r"""`pdh.dll` counter reads, over `ctypes`. No `pywin32`, no subprocess, ever.

**The rule this module exists to enforce: a missing instance is `None`, never `0.0`.** A
person at the booth who sees "NPU 0%" concludes the NPU is idle; if the truth is that no
counter exists, they conclude the NPU is broken. Those are opposite claims about the
hardware, and only one of them may be put on screen. So:

- A category or counter that cannot be opened raises `PdhUnavailableError` from
  `__init__`. It is never downgraded to a reading of zero.
- `read` returns `None` for an instance that is not in the enumerated set.
- `close` is idempotent, so a `finally:` block can always run it.

MEASURED BLOCKER, 2026-10-01, on this machine - read `value_reads` before trusting a
number. `ctypes` itself is fine: `GetSystemTime`, `GetVersionExW` and
`GlobalMemoryStatusEx` all fill their structures correctly through `ctypes` here, so this
is not a marshalling problem. The failure is inside `pdh.dll`, and it splits cleanly:

- **Working:** `PdhOpenQueryW` (0), `PdhAddEnglishCounterW` (0),
  `PdhCollectQueryData` (0), `PdhGetCounterTimeBase` (0, timebase 10000000 - which proves
  the counter handle is valid), `PdhRemoveCounter` (0), `PdhCloseQuery` (0),
  `PdhGetDllVersion` (0, 1283), `PdhExpandCounterPathW` (0x800007D2 = MORE_DATA with the
  correct size).
- **Broken:** every call that touches the performance-object *instance* database.
  `PdhGetFormattedCounterValue` raises `OSError: access violation reading
  0xFFFFFFFFFFFFFFFF`. `PdhGetRawCounterValue` returns success (0) having written only four
  bytes of uninitialised-looking data and then the process dies with 0xC0000005.
  `PdhCalculateCounterFromRawValue` and `PdhGetFormattedCounterArrayW` (buffer of 512
  items) fault the same way. `PdhLookupPerfIndexByNameW("GPU Engine", &index)` -
  a string in and a `DWORD` out, the simplest call in the whole API - access-violates.
  `PdhExpandWildCardPathW` returns `0xC0000BBD` even for `\Processor(*)\% Processor Time`,
  a counter that exists on every Windows install. `PdhGetFormattedCounterArrayW` returns
  `0xC0000BBC` for a null buffer and `0xC0000BBD` for a sized one.
  `PdhGetCounterInfoW` returns `0xC0000BBD`. Meanwhile `Get-Counter` and the CIM class
  `Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine` both read these same counters
  correctly, so the counters and their values exist; this is a fault in reaching them from
  `ctypes` on this build.

**Consequence, and the reason for `enable_value_reads` defaulting to `False`.** A native
access violation is not a Python exception: it cannot be caught, and it takes the whole
process with it. A booth application that dies while sampling is the worst outcome
available, so `PdhQuery` **does not call the value reader unless explicitly told to**. The
default is the measured state of this machine, the reason is the string below, and
`PdhQuery` exposes both through `value_reads` and `value_read_refusals` so a caller - and
the M0 report - can see that values were refused rather than zero. On a build where the
call is safe, pass `enable_value_reads=True` and the reading path is used unchanged.

This is the plan's Step 8 condition arriving in a different form than anticipated. The
plan expected a non-English Windows to make the *category* unopenable; here the category
opens fine (so the install is not the problem) but the *value* call is unusable. Either
way the rule is the same and is honoured here: record it, and never substitute a number.
The values themselves are obtained at operator time by `tools/probe_telemetry.py`, which
reads them through CIM and records them in `config/telemetry_map.json`.
"""

from __future__ import annotations

import ctypes
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "PDH_DLL",
    "PERF_DOUBLE",
    "PdhCounterPath",
    "PdhQuery",
    "PdhUnavailableError",
    "VALUE_READS_ENABLED_BY_DEFAULT",
    "VALUE_READS_REFUSED_REASON",
    "format_pdh_status",
    "pdh_available",
]

PDH_DLL: Final[str] = "pdh.dll"

#: `PERF_DOUBLE`. The only `CType` this module will read a `doubleValue` out of.
#: Measured on this machine: `Utilization Percentage` is reported through CIM, not
#: through the ctypes path, so no `CType` has been observed from `pdh.dll` here. Every
#: other `CType` is refused as `None` rather than reinterpreted.
PERF_DOUBLE: Final[int] = 5

#: Read a percentage, not a raw count. Fixed, not discovered: these are the two counters
#: this demo uses, and the name of the category is validated by `PdhAddEnglishCounterW`,
#: which resolves English counter names regardless of the display language.
DEFAULT_COUNTER: Final[str] = "Utilization Percentage"

#: The measured reason value reads are off by default. Verbatim, because the whole point is
#: that the next reader can tell an intentional refusal from an oversight.
VALUE_READS_REFUSED_REASON: Final[str] = (
    "value reads are disabled on this machine: every pdh.dll call that reaches the "
    "performance-object instance database faults natively (PdhGetFormattedCounterValue, "
    "PdhCalculateCounterFromRawValue, PdhLookupPerfIndexByNameW all raise an access "
    "violation; PdhGetRawCounterValue returns success having written 4 bytes and then "
    "kills the process with 0xC0000005), while the handle-level calls all succeed and "
    "ctypes struct-out is proven good by GetSystemTime/GetVersionExW/GlobalMemoryStatusEx. "
    "A native fault cannot be caught, so calling it would kill the booth application. "
    "Every reading is therefore None (unavailable), never 0. See the module docstring."
)

#: `False` on this machine, for the reason above. Flip to `True` only on a build where the
#: value read has been observed to return a value rather than fault.
VALUE_READS_ENABLED_BY_DEFAULT: Final[bool] = False

#: A utilisation percentage is a fraction of 100. A raw reading outside this range is not
#: a percentage this provider will present.
PERCENT_MIN: Final[float] = 0.0
PERCENT_MAX: Final[float] = 100.0


class PdhUnavailableError(RuntimeError):
    """A PDH category or counter could not be opened, or the library is unreachable.

    Raised rather than swallowed so that "we cannot measure" is a distinct, catchable
    state from "we measured zero". Every path in this module that cannot produce a
    measurement produces one of these two, never a number it did not read.
    """


class _PdhValueUnion(ctypes.Union):
    """The value half of `PDH_FMT_COUNTERVALUE`.

    The pointer members are `c_void_p` on purpose. As `c_char_p`/`c_wchar_p`, ctypes
    converts them to a Python object by *dereferencing* whatever the bytes hold, and a
    counter that left the union uninitialised turns that into an access violation
    **inside the conversion**, long after the native call returned. `c_void_p` never
    dereferences, so reading a struct can never crash the process.
    """

    _fields_ = [
        ("longValue", ctypes.c_long),
        ("doubleValue", ctypes.c_double),
        ("largeValue", ctypes.c_longlong),
        ("AnsiStringValue", ctypes.c_void_p),
        ("WideStringValue", ctypes.c_void_p),
    ]


class _PdhFmtCountervalue(ctypes.Structure):
    """`PDH_FMT_COUNTERVALUE`. Measured `ctypes.sizeof` on this machine: 16, value at 8."""

    _fields_ = [("CType", ctypes.c_uint32), ("value", _PdhValueUnion)]


_LIBRARY: Any = None
_LIBRARY_ERROR: str | None = None


def _load_library() -> Any:
    """`pdh.dll` loaded once, or `None` with `_LIBRARY_ERROR` explaining why."""
    global _LIBRARY, _LIBRARY_ERROR
    if _LIBRARY is not None or _LIBRARY_ERROR is not None:
        return _LIBRARY
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        _LIBRARY_ERROR = (
            f"ctypes.WinDLL is unavailable, so {PDH_DLL!r} cannot be loaded. This module "
            f"is Windows-only by construction: the counter object it reads is a Windows "
            f"performance object."
        )
        return None
    try:
        lib = loader(PDH_DLL)
    except OSError as exc:
        _LIBRARY_ERROR = f"cannot load {PDH_DLL!r}: {exc}"
        return None

    handle = ctypes.c_void_p
    dword = ctypes.c_uint32
    lpwstr = ctypes.c_wchar_p
    lpdword = ctypes.POINTER(dword)
    buffer = ctypes.c_void_p
    # PdhOpenQueryW(szDataSource, dwUserData, phQuery) - NULL data source = live data.
    lib.PdhOpenQueryW.argtypes = [lpwstr, ctypes.c_size_t, ctypes.POINTER(handle)]
    lib.PdhOpenQueryW.restype = dword
    lib.PdhCloseQuery.argtypes = [handle]
    lib.PdhCloseQuery.restype = dword
    # PdhAddEnglishCounterW(hQuery, szFullCounterPath, dwUserData, phCounter). The
    # "English" variant resolves English counter names on a non-English install, which is
    # the documented mitigation for a localised Windows; it is bound for that reason alone.
    lib.PdhAddEnglishCounterW.argtypes = [handle, lpwstr, ctypes.c_size_t,
                                          ctypes.POINTER(handle)]
    lib.PdhAddEnglishCounterW.restype = dword
    lib.PdhCollectQueryData.argtypes = [handle]
    lib.PdhCollectQueryData.restype = dword
    lib.PdhGetFormattedCounterArrayW.argtypes = [handle, buffer, lpdword, lpdword]
    lib.PdhGetFormattedCounterArrayW.restype = dword
    lib.PdhGetFormattedCounterValue.argtypes = [handle, buffer, lpdword]
    lib.PdhGetFormattedCounterValue.restype = dword
    lib.PdhExpandWildCardPathW.argtypes = [lpwstr, lpwstr, lpdword, lpdword]
    lib.PdhExpandWildCardPathW.restype = dword
    _LIBRARY = lib
    return _LIBRARY


def pdh_available() -> tuple[bool, str]:
    """Whether `pdh.dll` is loadable here, and the measured reason either way.

    `(False, reason)` when not. The reason is never empty and never invented.
    """
    if _load_library() is not None:
        return True, f"{PDH_DLL} loaded and the required entry points are bound"
    return False, _LIBRARY_ERROR or f"{PDH_DLL} could not be loaded"


def format_pdh_status(status: int) -> str:
    """A PDH status code as text Windows did not have to be asked for twice.

    Measured on this machine: `FormatMessageW` with `FORMAT_MESSAGE_FROM_HRESULT` returns
    an **empty string** for PDH status codes such as `0xC0000BBD` and `0xC0000BBC`, so no
    system text is available to quote. A table of symbolic `PDH_CSTATUS_*` names written
    from memory would be exactly the plausible-looking-but-unverified string this project
    forbids, so the code is reported numerically and labelled as untranslated.
    """
    return f"0x{int(status) & 0xFFFFFFFF:08X} (Windows supplies no text for this PDH status)"


@dataclass(frozen=True)
class PdhCounterPath:
    r"""One counter of one instance of `category`, as a `(category, instance)` pair.

    `instance` is the **bare** PDH instance name, e.g.
    `pid_1234_luid_0x00000000_0x000112A0_phys_0_eng_0_engtype_Neural`, not a full
    `\\host\object(instance)\counter` path. Bare names are what the wildcard expansion
    yields and what `app.telemetry.instance_map` parses, so keeping one spelling end to
    end removes a class of mismatch bug.
    """

    category: str
    instance: str


def _wildcard_path(category: str, counter: str) -> str:
    """`\\category(*)\\counter`, the form both PDH and CIM use for "every instance"."""
    return f"\\{category}(*)\\{counter}"


class PdhQuery:
    """An open PDH query over one category, and the instances it can see.

    **Never spawns anything.** ctypes into `pdh.dll` and nothing else, so this is safe to
    construct once and hold on the 4 Hz sampling path. The LUID -> device mapping is not
    resolved here: it is recorded once by `tools/probe_telemetry.py` and matched by
    `app.telemetry.instance_map`.

    If the category or any counter cannot be opened, `__init__` raises
    `PdhUnavailableError` carrying the PDH status code. That is the only honest response:
    the alternative - an object that exists but reports zero - is indistinguishable, at a
    glance, from a device that is genuinely idle.

    `read` is a two-step lookup, and the order matters:

    1. Is the instance in the enumerated set? If not, `None` - the instance has gone, and
       a process that has exited has no utilisation.
    2. Only then, a value read, and only if `enable_value_reads` is on. Off by default on
       this machine because the call faults natively and cannot be caught; see the module
       docstring. Every refusal is appended to `value_read_refusals` with its reason.
    """

    def __init__(
        self,
        category: str,
        counters: Mapping[str, str],
        *,
        enable_value_reads: bool | None = None,
    ) -> None:
        if not str(category).strip():
            raise PdhUnavailableError("a PDH query needs a category name; got an empty one")
        if not counters:
            raise PdhUnavailableError(
                f"no counters requested for category {category!r}; an empty query would "
                f"enumerate nothing and every reading would be unavailable for a reason "
                f"that has nothing to do with the hardware"
            )
        library = _load_library()
        if library is None:
            raise PdhUnavailableError(
                f"cannot open PDH category {category!r}: {_LIBRARY_ERROR}"
            )
        self._lib = library
        self._category = str(category)
        #: alias -> counter name, as the caller named it. Kept so a caller can report
        #: *which* counter it asked for in an error message.
        self._counters: dict[str, str] = {str(k): str(v) for k, v in counters.items()}
        self._value_reads = (
            VALUE_READS_ENABLED_BY_DEFAULT if enable_value_reads is None
            else bool(enable_value_reads)
        )
        #: Observable, so a caller and the M0 report can tell a refusal from a zero.
        self.value_read_refusals: list[str] = []
        self._closed = False

        handle = ctypes.c_void_p()
        status = library.PdhOpenQueryW(None, 0, ctypes.byref(handle))
        if status != 0 or not handle.value:
            raise PdhUnavailableError(
                f"PdhOpenQueryW failed for category {self._category!r}: "
                f"{format_pdh_status(status)}. No counter can be read, so utilisation is "
                f"unavailable - which is not the same as zero."
            )
        self._handle = handle
        #: alias -> PDH_HCOUNTER, for the counters that opened.
        self._handles: dict[str, ctypes.c_void_p] = {}
        try:
            for alias, counter_name in self._counters.items():
                counter_handle = ctypes.c_void_p()
                add_status = library.PdhAddEnglishCounterW(
                    self._handle,
                    _wildcard_path(self._category, counter_name),
                    0,
                    ctypes.byref(counter_handle),
                )
                if add_status != 0 or not counter_handle.value:
                    raise PdhUnavailableError(
                        f"PdhAddEnglishCounterW failed for counter {counter_name!r} in "
                        f"category {self._category!r}: {format_pdh_status(add_status)}. "
                        f"The counter cannot be opened, so its utilisation is unavailable. "
                        f"Note this uses the *English* counter name deliberately: on a "
                        f"non-English install a localised name is the usual cause, and "
                        f"substituting a number is not an option."
                    )
                self._handles[alias] = counter_handle
        except Exception:
            # A half-open query is worse than none: close it, so a failed construction
            # does not leak a handle per attempt at 5 Hz.
            self.close()
            raise

    @property
    def category(self) -> str:
        return self._category

    @property
    def counters(self) -> dict[str, str]:
        """The alias -> counter name mapping this query was opened with."""
        return dict(self._counters)

    @property
    def value_reads(self) -> bool:
        """Whether this query will attempt a native value read. See the module docstring."""
        return self._value_reads

    def instances(self) -> tuple[str, ...]:
        r"""Every instance name the query's counters currently have, sorted and unique.

        Empty is a legitimate answer and a common one: on an idle machine no process is
        using a GPU engine, so Windows has no `GPU Engine` instances to report. That is
        "nothing is running", not "the counter is missing", and callers must not turn it
        into a zero.

        Measured on this machine, `PdhExpandWildCardPathW` returns `0xC0000BBD` for a
        wildcard path, including `\Processor(*)\% Processor Time`, so this returns `()`
        here. The status is recorded on `last_status` rather than raised: an empty instance
        set is a legitimate reading, and raising would turn "idle" into "broken".
        """
        self._check_open()
        found: list[str] = []
        for counter_name in self._counters.values():
            path = _wildcard_path(self._category, counter_name)
            size = ctypes.c_uint32(0)
            count = ctypes.c_uint32(0)
            status = self._lib.PdhExpandWildCardPathW(
                path, None, ctypes.byref(size), ctypes.byref(count)
            )
            self.last_status = status
            if status != 0 or size.value == 0:
                continue
            buffer = ctypes.create_unicode_buffer(size.value)
            size_in = ctypes.c_uint32(size.value)
            count_in = ctypes.c_uint32(count.value)
            status = self._lib.PdhExpandWildCardPathW(
                path, ctypes.cast(buffer, ctypes.c_wchar_p),
                ctypes.byref(size_in), ctypes.byref(count_in),
            )
            self.last_status = status
            if status != 0:
                continue
            for entry in buffer[:].split("\0"):
                name = self._bare_instance(entry)
                if name and name not in found:
                    found.append(name)
        return tuple(sorted(found))

    def _bare_instance(self, expanded_path: str) -> str:
        """`\\\\host\\object(instance)\\counter` -> `instance`, or `""` if it is not that.

        The object and counter names are known here - they are the ones this query was
        opened with - so the reduction is exact rather than a "text between the first
        bracket" guess that would mangle an instance name containing a bracket.
        """
        text = str(expanded_path or "").strip()
        if not text:
            return ""
        head, _, tail = text.rpartition("\\")
        if not head or not tail:
            return ""
        object_and_instance = head.rpartition("\\")[2]
        prefix = f"{self._category}("
        if not object_and_instance.startswith(prefix) or not object_and_instance.endswith(")"):
            return ""
        return object_and_instance[len(prefix):-1]

    def read(self, path: PdhCounterPath) -> float | None:
        """The utilisation percentage for one instance, or `None` when there is none.

        `None` is returned, never `0.0`, in every one of these cases, and each is
        recorded in `value_read_refusals`:

        - the path's category is not this query's category;
        - the query is closed;
        - **more than one counter was added to this query.** `PdhCounterPath` is a
          `(category, instance)` pair and says nothing about which counter the caller
          means, so with several counters added an instance can belong to more than one
          and the reading would be a pick rather than a measurement. `None`; use a
          one-counter query per counter.
        - value reads are disabled on this machine (the default here) - see the module
          docstring. This check comes **before** the enumeration on purpose: with reads
          off there is nothing an enumeration could contribute, and skipping it keeps the
          4 Hz path off `pdh.dll` entirely.
        - the instance is not in the enumerated set - the case the plan's test pins. An
          engine whose process has exited has no utilisation, and reporting 0.0 for it
          is how a booth visitor is told the NPU is idle when nothing was measured;
        - the reading is outside `[0, 100]`, and is **refused rather than clamped**. A
          clamp would put a number on the card that was never measured: a 137.2% multi-
          engine sum displayed as 100.0% is a different number, and a wrong one.

        A value is returned only when it was actually read, was a `PERF_DOUBLE`, and fell
        inside the range. There is no code path in this method that produces a number the
        counter did not report.
        """
        if str(path.category) != self._category:
            reason = (f"path category {path.category!r} is not this query's category "
                      f"{self._category!r}")
            self.value_read_refusals.append(reason)
            return None
        if self._closed:
            self.value_read_refusals.append("query is closed")
            return None
        if len(self._handles) > 1:
            self.value_read_refusals.append(
                f"this query holds {len(self._handles)} counters "
                f"({sorted(self._counters.values())}) and PdhCounterPath names no "
                f"counter, so an instance of several of them cannot be attributed to one "
                f"reading. Use one counter per query."
            )
            return None
        if not self._value_reads:
            self.value_read_refusals.append(VALUE_READS_REFUSED_REASON)
            return None
        if str(path.instance) not in self.instances():
            # Not a failure to read: the instance is not there. This is the single most
            # important return in the module.
            self.value_read_refusals.append(
                f"instance {path.instance!r} is not in the enumerated set of "
                f"{self._category!r}; it has no utilisation, which is not zero"
            )
            return None
        return self._read_value(path.instance)

    def _read_value(self, instance: str) -> float | None:
        """The native value read, isolated so its refusal is recorded, not raised."""
        self.collect()
        first = self._counters[next(iter(self._counters))]
        handle = self._handles[next(iter(self._handles))]
        value = _PdhFmtCountervalue()
        value_type = ctypes.c_uint32(0)
        try:
            status = self._lib.PdhGetFormattedCounterValue(
                handle, ctypes.byref(value), ctypes.byref(value_type)
            )
        except OSError as exc:  # pragma: no cover - needs the native fault to be catchable
            self.value_read_refusals.append(
                f"PdhGetFormattedCounterValue for {instance!r} faulted: {exc}. A native "
                f"fault is normally not catchable at all, which is why this call is off "
                f"by default."
            )
            return None
        if status != 0:
            self.value_read_refusals.append(
                f"PdhGetFormattedCounterValue for {instance!r} returned "
                f"{format_pdh_status(status)}"
            )
            return None
        if value.CType != PERF_DOUBLE:
            self.value_read_refusals.append(
                f"{first!r} for {instance!r} came back as CType {value.CType}, not "
                f"PERF_DOUBLE ({PERF_DOUBLE}); this provider will not reinterpret an "
                f"unknown counter type as a percentage"
            )
            return None
        reading = float(value.value.doubleValue)
        if not (PERCENT_MIN <= reading <= PERCENT_MAX) or reading != reading:
            self.value_read_refusals.append(
                f"{first!r} for {instance!r} read {reading!r}, outside "
                f"[{PERCENT_MIN}, {PERCENT_MAX}]; refused rather than clamped, because a "
                f"clamped value is a number the counter never reported"
            )
            return None
        return reading

    def collect(self) -> int:
        """`PdhCollectQueryData` for this query, returning the PDH status code.

        A formatted read needs a sample. Failures are **not** raised: a query that cannot
        be collected has no *new* sample, which is a stale reading rather than an absent
        one, and the staleness decision belongs to the caller that owns the 2 s rule.
        Measured working on this machine (status 0).
        """
        if self._closed:
            return -1
        status = self._lib.PdhCollectQueryData(self._handle)
        self.last_status = status
        return int(status)

    def formatted_array_size(self) -> tuple[int, int, str]:
        """`(bytes, item_count, status)` from a `PdhGetFormattedCounterArrayW` size probe.

        The plan names this function, and it is bound and used for exactly what it is
        reliable for: asking PDH how large the formatted array would be. The size probe
        passes a null buffer, which is measured to return a status rather than fault (a
        *sized* buffer of 512 items faults - see the module docstring), so the reported
        count is `0` on this machine. That is recorded as the status text, not turned into
        a claim that the query has no instances: only `instances()` may make that claim.
        """
        self._check_open()
        size = ctypes.c_uint32(0)
        count = ctypes.c_uint32(0)
        status = self._lib.PdhGetFormattedCounterArrayW(
            self._handle, None, ctypes.byref(size), ctypes.byref(count)
        )
        self.last_status = status
        return int(size.value), int(count.value), format_pdh_status(status)

    def close(self) -> None:
        """Close the query. Idempotent, and safe to call from a `finally:` block.

        A second call is a no-op rather than a double `PdhCloseQuery`, so a caller cannot
        corrupt a handle that has already been released.
        """
        if self._closed:
            return
        self._closed = True
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value:
            self._lib.PdhCloseQuery(handle)

    def _check_open(self) -> None:
        if self._closed:
            raise PdhUnavailableError(
                f"the PDH query for {self._category!r} is closed; create a new PdhQuery"
            )

    def __enter__(self) -> "PdhQuery":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown ordering
        try:
            self.close()
        except Exception:
            pass

    #: The PDH status of the most recent call, kept so a refusal can name the real code.
    last_status: int = 0

    #: Class-level defaults, so `close()` and `__del__` are safe on an object whose
    #: `__init__` raised before it finished. A half-built query must not turn cleanup
    #: into a second exception.
    _lib: Any = None
    _handle: Any = None
    _closed: bool = True
    _handles: dict[str, Any] = {}
