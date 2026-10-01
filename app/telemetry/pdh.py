r"""`pdh.dll` counter reads, over `ctypes`. No `pywin32`, no subprocess, ever.

**The rule this module exists to enforce: a missing instance is `None`, never `0.0`.** A
person at the booth who sees "NPU 0%" concludes the NPU is idle; if the truth is that no
counter exists, they conclude the NPU is broken. Those are opposite claims about the
hardware, and only one of them may be put on screen. So:

- A category or counter that cannot be opened raises `PdhUnavailableError` from
  `__init__`, carrying the real PDH status code. It is never downgraded to a reading of
  zero, and a **NULL handle is refused**, never used.
- `read` returns `None` for an instance that is not in the enumerated set, for a counter
  that has not been primed, and for any non-`dwCStatus == 0` read.
- `close` is idempotent, so a `finally:` block can always run it.

---

## What was fixed on 2026-10-01, and what is still measured unavailable

### 1. The counter *path* was wrong. Fixed, and it was the whole of the earlier crash.

`"GPU Engine"` is a counter **set**, not a counter. Measured on this machine, same
interpreter, same process:

```
PdhAddEnglishCounterW(q, "GPU Engine", 0, &c)                        -> rc = 0xC0000BC0, c == NULL
PdhAddEnglishCounterW(q, "\GPU Engine(*)\Utilization Percentage",0,&c) -> rc = 0, c valid
PdhGetCounterTimeBase(c) -> rc = 0, timebase 10000000   (proves c is a real handle)
```

Every call made after a failed add was writing through a **NULL handle**. That is a real
access violation, and it is a symptom of the wrong name - not a PDH defect. It is now
impossible: `PdhQuery.__init__` refuses a NULL handle and raises, so a wrong name is loudly
diagnosable instead of silently poisoning every later call. `GPU_ENGINE_COUNTER_PATH` is a
module constant so the name cannot drift.

### 2. `PdhExpandWildCardPathW` has **five** parameters, not four.

The earlier binding passed four, which is why every wildcard expansion returned
`0xC0000BBD` and `instances()` was empty. With the real five-parameter signature and the
documented two-call pattern it returns `rc = 0` and the real instance paths:

```
call 1, NULL buffer -> rc = 0x800007D2, characters required = 45899
call 2, real buffer -> rc = 0, 455-497 paths, e.g.
    \GPU Engine(pid_10228_luid_0x00000000_0x000112A0_phys_0_eng_0_engtype_Neural)\Utilization Percentage
```

Measured cost on this machine: **5.1 ms cold, 0.006 ms warm and cached**, for 455-497
paths (127 ms on the very first expansion in a fresh process, which is a perflib
first-touch cost). Still too slow to run at 4 Hz over the whole set, which is why the read
strategy is the *owned subset* (see below), and why `instances()` caches for
`instance_ttl`.

### 3. `PdhGetFormattedCounterValue` still faults natively. Not fixed, and not guessable.

With the correct counter name, a handle proven valid by `PdhGetCounterTimeBase`, three
`PdhCollectQueryData` samples, and stable 64-byte output buffers, this call kills the
process with `0xC0000005` ("access violation reading 0xFFFFFFFFFFFFFFFF") - for
`\GPU Engine(*)\Utilization Percentage` **and** for three controls that have nothing to do
with the GPU:

| counter | handle | timebase | `PdhGetFormattedCounterValue` |
|---|---|---|---|
| `\Processor(_Total)\% Processor Time` | valid | 10000000 | **0xC0000005** |
| `\Memory\Available Bytes` | valid | 10000000 | **0xC0000005** |
| `\System\Processor Queue Length` | valid | 10000000 | **0xC0000005** |
| `\GPU Engine(*)\Utilization Percentage` | valid | 10000000 | **0xC0000005** |

`PdhGetRawCounterValue` faults the same way. `pdh.dll` here is `10.0.26100.1` and exports
119 names; `PdhGetFormattedCounterValue` and `PdhGetRawCounterValue` are the only value
entry points, and neither has a `W`/`A` twin. Swept and measured, all still unusable:

- 2, 3 arguments (the documented arity): fault.
- 4-6 arguments: returns `0xC0000BBD`/`0xC0000BBC` and writes 4 bytes to two out
  parameters - no value, and no fault.
- `PdhGetFormattedCounterArrayW` at 3-6 arguments, 9 `dwFormat` values, both a byte count
  and an item count in the size slot: `rc = 0`, no fault, and **no percentage anywhere in
  the returned buffer**. What it actually writes is the *instance name list*: a count, then
  `{pointer, length}` headers and UTF-16 names. A scan of every 8-byte offset of the
  returned buffer for a double in `[0.05, 100]` found **zero** candidates, on the wildcard
  counter and on a concrete single-instance counter.
- `PdhGetFormattedCounterArrayA`, `PdhVbGetDoubleCounterValue`, `PdhGetCounterInfoW`,
  `PdhLookupPerfIndexByNameW`: measured, listed in the fix report. The last of these works
  (`rc = 0`, `index = 5802`) - the earlier report's "it access-violates" was the NULL
  handle from finding 1, not a PDH fault.

`ctypes` is not the problem: `GetSystemTime`, `GetVersionExW` and `GlobalMemoryStatusEx`
all fill their structures correctly here, and a hand-built `CFUNCTYPE` from the raw export
address behaves identically to the `WinDLL` binding. Nor is it a "that instance has no
data" fault: the same call on a **concrete instance that was busy at that moment** - with
`Get-Counter` reporting 18-63% on sibling instances of the same counter - faults
identically. The same counters read correctly through `Get-Counter` and through
`Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine`, so the counters and their values
exist; what is unavailable is reaching the *value* from this process.

**Consequence, and the reason `enable_value_reads` still defaults to `False`.** A native
access violation is not a Python exception: it cannot be caught, and it takes the whole
process with it. A booth application that dies while sampling is the worst outcome
available, so `PdhQuery` **does not call the value reader unless explicitly told to**. The
default is the measured state of this machine, the reason is the string below, and every
refusal is appended to `value_read_refusals` so a caller - and the M0 report - can see that
a value was **refused**, not zero. `PdhQuery(..., enable_value_reads=True)` turns the
reading path on unchanged, and the reading path itself is complete: it returns a `float` in
`[0, 100]` when `dwCStatus == 0` and `None` for every other state.

The reading path is exercised with an injected `pdh_api()`, which is the only safe way to
test a reader whose native call can fault. The **real** `pdh.dll` is still used for
everything that is real: the counter name, the add, the wildcard enumeration, the array
size probe, the handle guards, and `close`.

### 4. The read strategy: the owned subset, read one concrete counter each.

This machine has **455-497** live `GPU Engine` instances and the formatted array reports
**4328** items for the wildcard counter. Reading all of them 4 times a second would be
meaningless work and, on this build, would not return percentages anyway. So the ruling
implemented here is:

1. `instances()` enumerates the real names once and caches them (`instance_ttl`).
2. The caller selects the **owned** subset - the instances whose LUID the recorded map
   identifies as ours - with `instances_for_luid(luid)`.
3. `watch(names)` opens **one concrete counter per owned instance**, by adding
   `\GPU Engine(<bare instance>)\Utilization Percentage`, and primes them **once**.
4. Each sample is then `collect()` plus one `PdhGetFormattedCounterValue` per owned
   instance - a fixed, small number that does not grow with the machine's total instance
   count.

Measured with a real load running, on this machine: 13 NPU `Neural` instances + 31 GPU
`Compute` instances = 44 owned counters; `watch` of those costs **754 ms** once (dominated
by the 0.75 s prime), and a sample - `collect()` plus `read_many` over all 44 - costs
**median 0.640 ms, p95 0.844 ms, max 1.327 ms** over 50 samples. The 81 instances on the
third adapter are never opened: they are not ours, and a rule that swept them in because
their LUID sits next to the NPU's is the mistake this layer exists to prevent.

Why per-instance and not a batch: a concrete counter is what the value read needs, it
carries its own `dwCStatus` so a dead engine is distinguishable from a busy one, and its
handle survives for the life of the query, so the 4 Hz path does no path building and no
`PdhAddEnglishCounterW` at all.

### 5. Priming a rate counter, at open and not per read.

`Utilization Percentage` is a rate. It needs two `PdhCollectQueryData` samples separated in
time; with one sample `dwCStatus` comes back non-zero and the reading is "not yet
available", not zero. `prime_seconds` (default `PRIME_SECONDS`) is that interval. Priming
happens in `__init__` and in `watch()` - the setup path - and **never** per read, because a
4 Hz sampling path cannot afford a multi-second prime per sample. A counter added after
construction is not primed until the next `prime()`; reads on it return `None` with that
reason, so an unprimed counter can never be mistaken for an idle device.
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from app.telemetry.instance_map import normalise_luid, parse_instance_name

__all__ = [
    "FMT_COUNTERVALUE_SIZE",
    "GPU_ENGINE_COUNTER",
    "GPU_ENGINE_COUNTER_PATH",
    "GPU_ENGINE_COUNTER_SET",
    "INSTANCE_TTL_SECONDS",
    "PDH_DLL",
    "PDH_MORE_DATA",
    "PDH_MORE_DATA_UNPRIMED",
    "PERCENT_MAX",
    "PERCENT_MIN",
    "PRIME_SECONDS",
    "PdhApi",
    "PdhCounterPath",
    "PdhQuery",
    "PdhUnavailableError",
    "VALUE_READS_ENABLED_BY_DEFAULT",
    "VALUE_READS_REFUSED_REASON",
    "format_pdh_status",
    "instance_path",
    "pdh_api",
    "pdh_available",
    "wildcard_path",
]

PDH_DLL: Final[str] = "pdh.dll"

#: The counter **set**. Windows' name for the `GPU Engine` performance object. This is not
#: a counter and cannot be added: see `GPU_ENGINE_COUNTER_PATH`.
GPU_ENGINE_COUNTER_SET: Final[str] = "GPU Engine"

#: The counter within that set, as Windows spells it in English. The reader uses
#: `PdhAddEnglishCounterW`, which resolves English counter names regardless of the display
#: language, so this spelling is correct on a localised install too.
GPU_ENGINE_COUNTER: Final[str] = "Utilization Percentage"

#: The full counter path. **This is the constant the earlier bug was about.** Passing
#: `GPU_ENGINE_COUNTER_SET` on its own returns `0xC0000BC0` and a NULL handle.
GPU_ENGINE_COUNTER_PATH: Final[str] = (
    f"\\{GPU_ENGINE_COUNTER_SET}(*)\\{GPU_ENGINE_COUNTER}"
)

#: `PDH_MORE_DATA`, measured: the status the two-call PDH patterns return when the buffer
#: is too small or absent, together with the size the second call must supply. Windows
#: supplies no text for it via `FormatMessageW`, see `format_pdh_status`.
PDH_MORE_DATA: Final[int] = 0x800007D2

#: The size probe's status on a counter that has **no sample yet**, also measured here:
#:
#:     PdhGetFormattedCounterArrayW wildcard, fresh query   -> 0x800007D5, 0 bytes
#:     same counter, after one PdhCollectQueryData          -> 0x800007D2, 69256 bytes
#:
#: So the array size is only meaningful once the counter has been collected, which is the
#: same priming requirement a rate counter has. Both statuses mean "no buffer was filled";
#: only the second one tells you how big a buffer to ask for.
PDH_MORE_DATA_UNPRIMED: Final[int] = 0x800007D5

#: Every measured status that means "the buffer was not filled, and here is (or is not) the
#: size it needs". Anything else is a genuine failure and is reported as one.
_BUFFER_NOT_FILLED: Final[frozenset[int]] = frozenset({PDH_MORE_DATA, PDH_MORE_DATA_UNPRIMED})

#: Read a percentage, not a raw count. A reading outside this range is not a percentage
#: this provider will present; it is refused, not clamped.
PERCENT_MIN: Final[float] = 0.0
PERCENT_MAX: Final[float] = 100.0

#: Seconds between the two `PdhCollectQueryData` samples that prime a rate counter.
#: Measured: at 0.3 s and above `dwCStatus` comes back 0 for a live `GPU Engine` engine.
PRIME_SECONDS: Final[float] = 0.75

#: How long an enumeration is reused before it is refreshed. `PdhExpandWildCardPathW` costs
#: ~2 ms warm and ~127 ms cold on this machine, so re-expanding on every 4 Hz sample would
#: spend more time in enumeration than in reading.
INSTANCE_TTL_SECONDS: Final[float] = 2.0

#: The measured reason value reads are off by default. Verbatim, because the whole point is
#: that the next reader can tell an intentional refusal from an oversight.
VALUE_READS_REFUSED_REASON: Final[str] = (
    "the native PDH value read is not usable on this machine: PdhGetFormattedCounterValue "
    "and PdhGetRawCounterValue both terminate the process with 0xC0000005 (access "
    "violation reading 0xFFFFFFFFFFFFFFFF) for \\GPU Engine(*)\\Utilization Percentage and "
    "for the non-GPU controls \\Processor(_Total)\\% Processor Time, \\Memory\\Available "
    "Bytes and \\System\\Processor Queue Length, all with a handle proven valid by "
    "PdhGetCounterTimeBase (timebase 10000000) and three PdhCollectQueryData samples. "
    "PdhGetFormattedCounterArrayW returns rc=0 and no percentage: what it writes into the "
    "item buffer is the instance-name list. The same counters read correctly through "
    "Get-Counter and through Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine, so "
    "the counter is live and the in-process read is not. A native fault cannot be caught, "
    "so calling it would kill the booth application. Every reading is therefore None "
    "(unavailable), never 0."
)

#: `False` on this machine, for the reason above. Flip to `True` - or pass
#: `enable_value_reads=True` - only on a build where the value read has been observed to
#: return a value rather than fault.
VALUE_READS_ENABLED_BY_DEFAULT: Final[bool] = False


class PdhUnavailableError(RuntimeError):
    """A PDH counter could not be opened, or the library is unreachable.

    Raised rather than swallowed so that "we cannot measure" is a distinct, catchable
    state from "we measured zero". Every path in this module that cannot produce a
    measurement produces one of these two, never a number it did not read.
    """


class _PdhValueUnion(ctypes.Union):
    """The value half of `PDH_FMT_COUNTERVALUE`.

    The pointer members are `c_void_p` on purpose. As `c_char_p`/`c_wchar_p`, ctypes
    converts them to a Python object by *dereferencing* whatever the bytes hold, and a
    counter that left the union uninitialised turns that into an access violation **inside
    the conversion**, long after the native call returned. `c_void_p` never dereferences, so
    reading a struct can never crash the process.
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


FMT_COUNTERVALUE_SIZE: Final[int] = ctypes.sizeof(_PdhFmtCountervalue)


def format_pdh_status(status: int) -> str:
    """A PDH status code as text Windows did not have to be asked for twice.

    Measured on this machine: `FormatMessageW` with `FORMAT_MESSAGE_FROM_HRESULT` returns
    an **empty string** for PDH status codes such as `0xC0000BBD` and `0xC0000BBC`, so no
    system text is available to quote. A table of symbolic `PDH_CSTATUS_*` names written
    from memory would be exactly the plausible-looking-but-unverified string this project
    forbids, so the code is reported numerically and labelled as untranslated.
    """
    return f"0x{int(status) & 0xFFFFFFFF:08X} (Windows supplies no text for this PDH status)"


def wildcard_path(category: str, counter: str) -> str:
    r"""`\\category(*)\\counter`, the form both PDH and CIM use for "every instance"."""
    return f"\\{category}(*)\\{counter}"


def instance_path(category: str, instance: str) -> str:
    r"""`\\category(instance)\\counter` for one concrete instance.

    This is the path `watch` adds, and it is the only form a single-instance value read
    can be taken from.
    """
    return f"\\{category}({instance})\\{GPU_ENGINE_COUNTER}"


class PdhApi:
    """Every native `pdh.dll` call this module makes, in one place.

    Two reasons this is a class rather than loose `ctypes` calls:

    1. **Every entry point gets an explicit `restype`.** Without it ctypes returns a
       32-bit-truncated `c_long` on 64-bit and the PDH status codes - which are NTSTATUS
       values such as `0xC0000BC0` - are misread.
    2. **Handles cross this boundary as `int`, with `0` meaning NULL.** That makes the
       NULL-handle guard in `PdhQuery` total rather than hopeful, and it makes the whole
       native surface substitutable - which is the only safe way to test a reader whose
       value call faults the process. Pass a substitute to `PdhQuery(api=...)`.

    Never call `formatted_value` or `raw_counter_value` through this class without a
    handle obtained from `add_counter` and checked non-zero. `pdh.dll` on this machine
    dereferences a NULL or unprimed handle rather than returning a status.
    """

    def __init__(self, library: Any) -> None:
        self._lib = library
        self._bind()

    def _bind(self) -> None:
        lib, handle, dword = self._lib, ctypes.c_void_p, ctypes.c_uint32
        lpwstr, lpdword, buffer = ctypes.c_wchar_p, ctypes.POINTER(dword), ctypes.c_void_p
        # PdhOpenQueryW(szDataSource, dwUserData, phQuery). NULL data source = live data.
        lib.PdhOpenQueryW.argtypes = [lpwstr, ctypes.c_size_t, ctypes.POINTER(handle)]
        lib.PdhOpenQueryW.restype = dword
        lib.PdhCloseQuery.argtypes = [handle]
        lib.PdhCloseQuery.restype = dword
        # PdhAddEnglishCounterW(hQuery, szFullCounterPath, dwUserData, phCounter). The
        # "English" variant resolves English counter names on a non-English install.
        lib.PdhAddEnglishCounterW.argtypes = [
            handle, lpwstr, ctypes.c_size_t, ctypes.POINTER(handle)
        ]
        lib.PdhAddEnglishCounterW.restype = dword
        lib.PdhRemoveCounter.argtypes = [handle]
        lib.PdhRemoveCounter.restype = dword
        lib.PdhCollectQueryData.argtypes = [handle]
        lib.PdhCollectQueryData.restype = dword
        lib.PdhGetCounterTimeBase.argtypes = [handle, lpdword]
        lib.PdhGetCounterTimeBase.restype = dword
        lib.PdhLookupPerfIndexByNameW.argtypes = [lpwstr, lpwstr, lpdword]
        lib.PdhLookupPerfIndexByNameW.restype = dword
        # FIVE parameters, and no instance-count parameter. Measured: with four bound, the
        # call returns 0xC0000BBD for every wildcard path.
        lib.PdhExpandWildCardPathW.argtypes = [lpwstr, lpwstr, buffer, lpdword, lpdword]
        lib.PdhExpandWildCardPathW.restype = dword
        # SIX parameters: (hCounter, dwFormat, lpdwBufferSize, lpdwValueType,
        # ItemBuffer, lpdwStatus). lpdwBufferSize is in BYTES, not items.
        lib.PdhGetFormattedCounterArrayW.argtypes = [
            handle, dword, lpdword, lpdword, ctypes.POINTER(_PdhFmtCountervalue), lpdword
        ]
        lib.PdhGetFormattedCounterArrayW.restype = dword
        lib.PdhGetFormattedCounterValue.argtypes = [
            handle, ctypes.POINTER(_PdhFmtCountervalue), lpdword
        ]
        lib.PdhGetFormattedCounterValue.restype = dword
        lib.PdhGetRawCounterValue.argtypes = [handle, buffer, lpdword]
        lib.PdhGetRawCounterValue.restype = dword

    # -- handle lifecycle -----------------------------------------------------------

    def open_query(self) -> tuple[int, int]:
        """`(status, handle)`. Handle `0` means NULL and the caller must not use it."""
        handle = ctypes.c_void_p()
        status = self._lib.PdhOpenQueryW(None, 0, ctypes.byref(handle))
        return int(status), int(handle.value or 0)

    def close_query(self, query: int) -> int:
        return int(self._lib.PdhCloseQuery(ctypes.c_void_p(query)))

    def add_counter(self, query: int, path: str) -> tuple[int, int]:
        """`(status, handle)` for one counter path.

        **`(0, 0)` is possible and is a failure, not a success.** Measured: adding
        `GPU_ENGINE_COUNTER_SET` returns a non-zero status *and* leaves the handle NULL;
        a caller that only checked the status would then write through NULL.
        """
        handle = ctypes.c_void_p()
        status = self._lib.PdhAddEnglishCounterW(
            ctypes.c_void_p(query), path, 0, ctypes.byref(handle)
        )
        return int(status), int(handle.value or 0)

    def remove_counter(self, counter: int) -> int:
        return int(self._lib.PdhRemoveCounter(ctypes.c_void_p(counter)))

    def collect(self, query: int) -> int:
        return int(self._lib.PdhCollectQueryData(ctypes.c_void_p(query)))

    def counter_timebase(self, counter: int) -> tuple[int, int]:
        base = ctypes.c_uint32(0)
        status = self._lib.PdhGetCounterTimeBase(ctypes.c_void_p(counter), ctypes.byref(base))
        return int(status), int(base.value)

    def lookup_index(self, name: str) -> tuple[int, int]:
        """`(status, index)`. Measured working: `("GPU Engine") -> (0, 5802)`."""
        index = ctypes.c_uint32(0)
        status = self._lib.PdhLookupPerfIndexByNameW(None, name, ctypes.byref(index))
        return int(status), int(index.value)

    # -- enumeration ----------------------------------------------------------------

    def expand_wildcard(self, path: str) -> tuple[int, tuple[str, ...]]:
        """`(status, expanded paths)` from the documented two-call pattern.

        Call one passes a NULL buffer and reads back the required **character** count from
        `pcchPathListLength`; call two passes a buffer of that size. Measured working for
        `\\GPU Engine(*)\\Utilization Percentage`: `0x800007D2` then `0` with 455-469 paths.
        """
        chars = ctypes.c_uint32(0)
        size = ctypes.c_uint32(0)
        status = int(self._lib.PdhExpandWildCardPathW(
            None, path, None, ctypes.byref(chars), ctypes.byref(size)
        ))
        if status not in _BUFFER_NOT_FILLED or chars.value == 0:
            return status, ()
        buffer = ctypes.create_unicode_buffer(chars.value + 2)
        chars_in, size_in = ctypes.c_uint32(chars.value), ctypes.c_uint32(size.value + 8)
        status = int(self._lib.PdhExpandWildCardPathW(
            None, path, ctypes.cast(buffer, ctypes.c_void_p),
            ctypes.byref(chars_in), ctypes.byref(size_in),
        ))
        if status != 0:
            return status, ()
        return status, tuple(e for e in buffer[:].split("\0") if e)

    # -- values ---------------------------------------------------------------------

    def formatted_value(self, counter: int) -> tuple[int, int, int, float]:
        """`(status, dwCStatus, CType, doubleValue)` for one concrete counter.

        **Unsafe on this machine** - it terminates the process; see the module docstring.
        Reached only when `PdhQuery` has value reads explicitly enabled and a non-zero
        handle in hand.
        """
        value = _PdhFmtCountervalue()
        value_type = ctypes.c_uint32(0)
        status = int(self._lib.PdhGetFormattedCounterValue(
            ctypes.c_void_p(counter), ctypes.byref(value), ctypes.byref(value_type)
        ))
        return status, int(value.CType), int(value_type.value), float(value.value.doubleValue)

    def formatted_array_size(self, counter: int, dw_format: int = 0) -> tuple[int, int]:
        """`(status, bytes)` - the NULL-buffer probe of the two-call array pattern.

        `lpdwBufferSize` is in **bytes**, and the caller divides by
        `FMT_COUNTERVALUE_SIZE` to get an item count. Measured on this machine, primed:
        `0x800007D2` with 69256 bytes, i.e. 4328 items. Unprimed it returns
        `0x800007D5` with 0 bytes.
        """
        size, value_type, status_out = (
            ctypes.c_uint32(0), ctypes.c_uint32(0), ctypes.c_uint32(0)
        )
        status = int(self._lib.PdhGetFormattedCounterArrayW(
            ctypes.c_void_p(counter), dw_format, ctypes.byref(size), ctypes.byref(value_type),
            None, ctypes.byref(status_out),
        ))
        return status, int(size.value)

    def formatted_array(
        self, counter: int, items: int, dw_format: int = 0
    ) -> tuple[int, int, tuple[int, ...]]:
        """`(status, items filled, per-item status words)` with a **real** buffer.

        The second half of the two-call pattern: `lpdwBufferSize` in **bytes**, a buffer
        large enough for the probe, and never a NULL `ItemBuffer`.

        **On output PDH reports the counter's whole item count, not how many it wrote.**
        Measured: asking for 8 items of a 4328-item counter comes back with
        `lpdwBufferSize = 4328 * 16`. The count is therefore clamped to the buffer that
        was actually supplied, or reading it would index past the allocation - one more
        way to turn this call into a native fault.

        The returned status words are whatever PDH put in the first `DWORD` of each
        16-byte slot. **On this build those slots are not `PDH_FMT_COUNTERVALUE` items** -
        they hold the instance-name list, so no percentage is extracted here. That is
        recorded rather than papered over; see the module docstring.
        """
        size = ctypes.c_uint32(items * FMT_COUNTERVALUE_SIZE)
        value_type = ctypes.c_uint32(0)
        status_out = ctypes.c_uint32(0)
        buffer = (_PdhFmtCountervalue * items)()
        status = int(self._lib.PdhGetFormattedCounterArrayW(
            ctypes.c_void_p(counter), dw_format, ctypes.byref(size), ctypes.byref(value_type),
            buffer, ctypes.byref(status_out),
        ))
        written = min(int(size.value) // FMT_COUNTERVALUE_SIZE, items)
        return status, written, tuple(int(buffer[i].CType) for i in range(written))


_API: PdhApi | None = None
_API_ERROR: str | None = None


def _load_api() -> PdhApi | None:
    """`PdhApi` over `pdh.dll` built once, or `None` with `_API_ERROR` explaining why."""
    global _API, _API_ERROR
    if _API is not None or _API_ERROR is not None:
        return _API
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        _API_ERROR = (
            f"ctypes.WinDLL is unavailable, so {PDH_DLL!r} cannot be loaded. This module "
            f"is Windows-only by construction: the counter object it reads is a Windows "
            f"performance object."
        )
        return None
    try:
        _API = PdhApi(loader(PDH_DLL))
    except (OSError, AttributeError) as exc:
        _API_ERROR = f"cannot load or bind {PDH_DLL!r}: {exc}"
        _API = None
    return _API


def pdh_api() -> PdhApi | None:
    """The shared `PdhApi`, or `None` when `pdh.dll` is unreachable here.

    Public so that a caller - `tools/probe_telemetry.py`, or a test - can exercise the
    native surface directly and see the real status codes, rather than inferring them.
    """
    return _load_api()


def pdh_available() -> tuple[bool, str]:
    """Whether `pdh.dll` is loadable here, and the measured reason either way."""
    if _load_api() is not None:
        return True, f"{PDH_DLL} loaded and the required entry points are bound"
    return False, _API_ERROR or f"{PDH_DLL} could not be loaded"


@dataclass(frozen=True)
class PdhCounterPath:
    r"""One counter of one instance of `category`, as a `(category, instance)` pair.

    `instance` is the **bare** PDH instance name, e.g.
    `pid_1234_luid_0x00000000_0x000112A0_phys_0_eng_0_engtype_Neural`, not a full
    `\host\object(instance)\counter` path. Bare names are what the wildcard expansion
    yields and what `app.telemetry.instance_map` parses, so keeping one spelling end to
    end removes a class of mismatch bug.
    """

    category: str
    instance: str


class PdhQuery:
    """An open PDH query over one category, the instances it can see, and their readings.

    **Never spawns anything.** ctypes into `pdh.dll` and nothing else, so this is safe to
    construct once and hold on the 4 Hz sampling path. The LUID -> device mapping is not
    resolved here: it is recorded once by `tools/probe_telemetry.py` and matched by
    `app.telemetry.instance_map`.

    If the category or any counter cannot be opened, `__init__` raises
    `PdhUnavailableError` carrying the real PDH status code - **including when the add
    "succeeds" with a NULL handle**, which is what a counter *set* name produces. That is
    the only honest response: an object that exists but reports zero is indistinguishable,
    at a glance, from a device that is genuinely idle.

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
        api: PdhApi | None = None,
        enable_value_reads: bool | None = None,
        prime_seconds: float = PRIME_SECONDS,
        instance_ttl: float = INSTANCE_TTL_SECONDS,
    ) -> None:
        if not str(category).strip():
            raise PdhUnavailableError("a PDH query needs a category name; got an empty one")
        if not counters:
            raise PdhUnavailableError(
                f"no counters requested for category {category!r}; an empty query would "
                f"enumerate nothing and every reading would be unavailable for a reason "
                f"that has nothing to do with the hardware"
            )
        api = api if api is not None else _load_api()
        if api is None:
            raise PdhUnavailableError(
                f"cannot open PDH category {category!r}: {_API_ERROR}"
            )
        self._api = api
        self._category = str(category)
        #: alias -> counter name, as the caller named it. Kept so a caller can report
        #: *which* counter it asked for in an error message.
        self._counters: dict[str, str] = {str(k): str(v) for k, v in counters.items()}
        self._value_reads = (
            VALUE_READS_ENABLED_BY_DEFAULT if enable_value_reads is None
            else bool(enable_value_reads)
        )
        self._prime_seconds = max(float(prime_seconds), 0.0)
        self._instance_ttl = max(float(instance_ttl), 0.0)
        #: Observable, so a caller and the M0 report can tell a refusal from a zero.
        self.value_read_refusals: list[str] = []
        #: The PDH status of the most recent call, kept so a refusal can name the real code.
        self.last_status: int = 0
        self._closed = False
        self._primed = False
        self._instance_cache: tuple[str, ...] = ()
        self._instance_cache_at: float = 0.0
        #: bare instance name -> concrete counter handle, for the owned subset.
        self._instance_handles: dict[str, int] = {}

        status, self._query = api.open_query()
        self.last_status = status
        if status != 0 or not self._query:
            raise PdhUnavailableError(
                f"PdhOpenQueryW failed for category {self._category!r}: "
                f"{format_pdh_status(status)}"
                f"{'' if self._query else ', and the query handle is NULL'}. No counter "
                f"can be read, so utilisation is unavailable - which is not the same as "
                f"zero."
            )
        #: alias -> PDH_HCOUNTER, for the wildcard counters that opened.
        self._handles: dict[str, int] = {}
        try:
            for alias, counter_name in self._counters.items():
                self._handles[alias] = self._add(api, wildcard_path(self._category, counter_name),
                                                  counter_name, alias)
        except Exception:
            # A half-open query is worse than none: close it, so a failed construction
            # does not leak a handle per attempt at 5 Hz.
            self.close()
            raise
        if self._prime_seconds:
            self.prime()

    # -- construction helpers -------------------------------------------------------

    def _add(self, api: PdhApi, path: str, counter_name: str, alias: str) -> int:
        """Add one counter, or raise. Refuses a NULL handle even when the status is 0.

        This is the fix for the bug that made every later call on this machine fault: a
        wrong counter name returns a non-zero status *and* a NULL handle, and a NULL
        handle is what `pdh.dll` dereferences. Checking the status alone is not enough.
        """
        status, handle = api.add_counter(self._query, path)
        self.last_status = status
        if status != 0:
            raise PdhUnavailableError(
                f"PdhAddEnglishCounterW failed for {path!r}: {format_pdh_status(status)}. "
                f"Counter {counter_name!r} in category {self._category!r} (alias {alias!r}) "
                f"could not be opened, so its utilisation is unavailable. Note this uses "
                f"the *English* counter name deliberately: on a non-English install a "
                f"localised name is the usual cause, and substituting a number is not an "
                f"option. A counter SET name such as {GPU_ENGINE_COUNTER_SET!r} is not a "
                f"counter at all; the path is {GPU_ENGINE_COUNTER_PATH!r}."
            )
        if not handle:
            raise PdhUnavailableError(
                f"PdhAddEnglishCounterW reported success for {path!r} but returned a NULL "
                f"counter handle. Refusing to use it: every subsequent pdh.dll call on this "
                f"machine dereferences a NULL handle and terminates the process with "
                f"0xC0000005, which is not catchable. Counter {counter_name!r} in category "
                f"{self._category!r} is therefore unavailable, not zero."
            )
        return handle

    @property
    def category(self) -> str:
        return self._category

    @property
    def counters(self) -> dict[str, str]:
        """The alias -> counter name mapping this query was opened with."""
        return dict(self._counters)

    @property
    def counter_name(self) -> str:
        """The single counter name this query reads. Refuses an ambiguous multi-counter query."""
        if len(self._counters) != 1:
            raise PdhUnavailableError(
                f"this query holds {len(self._counters)} counters "
                f"({sorted(self._counters.values())}); PdhCounterPath names no counter, so "
                f"a reading could not be attributed to one of them. Use one counter per "
                f"query, or read the aliases with the wildcard counter path directly."
            )
        return next(iter(self._counters.values()))

    @property
    def value_reads(self) -> bool:
        """Whether this query will attempt a native value read. See the module docstring."""
        return self._value_reads

    @property
    def primed(self) -> bool:
        """Whether a rate counter here has two samples to compute from."""
        return self._primed

    # -- enumeration ----------------------------------------------------------------

    def instances(self, *, refresh: bool = False) -> tuple[str, ...]:
        r"""Every instance name the category currently has, sorted and unique.

        These are the **real** names, parsed out of the `PdhExpandWildCardPathW` two-call
        expansion, reduced from `\GPU Engine(<name>)\Utilization Percentage` to `<name>`.
        The reduction is exact rather than a "text between the first bracket" guess,
        because the object and counter names are known here.

        Empty is a legitimate answer and a common one: on an idle machine no process is
        using a GPU engine, so Windows has no `GPU Engine` instances to report. That is
        "nothing is running", not "the counter is missing", and callers must not turn it
        into a zero.

        Measured cost on this machine: ~127 ms cold, ~2 ms warm, for 455-469 paths, so the
        result is cached for `instance_ttl`. Pass `refresh=True` to force a re-expansion -
        for example after a process that owns an engine has exited.
        """
        self._check_open()
        now = time.monotonic()
        if (not refresh and self._instance_cache
                and (now - self._instance_cache_at) < self._instance_ttl):
            return self._instance_cache
        status, paths = self._api.expand_wildcard(
            wildcard_path(self._category, self.counter_name)
        )
        self.last_status = status
        found: list[str] = []
        if status != 0:
            self.value_read_refusals.append(
                f"PdhExpandWildCardPathW for {self._category!r} returned "
                f"{format_pdh_status(status)}; the instance set is unknown, which is not "
                f"the same as empty and is certainly not zero"
            )
        for entry in paths:
            name = self._bare_instance(entry)
            if name and name not in found:
                found.append(name)
        self._instance_cache = tuple(sorted(found))
        self._instance_cache_at = now
        return self._instance_cache

    def instances_for_luid(self, luid: str, *, engtype: str | None = None) -> tuple[str, ...]:
        """The enumerated instances belonging to one adapter LUID.

        This is how the *owned subset* is selected. The device attribution stays in
        `app.telemetry.instance_map`: this method filters on the LUID the caller names and
        makes no claim about which device that is, so a LUID the recorded map leaves
        unattributed is still returned here rather than being quietly dropped by a filter
        that guesses.

        `engtype` is matched case-sensitively against the `engtype_` field when given; it
        is not a filter by default, because an instance with a name this module cannot
        parse is evidence and hiding it would be the same mistake as inventing it.
        """
        wanted = normalise_luid(luid)
        out: list[str] = []
        for name in self.instances():
            try:
                parsed = parse_instance_name(name)
            except ValueError:
                continue
            if parsed.luid != wanted:
                continue
            if engtype is not None and parsed.engtype != engtype:
                continue
            out.append(name)
        return tuple(out)

    def _bare_instance(self, expanded_path: str) -> str:
        r"""`\\host\\object(instance)\\counter` -> `instance`, or `""` if it is not that."""
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

    # -- the owned subset -----------------------------------------------------------

    def watch(self, names: Iterable[str]) -> tuple[str, ...]:
        """Open one concrete counter per owned instance, and prime them once.

        Returns the instances whose counter actually opened. An instance that cannot be
        opened is recorded in `value_read_refusals` and left out - it is not an error that
        stops the others, and it is never a zero.

        This is the read strategy on this machine, which has 455-469 live instances and a
        formatted array of 4199-4330 items. Reading the owned subset one counter at a time
        means the per-sample cost is the number of engines we care about, and does not
        grow with the number of processes on the machine. The handles live for the life of
        the query, so the 4 Hz path does no path building and no counter adds.
        """
        self._check_open()
        counter_name = self.counter_name
        opened: list[str] = []
        for name in names:
            instance = str(name)
            if instance in self._instance_handles:
                opened.append(instance)
                continue
            try:
                self._instance_handles[instance] = self._add(
                    self._api, instance_path(self._category, instance), counter_name,
                    f"instance:{instance}",
                )
            except PdhUnavailableError as exc:
                self.value_read_refusals.append(
                    f"instance {instance!r}: {exc}"
                )
                continue
            opened.append(instance)
        if self._prime_seconds and opened:
            self.prime()
        return tuple(opened)

    def watched(self) -> tuple[str, ...]:
        """The instances with a concrete counter open, in watch order."""
        return tuple(self._instance_handles)

    # -- reading --------------------------------------------------------------------

    def read(self, path: PdhCounterPath) -> float | None:
        """The utilisation percentage for one instance, or `None` when there is none.

        `None` is returned, never `0.0`, in every one of these cases, and each is recorded
        in `value_read_refusals`:

        - the path's category is not this query's category;
        - the query is closed;
        - **more than one counter was added to this query.** `PdhCounterPath` is a
          `(category, instance)` pair and says nothing about which counter the caller
          means;
        - the instance is not in the enumerated set - the case the plan's test pins. An
          engine whose process has exited has no utilisation, and reporting 0.0 for it is
          how a booth visitor is told the NPU is idle when nothing was measured;
        - the instance has no concrete counter open, i.e. it has not been `watch`ed;
        - the query is not primed, so a rate counter has one sample and no rate;
        - value reads are disabled on this machine (the default here) - see the module
          docstring. This check comes **before** the enumeration on purpose: with reads
          off there is nothing an enumeration could contribute, and skipping it keeps the
          4 Hz path off `pdh.dll` entirely;
        - the native call returns a non-zero status, or `dwCStatus != 0`. A rate counter
          that has not been primed reports `PDH_INVALID_DATA` here, and that is "not yet
          available", not zero;
        - the reading is outside `[0, 100]`, and is **refused rather than clamped**. A
          clamp would put a number on the card that was never measured: a 137.2% multi-
          engine sum displayed as 100.0% is a different number, and a wrong one;
        - the reading is not a double. Every other counter type is refused rather than
          reinterpreted, because reinterpreting an unknown type as a percentage is how a
          plausible-looking wrong number gets made.

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
        if len(self._counters) > 1:
            self.value_read_refusals.append(
                f"this query holds {len(self._counters)} counters "
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
        return self._read_value(str(path.instance))

    def read_many(self, names: Iterable[str]) -> dict[str, float | None]:
        """`read` for a set of instances, in one call, preserving order.

        A convenience over the owned subset: the per-instance verdicts are exactly the ones
        `read` would give, including `None` for an instance that is not there.
        """
        self._check_open()
        return {str(n): self.read(PdhCounterPath(self._category, str(n))) for n in names}

    def _read_value(self, instance: str) -> float | None:
        """The native value read, isolated so its refusal is recorded, not raised."""
        handle = self._instance_handles.get(instance)
        if not handle:
            self.value_read_refusals.append(
                f"instance {instance!r} has no concrete counter open; call watch() with it "
                f"first. A wildcard counter cannot be read for one instance, and reading "
                f"one anyway would be a number attributed by guesswork"
            )
            return None
        if not self._primed:
            self.value_read_refusals.append(
                f"instance {instance!r}: {self.counter_name!r} is a rate counter and this "
                f"query has not been primed with two PdhCollectQueryData samples separated "
                f"in time, so no rate exists yet. That is not zero."
            )
            return None
        try:
            status, cstatus, ctype, reading = self._api.formatted_value(handle)
        except OSError as exc:  # pragma: no cover - needs the native fault to be catchable
            self.value_read_refusals.append(
                f"the native value read for {instance!r} faulted: {exc}. A native fault is "
                f"normally not catchable at all, which is why this call is off by default."
            )
            return None
        self.last_status = status
        if status != 0:
            self.value_read_refusals.append(
                f"PdhGetFormattedCounterValue for {instance!r} returned "
                f"{format_pdh_status(status)}"
            )
            return None
        if cstatus != 0:
            self.value_read_refusals.append(
                f"{self.counter_name!r} for {instance!r} came back with dwCStatus "
                f"{format_pdh_status(cstatus)}: no data for this sample. Unavailable, "
                f"never zero."
            )
            return None
        if ctype != _PERF_DOUBLE:
            self.value_read_refusals.append(
                f"{self.counter_name!r} for {instance!r} came back as CType {ctype}, not "
                f"the double this provider reads; refusing to reinterpret an unknown "
                f"counter type as a percentage"
            )
            return None
        if reading != reading or not (PERCENT_MIN <= reading <= PERCENT_MAX):
            self.value_read_refusals.append(
                f"{self.counter_name!r} for {instance!r} read {reading!r}, outside "
                f"[{PERCENT_MIN}, {PERCENT_MAX}]; refused rather than clamped, because a "
                f"clamped value is a number the counter never reported"
            )
            return None
        return reading

    # -- sampling -------------------------------------------------------------------

    def collect(self) -> int:
        """`PdhCollectQueryData` for this query, returning the PDH status code.

        A formatted read needs a sample. Failures are **not** raised: a query that cannot
        be collected has no *new* sample, which is a stale reading rather than an absent
        one, and the staleness decision belongs to the caller that owns the 2 s rule.
        Measured working on this machine (status 0).
        """
        if self._closed or not self._query:
            return -1
        status = self._api.collect(self._query)
        self.last_status = status
        return status

    def prime(self) -> bool:
        """Two `PdhCollectQueryData` samples separated by `prime_seconds`.

        `Utilization Percentage` is a rate: with one sample there is no rate, and the read
        reports non-zero `dwCStatus`. That is "not yet available", and it is exactly what
        would be misread as a zero if it were not distinguished.

        Done at construction and after `watch`, **never per read**: a 4 Hz sampling path
        cannot afford a multi-second prime per sample. Returns whether the query is primed;
        a failed collect leaves it unprimed rather than pretending.
        """
        if self._closed or not self._query:
            return False
        first = self.collect()
        if self._prime_seconds:
            time.sleep(self._prime_seconds)
        second = self.collect()
        self._primed = first == 0 and second == 0
        if not self._primed:
            self.value_read_refusals.append(
                f"priming {self._category!r} failed: first collect "
                f"{format_pdh_status(first)}, second {format_pdh_status(second)}"
            )
        return self._primed

    # -- the two-call array pattern -------------------------------------------------

    def formatted_array_size(self) -> tuple[int, int, str]:
        """`(bytes, item_count, status_text)` from the `PdhGetFormattedCounterArrayW` probe.

        The plan names this function, and it is the half of the array pattern that is
        reliable. It is the **first** of the two calls: a NULL `ItemBuffer` with a
        `lpdwBufferSize` out-parameter, from which PDH reports the buffer size **in bytes**.
        The second call is `PdhQuery._read_formatted_array`, and it always passes a real
        buffer - a NULL `ItemBuffer` on a *sized* call is what faulted the process in the
        earlier implementation.

        Measured on this machine, primed: `0x800007D2` with 69 kB, i.e. 4328 items for the
        wildcard counter. Unprimed, the same call returns `0x800007D5` and **zero** bytes -
        so call `prime()` first if the size matters. Those items are **not** percentages on
        this build; see `_read_formatted_array`.
        """
        self._check_open()
        if len(self._handles) != 1:
            raise PdhUnavailableError(
                f"the array pattern is per counter and this query holds "
                f"{len(self._handles)}; use one counter per query"
            )
        status, size = self._api.formatted_array_size(
            self._next(iter(self._handles.values()))
        )
        self.last_status = status
        if status not in _BUFFER_NOT_FILLED:
            self.value_read_refusals.append(
                f"PdhGetFormattedCounterArrayW size probe returned {format_pdh_status(status)}"
            )
        elif status == PDH_MORE_DATA_UNPRIMED:
            self.value_read_refusals.append(
                "the array size probe returned "
                f"{format_pdh_status(PDH_MORE_DATA_UNPRIMED)} and 0 bytes: the counter has "
                "no sample yet. Call prime() before asking how large the array is."
            )
        return size, size // FMT_COUNTERVALUE_SIZE, format_pdh_status(status)

    def _read_formatted_array(self, max_items: int = 16) -> tuple[int, int, tuple[int, ...]]:
        """The **second** call of the array pattern, with a correctly sized real buffer.

        The buffer is sized from the probe plus a small slack - a deliberately short buffer
        is answered with `PDH_MORE_DATA` again, and the probed size is not always an exact
        multiple of the item size, so asking for "just eight" is not a thing this API can
        do. Only the first `max_items` are reported back, so a 4328-item wildcard counter
        does not put 69 kB of instance names on the caller's stack.
        """
        self._check_open()
        status, size = self._api.formatted_array_size(
            self._next(iter(self._handles.values()))
        )
        if status != PDH_MORE_DATA or size == 0:
            return status, 0, ()
        needed = size // FMT_COUNTERVALUE_SIZE + 8
        status, written, statuses = self._api.formatted_array(
            self._next(iter(self._handles.values())), needed
        )
        self.last_status = status
        return status, min(written, max(1, int(max_items))), statuses[:max(1, int(max_items))]

    def formatted_array_probe(self, max_items: int = 16) -> tuple[str, int, int, tuple[int, ...]]:
        """`(status_text, items_written, item_count_from_probe, per_item_status)` for the
        two-call array pattern, run end to end.

        Exposed so the argument order is pinned by a test against the **real** `pdh.dll`:
        a NULL-buffer probe, then a real buffer sized in bytes. On this build the items PDH
        writes are the instance-name list rather than `PDH_FMT_COUNTERVALUE` entries, so
        the per-item words are reported raw and no percentage is derived from them.
        """
        self._check_open()
        _, expected, _ = self.formatted_array_size()
        status, written, statuses = self._read_formatted_array(max_items)
        return format_pdh_status(status), written, expected, statuses

    # -- teardown -------------------------------------------------------------------

    def close(self) -> None:
        """Close the query. Idempotent, and safe to call from a `finally:` block.

        A second call is a no-op rather than a double `PdhCloseQuery`, so a caller cannot
        corrupt a handle that has already been released. `PdhCloseQuery` releases the
        counters with the query, so they are not removed individually - measured `rc = 0`
        for both, and removing them first would add one native call per owned instance to
        teardown for no benefit.
        """
        if self._closed:
            return
        self._closed = True
        self._instance_handles.clear()
        self._instance_cache = ()
        if getattr(self, "_query", 0):
            self._api.close_query(self._query)
            self._query = 0

    @staticmethod
    def _next(values: Iterable[int]) -> int:
        for value in values:
            return value
        raise PdhUnavailableError("this query holds no open counter")

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

    #: Class-level defaults, so `close()` and `__del__` are safe on an object whose
    #: `__init__` raised before it finished. A half-built query must not turn cleanup
    #: into a second exception.
    _api: Any = None
    _query: int = 0
    _closed: bool = True
    _handles: dict[str, int] = {}


#: `PERF_DOUBLE`, the `CType` of a counter measured as a `double`. The only `CType` this
#: module will read a `doubleValue` out of; every other is refused as `None` rather than
#: reinterpreted. It is not observed from `pdh.dll` on this machine - the value read does
#: not return - so it is checked for, never assumed.
_PERF_DOUBLE: Final[int] = 5
