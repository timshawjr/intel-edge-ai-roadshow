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

### 3. The value read was mis-bound, and that was the "fault". Corrected 2026-10-27.

The M0 record said the native value read terminates the process with `0xC0000005`. That is
reproducible, and it was never a PDH defect: **the binding passed three arguments to a
four-argument function.**

Measured on this machine (same interpreter, same process, exit code observed from a child
process because a native fault is not catchable):

| binding | call | result |
|---|---|---|
| `(hCounter, pValue, lpType)` - the old one | `PdhGetFormattedCounterValue` | **exit `0xC0000005`**: `dwFormat` read `pValue` as the format, `lpType` read the old third argument, and the real `pValue` slot held garbage |
| `(hCounter, dwFormat, lpType, pValue)` - the documented one | `PdhGetFormattedCounterValue` | `rc = 0`, `CStatus = 0`, `\Memory\Available Bytes` = 25,606,934,528.0, `\Processor(_Total)\% Processor Time` = 10.0 |

The same class of error explains the "array writes the instance-name list" finding: a
`PDH_FMT_COUNTERVALUE_ITEM_W` is **24 bytes** - an 8-byte `LPWSTR szName` followed by the
16-byte `PDH_FMT_COUNTERVALUE` - and the old reader walked it as 16-byte values, so every
item after the first was read at the wrong offset. With the real struct the array read
returns names *and* percentages: 335 instances, a busy `engtype_Compute` instance at
97.91%.

So `PdhGetFormattedCounterValue` and `PdhGetFormattedCounterArrayW` are both usable here,
and the refusal they used to produce was a bug in this module, not a property of the
machine. `VALUE_READS_ENABLED_BY_DEFAULT` is now `True`, and `PdhQuery.read_array()` is the
reader the accelerator provider uses: one wildcard counter, one array read per sample,
names and percentages together, so a value cannot be attributed to the wrong instance.

**Why the array read and not one concrete counter per instance.** A `GPU Engine` instance
comes and goes with its process's engine; a concrete handle for an instance that has
vanished returns `PDH_NO_DATA` on every later collect (measured: `0x800007D5`), and
re-adding handles per sample is path building at 4 Hz. The array read is one call over the
live set. On this machine it costs **0.37-0.84 ms** for 335 items (n=30: p50 0.38, p95 0.72),
which fits the 250 ms budget with no priming per sample.

The reading path is exercised with an injected `PdhApi`, so its decision logic is pinned
without a native call. The **real** `pdh.dll` is used for everything that is real: the
counter name, the add, the wildcard enumeration, the array size probe and values, the handle
guards, and `close`.

### 4. The read strategies: the array path for sampling, the owned subset for one instance.

The array path is what the accelerator provider uses (`read_array()`): one
`PdhCollectQueryData` plus one `PdhGetFormattedCounterArrayW`, names and percentages
together. This machine has 335-361 live instances (it had 455-497 on 2026-10-01, under
different load); the array buffer measured 49,352 bytes and the call 0.37-0.84 ms, so reading
the whole set at 4 Hz is affordable and is one native call, not one per instance.

`PdhQuery` also keeps the **owned subset** API, for a caller that wants one named instance:

1. `instances()` enumerates the real names once and caches them (`instance_ttl`).
2. The caller selects the owned subset with `instances_for_luid(luid)`.
3. `watch(names)` opens **one concrete counter per owned instance**, by adding
   `\GPU Engine(<bare instance>)\Utilization Percentage`, and primes them **once**.
4. `read()` then takes one `PdhGetFormattedCounterValue` per watched instance.

That path is not the sampling path: an instance comes and goes with its engine, and a
concrete handle for a vanished instance returns `PDH_NO_DATA` on every later collect
(measured `0x800007D5`). Only the array read sees the live set as it is.

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

### 6. The refusal log is bounded. Fixed 2026-10-01 after a scoped re-review.

`value_read_refusals` was a `list` that grew forever, and the 4 Hz path appends to it once
per instance per tick. Measured on this machine, 60 s of `collect()` + `read_many(40)`:

```
9600 refusals, 8,841,600 characters, 160 refusals/s, ~1012 MiB/hour of character count
and 9600 still held after close()
```

Two things about those numbers, both measured:

- The retained cost is **smaller** than the character count suggests. All 9600 entries are
  the *same* `VALUE_READS_REFUSED_REASON` string object - `len({id(s) for s in log}) == 1` -
  so the heap holds 9600 references (8 bytes each, ~75 KiB) plus one 921-character string.
  The character count overstates the real allocation by ~100x in this specific case. It is
  not always one shared string: with value reads enabled and several distinct refusal
  reasons in play the strings do differ, and then the character count is the real cost. The
  bound is required either way, because an unbounded list fails M6's "no continuing memory
  growth" gate on the direction alone.
- `close()` did **not** clear it, so the growth outlived the query.

The log is now a `collections.deque` with `maxlen=MAX_VALUE_READ_REFUSALS` (512), which
drops the **oldest** entry - the useful direction, since a diagnostic arriving at a booth
needs the current symptom. `PdhQuery.refusal_count` carries the unbounded total as an `int`,
so "are refusals still happening?" survives the bound. Every refusal path goes through one
`_refuse` helper, so the bound cannot be bypassed by a path that forgets it. `close()` leaves
the log alone on purpose: it is already bounded, and the post-mortem is exactly when someone
goes looking for the reason.
"""

from __future__ import annotations

import ctypes
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from app.telemetry.instance_map import normalise_luid, parse_instance_name

__all__ = [
    "FMT_COUNTERVALUE_ITEM_SIZE",
    "FMT_COUNTERVALUE_SIZE",
    "GPU_ENGINE_COUNTER",
    "GPU_ENGINE_COUNTER_PATH",
    "GPU_ENGINE_COUNTER_SET",
    "INSTANCE_TTL_SECONDS",
    "MAX_VALUE_READ_REFUSALS",
    "PDH_DLL",
    "PDH_FMT_DOUBLE",
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

#: The reason a query gives when value reads are explicitly turned **off** - not a claim
#: about this machine. `VALUE_READS_ENABLED_BY_DEFAULT` is `True` because the read works;
#: a caller that passes `enable_value_reads=False` gets this stated refusal instead of a
#: silent zero, and `read_many`/`read` never reach `pdh.dll`.
VALUE_READS_REFUSED_REASON: Final[str] = (
    "value reads are disabled for this query (enable_value_reads=False), so no counter "
    "value was read. This is a caller's choice, not a fault in the counter: the PDH value "
    "read works on this machine and returns a double with dwCStatus==0. Every reading is "
    "therefore None (unavailable), never 0."
)

#: `True`: measured on 2026-10-27, with the four-argument binding, the value read returns
#: a real double (`\Memory\Available Bytes` 25,606,934,528.0; `\Processor(_Total)\%...`
#: 10.0) and the array read returns per-instance percentages. It defaults on because the
#: booth's accelerator cards have no other per-process source, and a card that cannot
#: read has to say so rather than show 0.
VALUE_READS_ENABLED_BY_DEFAULT: Final[bool] = True

#: **How many refusal reasons are retained, newest kept.** Measured 2026-10-01: the 4 Hz
#: path (`collect()` + `read_many`) over 40 owned instances appends **160 refusals a
#: second**, one per instance per tick, and each one is `VALUE_READS_REFUSED_REASON` in
#: full. Unbounded, that is a list that grows for as long as the application runs, and M6's
#: gate is a four-hour soak with no continuing memory growth. 60 s of it measured 9600
#: entries; four hours at that rate is 2.3 million.
#:
#: 512 is chosen so that the *current* symptom is always in hand - which is what a
#: diagnostic arriving at a booth needs - while the retained cost is bounded by a few tens
#: of kilobytes whatever the refusal text is. At the measured 160/s that is ~3 s of
#: history; the trade is deliberate, and `PdhQuery.refusal_count` carries the unbounded
#: total as an `int` so "are refusals still happening?" survives the bound even though the
#: individual strings do not.
MAX_VALUE_READ_REFUSALS: Final[int] = 512


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
    """`PDH_FMT_COUNTERVALUE`. Measured `ctypes.sizeof` on this machine: 16, value at 8.

    The first field is `CStatus`, not a counter type. `dwType` - the third argument of
    `PdhGetFormattedCounterValue` - was measured **not** to be filled meaningfully here
    (`0x00010100` for `\\Memory\\Available Bytes`, `0x21510500` for
    `\\Processor(_Total)\\% Processor Time`, unchanged when `dwFormat` changed), so it is
    reported as a diagnostic and is never the validity gate. `CStatus` is the gate.
    """

    _fields_ = [("CStatus", ctypes.c_uint32), ("value", _PdhValueUnion)]


FMT_COUNTERVALUE_SIZE: Final[int] = ctypes.sizeof(_PdhFmtCountervalue)


class _PdhFmtCountervalueItem(ctypes.Structure):
    """`PDH_FMT_COUNTERVALUE_ITEM_W`: `LPWSTR szName` then `PDH_FMT_COUNTERVALUE`.

    Measured `ctypes.sizeof` on this machine: **24** (8 + 16), and 335 items for the
    wildcard `GPU Engine` counter come back with their real names and percentages when the
    buffer is walked with this stride. The earlier 16-byte stride is why the same call was
    reported as writing "the instance-name list".
    """

    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _PdhFmtCountervalue)]


FMT_COUNTERVALUE_ITEM_SIZE: Final[int] = ctypes.sizeof(_PdhFmtCountervalueItem)

#: `PDH_FMT_DOUBLE`. The value read is always requested in this format, so the union's
#: `doubleValue` is the reading; no other member is ever interpreted as a percentage.
PDH_FMT_DOUBLE: Final[int] = 0x00000200


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
       native surface substitutable, so the reader's decisions can be tested without a
       native call. Pass a substitute to `PdhQuery(api=...)`.

    Never call `formatted_value` or `formatted_array_values` through this class without a
    handle obtained from `add_counter` and checked non-zero. `pdh.dll` dereferences a NULL
    handle rather than returning a status - measured, `0xC0000005`.
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
        # SIX parameters: (hCounter, dwFormat, lpdwBufferSize, lpdwItemCount,
        # ItemBuffer, lpdwStatus). lpdwBufferSize is in BYTES, not items, and ItemBuffer
        # is a byte buffer walked as `PDH_FMT_COUNTERVALUE_ITEM_W` (24 bytes) - not as
        # 16-byte `PDH_FMT_COUNTERVALUE`s. `c_void_p` so a `create_string_buffer` can be
        # passed without a cast that hides the stride.
        lib.PdhGetFormattedCounterArrayW.argtypes = [
            handle, dword, lpdword, lpdword, buffer, lpdword
        ]
        lib.PdhGetFormattedCounterArrayW.restype = dword
        # FOUR parameters: (hCounter, dwFormat, lpdwType, pValue). The M0 binding passed
        # three - (hCounter, pValue, lpdwType) - so `dwFormat` read the value pointer and
        # the real `pValue` slot held garbage: that is the measured 0xC0000005, not a PDH
        # defect. `lpdwType` is filled unpredictably here and is diagnostic only.
        lib.PdhGetFormattedCounterValue.argtypes = [
            handle, dword, lpdword, ctypes.POINTER(_PdhFmtCountervalue)
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
        `\\GPU Engine(*)\\Utilization Percentage`: `0x800007D2` then `0` with 335-497 paths.

        **The whole pattern is retried up to three times.** The `GPU Engine` instance set
        changes whenever a process opens or closes an engine, and a set that grows between
        call one and call two makes call two return `PDH_MORE_DATA` again with a larger
        size. Measured on this machine: the five-parameter expansion returned `()` with
        `0x800007D2` in a suite run while worker processes were exiting, and the same call
        succeeded immediately on a retry. A single attempt would make `instances()` flap
        empty at 4 Hz - which is an absence, not a zero, but it is an absence caused by our
        own call pattern rather than by the machine.
        """
        status = -1
        for _attempt in range(3):
            chars = ctypes.c_uint32(0)
            size = ctypes.c_uint32(0)
            status = int(self._lib.PdhExpandWildCardPathW(
                None, path, None, ctypes.byref(chars), ctypes.byref(size)
            ))
            if status not in _BUFFER_NOT_FILLED or chars.value == 0:
                return status, ()
            buffer = ctypes.create_unicode_buffer(chars.value + 4096)
            chars_in, size_in = (
                ctypes.c_uint32(chars.value + 2048), ctypes.c_uint32(size.value + 8)
            )
            status = int(self._lib.PdhExpandWildCardPathW(
                None, path, ctypes.cast(buffer, ctypes.c_void_p),
                ctypes.byref(chars_in), ctypes.byref(size_in),
            ))
            if status == 0:
                return status, tuple(e for e in buffer[:].split("\0") if e)
            if status != PDH_MORE_DATA:
                return status, ()
        return status, ()

    # -- values ---------------------------------------------------------------------

    def formatted_value(
        self, counter: int, dw_format: int = PDH_FMT_DOUBLE
    ) -> tuple[int, int, int, float]:
        """`(status, dwCStatus, dwType, doubleValue)` for one concrete counter.

        The four-argument call. `dwFormat` is passed explicitly and is `PDH_FMT_DOUBLE` by
        default, so `doubleValue` is the reading. `dwType` is returned for diagnostics and
        was measured **not** to be filled meaningfully on this build; it is not a gate.
        `dwCStatus` is the gate: non-zero means "no data for this sample", never zero.

        Measured on this machine, 2026-10-27: `rc = 0`, `CStatus = 0`, and
        `\\Memory\\Available Bytes` returns 25,560,748,032.0. The old three-argument form
        terminated the process with `0xC0000005`.
        """
        value = _PdhFmtCountervalue()
        value_type = ctypes.c_uint32(0)
        status = int(self._lib.PdhGetFormattedCounterValue(
            ctypes.c_void_p(counter), dw_format, ctypes.byref(value_type),
            ctypes.byref(value),
        ))
        return status, int(value.CStatus), int(value_type.value), float(value.value.doubleValue)

    def formatted_array_size(
        self, counter: int, dw_format: int = PDH_FMT_DOUBLE
    ) -> tuple[int, int]:
        """`(status, bytes)` - the NULL-buffer probe of the two-call array pattern.

        `lpdwBufferSize` is in **bytes**, and the caller divides by
        `FMT_COUNTERVALUE_ITEM_SIZE` to get an item count. Measured on this machine,
        primed: `0x800007D2` with 49,352 bytes for 335 `GPU Engine` items. Unprimed it
        returns `0x800007D5` with 0 bytes.
        """
        size, item_count, status_out = (
            ctypes.c_uint32(0), ctypes.c_uint32(0), ctypes.c_uint32(0)
        )
        status = int(self._lib.PdhGetFormattedCounterArrayW(
            ctypes.c_void_p(counter), dw_format, ctypes.byref(size),
            ctypes.byref(item_count), None, ctypes.byref(status_out),
        ))
        return status, int(size.value)

    def formatted_array(
        self, counter: int, items: int, dw_format: int = PDH_FMT_DOUBLE
    ) -> tuple[int, int, tuple[int, ...]]:
        """`(status, items filled, per-item CStatus words)` with a **real** byte buffer.

        The second half of the two-call pattern: `lpdwBufferSize` in **bytes**, a buffer
        large enough for the probe, and never a NULL `ItemBuffer`.

        **On output PDH reports the counter's whole item count, not how many it wrote.**
        Measured: asking for 8 items of a 335-item counter comes back with
        `lpdwBufferSize = 335 * 24`. The count is therefore clamped to the buffer that was
        actually supplied, or reading it would index past the allocation.

        The stride is `FMT_COUNTERVALUE_ITEM_SIZE` (24), **not** `FMT_COUNTERVALUE_SIZE`
        (16): each item is an 8-byte name pointer followed by the 16-byte value. The
        earlier 16-byte stride is why this was reported as writing "the instance-name
        list". `formatted_array_values` is the one that returns the names and readings.
        """
        size = ctypes.c_uint32(max(1, items) * FMT_COUNTERVALUE_ITEM_SIZE)
        item_count = ctypes.c_uint32(0)
        status_out = ctypes.c_uint32(0)
        buffer = ctypes.create_string_buffer(size.value)
        status = int(self._lib.PdhGetFormattedCounterArrayW(
            ctypes.c_void_p(counter), dw_format, ctypes.byref(size),
            ctypes.byref(item_count), ctypes.cast(buffer, ctypes.c_void_p),
            ctypes.byref(status_out),
        ))
        written = min(int(item_count.value), items, int(size.value) // FMT_COUNTERVALUE_ITEM_SIZE)
        walk = (_PdhFmtCountervalueItem * max(written, 0)).from_buffer(buffer)
        return status, written, tuple(int(walk[i].FmtValue.CStatus) for i in range(written))

    def formatted_array_values(
        self, counter: int, items: int, dw_format: int = PDH_FMT_DOUBLE
    ) -> tuple[int, tuple[tuple[str, int, float], ...]]:
        """`(status, ((instance_name, dwCStatus, double), ...))` from one array read.

        This is the per-process read the accelerator provider uses. The names come back
        **with** their values from one call, so a percentage cannot be paired with the
        wrong instance; the caller filters on the `pid_` field and never sees a machine-wide
        total. Each item carries its own `dwCStatus`, so a dead engine is distinguishable
        from a busy one.

        Measured on this machine: 335 items in 49,352 bytes, `rc = 0`, a busy
        `engtype_Compute` instance at 97.91%, in 0.37-0.84 ms (n=30).
        """
        size = ctypes.c_uint32(max(1, items) * FMT_COUNTERVALUE_ITEM_SIZE)
        item_count = ctypes.c_uint32(0)
        status_out = ctypes.c_uint32(0)
        buffer = ctypes.create_string_buffer(size.value)
        status = int(self._lib.PdhGetFormattedCounterArrayW(
            ctypes.c_void_p(counter), dw_format, ctypes.byref(size),
            ctypes.byref(item_count), ctypes.cast(buffer, ctypes.c_void_p),
            ctypes.byref(status_out),
        ))
        written = min(int(item_count.value), max(1, items),
                      int(size.value) // FMT_COUNTERVALUE_ITEM_SIZE)
        walk = (_PdhFmtCountervalueItem * max(written, 0)).from_buffer(buffer)
        rows = tuple(
            (str(walk[i].szName or ""), int(walk[i].FmtValue.CStatus),
             float(walk[i].FmtValue.value.doubleValue))
            for i in range(written)
        )
        return status, rows


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
    2. Only then, a value read, and only if `enable_value_reads` is on (the default). The
       read is the four-argument `PdhGetFormattedCounterValue`, measured working on this
       machine; see the module docstring. Every refusal is appended to
       `value_read_refusals` with its reason.
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
        #: **Bounded** to `MAX_VALUE_READ_REFUSALS` entries, oldest dropped: the 4 Hz path
        #: appends one of these per instance per tick, so an unbounded list here is a
        #: process-lifetime leak. See `MAX_VALUE_READ_REFUSALS` for the measured rate and
        #: `refusal_count` for the unbounded total, which costs one `int` per query.
        self.value_read_refusals: deque[str] = deque(
            maxlen=max(1, int(MAX_VALUE_READ_REFUSALS))
        )
        #: Every refusal ever appended, including the ones the bound has dropped. An `int`,
        #: so it cannot leak, and it is the number to watch for "refusals are still
        #: happening" once the reason strings have rolled over.
        self._refusal_count = 0
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
    def refusal_count(self) -> int:
        """How many refusals this query has recorded, including those the bound dropped.

        An `int`, so it cannot leak and it never loses information: this is the number to
        watch for "refusals are still happening" once the retained reason strings in
        `value_read_refusals` have rolled over at `MAX_VALUE_READ_REFUSALS`.
        """
        return self._refusal_count

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
            self._refuse(
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
                self._refuse(f"instance {instance!r}: {exc}")
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
        in `value_read_refusals` - a log **bounded** to `MAX_VALUE_READ_REFUSALS` entries,
        oldest dropped, because this method is called once per instance per tick and an
        unbounded log here is a process-lifetime leak. `refusal_count` carries the
        unbounded total.

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
        - value reads are disabled for this query (`enable_value_reads=False`). This check
          comes **before** the enumeration on purpose: with reads off there is nothing an
          enumeration could contribute, and skipping it keeps that path off `pdh.dll`
          entirely. When it fires, `VALUE_READS_REFUSED_REASON` is logged once per read;
        - the native call returns a non-zero status, or `dwCStatus != 0`. A rate counter
          that has not been primed reports `PDH_INVALID_DATA` here, and that is "not yet
          available", not zero;
        - the reading is outside `[0, 100]`, and is **refused rather than clamped**. A
          clamp would put a number on the card that was never measured: a 137.2% multi-
          engine sum displayed as 100.0% is a different number, and a wrong one;
        - the reading is `NaN`, which fails every comparison and would otherwise look
          fresh for ever.

        A value is returned only when it was actually read, `dwCStatus` was 0, and it fell
        inside the range. There is no code path in this method that produces a number the
        counter did not report. The `dwType` out-parameter is **not** a gate: measured on
        this build it is filled with stale values, so gating on it would refuse every real
        reading (see `_MEASURED_UNFILLED_TYPES`).
        """
        if str(path.category) != self._category:
            reason = (f"path category {path.category!r} is not this query's category "
                      f"{self._category!r}")
            self._refuse(reason)
            return None
        if self._closed:
            self._refuse("query is closed")
            return None
        if len(self._counters) > 1:
            self._refuse(
                f"this query holds {len(self._counters)} counters "
                f"({sorted(self._counters.values())}) and PdhCounterPath names no "
                f"counter, so an instance of several of them cannot be attributed to one "
                f"reading. Use one counter per query."
            )
            return None
        if not self._value_reads:
            self._refuse(VALUE_READS_REFUSED_REASON)
            return None
        if str(path.instance) not in self.instances():
            # Not a failure to read: the instance is not there. This is the single most
            # important return in the module.
            self._refuse(
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

    def _refuse(self, reason: str) -> None:
        """Record one refusal: bump the unbounded count, append to the bounded log.

        Every refusal path in this module goes through here, so the bound cannot be
        bypassed by a path that forgets it. `deque(maxlen=...)` drops the **oldest** entry,
        which is the useful direction: a diagnostic arriving at a booth needs the current
        symptom, not the one from four hours ago.
        """
        self._refusal_count += 1
        self.value_read_refusals.append(reason)

    def _read_value(self, instance: str) -> float | None:
        """The native value read, isolated so its refusal is recorded, not raised."""
        handle = self._instance_handles.get(instance)
        if not handle:
            self._refuse(
                f"instance {instance!r} has no concrete counter open; call watch() with it "
                f"first. A wildcard counter cannot be read for one instance, and reading "
                f"one anyway would be a number attributed by guesswork"
            )
            return None
        if not self._primed:
            self._refuse(
                f"instance {instance!r}: {self.counter_name!r} is a rate counter and this "
                f"query has not been primed with two PdhCollectQueryData samples separated "
                f"in time, so no rate exists yet. That is not zero."
            )
            return None
        try:
            status, cstatus, ctype, reading = self._api.formatted_value(handle)
        except OSError as exc:  # pragma: no cover - a native fault is not normally catchable
            self._refuse(
                f"the native value read for {instance!r} raised {exc}. No value is "
                f"available."
            )
            return None
        self.last_status = status
        if status != 0:
            self._refuse(
                f"PdhGetFormattedCounterValue for {instance!r} returned "
                f"{format_pdh_status(status)}"
            )
            return None
        if cstatus != 0:
            self._refuse(
                f"{self.counter_name!r} for {instance!r} came back with dwCStatus "
                f"{format_pdh_status(cstatus)}: no data for this sample. Unavailable, "
                f"never zero."
            )
            return None
        if ctype and ctype != _PERF_DOUBLE and ctype not in _MEASURED_UNFILLED_TYPES:
            # Kept as a diagnostic, not as a gate. `dwType` was measured not to be filled
            # meaningfully on this build (0x00010100 for a memory counter, 0x21510500 for a
            # CPU counter), so refusing on it would refuse every real reading. The gate is
            # `dwCStatus` above and the request format is `PDH_FMT_DOUBLE`, so the value
            # interpreted is always the double PDH was asked for.
            self._refuse(
                f"{self.counter_name!r} for {instance!r} reported dwType {ctype:#x}, which "
                f"is neither the requested PDH_FMT_DOUBLE nor a measured filler value; "
                f"refusing rather than guessing"
            )
            return None
        if reading != reading or not (PERCENT_MIN <= reading <= PERCENT_MAX):
            self._refuse(
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
            self._refuse(
                f"priming {self._category!r} failed: first collect "
                f"{format_pdh_status(first)}, second {format_pdh_status(second)}"
            )
        return self._primed

    # -- the two-call array pattern -------------------------------------------------

    def read_array(self) -> tuple[tuple[str, float], ...]:
        """Every instance's percentage from **one** array read, names paired with values.

        Returns `((instance_name, percentage), ...)` for the items whose `dwCStatus` is 0
        and whose value is a finite percentage in `[0, 100]`. An item with a non-zero
        status, a `NaN`, or an out-of-range value is **left out and recorded** in
        `value_read_refusals` - it is never turned into a 0. An empty tuple means "no
        instance has a reading": on an idle machine Windows has no `GPU Engine` instances
        for a process that has not used the GPU, which is an absence, not a zero.

        One call, one sample. The caller filters by `pid_` and by adapter LUID; this method
        makes no attribution decision of its own and never sums anything.

        Measured on this machine: 335-361 instances, 0.37-0.84 ms per call (n=30), `rc = 0`, and a
        busy worker's `engtype_Compute` instance at 97.91% while every other process sat at
        or near 0.
        """
        self._check_open()
        if len(self._handles) != 1:
            raise PdhUnavailableError(
                f"the array pattern is per counter and this query holds "
                f"{len(self._handles)}; use one counter per query"
            )
        # **One collect, then one array read.** Without the collect the counter is never
        # re-sampled and every "reading" is the rate from the last prime - measured: a
        # worker at 98% read 0.0 on every tick because its busy interval began after the
        # prime. That is a stale value presented as current, which is the defect class
        # this module exists to prevent.
        collect_status = self.collect()
        if collect_status != 0:
            self._refuse(
                f"PdhCollectQueryData for {self._category!r} returned "
                f"{format_pdh_status(collect_status)}; no new sample exists, so no value "
                f"is reported. Unavailable, not 0."
            )
            return ()
        handle = self._next(iter(self._handles.values()))
        status, size = self._api.formatted_array_size(handle)
        self.last_status = status
        if size == 0:
            self._refuse(
                f"the array size probe for {self._category!r} returned "
                f"{format_pdh_status(status)} and 0 bytes: no sample exists yet. Call "
                f"prime(); an absence of items is not a reading of zero."
            )
            return ()
        if status not in _BUFFER_NOT_FILLED:
            self._refuse(
                f"the array size probe for {self._category!r} returned "
                f"{format_pdh_status(status)}; no value was read"
            )
            return ()
        # Slack for instances appearing between the probe and the read, and because the
        # probed size is not guaranteed to be an exact multiple of the stride.
        items = size // FMT_COUNTERVALUE_ITEM_SIZE + 8
        status, rows = self._api.formatted_array_values(handle, items)
        self.last_status = status
        if status != 0:
            self._refuse(
                f"PdhGetFormattedCounterArrayW for {self._category!r} returned "
                f"{format_pdh_status(status)}; no reading is available"
            )
            return ()
        readings: list[tuple[str, float]] = []
        for name, cstatus, value in rows:
            if not name:
                continue
            if cstatus != 0:
                self._refuse(
                    f"instance {name!r} came back with dwCStatus {format_pdh_status(cstatus)}"
                    f": no data for this sample. Unavailable, never zero."
                )
                continue
            if value != value or not (PERCENT_MIN <= value <= PERCENT_MAX):
                self._refuse(
                    f"instance {name!r} read {value!r}, outside "
                    f"[{PERCENT_MIN}, {PERCENT_MAX}]; refused rather than clamped"
                )
                continue
            readings.append((name, float(value)))
        return tuple(readings)

    def formatted_array_size(self) -> tuple[int, int, str]:
        """`(bytes, item_count, status_text)` from the `PdhGetFormattedCounterArrayW` probe.

        This is the **first** of the two calls: a NULL `ItemBuffer` with a
        `lpdwBufferSize` out-parameter, from which PDH reports the buffer size **in bytes**.
        The second call passes a real buffer; a NULL `ItemBuffer` on a *sized* call is what
        the earlier implementation did, and it faulted the process.

        Measured on this machine, primed: `0x800007D2` with 49,352 bytes, i.e. 335 items for
        the wildcard counter (the item stride is `FMT_COUNTERVALUE_ITEM_SIZE`, 24 bytes).
        Unprimed, the same call returns `0x800007D5` and **zero** bytes - so call `prime()`
        first if the size matters. `read_array()` is the call that returns the percentages.
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
            self._refuse(
                f"PdhGetFormattedCounterArrayW size probe returned {format_pdh_status(status)}"
            )
        elif status == PDH_MORE_DATA_UNPRIMED:
            self._refuse(
                "the array size probe returned "
                f"{format_pdh_status(PDH_MORE_DATA_UNPRIMED)} and 0 bytes: the counter has "
                "no sample yet. Call prime() before asking how large the array is."
            )
        return size, size // FMT_COUNTERVALUE_ITEM_SIZE, format_pdh_status(status)

    def _read_formatted_array(self, max_items: int = 16) -> tuple[int, int, tuple[int, ...]]:
        """The **second** call of the array pattern, with a correctly sized real buffer.

        The buffer is sized from the probe plus a small slack - a deliberately short buffer
        is answered with `PDH_MORE_DATA` again, and the probed size is not always an exact
        multiple of the item size, so asking for "just eight" is not a thing this API can
        do. Only the first `max_items` are reported back, so a 335-item wildcard counter
        does not put 8 kB of instance names on the caller's stack.
        """
        self._check_open()
        status, size = self._api.formatted_array_size(
            self._next(iter(self._handles.values()))
        )
        if status != PDH_MORE_DATA or size == 0:
            return status, 0, ()
        needed = size // FMT_COUNTERVALUE_ITEM_SIZE + 8
        status, written, statuses = self._api.formatted_array(
            self._next(iter(self._handles.values())), needed
        )
        self.last_status = status
        return status, min(written, max(1, int(max_items))), statuses[:max(1, int(max_items))]

    def formatted_array_probe(self, max_items: int = 16) -> tuple[str, int, int, tuple[int, ...]]:
        """`(status_text, items_written, item_count_from_probe, per_item_status)` for the
        two-call array pattern, run end to end.

        Exposed so the argument order is pinned by a test against the **real** `pdh.dll`:
        a NULL-buffer probe, then a real buffer sized in bytes. Each item is walked with the
        24-byte `PDH_FMT_COUNTERVALUE_ITEM_W` stride, and `read_array()` is the call that
        returns the names and percentages together.
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

        **The refusal log survives this, deliberately.** It is already bounded to
        `MAX_VALUE_READ_REFUSALS`, so it cannot leak past close, and keeping it means a
        caller can still read the reason a reading was refused *after* the query is gone -
        which is the post-mortem case, and the only time anyone actually goes looking.
        Dropping it here would trade a bounded few kilobytes for losing the diagnostic
        exactly when it is wanted.
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


#: `PERF_DOUBLE`, the counter type a double-valued counter reports. The value read asks
#: PDH for `PDH_FMT_DOUBLE`, so the union member read is always the requested double; this
#: constant and `_MEASURED_UNFILLED_TYPES` are a **diagnostic** on the `dwType` out-param,
#: which was measured to be filled with stale/unrelated values on this build.
_PERF_DOUBLE: Final[int] = 5

#: The `dwType` words measured on this machine for real counters, in a clean subprocess:
#: `0x00010100` (`\Memory\Available Bytes`) and `0x21510500`
#: (`\Processor(_Total)\% Processor Time`), unchanged when `dwFormat` changed. They are not
#: counter types; they are what this `pdh.dll` leaves in the third out-parameter. Treating
#: them as a gate would refuse every real reading, so they are accepted as "unfilled" and
#: the reading is gated on `dwCStatus` instead.
_MEASURED_UNFILLED_TYPES: Final[frozenset[int]] = frozenset({0x00010100, 0x21510500, 0x20510500})
