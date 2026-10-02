"""`MetricCollector` - the 4 Hz background sampler, and what it refuses to say.

Three constraints shape this module, and all three are measured facts about this machine
rather than preferences:

**1. `DeviceRegistry.describe()` cannot be on this path.** Measured at **3.3-3.8 s** -
one memoised Windows PnP enumeration. At `interval = 0.25` the whole budget for a sample
is 250 ms, and the measured cost of a full sample here is **~0.03 ms with one tracked pid
and ~0.04 ms with two** (five repeats of 200 runs each, p50 of each: 0.028-0.062 ms for
one pid and 0.041-0.048 ms for two, timed with `perf_counter`). So this module
does not import `app.inference.devices`, `openvino`, or `app.telemetry.pdh` at all:
device discovery is a **startup** cost, paid once by whoever builds the device panel, and
a collector that reached for it would stall the telemetry thread for four seconds every
time the panel was rebuilt. `tests/test_collector.py` reads this module's own source and
fails if any of those names appears.

An earlier version of this paragraph put the sample cost at "~0.2 ms", which was an order
of magnitude too high; the measured figures are above.

**2. `psutil`'s first `cpu_percent()` returns `0.0`, and that zero is not a measurement.**
Measured here: `psutil.cpu_percent(interval=None)` returns exactly `0.0` on the first call
after import. There is no previous sample, so there is no delta, and `psutil` reports
nothing as `0.0` rather than as "no measurement". A booth card showing "CPU 0%" during
that first sample is stating the opposite of the truth. This module therefore **primes at
construction**, discards the priming value, and marks the very first `sample()` as
`UNAVAILABLE` with a stated reason rather than publishing that zero.

The second call's value depends entirely on the machine's load, and this module used to
quote one without saying what the load was. Measured here, 8 logical cores, 300 ms
windows, five readings each:

    idle                   9.8 - 21.4   (the collector and the test harness are work)
    one core saturated    19.7 - 25.0   (one native busy loop)
    four cores saturated  54.6 - 60.4   (four native busy loops; 4/8 = 50%)

A single core of eight is 12.5% of the machine, so the "51.3 on the second call" this
module previously quoted - 51.3% of 8 cores being 4.1 cores busy - described roughly four
saturated cores, not one. The load is now stated with every figure.

**3. A process that died is not an idle process.** `psutil.Process(pid)` raises
`NoSuchProcess` for a pid that has exited. A dead worker process is exactly the thing an
operator needs to see at a booth, and summing it as `0.0` would report "the worker is
idle" for a worker that has crashed. So a dead pid makes the aggregate **`None`** with
the pid named in the reason, and its own per-process sample is `UNAVAILABLE`. It is
never `0.0`.

**App CPU is normalised to the machine, not to a core.** `psutil` reports per-process
CPU as a percentage that may exceed 100 on a multi-core host - a process using two of
this machine's eight cores reads near 200. Summing that across the UI and the worker and
publishing it next to "CPU 0%" would put a number in the same column with a different
meaning, so the sum is divided by the logical core count and is bounded by 100 per core
of headroom. `cpu_capacity()` is that denominator, measured not assumed, and a
`cpu_count()` of `None` is a refusal rather than a guess.

**`rss` is host process memory and is never presented as accelerator memory.**
`psutil.Process.memory_info().rss` is this process's resident set size in the host's RAM.
Every memory sample here carries `device_scope = None`, the unit `bytes`, and a
`process_scope` naming the process. No accelerator-memory figure is available from
`psutil` on this machine, so none is offered - and a card labelled "NPU VRAM" fed by this
number would be the most confident lie the booth could tell.

**The samples are produced on a background thread** (AGENTS.md rule 11) so a slow
measurement never blocks the frame path, and `sample()` is also callable directly for a
test or a one-shot reading. `stop()` is idempotent and returns promptly.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import psutil

from app.contracts.metric_sample import MetricSample, Validity

__all__ = ["MetricCollector", "cpu_capacity"]

PROVIDER = "psutil"

#: The reasons a reading can be absent, as text a presenter can read. `psutil` gives none
#: of these itself; the point is that "why" travels with the absence rather than being
#: reconstructed later from a number that is not there.
FIRST_WINDOW_REASON = (
    "no previous sample to compute a delta from: psutil.cpu_percent(interval=None) "
    "returns 0.0 on its first call after import, which measures nothing. The priming "
    "reading was discarded and this window is reported as unavailable rather than as "
    "the 0.0 psutil returned."
)
NO_PROCESSES_REASON = (
    "no process pids were supplied to the collector, so there is no application whose "
    "CPU could be measured. This is not an idle application."
)
NO_WINDOW_ELAPSED_REASON = (
    "no measurable window elapsed since the previous sample, so nothing was measured. "
    "The reported window is 0.0 - the measured value - not a substituted interval."
)

#: The most process labels an aggregate reason will name. Measured, not chosen for
#: tidiness - see `MetricCollector._aggregate_reason`.
MAX_REASONED_PROCESSES = 8


def cpu_capacity() -> int:
    """How many logical cores the machine has - the denominator for app CPU.

    Measured from `psutil`, not assumed, and **refusing** rather than defaulting when
    `psutil` cannot say: `cpu_count()` returns `None` on some platforms and when it
    cannot count, a default of 1 would report a two-core process's 190% as 190% instead
    of 24% - a number on the wrong scale rather than an absent one.
    """
    count = psutil.cpu_count(logical=True)
    if count is None:
        raise RuntimeError(
            "psutil.cpu_count(logical=True) returned None, so the app CPU cannot be "
            "normalised to the machine's capacity. A default of 1 would report a "
            "multi-core process's percentage on a single-core scale, which is a wrong "
            "number rather than a missing one."
        )
    count = int(count)
    if count <= 0:
        raise RuntimeError(
            f"psutil.cpu_count(logical=True) returned {count}, which is not a capacity. "
            f"The app CPU cannot be normalised against it."
        )
    return count


@dataclass(frozen=True)
class _Tick:
    """One tick's raw readings, and the window they were taken over.

    Frozen because it is a record of a measurement, and a measurement that can be edited
    after it was taken is not one. `window` is the elapsed time between the previous
    tick and the moment these readings were taken, which is the interval `psutil`
    actually integrated over.
    """

    now: float
    window: float
    system_cpu: float | None
    per_process_cpu: list[float | None]
    per_process_rss: list[int | None]


class _TrackedProcess:
    """One pid, its label, and a cached `psutil.Process` plus its last failure.

    **Every** read failure is recorded, not only the one from resolving the handle. That
    distinction was a real bug found by measurement. Over pids 1..199 on this machine,
    measured: **187 failed to resolve**, **12 resolved**, and of those 12, **9 then
    raised** when `cpu_percent()` was called on them - a zombie, or a process whose
    counters are denied. So "the handle exists" and "the handle can be read" are two
    different facts on Windows, and recording only the first left `_failure` as `None`
    when the second failed. The sample then carried `value=None` with no reason, and
    `MetricSample` correctly refused to build it: an absent reading with no stated
    reason is exactly the state this layer exists to prevent.

    The failure is **cached** so a pid that has exited does not raise out of every
    subsequent sample, and so the reason stays the one that was actually observed rather
    than a fresh string each tick - `_fail` is idempotent for a given kind of failure,
    which is what makes that true. See `_fail` for the measurement that forced it.
    """

    __slots__ = ("pid", "label", "_process", "_failure", "_failure_kind")

    def __init__(self, pid: int, label: str) -> None:
        self.pid = pid
        self.label = label
        self._process: Any = None
        self._failure: str | None = None
        self._failure_kind: str | None = None

    @property
    def failure(self) -> str | None:
        """Why this process's last read produced nothing, verbatim. `None` when it read."""
        return self._failure

    def _fail(self, exc: BaseException) -> None:
        """Record why this process could not be read.

        **A repeated failure of the same kind keeps the reason already recorded.** This is
        not an optimisation, it is a correctness requirement, and the reason was measured
        rather than imagined. `handle()` re-raises the recorded reason as the `msg` of a
        fresh `NoSuchProcess`, and `_fail` was re-recording on every read, so the text
        came to read

            NoSuchProcess: NoSuchProcess: NoSuchProcess: process PID not found (pid=1)

        and grew by one layer per reading - four times a second, per process. A reason
        that describes itself recursively stops being a reason, and a string that grows
        without bound on a 4 Hz path is a way to lose the process while explaining it.
        Keeping the first rendering makes `_fail` idempotent for a given kind, which is
        what the class docstring always claimed.

        **Rendering the exception is itself guarded.** The reason is built by
        interpolating a third-party exception, and a third party's `__str__` is not under
        this module's control. A sampler whose job is to keep producing readings cannot be
        taken down by a message it was writing - and the hazard is demonstrable, because
        `tests/test_collector.py` drives this branch with a `psutil.Error` subclass whose
        `__str__` raises and shows the unguarded interpolation propagating a `MemoryError`
        out of `sample()`, killing the tick.

        An earlier version of this docstring claimed such an exception had been
        **observed** on this machine over 199 pids. It was not, and it does not
        reproduce: measured, all 199 pids raised cleanly-rendering errors. The guard is
        kept because the hazard is real and the cost is one `try` - not because a
        measurement supports it.

        So the type name is always recorded, and the message is included only if
        rendering it is safe. A failure whose **kind** differs from the recorded one does
        replace it, because `AccessDenied` becoming `NoSuchProcess` is a real change in
        what is wrong with the process.
        """
        kind = type(exc).__name__
        if self._failure is not None and self._failure_kind == kind:
            return
        name = kind
        try:
            detail = str(exc)
        except Exception:  # noqa: BLE001 - explaining an error must not raise
            detail = None
        self._failure = f"{name}: {detail}" if detail else name
        self._failure_kind = kind

    def handle(self) -> Any:
        """The live `psutil.Process`, or raise the recorded reason.

        Resolves once and caches. `psutil.Process(pid)` is a constructor call that reads
        the process table, so calling it per sample at 4 Hz for each pid is waste on the
        one path where waste is measured against a 250 ms budget.
        """
        if self._process is not None:
            return self._process
        if self._failure is not None:
            # `msg=`, not the second positional: psutil's signature is
            # `NoSuchProcess(pid, name=None, msg=None)`, so a positional second argument
            # sets `name` and renders as
            # "process no longer exists (pid=5, name='NoSuchProcess: process no longer
            # exists (pid=5)')" - the reason quoted inside a reason, which reads as a
            # second failure rather than as the recorded one.
            raise psutil.NoSuchProcess(self.pid, msg=self._failure)
        try:
            self._process = psutil.Process(self.pid)
        except psutil.Error as exc:
            self._fail(exc)
            raise
        return self._process

    def cpu_percent(self) -> float | None:
        """This process's CPU since its previous call, or `None` with `failure` set."""
        try:
            return float(self.handle().cpu_percent(interval=None))
        except psutil.Error as exc:
            self._fail(exc)
            return None

    def rss_bytes(self) -> int | None:
        """Resident set size in **bytes**, or `None`. Host memory; never a device's."""
        try:
            return int(self.handle().memory_info().rss)
        except psutil.Error as exc:
            self._fail(exc)
            return None

    def prime(self) -> None:
        """Take the reading that the next `cpu_percent()` will be a delta against.

        The value is discarded. It is `0.0` by construction on the first call, and it is
        the reading that must never reach a screen.

        **A *read* failure here is not recorded.** Only `handle()` records, and only a
        failure to resolve does so; if the handle resolves but this one read fails, the
        reason is left for the first real sample to capture, because a reason recorded
        from a priming attempt would be reported as the current state on the strength of
        a discarded reading.
        """
        try:
            self.handle().cpu_percent(interval=None)
        except psutil.Error:
            pass


class MetricCollector:
    """Samples CPU and resident memory on a background thread and produces `MetricSample`s.

    `process_pids` are the processes whose CPU and RSS are attributed to the application
    as a whole. Labels come from `process_labels`, or default to `pid_<n>`; they are what
    appears in `process_scope`, so a card can say *which* process a figure belongs to.

    `interval` is the sampling period, **0.25 s (4 Hz)** by default, matching the design's
    telemetry cadence. The thread waits on an `Event` rather than sleeping, so `stop()`
    returns promptly instead of after up to one interval.

    Nothing here touches a widget, an OpenVINO device, or the network. The thread only
    reads `psutil` and hands out immutable records; how they reach a UI is the UI layer's
    problem, and AGENTS.md rule 11 requires that to be through signals.
    """

    def __init__(
        self,
        process_pids: Sequence[int],
        interval: float = 0.25,
        *,
        process_labels: Sequence[str] | None = None,
        background: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        interval = float(interval)
        if interval != interval or interval in (float("inf"), float("-inf")):
            raise ValueError(f"interval must be finite; got {interval!r}")
        if interval <= 0.0:
            raise ValueError(
                f"interval must be > 0; got {interval}. A zero or negative period is a "
                f"busy loop at 4 Hz, or never a sample."
            )
        pids = self._checked_pids(process_pids)
        labels = self._checked_labels(pids, process_labels)

        self._tracked = [_TrackedProcess(pid, labels[index]) for index, pid in enumerate(pids)]
        self._interval = interval
        self._clock = clock
        self._capacity = cpu_capacity()

        self._lock = threading.Lock()
        self._measure_lock = threading.Lock()
        self._tick: _Tick | None = None
        self._latest: tuple[MetricSample, ...] = ()
        self._samples_taken = 0
        # M-3: the reset comes **before** `_prime_global_cpu()`. It used to come two
        # lines after it, which silently discarded the reason the prime had recorded: an
        # instrumented run showed `_last_error` holding 'RuntimeError: prime exploded'
        # immediately after the prime and `None` by the end of `__init__`.
        self._last_error: str | None = None
        self._prime_global_cpu()
        for tracked in self._tracked:
            tracked.prime()
        self._window_started_at = float(self._clock())

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if background:
            self._thread = threading.Thread(
                target=self._run, name="metric-collector", daemon=True)
            self._thread.start()

    # -- construction helpers -------------------------------------------------------

    @staticmethod
    def _checked_pids(process_pids: Sequence[int]) -> tuple[int, ...]:
        if isinstance(process_pids, (str, bytes)):
            raise TypeError(
                f"process_pids must be a sequence of pids, not a {type(process_pids).__name__}"
            )
        try:
            candidates = list(process_pids)
        except TypeError as exc:
            raise TypeError(f"process_pids must be iterable: {exc}") from exc
        pids: list[int] = []
        for entry in candidates:
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise ValueError(
                    f"process_pids entries must be ints; got {entry!r} "
                    f"({type(entry).__name__}). A string pid would be looked up as a "
                    f"process name and nothing would be measured."
                )
            if entry <= 0:
                raise ValueError(
                    f"process_pids entries must be positive; got {entry}. A pid has to "
                    f"exist for a reading about it to mean anything."
                )
            pids.append(entry)
        return tuple(pids)

    @staticmethod
    def _checked_labels(pids: tuple[int, ...],
                        process_labels: Sequence[str] | None) -> tuple[str, ...]:
        if process_labels is None:
            return tuple(f"pid_{pid}" for pid in pids)
        labels = tuple(str(label) for label in process_labels)
        if len(labels) != len(pids):
            raise ValueError(
                f"{len(pids)} pids but {len(labels)} labels. Every pid this collector "
                f"reports on has to be nameable, because an unnamed process is a figure "
                f"a presenter cannot attribute."
            )
        for label in labels:
            if not label.strip():
                raise ValueError(
                    f"a process label is blank ({label!r}). A blank label is "
                    f"indistinguishable from no label."
                )
        return labels

    @staticmethod
    def _describe(exc: BaseException) -> str:
        """An exception as text, with the rendering itself guarded.

        Same hazard and same reason as `_TrackedProcess._fail`: the message is built from
        a third-party exception whose `__str__` is not ours, and `_run` calls this on the
        4 Hz path. If rendering raised, the failure of one tick would become the failure
        of every tick after it - the thread would stop producing evidence, and the only
        thing left on screen would be a frozen last reading. See `_TrackedProcess._fail`
        for the measured position on the `psutil.Error` claim, including the fact that
        the original observation did not reproduce.
        """
        name = type(exc).__name__
        try:
            detail = str(exc)
        except Exception:  # noqa: BLE001 - explaining an error must not raise
            detail = None
        return f"{name}: {detail}" if detail else name

    def _prime_global_cpu(self) -> None:
        try:
            psutil.cpu_percent(interval=None)
        except Exception as exc:  # noqa: BLE001 - the first reading's absence is state
            self._last_error = self._describe(exc)

    # -- the two direct measurements -------------------------------------------------

    def _readings(self) -> "_Tick":
        """This tick's raw readings, taken once and then reused.

        **This acquires `self._measure_lock` itself**, so both `sample()` and the two
        public accessors can call it. An earlier version documented "caller holds the
        lock" while the accessors called it without one - a documented precondition the
        code violated. It was benign in practice, because `_Tick` is frozen and
        publication is a single atomic attribute assignment, but it was not a contract
        anyone could rely on, and a second thread racing here could have taken two
        readings and kept the second.

        The measurement lock is **separate** from `self._lock`, which guards the sample
        set. `sample()` takes `_lock` and then `_measure_lock`; an accessor takes only
        `_measure_lock`; nothing takes `_lock` while holding `_measure_lock`. So the
        order is fixed, there is no cycle, and `_measure_lock` does not have to be
        reentrant.

        `psutil.cpu_percent(interval=None)` is a **delta since its previous call**, so the
        second reader of a tick does not re-measure the machine - it measures the
        microseconds since the first reader. That is not a rounding detail: a value read
        over ~0 s and a value read over the tick's real window are different claims, and
        the second one is the wrong one.

        So there is exactly one reading per tick, whoever asks first. The time the
        readings were taken is recorded with them, because the window has to be the
        interval `psutil` actually integrated over - not a wider one measured
        afterwards.
        """
        with self._measure_lock:
            if self._tick is not None:
                return self._tick
            now = float(self._clock())
            system = self._raw_system_cpu()
            per_process_cpu = [tracked.cpu_percent() for tracked in self._tracked]
            per_process_rss = [tracked.rss_bytes() for tracked in self._tracked]
            self._tick = _Tick(now=now, window=max(0.0, now - self._window_started_at),
                               system_cpu=system, per_process_cpu=per_process_cpu,
                               per_process_rss=per_process_rss)
            return self._tick

    @staticmethod
    @staticmethod
    def _raw_system_cpu() -> float | None:
        """The one direct `psutil` system-CPU read. `None` if it cannot be taken."""
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:  # noqa: BLE001 - unavailable is a result, not a crash
            return None

    def system_cpu_percent(self) -> float | None:
        """CPU across **all** logical cores, `0`-`100`, or `None` when unreadable.

        **This returns the current tick's reading, and takes it at most once.** A card
        that calls this between two `sample()` calls gets the figure `sample()` will
        report, not a second measurement over the microseconds since that figure was
        taken. See `_readings`.

        **Calling this moves the tick, and that is the cost.** Whoever reads first
        measures, so an accessor called immediately after a `sample()` takes the next
        reading at that moment rather than at the next tick. The `sample()` that follows
        then reports *the shorter window this call created* - and if the call left no time
        at all, that sample is `UNAVAILABLE` with `window_seconds = 0.0` and the reason
        "no measurable window elapsed since the previous sample". That is the correct
        outcome, but a card polling this on its own timer can flip the booth's sample to
        UNAVAILABLE with no other visible cause. Drive the read from the sampler's cadence
        rather than an independent timer.

        `None` - never `0.0` - when `psutil` cannot be read. A CPU reading that is not
        available is not zero.

        The docstring's example figures were wrong and are corrected in the module
        header; see the measured numbers there.
        """
        return self._readings().system_cpu

    def app_cpu_percent(self) -> float | None:
        """The summed CPU of every tracked pid, normalised to **0-100% of the machine**.

        Summed over the pids and then divided by `cpu_capacity()`, because `psutil`
        reports per-process CPU against a *single* core: on this eight-core machine a
        process using four cores reads near 400. Reporting that next to a system figure
        that cannot exceed 100 would put two different scales in one column.

        **`None` when any tracked pid cannot be read.** A dead worker's contribution is
        unknown, not zero, and a partial sum reported as the application's total is a
        number that understates by exactly the part that failed - which is the part an
        operator most needs to see.

        **Like `system_cpu_percent`, this reads at most once per tick** and then returns
        the cached reading, so calling it cannot corrupt the next `sample()`.

        **Calling this also moves the tick, and that is the cost.** An accessor called
        immediately after a `sample()` takes the next reading at that moment, so the
        following `sample()` reports the shorter window this call created - and reports
        `UNAVAILABLE` with `window_seconds = 0.0` if the call left no measurable time. A
        card polling this on its own timer can flip the booth's sample to UNAVAILABLE with
        no other visible cause. Drive the read from the sampler's cadence rather than an
        independent timer.
        """
        return _mean_over_capacity(self._readings().per_process_cpu, self._capacity)

    # -- the sample set --------------------------------------------------------------

    def sample(self) -> tuple[MetricSample, ...]:
        """One full set of readings, in a fixed order.

        **Each pid is read exactly once per tick, and every figure is derived from those
        same readings.** `psutil`'s `cpu_percent(interval=None)` is a delta since *its
        previous call*, so a second reader of the same tick does not re-measure the
        machine - it measures the microseconds since the first reader. An earlier version
        called `tracked.cpu_percent()` for the per-process list and then again inside
        `app_cpu_percent()`, which silently corrupted both; and because the two accessors
        were public, a card calling one between ticks corrupted the next `sample()` the
        same way while still reporting a plausible-looking number under the tick's full
        `window_seconds`. All of them now share `_readings()`, so whoever asks first
        takes the measurement and everyone reports the same one.

        **The very first sample after construction is `UNAVAILABLE`, and says why.**
        Priming happens in `__init__`, so the first real reading covers only the
        microseconds between construction and the first call. That window is not a
        measurement window: the percentage over it is noise, and on `psutil` it can be
        `0.0` for the same reason the pre-priming call was - which is a booth card
        reading "CPU 0%" on a machine that is about to be measured properly.

        `window_seconds` is the **measured** elapsed time since the previous sample, not
        the requested `interval`. If the thread falls behind, the window says so, and a
        reader can tell a 250 ms average from a 2 s one. A window of exactly zero - only
        reachable with an injected clock, since the default clock on this machine has
        ~16 ms resolution - is reported unavailable rather than substituted with
        `interval`, because a declared window that did not elapse is a window the number
        was not averaged over.
        """
        with self._lock:
            tick = self._readings()
            self._tick = None  # the next tick measures afresh
            now = tick.now
            window = tick.window
            first = self._samples_taken == 0
            self._window_started_at = now
            self._samples_taken += 1

            per_process_cpu = tick.per_process_cpu
            per_process_rss = tick.per_process_rss
            system = tick.system_cpu
            app_cpu = _mean_over_capacity(per_process_cpu, self._capacity)
            app_rss = _sum_or_none(per_process_rss)
            labels = tuple(tracked.label for tracked in self._tracked)

            # I-1: a failure is only a fact about *this* sample. `_TrackedProcess.failure`
            # is sticky by design - it caches the reason a pid died so the reason does
            # not change each tick - so it is not evidence that the pid failed now. The
            # current readings are, and a pid that read a number this sample is not named
            # in this sample's reason.
            cpu_failures = {tracked.label: tracked.failure
                            for tracked, value in zip(self._tracked, per_process_cpu)
                            if value is None and tracked.failure is not None}
            rss_failures = {tracked.label: tracked.failure
                            for tracked, value in zip(self._tracked, per_process_rss)
                            if value is None and tracked.failure is not None}

            unusable = (first or window <= 0.0)
            window_note = FIRST_WINDOW_REASON if first else (
                NO_WINDOW_ELAPSED_REASON if window <= 0.0 else None)
            # M-4: when no window elapsed, the reported window is the measured `0.0`, not
            # a placeholder. `MetricSample` permits a zero window only on an UNAVAILABLE
            # sample, and the sample carrying it is always one.
            reported_window = window

            samples: list[MetricSample] = [
                self._percent_sample(
                    "system_cpu_percent", system if not unusable else None,
                    reported_window, sampled_at=now, process_scope=(),
                    device_scope=None,
                    reason=self._reason(window_note, None if system is not None else
                                        (self._last_error or
                                         "psutil.cpu_percent did not return a reading"))),
                self._percent_sample(
                    "app_cpu_percent", app_cpu if not unusable else None,
                    reported_window, sampled_at=now, process_scope=labels,
                    device_scope=None,
                    reason=self._reason(window_note,
                                        self._aggregate_reason(cpu_failures,
                                                               per_process_cpu))),
                self._bytes_sample(
                    "app_rss_bytes", app_rss if not unusable else None,
                    reported_window, sampled_at=now, process_scope=labels,
                    device_scope=None,
                    reason=self._reason(window_note,
                                        self._aggregate_reason(rss_failures,
                                                               per_process_rss))),
            ]
            for index, tracked in enumerate(self._tracked):
                scope = (tracked.label,)
                samples.append(self._percent_sample(
                    f"cpu:{tracked.label}",
                    per_process_cpu[index] if not unusable else None,
                    reported_window, sampled_at=now, process_scope=scope,
                    device_scope=None,
                    reason=self._reason(window_note,
                                        cpu_failures.get(tracked.label))))
                samples.append(self._bytes_sample(
                    f"rss:{tracked.label}",
                    per_process_rss[index] if not unusable else None,
                    reported_window, sampled_at=now, process_scope=scope,
                    device_scope=None,
                    reason=self._reason(window_note,
                                        rss_failures.get(tracked.label))))

            self._latest = tuple(samples)
            return self._latest

    @staticmethod
    def _reason(window_note: str | None, own: str | None) -> str | None:
        """Both reasons when both apply, because dropping either loses information."""
        parts = [part for part in (window_note, own) if part]
        return " | ".join(parts) if parts else None

    def _aggregate_reason(self, failures: dict[str, str],
                          values: Sequence[float | int | None]) -> str | None:
        """Why an aggregate is absent, naming the processes that could not be read.

        **The number of processes named is bounded**, by `MAX_REASONED_PROCESSES`. This
        is a measured bound, not tidiness.

        **The "before" figures below are not reproducible from this commit** and are
        recorded as history, not as something to re-run: they came from an instrumented
        pre-fix run on 199 tracked pids, over ten samples, in which the recorded failure
        text compounded one `NoSuchProcess:` layer per reading and this method joined all
        199 of them twice per sample. RSS went

            56.2 -> 57.0 -> 57.5 -> 58.7 -> 59.9 -> 67.9 -> 95.0 -> 206.7 -> 647.1 -> 2411.9 MB

        and the eleventh tick raised `MemoryError`. **A reader cannot verify that from
        HEAD**, and should not be invited to try.

        **What is reproducible from HEAD** is that the bound holds. Measured after the
        fix, 199 tracked pids, twelve samples, two runs:

            57.4 -> 57.6 MB    and    57.6 -> 57.6 MB      (flat)

        and independently by the reviewer at 400 pids over 30 samples: RSS 56.6 -> 57.4 MB,
        flat from sample 2, with the aggregate reason constant at 673 characters. The
        post-fix figure to quote is **roughly 56-58 MB flat**, and the starting point moves
        by a megabyte or two between runs, which is why a single precise number here is
        over-claimed. An earlier version of this docstring said 56.6 MB, and the matching
        test said 44.1 MB for the same experiment - the latter was a different run's
        starting point rather than a settled value.

        What made it grow was the reason itself, not the process table: handles, thread
        count, `psutil._pmap` and the tracked Python object count were all flat
        throughout, and a control loop performing the same `psutil` reads without the
        collector did not move RSS at all (flat at 17.6 MB over ten rounds).

        So past the bound the reason says how many more there were rather than naming
        them. That loses no evidence the reader needs - the count is the number a
        presenter asks for, and the per-process samples still carry their own reasons.
        """
        if values is None or all(v is None for v in values):
            if not self._tracked:
                return NO_PROCESSES_REASON
        dead = sorted(label for label, reason in failures.items() if reason)
        if dead:
            named, omitted = _name_within_bound(dead)
            detail = "; ".join(f"{label}: {failures[label]}" for label in named)
            return (
                f"{', '.join(named)}{omitted} could not be read, so the application "
                f"total is unavailable rather than a partial sum. A process that has "
                f"exited is not an idle process. Detail: {detail}"
            )
        missing = sorted(
            tracked.label for tracked, value in zip(self._tracked, values)
            if value is None
        ) if values is not None else []
        if missing:
            named, omitted = _name_within_bound(missing)
            return (
                f"{', '.join(named)}{omitted} returned no reading this window. The "
                f"total is reported unavailable rather than as the sum of the rest, "
                f"because a partial sum understates the application by exactly the part "
                f"that could not be measured."
            )
        return None

    @staticmethod
    def _percent_sample(name: str, value: float | None, window: float, *,
                        sampled_at: float, process_scope: tuple[str, ...],
                        device_scope: str | None, reason: str | None) -> MetricSample:
        return MetricSample(
            name=name, value=value, unit="percent", provider=PROVIDER,
            device_scope=device_scope, process_scope=process_scope,
            window_seconds=window, sampled_at=sampled_at,
            validity=(Validity.VALID if value is not None else Validity.UNAVAILABLE),
            unavailable_reason=reason if value is None else None,
        )

    @staticmethod
    def _bytes_sample(name: str, value: int | None, window: float, *,
                      sampled_at: float, process_scope: tuple[str, ...],
                      device_scope: str | None, reason: str | None) -> MetricSample:
        return MetricSample(
            name=name,
            value=float(value) if value is not None else None,
            unit="bytes", provider=PROVIDER, device_scope=device_scope,
            process_scope=process_scope, window_seconds=window,
            sampled_at=sampled_at,
            validity=(Validity.VALID if value is not None else Validity.UNAVAILABLE),
            unavailable_reason=reason if value is None else None,
        )

    # -- the background thread -------------------------------------------------------

    def _run(self) -> None:
        """Sample every `interval` until `stop()`.

        Exceptions are caught and recorded in `last_error` rather than killing the
        thread: a collector that dies silently stops producing evidence, and the absence
        of a reason is the hardest thing to debug at a booth (AGENTS.md rule 12).
        """
        while not self._stop.wait(self._interval):
            try:
                self.sample()
            except Exception as exc:  # noqa: BLE001 - one bad tick is not a dead thread
                self._last_error = self._describe(exc)

    @property
    def running(self) -> bool:
        """Whether the background thread is alive. `False` for a manual collector."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def samples_taken(self) -> int:
        """How many sample sets this collector has produced. An `int`, so it cannot leak."""
        return self._samples_taken

    @property
    def interval(self) -> float:
        """The requested sampling period. See `sample()`'s `window_seconds` for the real one."""
        return self._interval

    @property
    def last_error(self) -> str | None:
        """The most recent collector-level failure, or `None`. Never a zero."""
        return self._last_error

    def latest(self) -> tuple[MetricSample, ...]:
        """The most recent sample set. `()` before the first one, never a fake empty set."""
        return self._latest

    def stop(self) -> None:
        """Stop the background thread and wait for it. Idempotent.

        Called from `__del__`-free teardown rather than a signal handler, and it is safe
        to call twice - a supervisor's shutdown path and a test's `addCleanup` both reach
        it.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self._interval * 4.0))
        self._thread = None

    def __enter__(self) -> "MetricCollector":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        labels = [tracked.label for tracked in self._tracked]
        return (f"MetricCollector(pids={labels!r}, interval={self._interval}s, "
                f"running={self.running}, samples={self._samples_taken})")


def _sum_or_none(values: Sequence[int | None]) -> int | None:
    """The sum, or `None` if any member is absent.

    A sum that quietly treats an absent member as zero is a number that understates by
    exactly the missing part, and the missing part is the interesting part.
    """
    if not values:
        return None
    total = 0
    for value in values:
        if value is None:
            return None
        total += int(value)
    return total


def _mean_over_capacity(values: Sequence[float | None], capacity: int) -> float | None:
    """Sum `values`, normalised to `0`-`100%` of `capacity`, or `None` if any is absent.

    The division is by the whole machine's logical core count because `psutil` reports
    per-process CPU against a *single* core: on this eight-core machine a process using
    four cores reads near 400, and printing that beside a system figure that cannot
    exceed 100 would put two different scales in one column.

    `None` - never a partial sum - when any member is absent. The whole point of review
    focus #3 is that "we could not measure that process" and "that process used no CPU"
    must not be the same number, and a partial sum is the first of those wearing the
    second's clothes.
    """
    if not values:
        return None
    total = 0.0
    for value in values:
        if value is None:
            return None
        total += float(value)
    return total / float(capacity)


def _name_within_bound(labels: Sequence[str],
                       limit: int = MAX_REASONED_PROCESSES) -> tuple[list[str], str]:
    """At most `limit` labels, and a phrase naming however many were left out.

    Returns `(named, omitted)`, where `omitted` is `""` when everything fitted and
    otherwise reads `" and 191 more"`. The count is the part a presenter asks for; the
    identities beyond the bound are still on the per-process samples.
    """
    ordered = list(labels)
    if len(ordered) <= limit:
        return ordered, ""
    extra = len(ordered) - limit
    return ordered[:limit], f" and {extra} more"
