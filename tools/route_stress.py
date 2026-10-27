"""M2 Task 3 - 50 alternating warm route changes, with leak and placement evidence.

    .venv\\Scripts\\python.exe -m tools.route_stress                 # the real 50
    .venv\\Scripts\\python.exe -m tools.route_stress --transitions 6

Writes `logs/qualification/m2-route-stress.json` and prints the distribution.

## What this measures, and what it deliberately does not

**The 2 s warm-transition gate.** Spec section 5 asks for a toggle to reach a *confirmed
placement* within two seconds. This runs 50 alternating warm toggles and reports the
**whole distribution** - every sample, min, p50, p95, max, and the count over 2 s.

**Three columns, because one of them was quietly hiding the gate.** `begin_transition` is
*synchronous*: it terminates and joins the old worker, retires its queues, drains, and
starts the replacement process, all before it returns an id. `TransitionOutcome.elapsed_s`
is measured from `_Transition.started`, which `begin_transition` stamps **after** that
work. So the figure M0-M2 reported as "the warm transition" is the confirmation *wait* and
leaves the teardown and the spawn out of it. The report prints

- `setup` - wall seconds inside `begin_transition`,
- `wait`  - `outcome.elapsed_s`, excluding `setup`,
- `total` - `setup + wait`, **the toggle**, and the number the gate is judged on.

Both readings are printed because only one of them is the spec's figure and the tool
should not decide that silently.

**The deadline is left at its real value.** `Supervisor.await_transition`'s default is
2.0 s and `--timeout-s` defaults to it. It is *not* widened to make samples pass. Note
what a timeout means for the distribution: the sample's `wait` is recorded **at** the
deadline (censored) and its `total` additionally includes the recovery, so both overstate
the confirmation time. `--timeout-s 30` is how the true tail is measured on purpose.

**Stale painting, observed rather than asserted.** Every `result_ready` delivery is
checked against the frame the stress itself submitted. The stress owns the frame ids: it
records, for each stream, the highest frame id it had submitted *at the moment
`begin_transition` was called*. Any result that arrives after that with a frame id at or
below the mark is a picture from the superseded generation reaching the UI - a stale
overlay - and is counted. This is the check Tasks 1-2 could not do, because they drove
`accept_result` with a chosen generation.

**Mislabelled devices, likewise.** Every delivered result must name the device its stream
is actually routed to, judged by `app.inference.devices.is_device` - the same helper the
production code uses, because `'GPU' in 'GPU.0'` is `True` as a substring test and
`False` as a tuple test, and one of those is a lie.

**Placement on the new pid.** For every transition that claims
`bounded_by='confirmed'`, the stress records the pid that was live when the confirming
result was delivered and the pid live when `await_transition` returned, and requires them
to agree. A confirmation carried by the *old* worker is exactly the defect this milestone
is about, so it is checked from the live path and never taken on trust.

**A real fallback.** M0's unmet gate 9 is open because every device on this machine
executes where it is asked. A genuine *silent* fallback therefore does not occur here and
this tool does not pretend otherwise. What it does produce is the **visible** half: the
`--fault-stream` option makes one camera's worker really die, and the run then records that
the stream recovered to a stated device with a stated reason, which is what Task 4's screen
work will assert on.

## Leak, with the retention subtracted rather than explained away

`rss_start`/`rss_end`/`leak_bytes` are the **host process** resident set (psutil), which
contains the OpenVINO runtime and the Python allocator's high-water mark, so it is a noisy
upper bound and not a per-stream figure.

`_transitions` and `_retired_queues` **grow by design** - a cap would drop an open record
and turn a legitimate `await_transition` into `ValueError` - so a raw `leak_bytes` would
attribute that retention to a leak. Two things are therefore measured and reported
separately:

- `retained_transitions` / `retained_queues` - the object counts that grew.
- `leak_bytes_after_release` - `rss` re-read **after** those containers have been emptied
  and `gc.collect()` has run. The difference between `leak_bytes` and that figure is the
  part of the growth explained by retention; what is left is what is not. Python's
  allocator does not return freed arenas to the OS, so this is a floor on the retained
  cost, not a subtraction that balances exactly - which is why both numbers are printed.

## The frame ages printed at the end are a different claim

`frame_age_p50_ms` / `frame_age_p95_ms` here are measured **under churn**: every stream is
being torn down and rebuilt while it is measured, and the all-CPU routing in the alternation
means three workers are competing for one CPU plugin. That is not the app's steady state.
The unsaturated spec section 5 row 4 figure (three streams on the CPU, no churn) is the
M2 headline latency number and is measured separately; the two must never be quoted as one.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from PySide6.QtCore import QCoreApplication, Qt

if __package__ in (None, ""):  # pragma: no cover - the `python tools/x.py` spelling
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts.frame import FramePacket
from app.inference.devices import is_device
from app.supervisor.runner import Supervisor

#: Spec section 5's four combinations, in the table's own order. **`table` walks these**,
#: which is the shape the 2 s gate is about: every toggle flips one hardware class, so
#: exactly one camera moves and the other two keep running. The `both` pattern alternates
#: row 1 against row 4, which moves **two** cameras at once - a change the spec table never
#: asks for, and measurably slower. Both are run and reported, because quoting the harder
#: one as if it were the gate (or the easier one as if it covered the harder) would each
#: be a lie in opposite directions.
ROW_HW = {"A": "CPU", "B": "GPU", "C": "NPU"}
ROW_GPU_OFF = {"A": "CPU", "B": "CPU", "C": "NPU"}
ROW_NPU_OFF = {"A": "CPU", "B": "GPU", "C": "CPU"}
ROW_CPU = {"A": "CPU", "B": "CPU", "C": "CPU"}

#: The table walk. Index 0 is the warm-up's own row, so the first toggle must move: the
#: walk starts at index 1.
TABLE = (ROW_GPU_OFF, ROW_NPU_OFF, ROW_CPU, ROW_HW)
#: Row 1 against row 4. Also starts off row 1, so the first toggle moves.
BOTH = (ROW_CPU, ROW_HW)

#: Named so the CLI and the record can both say which one ran.
PATTERNS = {"table": TABLE, "both": BOTH}

#: The gate. Spec section 5, restated here so the tool does not read it out of a docstring.
GATE_SECONDS = 2.0

#: Seconds to let the workers drain between the feeder stopping and `supervisor.stop()`.
#: See the `finally` block in `run_stress` for the measurement behind it.
SETTLE_SECONDS = 1.5

#: Frame size for the synthetic feed. 240x320 rather than 1920x1080, for the reason
#: `tests/test_runner.py`'s `_Feeder` gives: the worker letterboxes to the pinned 640x640,
#: and a 6 MB array per frame per stream at 45 frames/s would make this tool cost minutes
#: of pickling. The frame rate is the spec's 15 fps per stream.
FEED_SIZE = (240, 320)


@dataclass(frozen=True)
class TransitionSample:
    """One toggle, as measured. Nothing here is derived from another field."""

    index: int
    target: dict[str, str]
    acknowledged: bool
    bounded_by: str
    generation: int
    #: Wall seconds from `begin_transition` returning to `await_transition` returning.
    elapsed_s: float
    #: `outcome.elapsed_s` - the transition's own age, measured from when it *began*.
    #: **`_Transition.started` is stamped after `_apply_devices` returns**, so this
    #: EXCLUDES the synchronous teardown and spawn inside `begin_transition`. See
    #: `setup_s`, and see the module docstring: the two together are the toggle.
    transition_elapsed_s: float
    #: Wall seconds inside `begin_transition` itself - terminate/join/terminate the old
    #: worker, retire its queues, drain, and start the replacement process.
    setup_s: float
    unconfirmed: tuple[str, ...]
    confirmed: dict[str, tuple[str, ...]]
    discarded_results: int
    pids_at_confirm: dict[str, int | None]
    pids_after: dict[str, int | None]
    rss_before: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or self.index < 0:
            raise ValueError(f"index must be >= 0; got {self.index!r}")
        if self.bounded_by not in ("confirmed", "timeout"):
            raise ValueError(
                f"bounded_by must be one of the two spellings the supervisor emits; got "
                f"{self.bounded_by!r}")
        if self.acknowledged != (self.bounded_by == "confirmed"):
            raise ValueError(
                f"acknowledged={self.acknowledged!r} contradicts "
                f"bounded_by={self.bounded_by!r}; a transition is acknowledged only by a "
                f"confirmed placement")
        for name, value in (("elapsed_s", self.elapsed_s),
                            ("transition_elapsed_s", self.transition_elapsed_s),
                            ("setup_s", self.setup_s)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number; got {value!r}")
            if value != value or value < 0.0:
                raise ValueError(f"{name} must be a finite, non-negative number; got "
                                 f"{value!r}")
        if self.target and not self.confirmed and self.acknowledged:
            raise ValueError("acknowledged with no confirmed stream")
        if self.discarded_results < 0:
            raise ValueError(f"discarded_results must be >= 0; got "
                             f"{self.discarded_results}")
        if self.rss_before < 0:
            raise ValueError(f"rss_before must be >= 0; got {self.rss_before}")


@dataclass(frozen=True)
class StressResult:
    """The brief's record, plus the evidence the honesty requirements need.

    The brief's eleven fields are all here, with the brief's names. Everything after them
    exists because a verdict without its distribution, or a leak figure without its
    retention, is not reportable.
    """

    # -- the brief's fields ------------------------------------------------------------
    transitions: int
    warm_elapsed_p50: float
    warm_elapsed_p95: float
    warm_elapsed_max: float
    over_two_second: int
    discarded_results: int
    rss_start: int
    rss_end: int
    leak_bytes: int
    stale_overlays: int
    mislabelled_devices: int
    crashes: int

    # -- the distribution, not the verdict ---------------------------------------------
    warm_elapsed_min: float = 0.0
    warm_elapsed_samples: tuple[float, ...] = ()
    setup_p50: float = 0.0
    setup_p95: float = 0.0
    setup_max: float = 0.0
    setup_samples: tuple[float, ...] = ()
    wait_p50: float = 0.0
    wait_p95: float = 0.0
    wait_max: float = 0.0
    acknowledged: int = 0
    unconfirmed_samples: tuple[int, ...] = ()
    timeout_s: float = GATE_SECONDS
    censored_at_gate: bool = False
    #: Of those, how many were **acknowledged** - i.e. the toggle reached a confirmed
    #: placement at all. A censored sample's total includes the recovery that ran after
    #: the deadline, so it is over the gate by construction; this is the subset whose
    #: total is a real toggle time.
    over_two_second_confirmed: int = 0
    #: How many samples' `wait` alone - the confirmation time with no recovery in it -
    #: exceeded the gate. Zero by construction when the deadline is the gate, because a
    #: sample that did not confirm inside the deadline is censored AT it.
    wait_over_two_second: int = 0

    # -- leak, with the retention named ------------------------------------------------
    retained_transitions_start: int = 0
    retained_transitions_end: int = 0
    retained_queues_start: int = 0
    retained_queues_end: int = 0
    leak_bytes_after_release: int = 0
    rss_peak: int = 0
    leak_bytes_per_transition: float = 0.0
    #: **Live Python object count before and after the retained records were emptied.**
    #: RSS cannot show that the records were released - Python's allocator keeps freed
    #: arenas - so this is the hard number behind "the retention was released", and it
    #: is what makes the RSS remainder an explanation rather than a measurement.
    objects_before_release: int = 0
    objects_after_release: int = 0

    # -- placement evidence -------------------------------------------------------------
    placements_confirmed: int = 0
    placement_pids_matched: int = 0
    discarded_mismatched: int = 0

    # -- what was fed ------------------------------------------------------------------
    frames_submitted: int = 0
    frames_refused: int = 0
    drops: int = 0
    results_delivered: int = 0
    error_results: int = 0
    per_stream: tuple[dict[str, Any], ...] = ()
    frame_age_p50_ms: float | None = None
    frame_age_p95_ms: float | None = None

    #: Every sample, not just the summary. A distribution that cannot be re-read is a
    #: verdict wearing a distribution's clothes.
    samples: tuple[TransitionSample, ...] = ()
    #: The individual offending deliveries, not just their count. A count with no examples
    #: cannot be triaged, and these are the evidence for the headline number.
    stale_overlay_detail: tuple[dict[str, Any], ...] = ()
    mislabelled_detail: tuple[dict[str, Any], ...] = ()
    #: Whatever the supervisor had on record when the run ended, in full. A
    #: stress that swallows its own supervisor's complaints reports a clean run
    #: that the supervisor did not agree with.
    errors_at_end: tuple[str, ...] = ()
    pattern: str = "table"
    faulted_stream: str | None = None
    fault_recovery: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Refuse a number that would make the report a lie.

        Every field here is one the whole exercise turns on, and each has exactly one
        honest range. A record that accepts a negative leak, a `p95` below its `p50`, a
        count over the gate larger than the number of transitions, or a `bounded_by`
        spelling the supervisor does not emit, is a record nobody has to check - and
        `tests/test_route_stress.py` pins each of these refusals.
        """
        for name in ("transitions", "over_two_second", "discarded_results",
                     "stale_overlays", "mislabelled_devices", "crashes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int; got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0; got {value}")
        for name in ("rss_start", "rss_end", "leak_bytes", "leak_bytes_after_release",
                     "rss_peak"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int; got {value!r}")
        if self.transitions <= 0:
            raise ValueError(f"transitions must be >= 1; got {self.transitions}. A run "
                             f"that measured nothing has no percentile to report.")
        if self.over_two_second > self.transitions:
            raise ValueError(f"over_two_second ({self.over_two_second}) cannot exceed "
                             f"the number of transitions measured "
                             f"({self.transitions})")
        if self.leak_bytes != self.rss_end - self.rss_start:
            raise ValueError(f"leak_bytes ({self.leak_bytes}) must be rss_end - "
                             f"rss_start ({self.rss_end - self.rss_start}); it is a "
                             f"difference, not an independent claim")
        if self.rss_start <= 0:
            raise ValueError("rss_start must be a positive byte count")
        # `rss_peak` defaults to 0, which means "not measured" - the brief's eleven fields
        # do not include it, so a record built from those alone must still be valid.
        if self.rss_peak and self.rss_peak < self.rss_start:
            raise ValueError(f"rss_peak ({self.rss_peak}) is below rss_start "
                             f"({self.rss_start}); the peak cannot precede the start")
        for low, high, what in ((self.warm_elapsed_p50, self.warm_elapsed_p95, "p95"),
                                (self.setup_p50, self.setup_p95, "setup p95")):
            if high < low:
                raise ValueError(f"the {what} ({high}) is below its p50 ({low}); the "
                                 f"percentiles are out of order")
        if self.warm_elapsed_max < self.warm_elapsed_p95:
            raise ValueError(f"warm_elapsed_max ({self.warm_elapsed_max}) is below the "
                             f"p95 ({self.warm_elapsed_p95})")
        if self.warm_elapsed_p95 > GATE_SECONDS < self.over_two_second == 0:
            raise ValueError("a p95 over the gate with nothing counted over it")
        if self.placement_pids_matched > self.placements_confirmed:
            raise ValueError("more placements matched a live pid than were confirmed")

    @property
    def leak_after_retention(self) -> int:
        """`leak_bytes` minus what emptying the retained records gave back."""
        return self.leak_bytes - self.leak_bytes_after_release

    def to_json(self) -> dict[str, Any]:
        payload = {k: v for k, v in asdict(self).items() if k != "samples"}
        payload["samples"] = [sample.to_json() for sample in self.samples]
        payload["leak_after_retention"] = self.leak_after_retention
        return payload


class _Feeder:
    """A 15 fps submit loop for every stream - the thing a route needs to confirm.

    Owns the frame ids, which is what makes a stale overlay measurable: it records the
    highest id it submitted per stream, and the result side compares against that.
    """

    def __init__(self, supervisor: Supervisor, fps: float = 15.0) -> None:
        self._supervisor = supervisor
        self._interval = 1.0 / float(fps)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_id = 0
        self._lock = threading.Lock()
        self._high: dict[str, int] = {sid: -1 for sid in supervisor.stream_ids}
        self.submitted = 0
        #: `submit` returned `False`. **This is mostly not a refusal** - a full queue
        #: returns `False` too, and the supervisor counts that as a `drop`, not a
        #: refusal. So this number is the offer rate that did not land, and the
        #: authoritative split is `Supervisor.drops` and `StreamStats.frames_refused`.
        self.rejected = 0
        self.crashes: list[str] = []

    def mark(self) -> dict[str, int]:
        """The high-water mark per stream, for the next stale-overlay comparison."""
        with self._lock:
            return dict(self._high)

    def _frame(self, stream_id: str) -> FramePacket:
        with self._lock:
            frame_id = self._next_id
            self._next_id += 1
        height, width = FEED_SIZE
        # **Monotonic-clock seconds, not a frame counter.** `FramePacket` documents
        # `captured_at`/`decoded_at` as monotonic-clock seconds taken at ingress and after
        # decode, and `age_at(now)` subtracts `captured_at` from the supervisor's clock
        # (`perf_counter`). Feeding `frame_id / 30.0` produced a frame age of 789552831 ms,
        # which is a plausible-looking number and completely meaningless.
        now = time.perf_counter()
        return FramePacket(
            session_id="stress", scenario_id="retail_sanity", stream_id=stream_id,
            frame_id=frame_id, source_pts=now, source_loop_id=0,
            captured_at=now, decoded_at=now,
            width=width, height=height,
            image=np.zeros((height, width, 3), np.uint8))

    def _run(self) -> None:
        try:
            next_at = time.perf_counter()
            streams = self._supervisor.stream_ids
            while not self._stop.is_set():
                for stream in streams:
                    if self._stop.is_set():
                        return
                    frame = self._frame(stream)
                    if self._supervisor.submit(frame):
                        with self._lock:
                            self._high[stream] = max(self._high[stream],
                                                     frame.frame_id)
                        self.submitted += 1
                    else:
                        self.rejected += 1
                next_at += self._interval
                delay = next_at - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    # A slow machine falls behind in the *offer rate*, which the queue then
                    # counts as drops, rather than sprinting to catch up.
                    next_at = time.perf_counter()
        except Exception as exc:  # noqa: BLE001 - a feeder death is a result, not a crash
            self.crashes.append(f"{type(exc).__name__}: {exc}")

    def start(self) -> "_Feeder":
        self._thread = threading.Thread(target=self._run, name="stress-feeder",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)


class _Observer:
    """Counts, from the live delivery path, what the UI would have painted.

    Connected to `result_ready` with **`Qt.ConnectionType.DirectConnection`**, and that is
    deliberate. `app/main.py` uses a plain `connect`, so Qt queues the hop to the UI thread -
    which is right for painting, but it means a result emitted *before* a toggle is
    *delivered* after it. A stale-overlay check that compared delivery time against the
    pre-toggle mark therefore counted 104 stale overlays in a four-transition run where
    every one of them was a legitimate result delivered late by the event queue. The
    question being asked is "did the supervisor **emit** a superseded result", so the
    observation has to happen at the emission; the queued hop afterwards is Qt's business
    and cannot turn an unemitted result into a painted one.

    Every check is made against what the supervisor published **at emission time**, never
    against a value captured earlier.
    """

    def __init__(self, supervisor: Supervisor) -> None:
        self._supervisor = supervisor
        self._lock = threading.Lock()
        self._marks: dict[str, int] = {sid: -1 for sid in supervisor.stream_ids}
        #: Each stream's generation before the toggle in flight, so "this result is stale"
        #: can be asked properly - see `_inspect`.
        self._marks_generation: dict[str, int] = {}
        self.stale_overlays: list[dict[str, Any]] = []
        self.mislabelled: list[dict[str, Any]] = []
        self.delivered = 0
        self.error_results = 0
        self.confirmations: list[dict[str, Any]] = []
        #: `(stream, generation) -> the pid that was live when that generation's
        #: placement was first proved by a delivery`. Keyed on the generation so it
        #: survives the next toggle and cannot be confused with the previous worker's.
        self._first_confirm: dict[tuple[str, int], int | None] = {}
        self.crashes: list[str] = []

    def set_marks(self, marks: dict[str, int], generations: dict[str, int]) -> None:
        with self._lock:
            self._marks = dict(marks)
            self._marks_generation = dict(generations)

    def clear_marks(self) -> None:
        with self._lock:
            self._marks = {sid: -1 for sid in self._supervisor.stream_ids}
            self._marks_generation = {}

    def reset_confirmations(self) -> None:
        """Forget the confirming deliveries seen so far, so the next transition's are
        separable. With a queued connection these accumulate across transitions and a
        later transition is credited with an earlier one's proof."""
        with self._lock:
            self.confirmations = []

    def first_confirm_pid(self, stream_id: str, generation: int) -> int | None:
        """The pid that first proved `stream_id`'s placement on `generation`, or `None`."""
        with self._lock:
            return self._first_confirm.get((str(stream_id), int(generation)))

    def _on_result(self, result: Any) -> None:
        try:
            self._inspect(result)
        except Exception as exc:  # noqa: BLE001 - an observation failure is a result
            self.crashes.append(f"observer: {type(exc).__name__}: {exc}")

    def _inspect(self, result: Any) -> None:
        frame = result.frame
        stream_id = frame.stream_id
        requested = self._supervisor.stream_stats(stream_id).requested_device
        with self._lock:
            self.delivered += 1
            mark = self._marks.get(stream_id, -1)
            mark_generation = self._marks_generation.get(stream_id)
            stats = self._supervisor.stream_stats(stream_id)
            confirmed = stats.confirmed_devices
            pid = self._supervisor.pid_for(stream_id)
        if result.error is not None:
            self.error_results += 1
            return
        # **A picture from before the toggle, from a worker that has been superseded.**
        #
        # Both halves are needed, and the first version of this check had only the first:
        # Camera A never moves (it is on the CPU in all four rows), so every result for A
        # carries a pre-toggle frame id - and reporting those as stale overlays produced
        # three of them in a four-transition run, all of them frame_id **exactly equal** to
        # the mark, all of them perfectly current. The mark says *when the frame was
        # submitted*; only the generation says whether the worker that answered it is still
        # the one in charge. A pre-toggle frame answered by an unmoved stream is not a
        # stale overlay, it is a frame that was in flight.
        if (frame.frame_id <= mark and mark_generation is not None
                and stats.generation != mark_generation):
            self.stale_overlays.append(
                {"stream": stream_id, "frame_id": int(frame.frame_id),
                 "mark": int(mark),
                 "generation_at_mark": int(mark_generation),
                 "generation_now": int(stats.generation),
                 "execution_devices": list(result.execution_devices)})
        # **A placement that disagrees with the route.** `is_device` rather than a
        # substring test: `'GPU' in 'GPU.0'` is True and would let a mislabel through.
        if not is_device(result.execution_devices, requested):
            self.mislabelled.append(
                {"stream": stream_id, "requested": requested,
                 "execution_devices": list(result.execution_devices),
                 "frame_id": int(frame.frame_id)})
        elif confirmed and is_device(confirmed, requested):
            self.confirmations.append({"stream": stream_id, "pid": pid,
                                       "generation": int(stats.generation),
                                       "frame_id": int(frame.frame_id)})
        # **The pid that actually proved this generation's placement.** Recorded on the
        # *first* delivery that reads back the requested device under this generation, so
        # the pid is the one that carried `EXECUTION_DEVICES` for it. The earlier version
        # of this took the most recent confirming delivery *after* `await_transition`
        # returned, and that measured the wrong thing: `await_transition` returns the
        # instant the first confirming result is accepted, so no later result had arrived
        # yet and the "most recent" was still the **old** worker's. Mutant S4 read
        # `pids_after` before the wait instead of after, and survived - which is exactly
        # how the weakness was found.
        key = (stream_id, int(stats.generation))
        if (is_device(result.execution_devices, requested)
                and key not in self._first_confirm):
            self._first_confirm[key] = pid


def _percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile. Sorted copy; `values` is not mutated."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _rss() -> int:
    """The **host** process resident set, in bytes.

    Not per-stream and not per-device, and it carries the OpenVINO runtime and the Python
    allocator's high-water mark. The reason it is reported at all is that a *trend* across
    50 transitions is worth looking at even when the absolute number is meaningless.
    """
    return int(psutil.Process().memory_info().rss)


def _pump(application: Any, seconds: float = 0.0) -> None:
    """Let Qt deliver the queued `result_ready` emissions, and `yield` to them.

    Without this the stress would measure a supervisor that emits into a void: the
    queued signals would sit in the event queue and every stale-overlay count would be 0
    for the wrong reason.
    """
    deadline = time.perf_counter() + max(0.0, seconds)
    while True:
        application.processEvents()
        if time.perf_counter() >= deadline:
            return
        time.sleep(0.005)


def run_stress(*, transitions: int = 50, timeout_s: float = GATE_SECONDS,
               fps: float = 15.0, fault_stream: str | None = None,
               warmup_s: float = 30.0, application: Any = None,
               pattern: str = "table", progress: bool = True) -> StressResult:
    """Run the stress and return the record. Never raises for a failed run.

    `pattern` is `"table"` (spec section 5's four combinations, one camera per toggle) or
    `"both"` (row 1 against row 4, two cameras per toggle). Both start off the warm-up's
    row, so the first toggle always moves something.
    """
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {sorted(PATTERNS)}; got {pattern!r}")
    walk = PATTERNS[pattern]
    own_application = application is None
    if own_application:
        application = QCoreApplication.instance() or QCoreApplication([])

    faults: list[str] = [fault_stream] if fault_stream else []
    supervisor = Supervisor(device_ids=tuple(ROW_HW[s] for s in sorted(ROW_HW)),
                            compile_failure=faults)
    observer = _Observer(supervisor)
    supervisor.result_ready.connect(observer._on_result,
                                    Qt.ConnectionType.DirectConnection)
    feeder = _Feeder(supervisor, fps=fps).start()

    # -- warm up: the initial placement, which is a cold compile and NOT a transition ---
    samples: list[TransitionSample] = []
    faults_record: dict[str, Any] | None = None
    try:
        supervisor.start()
        first = supervisor.begin_transition(dict(ROW_HW))
        cold = supervisor.await_transition(first, timeout_s=warmup_s)
        if progress:
            print(f"[warmup] cold placement {cold.elapsed_s:.3f} s "
                  f"bounded_by={cold.bounded_by!r} "
                  f"(three cold compiles; not a transition)", flush=True)
        # **RSS is read after the warm-up, not before it.** The first placement compiles
        # three models and loads the OpenVINO runtime into the process; attributing that
        # to the transitions would put tens of MiB of runtime into the leak figure.
        rss_start = _rss()
        retained_transitions_start = len(supervisor._transitions)  # noqa: SLF001
        retained_queues_start = len(supervisor._retired_queues)  # noqa: SLF001
        if fault_stream is not None:
            faults_record = _observe_fault(supervisor, fault_stream, application)
            # The faulted stream is failed now; put it back so the churn under test is
            # the same churn as an unfaulted run.
            supervisor.select_devices({fault_stream: ROW_HW[fault_stream]})

        observer.clear_marks()
        rss_peak = rss_start
        for index in range(transitions):
            # **The first toggle is always a real move.** The warm-up is already on row 1,
            # so a walk that started on row 1 would make transition 1 a restatement of the
            # routing already in force - acknowledged instantly, 0.000 s, and a sample that
            # proves nothing.
            target = walk[index % len(walk)]
            marks = feeder.mark()
            generations = {sid: supervisor.stream_stats(sid).generation
                           for sid in sorted(target)}
            observer.set_marks(marks, generations)
            rss_before = _rss()
            started = time.perf_counter()
            transition_id = supervisor.begin_transition(dict(target))
            setup = time.perf_counter() - started
            outcome = supervisor.await_transition(transition_id, timeout_s=timeout_s)
            elapsed = time.perf_counter() - started
            pids_after = {sid: supervisor.pid_for(sid) for sid in sorted(target)}
            # **The pid that proved each stream's placement on the generation it is on
            # now**, from the observer's first-confirmation record - not from a delivery
            # looked up after the wait, which is the bug mutant S4 found. It must equal
            # the pid that is live after the transition: a confirmation carried by the
            # worker the toggle replaced is the defect this is looking for, and this is
            # the check that would catch it.
            pids_at_confirm = {
                sid: observer.first_confirm_pid(
                    sid, supervisor.stream_stats(sid).generation)
                for sid in sorted(target)}
            samples.append(TransitionSample(
                index=index, target=dict(target),
                acknowledged=outcome.acknowledged, bounded_by=outcome.bounded_by,
                generation=outcome.generation, elapsed_s=elapsed,
                transition_elapsed_s=outcome.elapsed_s, setup_s=setup,
                unconfirmed=outcome.unconfirmed_streams(),
                confirmed={k: tuple(v) for k, v in outcome.confirmed.items()},
                discarded_results=outcome.discarded_results,
                pids_at_confirm=pids_at_confirm, pids_after=pids_after,
                rss_before=rss_before))
            rss_peak = max(rss_peak, _rss())
            if progress:
                print(f"[{index + 1:>3}/{transitions}] {elapsed:6.3f} s "
                      f"{outcome.bounded_by:<9} "
                      f"unconfirmed={list(outcome.unconfirmed_streams())} "
                      f"discarded={supervisor.discarded_results} "
                      f"stale={len(observer.stale_overlays)} "
                      f"mislabelled={len(observer.mislabelled)}", flush=True)
        rss_end = _rss()
        retained_transitions_end = len(supervisor._transitions)  # noqa: SLF001
        retained_queues_end = len(supervisor._retired_queues)  # noqa: SLF001
        per_stream = [stats.to_json() for stats in supervisor.per_stream()]
        errors_at_end = supervisor.errors()
        ages = supervisor.frame_ages_ms()
        p50 = _percentile(ages, 50.0) if ages else None
        p95 = _percentile(ages, 95.0) if ages else None
        discarded = supervisor.discarded_results
        mismatched = supervisor.discarded_mismatched
        drops = supervisor.drops
        submitted = feeder.submitted
        rejected = feeder.rejected
    finally:
        feeder.stop()
        # **Let the workers drain before stopping the supervisor, and this is not
        # cosmetic - it is 15 s.** Measured 2026-10-03 with and without:
        # `supervisor.stop()` took **15.18 s** when called straight after the feeder
        # stopped and **0.27 s** after this settle, with everything else identical.
        #
        # The reason is in `stop()` itself: it detaches the streams, calls
        # `_halt_threads` - which joins each of the three threads per stream with a
        # `STOP_GRACE_SECONDS` (5.0 s) grace - and only *then* puts the `None` sentinel
        # into the input queues. A feeder blocked in `in_queue.put()` on a full pipe is
        # unblocked by nothing the join does, so it waits out the grace; three streams is
        # 15 s. **Not fixed here**: it is `stop()`'s ordering, it is M1/M2 behaviour, and
        # the instruction for this task is to report what the stress finds rather than to
        # fix it and re-run until green. It is a booth shutdown cost, not just a test cost.
        time.sleep(SETTLE_SECONDS)
        supervisor.stop()

    # -- the retention, released, and RSS read again ------------------------------------
    # Measurement only, and after everything else has been recorded: this is the point of
    # the whole exercise, so nothing that follows may depend on it.
    objects_before_release = len(gc.get_objects())
    with supervisor._lock:  # noqa: SLF001
        supervisor._transitions.clear()  # noqa: SLF001
        supervisor._retired_queues.clear()  # noqa: SLF001
    gc.collect()
    # **RSS is read here, before any further Qt pumping or any further census.** Pumping
    # the event queue allocates - every queued `state_changed` carries a whole `RoutePlan` -
    # and `gc.get_objects()` allocates a list of every tracked object, which on a suite
    # process is tens of thousands of pointers. Either allocation lands inside the very
    # figure used to say whether the retained records explain the growth. An earlier version
    # pumped first and produced a `leak_bytes_after_release` above `leak_bytes` roughly one
    # run in five; the census was still doing it afterwards. Measured on this machine: one
    # `len(gc.get_objects())` over 9 852 tracked objects raised RSS by **176 KiB and never
    # gave it back** (five consecutive calls, plateau at 176 KiB).
    leak_after_release = _rss() - rss_start
    objects_after_release = len(gc.get_objects())

    elapsed_samples = tuple(sample.elapsed_s for sample in samples)
    setup_samples = tuple(sample.setup_s for sample in samples)
    wait_samples = tuple(sample.transition_elapsed_s for sample in samples)
    over = sum(1 for value in elapsed_samples if value > GATE_SECONDS)
    over_confirmed = sum(1 for sample in samples
                         if sample.acknowledged and sample.elapsed_s > GATE_SECONDS)
    wait_over = sum(1 for value in wait_samples if value > GATE_SECONDS)
    unconfirmed = tuple(sample.index for sample in samples
                        if not sample.acknowledged)
    censored = bool(unconfirmed)
    matched = sum(
        1 for sample in samples
        if sample.acknowledged
        and all(sample.pids_at_confirm.get(sid) is not None
                and sample.pids_at_confirm.get(sid) == sample.pids_after.get(sid)
                for sid in sample.target))
    crashes = list(feeder.crashes) + list(observer.crashes)
    leak_bytes = rss_end - rss_start
    return StressResult(
        transitions=len(samples),
        warm_elapsed_p50=_percentile(elapsed_samples, 50.0),
        warm_elapsed_p95=_percentile(elapsed_samples, 95.0),
        warm_elapsed_max=max(elapsed_samples) if elapsed_samples else 0.0,
        over_two_second=over,
        discarded_results=discarded,
        rss_start=rss_start,
        rss_end=rss_end,
        leak_bytes=leak_bytes,
        stale_overlays=len(observer.stale_overlays),
        mislabelled_devices=len(observer.mislabelled),
        crashes=len(crashes),
        stale_overlay_detail=tuple(observer.stale_overlays),
        mislabelled_detail=tuple(observer.mislabelled),
        warm_elapsed_min=min(elapsed_samples) if elapsed_samples else 0.0,
        warm_elapsed_samples=elapsed_samples,
        setup_p50=_percentile(setup_samples, 50.0),
        setup_p95=_percentile(setup_samples, 95.0),
        setup_max=max(setup_samples) if setup_samples else 0.0,
        setup_samples=setup_samples,
        wait_p50=_percentile(wait_samples, 50.0),
        wait_p95=_percentile(wait_samples, 95.0),
        wait_max=max(wait_samples) if wait_samples else 0.0,
        acknowledged=sum(1 for s in samples if s.acknowledged),
        unconfirmed_samples=unconfirmed,
        timeout_s=timeout_s,
        censored_at_gate=censored,
        over_two_second_confirmed=over_confirmed,
        wait_over_two_second=wait_over,
        retained_transitions_start=retained_transitions_start,
        retained_transitions_end=retained_transitions_end,
        retained_queues_start=retained_queues_start,
        retained_queues_end=retained_queues_end,
        leak_bytes_after_release=leak_after_release,
        rss_peak=rss_peak,
        leak_bytes_per_transition=(leak_bytes / len(samples)) if samples else 0.0,
        objects_before_release=objects_before_release,
        objects_after_release=objects_after_release,
        placements_confirmed=sum(1 for s in samples if s.acknowledged),
        placement_pids_matched=matched,
        discarded_mismatched=mismatched,
        frames_submitted=submitted,
        frames_refused=rejected,
        drops=drops,
        results_delivered=observer.delivered,
        error_results=observer.error_results,
        per_stream=tuple(per_stream),
        frame_age_p50_ms=p50,
        frame_age_p95_ms=p95,
        pattern=pattern,
        faulted_stream=fault_stream,
        fault_recovery=faults_record,
        errors_at_end=tuple(errors_at_end),
        samples=tuple(samples))


def measure_steady_state(*, seconds: float = 120.0, fps: float = 15.0,
                         application: Any = None) -> dict[str, Any]:
    """Hold spec section 5 row 4 - three streams on the CPU - and measure the latencies.

    **This is a different claim from anything the transition stress reports**, and the two
    must never be quoted as one:

    - here nothing is torn down or rebuilt, and the routing is the worst case on this
      machine: three workers on one CPU plugin, each offered 15 fps. It is the app's
      steady state under spec section 5 row 4, and it is the number M2's headline latency
      figure should be.
    - the stress's `frame_age_p50_ms`/`p95_ms` are measured **under churn**, with workers
      being killed and respawned around them, and they fill a different window.

    `backlog` is `frames_accepted - frames_analysed - frames_invalidated` per stream: the
    frames submitted, taken off the queue, and thrown away by a route change, none of
    which are the queue's own `queue_size`. It is the honest measure of how far behind a
    stream is, and the frame age is a consequence of it rather than a cause.

    The percentile window is the supervisor's bounded `deque(maxlen=256)`, so it is the
    most recent 256 analysed frames' ages, not a lifetime figure - see
    `app/main.py`'s `_publish_derived_metrics` docstring for why that is 256 and what it
    spans at these rates.
    """
    if application is None:
        application = QCoreApplication.instance() or QCoreApplication([])
    supervisor = Supervisor(device_ids=("CPU", "CPU", "CPU"))
    feeder = _Feeder(supervisor, fps=fps).start()
    started = time.perf_counter()
    elapsed = 0.0
    try:
        supervisor.start()
        deadline = started + seconds
        while time.perf_counter() < deadline:
            _pump(application, 0.2)
        elapsed = time.perf_counter() - started
        per_stream = []
        for stats in supervisor.per_stream():
            # **Per-stream frame ages are read off the internal `_Stream`, because
            # `StreamStats` does not carry them** - `Supervisor.frame_ages_ms()`
            # concatenates every stream's and so cannot say which one lost its
            # history, which is the whole point of reporting this per stream. Read
            # under the supervisor's own lock, as the public accessor reads it.
            with supervisor._lock:  # noqa: SLF001
                ages = tuple(supervisor._streams[stats.stream_id]  # noqa: SLF001
                             .frame_ages_ms)
            per_stream.append({
                "stream_id": stats.stream_id,
                "requested_device": stats.requested_device,
                "state": stats.state,
                "confirmed_devices": list(stats.confirmed_devices),
                "frames_accepted": stats.frames_accepted,
                "frames_refused": stats.frames_refused,
                "drops": stats.drops,
                "frames_analysed": stats.frames_analysed,
                "frames_invalidated": stats.frames_invalidated,
                "backlog": (stats.frames_accepted - stats.frames_analysed
                            - stats.frames_invalidated),
                "frame_age_p50_ms": _percentile(ages, 50.0),
                "frame_age_p95_ms": _percentile(ages, 95.0),
                "window_samples": len(ages),
            })
    finally:
        feeder.stop()
        time.sleep(SETTLE_SECONDS)
        supervisor.stop()
    return {
        "claim": ("spec section 5 row 4, three streams on the CPU, no route change and no "
                  "churn - the app's steady state. NOT comparable with the stress's frame "
                  "ages, which are measured under churn."),
        "seconds": elapsed,
        "fps_offered_per_stream": fps,
        "fps_offered_total": fps * 3,
        "frame_age_p50_ms": _percentile(supervisor.frame_ages_ms(), 50.0),
        "frame_age_p95_ms": _percentile(supervisor.frame_ages_ms(), 95.0),
        "per_stream": per_stream,
        "errors": list(supervisor.errors()),
    }


def _observe_fault(supervisor: Supervisor, stream_id: str, application: Any) -> dict[str, Any]:
    """Wait for one camera's real worker death and record what the supervisor said.

    M0's unmet gate 9 is a **silent** fallback, and on this machine every device executes
    where it is asked, so one does not occur. This records the visible half - a worker that
    really dies, and the stream recovering to a stated device with a stated reason - which
    is the evidence Task 4's screen work will assert on.
    """
    deadline = time.perf_counter() + 120.0
    while time.perf_counter() < deadline:
        _pump(application, 0.1)
        if supervisor.stream_state(stream_id) == "failed":
            break
    record: dict[str, Any] = {
        "stream": stream_id,
        "state": supervisor.stream_state(stream_id),
        "requested_device": supervisor.stream_stats(stream_id).requested_device,
        "confirmed_devices": list(
            supervisor.stream_stats(stream_id).confirmed_devices),
        "pid": supervisor.pid_for(stream_id),
        "restart_count": supervisor.restart_count_for(stream_id),
        "errors": list(supervisor.errors()),
    }
    record["states_a_failed_route_must_never_claim"] = (
        record["confirmed_devices"] == []
        and "failed" in (record["state"],))
    # The recovery the supervisor offers on its own, exercised here so Task 4 has a
    # measured outcome to assert on rather than a prediction.
    supervisor.select_devices({stream_id: "GPU" if record["requested_device"] == "CPU"
                               else "CPU"})
    deadline = time.perf_counter() + 60.0
    while time.perf_counter() < deadline:
        _pump(application, 0.1)
        if supervisor.stream_state(stream_id) == "active":
            break
    record["after_reselect_state"] = supervisor.stream_state(stream_id)
    record["after_reselect_device"] = supervisor.stream_stats(stream_id).requested_device
    record["after_reselect_confirmed"] = list(
        supervisor.stream_stats(stream_id).confirmed_devices)
    record["after_reselect_pid"] = supervisor.pid_for(stream_id)
    return record


def _fmt_bytes(value: int) -> str:
    sign = "-" if value < 0 else ""
    value = abs(int(value))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{sign}{value:.1f} {unit}"
        value /= 1024.0
    return f"{sign}{value:.1f} GiB"  # pragma: no cover - unreachable


def _report(result: StressResult) -> None:
    line = "-" * 78
    print(line)
    print("M2 Task 3 - route-change stress")
    print(line)
    gate = f"{GATE_SECONDS:.0f}"
    deadline_is_default = result.timeout_s == GATE_SECONDS
    print(f"pattern                     : {result.pattern}")
    print(f"transitions measured        : {result.transitions}")
    print(f"acknowledged (confirmed)    : {result.acknowledged}/{result.transitions}")
    print(f"await deadline used         : {result.timeout_s:.1f} s"
          + ("  (Supervisor.await_transition's own default, unchanged)"
             if deadline_is_default else
             "  **NOT the default - this run measures the tail on purpose; see the "
             "module docstring**"))
    print(f"over {gate} s{' ' * max(1, 10 - len(gate))}     : "
          f"{result.over_two_second}")
    print(f"  ... of which acknowledged: {result.over_two_second_confirmed}  "
          f"(the subset with no recovery in its total)")
    print(f"  wait alone over {gate} s   : "
          f"{result.wait_over_two_second}")
    print()
    print("warm transition distribution, seconds, every sample.")
    print("  'total' is the toggle: begin_transition() returning to await_transition()")
    print("  returning. 'setup' is the synchronous teardown and spawn inside")
    print("  begin_transition. 'wait' is TransitionOutcome.elapsed_s, which is stamped")
    print("  AFTER that work and therefore excludes it.")
    print(f"  {'#':>3}  {'setup':>7} {'wait':>7} {'total':>7}")
    for index, sample in enumerate(result.samples, start=1):
        flag = "  OVER GATE" if sample.elapsed_s > GATE_SECONDS else ""
        print(f"  {index:>3}  {sample.setup_s:7.3f} {sample.transition_elapsed_s:7.3f} "
              f"{sample.elapsed_s:7.3f}{flag}")
    print(f"       min  {result.warm_elapsed_min:.3f}   "
          f"p50 {result.warm_elapsed_p50:.3f}   "
          f"p95 {result.warm_elapsed_p95:.3f}   "
          f"max {result.warm_elapsed_max:.3f}   <- TOTAL, the toggle")
    print(f"   setup                            "
          f"p50 {result.setup_p50:.3f}   "
          f"p95 {result.setup_p95:.3f}   "
          f"max {result.setup_max:.3f}")
    print(f"   wait                             "
          f"p50 {result.wait_p50:.3f}   "
          f"p95 {result.wait_p95:.3f}   "
          f"max {result.wait_max:.3f}")
    if result.censored_at_gate:
        print(f"  CENSORED: samples {list(result.unconfirmed_samples)} did not confirm "
              f"within the {result.timeout_s:.1f} s deadline. Their 'wait' is AT the "
              f"deadline and their 'total' includes the recovery, so both overstate the "
              f"confirmation time - the true tail is longer than the 'wait' column "
              f"shows.")
    else:
        print("  every sample confirmed inside the deadline; max is a real maximum, "
              "not a censoring artefact.")
    print()
    print("stale painting / placement, counted from the live delivery path:")
    print(f"  results delivered         : {result.results_delivered} "
          f"({result.error_results} error results)")
    print(f"  stale overlays            : {result.stale_overlays}")
    for entry in result.stale_overlay_detail[:20]:
        print(f"      STALE {entry}")
    print(f"  mislabelled devices       : {result.mislabelled_devices}")
    print(f"  discarded results (total) : {result.discarded_results}   "
          f"(mismatched: {result.discarded_mismatched})")
    print(f"  placements confirmed      : {result.placements_confirmed}, of which the "
          f"confirming pid matched the live pid: {result.placement_pids_matched}")
    print(f"  crashes                   : {result.crashes}")
    print(f"  supervisor errors at end  : {len(result.errors_at_end)}")
    for text in result.errors_at_end[:6]:
        shown = text if len(text) < 100 else text[:97] + "..."
        print(f"      {shown}")
    print()
    print("leak, host process RSS, with the designed retention named:")
    print(f"  rss_start                 : {_fmt_bytes(result.rss_start)}")
    print(f"  rss_peak                  : {_fmt_bytes(result.rss_peak)}")
    print(f"  rss_end                   : {_fmt_bytes(result.rss_end)}")
    print(f"  leak_bytes (end - start)  : {_fmt_bytes(result.leak_bytes)}  "
          f"({result.leak_bytes_per_transition / 1024.0:+.1f} KiB/transition)")
    print(f"  retained transitions      : {result.retained_transitions_start} -> "
          f"{result.retained_transitions_end}")
    print(f"  retained retired queues   : {result.retained_queues_start} -> "
          f"{result.retained_queues_end}")
    print(f"  live objects              : {result.objects_before_release} -> "
          f"{result.objects_after_release} after emptying them and collecting")
    print(f"  after emptying them + gc  : {_fmt_bytes(result.leak_bytes_after_release)}")
    print(f"  so retention explains     : "
          f"{_fmt_bytes(result.leak_after_retention)} of the growth; the rest "
          f"({_fmt_bytes(result.leak_bytes_after_release)}) is not accounted for by it")
    print()
    print("what was fed:")
    print(f"  frames accepted {result.frames_submitted}, submit() returned False "
          f"{result.frames_refused} (mostly a full queue), supervisor drops "
          f"{result.drops}")
    for stats in result.per_stream:
        print(f"  {stats['stream_id']}: requested={stats['requested_device']:<4} "
              f"state={stats['state']:<8} confirmed={list(stats['confirmed_devices'])} "
              f"accepted={stats['frames_accepted']} refused={stats['frames_refused']} "
              f"drops={stats['drops']} analysed={stats['frames_analysed']} "
              f"invalidated={stats['frames_invalidated']}")
    if result.frame_age_p50_ms is not None:
        print()
        print("frame ages UNDER CHURN - a different claim from the app's steady state:")
        print(f"  p50 {result.frame_age_p50_ms:.0f} ms   "
              f"p95 {result.frame_age_p95_ms:.0f} ms")
    if result.fault_recovery:
        print()
        print("visible fallback (a worker that really died):")
        for key, value in result.fault_recovery.items():
            text = str(value)
            print(f"  {key:<44}: {text if len(text) < 90 else text[:87] + '...'}")


def _report_steady(args: argparse.Namespace) -> int:
    """Print and publish the steady-state row. Exits non-zero only on a real failure."""
    print("-" * 78)
    print("M2 Task 3 - spec section 5 row 4 steady state (a DIFFERENT claim)")
    print("-" * 78)
    result = measure_steady_state(seconds=args.steady_seconds, fps=args.fps)
    print(f"held for {result['seconds']:.1f} s at {result['fps_offered_total']:.0f} fps "
          f"offered in total ({result['fps_offered_per_stream']:.0f} per stream)")
    print(f"frame age p50 {result['frame_age_p50_ms']:.0f} ms   "
          f"p95 {result['frame_age_p95_ms']:.0f} ms")
    print()
    print(f"  {'stream':<7} {'dev':<5} {'state':<9} {'accepted':>9} {'refused':>8} "
          f"{'drops':>7} {'analysed':>9} {'invalid':>8} {'backlog':>8} "
          f"{'p50 ms':>8} {'p95 ms':>8}")
    for entry in result["per_stream"]:
        print(f"  {entry['stream_id']:<7} {entry['requested_device']:<5} "
              f"{entry['state']:<9} {entry['frames_accepted']:>9} "
              f"{entry['frames_refused']:>8} {entry['drops']:>7} "
              f"{entry['frames_analysed']:>9} {entry['frames_invalidated']:>8} "
              f"{entry['backlog']:>8} {entry['frame_age_p50_ms']:>8.0f} "
              f"{entry['frame_age_p95_ms']:>8.0f}")
    print()
    print(f"  errors: {len(result['errors'])}")
    for text in result["errors"][:4]:
        print(f"    {text if len(text) < 110 else text[:107] + '...'}")
    print()
    print(f"  {result['claim']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print()
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    confirmed = [e for e in result["per_stream"] if e["state"] != "active"]
    print(f"verdict: {'all three active' if not confirmed else 'NOT ACTIVE: ' + str(confirmed)}; "
          f"a stream that cannot keep up shows here as drops and backlog, not as an error, "
          f"and that is the finding rather than a failure")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transitions", type=int, default=50,
                        help="how many alternating warm toggles (default 50)")
    parser.add_argument("--timeout-s", type=float, default=GATE_SECONDS,
                        help="await_transition deadline; the spec gate is 2.0 s and "
                             "widening it is for measuring a tail, never for passing")
    parser.add_argument("--fps", type=float, default=15.0,
                        help="offered frames per second per stream (default 15)")
    parser.add_argument("--pattern", choices=sorted(PATTERNS), default="table",
                        help="table: spec section 5's four combinations, one camera "
                             "per toggle (the shape the 2 s gate is about). both: "
                             "row 1 against row 4, two cameras per toggle.")
    parser.add_argument("--fault-stream", choices=sorted(ROW_HW), default=None,
                        help="make this camera's worker really die, for the visible "
                             "fallback evidence")
    parser.add_argument("--out", default="logs/qualification/m2-route-stress.json",
                        help="where the JSON record is written")
    parser.add_argument("--steady-seconds", type=float, default=0.0,
                        help="measure spec section 5 row 4's steady state instead of "
                             "running transitions: three streams on the CPU, no churn. A "
                             "different claim from the stress, and the one M2's headline "
                             "latency figure belongs to.")
    args = parser.parse_args(argv)

    if args.steady_seconds > 0.0:
        return _report_steady(args)

    result = run_stress(transitions=args.transitions, timeout_s=args.timeout_s,
                        fps=args.fps, fault_stream=args.fault_stream,
                        pattern=args.pattern)
    _report(result)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_json()
    payload["host"] = {"pid": os.getpid(), "python": sys.version.split()[0]}
    payload["note"] = (
        "frame_age_p50_ms/p95_ms here are measured under churn and are NOT the app's "
        "steady-state latency; spec section 5 row 4 unsaturated is the M2 headline.")
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print()
    print(f"wrote {out} ({out.stat().st_size} bytes)")

    clean = (result.stale_overlays == 0 and result.mislabelled_devices == 0
             and result.crashes == 0 and result.discarded_mismatched == 0
             and result.placement_pids_matched == result.placements_confirmed)
    print(f"verdict: {'clean' if clean else 'DEFECTS FOUND - see the counts above'}; "
          f"over the {GATE_SECONDS:.0f} s gate: {result.over_two_second}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())