"""`tools/leak_probe.py` - the leak at the deadline a presenter actually runs at.

## What this measures, and why it is two runs and not one

M2 measured the same 50 warm route changes under two deadlines and got two different
answers:

- **30 s deadline: 15.5 MiB total, 5.76 MiB after the designed retention was emptied.**
  The tail was censored on purpose so the distribution was measurable.
- **2.0 s deadline - the product's own default: 182.4 MiB, 175.3 MiB of it unexplained.**
  The run a presenter experiences, and the run nobody had explained.

The difference is not noise and it is not the extra transitions. **A shorter deadline means
more transitions time out**, and a transition that times out kills a worker that is still
compiling - so nothing has drained the `multiprocessing.Queue` the worker was reading. M4
Task 3 found the leak there: a retired queue keeps every frame its feeder thread never sent,
and `Supervisor._retired_queues` used to hold every queue it had ever created, for the life
of the process. `cancel_join_thread()` prevented the **atexit hang** and released nothing.

So this tool runs the same 50 toggles at both deadlines and reports, for each:

| figure | what it is |
|---|---|
| `leak_bytes` | RSS end - RSS start, measured **after** a warm-up that pays for the three cold compiles |
| `designed_bytes` | the part explained by records the supervisor deliberately retains (`_transitions`, and any queue the ledger is still holding), measured by emptying them and re-reading RSS |
| `unexplained_bytes` | RSS after emptying the retention and collecting, minus RSS start. **The remainder** |

They are reported separately and never added into one "leak" number: the retention is by
design, and folding it in would make the figures look worse than the code.

## The rules this probe follows

- **The machine is asserted clean before anything is measured.** 66 orphaned workers once
  poisoned two milestones of timings. The census is `tools/input_ack_under_load.py`'s, not a
  second implementation, and a dirty machine refuses the run unless
  `ACKLD_ALLOW_DIRTY=1` is set - in which case the JSON says `overridden: true`.
- **The 2.0 s deadline is the product's default and is not widened here.** `--timeout-s`
  exists so the 30 s condition can be reproduced; nothing in this file changes what
  `await_transition` defaults to.
- **`time.perf_counter` for every interval.** `time.monotonic` on this machine returned a
  single distinct value in 20 000 consecutive calls.
- **The feed is 240x320 at 15 fps, exactly M2's `FEED_SIZE`**, so the figures are comparable
  with M2's. This matters for the size of the number: a 1920x1080 frame is 6.2 MB against
  0.23 MB here, so the same *count* of retained frames would be ~27x the bytes. That ratio is
  arithmetic, not a measurement, and no 1080p leak figure is claimed here.
- **The distribution is reported, not one number.** Per-transition elapsed and per-transition
  RSS deltas are kept in `samples`, and p50/p95/max are printed next to the total.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
#: No display is opened: this is a `QCoreApplication`, and the supervisor only needs an
#: event loop for its queued signals. Set before Qt is imported, as the rest of the repo does.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import psutil  # noqa: E402
from PySide6.QtCore import QCoreApplication, Qt  # noqa: E402

from app.contracts.frame import FramePacket  # noqa: E402
from app.supervisor.runner import Supervisor  # noqa: E402
from tools.input_ack_under_load import assert_machine_clean  # noqa: E402

#: The same three requested devices the booth runs, and the same ones M2's stress walked.
ROW_HW = {"A": "CPU", "B": "GPU", "C": "NPU"}

#: Spec section 5's four combinations, in the order M2's stress walked them.
ROW_GPU_OFF = {"A": "CPU", "B": "CPU", "C": "NPU"}
ROW_NPU_OFF = {"A": "CPU", "B": "GPU", "C": "CPU"}
ROW_CPU = {"A": "CPU", "B": "CPU", "C": "CPU"}
TABLE = (ROW_CPU, ROW_GPU_OFF, ROW_NPU_OFF, ROW_HW)

#: The product's own default deadline. **Not a parameter of the fixed condition**, only the
#: value a `--timeout-s` of 2.0 means.
GATE_SECONDS = 2.0

#: M2's feed. 240x320, not 1920x1080, so this run is comparable with M2's table.
FEED_SIZE = (240, 320)
FEED_FPS = 15.0

#: How long the warm-up may take: three cold compiles plus the first inference.
WARMUP_SECONDS = 60.0

#: Let the workers finish the frames already in flight before `stop()`, so the shutdown cost
#: is not part of the measurement. **Not a workaround for `stop()`** - `stop()` now signals
#: its sentinels before joining threads and costs a fraction of a second on its own.
SETTLE_SECONDS = 1.5


@dataclass(frozen=True)
class LeakSample:
    """One warm transition: what it cost, how it ended, and what RSS did around it."""

    index: int
    elapsed_s: float
    #: `"confirmed"` or `"timeout"` - the `TransitionOutcome.bounded_by`.
    bounded_by: str
    rss_before: int
    rss_after: int

    def __post_init__(self) -> None:
        if self.bounded_by not in ("confirmed", "timeout"):
            raise ValueError(f"bounded_by must be confirmed or timeout; got "
                             f"{self.bounded_by!r}")
        if self.rss_before < 0 or self.rss_after < 0:
            raise ValueError("RSS is a byte count and cannot be negative")
        if self.elapsed_s < 0.0:
            raise ValueError(f"elapsed_s must be >= 0; got {self.elapsed_s}")

    @property
    def rss_delta(self) -> int:
        return self.rss_after - self.rss_before

    def to_json(self) -> dict[str, Any]:
        return {"index": self.index, "elapsed_s": round(self.elapsed_s, 6),
                "bounded_by": self.bounded_by, "rss_before": self.rss_before,
                "rss_after": self.rss_after, "rss_delta": self.rss_delta}


@dataclass(frozen=True)
class LeakResult:
    """One deadline's run, with the designed retention named separately.

    Every field is a measurement or a condition. `unexplained_bytes` is **not** an estimate:
    it is RSS re-read after the retention was emptied and `gc.collect()` ran, minus the RSS
    taken after the warm-up. Python's allocator does not return every freed arena to the OS,
    so this is a **floor** on what is still held - which is the direction that matters for a
    leak.
    """

    #: The `await_transition` deadline this run used. 2.0 is the product's default.
    timeout_s: float
    transitions: int
    #: Host-process RSS after the warm-up and after the last transition.
    rss_start: int
    rss_end: int
    #: `rss_end - rss_start`.
    leak_bytes: int
    #: RSS after `_transitions` and `_retired_queues` were emptied and `gc.collect()` ran,
    #: minus `rss_start`. The remainder, and the figure M2 called "remainder".
    unexplained_bytes: int
    #: Live Python objects **after** that emptying and collection. A count that goes *down*
    #: across transitions is the evidence that nothing accumulates per transition.
    live_objects: int
    samples: tuple[LeakSample, ...]
    #: `leak_bytes - unexplained_bytes`: what emptying the deliberately-retained records
    #: gave back. **A residual, not an attribution.** When the total is large this is the
    #: retained records (M2 read 9.7 MiB for 50 transitions and 150 retired queues); when
    #: the total is inside allocator noise it can be negative, because emptying the records
    #: also lets `gc` hand arenas back to the OS. `report` says so beside the number.
    designed_bytes: int
    rss_after_release: int
    live_objects_start: int
    retained_transitions_start: int
    retained_transitions_end: int
    #: Queues in `Supervisor._retired_queues` at the start and the end of the walk. The
    #: start is the live pairs; the end should be the live pairs - anything else is a queue
    #: nothing is reading and nothing released.
    retained_queues_start: int
    retained_queues_end: int
    #: And after `stop()`. **This is the one that must be 0**: `stop()` sweeps the ledger
    #: after every stream has been claimed, so a queue still listed here is one whose feeder
    #: `multiprocessing`'s atexit finaliser would eventually have to join.
    retained_queues_after_stop: int
    #: How many transitions ended `timeout`. **The whole point of the shorter deadline.**
    timeouts: int
    elapsed_min_s: float
    elapsed_p50_s: float
    elapsed_p95_s: float
    elapsed_max_s: float
    #: p50/p95/max of the per-transition RSS delta, so a grower is visible in the
    #: distribution rather than only in the total.
    rss_delta_p50: int
    rss_delta_p95: int
    rss_delta_max: int
    stop_s: float
    wall_s: float
    machine: dict[str, Any]
    feeds_frames: int

    def __post_init__(self) -> None:
        if self.transitions != len(self.samples):
            raise ValueError(f"transitions ({self.transitions}) must equal the sample "
                             f"count ({len(self.samples)}); one of them is a claim about "
                             f"the other")
        if self.leak_bytes != self.rss_end - self.rss_start:
            raise ValueError(f"leak_bytes ({self.leak_bytes}) must be rss_end - rss_start "
                             f"({self.rss_end - self.rss_start})")
        if self.unexplained_bytes != self.rss_after_release - self.rss_start:
            raise ValueError(f"unexplained_bytes ({self.unexplained_bytes}) must be "
                             f"rss_after_release - rss_start")
        if self.designed_bytes != self.leak_bytes - self.unexplained_bytes:
            raise ValueError("designed_bytes must be leak_bytes - unexplained_bytes")
        if self.rss_start <= 0:
            raise ValueError("rss_start must be a positive byte count")
        if not 0 <= self.timeouts <= self.transitions:
            raise ValueError(f"timeouts ({self.timeouts}) must be within 0..transitions")

    def to_json(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record["samples"] = [sample.to_json() for sample in self.samples]
        return record


class _Feeder:
    """A 15 fps submit loop on every stream - what a route needs in order to confirm.

    `captured_at`/`decoded_at` are `perf_counter` seconds, not a frame counter: the
    supervisor computes frame age by subtracting them from its own clock, and a counter
    produced a frame age of 789552831 ms in an earlier version of the M2 stress.
    """

    def __init__(self, supervisor: Supervisor, *, fps: float = FEED_FPS) -> None:
        self._supervisor = supervisor
        self._interval = 1.0 / float(fps)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._next_id = 0
        self.submitted = 0

    def _frame(self, stream_id: str) -> FramePacket:
        with self._lock:
            frame_id = self._next_id
            self._next_id += 1
        height, width = FEED_SIZE
        now = time.perf_counter()
        return FramePacket(session_id="leak-probe", scenario_id="retail_sanity",
                           stream_id=stream_id, frame_id=frame_id, source_pts=now,
                           source_loop_id=0, captured_at=now, decoded_at=now,
                           width=width, height=height,
                           image=np.zeros((height, width, 3), np.uint8))

    def _run(self) -> None:
        next_at = time.perf_counter()
        while not self._stop.is_set():
            for stream_id in self._supervisor.stream_ids:
                if self._stop.is_set():
                    return
                if self._supervisor.submit(self._frame(stream_id)):
                    self.submitted += 1
            next_at += self._interval
            delay = next_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                # Fall behind in the *offer* rate rather than sprinting to catch up.
                next_at = time.perf_counter()

    def start(self) -> "_Feeder":
        self._thread = threading.Thread(target=self._run, name="leak-probe-feeder",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)


def _rss() -> int:
    """This process's resident set, in bytes.

    The host process only: the workers are separate processes, and the supervisor's own
    bookkeeping is what this probe is about.
    """
    return int(psutil.Process().memory_info().rss)


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _pump(application: Any, seconds: float = 0.0) -> None:
    """Let Qt deliver the queued `state_changed` emissions the supervisor makes.

    Without this the probe would measure a supervisor whose published plans pile up in the
    event queue - a retention of its own that is not the supervisor's.
    """
    deadline = time.perf_counter() + max(0.0, seconds)
    while True:
        application.processEvents()
        if time.perf_counter() >= deadline:
            return
        time.sleep(0.005)


def measure(*, timeout_s: float = GATE_SECONDS, transitions: int = 50,
            application: Any = None, progress: bool = True) -> LeakResult:
    """Run `transitions` warm route changes at `timeout_s` and return the record.

    Never raises for a failed run: a timeout is an outcome, not an error. A **dirty
    machine** does raise, because a leak figure measured beside a competing suite is not a
    measurement of this application.
    """
    if transitions < 1:
        raise ValueError(f"transitions must be >= 1; got {transitions}")
    if timeout_s <= 0.0:
        raise ValueError(f"timeout_s must be > 0; got {timeout_s}")
    machine = assert_machine_clean(settle_s=10.0)

    own_application = application is None
    if own_application:
        application = QCoreApplication.instance() or QCoreApplication([])

    supervisor = Supervisor(device_ids=tuple(ROW_HW[sid] for sid in sorted(ROW_HW)))
    feeder = _Feeder(supervisor).start()
    samples: list[LeakSample] = []
    stop_s = -1.0
    wall_start = time.perf_counter()
    try:
        supervisor.start()
        first = supervisor.begin_transition(dict(ROW_HW))
        cold = supervisor.await_transition(first, timeout_s=WARMUP_SECONDS)
        if progress:
            print(f"[warmup] cold placement {cold.elapsed_s:.3f} s "
                  f"bounded_by={cold.bounded_by!r} (three cold compiles; not a "
                  f"transition)", flush=True)
        # **RSS is read after the warm-up, not before it.** The first placement compiles
        # three models and loads the runtime into the process; attributing that to the
        # transitions would put tens of MiB of runtime into the leak figure.
        _pump(application)
        gc.collect()
        rss_start = _rss()
        live_objects_start = len(gc.get_objects())
        retained_transitions_start = len(supervisor._transitions)  # noqa: SLF001
        retained_queues_start = len(supervisor._retired_queues)  # noqa: SLF001

        for index in range(transitions):
            target = dict(TABLE[index % len(TABLE)])
            rss_before = _rss()
            started = time.perf_counter()
            transition_id = supervisor.begin_transition(target)
            outcome = supervisor.await_transition(transition_id, timeout_s=timeout_s)
            elapsed = time.perf_counter() - started
            _pump(application)
            rss_after = _rss()
            samples.append(LeakSample(index=index, elapsed_s=elapsed,
                                      bounded_by=outcome.bounded_by,
                                      rss_before=rss_before, rss_after=rss_after))
            if progress:
                print(f"[{index + 1:>3}/{transitions}] {elapsed:6.3f} s "
                      f"{outcome.bounded_by:<9} unconfirmed="
                      f"{list(outcome.unconfirmed_streams())} "
                      f"rss {rss_after / 2**20:7.1f} MiB "
                      f"(+{samples[-1].rss_delta / 2**20:+6.1f}) "
                      f"ledger {len(supervisor._retired_queues)}",  # noqa: SLF001
                      flush=True)

        rss_end = _rss()
        elapsed_samples = tuple(sample.elapsed_s for sample in samples)
        rss_deltas = tuple(sample.rss_delta for sample in samples)
        retained_transitions_end = len(supervisor._transitions)  # noqa: SLF001
        retained_queues_end = len(supervisor._retired_queues)  # noqa: SLF001
        timeouts = sum(1 for sample in samples if sample.bounded_by == "timeout")
        submitted = feeder.submitted
    finally:
        feeder.stop()
        time.sleep(SETTLE_SECONDS)
        stop_started = time.perf_counter()
        supervisor.stop()
        stop_s = time.perf_counter() - stop_started
        retained_queues_after_stop = len(supervisor._retired_queues)  # noqa: SLF001
    wall_s = time.perf_counter() - wall_start

    # -- the designed retention, emptied, and RSS read again -----------------------------
    # Measurement only, and after everything else has been recorded: nothing that follows
    # may change a figure above it.
    gc.collect()
    objects_before_release = len(gc.get_objects())
    with supervisor._lock:  # noqa: SLF001
        supervisor._transitions.clear()  # noqa: SLF001
        supervisor._retired_queues.clear()  # noqa: SLF001
    gc.collect()
    objects_after_release = len(gc.get_objects())
    # **Read before any further Qt pumping.** Pumping allocates a whole `RoutePlan` per
    # queued `state_changed`, and that allocation would land inside the figure used to say
    # whether the retention explains the growth.
    rss_after_release = _rss()

    leak_bytes = rss_end - rss_start
    unexplained_bytes = rss_after_release - rss_start
    return LeakResult(
        timeout_s=float(timeout_s),
        transitions=len(samples),
        rss_start=rss_start,
        rss_end=rss_end,
        leak_bytes=leak_bytes,
        unexplained_bytes=unexplained_bytes,
        live_objects=objects_after_release,
        samples=tuple(samples),
        designed_bytes=leak_bytes - unexplained_bytes,
        rss_after_release=rss_after_release,
        live_objects_start=live_objects_start,
        retained_transitions_start=retained_transitions_start,
        retained_transitions_end=retained_transitions_end,
        retained_queues_start=retained_queues_start,
        retained_queues_end=retained_queues_end,
        retained_queues_after_stop=retained_queues_after_stop,
        timeouts=timeouts,
        elapsed_min_s=min(elapsed_samples) if elapsed_samples else 0.0,
        elapsed_p50_s=_percentile(elapsed_samples, 50.0),
        elapsed_p95_s=_percentile(elapsed_samples, 95.0),
        elapsed_max_s=max(elapsed_samples) if elapsed_samples else 0.0,
        rss_delta_p50=int(_percentile(rss_deltas, 50.0)),
        rss_delta_p95=int(_percentile(rss_deltas, 95.0)),
        rss_delta_max=max(rss_deltas) if rss_deltas else 0,
        stop_s=stop_s,
        wall_s=wall_s,
        machine=machine,
        feeds_frames=submitted,
    )


def _fmt(value: int) -> str:
    sign = "-" if value < 0 else "+"
    return f"{sign}{abs(value) / 2**20:.1f} MiB"


def report(result: LeakResult) -> None:
    """Print one run. The three lines that matter are total / designed / remainder."""
    deadline = "the product's default" if result.timeout_s == GATE_SECONDS else (
        "a deliberately widened deadline; NOT the product's")
    print("", flush=True)
    print(f"deadline used              : {result.timeout_s:.1f} s ({deadline})", flush=True)
    print(f"machine clean              : {result.machine.get('clean')} "
          f"(competing pids {result.machine.get('competing_pids')})", flush=True)
    print(f"warm transitions           : {result.transitions}, of which "
          f"{result.timeouts} timed out", flush=True)
    print(f"elapsed                    : min {result.elapsed_min_s:.3f}  "
          f"p50 {result.elapsed_p50_s:.3f}  p95 {result.elapsed_p95_s:.3f}  "
          f"max {result.elapsed_max_s:.3f} s", flush=True)
    print("", flush=True)
    print("leak, host process RSS, with the designed retention named separately:", flush=True)
    print(f"  rss_start                : {result.rss_start / 2**20:.1f} MiB", flush=True)
    print(f"  rss_end                  : {result.rss_end / 2**20:.1f} MiB", flush=True)
    print(f"  total growth             : {_fmt(result.leak_bytes)} "
          f"({result.leak_bytes / max(1, result.transitions) / 1024:.1f} KiB/transition)",
          flush=True)
    print(f"  retained transitions     : {result.retained_transitions_start} -> "
          f"{result.retained_transitions_end}", flush=True)
    print(f"  retained queues (ledger) : {result.retained_queues_start} -> "
          f"{result.retained_queues_end} during the walk, "
          f"{result.retained_queues_after_stop} after stop()", flush=True)
    print(f"  live objects             : {result.live_objects_start} -> "
          f"{result.live_objects} (emptied, then gc)", flush=True)
    print(f"  designed retention       : {_fmt(result.designed_bytes)} - `_transitions` "
          f"plus any queue the ledger still holds; emptied to measure it", flush=True)
    print(f"  UNEXPLAINED remainder    : {_fmt(result.unexplained_bytes)} "
          f"(a floor: the allocator does not return every arena)", flush=True)
    if abs(result.leak_bytes) < 8 * 2**20:
        print("  note                     : the total is inside allocator noise, so the "
              "split above is a residual and neither half is attributable to it",
              flush=True)
    print(f"  per-transition RSS delta : p50 {_fmt(result.rss_delta_p50)}  "
          f"p95 {_fmt(result.rss_delta_p95)}  max {_fmt(result.rss_delta_max)}", flush=True)
    print(f"  stop()                   : {result.stop_s:.3f} s", flush=True)
    print(f"  wall                     : {result.wall_s:.1f} s "
          f"({result.feeds_frames} frames offered)", flush=True)
    print("", flush=True)


def _default_out(timeout_s: float) -> Path:
    return REPO_ROOT / "logs" / "qualification" / f"m4-leak-{timeout_s:g}s.json"


def _write(result: LeakResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    print(f"written                    : {path}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure the supervisor's leak at a route-transition deadline. The "
                    "2.0 s default is the product's; the 30 s condition exists to "
                    "reproduce M2's comparison.")
    parser.add_argument("--timeout-s", type=float, default=GATE_SECONDS,
                        help="await_transition deadline. 2.0 is the product's default and "
                             "the only one a booth runs.")
    parser.add_argument("--transitions", type=int, default=50,
                        help="warm route changes. 50 is M2's count and the one that "
                             "produced 182.4 MiB at the 2 s deadline.")
    parser.add_argument("--out", type=Path, default=None,
                        help="JSON path. Defaults to "
                             "logs/qualification/m4-leak-<deadline>s.json")
    parser.add_argument("--both", action="store_true",
                        help="Run the 2.0 s condition and then the 30 s one, each in its "
                             "own process, and write one JSON per condition. This is the "
                             "reproduction: the two deadlines are the finding.")
    args = parser.parse_args(argv)

    if args.both:
        for deadline in (GATE_SECONDS, 30.0):
            if args.out is None:
                out = _default_out(deadline)
            else:
                out = args.out.with_name(f"{args.out.stem}-{deadline:g}s"
                                         f"{args.out.suffix or '.json'}")
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--timeout-s", f"{deadline:g}",
                       "--transitions", str(args.transitions),
                       "--out", str(out)]
            print(f"=== running the {deadline:g} s condition in its own process ===",
                  flush=True)
            completed = subprocess_run(command)
            if completed != 0:
                print(f"the {deadline:g} s condition exited {completed}", file=sys.stderr)
                return completed
        return 0

    result = measure(timeout_s=args.timeout_s, transitions=args.transitions)
    report(result)
    _write(result, args.out or _default_out(args.timeout_s))
    return 0


def subprocess_run(command: Sequence[str]) -> int:
    """Run a child and stream its output through, keeping the exit code."""
    import subprocess

    return subprocess.call(list(command), cwd=str(REPO_ROOT),
                           env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})


if __name__ == "__main__":
    raise SystemExit(main())
