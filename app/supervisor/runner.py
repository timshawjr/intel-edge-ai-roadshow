"""`app.supervisor.runner` - the process owner, the routes, and the transition protocol.

M1 built one stream, one worker, one verdict. M2 owns **three** streams, three worker
processes, and a route transition that must never be acknowledged before it happened.
Five decisions here are the difference between a booth screen that is truthful and one
that is not, and each is enforced structurally rather than by convention.

## 1. `pending` until an inference has proved a placement

A fresh supervisor knows what it was *asked* for. It does not know where anything ran.
So `route_plan().state` is `"pending"` from construction, and only a result carrying
`EXECUTION_DEVICES` read **after** its own inference - which is the only proof of
placement there is (AGENTS.md rule 5) - moves it to `"active"`. Setting `active` in
`start()` would publish a placement nobody observed, which is review focus #5 with a
different mechanism: not "the card shows the requested device", but "the card shows a
device that has not run yet".

`is_device()` therefore reads the **confirmed** map and never the request. So
`Supervisor(device_id="NPU").is_device("NPU")` is `False` until an NPU inference comes
back saying so. `RoutePlan.requested_device(stream)` exists for the separate question
"what was asked", and a caller that wants that has to ask for it by name.

**With three streams that is per stream.** `plan.confirmed_devices("C")` is the only
thing that says where Camera C ran, and `plan.stream_state("C")` the only thing that says
whether it is healthy. `plan.state` is the aggregate (see `RoutePlan`'s docstring) and
answering "is anything wrong", never "what is Camera C doing".

## 2. One restart per stream, then a visible failure

AGENTS.md rule 12: a worker that fails is restarted **once** automatically, and a
second failure is a failure somebody has to see. Two kinds count as a failure here -
the worker process **exiting**, and a result carrying an `error` - because on this
machine the second is how a device that will not compile announces itself:
`run_worker` turns a refused compile into an `InferenceResult` with `error` set and
carries on, so the process stays alive while doing nothing useful. Watching only for
`exitcode` would leave a permanently broken route sitting in `pending` with no error on
screen.

After the second failure **that stream's** plan is `failed`, **that stream's**
`confirmed` is emptied, and the error text is published on `error_raised`. Emptying
`confirmed` is the part that matters: a route that has failed but still reports
`confirmed={"C": ("NPU",)}` renders as a working NPU, and that is precisely the claim the
review exists to forbid. The requested device is *kept*, because "NPU was asked for and
could not run" is what an operator needs to read. **The other two streams are not
touched** - a Camera C that will not compile is a Camera C problem, and taking the whole
panel down for it is the M2 version of the same failure.

`restart_count` is a property, not a log line, so a supervisor that restarted in a loop
would be visible in a test. It is the **total** over streams. A new generation resets a
stream's budget: a new generation is a new opportunity, and holding a spent budget
against the next device would mean the second device selected could not survive a single
hiccup.

**This is deliberately unforgiving.** A route whose frames error twice is `failed`, and
the operator re-selects. A single transient decode error costs one restart and the route
recovers; two do not. The alternative - counting consecutive failures, or resetting the
budget on the first success - would mean the budget is not a budget, and a device that
fails intermittently would never reach a visible failure at all.

## 3. Frames cross the spawn boundary through a bounded, counted queue

`submit` goes through `app.engine.queue.BoundedFrameQueue` before anything is handed to
the child, so the drop counter and the newest-frame refusal are the ones M1 wrote and
tested. One queue per stream, `maxsize=2` each, which is spec section 7's "at most two
waiting frames". A feeder thread drains that queue into the `multiprocessing` queue; the
worker consumes it and puts `InferenceResult`s on a second one.

**The transport is pickled numpy arrays, not shared memory, and that is a measured cost
not an oversight.** The design's Approach C names `multiprocessing.shared_memory` frame
rings; that is not built here. Measured on this machine, a 1920x1080x3 BGR frame
round-tripped through a spawn `multiprocessing.Queue` is **21.49 ms p50, 23.24 ms max**
over 30 frames (probe: put, child gets, child puts the same array back, parent gets).
Pickling alone is 3.1-3.6 ms and unpickling 1.7-1.9 ms for 6.2 MB. That is inside the
200 ms input-acknowledgement budget with room over, and a shared-memory ring is a
M2-scale change with its own lifetime and teardown hazards. It is recorded here as the
known cost of this transport rather than left for a reader to discover as latency.

**Three threads per stream - a feeder, a reader and a watchdog** - so nine for three
streams. One shared reader was considered and rejected: it would make one stream's
blocked `get()` another's latency, and "a stream whose worker fails does not take down
the other two" is easier to hold when a thread belongs to exactly one stream. The cost is
measured, not assumed - see the M2 report for input acknowledgement with three streams
live under 6 saturators.

**Every counter `submit` moves is moved inside one hold of `self._lock`.** It was not:
`attempts` was incremented under the lock, then the bounded-queue `put` and
`accepted += 1` ran outside it. A `per_stream()` snapshot landing in that window saw
`attempts = 1, accepted = 0` and the per-stream identity `accepted + refused + drops ==
attempts` was **transiently false under live traffic** - a checkable invariant that fails
intermittently is worse than one that is never checkable, because it fails on the booth
and not in the test. `BoundedFrameQueue.put` returns immediately when full and never
waits, so holding the lock across it costs nothing.

The watchdog polls `Process.is_alive()` because `multiprocessing` offers no exit callback
on Windows that survives a queue that nobody is reading; `EXIT_POLL_SECONDS` is 0.05 s
because the deadline that matters is a human watching a booth, not a benchmark.

## 4. A generation is a property of a *stream*, and a result carries its own

Review focus #1 for routing: *"a result from a superseded generation painting onto a newer
frame."* The reader thread for a stream is created **bound to that stream's generation at
the moment the worker was spawned**. Every result it takes is offered through one public
method, `accept_result(stream_id, generation, result)`, which discards and counts anything
whose generation is not the stream's current one. That is not a heuristic: a superseded
worker can only ever be read by the reader thread that was spawned with it, so the
generation cannot be lost, guessed, or read off a frame that does not carry one.

The check is **per stream**, not against `route_plan().generation`. `RoutePlan.generation`
is the *highest* generation across streams (spec section 7 asks for the age of the newest
routing decision), so filtering on it would throw away valid results from the two streams
whose route did not change. `StreamStats.generation` and `RoutePlan.stream_states` are
the per-stream answers.

**A superseded result is discarded, counted, and never delivered.** The counter is
`Supervisor.discarded_results` and the per-transition count is
`TransitionOutcome.discarded_results`. Discarding is not the same as coasting: a coasted
*track* is a tracker concept (the object left the picture and the box is held briefly),
and an obsolete-generation *result* is a routing concept. They are counted separately and
never summed.

## 5. A transition is acknowledged by a **confirmed placement**, never by a request

`begin_transition(mapping) -> int` is the toggle. It does exactly three things: bump the
generation, clear the proof for the affected streams only, and respawn those streams'
workers on the new devices.

**The old route stops accepting new submissions by ceasing to exist, not by refusing
them.** `select_devices` detaches the old queues, terminates the old worker and drains
the bounded queue before the new one is spawned, so there is no route left for a frame to
be refused by. A frame offered after `begin_transition` returns goes onto the **new**
worker's queue, which is exactly spec section 5's "new target begins submitting".

The alternative was tried first and is a deadlock: an earlier version refused mid-
transition submissions with a `"transitioning"` reason, and then nothing could be
submitted, so no inference could run on the new worker, so the new route could never
confirm, so every transition could only ever end in a timeout. Refusing frames does not
make a transition clean, and the feed is a 15 fps stream the presenter is not pausing.

`await_transition(transition_id, timeout_s=2.0) -> TransitionOutcome` then waits for the
**new** route to confirm. The deadline is measured from when the transition *began*, not
from when `await_transition` was called, because the 2 s spec gate is a time from toggle
to confirmed placement. It ends in one of two ways:

- `"confirmed"` - every affected stream proved a placement under that generation. This
  is the only path that is an acknowledgement, and it is reachable only through
  `EXECUTION_DEVICES` read after an inference.
- `"timeout"` - the deadline passed, **or** an affected stream failed visibly, which
  cannot become a confirmation. `elapsed_s` is reported either way, honestly, and
  `confirmed` names only the streams that did prove something, so `set(requested) -
  set(confirmed)` **is** the set of streams that did not confirm
  (`TransitionOutcome.unconfirmed_streams`).

On `"timeout"` the supervisor **recovers to the last healthy route**: every affected
stream that did not confirm is put back on the device it was running on before the
toggle, with a stated reason on `error_raised` and in `RoutePlan.errors`. The recovery
itself does not re-enter recovery, so a chain of failed toggles terminates rather than
recursing.

**A recovery that cannot move anything says so instead of claiming it did.** A stream
that failed *before* the toggle is already on the device the recovery wants to restore it
to, so there is no route change to make and no worker to bring up. Publishing
*"Recovering C - the last healthy route"* there would be a claim about an action that did
not happen, on the one stream an operator is watching; it was observed doing exactly
that. The published text in that case states that nothing moved, that the route is failed
and idle, and that **re-selecting the same device will not fix it** - the remedy is a
different device, or a direct `select_device`, because that is what spawns a worker.

### The two ids, and why they are not one

`begin_transition` returns a **monotonic transition id**, not the route generation. They
are equal when the call moved something and differ when it did not. Keying transitions by
the route generation let two restatements inside one generation collide, and the second
overwrote the first's `started`, so a late `await_transition` was handed a deadline
measured from the wrong moment. `TransitionOutcome.generation` still reports the route
generation, so the reporting surface is unchanged.

### Queued frames a route change throws away are counted

`select_devices` drains each affected stream's bounded queue, discarding up to
`queue_size` (2) frames that had been accepted and whose analysis is now moot. They are
neither drops (the queue was not full) nor refusals (the caller was told `True`), so they
are counted as their own population, `StreamStats.frames_invalidated` — a fourth kind
alongside `drops`, `frames_refused` and `frames_analysed`, and deliberately outside the
per-stream identity because the frame was already counted once, as accepted, at `submit`
time. Before this counter existed, `frames_accepted` permanently exceeded
`frames_analysed` with nothing on the record to explain the gap.

**Spec deviation, recorded deliberately:** spec section 5 says outstanding requests
"finish or time out". Here they are discarded at the moment the route changes, without
being waited for. The alternative was to let the old worker finish them, which would mean
keeping its process alive across the transition and would reopen exactly the "old worker
still running" window review focus #5 is about. At most 2 frames per stream per toggle —
about 133 ms of picture at 15 fps — and the discard is now counted rather than silent.

**Coalescing.** Toggling mid-flight bumps the generation again, so the earlier one is
superseded. `await_transition` on a superseded generation returns at once with
`bounded_by="timeout"` and an empty `confirmed`: a coalesced toggle resolves to the final
requested state, and the generation that was replaced is never acknowledged as though it
had landed. `route_plan().requested` is always the final mapping.

**The four combinations of spec section 5 are four `begin_transition` calls**, and each
must execute for real. The camera A slot stays on the CPU throughout: the toggles move
inference *jobs* onto the CPU, they do not disable hardware.

## Telemetry

`samples()` returns `MetricCollector.sample()` and nothing else - **never the
collector's `system_cpu_percent()` / `app_cpu_percent()` accessors.** Those share one
frozen `_Tick` with `sample()`, and calling one between samples moves the tick earlier
for everybody, so a card polling an accessor on its own timer can flip the booth's own
sample to `UNAVAILABLE` with no other visible cause. The docstrings on those accessors
say so; this module is the in-tree consumer that honours it.

The collector tracks **this process and every worker**, labelled `ui` and `worker`. With
three streams the worker labels become `worker_A`, `worker_B`, `worker_C`, so a metric can
say which process it belongs to; with one stream the label stays exactly `worker` so the
M1 single-stream shape is unchanged. It is rebuilt when a worker is respawned, because
the new process has a new pid and a stale pid would report a dead worker's CPU as
absent forever.

## `compile_failure` is a test seam, and says so

`Supervisor(device_id="NPU", compile_failure=True)` makes the spawned child raise before
it compiles anything, and the child says exactly that in the text the supervisor
publishes. `compile_failure=["B"]` fails **only** stream B, which is how "one failing
stream does not take down the other two" is exercised for real rather than simulated.
It is a fault injection, not a measurement of the NPU: it exercises the supervisor's
detect-a-dead-worker, restart-once, fail-visibly path without needing a device that
genuinely will not compile. A **real** compile refusal is exercised by
`tests/test_worker.py`, which asks for a device id this machine does not have and
requires an error rather than a run somewhere else.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from PySide6.QtCore import QObject, Signal

from app.contracts.frame import FramePacket
from app.contracts.metric_sample import MetricSample
from app.contracts.route_plan import RoutePlan, StreamStats
from app.engine.queue import BoundedFrameQueue
from app.engine.worker import InferenceResult, WorkerConfig, run_worker
from app.inference.devices import is_device
from app.telemetry.collector import MetricCollector

__all__ = [
    "COMPILE_FAILURE_REASON",
    "DEFAULT_STREAM_IDS",
    "REFUSAL_REASONS",
    "Supervisor",
    "StreamStats",
    "TransitionOutcome",
]

#: How often the watchdog asks the worker process whether it is still there. 50 ms
#: because the deadline that matters is a person watching a screen, not a benchmark.
EXIT_POLL_SECONDS: Final[float] = 0.05

#: How long a blocking queue read waits before looping, so the reader thread notices a
#: stop request promptly instead of sitting on `get()` until the queue is closed.
READ_POLL_SECONDS: Final[float] = 0.05

#: How long `stop()` waits for a worker to notice the `None` sentinel before it is
#: terminated. One CPU inference on this machine is 21 ms; 5 s is ~200 of them.
STOP_GRACE_SECONDS: Final[float] = 5.0

#: How long `_retire` spends discarding what a retiring queue still holds before it is
#: closed. **A ceiling, not a wait**: the loop returns as soon as the pipe and the
#: feeder's buffer are empty, which on the healthy path is the first read. It exists so
#: that a queue whose feeder will not drain - a surviving feeder, a queue a platform will
#: not read - cannot turn a teardown into an unbounded wait.
#:
#: Measured on this machine at the 2.0 s deadline, 20 transitions, 52 drain calls
#: (`_discard_pending` timed around its own body): 20 calls discarded anything, **22-31
#: frames** each for the retired input queues on the timeout path, and the whole drain cost
#: **p50 15.2 ms, max 23.8 ms, 0.73 s over 20 transitions**. No call came close to the
#: ceiling.
RETIRE_DRAIN_SECONDS: Final[float] = 0.5

#: How long a single read in `_retire`'s drain waits. Two milliseconds because the read
#: exists to find a frame whose write is in flight, and on this platform the pipe handle
#: is signalled immediately in that state; the timeout is only paid when the queue is
#: genuinely empty, once per retired queue.
RETIRE_POLL_SECONDS: Final[float] = 0.002

#: How long `_retire` waits for a closed queue's feeder thread to see the sentinel and
#: return. It is what makes it safe to forget the queue: the atexit finaliser only hangs on
#: a feeder thread that is still blocked, so the ledger entry is dropped only after the
#: thread is confirmed gone (see `_retire`). **A grace, not a guarantee**: in one of the
#: probe's runs at the 2.0 s deadline the walk ended with one queue still inside this wait,
#: and `stop()`'s sweep released it. That is why the entry is dropped on the evidence of a
#: dead thread and not on the fact that a drain was attempted.
RETIRE_FEEDER_GRACE_SECONDS: Final[float] = 0.2

#: How often `await_transition` re-reads the streams while it waits. 10 ms, because a
#: `Condition.wait` on this platform has a measured ~15 ms floor (see
#: `BoundedFrameQueue.get`), so a finer poll would cost the same wall-clock time and
#: report a coarser deadline. Polling on `time.sleep` is honest here: the loop reads
#: immutable state under the supervisor's own lock and sleeps between reads.
TRANSITION_POLL_SECONDS: Final[float] = 0.01

#: How many recent frame ages each stream keeps, for the p50/p95 a card renders.
#:
#: **Bounded, and this is the reason.** The age list was a plain `list` appended once per
#: analysed frame, which is ~90 000 floats an hour per stream and grows for ever.
#: `app/main.py`'s summary path copies it under the lock and double-`sorted()` it twice a
#: second on the Qt UI thread, so the cost of that copy and sort grew without bound during
#: a shift. A `deque(maxlen=...)` bounds both the memory and the sort.
#:
#: 256 samples is **deliberately the transport's window, not since-construction**: at 15
#: fps a stream contributes ~15 ages/s, so 256 is roughly **the last 17 seconds** of
#: analysis. That is the right scale for a number a person reads as "is the video
#: keeping up", and it is the figure the M2 report quotes, so it is stated here rather
#: than left for a reader to infer from `deque.maxlen`.
FRAME_AGE_WINDOW: Final[int] = 256

#: The stream ids used when a caller asks for several devices and names no ids.
DEFAULT_STREAM_IDS: Final[tuple[str, ...]] = ("A", "B", "C")

#: Every reason `submit` can refuse, and what each one means to the operator. A
#: refusal with no reason is the failure review focus #4 is about, so the vocabulary is
#: named here once and pinned by a test.
#:
#: **There is deliberately no `"transitioning"`.** `begin_transition` spawns the new
#: worker immediately, so a frame offered mid-transition goes straight onto the new
#: route's queue - which is spec section 5's "new target begins submitting", read
#: literally. An earlier version of this code refused mid-transition submissions with a
#: `"transitioning"` reason, and it was a deadlock: nothing could be submitted, so no
#: inference could run, so the new route could never confirm, so the transition could
#: only ever end in a timeout. Refusing frames to make a transition "clean" is not
#: available here, because the frames arrive from a 15 fps feed the presenter is not
#: pausing.
REFUSAL_REASONS: Final[tuple[str, ...]] = (
    "not_running",     # never started, or already stopped
    "closed",          # shutting down
    "failed",          # this stream's route gave up
    "unknown_stream",  # no such stream on this supervisor
)

#: The text published when `compile_failure=True`. Verbatim, and it says what it is:
#: an injected fault, not a measurement of any device.
COMPILE_FAILURE_REASON: Final[str] = (
    "COMPILE FAILURE INJECTED BY THE SUPERVISOR'S TEST SEAM (compile_failure=True). "
    "The worker process raised before compiling anything. This says nothing about the "
    "requested device's health; it exists so the supervisor's detect-a-dead-worker, "
    "restart-once and fail-visibly path can be exercised without a device that "
    "genuinely will not compile. A real compile refusal is covered by "
    "tests/test_worker.py, which asks for a device this machine does not have."
)


# ------------------------------------------------------------------------------------
# The records
# ------------------------------------------------------------------------------------

def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str; got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string; got {value!r}. An empty "
                         f"device id cannot be told apart from a missing one.")
    return value


def _require_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int; got {value!r} ({type(value).__name__})")
    if value < 0:
        raise ValueError(f"{name} must be >= 0; got {value}")
    return int(value)


@dataclass(frozen=True)
class TransitionOutcome:
    """What one `begin_transition` actually achieved, in measured numbers.

    ## `bounded_by` is `"timeout"` for every outcome that is not a confirmation

    There are exactly two spellings on purpose: `"confirmed"` and `"timeout"`, and the
    brief fixes them. `"timeout"` therefore means **"the transition was not
    acknowledged"**, and covers both of the ways that happens - the deadline passed, or an
    affected stream failed visibly, which cannot become a confirmation. `elapsed_s` and
    `confirmed` distinguish them: an early failure has `elapsed_s` well under
    `timeout_s`, and the reason it failed is on `Supervisor.errors()` and in
    `RoutePlan.errors`. Inventing a third spelling would have been more precise and would
    have put a word on the wire that every existing consumer of this record does not know.

    **`elapsed_s` is the age of the transition, not the time this call blocked.** The 2 s
    spec gate is a time from *toggle* to *confirmed placement*, so the deadline is
    measured from the transition's start; a caller that calls `await_transition` a
    second time on the same generation gets a larger `elapsed_s`, not a smaller one.

    ## `confirmed` carries only streams that proved something

    Same convention as `RoutePlan.confirmed`, for the same reason: a stream with no entry
    is **unproven**, not confirmed-on-nothing. So the streams that did *not* confirm are
    `set(requested) - set(confirmed)`, and `unconfirmed_streams` hands that over as a
    named call rather than as arithmetic every caller repeats.

    ## `discarded_results` belongs to *this* transition

    Results the superseded generation produced between `begin_transition` and
    `await_transition` returning. It is a count of work thrown away, kept because a
    discard that is not counted is a discard that can be argued about - and the count is
    attributable to the transition that caused it rather than to a process-wide total.
    """

    generation: int
    requested: MappingABC[str, str]
    confirmed: MappingABC[str, tuple[str, ...]]
    elapsed_s: float
    bounded_by: str
    discarded_results: int

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise TypeError(f"generation must be an int; got {self.generation!r}")
        if self.generation < 0:
            raise ValueError(f"generation must be >= 0; got {self.generation}")
        if self.bounded_by not in ("timeout", "confirmed"):
            raise ValueError(
                f"bounded_by must be 'timeout' or 'confirmed'; got {self.bounded_by!r}. A "
                f"transition is acknowledged only by a confirmed placement, so those are "
                f"the only two honest answers."
            )
        if not isinstance(self.requested, MappingABC):
            raise TypeError(f"requested must be a mapping; got "
                            f"{type(self.requested).__name__}")
        if not isinstance(self.confirmed, MappingABC):
            raise TypeError(f"confirmed must be a mapping; got "
                            f"{type(self.confirmed).__name__}")
        object.__setattr__(self, "requested", MappingProxyType({
            _require_text("requested key", str(k)): _require_text("requested value", v)
            for k, v in self.requested.items()
        }))
        object.__setattr__(self, "confirmed", MappingProxyType({
            _require_text("confirmed key", str(k)): tuple(
                _require_text("confirmed value entry", str(d)) for d in v)
            for k, v in self.confirmed.items()
        }))
        elapsed = float(self.elapsed_s)
        if elapsed != elapsed or elapsed in (float("inf"), float("-inf")):
            raise ValueError(f"elapsed_s must be finite; got {self.elapsed_s!r}")
        if elapsed < 0.0:
            raise ValueError(
                f"elapsed_s must be >= 0.0; got {self.elapsed_s}. A negative transition "
                f"time is a broken clock, and it is the one number in this record that "
                f"must never be adjusted to make a gate pass."
            )
        object.__setattr__(self, "elapsed_s", elapsed)
        object.__setattr__(self, "discarded_results",
                           _require_non_negative_int("discarded_results",
                                                     self.discarded_results))
        unrequested = sorted(set(self.confirmed) - set(self.requested))
        if unrequested:
            raise ValueError(
                f"confirmed names streams {unrequested} that requested does not. A "
                f"placement is reported for a route that was requested."
            )

    @property
    def acknowledged(self) -> bool:
        """Whether this transition was acknowledged by a confirmed placement.

        The single question a caller should ask. It is `True` only when
        `bounded_by == "confirmed"`, which is reachable only through `EXECUTION_DEVICES`
        read after an inference - never through the toggle being made.
        """
        return self.bounded_by == "confirmed"

    def unconfirmed_streams(self) -> tuple[str, ...]:
        """The streams this transition asked for and did **not** prove, sorted.

        Named, because "which camera did not move" is the question an operator asks at
        the moment a transition times out, and it should not be every caller's job to
        re-derive it from two mappings.
        """
        return tuple(sorted(set(self.requested) - set(self.confirmed)))

    def to_json(self) -> dict[str, Any]:
        """A `json.dumps`-acceptable payload."""
        return {
            "generation": self.generation,
            "requested": dict(self.requested),
            "confirmed": {k: list(v) for k, v in self.confirmed.items()},
            "elapsed_s": self.elapsed_s,
            "bounded_by": self.bounded_by,
            "acknowledged": self.acknowledged,
            "discarded_results": self.discarded_results,
            "unconfirmed_streams": list(self.unconfirmed_streams()),
        }


@dataclass
class _Transition:
    """One open or finished `begin_transition`, as the supervisor remembers it.

    **`id` is the transition's own identity and is monotonic per call**, independent of
    `generation`. The route generation only moves when a device actually changes, so it
    cannot key this record: two restatements inside one generation would collide, and the
    second would overwrite the first's `started` - handing a late `await_transition` a
    deadline measured from the *wrong* moment. `generation` is still carried, because it
    is what `TransitionOutcome.generation` reports and what a reader of a log wants.
    """

    id: int
    generation: int
    streams: tuple[str, ...]
    requested: dict[str, str]
    previous: dict[str, str]
    started: float
    discarded: int = 0
    recovered: bool = False


class _Stream:
    """Everything one stream owns: its queue, its worker, its counters, its proof.

    Mutable and guarded by `Supervisor._lock`; never handed out. `StreamStats` is the
    immutable snapshot of it that a consumer sees.
    """

    __slots__ = (
        "stream_id", "requested_device", "config", "queue", "compile_failure",
        "generation", "generations_seen", "superseded_by", "confirmed", "state",
        "errors", "restarts", "attempts", "accepted", "refused", "refusal_reasons",
        "analysed", "frame_ages_ms", "latest", "failure_published", "invalidated",
        "process", "in_queue", "out_queue", "pid", "feeder", "reader", "watchdog",
    )

    def __init__(self, stream_id: str, requested_device: str, config: WorkerConfig,
                 queue: BoundedFrameQueue, compile_failure: bool,
                 generation: int) -> None:
        self.stream_id = stream_id
        self.requested_device = requested_device
        self.config = config
        self.queue = queue
        self.compile_failure = compile_failure
        self.generation = int(generation)
        self.generations_seen: list[int] = [int(generation)]
        #: The generation that superseded this stream's current one, when there is one.
        #: This is what attributes a discarded result to the transition that caused it.
        self.superseded_by: int | None = None
        self.confirmed: tuple[str, ...] = ()
        self.state = "pending"
        self.errors: list[str] = []
        self.restarts = 0
        self.attempts = 0
        self.accepted = 0
        self.refused = 0
        self.refusal_reasons: dict[str, int] = {}
        self.analysed = 0
        self.frame_ages_ms: deque[float] = deque(maxlen=FRAME_AGE_WINDOW)
        self.latest: InferenceResult | None = None
        self.failure_published = False
        #: Frames accepted into the bounded queue and then thrown away by a route change
        #: before the worker saw them (`_drain_queue`). A fourth population, distinct from
        #: a drop (the queue was full), a refusal (the pipeline was not there) and a
        #: discarded *result* (work that ran and was superseded). See `StreamStats`.
        self.invalidated = 0
        self.process: mp.process.BaseProcess | None = None
        self.in_queue: Any = None
        self.out_queue: Any = None
        self.pid: int | None = None
        self.feeder: threading.Thread | None = None
        self.reader: threading.Thread | None = None
        self.watchdog: threading.Thread | None = None

    def stats(self) -> StreamStats:
        """The immutable per-stream record. A snapshot, safe to hand to a card."""
        return StreamStats(
            stream_id=self.stream_id,
            requested_device=self.requested_device,
            confirmed_devices=tuple(self.confirmed),
            frames_accepted=self.accepted,
            frames_refused=self.refused,
            frames_refused_reasons=dict(self.refusal_reasons),
            drops=self.queue.drops,
            state=self.state,
            generations_seen=tuple(self.generations_seen),
            frame_attempts=self.attempts,
            frames_analysed=self.analysed,
            generation=self.generation,
            frames_invalidated=self.invalidated,
        )

    def clear_proof(self) -> None:
        """Everything that belonged to the previous routing decision.

        A placement proved under the old routing is not a proof under the new one, and a
        plan that kept `GPU.0` in `confirmed` while the new generation is still `pending`
        would be claiming a placement for a route that has not run.

        **The frame-age history is deliberately NOT cleared here.** It was, and it was
        wrong: the lag percentile is a measurement of the *transport* - ingress to result,
        through a pickling queue - not of the device. Wiping it on every device change
        meant the displayed p50/p95 changed the instant a presenter pressed a toggle,
        with nothing about latency having changed, so the number on the card moved for a
        reason the card could not show. The bounded window (`FRAME_AGE_WINDOW`) does the
        bounding; the route change does not.
        """
        self.confirmed = ()
        self.errors = []
        self.latest = None
        self.failure_published = False
        self.restarts = 0


# ------------------------------------------------------------------------------------
# The spawn entry points. Module-level functions, because spawn needs them.
# ------------------------------------------------------------------------------------

def _faulting_worker_entry(config: WorkerConfig, in_queue: Any, out_queue: Any) -> None:
    """The spawn entry point used for a stream with `compile_failure=True`.

    Raises, so the process exits non-zero with this text on stderr and the supervisor
    observes a real dead worker rather than a simulated notification. The supervisor
    names the reason itself; this is what makes the death real.
    """
    raise RuntimeError(COMPILE_FAILURE_REASON)


def _worker_entry(config: WorkerConfig, in_queue: Any, out_queue: Any) -> None:
    """The spawn entry point. A module-level function, because spawn needs one.

    Windows `spawn` re-imports this module in the child and looks the target up by
    qualified name. A closure or a bound method cannot be found, and the failure is a
    `PicklingError`/`AttributeError` in the child at start time rather than anything
    diagnosable from the parent.
    """
    run_worker(config, in_queue, out_queue)


# ------------------------------------------------------------------------------------
# The supervisor
# ------------------------------------------------------------------------------------

class Supervisor(QObject):
    """Owns N inference worker processes, their routes, and the transitions between them.

    Construct one stream with `device_id=` - which is what M1's `app.main` does and what
    keeps working unchanged - or several with `device_ids=`, which is M2's three-camera
    panel:

    ```python
    Supervisor(device_ids=("CPU", "GPU", "NPU"))     # streams A, B, C
    Supervisor(device_ids=("CPU", "CPU", "CPU"))     # spec section 5, row 4
    ```

    `device_id` is the **requested** device and is recorded as such. Nothing in this class
    ever promotes it to a confirmed placement; only `InferenceResult.execution_devices`
    can, and only after its own inference has run.

    The single-stream accessors (`frames_submitted`, `drops`, `latest_result`, ...) are
    **sums over the streams**, which is why the M1 code paths and the M1 tests are
    unchanged: with one stream the sum is that stream.
    """

    #: An `InferenceResult` for a frame that has been analysed (or has failed). Emitted
    #: from the reader thread of the stream that produced it; connect with a queued
    #: connection, per AGENTS.md rule 11. **Never emitted for a superseded generation.**
    result_ready = Signal(object)
    #: The new `RoutePlan`, on every state change. Emitted from whichever thread
    #: observed the change.
    state_changed = Signal(object)
    #: One stated reason, as text. Never a formatted exception object.
    error_raised = Signal(str)

    def __init__(
        self,
        device_id: str | None = None,
        *,
        device_ids: Sequence[str] | None = None,
        stream_id: str | None = None,
        stream_ids: Sequence[str] = DEFAULT_STREAM_IDS,
        session_id: str = "supervisor",
        scenario_id: str = "retail_sanity",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        queue_size: int = 2,
        compile_failure: bool | Sequence[str] = False,
        collector_interval: float = 0.25,
        clock=time.perf_counter,
    ) -> None:
        super().__init__()
        requested = self._requested_devices(device_id, device_ids)
        ids = self._stream_identifiers(stream_id, stream_ids, len(requested))

        self._clock = clock
        self._queue_size = int(queue_size)
        self._conf_threshold = float(conf_threshold)
        self._iou_threshold = float(iou_threshold)
        self._collector_interval = float(collector_interval)

        self._lock = threading.RLock()
        self._order: tuple[str, ...] = ids
        self._generation = 1
        self._transitions: dict[int, _Transition] = {}
        #: The last id handed out by `begin_transition`. Monotonic per call; see
        #: `_Transition.id`. Not the generation, which only moves on a real change.
        self._transition_seq = self._generation
        self._discarded_results = 0
        self._discarded_mismatched = 0
        self._frame_attempts_globally = 0
        self._unknown_stream_refusals = 0
        self._unknown_stream_reasons: dict[str, int] = {}
        self._stopping = False
        self._running = False
        self._streams: dict[str, _Stream] = {}

        failing = self._failing_streams(compile_failure, ids)
        for index, sid in enumerate(ids):
            config = self._worker_config(requested[index])
            self._streams[sid] = _Stream(
                stream_id=sid,
                requested_device=requested[index],
                config=config,
                queue=BoundedFrameQueue(maxsize=int(queue_size), clock=clock),
                compile_failure=sid in failing,
                generation=self._generation,
            )

        #: Every queue created and **not yet released**, across generations and streams -
        #: live ones included, because a queue that is live at `stop()` has to be reached
        #: by the final sweep. `_retire` drops an entry only once its feeder thread is
        #: confirmed gone, so what stays here is exactly what `multiprocessing`'s atexit
        #: finaliser could still block on. See `_retire`.
        self._retired_queues: list[Any] = []
        self._collector: MetricCollector | None = None
        #: The most recent `samples()` result, kept so a run summary can be built after
        #: `stop()` has torn the collector down. See `last_samples`.
        self._last_samples: tuple[MetricSample, ...] = ()

    # -- construction helpers ---------------------------------------------------------

    @staticmethod
    def _requested_devices(device_id: str | None,
                           device_ids: Sequence[str] | None) -> tuple[str, ...]:
        """The per-stream requested devices, from either spelling of the constructor.

        `device_id` (one str) and `device_ids` (a sequence) are two ways of saying the
        same thing; passing both is refused rather than silently resolved, because
        "which did the caller mean" has no safe default once there are three streams.
        """
        if device_ids is not None:
            if device_id is not None:
                raise ValueError(
                    "pass either device_id (one stream) or device_ids (several), not "
                    f"both. Got device_id={device_id!r} and device_ids={list(device_ids)!r}; "
                    "resolving that silently would pick a route nobody asked for."
                )
            if isinstance(device_ids, (str, bytes)):
                raise TypeError(
                    f"device_ids must be a sequence of device ids, not a single string; "
                    f"got {device_ids!r}. A bare string would iterate to characters."
                )
            entries = tuple(device_ids)
            if not entries:
                raise ValueError(
                    "device_ids must name at least one device. A supervisor with no "
                    "streams has no route, no proof and no place to put a frame."
                )
            for entry in entries:
                _require_text("device_ids entry", str(entry))
            # **Repeats are legal and required.** Spec section 5's fourth row is three
            # streams all on the CPU, and refusing a duplicate here would make the one
            # configuration the table is built around unbuildable. An earlier version of
            # this code did refuse it, on the reasoning that "two streams on one device
            # must be a mistake"; the spec table says otherwise, and the spec is right.
            return tuple(str(entry) for entry in entries)
        chosen = "CPU" if device_id is None else device_id
        if not isinstance(chosen, str) or not chosen.strip():
            raise ValueError(
                f"device_id must be a non-empty string; got {chosen!r}. The supervisor "
                f"records what it was asked to run on, and an empty request cannot be "
                f"distinguished from no request."
            )
        return (chosen,)

    @staticmethod
    def _stream_identifiers(stream_id: str | None, stream_ids: Sequence[str],
                            count: int) -> tuple[str, ...]:
        """One stream id per stream, in order, unique, and never invented mid-list."""
        if stream_id is not None:
            _require_text("stream_id", stream_id)
        if isinstance(stream_ids, (str, bytes)):
            raise TypeError(
                f"stream_ids must be a sequence of ids, not a single string; got "
                f"{stream_ids!r}"
            )
        entries = tuple(str(e) for e in stream_ids)
        for entry in entries:
            _require_text("stream_ids entry", entry)
        if len(entries) < count:
            raise ValueError(
                f"{count} device(s) were requested but stream_ids names only "
                f"{len(entries)}: {list(entries)}. Inventing an id for a camera the "
                f"caller did not name would put a stream on screen that nobody "
                f"configured."
            )
        chosen = entries[:count]
        if len(set(chosen)) != len(chosen):
            raise ValueError(
                f"stream ids must be unique; got {list(chosen)}. Two streams sharing an "
                f"id would file their frames and their counters against each other."
            )
        if stream_id is not None:
            # M1's spelling, for one stream. In multi-stream mode it may only agree
            # with the first of the named ids; anything else is two answers to one
            # question.
            if stream_id != chosen[0]:
                raise ValueError(
                    f"stream_id={stream_id!r} disagrees with stream_ids[0]="
                    f"{chosen[0]!r}. Name the streams once, through stream_ids."
                )
        return chosen

    @staticmethod
    def _failing_streams(compile_failure: bool | Sequence[str],
                         ids: tuple[str, ...]) -> tuple[str, ...]:
        """Which streams get the injected fault: all of them, or a named few.

        A sequence is the point of this shape: "one stream fails, the other two carry on"
        is a Task 1 requirement, and with a single global flag it could only be expressed
        as "three streams fail", which is the wrong test.
        """
        if isinstance(compile_failure, bool):
            return ids if compile_failure else ()
        if isinstance(compile_failure, (str, bytes)):
            raise TypeError(
                f"compile_failure must be a bool or a sequence of stream ids; got the "
                f"string {compile_failure!r}. A bare string would be iterated as a name."
            )
        names = tuple(str(name) for name in compile_failure)
        unknown = sorted(set(names) - set(ids))
        if unknown:
            raise ValueError(
                f"compile_failure names streams {unknown} that this supervisor does not "
                f"own. Its streams are {list(ids)}."
            )
        return tuple(name for name in ids if name in names)

    def _worker_config(self, device_id: str) -> WorkerConfig:
        return WorkerConfig(
            device_id=device_id,
            model_id=self._model_id(),
            input_width=self._input_extent(3),
            input_height=self._input_extent(2),
            conf_threshold=self._conf_threshold,
            iou_threshold=self._iou_threshold,
        )

    # -- the pinned model's facts, read once, never assumed ---------------------------

    @staticmethod
    def _model_id() -> str:
        from app.inference.pinned import MODEL_ID

        return MODEL_ID

    @staticmethod
    def _input_extent(axis: int) -> int:
        from app.inference.pinned import pinned_input_shape

        return int(pinned_input_shape()[axis])

    # -- the plan ---------------------------------------------------------------------

    def route_plan(self) -> RoutePlan:
        """The current generation's plan. Safe to call from any thread.

        `requested` names every stream this supervisor owns, so it is populated from
        construction - what was asked is known immediately. `confirmed` is populated only
        by a result, and is **emptied per stream** when that stream's route fails.

        `generation` is the **highest** generation across the streams: the age of the
        newest routing decision on this plan. A stream that did not move keeps its own
        older generation, and `stream_states` says what each stream is doing - see
        `RoutePlan`'s docstring for why the aggregate `state` must not be read as a
        per-stream answer.
        """
        with self._lock:
            return RoutePlan(
                generation=max(s.generation for s in self._streams.values()),
                requested={sid: s.requested_device for sid, s in self._streams.items()},
                confirmed={sid: tuple(s.confirmed) for sid, s in self._streams.items()
                           if s.confirmed},
                state=self._aggregate_state(),
                errors=self._all_errors(),
                stream_states={sid: s.state for sid, s in self._streams.items()},
                # **M2 Task 5: the per-stream reason channel.** `_Stream.errors` has
                # always been per stream - it is cleared by that stream's own
                # `clear_proof()` and appended to by that stream's own failures - and only
                # the *concatenation* was ever published. The concatenation is the wrong
                # thing to put on a card: with three streams it shows Camera C's failure
                # carrying Camera A's and B's sentences, and a card reading `failed` with
                # no reason of its own asks the presenter to work out which camera died.
                stream_errors={sid: tuple(s.errors) for sid, s in self._streams.items()
                               if s.errors},
            )

    def _aggregate_state(self) -> str:
        """`failed` only when every stream failed; `active` if any one has proved a
        placement; otherwise `pending`.

        With one stream this is exactly that stream's state, which is what M1 read. With
        three it answers "is anything wrong", and `stream_states` answers "what is Camera
        C doing" - a consumer that renders one camera must never use this.
        """
        states = {s.state for s in self._streams.values()}
        if states and states == {"failed"}:
            return "failed"
        if "active" in states:
            return "active"
        return "pending"

    def _all_errors(self) -> tuple[str, ...]:
        out: list[str] = []
        for sid in self._order:
            out.extend(self._streams[sid].errors)
        return tuple(out)

    # -- the M1 single-stream accessors, now sums over the streams -------------------

    @property
    def device_id(self) -> str:
        """The **requested** device of the first stream. Not a placement claim.

        A convenience for the single-stream caller (`app.main`, M1's tests). With three
        streams the question "what was this stream asked to run on" has a per-stream
        answer: `RoutePlan.requested_device(stream_id)`, or `StreamStats.requested_device`.
        """
        return self._streams[self._order[0]].requested_device

    @property
    def stream_id(self) -> str:
        """The first stream's id. See `device_id`."""
        return self._order[0]

    @property
    def stream_ids(self) -> tuple[str, ...]:
        """Every stream id this supervisor owns, in construction order."""
        return tuple(self._order)

    @property
    def generation(self) -> int:
        """The highest generation across the streams - see `route_plan`."""
        with self._lock:
            return max(s.generation for s in self._streams.values())

    @property
    def restart_count(self) -> int:
        """How many times a failed worker has been restarted, **summed over streams**.

        At most 1 per stream per generation. A supervisor that restarted in a loop would
        be visible in a test.
        """
        with self._lock:
            return sum(s.restarts for s in self._streams.values())

    def restart_count_for(self, stream_id: str) -> int:
        """How many times **this** stream's worker has been restarted."""
        with self._lock:
            return self._require_stream(stream_id).restarts

    def errors(self) -> tuple[str, ...]:
        """Every stated reason any stream has published, oldest first, stream order.

        A **method**, not a property, per the interface the brief specifies
        (`self.assertTrue(s.errors())`). A property of the same name would make
        `s.errors` a tuple - truthy, and silently passing an assertion that meant
        "is there a stated reason", which is the exact failure mode this repository
        keeps finding in its own test suite.
        """
        with self._lock:
            return self._all_errors()

    def is_device(self, device_id: str, stream_id: str | None = None) -> bool:
        """Whether a placement has been **confirmed** on `device_id`.

        Reads the confirmed map only, and only for the named stream (the first one by
        default, which is M1's behaviour). `Supervisor(device_id="NPU").is_device("NPU")`
        is `False` until an NPU inference reports `EXECUTION_DEVICES`, and stays `False`
        after a failure - **for that stream only**: with three streams, `is_device("CPU")`
        being `True` says Camera A is confirmed on the CPU and says nothing about B or C.

        Compare through `app.inference.devices.is_device`, because the readings are
        inconsistent on this machine: CPU `['CPU']`, GPU `['GPU.0']`, NPU the bare string
        `'NPU'`.
        """
        with self._lock:
            stream = self._require_stream(stream_id or self._order[0])
            confirmed = tuple(stream.confirmed)
            state = stream.state
        if state == "failed":
            return False
        return is_device(confirmed, device_id)

    # -- per-stream reads --------------------------------------------------------------

    def _require_stream(self, stream_id: str) -> _Stream:
        stream = self._streams.get(str(stream_id))
        if stream is None:
            raise KeyError(
                f"no stream {stream_id!r} on this supervisor. Its streams are "
                f"{list(self._order)}. Routing a frame at a camera that does not exist "
                f"is how a result ends up attributed to the wrong stream."
            )
        return stream

    def per_stream(self) -> tuple[StreamStats, ...]:
        """One `StreamStats` per stream, in construction order.

        The per-stream accounting, and the per-stream identity
        `accepted + refused + drops == attempts` is checkable against
        (`StreamStats.reconciles`). An aggregate that reconciles while one stream's does
        not is exactly the failure this exists to catch.
        """
        with self._lock:
            return tuple(self._streams[sid].stats() for sid in self._order)

    def stream_stats(self, stream_id: str) -> StreamStats:
        """One stream's `StreamStats`."""
        with self._lock:
            return self._require_stream(stream_id).stats()

    def pid_for(self, stream_id: str) -> int | None:
        """The **live** worker process id for `stream_id`, or `None` if it has no worker.

        Exposed because "is there a process behind this camera?" is a question a
        diagnostics screen and a test both need, and both were reaching into `_streams` to
        answer it. A pid is **not** a booth metric - nothing renders it - but it is the
        only way to say whether a route change actually replaced the worker, which is the
        load-bearing claim of the transition protocol.

        **A dead process reports `None`, not its stale pid.** `_Stream.pid` is *retained*
        after a failure on purpose - `stop()` and `_restart` need the `Process` object to
        join and terminate it - so the raw field says 9264 for a worker that exited with
        code 1. Reporting that would say "there is a process behind Camera C" when there is
        none, which is the failure this whole report is about. Measured: with
        `compile_failure=["C"]` the raw field reads `9264` with `is_alive() == False`, and
        this returns `None`.

        So `None` is the honest answer for a `failed` stream: it has spent its restart
        budget and has no running worker.
        """
        with self._lock:
            stream = self._streams.get(str(stream_id))
            if stream is None:
                return None
            process = stream.process
            if process is None or not process.is_alive():
                return None
            return stream.pid

    def stream_state(self, stream_id: str) -> str | None:
        """This stream's own state: `pending`, `active`, `failed`, or `None` for unknown.

        The per-stream answer, and the one a card for one camera must read.
        `route_plan().state` is the aggregate over three streams and answers "is anything
        wrong", never "what is Camera C doing".
        """
        with self._lock:
            stream = self._streams.get(str(stream_id))
            return None if stream is None else stream.state

    def results(self) -> MappingABC[str, InferenceResult]:
        """The newest result per stream, read-only. Streams with none are **absent**.

        Same convention as `RoutePlan.confirmed`: absent means nothing has come back for
        that stream, which is not the same as a result with nothing in it.
        """
        with self._lock:
            return MappingProxyType({
                sid: stream.latest for sid, stream in self._streams.items()
                if stream.latest is not None
            })

    def latest_result(self) -> InferenceResult | None:
        """The newest result of the **first** stream, or `None`.

        Kept for M1's `app.main`, which is single-stream. `results()` is the per-stream
        question, and mixing the two up is how one camera's boxes land on another's
        frame.
        """
        with self._lock:
            return self._streams[self._order[0]].latest

    # -- submission -------------------------------------------------------------------

    def submit(self, frame: FramePacket) -> bool:
        """Offer a frame for analysis. `False` means it was refused, with a reason.

        The frame's own `stream_id` chooses the stream, so a caller cannot put Camera B's
        pixels into Camera A's queue by accident. A frame naming a stream this supervisor
        does not own is refused and counted under `"unknown_stream"` - **not** silently
        dropped, because a frame that went nowhere has to be attributable.

        `False` for a closed or failed route is a **refusal, not a drop**, and does not
        move `drops`: a frame arriving after the consumer has gone is a shutdown, and
        mixing the two would make the drop counter unreadable exactly when somebody is
        diagnosing it. A full queue is a drop and does move it.

        **Every counter this method moves is moved inside one hold of `self._lock`,
        including the queue `put`.** It did not, and that made the per-stream identity
        `accepted + refused + drops == attempts` transiently *false* under live traffic:
        `attempts` was incremented under the lock, the `put` and the `accepted += 1` ran
        outside it, and a `per_stream()` snapshot landing in that window saw
        `attempts = 1, accepted = 0` and did not reconcile. A checkable invariant that
        fails intermittently is worse than one that is never checkable, because it fails
        on the booth and not in the test.
        `BoundedFrameQueue.put` cannot block - it returns `False` when full and never
        waits - so holding the lock across it costs nothing.

        **A refusal increments that stream's `frames_refused` and records why.** Measured
        on the shape this replaces: never started, stopped, and route-failed all produced
        `frames_submitted == 0` and `drops == 0`, so a caller that forgot to call
        `start()` produced a summary indistinguishable from a healthy run that had not
        been given any frames. The four reasons are kept apart - `not_running`, `closed`,
        `failed` and `unknown_stream` - because they call for four different responses.

        **A frame offered mid-transition is accepted, not refused.** `begin_transition`
        has already spawned the new worker and its queue is empty, so the frame becomes
        the new target's first submission and spec section 5's "new target begins
        submitting" is what actually happens. See `REFUSAL_REASONS` for why refusing here
        was tried first and is a deadlock.

        Raises `TypeError` for anything that is not a `FramePacket` - a malformed put is
        a bug in the caller, and counting it as lag would put a wrong number on the lag
        counter. A malformed put has no stream to be charged to, so it is charged to
        nothing.
        """
        if not isinstance(frame, FramePacket):
            raise TypeError(
                f"submit takes a FramePacket; got {type(frame).__name__}. A bare list of "
                f"detections has no frame identity, so a result carrying one could not be "
                f"matched to the picture it came from."
            )
        with self._lock:
            self._frame_attempts_globally += 1
            stream = self._streams.get(frame.stream_id)
            if stream is None:
                # An unknown stream is still an attempt somebody made and somebody has to
                # see. It is counted here, on the supervisor, because there is no stream
                # to count it on.
                self._unknown_stream_refusals += 1
                self._unknown_stream_reasons[frame.stream_id] = (
                    self._unknown_stream_reasons.get(frame.stream_id, 0) + 1)
                return False
            stream.attempts += 1
            reason = ""
            if stream.state == "failed":
                reason = "failed"
            elif self._stopping:
                reason = "closed"
            elif not self._running:
                reason = "not_running"
            if reason:
                stream.refused += 1
                stream.refusal_reasons[reason] = stream.refusal_reasons.get(reason, 0) + 1
                return False
            # Inside the same hold as `attempts`, deliberately - see the docstring. The
            # put cannot block, so this narrows no window that mattered.
            accepted = stream.queue.put(frame)
            if accepted:
                stream.accepted += 1
            return accepted

    @property
    def frames_refused(self) -> int:
        """Frames `submit` refused, **summed over the streams**, with reasons.

        Never conflated with `drops`: a drop means the queue was full (the pipeline fell
        behind), a refusal means the pipeline was not there at all. Frames aimed at a
        stream this supervisor does not own are refused too and are reported by
        `unknown_stream_refusals` - there is no stream to put them on.
        """
        with self._lock:
            return (sum(s.refused for s in self._streams.values())
                    + self._unknown_stream_refusals)

    @property
    def frames_refused_reasons(self) -> dict[str, int]:
        """Why each refusal happened, summed over the streams.

        Plus `unknown_stream`, which has no stream to belong to.
        """
        with self._lock:
            reasons: dict[str, int] = {}
            for stream in self._streams.values():
                for key, count in stream.refusal_reasons.items():
                    reasons[key] = reasons.get(key, 0) + count
            if self._unknown_stream_refusals:
                reasons["unknown_stream"] = (
                    reasons.get("unknown_stream", 0) + self._unknown_stream_refusals)
            return reasons

    @property
    def unknown_stream_refusals(self) -> int:
        """Frames refused because their `stream_id` names no stream here.

        Counted separately rather than folded into a stream's numbers, because putting
        another stream's frame on this stream's counters is the error this count exists
        to surface - it would reconcile perfectly and be entirely wrong.
        """
        with self._lock:
            return self._unknown_stream_refusals

    def unknown_stream_refusal_detail(self) -> dict[str, int]:
        """Which unknown stream id was aimed at, and how often."""
        with self._lock:
            return dict(self._unknown_stream_reasons)

    @property
    def frame_attempts(self) -> int:
        """Every `submit` call that reached the routing decision, accepted or not.

        So that `frames_submitted + frames_refused + drops == frame_attempts` is a check
        somebody can make rather than a hope - for the aggregate **and, through
        `StreamStats`, for each stream separately**.
        """
        with self._lock:
            return self._frame_attempts_globally

    @property
    def drops(self) -> int:
        """Frames refused for want of room, summed over the streams."""
        with self._lock:
            return sum(s.queue.drops for s in self._streams.values())

    @property
    def frames_submitted(self) -> int:
        """Frames accepted into the queues, summed over the streams."""
        with self._lock:
            return sum(s.accepted for s in self._streams.values())

    @property
    def frames_analysed(self) -> int:
        """Frames that came back with a placement. Failures are not counted here."""
        with self._lock:
            return sum(s.analysed for s in self._streams.values())

    @property
    def frames_invalidated(self) -> int:
        """Frames a route change threw away, summed over the streams.

        For symmetry with `drops`, `frames_submitted` and `frames_analysed`, so a
        three-stream caller does not have to sum a per-stream field by hand. It is the
        **fourth** population, not a fifth way of accounting for the same `submit`: a frame
        it counts was already counted once, as accepted, at `submit` time, and the route
        change is what discarded it. See `StreamStats.frames_invalidated`.
        """
        with self._lock:
            return sum(s.invalidated for s in self._streams.values())

    def frame_ages_ms(self) -> tuple[float, ...]:
        """Measured ingress-to-result age per analysed frame, in milliseconds.

        **A bounded recent window, not since-construction.** Each stream keeps the last
        `FRAME_AGE_WINDOW` ages - 256, roughly the last 17 seconds of a 15 fps stream - so
        the copy under the lock and the sort a caller does on it are both bounded. The
        sort is the caller's, but the cost of it is set here.
        """
        with self._lock:
            out: list[float] = []
            for sid in self._order:
                out.extend(self._streams[sid].frame_ages_ms)
            return tuple(out)

    def stream_frame_ages_ms(self, stream_id: str) -> tuple[float, ...]:
        """**This one stream's** recent frame ages, in milliseconds.

        **M2 Task 5 addition, and it is not a convenience.** `frame_ages_ms()` merges all
        three streams into one population, and the measured unsaturated three-stream
        condition - spec section 5 row 4, all three on the CPU - is bimodal enough that a
        merged p50 is a number about no camera at all. A card that shows it would be
        showing the average lateness of three streams on a card for one of them, and the
        presenter would read it as that camera's.

        Same bounded window as the merged accessor: the most recent
        `FRAME_AGE_WINDOW` ages **for this stream**, so the merged figure is not the mean
        of three windows either - it is the percentile of up to 3 x 256 samples.
        """
        with self._lock:
            stream = self._streams.get(str(stream_id))
            return tuple(stream.frame_ages_ms) if stream is not None else ()

    # -- generation-scoped result acceptance -------------------------------------------

    def accept_result(self, stream_id: str, generation: int, result: InferenceResult) -> bool:
        """The one path every result takes. `False` means it was **discarded**.

        This is not a test seam: it is what each stream's reader thread calls, and it is
        public so a test can drive the production path with a chosen generation instead of
        trying to win a race against a real worker.

        The rule is one comparison. A result read by the reader thread of generation `g`
        for stream `S` is delivered only if `g` is `S`'s current generation; otherwise it
        is **discarded, counted, and never emitted**. `Supervisor.discarded_results` is
        the total and `TransitionOutcome.discarded_results` is this transition's share.

        A result for a stream this supervisor does not own, or one whose frame does not
        carry that stream's id, is discarded the same way and counted under
        `discarded_mismatched` - a result whose frame says Camera B while its worker was
        Camera A's has no business being drawn.
        """
        with self._lock:
            stream = self._streams.get(str(stream_id))
            if stream is None:
                self._discarded_results += 1
                self._discarded_mismatched += 1
                return False
            if int(generation) != stream.generation:
                self._discarded_results += 1
                self._attribute_discard(stream)
                return False
            if result.frame.stream_id != stream.stream_id:
                # The worker's own frame disagrees with the stream it was reading for.
                # Never delivered: this is the frame-identity defect one layer up.
                self._discarded_results += 1
                self._discarded_mismatched += 1
                return False
            return self._apply_result(stream, int(generation), result)

    def _attribute_discard(self, stream: _Stream) -> None:
        """Charge one discarded result to the transition that superseded `stream`.

        `_Stream.superseded_by` records the *route* generation, so the lookup is by
        generation rather than by transition id. A restatement creates a new transition
        at the **same** route generation, and the later one then receives discards that
        the earlier one caused - so the per-transition `discarded_results` is an
        approximation in exactly the coalescing case, while the process-wide
        `discarded_results` is exact.

        Making it exact would mean stamping the transition id on the stream at supersede
        time, and `superseded_by` is already recorded before the id exists (`_apply_devices`
        runs before `begin_transition` allocates it). The right fix is to allocate the id
        in `_apply_devices` and pass it down, and it is recorded here as a known
        approximation rather than papered over: **the total is the honest number and the
        per-transition split is not.** Task 3's stress reporting must use the total.
        """
        owner = stream.superseded_by
        transition = self._transitions.get(owner) if owner is not None else None
        if transition is not None:
            transition.discarded += 1

    @property
    def discarded_results(self) -> int:
        """Results thrown away because their generation was superseded.

        Counted because a discard that is not counted is a discard that can be argued
        about. Distinct from a drop (a frame nobody analysed) and distinct from a coasted
        track (a tracker holding a box for an object that left) - three different things
        that must not be summed into one "lost frames" figure.
        """
        with self._lock:
            return self._discarded_results

    @property
    def discarded_mismatched(self) -> int:
        """Results discarded because their frame named a different stream than the reader.

        Always 0 on a correct run. It is a separate counter so that "we threw away old
        results" and "we threw away a result that belonged to another camera" cannot be
        reported as one benign number.
        """
        with self._lock:
            return self._discarded_mismatched

    # -- lifecycle --------------------------------------------------------------------

    def start(self) -> None:
        """Spawn every stream's worker. Idempotent; does not wait for a placement.

        Returns as soon as the processes exist. Every stream's `state` stays `pending`
        until a result proves a placement, so a slow compile shows as "moving..." on
        screen rather than as a device that has not run anything.
        """
        with self._lock:
            if self._running and not self._stopping:
                return
            self._stopping = False
            self._running = True
        for sid in self._order:
            self._spawn(sid)

    def stop(self) -> None:
        """Stop every worker, its threads and the collector. Idempotent, and prompt.

        `submit` returns `False` afterwards without counting a drop: a frame arriving
        after the consumer has gone is a shutdown, not the analysis falling behind.
        """
        with self._lock:
            if self._stopping and not self._running and not self._retired_queues:
                return
            self._stopping = True
            self._running = False
            claimed = []
            for sid in self._order:
                stream = self._streams[sid]
                claimed.append((stream, stream.process, stream.in_queue, stream.out_queue))
                stream.process = None
                stream.in_queue = None
                stream.out_queue = None
                stream.pid = None
                stream.queue.close()
        for _stream, _process, in_queue, _out in claimed:
            # **The stop sentinel goes in BEFORE the threads are joined**, which is the
            # whole of this fix. It used to go in after, which meant the worker never saw
            # it, so a feeder blocked writing to a full pipe waited out its full 5 s grace
            # and `stop()` cost three streams' worth of that: measured **15.18 s before and
            # 0.84 s after**, same probe, six toggles first. (Task 3 also measured 0.27 s
            # by letting the queues drain for 1.5 s *before* calling `stop()` at all - that
            # was a workaround around the ordering, and this is the ordering, so the honest
            # comparison is against the 15.18 s and not against the workaround.)
            # See `_signal_stop` and `_reap_worker`.
            self._signal_stop(in_queue)
        for sid in self._order:
            self._halt_threads(sid)
        for stream, process, in_queue, _out in claimed:
            if process is not None:
                # Draining while it exits: see `_reap_worker`. Without it one worker in
                # three never returns and this join blocks its whole grace.
                if not self._reap_worker(stream.stream_id, process, _out):
                    process.terminate()
                    process.join(timeout=STOP_GRACE_SECONDS)
        # **Never `join_thread()` a shutdown queue.** Measured here: with a
        # `join_thread()`, `stop()` on a supervisor whose worker had died **hung
        # indefinitely** - reproduced twice, the process killed at 90 s with the last
        # line printed being `join_thread`. The cause is `multiprocessing`'s own feeder
        # thread: it is still trying to flush a 6.2 MB frame into a pipe whose reader
        # (the child) is gone, and nothing drains it. See `_retire`.
        with self._lock:
            for _stream, _process, in_queue, out_queue in claimed:
                self._retire(in_queue, out_queue)
            # A final sweep over everything still in the ledger. It used to be every queue
            # this supervisor had ever created; `_retire` now drops an entry as soon as its
            # feeder is gone, so what is left is what has not been released - and every
            # one of those still gets `cancel_join_thread()`, which is the atexit
            # protection. A ledger that emptied during the run is not a skipped sweep.
            self._retire(*tuple(self._retired_queues))
        collector = self._collector
        if collector is not None:
            collector.stop()
        self._collector = None

    def _retire(self, *queues: Any) -> None:
        """Release queues: discard what they still hold, close them, forget them.

        **Every queue this supervisor has created and not yet released is registered in
        `_retired_queues`, and every one of them is cancelled here - including the ones an
        earlier generation created, and including the ones belonging to streams this call
        is not about.** That last clause is why the ledger exists and why the cancellation
        is of the *arguments* only: with three streams, cancelling "the previous
        generation's queues" as a single shared slot would have torn down a live Camera B
        because Camera C moved.

        ## The hang this method exists for, which is not the leak it was mistaken for

        The original hang was measured, not hypothesised: a respawn dropped the old pair
        on the floor without closing it, and `multiprocessing`'s own atexit finaliser then
        ran `_finalize_join` on a queue whose feeder thread was blocked forever in
        `Connection._send_bytes` trying to flush a 6.2 MB frame into a pipe whose reader
        (the dead worker) had gone. Reproduced: the test suite finished with `OK` and the
        interpreter then never exited; `faulthandler.dump_traceback_later` showed
        `multiprocessing/queues.py:219 _finalize_join -> Thread.join` in
        `_run_finalizers`, with two feeder threads parked in `connection.py:293
        _send_bytes`.

        `cancel_join_thread()` says "do not wait for that flush", and that fixed the hang.
        **It did not release anything**, and the two were conflated for two milestones. A
        `multiprocessing.Queue` whose feeder is blocked still holds, in `_buffer`, every
        object handed to `put` that the feeder has not sent - and because the queue was
        appended to this ledger and never removed, the queue, its buffer and the blocked
        feeder thread were all kept alive for the life of the process.

        **The ledger is the way in, not the thing that holds the memory, and that is
        measured.** `tools/leak_probe.py` empties `_transitions` and this list, collects, and
        re-reads RSS: on the unfixed code, after 50 transitions at the product's 2.0 s
        deadline, that released **nothing** - the remainder was **+154.1 MiB of a +153.7 MiB
        total**. Dropping the list cannot free the frames, because the feeder thread owns
        the buffer; only draining the queue can. That is why the fix is a drain and a
        bounded wait, and pruning is only the accounting that follows it.

        That is also why the leak **inverted with the deadline.** Same walk, same machine,
        fresh process: 30 s deadline **+8.2 MiB**, 0 of 50 transitions timed out; 2.0 s
        deadline **+153.7 MiB**, 25 of 50 timed out. A 2 s deadline kills a worker that is
        still compiling, so nothing drains the pipe, and a retired input queue keeps the
        frames offered while it was starting: measured **22-31 frames** per queue (2 s of a
        15 fps feed). A 30 s deadline lets the worker drain them and the same queue retires
        empty.

        ## What it does now, in the order that matters

        1. `cancel_join_thread()` - the atexit protection, unchanged and still first.
        2. **Discard what the queue still holds** (`_discard_pending`). Those frames are
           addressed to a worker that no longer exists; the docstring on `_signal_stop`
           already says abandoning them costs nothing and no counter changes, because a
           frame that has left the bounded queue has left the supervisor's accounting.
        3. `close()` - which appends the feeder's sentinel *behind* the remaining buffer.
        4. Wait **a bounded** `RETIRE_FEEDER_GRACE_SECONDS` for the feeder to see the
           sentinel and return. It is no longer blocked, because step 2 emptied the pipe.
        5. **Forget the queue - only if that thread is confirmed gone.** A closed queue
           whose feeder has exited has nothing left for `multiprocessing`'s atexit
           finaliser to join, so dropping the reference is safe; a queue whose feeder
           survives is **left in the ledger**, and `stop()`'s final sweep cancels it again.
           That is the whole of the pruning rule: released means the feeder is gone, not
           means we tried.

        Step 5 is why the ledger can be both the atexit registry and bounded. It is still
        read by `stop()`'s re-entry guard and by `stop()`'s final sweep; neither becomes
        unreachable, because a queue that has not been released stays in it.

        Every call site detaches the pair from its stream before calling this
        (`_apply_devices`, `_restart`, `stop`), and `stop()`'s sweep runs after every stream
        has been claimed, so a queue a worker is still reading is never closed here.
        """
        with self._lock:
            known = list(self._retired_queues)
            for candidate in queues:
                if candidate is None:
                    continue
                if all(candidate is not existing for existing in known):
                    known.append(candidate)
            self._retired_queues = known
        released: list[Any] = []
        for candidate in queues:
            if candidate is None:
                continue
            try:
                candidate.cancel_join_thread()
            except Exception:  # noqa: BLE001 - teardown
                pass
            self._discard_pending(candidate)
            try:
                candidate.close()
            except Exception:  # noqa: BLE001 - teardown
                pass
            if self._feeder_gone(candidate):
                released.append(candidate)
        if released:
            dropped = {id(queue_obj) for queue_obj in released}
            with self._lock:
                self._retired_queues = [queue_obj for queue_obj in self._retired_queues
                                        if id(queue_obj) not in dropped]

    @staticmethod
    def _discard_pending(retired: Any) -> int:
        """Throw away everything a retiring queue still holds, and count it.

        **The frames in a retired input queue are addressed to a worker that no longer
        exists**, so discarding them is the only correct disposition; they are not results
        and cannot be delivered. Reading them is what unblocks the queue's feeder thread:
        it is parked in `Connection._send_bytes` on a pipe the dead child was the reader
        of, and each read here is the read that never happened.

        ## `get(timeout=...)`, not `get_nowait()`, and this is measured

        `get_nowait()` is the obvious choice and it **does not work here**. On Windows a
        queue is a message-mode named pipe, and `get_nowait()` asks `PeekNamedPipe` whether
        any bytes are available. A frame whose write is still in flight reports **zero**
        bytes available, so `get_nowait()` answers "empty" while the feeder is parked
        mid-message - and the queue is then closed with the feeder still blocked, which is
        exactly the retention this method exists to remove. Measured on this machine: an
        early version used `get_nowait()`, and of eight retired queues **four kept a live
        blocked feeder with the sentinel still behind it**, each holding the frame it could
        not finish writing. The four that succeeded were the ones whose last write happened
        to have completed - a race, not a rule.

        `get(timeout=...)` polls the pipe handle instead (`Connection.poll(timeout)` falls
        back to `wait([self], timeout)`), which *is* signalled for a partial message, and
        its read completes the message. Measured: **6 of 6 trials** drained all eight frames
        and the feeder exited within 300 ms. The timeout is `RETIRE_POLL_SECONDS` so an idle
        queue costs two milliseconds rather than blocking.

        Returns the number of items discarded. Nothing here is counted against `drops`,
        `frames_invalidated` or `discarded_results`: those populations have definitions of
        their own and none of them is "a frame abandoned with the transport".
        """
        deadline = time.perf_counter() + RETIRE_DRAIN_SECONDS
        discarded = 0
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                return discarded
            try:
                retired.get(timeout=min(RETIRE_POLL_SECONDS, remaining))
            except queue.Empty:
                thread = getattr(retired, "_thread", None)
                buffered = (thread is not None and thread.is_alive()
                            and bool(getattr(retired, "_buffer", ())))
                if not buffered:
                    return discarded
                continue
            except (OSError, EOFError, ValueError):
                # Closed under us, or a pipe whose other end is gone. Nothing left to
                # read; the feeder's own error path deals with the rest.
                return discarded
            discarded += 1

    @staticmethod
    def _feeder_gone(retired: Any) -> bool:
        """Whether a closed queue's feeder thread has returned.

        `True` for a queue whose feeder never started (a results queue nobody put to from
        this process), which is the common case for the outbound half of a pair. Otherwise
        it gives the thread `RETIRE_FEEDER_GRACE_SECONDS` to see the sentinel `close()`
        appended, and answers with what it observed rather than what it hoped.
        """
        thread = getattr(retired, "_thread", None)
        if thread is None or thread is threading.current_thread():
            return True
        if thread.is_alive():
            thread.join(timeout=RETIRE_FEEDER_GRACE_SECONDS)
        return not thread.is_alive()

    def _reap_worker(self, stream_id: str, process: Any, out_queue: Any,
                 timeout_s: float = STOP_GRACE_SECONDS) -> bool:
        """Wait for a retiring worker to exit, **draining its output while it does**.

        Returns `True` if the process exited inside the deadline.

        ## Why the drain is not optional

        The sentinel in `_signal_stop` is only read by a worker that comes back round to
        `in_queue.get()`. A worker blocked in `out_queue.put(result)` never gets there,
        and it blocks there as soon as the parent's reader thread is gone and the pipe
        fills - a 240x320 `FramePacket` is 230 KB, so one unread result is enough.

        Measured, after the sentinel was added and before this: of three workers, **one**
        never exited, so `join(timeout=5.0)` blocked its full five seconds and `stop()`
        took **5.16 s**. With the drain below it takes **0.05 s**. `multiprocessing`'s own
        behaviour is on both sides of that: the child blocks writing to a pipe nobody
        reads, and the parent's `join` blocks waiting for a child that is blocked.

        ## What the drained items are, and why they are counted

        Everything still in flight belongs to the generation being retired, so it is
        **discarded and counted** through `_discard_superseded` rather than dropped
        silently and rather than delivered - delivering a retired generation's result is
        exactly the defect the transition protocol exists to prevent. The counter's
        documented reason is "superseded", and a shutdown discard is that too: the result
        was produced by a worker this call is retiring.
        """
        if process is None:
            return True
        deadline = time.perf_counter() + timeout_s
        while process.is_alive():
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                return False
            if out_queue is None:
                time.sleep(min(0.02, remaining))
                continue
            try:
                item = out_queue.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            except (OSError, EOFError, ValueError):
                # The queue was closed under us; the child is on its way out.
                break
            if item is not None:
                self._discard_superseded(stream_id, item)
        return not process.is_alive()

    def _discard_superseded(self, stream_id: str, item: Any) -> None:
        """Count one result from a retiring worker, and never deliver it.

        A named method rather than a call to `accept_result` with a fake generation: the
        reason is not "the generation did not match", it is "this result belongs to a
        worker being torn down", and the two should not be able to drift apart silently.
        """
        with self._lock:
            self._discarded_results += 1
            stream = self._streams.get(str(stream_id))
            if stream is not None:
                self._attribute_discard(stream)

    def _signal_stop(self, in_queue: Any) -> None:
        """Put the stop sentinel into a worker's input queue, **while it can still read it**.

        ## Why this exists, and why it is called *before* the join

        `run_worker` returns when it reads `None` from its input queue. That is the only
        graceful exit a worker has, and the only way to reach it is to write the sentinel
        before the process is torn down.

        Both teardown paths used to do the opposite, and both paid for it in measured
        seconds:

        - `stop()` called `_halt_threads` **first** and put the sentinel after it.
        - `_apply_devices` and `_restart` never put it at all; they joined for
          `join(timeout=1.0)` and then terminated the process.

        So the worker never saw the sentinel, never returned, and the join always ran out
        its full second before `terminate()`. That is the whole of the ~1 s per moved
        stream in the transition times, and it is why one stream cost ~1.1 s of `setup`
        and two cost ~2.15 s.

        The feeder makes it worse in the other direction, and this ordering fixes that
        too. A feeder blocked in `in_queue.put()` on a full pipe is unblocked only by the
        worker draining that pipe; `_halt_threads` joins it with a `STOP_GRACE_SECONDS`
        (5.0 s) grace and then reports it as surviving. Three streams of that is 15 s of
        `stop()`. Writing the sentinel first lets the worker drain and return, which
        empties the pipe, which unblocks the feeder.

        ## What is given up, and why it costs nothing

        Frames already handed to the worker's input queue are abandoned rather than
        analysed. That is **already** what happens: the generation is over, `_drain_queue`
        throws away everything still in this supervisor's bounded queue and counts it as
        `frames_invalidated`, and a frame that has left the bounded queue has left the
        supervisor's accounting entirely. Terminating mid-inference abandons exactly the
        same frames; the sentinel only makes the child exit on its own terms instead of
        being killed. No counter changes, and `_halt_threads` still reports any thread
        that survives its grace.

        Best-effort by design: `mp.Queue.put` can raise on a pipe whose reader has already
        gone, and a teardown that raised here would be worse than one that terminates.
        """
        if in_queue is None:
            return
        try:
            in_queue.put(None)
        except Exception:  # noqa: BLE001 - teardown; the process is terminated anyway
            pass

    def _halt_threads(self, stream_id: str) -> None:
        """Join one stream's feeder, reader and watchdog threads and clear the handles.

        Called before a respawn. The threads loop on `self._stopping`, which `stop()`
        sets, so a plain `stop()` is not the mechanism here - a restart must not look
        like a shutdown to the caller's `submit`.

        **The calling thread is never joined.** This method is reached from the watchdog,
        which is one of the three threads it would join, and `Thread.join()` on the
        current thread raises `RuntimeError("cannot join current thread")`. That exception
        propagated out of `_restart`, killed the watchdog, and left the route in
        `pending` for ever after a single failure - observed, not theorised: the
        watchdog's traceback is in the M1 fix round's output. The old thread is left to
        exit on its own; it has already returned or is about to.

        **The frames still queued are discarded by `_drain_queue`, not by closing the
        queue.** `BoundedFrameQueue.close()` is permanent, and a queue that refused every
        put after a device change would make `submit` return `False` for ever while
        looking like a healthy route that is simply full - which is exactly the
        silent-drop failure review focus #2 is about.
        """
        with self._lock:
            stream = self._streams[stream_id]
            threads = (stream.feeder, stream.reader, stream.watchdog)
            generation = stream.generation
            stream.feeder = stream.reader = stream.watchdog = None
        current = threading.current_thread()
        surviving: list[str] = []
        for name, thread in zip(("feeder", "reader", "watchdog"), threads):
            if thread is None or thread is current or not thread.is_alive():
                continue
            thread.join(timeout=STOP_GRACE_SECONDS)
            if thread.is_alive():
                surviving.append(name)
        if surviving:
            # **Say so, because a surviving thread overlaps its replacement.** The old
            # feeder drains the *same* `BoundedFrameQueue` the new one will drain, so a
            # feeder that does not exit can take a frame across the handover and write it
            # into the retired worker: a frame lost uncounted, which is exactly what the
            # per-stream accounting exists to make impossible. The join is bounded so
            # `stop()` cannot hang, and it is bounded so a `select_devices` cannot hang
            # either - so this is the price, and it has to be a stated reason rather than
            # a silent one.
            reason = (
                f"stream {stream_id!r} generation {generation}: the {', '.join(surviving)} "
                f"thread(s) did not exit within {STOP_GRACE_SECONDS} s of being asked to "
                f"stop and may still overlap this stream's replacement worker. A retiring "
                f"feeder sharing the bounded queue with its replacement can move a frame "
                f"into the retired transport, which would be lost uncounted. The frames "
                f"themselves are not at risk of corruption - `BoundedFrameQueue` is "
                f"thread-safe - but this stream's per-stream counts may no longer "
                f"account for every frame that entered its queue. Treat this stream's "
                f"counters as suspect until the next `select_devices`."
            )
            with self._lock:
                stream.errors.append(reason)
            self.error_raised.emit(reason)

    def _drain_queue(self, stream_id: str) -> None:
        """Throw away frames queued for a generation that is over.

        They are real pictures that were paid for, and they are being dropped because the
        routing decision that queued them has been superseded - the same reason
        `ClipSource.restart` invalidates every result in flight. They are not counted as
        drops: a drop is a frame nobody analysed because the analysis was falling behind,
        and conflating the two would make the lag counter lie. They are also not counted
        as refusals: they were accepted, and the caller was told `True`.

        **They are counted, as `frames_invalidated`** - see the return value and
        `StreamStats.frames_invalidated`. At most `queue_size` (2) per affected stream per
        toggle, which is ~133 ms of picture per stream at 15 fps.
        """
        queue_obj = self._streams[stream_id].queue
        invalidated = 0
        while queue_obj.get(timeout=0.0) is not None:
            invalidated += 1
        if invalidated:
            # **Recorded, because before this counter existed they were not.** These
            # frames were accepted at `submit` time and were told so, and the route
            # change is what threw them away. Counting them as drops would put a lie in
            # the lag counter; counting them as refusals would put a lie in the "the
            # pipeline was not there" counter; counting them as nothing left
            # `frames_accepted` permanently exceeding `frames_analysed` with no field
            # explaining the gap. Spec section 5's "outstanding requests finish or time
            # out" has an unrecorded half, and this is the other half.
            with self._lock:
                self._streams[stream_id].invalidated += invalidated
        return invalidated

    def _spawn(self, stream_id: str) -> None:
        """(Re)create one stream's queues, its process and its three threads.

        The reader thread is created **bound to this stream's current generation**, which
        is what makes "discard a superseded result" a comparison rather than a guess. See
        the module docstring, section 4.
        """
        context = mp.get_context("spawn")
        in_queue = context.Queue()
        out_queue = context.Queue()
        with self._lock:
            # **Registered while they are still live**, not when they are retired: a queue
            # that is live at `stop()` is reached by the final sweep through this list, and
            # a queue that is never registered would be handed to `multiprocessing`'s
            # atexit finaliser with its feeder thread attached. `_retire` drops the entry
            # once the feeder is gone; until then it belongs here.
            self._retired_queues.append(in_queue)
            self._retired_queues.append(out_queue)
            stream = self._streams[stream_id]
            generation = stream.generation
            target = _faulting_worker_entry if stream.compile_failure else _worker_entry
            config = stream.config
        process = context.Process(
            target=target,
            args=(config, in_queue, out_queue),
            name=f"inference-worker-{stream_id}",
            daemon=True,
        )
        process.start()
        with self._lock:
            self._streams[stream_id].process = process
            self._streams[stream_id].in_queue = in_queue
            self._streams[stream_id].out_queue = out_queue
            self._streams[stream_id].pid = int(process.pid or 0) or None
            self._streams[stream_id].state = "pending"
        self._rebuild_collector()
        feeder = threading.Thread(target=self._feed, args=(stream_id, generation),
                                  name=f"supervisor-feeder-{stream_id}", daemon=True)
        reader = threading.Thread(target=self._read, args=(stream_id, generation, out_queue),
                                  name=f"supervisor-reader-{stream_id}", daemon=True)
        watchdog = threading.Thread(target=self._watch, args=(stream_id, generation),
                                    name=f"supervisor-watchdog-{stream_id}", daemon=True)
        with self._lock:
            stream = self._streams[stream_id]
            stream.feeder, stream.reader, stream.watchdog = feeder, reader, watchdog
        for thread in (feeder, reader, watchdog):
            thread.start()
        self.state_changed.emit(self.route_plan())

    # -- the per-stream threads --------------------------------------------------------

    def _feed(self, stream_id: str, generation: int) -> None:
        """Drain the bounded queue into the child's queue, one frame at a time.

        Reusing the bounded queue's frames rather than reading its depth: it is FIFO,
        and the queue's own `get` is the accessor its tests pin.
        """
        while True:
            with self._lock:
                if self._stopping:
                    return
                stream = self._streams[stream_id]
                # A superseded feeder has no business writing into a queue the new
                # generation's feeder is about to create.
                if stream.generation != generation:
                    return
                in_queue = stream.in_queue
            if in_queue is None:
                return
            frame = stream.queue.get(timeout=READ_POLL_SECONDS)
            if frame is None:
                with self._lock:
                    stopping = self._stopping
                if stopping:
                    return
                continue
            try:
                in_queue.put(frame)
            except Exception:  # noqa: BLE001 - a dead pipe is the watchdog's business
                return

    def _read(self, stream_id: str, generation: int, out_queue: Any) -> None:
        """Take one stream's results off the child's queue and publish the current ones.

        **A reader whose queue has been detached returns.** It used to loop until
        `self._stopping`, which only a real `stop()` sets - so a reader being replaced by a
        respawn sat on the *retired* queue for ever, and `_halt_threads` waited the full
        `STOP_GRACE_SECONDS` for a thread that was never going to exit. That was not
        visible before the thread-handover check reported it: the join timed out, the
        timeout was swallowed, and the old reader and its replacement then both polled, one
        on a retired queue. The check is `out_queue is stream.out_queue`, which is false the
        moment `_apply_devices` detaches it - so the reader retires with its transport
        instead of outliving it.
        """
        while True:
            with self._lock:
                if self._stopping:
                    return
                stream = self._streams.get(stream_id)
                if stream is None or stream.out_queue is not out_queue:
                    # Superseded: this reader belongs to a transport the stream no longer
                    # owns. Leaving would keep a thread alive on a retired queue.
                    return
            try:
                item = out_queue.get(timeout=READ_POLL_SECONDS)
            except queue.Empty:
                # **Not** a failure: nothing arrived inside the poll window. Catching it
                # as `Exception` instead kills the reader thread on its first empty
                # poll, and a supervisor with a dead reader keeps reporting "pending"
                # for ever while a perfectly healthy worker sits behind it - a stall with
                # no error anywhere.
                continue
            except (OSError, EOFError, ValueError):
                # The queue was closed under us during shutdown.
                return
            if item is None:
                return
            if isinstance(item, InferenceResult):
                self.accept_result(stream_id, generation, item)

    def _watch(self, stream_id: str, generation: int) -> None:
        """Notice one worker that has exited. `multiprocessing` gives no usable callback.

        ## What this is responsible for, and what it is not

        **It is responsible for one thing: a child process that stops being alive becomes
        a stated reason.** It polls `is_alive()` every `EXIT_POLL_SECONDS` (0.05 s),
        works out *why* the process is gone - the injected compile-failure text plus which
        stream and which device it happened to, or the bare exit code - and hands it to
        `_on_failure`, which restarts the stream once and then fails it visibly.

        It does **not** wait for a result, and it must not: a worker that wedges without
        exiting is a different failure, and one that exits without producing anything is
        exactly the case this handles. `is_alive()` is the only signal that does not
        depend on the child cooperating.

        ## Why it returns instead of failing the stream

        Each of these returns is a case where this watchdog has nothing to say, and
        getting that wrong is how a healthy transition turns into a failed one:

        - `self._stopping` - `stop()` owns the outcome from here.
        - `stream.generation != generation` - **a superseded worker dying says nothing
          about the route that replaced it.** The replacement brought its own watchdog,
          bound to its own generation; two of them racing to fail one stream would be the
          bug, not the fix.
        - `process is None` - the stream has been detached (by `_apply_devices`,
          `_restart` or `stop`) and either has a new generation or is going away.

        ## The invariant this depends on, and where it was broken

        **A stream can only be watched if it was spawned.** There is no separate
        liveness check for "pending with no process", so a stream that reaches `pending`
        without a worker is invisible to every mechanism in this class: no child to die, no
        watchdog to notice, no restart, no failure. That is not hypothetical - it is
        exactly what happened (Task 3) while `_running` was also being used as a global
        "some stream is failed" flag, which suppressed the spawn in
        `_apply_devices`. `_running` no longer carries that meaning; see `_on_failure`.

        The reason is the asymmetry: **a false negative here is silent and permanent, and a
        false positive is a stated error an operator can act on.** Every loop above prefers
        to stay quiet rather than blame the wrong generation.
        """
        while True:
            time.sleep(EXIT_POLL_SECONDS)
            with self._lock:
                if self._stopping:
                    return
                stream = self._streams.get(stream_id)
                if stream is None or stream.generation != generation:
                    return
                process = stream.process
                compile_failure = stream.compile_failure
                device_id = stream.requested_device
            if process is None:
                return
            if process.is_alive():
                continue
            exitcode = process.exitcode
            reason = (
                f"the inference worker process for stream {stream_id!r} exited with code "
                f"{exitcode} on device {device_id!r}"
            )
            if compile_failure and exitcode is not None and exitcode != 0:
                # The injected text, **plus which stream and which device it happened
                # to**. M1's version was the injected reason alone, which was complete
                # with one stream and is unattributable with three: three identical
                # error lines, and no way to tell which camera died. Found by this
                # milestone's own test asserting the reason names its stream.
                reason = (f"{COMPILE_FAILURE_REASON} (worker exit code {exitcode}; the "
                          f"worker for stream {stream_id!r} on device {device_id!r})")
            self._on_failure(stream_id, generation, reason)
            return

    # -- routing -----------------------------------------------------------------------

    def select_devices(self, mapping: MappingABC[str, str]) -> RoutePlan:
        """Request new devices for the named streams: bump the generation, clear only
        their proof, restart only their workers. Returns the **new** plan.

        A stream is **affected** when its requested device actually changes. Asking for
        the device a stream is already on is not a transition: nothing has been
        superseded, so nothing is cleared and no worker is restarted. That is also what
        makes a coalesced toggle cheap - asking for the final state while an earlier one
        is in flight resolves to "no change" for the streams that already got there.

        For the affected streams: everything that belonged to the old generation goes -
        `confirmed` (a placement proved under the old routing is not a proof under the
        new one), `errors`, the restart budget, and `latest_result`. Streams not named
        keep their generation, their proof and their counters, which is why
        `RoutePlan.generation` is the highest across streams.

        Raises `KeyError` for a stream this supervisor does not own: routing a camera
        that does not exist would otherwise create one on demand.
        """
        plan, _moved = self._apply_devices(self._validate_mapping(mapping), force=False)
        return plan

    def _apply_devices(self, validated: dict[str, str], *,
                       force: bool = False) -> tuple[RoutePlan, tuple[str, ...]]:
        """`select_devices`'s body. Returns the plan and the ids that actually moved.

        Split out because `begin_transition` needs the moved set to record what a
        transition was for, and recomputing "which streams did this change" from the
        returned plan would be a guess.

        `force=True` treats every named stream as moved even when its device is
        unchanged. That is M1's `select_device`, whose whole meaning is "prove this route
        again" - an operator re-selecting the device that is already running is asking
        for a fresh compile and a fresh `EXECUTION_DEVICES` reading, not for a no-op.
        The multi-stream `select_devices` does not force, because the routing table is
        re-asserted on every toggle and a toggle that changes nothing must not cost three
        compiles.

        ## A **failed** stream is always moved, even onto the device it already names

        This is the third condition and it was missing, and the omission was reachable
        from the booth. Spec section 5 toggles restate the **whole** A/B/C mapping, so
        flipping the NPU switch while Camera C has failed and is already sitting on the
        CPU asks for `{"A": "CPU", "B": "CPU", "C": "CPU"}` - and Camera C's requested
        device does not change. Before this condition existed, that stream was not
        affected, no worker was spawned, the generation did not move, and the caller's
        only outcome was a transition over a stream with **no worker at all**. Observed:
        `begin_transition` returned generation 1, `bounded_by` was `'timeout'` with
        `elapsed_s` 5e-05 s, and Camera C stayed `failed` on the same pid for ever.

        Worse, the recovery path then published *"Recovering C - the last healthy route"*
        for a `select_devices` that was a no-op, and the failure guidance told the
        operator to *"re-select one to try again"* - which is precisely the action that
        did nothing. A failed stream has no worker and has spent its budget; naming it
        `affected` is what makes the operator's retry actually retry.
        """
        with self._lock:
            for stream_id in validated:
                self._require_stream(stream_id)
            affected = tuple(
                sid for sid, device in validated.items()
                if force
                or self._streams[sid].requested_device != device
                or self._streams[sid].state == "failed")
            if not affected:
                return self.route_plan(), ()
            # The generation moves first, and every affected stream is stamped with the
            # new one **inside the same lock hold**, so no reader thread can observe a
            # half-applied routing decision and compare a result against a generation
            # that is about to change under it.
            #
            # `superseded_by` is the *route* generation, and a discarded result is
            # attributed to `_transitions.get(superseded_by)`. With the id/generation
            # split above that lookup is now exact rather than approximate: a restatement
            # gets a fresh id, so a discard is attributed to the transition that
            # actually superseded the worker rather than to whatever now occupies that
            # generation slot.
            self._generation += 1
            claims: list[tuple[str, Any, Any, Any]] = []
            for stream_id in affected:
                device = validated[stream_id]
                stream = self._streams[stream_id]
                stream.requested_device = device
                stream.config = self._worker_config(device)
                stream.superseded_by = self._generation
                stream.generation = self._generation
                stream.generations_seen.append(self._generation)
                stream.clear_proof()
                stream.state = "pending"
                claims.append((stream_id, stream.process, stream.in_queue, stream.out_queue))
                stream.process = None
                stream.in_queue = None
                stream.out_queue = None
                stream.pid = None
            running = self._running and not self._stopping
        # **Publish the unproven state before the teardown, not after it.**
        #
        # Inside the lock above, every affected stream's `state` became `pending` and its
        # proof was cleared - so the supervisor already knows the old device is no longer
        # confirmed. But the teardown (terminate/join the old worker, retire its queues,
        # drain, spawn the replacement) then runs for **~1.1 s per moved stream** before
        # the `state_changed` at the end of this method. For that whole second the screen
        # still rendered the *previous* device as if it were current.
        #
        # Measured on the live path (Task 4): a presenter-driven move produced **10
        # consecutive 100 ms samples** in which the badge read `device: CPU` while the
        # supervisor's `confirmed_devices` was already empty. The queued hop cannot fix
        # that - the bridge was not *told* anything for the whole second. Emitting here
        # does, and it is also what spec section 5 asks for: the UI shows "Moving Camera B
        # inference to CPU..." with pending vs actual, which means pending has to be
        # published while the move is still in progress.
        self.state_changed.emit(self.route_plan())
        for _stream_id, _process, in_queue, _out_queue in claims:
            # **The stop sentinel goes in BEFORE the join**, which this code never did and
            # `stop()` did only after its own join. Measured: with it, one moved stream
            # costs ~1.1 s of `setup` because `join(timeout=1.0)` always runs out before
            # `terminate()`; the sentinel lets the worker return from `run_worker` and the
            # join finds it already gone. See `_signal_stop`.
            self._signal_stop(in_queue)
        for stream_id, process, _in_queue, _out_queue in claims:
            self._halt_threads(stream_id)
        for _stream_id, process, _in_queue, out_queue in claims:
            if process is None:
                continue
            # **Reaped, not joined-and-waited.** `join(timeout=1.0)` here always ran its
            # full second because the worker never saw the sentinel and never returned;
            # draining its output while it exits is what makes the join cheap. See
            # `_reap_worker` for the measurement.
            if not self._reap_worker(_stream_id, process, out_queue, timeout_s=1.0):
                process.terminate()
                process.join(timeout=1.0)
        for _stream_id, _process, in_queue, out_queue in claims:
            # The old transport is abandoned here, not at exit. See `_retire`: a queue
            # left un-cancelled hangs the interpreter in `multiprocessing`'s atexit
            # finaliser, and this is the path that used to leave one behind.
            self._retire(in_queue, out_queue)
        for stream_id, _process, _in_queue, _out_queue in claims:
            self._drain_queue(stream_id)
        if running:
            for stream_id, _process, _in_queue, _out_queue in claims:
                self._spawn(stream_id)
        plan = self.route_plan()
        self.state_changed.emit(plan)
        return plan, affected

    def _validate_mapping(self, mapping: MappingABC[str, str]) -> dict[str, str]:
        """`mapping` as `stream_id -> device_id`, checked before anything is changed."""
        if not isinstance(mapping, MappingABC):
            raise TypeError(
                f"a device mapping must be a mapping of stream_id -> device id; got "
                f"{type(mapping).__name__}. A bare device id routes one stream by "
                f"accident of ordering, which is not a decision anybody made."
            )
        out: dict[str, str] = {}
        for key, value in mapping.items():
            out[_require_text("stream id", str(key))] = _require_text("device id", value)
        return out

    def select_device(self, device_id: str) -> RoutePlan:
        """Re-select the first stream's device: bump the generation, clear its proof,
        restart its worker. M1's spelling, kept because M1's callers use it.

        Unlike `select_devices({"A": device_id})` this **always** acts, even when the
        stream is already on `device_id`. See `_apply_devices`'s `force`.
        """
        with self._lock:
            first = self._order[0]
        plan, _moved = self._apply_devices({first: _require_text("device_id", device_id)},
                                          force=True)
        return plan

    # -- the transition protocol ------------------------------------------------------

    def begin_transition(self, mapping: MappingABC[str, str]) -> int:
        """Open a route transition. Returns the **new** generation.

        The toggle. It asks for devices, it does not acknowledge anything: the streams
        named have their proof cleared and their workers respawned, and it returns the
        new generation number. The transition is acknowledged only by `await_transition`
        seeing a **confirmed placement** for every stream it moved.

        Coalescing: calling this again while an earlier transition is open supersedes it -
        the generation moves on, and the earlier one can never be acknowledged (see
        `await_transition`).
        """
        validated = self._validate_mapping(mapping)
        with self._lock:
            previous = {sid: self._require_stream(sid).requested_device
                        for sid in validated}
        plan, moved = self._apply_devices(validated)
        with self._lock:
            # A mapping that restates the device a stream is already on moves nothing.
            # It is still a transition over *those streams*: the caller asked for a
            # routing, and the acknowledgement it gets back is whether that routing has a
            # confirmed placement - which is a question about the streams it named, not
            # about how many of them had to be restarted.
            named = moved or tuple(sorted(validated))
            # **The returned id is monotonic and strictly greater than any generation
            # seen so far**, so it is unique per `begin_transition` call whether or not
            # the call moved anything. Keying `_transitions` by the *generation* did not
            # give that: two restatements inside one generation collided, the second
            # overwrote the first's `started`, and a late `await_transition` was handed a
            # fresh deadline for a transition that had been running for seconds. That is
            # the defect two tests now cover: the late-caller deadline, and two
            # restatements sharing one id. That second case only became reachable once a
            # restatement could exist at all.
            self._transition_seq = max(self._transition_seq, plan.generation) + 1
            transition = _Transition(
                id=self._transition_seq,
                generation=plan.generation,
                streams=named,
                requested={sid: validated[sid] for sid in named},
                previous={sid: previous[sid] for sid in named},
                started=float(self._clock()),
            )
            self._transitions[transition.id] = transition
        self.state_changed.emit(plan)
        return transition.id

    def await_transition(self, transition_id: int,
                         timeout_s: float = 2.0) -> TransitionOutcome:
        """Wait for the transition `transition_id` to be acknowledged, or for the deadline.

        The deadline is measured from when the transition **began**, so `elapsed_s` is the
        spec section 5 warm-transition figure and is unaffected by how late the caller got
        round to asking. `bounded_by == "confirmed"` means every stream the transition
        moved proved a placement under that generation; `bounded_by == "timeout"` means it
        did not, and the supervisor has **recovered every stream that did not confirm to
        the device it was running on before the toggle**, with the reason stated on
        `error_raised` and in `RoutePlan.errors`.

        A superseded transition returns at once with `bounded_by="timeout"`, because one
        that has been replaced can never be acknowledged and blocking on it would be
        waiting for a state the supervisor has left. That is what coalescing looks like
        from here: the last requested state is the one that runs.

        `transition_id` is what `begin_transition` returned - the monotonic id, **not** the
        route generation. They are the same number when the call moved something and they
        differ when it did not; `TransitionOutcome.generation` reports the route
        generation, so nothing is lost by the caller.
        """
        timeout = float(timeout_s)
        if timeout != timeout or timeout in (float("inf"), float("-inf")):
            raise ValueError(f"timeout_s must be finite; got {timeout_s!r}")
        if timeout < 0.0:
            raise ValueError(f"timeout_s must be >= 0.0; got {timeout_s!r}")
        if isinstance(transition_id, bool) or not isinstance(transition_id, int):
            raise TypeError(f"transition_id must be an int; got {transition_id!r}")
        with self._lock:
            transition = self._transitions.get(transition_id)
            if transition is None:
                raise ValueError(
                    f"no transition was opened with id {transition_id}. The highest id "
                    f"handed out is {self._transition_seq} and the highest route "
                    f"generation is "
                    f"{max(s.generation for s in self._streams.values())}. "
                    f"`begin_transition` is what opens one; awaiting an id that was never "
                    f"handed out would wait for nothing."
                )
            deadline = transition.started + timeout
        while True:
            with self._lock:
                now = float(self._clock())
                confirmed = {sid: tuple(self._streams[sid].confirmed)
                             for sid in transition.streams
                             if self._streams[sid].confirmed}
                failed = [sid for sid in transition.streams
                          if self._streams[sid].state == "failed"]
                superseded = self._generation > transition.generation
                done = len(confirmed) == len(transition.streams)
            if done:
                break
            if failed or superseded or now >= deadline:
                break
            time.sleep(TRANSITION_POLL_SECONDS)
        elapsed = max(0.0, float(self._clock()) - transition.started)
        with self._lock:
            confirmed = {sid: tuple(self._streams[sid].confirmed)
                         for sid in transition.streams if self._streams[sid].confirmed}
            bounded_by = "confirmed" if len(confirmed) == len(transition.streams) else "timeout"
            discarded = transition.discarded
        outcome = TransitionOutcome(
            generation=transition.generation,
            requested=dict(transition.requested),
            confirmed=confirmed,
            elapsed_s=elapsed,
            bounded_by=bounded_by,
            discarded_results=discarded,
        )
        if not outcome.acknowledged and self._is_current(transition):
            self._recover(transition, outcome)
        return outcome

    def _is_current(self, transition: _Transition) -> bool:
        """Whether `transition` is still the newest generation on this supervisor.

        A **superseded** transition must never recover. The presenter asked for something
        else while it was in flight, and recovering to "the last healthy route" for the
        request they have already replaced would undo the request they made - a toggle to
        the NPU, answered by a recovery back to the CPU because an earlier toggle to the
        GPU had not finished. Coalescing means the last request wins, and this is where
        that is enforced.
        """
        with self._lock:
            return self._generation <= transition.generation

    def _recover(self, transition: _Transition, outcome: TransitionOutcome) -> None:
        """Put every unconfirmed stream back on its last healthy device, and say so.

        Spec section 5: "on target failure the UI states the failure and the real fallback
        route and recovers to the last healthy route." The reason is published **after**
        `select_devices`, which clears the stream's errors - otherwise the recovery would
        erase the account of why the recovery happened.

        The recovery is itself a transition but it does not re-enter recovery, so a
        device that will not compile at all cannot produce an unbounded chain of
        recoveries; the caller may `await_transition` the new generation and see the
        result.

        ## **A recovery that cannot recover says so, and does not claim to have.**

        Some unconfirmed stream is already on the device the recovery wants to put it
        back on - which is the normal case for a stream that failed *before* the toggle,
        because `select_devices` already moved it there. `select_devices` would then be a
        no-op, and the text *"Recovering C - the last healthy route"* would be a claim
        about an action that did not happen, on the exact stream an operator is looking
        at. That was observed, and it is worse than a no-op: the message is the recovery
        report.

        So the targets are filtered to the streams that would actually move. If none
        would, what happens next depends on **why** the stream has no worker, and that is
        read from its state rather than assumed:

        - **`failed`** - it has spent its restart budget, and `_apply_devices` always
          treats a failed stream as affected. So naming it re-selects it onto the same
          device, which **does** spawn a worker and **does** give it a fresh budget. The
          recovery does that here rather than telling the operator to.
        - **`pending`, no worker** - its device has not changed and it is not failed, so
          naming it again is a no-op. The only remedy is a device *change*.

        The earlier text claimed re-selection was useless for both. That was false for
        `failed` (the cheap correct remedy, denied) and named `select_device` as the
        alternative, which uses `force=True` and so always moved anyway - a sentence that
        contradicted itself inside one clause.
        """
        with self._lock:
            if transition.recovered:
                return
            transition.recovered = True
            unconfirmed = outcome.unconfirmed_streams()
            requested = {sid: transition.requested[sid] for sid in unconfirmed}
            # Read the failing streams' own reasons **before** `select_devices` clears
            # them. Without this the recovery would erase the account of *why* the
            # recovery happened, and the operator would be left with "it went back to the
            # CPU" and no word about the device that refused.
            reasons = {sid: tuple(self._streams[sid].errors)
                       for sid in unconfirmed}
            moved = {sid: transition.previous[sid]
                     for sid in unconfirmed
                     if self._streams[sid].requested_device != transition.previous[sid]}
            stuck = {sid: self._streams[sid].requested_device
                     for sid in unconfirmed
                     if sid not in moved}
        if not unconfirmed:
            return
        quoted = (" | ".join(f"stream {sid}: {texts}"
                             for sid, texts in sorted(reasons.items()) if texts)
                  or "nothing; it never reported an error.")
        head = (
            f"route transition to generation {transition.generation} was not "
            f"acknowledged: stream(s) {list(unconfirmed)} did not confirm a placement on "
            f"{ {sid: requested[sid] for sid in sorted(requested)} } within "
            f"{outcome.elapsed_s:.3f} s (bounded_by={outcome.bounded_by!r}). "
            f"{outcome.discarded_results} result(s) from the superseded generation were "
            f"discarded. "
        )
        # Streams that are `failed` are **always** `affected` by `_apply_devices`, even onto
        # the device they already name, so naming one re-selects it and gives it a worker
        # back. Naming a `pending` route with no worker does not - its device has not
        # changed - and that is the only case where a different device is the remedy.
        # So one `select_devices` call handles both, and the published text can say which
        # is which because the states are read here, not guessed.
        retry_failed = {sid: stuck[sid] for sid in stuck
                        if self._streams[sid].state == "failed"}
        needs_device_change = sorted(sid for sid in stuck
                                     if sid not in retry_failed)
        if moved:
            reason = (
                head
                + f"Recovering { {sid: moved[sid] for sid in sorted(moved)} } - "
                f"the last healthy route - and stating the failure rather than claiming "
                f"the requested device. "
                + (f"Stream(s) {sorted(stuck)} were already on the device the recovery "
                   f"wanted, so nothing moved for them: "
                   + (f"{sorted(retry_failed)} are failed with no worker, so they were "
                      f"re-selected on the same device to give them a fresh worker and a "
                      f"fresh restart budget. "
                      if retry_failed else "")
                   + (f"{needs_device_change} are pending with no worker, and a pending "
                      f"route is not re-selected onto the device it already names - those "
                      f"need a device change to spawn a worker. "
                      if needs_device_change else "")
                   if stuck else "")
                + f"What the failing route said: {quoted}"
            )
            self.select_devices(moved)
            if retry_failed:
                self.select_devices(retry_failed)
        elif retry_failed:
            # **Not a route recovery, but not nothing either.** The streams are already on
            # the device the recovery wanted, so there is no *route change* to make - and
            # saying "re-selecting the same device will not fix this" was **false**,
            # because a `failed` stream is always `affected`: naming it does spawn a worker.
            # So it is named here, the worker comes back, and the text says exactly that.
            reason = (
                head
                + f"Nothing to recover for stream(s) {sorted(retry_failed)}: they are "
                f"already on "
                f"{ {sid: stuck[sid] for sid in sorted(retry_failed)} }, the device the "
                f"recovery wanted, so there was no route change to make. They were "
                f"**failed** with no worker, so they have been re-selected on the same "
                f"device, which does spawn a worker and gives them a fresh restart budget. "
                + (f"Stream(s) {needs_device_change} are pending with no worker, and a "
                   f"pending route is not re-selected onto the device it already names - "
                   f"those need a device change to spawn a worker. "
                   if needs_device_change else "")
                + f"What the failing route said: {quoted}"
            )
            self.select_devices(retry_failed)
        else:
            # The one case where a different device really is the only remedy: a `pending`
            # route with no worker. Re-selecting it would be a no-op, because its device
            # has not changed and it is not failed.
            reason = (
                head
                + f"Nothing moved, and a device change is the only remedy: "
                f"stream(s) {needs_device_change} are already on "
                f"{ {sid: stuck[sid] for sid in needs_device_change} }, the device the "
                f"recovery wanted, and they are **pending** rather than failed, so naming "
                f"them again would be a no-op - a pending route is only re-selected when "
                f"its device changes. They have no worker, and the remedy is to select a "
                f"*different* device for the stream. What the failing route said: "
                f"{quoted}"
            )
        with self._lock:
            for sid in unconfirmed:
                self._streams[sid].errors.append(reason)
        self.error_raised.emit(reason)
        self.state_changed.emit(self.route_plan())

    def transition(self, transition_id: int) -> MappingABC[str, str] | None:
        """What transition `transition_id` asked for, or `None` if there is no such id."""
        with self._lock:
            found = self._transitions.get(transition_id)
            return None if found is None else MappingProxyType(dict(found.requested))

    def latest_transition(self) -> TransitionOutcome | None:
        """The newest transition as a fresh outcome, or `None`.

        Convenience for a caller that only ever has the latest toggle to hand. It never
        blocks and never recovers - `await_transition` is the call that does.
        """
        with self._lock:
            if not self._transitions:
                return None
            transition = self._transitions[max(self._transitions)]
            confirmed = {sid: tuple(self._streams[sid].confirmed)
                         for sid in transition.streams if self._streams[sid].confirmed}
            return TransitionOutcome(
                generation=transition.generation,
                requested=dict(transition.requested),
                confirmed=confirmed,
                elapsed_s=max(0.0, float(self._clock()) - transition.started),
                bounded_by=("confirmed"
                            if len(confirmed) == len(transition.streams) else "timeout"),
                discarded_results=transition.discarded,
            )

    # -- the two state transitions ----------------------------------------------------

    def _apply_result(self, stream: _Stream, generation: int,
                      result: InferenceResult) -> bool:
        """Apply a result that already passed the generation check, and publish it.

        **This is the only path that can mark a route `active`, and therefore the only
        path that can acknowledge a transition.** A stream proves its placement when an
        inference on its *current* generation comes back saying where it ran; nothing that
        merely *asked* for a device can do it.

        **The check and the application are in one critical section, and this method must
        not be called without `self._lock` held.** `accept_result` is its only caller, it
        calls it from inside its own `with self._lock:` on an `RLock`, and the lock is
        never dropped between the two. That is why there is no second generation check
        here: this method used to have one, on the belief that the lock was dropped
        between the check and the apply and a toggle could land there. It never is - the
        lock is an `RLock` and the call is within its hold - so the second check was one
        layer expressed twice.

        That is worse than duplication. A duplicated check invites a refactor that moves
        the call outside the lock, and then the refactor looks correct (both checks still
        present) while the window it reopens is the real defect this module exists to
        prevent: a result admitted just before a transition and applied just after it
        would confirm the placement of a route already torn down. The protection is the
        **critical section**, not a second comparison inside it.

        Reads `stream.generation` **here**, after the caller's lock hold ends, rather than from
        the argument, because that is the value `_on_failure` must be handed: it is the
        generation this result is being applied to, and a failure must fail *that*
        generation or not at all.
        """
        if True:  # the caller's lock hold; see the docstring above
            stream.latest = result
            if result.error is None:
                stream.analysed += 1
                stream.frame_ages_ms.append(result.frame.age_at(self._clock()) * 1000.0)
                if result.execution_devices:
                    stream.confirmed = tuple(result.execution_devices)
                stream.state = "active"
            stream.failure_published = False
            device_id = stream.requested_device
            generation = stream.generation
        if result.error is not None:
            self._on_failure(stream.stream_id, generation,
                             f"inference failed on device {device_id!r}: {result.error}",
                             result=result)
            return True
        self.result_ready.emit(result)
        self.state_changed.emit(self.route_plan())
        return True

    def _on_failure(self, stream_id: str, generation: int, reason: str,
                    *, result: InferenceResult | None = None) -> None:
        """One stream failed. Restart that stream once, then fail that stream visibly."""
        with self._lock:
            stream = self._streams.get(str(stream_id))
            if stream is None or stream.generation != int(generation):
                # A superseded worker dying says nothing about the route that replaced
                # it. Failing the new generation for an old worker's exit is how a
                # successful transition turns into a failed one.
                return
            stream.errors.append(reason)
            if result is not None:
                stream.latest = result
            restart = stream.restarts == 0
            if restart:
                stream.restarts += 1
            device_id = stream.requested_device
        self.error_raised.emit(reason)
        if restart:
            self._restart(stream_id, generation, reason)
            return
        with self._lock:
            stream = self._streams.get(str(stream_id))
            if stream is None or stream.generation != int(generation):
                return
            stream.state = "failed"
            # The load-bearing part: a failed route must not keep claiming a device.
            stream.confirmed = ()
            # **Nothing else changes here, and that is the fix.** This used to read
            # `self._running = self._running and any(s.state != "failed" ...)`, so once
            # *every* stream had failed, `_running` went `False` - and `_running` is what
            # gates the spawn: `_apply_devices` computes
            # `running = self._running and not self._stopping` and skips `_spawn`
            # altogether, and `_restart` returns early on the same flag. The consequence
            # was measured (Task 3): after all three cameras had really died, the
            # operator's retry left all three `pending` with **no process, no watchdog and
            # `restart_count` 0**, for ever. Nothing could fail them and nothing could
            # bring them back, and the recovery's published remedy - "re-selected on the
            # same device, which does spawn a worker" - was false in exactly that state.
            #
            # A stream's health is `stream.state`, which `RoutePlan` and `StreamStats`
            # already publish per stream. `_running` means what it is documented to mean
            # and only that: *never started, or already stopped*. Conflating the two made
            # a per-stream failure a global stop.
        guidance = (
            f"{reason}. The restart budget for stream {stream_id!r} at generation "
            f"{generation} is spent, so this route is failed and no device is being "
            f"claimed for it. The requested device {device_id!r} is recorded; "
            f"re-select one to try again. The other streams are unaffected."
        )
        # **Recorded on the stream, not only emitted.** It used to be only emitted, and it
        # is guidance an operator acts on: `errors()` is what survives into
        # `RoutePlan.errors` and what a card reads, and a signal that nobody is connected
        # to takes the instruction with it. A stated reason that only exists on a signal
        # is a reason nobody will see.
        with self._lock:
            stream = self._streams.get(str(stream_id))
            if stream is not None and guidance not in stream.errors:
                stream.errors.append(guidance)
        self.error_raised.emit(guidance)
        self.state_changed.emit(self.route_plan())

    def _restart(self, stream_id: str, generation: int, reason: str) -> None:
        """Bring a fresh worker up for **this stream**, at the same generation, once.

        A restart is skipped when the supervisor is not running or is shutting down:
        `accept_result` can carry a failure on a supervisor nobody started, and
        respawning a worker there would bring up a process the caller never asked for -
        and, in a test, a real compile on a device nothing was measuring.
        """
        with self._lock:
            if self._stopping or not self._running:
                return
            stream = self._streams.get(str(stream_id))
            if stream is None or stream.generation != int(generation):
                return
            process = stream.process
            in_queue = stream.in_queue
            out_queue = stream.out_queue
            stream.process = None
            stream.in_queue = None
            stream.out_queue = None
            stream.pid = None
        self._signal_stop(in_queue)
        self._halt_threads(stream_id)
        if process is not None:
            # Reaped, not joined: see `_reap_worker`. A restart used to wait out a full
            # `join(timeout=1.0)` on every dead worker, because the sentinel was never
            # sent and the child never returned.
            if not self._reap_worker(stream_id, process, out_queue, timeout_s=1.0):
                process.terminate()
                process.join(timeout=1.0)
        # The old transport is abandoned here, not at exit. See `_retire`: a queue left
        # un-cancelled hangs the interpreter in `multiprocessing`'s atexit finaliser,
        # and this is the path that used to leave one behind.
        self._retire(in_queue, out_queue)
        self._drain_queue(stream_id)
        with self._lock:
            if self._stopping:
                return
            self._streams[stream_id].state = "pending"
        self.state_changed.emit(self.route_plan())
        self._spawn(stream_id)

    # -- telemetry --------------------------------------------------------------------

    @property
    def last_samples(self) -> tuple[MetricSample, ...]:
        """The most recent sample set this supervisor took, or `()`.

        Kept so a run summary can be built **after** `stop()` has torn the collector
        down - which is the only safe order, because stopping the collector is what
        guarantees no telemetry thread is still reading. Taking a fresh sample after
        `stop()` would rebuild a collector, start a thread, and immediately leave it
        running.
        """
        with self._lock:
            return tuple(self._last_samples)

    def samples(self) -> tuple[MetricSample, ...]:
        """One telemetry sample set, from `MetricCollector.sample()`.

        **Not the accessors.** `system_cpu_percent()` and `app_cpu_percent()` share one
        frozen `_Tick` with `sample()`; calling one between samples takes the next
        reading at that moment, and the `sample()` that follows reports the shorter
        window it created - or `UNAVAILABLE` with `window_seconds = 0.0` when it left no
        measurable time at all. A card polling an accessor on its own timer can flip the
        booth's own sample to unavailable with no other visible cause.
        """
        collector = self._collector
        if collector is None:
            collector = self._rebuild_collector()
        result = collector.sample()
        with self._lock:
            self._last_samples = result
        return result

    def _rebuild_collector(self) -> MetricCollector:
        """A collector over this process and every live worker pid.

        Rebuilt on every spawn because a respawned worker has a new pid, and a stale pid
        would be reported as an unreadable process - `app_cpu_percent` would be `None`
        for ever, which is honest about a dead worker and useless about a live one.

        The worker label is exactly `"worker"` for one stream, so M1's
        `("ui", "worker")` process scope is unchanged, and `worker_A` / `worker_B` /
        `worker_C` for three, so a sample says which camera's process it is about.
        """
        pids = [os.getpid()]
        labels = ["ui"]
        with self._lock:
            multi = len(self._order) > 1
            for sid in self._order:
                pid = self._streams[sid].pid
                if pid is None:
                    continue
                pids.append(int(pid))
                labels.append(f"worker_{sid}" if multi else "worker")
        previous = self._collector
        if previous is not None:
            previous.stop()
        collector = MetricCollector(
            pids, interval=self._collector_interval, process_labels=labels,
            background=True,
        )
        self._collector = collector
        return collector

    def __enter__(self) -> "Supervisor":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        with self._lock:
            stats = [f"{s.stream_id}:{s.requested_device}->"
                     f"{list(s.confirmed) or 'unproven'}"
                     for s in (self._streams[sid] for sid in self._order)]
            return (f"Supervisor(streams={stats!r}, state={self._aggregate_state()!r}, "
                    f"generation={max(s.generation for s in self._streams.values())}, "
                    f"restarts={self.restarts})")