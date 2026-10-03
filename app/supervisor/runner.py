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

The watchdog polls `Process.is_alive()` because `multiprocessing` offers no exit callback
on Windows that survives a queue that nobody is reading; `EXIT_POLL_SECONDS` is 0.05 s
because the deadline that matters is a human watching a booth, not a benchmark.

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

#: How often `await_transition` re-reads the streams while it waits. 10 ms, because a
#: `Condition.wait` on this platform has a measured ~15 ms floor (see
#: `BoundedFrameQueue.get`), so a finer poll would cost the same wall-clock time and
#: report a coarser deadline. Polling on `time.sleep` is honest here: the loop reads
#: immutable state under the supervisor's own lock and sleeps between reads.
TRANSITION_POLL_SECONDS: Final[float] = 0.01

#: The stream ids used when a caller asks for several devices and names no ids.
DEFAULT_STREAM_IDS: Final[tuple[str, ...]] = ("A", "B", "C")

#: Every reason `submit` can refuse, and what each one means to the operator. A
#: refusal with no reason is the failure review focus #4 is about, so the vocabulary is
#: named here once and pinned by a test.
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


class _Stream:
    """Everything one stream owns: its queue, its worker, its counters, its proof.

    Mutable and guarded by `Supervisor._lock`; never handed out. `StreamStats` is the
    immutable snapshot of it that a consumer sees.
    """

    __slots__ = (
        "stream_id", "requested_device", "config", "queue", "compile_failure",
        "generation", "generations_seen", "confirmed", "state",
        "errors", "restarts", "attempts", "accepted", "refused", "refusal_reasons",
        "analysed", "frame_ages_ms", "latest", "failure_published",
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
        self.confirmed: tuple[str, ...] = ()
        self.state = "pending"
        self.errors: list[str] = []
        self.restarts = 0
        self.attempts = 0
        self.accepted = 0
        self.refused = 0
        self.refusal_reasons: dict[str, int] = {}
        self.analysed = 0
        self.frame_ages_ms: list[float] = []
        self.latest: InferenceResult | None = None
        self.failure_published = False
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
        )

    def clear_proof(self) -> None:
        """Everything that belonged to the previous routing decision.

        A placement proved under the old routing is not a proof under the new one, and a
        plan that kept `GPU.0` in `confirmed` while the new generation is still `pending`
        would be claiming a placement for a route that has not run.
        """
        self.confirmed = ()
        self.errors = []
        self.latest = None
        self.frame_ages_ms = []
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

        #: Every queue created and not yet retired, across generations and streams. See
        #: `_retire` - this exists because a queue dropped without `cancel_join_thread()`
        #: hangs the interpreter at exit.
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
        accepted = stream.queue.put(frame)
        if accepted:
            with self._lock:
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

    def frame_ages_ms(self) -> tuple[float, ...]:
        """Measured ingress-to-result age per analysed frame, in milliseconds."""
        with self._lock:
            out: list[float] = []
            for sid in self._order:
                out.extend(self._streams[sid].frame_ages_ms)
            return tuple(out)

    # -- generation-scoped result acceptance -------------------------------------------

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
        for sid in self._order:
            self._halt_threads(sid)
        for stream, process, in_queue, _out in claimed:
            if in_queue is not None:
                # The stop sentinel, so a healthy worker returns from `run_worker` rather
                # than being killed. `put` only appends to the queue's buffer - the feeder
                # thread does the writing - so this does not block on the dead-pipe case.
                try:
                    in_queue.put(None)
                except Exception:  # noqa: BLE001 - shutdown, not a reportable failure
                    pass
            if process is not None:
                process.join(timeout=STOP_GRACE_SECONDS)
                if process.is_alive():
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
            # A final sweep over everything this supervisor ever created and never
            # retired. Without it a queue that was still live at `stop()` - which is
            # every stream that was running - would be handed to `multiprocessing`'s
            # atexit finaliser with its feeder thread still attached.
            self._retire(*tuple(self._retired_queues))
        collector = self._collector
        if collector is not None:
            collector.stop()
        self._collector = None

    def _retire(self, *queues: Any) -> None:
        """Abandon queues without waiting for their feeder threads.

        **Every queue this supervisor has ever created is registered, and every one of
        them is cancelled here - including the ones an earlier generation created, and
        including the ones belonging to streams this call is not about.** That last
        clause is why the ledger is append-only and the cancellation is of the *arguments*
        only: with three streams, cancelling "the previous generation's queues" as a
        single shared slot would have torn down a live Camera B because Camera C moved.

        The original hang this method exists for was measured, not hypothesised: a
        respawn dropped the old pair on the floor without closing it, and
        `multiprocessing`'s own atexit finaliser then ran `_finalize_join` on a queue
        whose feeder thread was blocked forever in `Connection._send_bytes` trying to
        flush a 6.2 MB frame into a pipe whose reader (the dead worker) had gone.
        Reproduced: the test suite finished with `OK` and the interpreter then never
        exited; `faulthandler.dump_traceback_later` showed
        `multiprocessing/queues.py:219 _finalize_join -> Thread.join` in
        `_run_finalizers`, with two feeder threads parked in `connection.py:293
        _send_bytes`.

        `cancel_join_thread()` says "do not wait for that flush", which is what retiring a
        transport means. Nothing is lost by it: the frames still in a retired queue are
        frames addressed to a worker that no longer exists.
        """
        with self._lock:
            known = list(self._retired_queues)
            for candidate in queues:
                if candidate is None:
                    continue
                if all(candidate is not existing for existing in known):
                    known.append(candidate)
            self._retired_queues = known
        for candidate in queues:
            if candidate is None:
                continue
            try:
                candidate.cancel_join_thread()
                candidate.close()
            except Exception:  # noqa: BLE001 - teardown
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
            stream.feeder = stream.reader = stream.watchdog = None
        current = threading.current_thread()
        for thread in threads:
            if thread is None or thread is current or not thread.is_alive():
                continue
            thread.join(timeout=STOP_GRACE_SECONDS)

    def _drain_queue(self, stream_id: str) -> None:
        """Throw away frames queued for a generation that is over.

        They are real pictures that were paid for, and they are being dropped because the
        routing decision that queued them has been superseded - the same reason
        `ClipSource.restart` invalidates every result in flight. They are not counted as
        drops: a drop is a frame nobody analysed because the analysis was falling behind,
        and conflating the two would make the lag counter lie. They are also not counted
        as refusals: they were accepted, and the caller was told `True`.
        """
        queue_obj = self._streams[stream_id].queue
        while queue_obj.get(timeout=0.0) is not None:
            continue

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
            self._retired_queues.append(in_queue)
            self._retired_queues.append(out_queue)
            stream = self._streams[stream_id]
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
        feeder = threading.Thread(target=self._feed, args=(stream_id,),
                                  name=f"supervisor-feeder-{stream_id}", daemon=True)
        reader = threading.Thread(target=self._read, args=(stream_id, out_queue),
                                  name=f"supervisor-reader-{stream_id}", daemon=True)
        watchdog = threading.Thread(target=self._watch, args=(stream_id,),
                                    name=f"supervisor-watchdog-{stream_id}", daemon=True)
        with self._lock:
            stream = self._streams[stream_id]
            stream.feeder, stream.reader, stream.watchdog = feeder, reader, watchdog
        for thread in (feeder, reader, watchdog):
            thread.start()
        self.state_changed.emit(self.route_plan())

    # -- the per-stream threads --------------------------------------------------------

    def _feed(self, stream_id: str) -> None:
        """Drain the bounded queue into the child's queue, one frame at a time.

        Reusing the bounded queue's frames rather than reading its depth: it is FIFO,
        and the queue's own `get` is the accessor its tests pin.
        """
        while True:
            with self._lock:
                if self._stopping:
                    return
                stream = self._streams[stream_id]
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

    def _read(self, stream_id: str, out_queue: Any) -> None:
        """Take one stream's results off the child's queue and publish the current ones."""
        while True:
            with self._lock:
                if self._stopping:
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
                self._on_result(stream_id, item)

    def _watch(self, stream_id: str) -> None:
        """Notice one worker that has exited. `multiprocessing` gives no usable callback."""
        while True:
            time.sleep(EXIT_POLL_SECONDS)
            with self._lock:
                if self._stopping:
                    return
                stream = self._streams.get(stream_id)
                if stream is None:
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
            self._on_failure(stream_id, reason)
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
        The multi-stream `select_devices` does not do that, because the routing table is
        re-asserted on every toggle and a toggle that changes nothing must not cost three
        compiles.
        """
        with self._lock:
            for stream_id in validated:
                self._require_stream(stream_id)
            affected = tuple(sid for sid, device in validated.items()
                             if force or self._streams[sid].requested_device != device)
            if not affected:
                return self.route_plan(), ()
            # The generation moves first, and every affected stream is stamped with the
            # new one **inside the same lock hold**, so no reader thread can observe a
            # half-applied routing decision and compare a result against a generation
            # that is about to change under it.
            self._generation += 1
            claims: list[tuple[str, Any, Any, Any]] = []
            for stream_id in affected:
                device = validated[stream_id]
                stream = self._streams[stream_id]
                stream.requested_device = device
                stream.config = self._worker_config(device)
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
        for stream_id, process, _in_queue, _out_queue in claims:
            self._halt_threads(stream_id)
        for _stream_id, process, _in_queue, _out_queue in claims:
            if process is None:
                continue
            process.join(timeout=1.0)
            if process.is_alive():
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

    # -- the two state transitions ----------------------------------------------------

    def _on_result(self, stream_id: str, result: InferenceResult) -> bool:
        """A result that passed the generation check. Updates the stream and publishes.

        **This is the only path that can mark a route `active`, and therefore the only
        path that can acknowledge a transition.** A stream proves its placement when an
        inference on its *current* generation comes back saying where it ran; nothing that
        merely *asked* for a device can do it.

        The generation is re-checked inside this lock rather than trusted from
        `accept_result`. Between the two there is a window - the result was validated, the
        lock was dropped, a toggle arrived - and a result admitted just before a
        transition and applied just after it would confirm the placement of a route that
        has already been torn down. That is the exact defect this module exists to
        prevent, arriving through a lock boundary instead of through a stale reader.
        """
        with self._lock:
            stream = self._streams.get(str(stream_id))
            if stream is None:
                return False
            stream.latest = result
            if result.error is None:
                stream.analysed += 1
                stream.frame_ages_ms.append(result.frame.age_at(self._clock()) * 1000.0)
                if result.execution_devices:
                    stream.confirmed = tuple(result.execution_devices)
                stream.state = "active"
            stream.failure_published = False
            device_id = stream.requested_device
        if result.error is not None:
            self._on_failure(stream.stream_id,
                             f"inference failed on device {device_id!r}: {result.error}",
                             result=result)
            return True
        self.result_ready.emit(result)
        self.state_changed.emit(self.route_plan())
        return True

    def _on_failure(self, stream_id: str, reason: str,
                    *, result: InferenceResult | None = None) -> None:
        """One stream failed. Restart that stream once, then fail that stream visibly."""
        with self._lock:
            stream = self._streams.get(str(stream_id))
            if stream is None:
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
            self._restart(stream_id, reason)
            return
        with self._lock:
            stream = self._streams.get(str(stream_id))
            if stream is None:
                return
            stream.state = "failed"
            # The load-bearing part: a failed route must not keep claiming a device.
            stream.confirmed = ()
            self._running = self._running and any(
                s.state != "failed" for s in self._streams.values())
        self.error_raised.emit(
            f"{reason}. The restart budget for stream {stream_id!r} is spent, so "
            f"this route is failed and no device is being "
            f"claimed for it. The requested device {device_id!r} is recorded; "
            f"re-select one to try again. The other streams are unaffected."
        )
        self.state_changed.emit(self.route_plan())

    def _restart(self, stream_id: str, reason: str) -> None:
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
            if stream is None:
                return
            process = stream.process
            in_queue = stream.in_queue
            out_queue = stream.out_queue
            stream.process = None
            stream.in_queue = None
            stream.out_queue = None
            stream.pid = None
        self._halt_threads(stream_id)
        if process is not None:
            process.join(timeout=1.0)
            if process.is_alive():
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