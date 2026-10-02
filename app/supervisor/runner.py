"""`app.supervisor.runner` - the process owner, the route, and the one-time restart.

Three decisions here are the difference between a booth screen that is truthful and one
that is not, and each is enforced structurally rather than by convention.

## 1. `pending` until an inference has proved a placement

A fresh supervisor knows what it was *asked* for. It does not know where anything ran.
So `route_plan().state` is `"pending"` from construction, and only a result carrying
`EXECUTION_DEVICES` read **after** its own inference - which is the only proof of
placement there is (AGENTS.md rule 5) - moves it to `"active"`. Setting `active` in
`start()` would publish a placement nobody observed, which is review focus #5 with a
different mechanism: not "the card shows the requested device", but "the card shows a
device that has not run yet".

`is_device(name)` therefore reads the **confirmed** map and never the request. So
`Supervisor(device_id="NPU").is_device("NPU")` is `False` until an NPU inference comes
back saying so. `RoutePlan.requested_device(stream)` exists for the separate question
"what was asked", and a caller that wants that has to ask for it by name.

## 2. One restart, then a visible failure

AGENTS.md rule 12: a worker that fails is restarted **once** automatically, and a
second failure is a failure somebody has to see. Two kinds count as a failure here -
the worker process **exiting**, and a result carrying an `error` - because on this
machine the second is how a device that will not compile announces itself:
`run_worker` turns a refused compile into an `InferenceResult` with `error` set and
carries on, so the process stays alive while doing nothing useful. Watching only for
`exitcode` would leave a permanently broken route sitting in `pending` with no error on
screen.

After the second failure the plan is `failed`, **`confirmed` is emptied**, and the
error text is published on `error_raised`. Emptying `confirmed` is the part that matters:
a route that has failed but still reports `confirmed={"A": ("NPU",)}` renders as a
working NPU, and that is precisely the claim the review exists to forbid. The requested
device is *kept*, because "NPU was asked for and could not run" is what an operator
needs to read.

`restart_count` is a property, not a log line, so a supervisor that restarted in a loop
would be visible in a test. `select_device` resets it: a new generation is a new
opportunity, and holding a spent budget against the next device would mean the second
device selected could not survive a single hiccup.

**This is deliberately unforgiving.** A route whose frames error twice is `failed`, and
the operator re-selects. A single transient decode error costs one restart and the
route recovers; two do not. The alternative - counting consecutive failures, or
resetting the budget on the first success - would mean the budget is not a budget, and
a device that fails intermittently would never reach a visible failure at all.

## 3. Frames cross the spawn boundary through a bounded, counted queue

`submit` goes through `app.engine.queue.BoundedFrameQueue` before anything is handed to
the child, so the drop counter and the newest-frame refusal are the ones Task 5 wrote
and tested. A feeder thread drains that queue into the `multiprocessing` queue; the
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

**One reader thread, one feeder thread, one watchdog.** The watchdog polls
`Process.is_alive()` because `multiprocessing` offers no exit callback on Windows that
survives a queue that nobody is reading; `EXIT_POLL_SECONDS` is 0.05 s because the
deadline that matters is a human watching a booth, not a benchmark.

## Telemetry

`samples()` returns `MetricCollector.sample()` and nothing else - **never the
collector's `system_cpu_percent()` / `app_cpu_percent()` accessors.** Those share one
frozen `_Tick` with `sample()`, and calling one between samples moves the tick earlier
for everybody, so a card polling an accessor on its own timer can flip the booth's own
sample to `UNAVAILABLE` with no other visible cause. The docstrings on those accessors
say so; this module is the in-tree consumer that honours it.

The collector tracks **this process and the worker**, labelled `ui` and `worker`, which
is what spec section 8 asks for. It is rebuilt when the worker is respawned, because
the new process has a new pid and a stale pid would report a dead worker's CPU as
absent forever.

## `compile_failure` is a test seam, and says so

`Supervisor(device_id="NPU", compile_failure=True)` makes the spawned child raise
before it compiles anything, and the child says exactly that in the text the supervisor
publishes. It is a fault injection, not a measurement of the NPU: it exercises the
supervisor's detect-a-dead-worker, restart-once, fail-visibly path without needing a
device that genuinely will not compile. A **real** compile refusal is exercised by
`tests/test_worker.py`, which asks for a device id this machine does not have and
requires an error rather than a run somewhere else.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Any, Final

from PySide6.QtCore import QObject, Signal

from app.contracts.frame import FramePacket
from app.contracts.metric_sample import MetricSample
from app.contracts.route_plan import RoutePlan
from app.engine.queue import BoundedFrameQueue
from app.engine.worker import InferenceResult, WorkerConfig, run_worker
from app.inference.devices import is_device
from app.telemetry.collector import MetricCollector

__all__ = ["COMPILE_FAILURE_REASON", "Supervisor"]

#: How often the watchdog asks the worker process whether it is still there. 50 ms
#: because the deadline that matters is a person watching a screen, not a benchmark.
EXIT_POLL_SECONDS: Final[float] = 0.05

#: How long a blocking queue read waits before looping, so the reader thread notices a
#: stop request promptly instead of sitting on `get()` until the queue is closed.
READ_POLL_SECONDS: Final[float] = 0.05

#: How long `stop()` waits for a worker to notice the `None` sentinel before it is
#: terminated. One CPU inference on this machine is 21 ms; 5 s is ~200 of them.
STOP_GRACE_SECONDS: Final[float] = 5.0

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


def _faulting_worker_entry(config: WorkerConfig, in_queue: Any, out_queue: Any) -> None:
    """The spawn entry point used when `compile_failure=True`.

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


class Supervisor(QObject):
    """Owns one inference worker process, its route, and its telemetry.

    `device_id` is the **requested** device and is recorded as such. Nothing in this
    class ever promotes it to a confirmed placement; only `InferenceResult.
    execution_devices` can, and only after its own inference has run.
    """

    #: An `InferenceResult` for a frame that has been analysed (or has failed). Emitted
    #: from the reader thread; connect with a queued connection, per AGENTS.md rule 11.
    result_ready = Signal(object)
    #: The new `RoutePlan`, on every state change. Emitted from whichever thread
    #: observed the change.
    state_changed = Signal(object)
    #: One stated reason, as text. Never a formatted exception object.
    error_raised = Signal(str)

    def __init__(
        self,
        device_id: str = "CPU",
        *,
        stream_id: str = "A",
        session_id: str = "supervisor",
        scenario_id: str = "retail_sanity",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        queue_size: int = 2,
        compile_failure: bool = False,
        collector_interval: float = 0.25,
        clock=time.perf_counter,
    ) -> None:
        super().__init__()
        if not isinstance(device_id, str) or not device_id.strip():
            raise ValueError(
                f"device_id must be a non-empty string; got {device_id!r}. The supervisor "
                f"records what it was asked to run on, and an empty request cannot be "
                f"distinguished from no request."
            )
        self._device_id = device_id
        self._stream_id = str(stream_id)
        self._compile_failure = bool(compile_failure)
        self._clock = clock
        self._queue = BoundedFrameQueue(maxsize=int(queue_size), clock=clock)

        self._lock = threading.RLock()
        self._generation = 1
        self._state = "pending"
        self._errors: list[str] = []
        self._confirmed: dict[str, tuple[str, ...]] = {}
        self._restarts = 0
        self._latest: InferenceResult | None = None
        self._frames_submitted = 0
        self._frames_analysed = 0
        self._frame_ages_ms: list[float] = []
        self._stopping = False
        self._running = False
        self._failure_published = False

        self._process: mp.process.BaseProcess | None = None
        self._in_queue: Any = None
        self._out_queue: Any = None
        #: Every queue created and not yet retired, across generations. See `_retire` -
        #: this exists because a queue dropped without `cancel_join_thread()` hangs the
        #: interpreter at exit.
        self._retired_queues: list[Any] = []
        self._worker_pid: int | None = None
        self._collector: MetricCollector | None = None
        #: The most recent `samples()` result, kept so a run summary can be built after
        #: `stop()` has torn the collector down. See `last_samples`.
        self._last_samples: tuple[MetricSample, ...] = ()

        self._config = WorkerConfig(
            device_id=device_id,
            model_id=self._model_id(),
            input_width=self._input_extent(3),
            input_height=self._input_extent(2),
            conf_threshold=float(conf_threshold),
            iou_threshold=float(iou_threshold),
        )
        self._reader: threading.Thread | None = None
        self._feeder: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None
        self._collector_interval = float(collector_interval)

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

        `requested` names this supervisor's declared stream, so it is populated from
        construction - what was asked is known immediately. `confirmed` is populated
        only by a result, and is **emptied** when the route fails.
        """
        with self._lock:
            return RoutePlan(
                generation=self._generation,
                requested={self._stream_id: self._device_id},
                confirmed=dict(self._confirmed),
                state=self._state,
                errors=tuple(self._errors),
            )

    @property
    def device_id(self) -> str:
        """The **requested** device. Not a placement claim - see `is_device`."""
        return self._device_id

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def restart_count(self) -> int:
        """How many times a failed worker has been restarted. At most 1 per generation."""
        with self._lock:
            return self._restarts

    def errors(self) -> tuple[str, ...]:
        """Every stated reason this generation has published, oldest first.

        A **method**, not a property, per the interface the brief specifies
        (`self.assertTrue(s.errors())`). A property of the same name would make
        `s.errors` a tuple - truthy, and silently passing an assertion that meant
        "is there a stated reason", which is the exact failure mode this repository
        keeps finding in its own test suite.
        """
        with self._lock:
            return tuple(self._errors)

    def is_device(self, device_id: str) -> bool:
        """Whether a placement has been **confirmed** on `device_id`.

        Reads the confirmed map only. `Supervisor(device_id="NPU").is_device("NPU")` is
        `False` until an NPU inference reports `EXECUTION_DEVICES`, and stays `False`
        after a failure. Compare through `app.inference.devices.is_device`, because the
        readings are inconsistent on this machine: CPU `['CPU']`, GPU `['GPU.0']`, NPU
        the bare string `'NPU'`.
        """
        with self._lock:
            confirmed = tuple(self._confirmed.get(self._stream_id, ()))
            state = self._state
        if state == "failed":
            return False
        return is_device(confirmed, device_id)

    def select_device(self, device_id: str) -> RoutePlan:
        """Request a new device: bump the generation, clear the proof, restart the worker.

        Returns the **new** plan so a caller can compare generations without a second
        call. Everything that belonged to the old generation goes: `confirmed` (a
        placement proved under the old routing is not a proof under the new one),
        `errors`, the restart budget, and `latest_result`.
        """
        if not isinstance(device_id, str) or not device_id.strip():
            raise ValueError(f"device_id must be a non-empty string; got {device_id!r}")
        config = WorkerConfig(
            device_id=device_id,
            model_id=self._model_id(),
            input_width=self._input_extent(3),
            input_height=self._input_extent(2),
            conf_threshold=self._config.conf_threshold,
            iou_threshold=self._config.iou_threshold,
        )
        with self._lock:
            was_running = self._running and not self._stopping
            self._device_id = device_id
            self._config = config
            self._generation += 1
            self._confirmed = {}
            self._errors = []
            self._state = "pending"
            self._restarts = 0
            self._latest = None
            self._frame_ages_ms = []
            self._failure_published = False
            process = self._process
            in_queue = self._in_queue
            out_queue = self._out_queue
            self._process = None
            self._in_queue = None
            self._out_queue = None
            self._worker_pid = None
        self._halt_threads()
        if process is not None:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        self._retire(in_queue, out_queue)
        self._drain_queue()
        if was_running:
            self._spawn()
        plan = self.route_plan()
        self.state_changed.emit(plan)
        return plan

    # -- lifecycle --------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the worker. Idempotent; does not wait for a placement.

        Returns as soon as the process exists. `state` stays `pending` until a result
        proves a placement, so a slow compile shows as "moving..." on screen rather
        than as a device that has not run anything.
        """
        with self._lock:
            if self._running and not self._stopping:
                return
            self._stopping = False
            self._running = True
        self._spawn()

    def stop(self) -> None:
        """Stop the worker, its threads and the collector. Idempotent, and prompt.

        `submit` returns `False` afterwards without counting a drop: a frame arriving
        after the consumer has gone is a shutdown, not the analysis falling behind.
        """
        with self._lock:
            if self._stopping and not self._running and not self._retired_queues:
                return
            self._stopping = True
            self._running = False
            process = self._process
            in_queue = self._in_queue
            out_queue = self._out_queue
            self._process = None
            self._in_queue = None
            self._out_queue = None
            self._worker_pid = None
        self._queue.close()
        self._halt_threads()
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
        self._retire(in_queue, out_queue)
        collector = self._collector
        if collector is not None:
            collector.stop()
        self._collector = None

    def _retire(self, *queues: Any) -> None:
        """Abandon queues without waiting for their feeder threads.

        **Every queue this supervisor has ever created is registered, and every one of
        them is cancelled here - including the ones an earlier generation created.** That
        last clause was a measured hang, not a hypothetical. A respawn dropped the old
        pair on the floor without closing it, and `multiprocessing`'s own atexit
        finaliser then ran `_finalize_join` on a queue whose feeder thread was blocked
        forever in `Connection._send_bytes` trying to flush a 6.2 MB frame into a pipe
        whose reader (the dead worker) had gone. Reproduced: the test suite finished with
        `OK` and the interpreter then never exited; `faulthandler.dump_traceback_later`
        showed `multiprocessing/queues.py:219 _finalize_join -> Thread.join` in
        `_run_finalizers`, with two feeder threads parked in `connection.py:293
        _send_bytes`.

        `cancel_join_thread()` says "do not wait for that flush", which is what retiring a
        transport means. Nothing is lost by it: the frames still in a retired queue are
        frames addressed to a worker that no longer exists.
        """
        with self._lock:
            live = [q for q in self._retired_queues if q is not None]
            self._retired_queues = [q for q in queues if q is not None]
        for queue in live:
            try:
                queue.cancel_join_thread()
                queue.close()
            except Exception:  # noqa: BLE001 - teardown
                pass

    def _halt_threads(self) -> None:
        """Join the feeder, reader and watchdog threads and clear the handles.

        Called before a respawn. The threads loop on `self._stopping`, which `stop()`
        sets, so a plain `stop()` is not the mechanism here - a restart must not look
        like a shutdown to the caller's `submit`.

        **The calling thread is never joined.** This method is reached from the
        watchdog, which is one of the three threads it would join, and
        `Thread.join()` on the current thread raises `RuntimeError("cannot join current
        thread")`. That exception propagated out of `_restart`, killed the watchdog, and
        left the route in `pending` for ever after a single failure - observed, not
        theorised: the watchdog's traceback is in the fix round's output. The old
        thread is left to exit on its own; it has already returned or is about to.

        **The frames still queued are discarded by `_drain_queue`, not by closing the
        queue.** `BoundedFrameQueue.close()` is permanent, and a queue that refused
        every put after a device change would make `submit` return `False` for ever
        while looking like a healthy route that is simply full - which is exactly the
        silent-drop failure review focus #2 is about.
        """
        current = threading.current_thread()
        threads = (self._feeder, self._reader, self._watchdog)
        for thread in threads:
            if thread is None or thread is current or not thread.is_alive():
                continue
            thread.join(timeout=STOP_GRACE_SECONDS)
        self._feeder = self._reader = self._watchdog = None

    def _drain_queue(self) -> None:
        """Throw away frames queued for a generation that is over.

        They are real pictures that were paid for, and they are being dropped because
        the routing decision that queued them has been superseded - the same reason
        `ClipSource.restart` invalidates every result in flight. They are not counted
        as drops: a drop is a frame nobody analysed because the analysis was falling
        behind, and conflating the two would make the lag counter lie.
        """
        while self._queue.get(timeout=0.0) is not None:
            continue

    def _spawn(self) -> None:
        """(Re)create the queues, the process and the three threads."""
        context = mp.get_context("spawn")
        in_queue = context.Queue()
        out_queue = context.Queue()
        with self._lock:
            self._retired_queues.append(in_queue)
            self._retired_queues.append(out_queue)
        target = _faulting_worker_entry if self._compile_failure else _worker_entry
        process = context.Process(
            target=target,
            args=(self._config, in_queue, out_queue),
            name="inference-worker",
            daemon=True,
        )
        process.start()
        with self._lock:
            self._process = process
            self._in_queue = in_queue
            self._out_queue = out_queue
            self._worker_pid = int(process.pid or 0) or None
            self._state = "pending"
        self._rebuild_collector()
        self._feeder = threading.Thread(target=self._feed, name="supervisor-feeder",
                                        daemon=True)
        self._reader = threading.Thread(target=self._read, name="supervisor-reader",
                                        daemon=True)
        self._watchdog = threading.Thread(target=self._watch, name="supervisor-watchdog",
                                          daemon=True)
        for thread in (self._feeder, self._reader, self._watchdog):
            thread.start()
        self.state_changed.emit(self.route_plan())

    # -- submission -------------------------------------------------------------------

    def submit(self, frame: FramePacket) -> bool:
        """Offer a frame for analysis. `False` means it was refused.

        `False` for a closed or failed route is a **refusal, not a drop**, and does not
        move `drops`: a frame arriving after the consumer has gone is a shutdown, and
        mixing the two would make the drop counter unreadable exactly when somebody is
        diagnosing it. A full queue is a drop and does move it.

        Raises `TypeError` for anything that is not a `FramePacket` - a malformed put is
        a bug in the caller, and counting it as lag would put a wrong number on the lag
        counter.
        """
        if not isinstance(frame, FramePacket):
            raise TypeError(
                f"submit takes a FramePacket; got {type(frame).__name__}. A bare list of "
                f"detections has no frame identity, so a result carrying one could not be "
                f"matched to the picture it came from."
            )
        with self._lock:
            closed = self._stopping or not self._running or self._state == "failed"
        if closed:
            return False
        accepted = self._queue.put(frame)
        if accepted:
            with self._lock:
                self._frames_submitted += 1
        return accepted

    @property
    def drops(self) -> int:
        """Frames refused for want of room. See `BoundedFrameQueue.drops`."""
        return self._queue.drops

    @property
    def frames_submitted(self) -> int:
        """Frames accepted into the queue since construction."""
        with self._lock:
            return self._frames_submitted

    @property
    def frames_analysed(self) -> int:
        """Frames that came back with a placement. Failures are not counted here."""
        with self._lock:
            return self._frames_analysed

    def frame_ages_ms(self) -> tuple[float, ...]:
        """Measured ingress-to-result age per analysed frame, in milliseconds."""
        with self._lock:
            return tuple(self._frame_ages_ms)

    def latest_result(self) -> InferenceResult | None:
        """The most recent result, or `None` before the first one."""
        with self._lock:
            return self._latest

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
        """A collector over this process and the current worker pid.

        Rebuilt on every spawn because a respawned worker has a new pid, and a stale pid
        would be reported as an unreadable process - `app_cpu_percent` would be
        `None` for ever, which is honest about a dead worker and useless about a live
        one.
        """
        pids = [os.getpid()]
        labels = ["ui"]
        with self._lock:
            worker_pid = self._worker_pid
        if worker_pid is not None:
            pids.append(int(worker_pid))
            labels.append("worker")
        previous = self._collector
        if previous is not None:
            previous.stop()
        collector = MetricCollector(
            pids, interval=self._collector_interval, process_labels=labels,
            background=True,
        )
        self._collector = collector
        return collector

    # -- the three threads ------------------------------------------------------------

    def _feed(self) -> None:
        """Drain the bounded queue into the child's queue, one frame at a time.

        Reusing the bounded queue's frames rather than reading its depth: it is FIFO,
        and the queue's own `get` is the accessor its tests pin.
        """
        while True:
            with self._lock:
                stopping = self._stopping
                in_queue = self._in_queue
            if stopping or in_queue is None:
                return
            frame = self._queue.get(timeout=READ_POLL_SECONDS)
            if frame is None:
                if stopping:
                    return
                continue
            try:
                in_queue.put(frame)
            except Exception:  # noqa: BLE001 - a dead pipe is the watchdog's business
                return

    def _read(self) -> None:
        """Take results off the child's queue and publish them."""
        while True:
            with self._lock:
                if self._stopping:
                    return
                out_queue = self._out_queue
            if out_queue is None:
                return
            try:
                item = out_queue.get(timeout=READ_POLL_SECONDS)
            except queue.Empty:
                # **Not** a failure: nothing arrived inside the poll window. Catching it
                # as `Exception` instead kills the reader thread on its first empty
                # poll, and a supervisor with a dead reader keeps reporting "pending"
                # for ever while a perfectly healthy worker sits behind it - a stall
                # with no error anywhere.
                continue
            except (OSError, EOFError, ValueError):
                # The queue was closed under us during shutdown.
                return
            if item is None:
                return
            if isinstance(item, InferenceResult):
                self._on_result(item)

    def _watch(self) -> None:
        """Notice a worker that has exited. `multiprocessing` gives no usable callback."""
        while True:
            time.sleep(EXIT_POLL_SECONDS)
            with self._lock:
                if self._stopping:
                    return
                process = self._process
            if process is None:
                return
            if process.is_alive():
                continue
            exitcode = process.exitcode
            reason = (
                f"the inference worker process exited with code {exitcode} on device "
                f"{self._device_id!r}"
            )
            if self._compile_failure and exitcode is not None and exitcode != 0:
                reason = f"{COMPILE_FAILURE_REASON} (worker exit code {exitcode})"
            self._on_failure(reason)
            return

    # -- the two state transitions ----------------------------------------------------

    def _on_result(self, result: InferenceResult) -> None:
        with self._lock:
            self._latest = result
            if result.error is None:
                self._frames_analysed += 1
                age_ms = result.frame.age_at(self._clock()) * 1000.0
                self._frame_ages_ms.append(age_ms)
                if result.execution_devices:
                    self._confirmed[result.frame.stream_id] = tuple(result.execution_devices)
                self._state = "active"
            self._failure_published = False
        if result.error is not None:
            self._on_failure(f"inference failed on device {self._device_id!r}: "
                             f"{result.error}", result=result)
            return
        self.result_ready.emit(result)
        self.state_changed.emit(self.route_plan())

    def _on_failure(self, reason: str, *, result: InferenceResult | None = None) -> None:
        """One failure has happened. Restart once, then fail visibly."""
        with self._lock:
            self._errors.append(reason)
            if result is not None:
                self._latest = result
            restart = self._restarts == 0
            if restart:
                self._restarts += 1
            self.error_raised.emit(reason)
        if restart:
            self._restart(reason)
            return
        with self._lock:
            self._state = "failed"
            # The load-bearing part: a failed route must not keep claiming a device.
            self._confirmed = {}
            self._running = False
        self.error_raised.emit(
            f"{reason}. The restart budget for generation {self._generation} is spent, so "
            f"this route is failed and no device is being claimed for it. The requested "
            f"device {self._device_id!r} is recorded; re-select one to try again."
        )
        self.state_changed.emit(self.route_plan())

    def _restart(self, reason: str) -> None:
        """Bring a fresh worker up for the same generation, once."""
        with self._lock:
            if self._stopping:
                return
            process = self._process
            in_queue = self._in_queue
            out_queue = self._out_queue
            self._process = None
            self._in_queue = None
            self._out_queue = None
            self._worker_pid = None
        self._halt_threads()
        if process is not None:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        # The old transport is abandoned here, not at exit. See `_retire`: a queue left
        # un-cancelled hangs the interpreter in `multiprocessing`'s atexit finaliser,
        # and this is the path that used to leave one behind.
        self._retire(in_queue, out_queue)
        self._drain_queue()
        with self._lock:
            if self._stopping:
                return
            self._state = "pending"
        self.state_changed.emit(self.route_plan())
        self._spawn()

    def __enter__(self) -> "Supervisor":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        plan = self.route_plan()
        return (f"Supervisor(requested={plan.requested!r}, "
                f"confirmed={ {k: list(v) for k, v in plan.confirmed.items()} !r}, "
                f"state={plan.state!r}, generation={plan.generation}, "
                f"restarts={self.restart_count})")