"""The supervisor: routing generations, placement proof, and a one-time restart.

Review Focus #5 (*"the overlay shows a different device from the one actually
executing"*) and #6 (*"the worker wedges and the UI freezes"*) are both decided here,
and both are decided by what the supervisor **refuses**:

- `Routing::test_a_fresh_supervisor_reports_a_pending_route_then_a_confirmed_one` is
  the honest-first-state test. A supervisor that has been handed `device_id="CPU"` and
  has spawned a worker still reports `state == "pending"` until a result comes back
  carrying `EXECUTION_DEVICES`. A supervisor that set `active` in `start()` would be
  claiming a placement nobody observed - the same defect as the frame-identity bug, one
  layer up. The polling loop is real: it waits for the compile and the inference, and a
  supervisor that never became active would hang it out rather than pass vacuously.
- `test_a_fresh_supervisor_does_not_already_claim_the_requested_device` is the
  negative control for the same thing, and it is separate on purpose: the positive test
  above passes just as well for a supervisor that hardcodes `confirmed = requested`.
  This one asserts that *before* anything has run, `is_device("CPU")` is `False` while
  `requested_device("A")` is `"CPU"`.
- `test_a_failed_device_surfaces_the_failure_and_never_claims_the_target` builds the
  fault for real - `compile_failure=True` makes the child process exit before it can
  compile - and requires that after the one permitted restart the plan is `failed`, that
  `confirmed` does not name `NPU`, and that the error text is stated rather than blank.

**The brief's version of that last test is vacuous and was replaced.** It read:

    self.assertNotEqual(plan.confirmed.get("A", ()), ("NPU",))

which passes for any supervisor whatsoever, including one that never confirms anything
at all, because the default `()` already differs from `("NPU",)`. I verified that by
deleting the whole confirmation path from a scratch copy and re-running: still green.
The replacement asserts on the thing the test is named for - `state == "failed"`,
`is_device("NPU") is False`, and a stated reason - and additionally asserts that the
supervisor **did** start attempting the requested device first, so a supervisor that
failed for an unrelated reason cannot satisfy it.

**Why `is_device` is the assertion and not string equality.** Measured on this machine:
CPU reports `['CPU']`, GPU `['GPU.0']` and NPU the bare string `'NPU'`. `'NPU' in 'NPU'`
is `True` by substring, and would also be `True` for `'NP' in 'NPU'`. Every assertion
here goes through `app.inference.devices.is_device`.

**`restart_count` is asserted as exactly 1.** The one-time restart is the whole of
AGENTS.md rule 12: a worker that dies is restarted once automatically, and a second
death is a failure somebody has to see. A supervisor that restarted in a loop would
keep `state == "pending"` forever and the test's `state == "failed"` assertion would
never be reached - so pinning both the count and the state is what makes the loop
visible.
"""

import os
import threading
import time
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.contracts.frame import FramePacket
from app.contracts.route_plan import RoutePlan, StreamStats
from app.engine.worker import InferenceResult
from app.inference.pinned import REPO_ROOT, verified_ir_path
from app.supervisor import runner
from app.supervisor.runner import (
    DEFAULT_STREAM_IDS,
    FRAME_AGE_WINDOW,
    REFUSAL_REASONS,
    Supervisor,
    TransitionOutcome,
)

#: The stream id the plan is keyed by, kept in one place so the brief's `confirmed.get`
#: shape and this file's assertions cannot drift apart.
STREAM = "A"


#: The four rows of spec section 5. `CPU` on Camera A is constant across every row, which
#: is the point: the toggles move inference *jobs* onto the CPU, they do not disable
#: hardware. Consumed by `SlowFourCombinationHardwareProof`, which prints the measured
#: transition times for each row.
SPEC_S5_TABLE = (
    # (gpu_toggle, npu_toggle) -> {stream: device}
    (("On", "On"), {"A": "CPU", "B": "GPU", "C": "NPU"}),
    (("Off", "On"), {"A": "CPU", "B": "CPU", "C": "NPU"}),
    (("On", "Off"), {"A": "CPU", "B": "GPU", "C": "CPU"}),
    (("Off", "Off"), {"A": "CPU", "B": "CPU", "C": "CPU"}),
)


def _frame(fid=0, loop=0, stream=STREAM, size=(1080, 1920)):
    height, width = size
    image = np.zeros((height, width, 3), np.uint8)
    return FramePacket(session_id="s", scenario_id="retail_sanity", stream_id=stream,
                       frame_id=fid, source_pts=fid / 30, source_loop_id=loop,
                       captured_at=fid / 30, decoded_at=fid / 30,
                       width=int(width), height=int(height), image=image)


class _Feeder:
    """A 15 fps submit loop for every stream, which is what a route needs to confirm.

    **`begin_transition` cannot confirm anything by itself**, and that is the design: a
    transition is acknowledged by an inference on the new route, and an inference needs a
    frame. In the booth `app.main`'s feeder never pauses - it is a 15 fps source - so a
    toggle mid-feed puts the next frames straight onto the new worker. A test that calls
    `begin_transition` and then waits without feeding is testing a supervisor nobody is
    using, and its timeout is not evidence about the protocol.

    240x320 rather than 1920x1080: `FramePacket` validates the declaration against the
    array, the worker letterboxes to the pinned 640x640, and a 6 MB array per frame per
    stream at 45 frames/s would make these tests cost minutes of pickling for no extra
    coverage.
    """

    def __init__(self, supervisor: Supervisor, fps: float = 15.0) -> None:
        self._supervisor = supervisor
        self._interval = 1.0 / float(fps)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.submitted = 0
        self.refused = 0

    def _run(self) -> None:
        frame_id = 0
        next_at = time.perf_counter()
        streams = self._supervisor.stream_ids
        while not self._stop.is_set():
            for stream in streams:
                if self._stop.is_set():
                    return
                accepted = self._supervisor.submit(
                    _frame(frame_id, stream=stream, size=(240, 320)))
                self.submitted += 1
                if not accepted:
                    self.refused += 1
                frame_id += 1
            next_at += self._interval
            delay = next_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                # Never accumulate an unbounded debt: a slow machine should fall behind
                # in the *offer rate*, which the queue then counts as drops, rather than
                # sprinting to catch up.
                next_at = time.perf_counter()

    def start(self) -> "_Feeder":
        self._thread = threading.Thread(target=self._run, name="test-feeder", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _result(fid=0, stream=STREAM, devices=("CPU",), requested="CPU",
            latency_ms=1.0, error=None, loop=0):
    """An `InferenceResult` built by hand, for the paths that do not need a worker.

    Deliberately constructed through the real constructor, never through a stand-in
    object: the constructor is where the "no latency beside an error, no placement
    beside a failure" rules live, and a fake result object would bypass exactly the
    checks under test.
    """
    if error is not None:
        return InferenceResult.failure(_frame(fid, loop, stream), error, requested)
    return InferenceResult(frame=_frame(fid, loop, stream), detections=(),
                           latency_ms=float(latency_ms),
                           execution_devices=tuple(devices), error=None,
                           requested_device=requested)


def _ages_of(supervisor: Supervisor, stream_id: str) -> tuple[float, ...]:
    """One stream's retained frame ages.

    `frame_ages_ms()` concatenates every stream's, so it cannot say which one lost its
    history - and that per-stream distinction is the whole of one of the tests using
    this. Read under the supervisor's own lock, like `frame_ages_ms` reads it.
    """
    with supervisor._lock:
        return tuple(supervisor._streams[stream_id].frame_ages_ms)  # noqa: SLF001


def _private_stream(supervisor: Supervisor, stream_id: str):
    """The supervisor's internal `_Stream`, for wiring assertions only.

    Admitted deliberately: two of the properties under test - which queue a stream's
    `drops` reads, and whether a restart spawned a process - are wiring, and there is no
    public accessor for either. Every use is marked and explained at the call site.
    """
    return supervisor._streams[stream_id]  # noqa: SLF001


def _require_cpu() -> str | None:
    """Why this machine cannot run the routing tests, or `None` if it can."""
    try:
        import openvino as ov
    except ImportError:
        return "openvino is not importable"
    if "CPU" not in ov.Core().available_devices:
        return "no CPU plugin on this machine"
    if not REPO_ROOT.joinpath("assets", "media", "retail-checkout.mp4").exists():
        return "the pinned clip is not provisioned"
    try:
        verified_ir_path()
    except Exception as exc:  # noqa: BLE001 - a skip reason, never a pass
        return f"the pinned IR is unusable: {exc}"
    return None


def _require_devices(*device_ids) -> str | None:
    """Why these device ids cannot be exercised here, or `None` if they can.

    Separated from `_require_cpu` because a CPU-only box can still run most of the M2
    tests, and a test that skips with "no NPU" must say that - never pass silently,
    which would leave a placement claim about the NPU unasserted and looking proven.
    """
    reason = _require_cpu()
    if reason:
        return reason
    try:
        import openvino as ov
    except ImportError:  # pragma: no cover - _require_cpu already returned
        return "openvino is not importable"
    present = tuple(ov.Core().available_devices)
    missing = [d for d in device_ids if d not in present]
    if missing:
        return f"this machine does not report {missing}; available_devices is {list(present)}"
    return None


def _wait_for(predicate, *, seconds=90.0, step=0.05):
    """Poll `predicate` until it is true or the deadline passes. Returns its last value.

    Bounded rather than a bare `sleep`, because the alternative - sleeping long enough
    and hoping - is how a test ends up passing on a machine three times slower than the
    one it was written on.
    """
    deadline = time.perf_counter() + seconds
    value = predicate()
    while not value and time.perf_counter() < deadline:
        time.sleep(step)
        value = predicate()
    return value


class RoutePlanContract(unittest.TestCase):
    """The record itself, with no worker involved.

    These are pure and fast, and they are what makes the supervisor's assertions
    meaningful: `is_confirmed` reading the confirmed map and never the request is a
    property of `RoutePlan`, tested here directly rather than only through a process.
    """

    def test_a_plan_with_nothing_confirmed_confirms_nothing(self):
        plan = RoutePlan(generation=0, requested={STREAM: "NPU"})
        self.assertEqual(plan.confirmed_devices(STREAM), ())
        self.assertFalse(plan.is_confirmed(STREAM, "NPU"))
        self.assertFalse(plan.fell_back(STREAM))

    def test_a_fallback_is_expressible_and_visible(self):
        plan = RoutePlan(generation=1, requested={STREAM: "NPU"},
                         confirmed={STREAM: ("CPU",)}, state="active")
        self.assertEqual(plan.requested_device(STREAM), "NPU")
        self.assertEqual(plan.confirmed_devices(STREAM), ("CPU",))
        self.assertTrue(plan.fell_back(STREAM))
        self.assertFalse(plan.is_confirmed(STREAM, "NPU"))

    def test_the_gpu_dotted_spelling_is_matched_not_substringed(self):
        plan = RoutePlan(generation=1, requested={STREAM: "GPU"},
                         confirmed={STREAM: ("GPU.0",)}, state="active")
        self.assertTrue(plan.is_confirmed(STREAM, "GPU"))
        self.assertFalse(plan.is_confirmed(STREAM, "GP"))

    def test_a_failed_plan_must_carry_a_reason(self):
        with self.assertRaises(ValueError):
            RoutePlan(generation=2, requested={STREAM: "NPU"}, state="failed")

    def test_a_failed_plan_may_not_carry_a_reasonless_error(self):
        with self.assertRaises(ValueError):
            RoutePlan(generation=2, requested={STREAM: "NPU"}, state="failed",
                      errors=("",))

    def test_confirmed_cannot_name_a_stream_nobody_requested(self):
        with self.assertRaises(ValueError):
            RoutePlan(generation=3, requested={}, confirmed={"B": ("CPU",)},
                      state="active")

    def test_the_maps_are_read_only(self):
        plan = RoutePlan(generation=0, requested={STREAM: "CPU"}, state="active",
                         confirmed={STREAM: ("CPU",)})
        with self.assertRaises(TypeError):
            plan.requested[STREAM] = "NPU"
        with self.assertRaises(TypeError):
            plan.confirmed[STREAM] = ("NPU",)

    def test_a_bare_string_is_not_a_whole_confirmed_tuple(self):
        # The NPU plugin reports `'NPU'`, and accepting a str here would file it as
        # ('N', 'P', 'U') - the exact normalisation bug `normalise_execution_devices`
        # exists to absorb, re-introduced one layer up.
        with self.assertRaises(TypeError):
            RoutePlan(generation=0, requested={STREAM: "NPU"}, confirmed={STREAM: "NPU"})

    def test_an_unknown_state_is_refused(self):
        with self.assertRaises(ValueError):
            RoutePlan(generation=0, requested={STREAM: "CPU"}, state="moving")


class Routing(unittest.TestCase):
    """The supervisor against the real CPU plugin, the real IR and a real frame."""

    def setUp(self):
        reason = _require_cpu()
        if reason:
            self.skipTest(
                f"cannot run the routing tests: {reason}. This is skipped with that "
                f"reason rather than passing silently: no placement assertion about "
                f"this machine has been made."
            )
        self._supervisors: list[Supervisor] = []

    def tearDown(self):
        for supervisor in self._supervisors:
            try:
                supervisor.stop()
            except Exception:  # noqa: BLE001 - teardown must not mask the real failure
                pass

    def _supervisor(self, **kwargs) -> Supervisor:
        supervisor = Supervisor(**kwargs)
        self._supervisors.append(supervisor)
        return supervisor

    def test_a_fresh_supervisor_reports_a_pending_route_then_a_confirmed_one(self):
        s = self._supervisor(device_id="CPU")
        self.assertEqual(s.route_plan().state, "pending")
        s.start()
        s.submit(_frame(0))
        _wait_for(lambda: s.route_plan().state == "active")
        plan = s.route_plan()
        self.assertEqual(plan.state, "active", plan.to_json())
        self.assertTrue(s.is_device("CPU"), plan.to_json())
        self.assertTrue(plan.is_confirmed(STREAM, "CPU"), plan.to_json())

    def test_a_fresh_supervisor_does_not_already_claim_the_requested_device(self):
        s = self._supervisor(device_id="CPU")
        plan = s.route_plan()
        self.assertEqual(plan.requested_device(STREAM), "CPU")
        self.assertFalse(plan.is_confirmed(STREAM, "CPU"))
        self.assertFalse(s.is_device("CPU"), plan.to_json())
        self.assertFalse(plan.is_confirmed(STREAM, "CPU"))

    def test_the_confirmed_placement_comes_from_the_result_not_the_request(self):
        # The positive control for the test above: this supervisor DOES confirm CPU, and
        # the confirmation is a reading off a result, not an echo of `requested`.
        s = self._supervisor(device_id="CPU")
        s.start()
        s.submit(_frame(0))
        _wait_for(lambda: s.route_plan().state == "active")
        result = s.latest_result()
        self.assertIsNotNone(result)
        self.assertIsNone(result.error)
        self.assertEqual(result.requested_device, "CPU")
        self.assertEqual(tuple(result.execution_devices), ("CPU",))
        self.assertFalse(result.fell_back)
        self.assertEqual(s.route_plan().confirmed_devices(STREAM), ("CPU",))

    def test_selecting_a_device_bumps_the_generation_and_clears_the_confirmation(self):
        s = self._supervisor(device_id="CPU")
        s.start()
        s.submit(_frame(0))
        _wait_for(lambda: s.route_plan().state == "active")
        before = s.route_plan().generation
        after = s.select_device("CPU").generation
        self.assertGreater(after, before)
        # The old generation's proof does not carry across: a plan that kept CPU in
        # `confirmed` while the new generation is still pending would be claiming a
        # placement for a route that has not run yet.
        self.assertEqual(s.route_plan().confirmed_devices(STREAM), ())
        self.assertEqual(s.route_plan().state, "pending")

    def test_submit_refuses_a_non_frame_and_does_not_count_it_as_a_drop(self):
        s = self._supervisor(device_id="CPU")
        s.start()
        drops_before = s.drops
        with self.assertRaises(TypeError):
            s.submit([{"box": (0, 0, 1, 1)}])
        self.assertEqual(s.drops, drops_before)

    def test_a_failed_device_surfaces_the_failure_and_never_claims_the_target(self):
        s = self._supervisor(device_id="NPU", compile_failure=True)
        s.start()
        # Submit repeatedly: the failure is observed on the polling of the worker, and a
        # single submit plus a fixed sleep is how this becomes a timing test.
        deadline = time.perf_counter() + 60.0
        while s.route_plan().state not in ("active", "failed") \
                and time.perf_counter() < deadline:
            s.submit(_frame(int(time.perf_counter() * 1000) % 3000))
            time.sleep(0.05)
        plan = s.route_plan()
        self.assertEqual(plan.state, "failed", plan.to_json())
        self.assertFalse(s.is_device("NPU"), plan.to_json())
        self.assertFalse(plan.is_confirmed(STREAM, "NPU"), plan.to_json())
        self.assertEqual(plan.confirmed_devices(STREAM), ())
        self.assertTrue(plan.errors, "a failed plan must state why")
        self.assertEqual(s.restart_count, 1,
                         "AGENTS.md rule 12 restarts a failed worker exactly once")

    def test_a_failed_device_still_reports_what_it_tried(self):
        s = self._supervisor(device_id="NPU", compile_failure=True)
        s.start()
        _wait_for(lambda: s.route_plan().state == "failed", seconds=60.0)
        self.assertEqual(s.route_plan().requested_device(STREAM), "NPU")

    def test_a_result_carries_the_frame_it_analysed(self):
        s = self._supervisor(device_id="CPU")
        s.start()
        frame = _frame(7)
        s.submit(frame)
        _wait_for(lambda: s.latest_result() is not None)
        result = s.latest_result()
        self.assertEqual(result.frame.identity, frame.identity)

    def test_the_worker_compiles_once_and_reports_a_latency(self):
        s = self._supervisor(device_id="CPU")
        s.start()
        s.submit(_frame(0))
        _wait_for(lambda: s.latest_result() is not None)
        result = s.latest_result()
        self.assertIsNotNone(result.latency_ms)
        self.assertGreater(result.latency_ms, 0.0)
        self.assertEqual(s.frames_analysed, 1)

    def test_samples_are_metric_samples_and_the_collector_is_not_read_via_accessors(self):
        s = self._supervisor(device_id="CPU")
        s.start()
        s.submit(_frame(0))
        _wait_for(lambda: s.route_plan().state == "active")
        first = s.samples()
        second = s.samples()
        self.assertTrue(first and second)
        self.assertTrue(all(sample.value is None or isinstance(sample.value, float)
                            for sample in first))

    def test_stop_is_idempotent_and_refuses_submissions_afterwards(self):
        s = self._supervisor(device_id="CPU")
        s.start()
        s.stop()
        s.stop()
        self.assertFalse(s.submit(_frame(0)))


class StreamStatsContract(unittest.TestCase):
    """The per-stream record itself, with no worker and no clock involved.

    These are pure and fast. `StreamStats.reconciles()` is the per-stream
    `accepted + refused + drops == attempts` identity, and it is here as a method on the
    record so that a test - and a card - asks the same question the same way.
    """

    def test_a_fresh_record_reconciles_and_claims_nothing(self):
        stats = StreamStats(stream_id="A", requested_device="NPU")
        self.assertEqual(stats.confirmed_devices, ())
        self.assertEqual(stats.state, "pending")
        self.assertTrue(stats.reconciles())
        # The negative control for the test above, and it is the one that matters: a
        # record that has confirmed nothing must NOT read as "running where you asked",
        # because `fell_back() == False` next to an empty confirmation is exactly that
        # reading.
        self.assertFalse(stats.fell_back())

    def test_a_reconciliation_that_does_not_hold_is_reported_as_not_holding(self):
        stats = StreamStats(stream_id="A", requested_device="CPU", frames_accepted=2,
                            drops=3, frame_attempts=9)
        self.assertFalse(stats.reconciles(),
                         "2 accepted + 3 dropped is 5, and 9 attempts were made")

    def test_a_fallback_is_visible_on_the_record(self):
        stats = StreamStats(stream_id="C", requested_device="NPU",
                            confirmed_devices=("CPU",), state="active")
        self.assertTrue(stats.fell_back())
        # The negative control, and it is the one a substring comparison would fail: the
        # GPU plugin reports `GPU.0`, and `is_device(("GPU.0",), "GPU")` is True through
        # `app.inference.devices`. A `== "GPU"` comparison would call this a fallback and
        # put "it fell back to the GPU" on a card for a route that ran exactly where it
        # was asked to.
        self.assertFalse(StreamStats(stream_id="B", requested_device="GPU",
                                     confirmed_devices=("GPU.0",),
                                     state="active").fell_back())
        self.assertFalse(StreamStats(stream_id="C", requested_device="NPU",
                                     confirmed_devices=("NPU",),
                                     state="active").fell_back())

    def test_a_negative_count_is_refused(self):
        with self.assertRaises(ValueError):
            StreamStats(stream_id="A", requested_device="CPU", drops=-1)

    def test_the_reasons_map_is_read_only(self):
        stats = StreamStats(stream_id="A", requested_device="CPU",
                            frames_refused_reasons={"closed": 1})
        with self.assertRaises(TypeError):
            stats.frames_refused_reasons["closed"] = 99

    def test_the_state_must_be_one_the_code_can_produce(self):
        with self.assertRaises(ValueError):
            StreamStats(stream_id="A", requested_device="CPU", state="moving")


class RoutePlanMultiStream(unittest.TestCase):
    """The plan's per-stream fields, with no worker involved."""

    def test_the_aggregate_state_is_failed_only_when_every_stream_failed(self):
        mixed = RoutePlan(generation=2, requested={"A": "CPU", "C": "NPU"},
                          confirmed={"A": ("CPU",)}, state="pending",
                          stream_states={"A": "active", "C": "failed"})
        self.assertEqual(mixed.state, "pending")
        self.assertEqual(mixed.stream_state("C"), "failed")
        self.assertEqual(mixed.stream_state("A"), "active")
        all_failed = RoutePlan(generation=2, requested={"A": "CPU", "C": "NPU"},
                               state="failed", errors=("A died", "C died"),
                               stream_states={"A": "failed", "C": "failed"})
        self.assertEqual(all_failed.state, "failed")

    def test_a_stream_with_no_declared_state_is_unknown_not_pending(self):
        plan = RoutePlan(generation=1, requested={"A": "CPU"}, state="pending")
        self.assertIsNone(plan.stream_state("A"))

    def test_an_unknown_per_stream_state_is_refused(self):
        with self.assertRaises(ValueError):
            RoutePlan(generation=1, requested={"A": "CPU"},
                      stream_states={"A": "moving"})

    def test_a_stream_state_for_a_stream_nobody_requested_is_refused(self):
        with self.assertRaises(ValueError):
            RoutePlan(generation=1, requested={"A": "CPU"},
                      stream_states={"Z": "active"})

    def test_the_per_stream_maps_are_read_only(self):
        plan = RoutePlan(generation=1, requested={"A": "CPU"}, stream_states={"A": "active"})
        with self.assertRaises(TypeError):
            plan.stream_states["A"] = "failed"


class SupervisorConstruction(unittest.TestCase):
    """Which streams exist, and what a bad request is refused with. No worker, no clock."""

    def setUp(self):
        self._made: list[Supervisor] = []

    def tearDown(self):
        for supervisor in self._made:
            supervisor.stop()

    def _supervisor(self, *args, **kwargs) -> Supervisor:
        supervisor = Supervisor(*args, **kwargs)
        self._made.append(supervisor)
        return supervisor

    def test_three_devices_make_three_streams_on_the_spec_ids(self):
        s = self._supervisor(device_ids=("CPU", "GPU", "NPU"))
        self.assertEqual(s.stream_ids, DEFAULT_STREAM_IDS)
        plan = s.route_plan()
        self.assertEqual(dict(plan.requested), {"A": "CPU", "B": "GPU", "C": "NPU"})
        self.assertEqual(s.per_stream()[0].stream_id, "A")
        self.assertEqual(s.per_stream()[2].requested_device, "NPU")

    def test_the_default_construction_is_still_one_cpu_stream(self):
        s = self._supervisor()
        self.assertEqual(s.stream_ids, ("A",))
        self.assertEqual(s.device_id, "CPU")
        self.assertEqual(s.stream_id, "A")

    def test_three_streams_may_all_name_the_same_device(self):
        # Spec section 5's fourth row is exactly this: `{"A": "CPU", "B": "CPU",
        # "C": "CPU"}`. An earlier version of `Supervisor` refused a repeated
        # `device_ids` entry on the reasoning that two streams on one device must be a
        # mistake, which would have made the configuration the table is built around
        # unbuildable. The spec is right and the refusal is gone; this test pins that.
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        self.assertEqual(s.stream_ids, DEFAULT_STREAM_IDS)
        self.assertEqual(dict(s.route_plan().requested),
                         {"A": "CPU", "B": "CPU", "C": "CPU"})

    def test_naming_both_spellings_of_the_route_is_refused(self):
        with self.assertRaises(ValueError):
            self._supervisor("CPU", device_ids=("CPU", "GPU"))

    def test_a_bare_string_for_several_devices_is_refused(self):
        # "CPUGPU" is three characters and three wrong stream ids. Accepted silently it
        # would build three streams that cannot compile.
        with self.assertRaises(TypeError):
            self._supervisor(device_ids="CPU")

    def test_asking_for_more_devices_than_named_streams_is_refused(self):
        with self.assertRaises(ValueError):
            self._supervisor(device_ids=("CPU", "GPU", "NPU"), stream_ids=("A", "B"))

    def test_two_streams_may_not_share_an_id(self):
        with self.assertRaises(ValueError):
            self._supervisor(device_ids=("CPU", "GPU"), stream_ids=("A", "A"))

    def test_an_empty_device_list_is_refused(self):
        with self.assertRaises(ValueError):
            self._supervisor(device_ids=())

    def test_routing_a_stream_this_supervisor_does_not_own_is_refused(self):
        s = self._supervisor(device_ids=("CPU",))
        with self.assertRaises(KeyError):
            s.select_devices({"Z": "CPU"})

    def test_a_bare_device_id_is_not_a_mapping(self):
        s = self._supervisor()
        with self.assertRaises(TypeError):
            s.select_devices("GPU")

    def test_a_refusal_reasons_map_that_names_an_unknown_reason_would_be_a_lie(self):
        # The vocabulary is named once and pinned here: a consumer switches on these
        # words, so a silent typo in one of them is a silent loss of a whole category.
        # Note what is **absent**: there is no "transitioning", because `begin_transition`
        # spawns the new worker immediately and a frame offered mid-transition is the new
        # target's first submission. Refusing it is a deadlock.
        self.assertEqual(set(REFUSAL_REASONS),
                         {"not_running", "closed", "failed", "unknown_stream"})


class PerStreamAccounting(unittest.TestCase):
    """Task 1: the per-stream identity, the unknown-stream refusal, and no cross-talk.

    These run against **unstarted** supervisors on purpose. Every one of them is about
    wiring - which counter belongs to which stream - and an unstarted supervisor produces
    that wiring in microseconds with no compile, no process and no flakiness. The
    accounting under a real running load is `PerStreamAccountingUnderLoad` below, and the
    placement proof is `Routing.*`.
    """

    def setUp(self):
        self._made: list[Supervisor] = []

    def tearDown(self):
        for supervisor in self._made:
            supervisor.stop()

    def _supervisor(self, **kwargs) -> Supervisor:
        supervisor = Supervisor(**kwargs)
        self._made.append(supervisor)
        return supervisor

    def test_a_frame_for_a_stream_nobody_owns_is_refused_and_counted_with_a_reason(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        for _ in range(3):
            self.assertFalse(s.submit(_frame(stream="Z")))
        self.assertEqual(s.unknown_stream_refusals, 3)
        self.assertEqual(s.unknown_stream_refusal_detail(), {"Z": 3})
        self.assertEqual(s.frames_refused_reasons.get("unknown_stream"), 3)
        # It is refused, not dropped: nothing was queued, so nothing was late.
        self.assertEqual(s.drops, 0)
        # And it is charged to no stream at all. Filing another camera's frame on this
        # stream's counters would reconcile perfectly and be entirely wrong.
        for stats in s.per_stream():
            self.assertEqual(stats.frame_attempts, 0, stats.to_json())
            self.assertEqual(stats.frames_refused, 0, stats.to_json())
            self.assertEqual(stats.drops, 0, stats.to_json())

    def test_the_identity_holds_per_stream_and_not_just_in_the_aggregate(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        self.assertFalse(s.submit(_frame(0, stream="A")))
        self.assertFalse(s.submit(_frame(1, stream="A")))
        self.assertFalse(s.submit(_frame(0, stream="B")))
        for stats in s.per_stream():
            self.assertTrue(stats.reconciles(), stats.to_json())
        by_id = {stats.stream_id: stats for stats in s.per_stream()}
        self.assertEqual(by_id["A"].frame_attempts, 2)
        self.assertEqual(by_id["A"].frames_refused, 2)
        self.assertEqual(by_id["A"].frames_refused_reasons, {"not_running": 2})
        self.assertEqual(by_id["B"].frame_attempts, 1)
        self.assertEqual(by_id["C"].frame_attempts, 0)

    def test_the_three_refusal_reasons_stay_apart_per_stream(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        for _ in range(2):
            s.submit(_frame(stream="A"))
        s.stop()
        s.submit(_frame(stream="A"))
        by_id = {stats.stream_id: stats for stats in s.per_stream()}
        self.assertEqual(by_id["A"].frames_refused_reasons,
                         {"not_running": 2, "closed": 1})
        self.assertTrue(by_id["A"].reconciles(), by_id["A"].to_json())
        self.assertEqual(by_id["B"].frames_refused_reasons, {})

    def test_one_streams_drops_do_not_land_on_another_streams_record(self):
        # A shared counter is the obvious way to get this wrong and it reconciles
        # perfectly, so the assertion is made by **filling one stream's queue past its
        # capacity** and then asking the other two for their numbers. `stats.drops` reads
        # that stream's own `BoundedFrameQueue`; a single shared counter - or an aggregate
        # that reads one arbitrary stream - would put Camera C's drops on A and B.
        #
        # Camera **C** is the one filled, deliberately: an implementation that reads the
        # *first* stream for the aggregate then reports 0 while `s.drops` should be C's.
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        # The queue is the supervisor's private wiring, so reaching one is admitted here
        # rather than pretended away; there is no public accessor for it.
        filled = _private_stream(s, "C").queue
        for index in range(40):
            filled.put(_frame(index, stream="C"))
        self.assertGreaterEqual(filled.drops, 38)
        by_id = {stats.stream_id: stats for stats in s.per_stream()}
        self.assertGreater(by_id["C"].drops, 0, by_id["C"].to_json())
        self.assertEqual(by_id["A"].drops, 0, by_id["A"].to_json())
        self.assertEqual(by_id["B"].drops, 0, by_id["B"].to_json())
        self.assertEqual(s.drops, by_id["C"].drops,
                         "the aggregate is the sum of the per-stream records")

    def test_select_devices_bumps_only_the_affected_stream_and_keeps_its_own_generation(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        before = s.route_plan()
        after = s.select_devices({"B": "GPU"})
        self.assertGreater(after.generation, before.generation)
        self.assertEqual(dict(after.requested), {"A": "CPU", "B": "GPU", "C": "CPU"})
        by_id = {stats.stream_id: stats for stats in s.per_stream()}
        self.assertEqual(by_id["B"].generation, after.generation)
        # A and C did not move, so their generation did not move either. This is the
        # property that makes `RoutePlan.generation` the *highest* and not "the" one, and
        # discarding results against the plan's generation would throw away valid work
        # from the two streams that are still running exactly where they were.
        self.assertEqual(by_id["A"].generation, before.generation)
        self.assertEqual(by_id["C"].generation, before.generation)
        self.assertEqual(by_id["A"].generations_seen, (before.generation,))
        self.assertEqual(by_id["B"].generations_seen, (before.generation, after.generation))

    def test_select_devices_clears_only_the_streams_it_moved(self):
        # The "**only** those streams' proof" half of the requirement, held still.
        #
        # This was a vacuous test twice, and mutation checks are what found both times.
        # First version asserted after a live transition finished: Camera A's own worker
        # re-confirms CPU about 20 ms later at 15 fps, so a supervisor that wrongly
        # cleared *every* stream's proof had already repaired the defect before the
        # assertion ran. A defect that repairs itself before the assertion is not
        # detectable by asserting after the repair.
        #
        # So the proof is **synthetic and the supervisor is fed nothing**. There is no
        # worker running to re-confirm it, and the property under test is wiring: which
        # streams `select_devices` clears. `TransitionUnderLoad`'s live test covers the
        # real-hardware version of the same claim.
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        for stream in ("A", "C"):
            s.accept_result(stream, s.stream_stats(stream).generation,
                            _result(stream=stream))
        for stream in ("A", "C"):
            self.assertEqual(s.stream_stats(stream).confirmed_devices, ("CPU",),
                             f"{stream} must have proved something first")
        self.assertEqual(s.stream_stats("B").confirmed_devices, (),
                         "Camera B starts with nothing proved")
        s.select_devices({"B": "GPU"})
        by_id = {stats.stream_id: stats for stats in s.per_stream()}
        self.assertEqual(by_id["A"].confirmed_devices, ("CPU",),
                         f"Camera A was not named: {by_id['A'].to_json()}")
        self.assertEqual(by_id["C"].confirmed_devices, ("CPU",),
                         f"Camera C was not named: {by_id['C'].to_json()}")
        self.assertEqual(by_id["A"].state, "active")
        self.assertEqual(by_id["C"].state, "active")
        # And the stream that *was* named lost its proof and went back to pending.
        self.assertEqual(by_id["B"].confirmed_devices, (), by_id["B"].to_json())
        self.assertEqual(by_id["B"].state, "pending", by_id["B"].to_json())

    def test_asking_for_the_device_a_stream_is_already_on_moves_nothing(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        before = s.route_plan()
        after = s.select_devices({"B": "CPU"})
        self.assertEqual(after.generation, before.generation)
        self.assertEqual({stats.stream_id: stats.generation for stats in s.per_stream()},
                         {"A": 1, "B": 1, "C": 1})

    def test_a_supervisor_with_three_streams_reports_three_route_entries(self):
        s = self._supervisor(device_ids=("CPU", "GPU", "NPU"))
        plan = s.route_plan()
        self.assertEqual(set(plan.requested), {"A", "B", "C"})
        self.assertEqual(dict(plan.stream_states), {"A": "pending", "B": "pending",
                                                    "C": "pending"})
        self.assertEqual(plan.confirmed, {},
                         "three pending streams have proved nothing at all")


class PerStreamAccountingUnderLoad(unittest.TestCase):
    """The same identity, with workers running and frames actually flowing."""

    def setUp(self):
        reason = _require_cpu()
        if reason:
            self.skipTest(f"cannot run the routing tests: {reason}")
        self._supervisors: list[Supervisor] = []

    def tearDown(self):
        for supervisor in self._supervisors:
            try:
                supervisor.stop()
            except Exception:  # noqa: BLE001 - teardown must not mask the real failure
                pass

    def test_hammering_one_stream_leaves_the_others_at_zero_and_all_three_reconcile(self):
        s = Supervisor(device_ids=("CPU", "CPU", "CPU"), queue_size=2)
        self._supervisors.append(s)
        s.start()
        for index in range(300):
            s.submit(_frame(index, stream="A"))
        _wait_for(lambda: s.stream_stats("B").frames_analysed >= 0 and
                  s.stream_stats("A").frames_accepted >= 0, seconds=2.0)
        by_id = {stats.stream_id: stats for stats in s.per_stream()}
        for stream_id, stats in by_id.items():
            self.assertTrue(stats.reconciles(), f"{stream_id}: {stats.to_json()}")
        # Nothing was ever offered to B or C, so their counters must be exactly zero -
        # not "small", zero.
        for stream_id in ("B", "C"):
            self.assertEqual(by_id[stream_id].frame_attempts, 0, by_id[stream_id].to_json())
            self.assertEqual(by_id[stream_id].frames_accepted, 0, by_id[stream_id].to_json())
            self.assertEqual(by_id[stream_id].drops, 0, by_id[stream_id].to_json())
        self.assertEqual(by_id["A"].frame_attempts, 300)
        # And the aggregate is the sum of the per-stream records, not a figure from one
        # of them.
        self.assertEqual(s.frame_attempts, 300)
        self.assertEqual(s.drops, sum(stats.drops for stats in by_id.values()))
        # Reported, not asserted: whether the feeder kept up is a property of this
        # machine's scheduler, and asserting a drop count would be asserting a
        # throughput result in a wiring test.
        self.assertGreaterEqual(by_id["A"].drops, 0)


class RoutingMultiStream(unittest.TestCase):
    """Three real streams, three real workers, three real placements."""

    def setUp(self):
        reason = _require_devices("CPU", "GPU", "NPU")
        if reason:
            self.skipTest(
                f"cannot run the three-stream placement tests: {reason}. Skipped with that "
                f"reason rather than passing silently: no placement assertion about the "
                f"accelerators would have been made."
            )
        self._supervisors: list[Supervisor] = []

    def tearDown(self):
        for supervisor in self._supervisors:
            try:
                supervisor.stop()
            except Exception:  # noqa: BLE001 - teardown must not mask the real failure
                pass

    def test_three_streams_each_place_on_the_device_they_were_asked_for(self):
        s = Supervisor(device_ids=("CPU", "GPU", "NPU"))
        self._supervisors.append(s)
        s.start()
        for index, stream in enumerate("ABC"):
            s.submit(_frame(index, stream=stream))
        _wait_for(lambda: all(stats.state == "active" for stats in s.per_stream()),
                  seconds=180.0)
        plan = s.route_plan()
        self.assertEqual(plan.state, "active", plan.to_json())
        for stream, device in (("A", "CPU"), ("B", "GPU"), ("C", "NPU")):
            self.assertTrue(plan.is_confirmed(stream, device),
                            f"{stream}: {plan.to_json()}")
            self.assertFalse(plan.fell_back(stream), f"{stream}: {plan.to_json()}")
            self.assertTrue(s.is_device(device, stream), f"{stream}: {plan.to_json()}")
        # The measured shapes, verbatim. GPU reports the dotted form and NPU the bare
        # string on this machine, which is why nothing here compares strings.
        by_id = {stats.stream_id: stats for stats in s.per_stream()}
        self.assertEqual(by_id["A"].confirmed_devices, ("CPU",))
        self.assertEqual(by_id["B"].confirmed_devices, ("GPU.0",))
        self.assertEqual(by_id["C"].confirmed_devices, ("NPU",))
        # And the cross-stream negatives, which are the point: Camera B's confirmation
        # says nothing about Camera C, and a supervisor that collapsed the three into one
        # verdict would answer these all `True`.
        self.assertFalse(s.is_device("NPU", "B"))
        self.assertFalse(s.is_device("GPU", "C"))
        self.assertFalse(s.is_device("GPU", "A"))
        for stats in by_id.values():
            self.assertEqual(stats.state, "active", stats.to_json())
            self.assertFalse(stats.fell_back(), stats.to_json())
            self.assertTrue(stats.reconciles(), stats.to_json())

    def test_one_failing_stream_does_not_take_down_the_other_two(self):
        # `compile_failure=["C"]` makes **only** Camera C's worker raise before it
        # compiles. This is the M2 version of the M1 failure test, and the assertion that
        # makes it worth having is the other two still running.
        #
        # Camera C asks for the **NPU**, deliberately. An earlier version had it on the
        # CPU, and a mutation check found the test was vacuous: C's worker dies before
        # any inference, so `confirmed` was already empty and asserting it stays empty
        # proved nothing about what a failed route *claims*. Asking for the NPU makes the
        # claim "the card says NPU on a route that will never run" the thing under test.
        s = Supervisor(device_ids=("CPU", "CPU", "NPU"), compile_failure=["C"])
        self._supervisors.append(s)
        s.start()
        _wait_for(lambda: s.stream_state("C") == "failed", seconds=180.0)
        deadline = time.perf_counter() + 180.0
        while not (s.stream_state("A") == "active" and s.stream_state("B") == "active") \
                and time.perf_counter() < deadline:
            s.submit(_frame(1, stream="A"))
            s.submit(_frame(1, stream="B"))
            time.sleep(0.05)
        plan = s.route_plan()
        self.assertEqual(plan.stream_state("C"), "failed", plan.to_json())
        self.assertEqual(plan.stream_state("A"), "active", plan.to_json())
        self.assertEqual(plan.stream_state("B"), "active", plan.to_json())
        # The aggregate is not "failed": two of three cameras are running, and a card
        # that renders the aggregate would tell the presenter the whole panel is dead.
        self.assertEqual(plan.state, "active", plan.to_json())
        # The failed stream claims nothing, and says why.
        self.assertEqual(plan.confirmed_devices("C"), (),
                         "a route whose worker died before compiling has proved nothing")
        self.assertFalse(plan.is_confirmed("C", "NPU"), plan.to_json())
        self.assertFalse(s.is_device("NPU", "C"), plan.to_json())
        self.assertEqual(plan.requested_device("C"), "NPU",
                         "the request is kept: 'NPU was asked for and could not run' is "
                         "what an operator needs to read")
        self.assertTrue(s.errors(), "a failed stream must state a reason")
        self.assertTrue(any("'C'" in text for text in s.errors()), s.errors())
        self.assertEqual(s.restart_count_for("C"), 1)
        self.assertEqual(s.restart_count, 1,
                         "A and B never failed, so the total is C's one restart")
        # And the two healthy streams still take work.
        self.assertTrue(s.submit(_frame(9, stream="A")))
        self.assertTrue(s.submit(_frame(9, stream="B")))
        self.assertFalse(s.submit(_frame(9, stream="C")))
        by_id = {stats.stream_id: stats for stats in s.per_stream()}
        self.assertEqual(by_id["C"].frames_refused_reasons, {"failed": 1})
        for stream_id, stats in by_id.items():
            self.assertTrue(stats.reconciles(), f"{stream_id}: {stats.to_json()}")


class TransitionOutcomeContract(unittest.TestCase):
    """The outcome record itself. No supervisor, no clock, no worker."""

    def test_an_unacknowledged_outcome_names_the_streams_that_did_not_confirm(self):
        outcome = TransitionOutcome(generation=4, requested={"B": "GPU", "C": "NPU"},
                                    confirmed={"C": ("NPU",)}, elapsed_s=2.01,
                                    bounded_by="timeout", discarded_results=3)
        self.assertFalse(outcome.acknowledged)
        # The brief's "states which stream did not confirm", as a named call rather than
        # as set arithmetic every caller repeats.
        self.assertEqual(outcome.unconfirmed_streams(), ("B",))
        self.assertEqual(outcome.to_json()["unconfirmed_streams"], ["B"])

    def test_an_acknowledged_outcome_confirms_every_stream_it_asked_about(self):
        outcome = TransitionOutcome(generation=5, requested={"B": "CPU"},
                                    confirmed={"B": ("CPU",)}, elapsed_s=0.42,
                                    bounded_by="confirmed", discarded_results=0)
        self.assertTrue(outcome.acknowledged)
        self.assertEqual(outcome.unconfirmed_streams(), ())

    def test_an_unrecognised_bounded_by_is_refused(self):
        with self.assertRaises(ValueError):
            TransitionOutcome(generation=1, requested={}, confirmed={}, elapsed_s=0.1,
                              bounded_by="probably", discarded_results=0)

    def test_a_negative_elapsed_time_is_refused(self):
        # The one number in this record that must never be adjusted to make a gate pass.
        with self.assertRaises(ValueError):
            TransitionOutcome(generation=1, requested={}, confirmed={}, elapsed_s=-1.0,
                              bounded_by="timeout", discarded_results=0)

    def test_a_confirmation_for_a_route_nobody_requested_is_refused(self):
        with self.assertRaises(ValueError):
            TransitionOutcome(generation=1, requested={}, confirmed={"B": ("CPU",)},
                              elapsed_s=0.1, bounded_by="confirmed", discarded_results=0)

    def test_the_mappings_are_read_only(self):
        outcome = TransitionOutcome(generation=1, requested={"B": "CPU"}, confirmed={},
                                    elapsed_s=0.1, bounded_by="timeout",
                                    discarded_results=0)
        with self.assertRaises(TypeError):
            outcome.requested["B"] = "NPU"


class SupersededResults(unittest.TestCase):
    """Review focus #1 for routing: an obsolete generation's result is discarded.

    **These drive `accept_result` directly, and that is deliberate rather than a
    shortcut.** `accept_result` is not a test seam - it is the call every stream's reader
    thread makes, and it is public so a test can put a chosen generation on a result
    instead of trying to win a race against a real worker. An in-flight result arriving
    exactly across a toggle is not reliably reproducible in a test, and a test that
    pretends it reproduced something is worse than a test that pins the rule.
    """

    def setUp(self):
        self._made: list[Supervisor] = []

    def tearDown(self):
        for supervisor in self._made:
            supervisor.stop()

    def _supervisor(self, **kwargs) -> Supervisor:
        supervisor = Supervisor(**kwargs)
        self._made.append(supervisor)
        return supervisor

    def test_a_result_from_a_superseded_generation_is_discarded_counted_and_never_delivered(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        delivered: list[InferenceResult] = []
        s.result_ready.connect(delivered.append)
        old_generation = s.stream_stats("B").generation
        s.begin_transition({"B": "GPU"})
        new_generation = s.stream_stats("B").generation
        self.assertNotEqual(new_generation, old_generation)

        self.assertFalse(s.accept_result("B", old_generation, _result(stream="B")))
        # Counted, not merely dropped on the floor.
        self.assertEqual(s.discarded_results, 1)
        # Never delivered - and the signal is how the overlay would have painted it.
        self.assertEqual(delivered, [])
        self.assertEqual(dict(s.results()), {})
        self.assertEqual(s.stream_stats("B").confirmed_devices, ())
        self.assertEqual(s.discarded_mismatched, 0,
                         "this is a stale generation, not a mismatched stream")

    def test_a_result_from_the_current_generation_is_delivered(self):
        # The positive control, and it is what makes the test above non-vacuous: a
        # supervisor that discarded everything would pass that one.
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        delivered: list[InferenceResult] = []
        s.result_ready.connect(delivered.append)
        current = s.stream_stats("B").generation
        self.assertTrue(s.accept_result("B", current, _result(stream="B")))
        self.assertEqual(s.discarded_results, 0)
        self.assertEqual(len(delivered), 1)
        self.assertEqual(s.stream_stats("B").confirmed_devices, ("CPU",))
        self.assertEqual(s.route_plan().stream_state("B"), "active")

    def test_the_discard_is_counted_and_attributed_to_the_transition_that_caused_it(self):
        # **The process-wide total is the assertion.** The per-transition share is an
        # approximation in the coalescing case and is documented as such in
        # `_attribute_discard`: `_Stream.superseded_by` records the route generation, and
        # two transitions can share one. Asserting the per-transition number here would
        # have been asserting an implementation detail, and the test that used to do that
        # broke when the ids were split - which is the correct outcome.
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        old_generation = s.stream_stats("B").generation
        transition = s.begin_transition({"B": "GPU"})
        # The in-flight old result lands *during* the transition, which is the only time it
        # can. Delivering it after `await_transition` would be testing the recovery's
        # bookkeeping rather than the transition's.
        s.accept_result("B", old_generation, _result(stream="B"))
        outcome = s.await_transition(transition, timeout_s=0.05)
        self.assertEqual(s.discarded_results, 1,
                         "the exact count: this result was thrown away because its "
                         "generation was superseded, and the total is what Task 3 must "
                         "report")
        self.assertEqual(s.discarded_mismatched, 0)
        # **The per-transition share is deliberately NOT asserted.** It is an
        # approximation in the coalescing case - `_attribute_discard` looks the owning
        # transition up by route generation, and two transitions can share one - so an
        # assertion on it would either pin an implementation detail or fail for a reason
        # that has nothing to do with the discard. The exact figure is the total above, and
        # the caveat is in `_attribute_discard`'s docstring and in the report.

    def test_a_restatement_does_not_lose_a_discard_from_the_process_total(self):
        # The coalescing case the approximation caveat is about: two transitions inside one
        # route generation. The total must not double-count or lose the discard, whatever
        # the per-transition split does.
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        old_generation = s.stream_stats("B").generation
        s.begin_transition({"B": "GPU"})
        s.begin_transition({"B": "GPU"})
        s.accept_result("B", old_generation, _result(stream="B"))
        self.assertEqual(s.discarded_results, 1,
                         "one superseded result is one discard, however many transitions "
                         "share the generation")

    def test_a_result_whose_frame_names_another_camera_is_never_delivered(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        current = s.stream_stats("B").generation
        delivered: list[InferenceResult] = []
        s.result_ready.connect(delivered.append)
        # Frame says Camera C, read by Camera B's reader. This is review focus #1 one
        # layer up: a result bound to a frame that is not this stream's.
        self.assertFalse(s.accept_result("B", current, _result(stream="C")))
        self.assertEqual(delivered, [])
        self.assertEqual(s.discarded_mismatched, 1)
        self.assertEqual(s.discarded_results, 1)

    def test_a_discard_is_not_a_drop_and_not_a_coasted_track(self):
        # Three populations that must never be summed into one "lost frames" figure:
        # a drop is a frame nobody analysed, a refusal is a frame the pipeline would not
        # take, and a discard is work thrown away because its routing is obsolete.
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        old_generation = s.stream_stats("C").generation
        s.begin_transition({"C": "NPU"})
        s.accept_result("C", old_generation, _result(stream="C"))
        stats = s.stream_stats("C")
        self.assertEqual(stats.drops, 0)
        self.assertEqual(stats.frames_refused, 0)
        self.assertEqual(s.discarded_results, 1)
        self.assertTrue(stats.reconciles(), stats.to_json())


class TeardownOrdering(unittest.TestCase):
    """**M2 Task 4: the stop sentinel goes in *before* the join, and the output is drained
    while the worker exits.** Both were found as costs, not as suspicions.

    - `stop()` used to join the threads first and put the sentinel after, so a feeder
      blocked writing to a full pipe waited out its full `STOP_GRACE_SECONDS` grace:
      **measured 15.18 s before, 0.84 s after**, three streams, same probe.
    - `_apply_devices` and `_restart` never sent the sentinel at all, so `join(timeout=1.0)`
      always ran out before `terminate()`. That was **~1.08 s per moved stream** of the
      warm-transition time.
    - And the sentinel alone was not enough: a worker blocked in `out_queue.put(result)`
      never comes back to read it, so one worker in three still waited out its whole grace
      and `stop()` took **5.16 s**. The parent has to drain while it waits.

    These are hardware tests and they skip loudly. The bounds are generous on purpose: the
    property is "did not wait out the grace", not "was instant".
    """

    def setUp(self):
        self._made: list[Supervisor] = []

    def tearDown(self):
        for supervisor in self._made:
            supervisor.stop()

    def _live(self, device_ids=("CPU", "GPU", "NPU")):
        """A started supervisor with a **running 15 fps feed** on every stream.

        The feed is not optional: `begin_transition`'s own docstring says a transition
        cannot confirm anything by itself, because a confirmation is an inference and an
        inference needs a frame. The first version of `_live` started the workers and no
        feeder, and every stream sat at `pending` for ever - which is correct behaviour and
        useless as a premise.
        """
        reason = _require_devices(*device_ids)
        if reason:
            self.skipTest(f"cannot build real workers: {reason}")
        supervisor = Supervisor(device_ids=device_ids)
        self._made.append(supervisor)
        feeder = _Feeder(supervisor)
        feeder.start()
        self.addCleanup(feeder.stop)
        supervisor.start()
        _wait_for(lambda: all(supervisor.stream_state(sid) == "active"
                              for sid in supervisor.stream_ids),
                  seconds=120.0)
        self.assertTrue(all(supervisor.stream_state(sid) == "active"
                            for sid in supervisor.stream_ids),
                        f"premise: every worker must really be running; states are "
                        f"{ {sid: supervisor.stream_state(sid) for sid in supervisor.stream_ids} } "
                        f"and errors {supervisor.errors()[:1]!r}")
        return supervisor

    def test_a_retiring_worker_is_asked_to_stop_before_it_is_waited_for(self):
        # The mechanism, asserted on the call rather than only on the clock: `_signal_stop`
        # must be reached, and it must be reached *before* `_halt_threads`, because a
        # feeder blocked on `in_queue.put` is unblocked by the worker draining that pipe
        # and nothing else.
        supervisor = self._live()
        order: list[str] = []
        real_signal = supervisor._signal_stop
        real_halt = supervisor._halt_threads

        def signal(in_queue):
            order.append("signal")
            return real_signal(in_queue)

        def halt(stream_id):
            order.append("halt")
            return real_halt(stream_id)

        supervisor._signal_stop = signal
        supervisor._halt_threads = halt
        supervisor.select_devices({"B": "CPU"})
        self.assertEqual(order[:2], ["signal", "halt"],
                         f"the sentinel must precede the join; the order was {order[:6]} "
                         f"for a one-stream move")
        self.assertGreaterEqual(order.count("signal"), 1,
                                "a retiring worker must be asked to stop at all")

    def test_a_route_change_does_not_wait_out_a_join_grace_per_moved_stream(self):
        supervisor = self._live()
        started = time.perf_counter()
        transition = supervisor.begin_transition({"B": "CPU"})
        supervisor.await_transition(transition, timeout_s=60.0)
        setup = time.perf_counter() - started
        # Measured on this machine after the fix: setup p50 0.723 s for the 50-toggle
        # walk, 0.866 s for a single-camera move in the six-toggle probe, and 1.1 s for a
        # two-camera move. Before the fix a single-camera move's setup was 1.127 s and a
        # two-camera move's 2.174 s - i.e. ~1.08 s of pure join grace per stream. The bound
        # here is deliberately loose: it catches "still waiting out the grace" and not a
        # slow machine.
        self.assertLess(
            setup, 3.0,
            f"the teardown of one moved stream took {setup:.2f} s; before the fix this "
            f"was a full join(1.0) before terminate(), and two streams cost 2.17 s")

    def test_stopping_a_busy_three_stream_supervisor_does_not_wait_out_the_thread_grace(self):
        supervisor = self._live()
        # Let the queues fill, so a feeder is genuinely blocked writing to a full pipe -
        # which is the state that cost the full grace per stream.
        _wait_for(lambda: supervisor.drops > 0, seconds=30.0)
        started = time.perf_counter()
        supervisor.stop()
        elapsed = time.perf_counter() - started
        # Three streams. `STOP_GRACE_SECONDS` is 5.0 and `_halt_threads` joins three
        # threads per stream, so waiting out the grace for every stream is at least 15 s.
        # Measured 15.18 s before the fix and 0.84 s after.
        self.assertLess(
            elapsed, runner.STOP_GRACE_SECONDS,
            f"stopping a busy three-stream supervisor took {elapsed:.2f} s, which is at "
            f"least one thread's full {runner.STOP_GRACE_SECONDS} s grace on one stream "
            f"and much more than that on three; the sentinel must reach the worker before "
            f"the join, and the parent must drain its output while it exits")


class DeadWorkerVisibility(unittest.TestCase):
    """**A worker that dies must become a failure somebody can see.** Task 3's first job.

    The symptom was found while proving that a `failed` stream reaches the no-move
    recovery branch, and reported as *"a worker that dies without producing a result
    leaves its stream `pending` for ever - `restart_count` 0, `pid_for` returns `None`,
    and the watchdog does not treat it as a failure"*.

    **The watchdog was not at fault, and the honest root cause is one level up.** Measured
    with every camera `compile_failure`-injected: after all three had really died,
    `Supervisor._running` was `False`; `begin_transition` computes
    `running = self._running and not self._stopping` (`runner.py:1813`) and therefore
    **skipped `_spawn` altogether**. The streams went `pending` with no process, so there
    was no child to die, no watchdog to notice, and nothing that could ever move them again.
    `restart_count` was 0 because `_on_failure` had not been called once.

    That also made round 2's published remedy false in this state - *"they have been
    re-selected on the same device, which does spawn a worker"* - when every stream is
    failed, because `_apply_devices` refused to spawn for exactly the same reason.

    So the class asserts the property, not the mechanism: **after every stream has failed,
    a route change that names them must give each one a worker**, and a stream with no
    worker must not sit at `pending` for ever.
    """

    def setUp(self):
        self._made: list[Supervisor] = []

    def tearDown(self):
        for supervisor in self._made:
            supervisor.stop()

    def _all_failed(self):
        """A started supervisor whose every stream really died, and is really `failed`.

        `compile_failure` makes the child process raise before it compiles, so this is a
        real death with a real exit code and a real watchdog noticing it - not an injected
        state variable.
        """
        reason = _require_cpu()
        if reason:
            self.skipTest(f"cannot build a real dead worker: {reason}")
        supervisor = Supervisor(device_ids=("CPU", "CPU", "CPU"),
                                compile_failure=["A", "B", "C"])
        self._made.append(supervisor)
        supervisor.start()
        gone = _wait_for(
            lambda: all(supervisor.stream_state(sid) == "failed"
                        for sid in ("A", "B", "C")),
            seconds=120.0)
        self.assertTrue(gone,
                        "premise: every stream must really have failed first; "
                        f"A={supervisor.stream_state('A')} "
                        f"B={supervisor.stream_state('B')} "
                        f"C={supervisor.stream_state('C')}")
        # `errors()` and not the `error_raised` signal: the signal is emitted from the
        # watchdog thread and Qt **queues** it to this thread, which has no event loop
        # running, so listening for it here would measure the absence of a `QApplication`
        # rather than the presence of a stated reason. `errors()` is the recorded text,
        # which is what a card reads anyway.
        errors = supervisor.errors()
        for sid in ("A", "B", "C"):
            self.assertTrue(any(f"stream {sid!r}" in text for text in errors),
                            f"the death of {sid} must be stated on the supervisor; "
                            f"errors={errors!r}")
        return supervisor, errors

    def test_a_retry_after_every_stream_failed_brings_the_workers_back(self):
        supervisor, _errors = self._all_failed()
        before = len(supervisor.errors())

        # The operator's action: re-select the devices every stream is already recorded on.
        # `failed` is always `affected`, so all three are named and the generation moves -
        # the route change is real and must be followed by a real spawn.
        supervisor.select_devices({"A": "CPU", "B": "CPU", "C": "CPU"})

        # **A worker exists for each one, synchronously, by the time the call returns.**
        # Read on the internal `_Stream` because the public `pid_for` is liveness-aware:
        # these children raise immediately, so a live pid is not observable and asserting
        # on it would be asserting on a race. The claim here is "a process was started".
        for sid in ("A", "B", "C"):
            self.assertIsNotNone(_private_stream(supervisor, sid).process,
                                 f"stream {sid} was re-selected but no worker was spawned; "
                                 f"state={supervisor.stream_state(sid)!r}. A stream with "
                                 f"no worker has no watchdog, so nothing can ever fail it "
                                 f"or bring it back.")

        # And the outcome is visible: each stream must leave `pending` and end `failed`
        # again, with the death stated. Before the fix this wait ran out with all three
        # still `pending` and `restart_count` 0.
        for sid in ("A", "B", "C"):
            self.assertIsNotNone(supervisor.restart_count_for(sid),
                                 f"premise: {sid} spent its budget once already")

        resolved = _wait_for(
            lambda: all(supervisor.stream_state(sid) == "failed"
                        for sid in ("A", "B", "C")),
            seconds=120.0)
        self.assertTrue(
            resolved,
            "a re-selected failed stream must end up failed again with the death stated, "
            f"not pending for ever: "
            + ", ".join(f"{sid}={supervisor.stream_state(sid)!r}"
                        for sid in ("A", "B", "C")))
        self.assertGreater(len(supervisor.errors()), before - 3,
                           f"the retry's own deaths must be on record; errors so far: "
                           f"{supervisor.errors()!r}")
        # **Generation-scoped, which is what makes it the retry's deaths rather than the
        # originals replayed.** `select_devices` clears each affected stream's errors and
        # moves the generation on, so the guidance left after a retry names the *new*
        # generation. Before the fix nothing was re-spawned, so the only text on record
        # still named generation 1.
        for sid in ("A", "B", "C"):
            self.assertTrue(
                any(f"restart budget for stream {sid!r} at generation 2" in text
                    for text in supervisor.errors()),
                f"the retry of {sid} failed at generation 2 and must say so; "
                f"errors={supervisor.errors()!r}")

    def test_a_retry_brings_back_exactly_one_stream_and_leaves_the_others_failed(self):
        # The negative control for the test above: a route change that names **one**
        # failed stream must not quietly revive the others. Without this, a fix that simply
        # cleared the failure bookkeeping would pass the test above while resurrecting
        # cameras the operator had not touched.
        supervisor, _errors = self._all_failed()
        before = {sid: _private_stream(supervisor, sid).process
                  for sid in ("A", "B", "C")}
        supervisor.select_devices({"B": "CPU"})
        self.assertIsNotNone(_private_stream(supervisor, "B").process,
                             "the named stream must get a worker")
        self.assertIsNot(before["B"], _private_stream(supervisor, "B").process,
                         "and it must be a *new* process, not the dead one it had")
        for sid in ("A", "C"):
            self.assertEqual(supervisor.stream_state(sid), "failed",
                             f"{sid} was not named and must be left exactly as it was")
            # Identity, not existence: a stream that has already died keeps its dead
            # `Process` object attached on purpose, because `stop()` and `_restart` need
            # it to join and terminate. `pid_for` already reports that as no worker. So
            # "nothing was started for A" means the object is *the same one*.
            self.assertIs(before[sid], _private_stream(supervisor, sid).process,
                          f"{sid} was not named, so no new process may have been started")
            self.assertIsNone(supervisor.pid_for(sid),
                              f"{sid} was not named and its old child is dead, so it has "
                              f"no worker")
        # A and C keep the reasons they already had; nothing new is claimed about them.
        for sid in ("A", "C"):
            self.assertTrue(any(f"restart budget for stream {sid!r} at generation 1" in text
                                for text in supervisor.errors()),
                            f"the original failure of {sid} is still on record, at the "
                            f"generation it happened: {supervisor.errors()!r}")


class FailedStreamRecovery(unittest.TestCase):
    """**The Critical finding:** a recovery that cannot recover must not claim it did.

    Observed on the code this replaces: a Camera C that had failed and was already sitting
    on the CPU was named by a spec section 5 toggle that **restates the whole mapping**.
    `begin_transition({"A": "CPU", "B": "CPU", "C": "CPU"})` moved nothing, the generation
    did not bump, no worker was spawned, and `await_transition` returned
    `bounded_by='timeout'` with `elapsed_s` 5e-05 s. Then `_recover` called
    `select_devices` - a no-op - and published *"Recovering C - the last healthy route"*,
    which is a claim about an action that did not happen, on the one stream an operator is
    watching. Camera C stayed `failed` on the same pid for ever.

    These are built on an **unstarted** supervisor with a synthetic failure, so each runs
    in microseconds. The hardware version of the same claim is
    `TransitionUnderLoad.test_a_failed_stream_can_be_recovered_by_selecting_a_different_
    device`.
    """

    def setUp(self):
        self._made: list[Supervisor] = []

    def tearDown(self):
        for supervisor in self._made:
            supervisor.stop()

    def _failed_stream(self, devices=("CPU", "CPU", "CPU"), failed="C"):
        """A supervisor with `failed` parked in `failed`, with no worker, at a known pid.

        The pid is left `None`, which is exactly the state that makes the recovery a no-op:
        there is no process to retire and none to bring up. `error_raised` is captured so
        the published text can be asserted on rather than inferred from `errors()`.
        """
        supervisor = Supervisor(device_ids=devices)
        self._made.append(supervisor)
        published: list[str] = []
        supervisor.error_raised.connect(published.append)
        generation = supervisor.stream_stats(failed).generation
        supervisor.accept_result(
            failed, generation, _result(stream=failed, error="the plugin said no",
                                        requested=supervisor.stream_stats(failed).requested_device))
        # The second failure spends the budget and parks the route.
        supervisor.accept_result(
            failed, supervisor.stream_stats(failed).generation,
            _result(stream=failed, error="the plugin said no again",
                    requested=supervisor.stream_stats(failed).requested_device))
        assert supervisor.stream_stats(failed).state == "failed", (
            supervisor.stream_stats(failed).to_json())
        assert _private_stream(supervisor, failed).pid is None, (
            "a failed route must have no worker; the pid is internal state, so it is "
            "read off the stream object rather than the public record, which does not "
            "expose it - and should not, since a pid is not a booth fact")
        return supervisor, published, failed

    def test_restating_the_mapping_of_a_failed_stream_moves_it_and_gives_it_a_new_pid(self):
        supervisor, _published, failed = self._failed_stream()
        before = supervisor.stream_stats(failed)
        self.assertEqual(before.generation, 1)
        self.assertIsNone(_private_stream(supervisor, failed).pid,
                          "a failed route has no worker; that is the whole defect")
        self.assertIsNone(supervisor.pid_for(failed))

        # Spec section 5 restates the WHOLE mapping on every toggle. Camera C is already
        # on the CPU, so its requested device does not change - and that is the case that
        # used to do nothing.
        transition = supervisor.begin_transition(
            {"A": "CPU", "B": "CPU", "C": "CPU"})
        self.assertGreater(transition, before.generation,
                           "a failed stream must move even when the device is unchanged; "
                           "otherwise the operator's retry is the action that does nothing")
        after = supervisor.stream_stats(failed)
        self.assertGreater(after.generation, before.generation, after.to_json())
        self.assertNotEqual(after.generations_seen, before.generations_seen)
        # A fresh budget: the retry is a new opportunity, not a continuation of the spent
        # one.
        self.assertEqual(after.state, "pending", after.to_json())
        self.assertEqual(supervisor.restart_count_for(failed), 0, after.to_json())

    def test_re_selecting_a_failed_stream_on_a_different_device_spawns_a_worker(self):
        # The remedy the failure guidance should name, exercised: a *different* device is
        # a real change, so a worker is spawned and the stream leaves `failed`.
        supervisor, _published, failed = self._failed_stream()
        supervisor.begin_transition({failed: "GPU"})
        self.assertEqual(supervisor.stream_stats(failed).requested_device, "GPU")
        self.assertNotEqual(supervisor.stream_stats(failed).state, "failed")

    def test_a_recovery_that_cannot_move_anything_does_not_claim_it_did(self):
        # A supervisor that was **never started**, restating the devices already in force:
        # nothing moves, nothing can confirm, and the recovery has nothing to restore.
        # Every stream is `pending` with no worker, which is the only shape for which a
        # re-selection really is a no-op - the next test covers the `failed` shape, which
        # reaches this same branch and is the booth-reachable one.
        supervisor = Supervisor(device_ids=("CPU", "CPU", "CPU"))
        self._made.append(supervisor)
        published: list[str] = []
        supervisor.error_raised.connect(published.append)
        transition = supervisor.begin_transition(
            {"A": "CPU", "B": "CPU", "C": "CPU"})
        supervisor.await_transition(transition, timeout_s=0.05)
        joined = " ".join(published)
        self.assertNotIn("Recovering", joined,
                         f"published a recovery for a route change that did not happen: "
                         f"{joined}")
        self.assertNotIn("the last healthy route", joined, joined)

    def test_the_published_text_says_what_to_do_when_nothing_moved(self):
        # A supervisor that was **never started**, restating the devices already in force:
        # no stream moves, no worker is spawned, and the transition times out having
        # achieved nothing - which is exactly the case where a "Recovering X" claim is a
        # lie. Every stream here is `pending` with no worker, so a re-selection really is a
        # no-op and a device change really is the only remedy.
        # **The `failed` shape reaches this same branch and is the one an operator hits**;
        # `test_a_failed_route_reaching_the_no_move_branch_is_re_selected_not_dismissed`
        # covers that, and its remedy is different.
        supervisor = Supervisor(device_ids=("CPU", "CPU", "CPU"))
        self._made.append(supervisor)
        published: list[str] = []
        supervisor.error_raised.connect(published.append)
        transition = supervisor.begin_transition(
            {"A": "CPU", "B": "CPU", "C": "CPU"})
        self.assertEqual(supervisor.route_plan().generation, 1,
                         "nothing moved, so the generation did not move")
        outcome = supervisor.await_transition(transition, timeout_s=0.05)
        self.assertFalse(outcome.acknowledged)
        self.assertEqual(outcome.unconfirmed_streams(), ("A", "B", "C"))
        joined = " ".join(published)
        self.assertNotIn("Recovering", joined,
                         f"a recovery was claimed for a route change that did not happen: "
                         f"{joined}")
        self.assertIn("Nothing moved", joined, joined)
        self.assertIn("no worker", joined, joined)
        # The remedy named for a *pending* route is a device change, and it is true: naming
        # the stream again would be a no-op, and a different device does spawn a worker.
        self.assertIn("a pending route is only re-selected when its device changes",
                      joined, joined)
        self.assertNotIn("re-selecting the same device will not fix",
                         joined.lower(), joined)
        # `select_device` must not be named as an alternative remedy. It uses
        # `force=True`, so it always moved - naming it as a *different* action from a
        # re-selection contradicts the sentence it sits in, which is what the old text
        # did ("re-selecting the same device will not fix this ... or call `select_device`
        # on it directly").
        self.assertNotIn("select_device", joined,
                         f"the message names `select_device` as an alternative remedy "
                         f"while calling re-selection useless, and it uses `force=True` so "
                         f"it always moved anyway: {joined}")
        supervisor.begin_transition({"B": "GPU"})
        self.assertEqual(supervisor.stream_stats("B").requested_device, "GPU")
        self.assertEqual(supervisor.route_plan().generation, 2)

    def test_a_failed_route_reaching_the_no_move_branch_is_re_selected_not_dismissed(self):
        """**N1, and it is Option A: the remedy is performed, not refused.**

        A `failed` stream reaches the same no-move branch as a `pending` one - it is already
        on the device the recovery wanted - and the earlier text told the operator that
        re-selecting would not help. That was **false**: `_apply_devices` treats a failed
        stream as always affected, so naming it spawns a worker and gives it a fresh
        budget. Denying the cheap correct remedy on the one path an operator walks is worse
        than the false recovery claim it replaced, so the recovery now performs it.

        This is **end-to-end**: a genuinely failed stream (the child really raises and
        really dies), a real transition, the real recovery, and the message built from that
        route's own errors. Measured on 2026-10-03, two runs: the transition timed out at
        1.276 s and 1.502 s having spawned a replacement that faulted again, and the
        recovery re-selected it - generation 2 to 3.

        The claim under test is **Option A's**: the remedy is *performed*, and the text
        says what was performed. So the assertions are the effect (the generation moved)
        and the text agreeing with it, not either alone.
        """
        supervisor = Supervisor(device_ids=("CPU", "CPU", "CPU"),
                                compile_failure=[STREAM])
        self._made.append(supervisor)
        published: list[str] = []
        supervisor.error_raised.connect(published.append)
        supervisor.start()
        deadline = time.perf_counter() + 120.0
        while supervisor.stream_stats(STREAM).state != "failed" \
                and time.perf_counter() < deadline:
            time.sleep(0.05)
        self.assertEqual(supervisor.stream_stats(STREAM).state, "failed",
                         supervisor.stream_stats(STREAM).to_json())
        self.assertIsNone(supervisor.pid_for(STREAM),
                          "a failed route has no worker; that is the premise")

        transition = supervisor.begin_transition({"A": "CPU", "B": "CPU", "C": "CPU"})
        # The generation the transition itself reached. Everything after this is the
        # recovery's doing, which is what makes it a clean way to see the effect.
        transitioned_to = supervisor.route_plan().generation
        outcome = supervisor.await_transition(transition, timeout_s=60.0)
        self.assertFalse(outcome.acknowledged, outcome.to_json())
        joined = " ".join(published)

        # **The effect, and it is the generation rather than the pid.** Re-selecting a
        # failed stream onto the device it already names is a route change - that is what
        # makes a failed stream `affected` - so it bumps the generation whether or not the
        # replacement child is alive when this line runs. Asserting on the pid instead
        # would be racy: this stream is `compile_failure`-injected, so its replacement
        # worker really does fault, and whether a live pid is visible here depends on
        # timing. The generation is the deterministic witness.
        after = supervisor.route_plan().generation
        self.assertGreater(after, transitioned_to,
                           f"the recovery re-selects the failed stream onto the same "
                           f"device, which is a route change and bumps the generation; it "
                           f"did not move from {transitioned_to} to {after}, so the "
                           f"published text is describing something that did not happen: "
                           f"{joined}")
        self.assertEqual(after, transitioned_to + 1,
                         "and exactly one re-selection: the one stream that was stuck")

        # **The text must match that effect.** It is not enough to re-select silently -
        # the operator has to be told, because the whole defect was a sentence about what
        # to do next being false. And it must not describe a failed stream as pending:
        # the `elif False` mutant (no re-selection at all) falls through to the
        # `pending` wording and would otherwise satisfy a message/effect agreement check
        # by being wrong in both halves.
        self.assertIn("re-selected on the same device", joined,
                      f"the recovery re-selected {STREAM}, so the text must say so: "
                      f"{joined}")
        self.assertIn("**failed**", joined,
                      f"the stream was failed, so the text must not describe it any other "
                      f"way: {joined}")
        self.assertNotIn("pending", joined,
                         f"the stream was failed, not pending, so the pending route's "
                         f"wording - and its device-change remedy - is false here: {joined}")
        # And it must not deny the re-selection remedy, which was the defect, nor name
        # `select_device` as the alternative: that uses `force=True` and always moved, so
        # the old sentence contradicted itself inside one clause.
        self.assertNotIn("re-selecting the same device will not fix",
                         joined.lower(), joined)
        self.assertNotIn("select_device", joined,
                         "the message named `select_device` as a remedy while also "
                         "calling re-selection useless - it uses `force=True` and always "
                         "moved, so it contradicted itself")
        self.assertNotIn("Recovering", joined,
                         f"no route change happened, so no recovery may be claimed: "
                         f"{joined}")

    def test_a_real_recovery_still_names_the_device_it_restored(self):
        # The negative control for the test above: a transition that genuinely moved a
        # stream and then failed to confirm must still publish the device, because
        # "Nothing moved" would be as wrong as the false recovery was.
        supervisor = Supervisor(device_ids=("CPU", "CPU"))
        self._made.append(supervisor)
        published: list[str] = []
        supervisor.error_raised.connect(published.append)
        transition = supervisor.begin_transition({"B": "GPU"})
        outcome = supervisor.await_transition(transition, timeout_s=0.05)
        self.assertFalse(outcome.acknowledged)
        joined = " ".join(published)
        self.assertIn("Recovering", joined, joined)
        self.assertIn("'B': 'CPU'", joined,
                      f"the recovery must name the device it put back: {joined}")
        self.assertNotIn("Nothing moved", joined, joined)
        self.assertEqual(supervisor.stream_stats("B").requested_device, "CPU")

    def test_a_mixed_recovery_names_the_stuck_stream_it_did_not_move(self):
        """The mixed branch of `_recover`: some streams move, some are already in place.

        The published text carries a **per-stream sentence** about the streams nothing
        moved for, and nothing asserted on it - so a supervisor could name them as moved
        and this would have passed. Stream A is already on the CPU when B is moved off it,
        so A is the stuck one, and its id has to appear in the sentence about being stuck.
        """
        # A **real failed stream** named alongside a stream that genuinely moves. The failed one
        # is affected (so it is in the transition) yet already on the device the recovery
        # wants - which is exactly the shape that puts one stream in the moved set and one
        # in the stuck set. A never-started supervisor cannot produce it: naming a
        # `pending` stream with an unchanged device does not make it affected, so it is
        # not in the transition at all.
        supervisor, published, failed = self._failed_stream()
        transition = supervisor.begin_transition({failed: "CPU", "B": "GPU"})
        outcome = supervisor.await_transition(transition, timeout_s=0.05)
        self.assertFalse(outcome.acknowledged)
        # Camera B moved, so it is the one being recovered.
        self.assertEqual(supervisor.stream_stats("B").requested_device, "CPU")
        joined = " ".join(published)
        self.assertIn("Recovering", joined, joined)
        stuck_clause = joined.split("Stream(s)", 1)
        self.assertEqual(len(stuck_clause), 2,
                         f"the mixed branch must publish a per-stream sentence about the "
                         f"streams it did not move: {joined}")
        sentence = stuck_clause[1]
        self.assertIn(repr(failed), sentence,
                      f"the stuck stream {failed} is not named in the sentence about the "
                      f"streams nothing moved: {sentence}")
        # The remedy stated must match the state the stream is actually in when the
        # recovery runs. It was `failed` going in - which is why `begin_transition` moved it
        # and gave it a worker - so by now it is `pending` with a worker, and the honest
        # sentence for a `pending` route is the device change. Asserting the *match* rather
        # than a fixed phrase is what makes this a test of honesty instead of of wording.
        state = supervisor.stream_state(failed)
        pid = supervisor.pid_for(failed)
        if state == "failed":
            self.assertIn("re-selected on the same device", sentence, sentence)
        else:
            self.assertIn("pending", sentence, sentence)
            self.assertIn("device change", sentence, sentence)
            # This supervisor is never started, so a `pending` stream correctly has no
            # worker and the device change really is the only remedy - which is what the
            # sentence must say.
            self.assertIsNone(pid, "an unstarted supervisor spawns nothing")

    def test_the_failure_guidance_is_recorded_on_the_stream_not_only_emitted(self):
        supervisor, published, failed = self._failed_stream()
        joined = " ".join(supervisor.errors())
        self.assertIn("re-select one to try again", joined,
                      "guidance an operator acts on must survive into "
                      "RoutePlan.errors; a signal nobody is connected to does not")
        self.assertIn(f"stream {failed!r}", joined, joined)
        # And it is still on the signal, because that is how the live UI gets it.
        self.assertIn("re-select one to try again", " ".join(published))


class PerStreamInvariantsUnderConcurrency(unittest.TestCase):
    """I1: the per-stream identity must hold on **every** snapshot, not eventually."""

    def setUp(self):
        self._made: list[Supervisor] = []

    def tearDown(self):
        for supervisor in self._made:
            supervisor.stop()

    def test_the_identity_holds_on_every_snapshot_while_another_thread_submits(self):
        # The defect: `submit` incremented `attempts` under the lock, then ran the queue
        # `put` and `accepted += 1` outside it. A `per_stream()` snapshot landing in that
        # window saw attempts=1, accepted=0 and the identity was false. Asserting only
        # "eventually reconciles" would have passed throughout.
        supervisor = Supervisor(device_ids=("CPU", "CPU", "CPU"))
        self._made.append(supervisor)
        # `compile_failure` keeps the workers out of it entirely - this is about the
        # supervisor's own accounting, and a dead worker would just make the test slow.
        supervisor.start()
        snapshot_failures: list[str] = []
        stop = threading.Event()

        def submitter():
            index = 0
            while not stop.is_set():
                supervisor.submit(_frame(index % 5000, stream="A"))
                index += 1

        def snapshotted():
            # Thousands of snapshots across the submit loop, each one checked *as taken*.
            while not stop.is_set():
                for stats in supervisor.per_stream():
                    if not stats.reconciles():
                        snapshot_failures.append(
                            f"{stats.stream_id}: {stats.to_json()}")
                        return

        feeder = threading.Thread(target=submitter, daemon=True)
        watcher = threading.Thread(target=snapshotted, daemon=True)
        feeder.start()
        watcher.start()
        feeder.join(timeout=5.0)
        stop.set()
        watcher.join(timeout=5.0)
        self.assertEqual(snapshot_failures, [],
                         "the per-stream identity went false while frames were arriving; "
                         "it must hold on every snapshot or it is not an identity")

    def test_a_malformed_submit_still_raises_before_any_counter_moves(self):
        supervisor = Supervisor(device_ids=("CPU", "CPU"))
        self._made.append(supervisor)
        before = supervisor.frame_attempts
        with self.assertRaises(TypeError):
            supervisor.submit([{"box": (0, 0, 1, 1)}])
        self.assertEqual(supervisor.frame_attempts, before,
                         "a malformed put has no stream to charge and must not move the "
                         "global attempt counter")


class FrameInvalidationAndAges(unittest.TestCase):
    """I3 and I4: the fourth population, and a bounded age window that survives a toggle."""

    def setUp(self):
        self._made: list[Supervisor] = []

    def tearDown(self):
        for supervisor in self._made:
            supervisor.stop()

    def _supervisor(self, **kwargs) -> Supervisor:
        supervisor = Supervisor(**kwargs)
        self._made.append(supervisor)
        return supervisor

    def test_the_supervisor_sums_invalidation_across_the_streams(self):
        # The symmetry check: `drops`, `frames_submitted` and `frames_analysed` all have a
        # supervisor-level sum, and `frames_invalidated` did not, so a three-stream caller
        # had to sum a per-stream field by hand. Asserted as an identity against the
        # per-stream records rather than a literal, because a literal would go stale the
        # moment the counts are produced differently.
        #
        # **Two streams, two different counts, on purpose.** A single invalidated stream
        # cannot tell "the sum" from "the one stream's count" - a mutant reading only the
        # first stream would pass. So A contributes 2 and B contributes 1, and the
        # aggregate must be 3. Each count is at `queue_size` (2) or under, because that is
        # the most one toggle can invalidate per stream: a third frame is refused at
        # `submit` as a drop, which is a different counter with a different reason.
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        for stream, count in (("A", 2), ("B", 1), ("C", 0)):
            queue = _private_stream(s, stream).queue
            for index in range(count):
                queue.put(_frame(index, stream=stream))
        s.select_devices({"A": "GPU"})
        s.select_devices({"B": "GPU"})
        by_id = {stats.stream_id: stats for stats in s.per_stream()}
        self.assertEqual(by_id["A"].frames_invalidated, 2, by_id["A"].to_json())
        self.assertEqual(by_id["B"].frames_invalidated, 1, by_id["B"].to_json())
        self.assertEqual(by_id["C"].frames_invalidated, 0, by_id["C"].to_json())
        self.assertEqual(s.frames_invalidated, 3,
                         "2 + 1 + 0 across the streams, not any one stream's count")
        self.assertEqual(s.frames_invalidated,
                         sum(stats.frames_invalidated for stats in by_id.values()),
                         "the aggregate must be the sum of the per-stream records")

    def test_frames_a_route_change_throws_away_are_counted_and_named(self):
        # **The frames here are pushed straight into the bounded queue**, not through
        # `submit`, so they never become `attempts`. That is a wiring assertion and it says
        # so: what is under test is which counter `_drain_queue` moves. The
        # identity-through-`submit` version is the live test below.
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"), queue_size=2)
        queue = _private_stream(s, "B").queue
        for index in range(6):
            queue.put(_frame(index, stream="B"))
        self.assertGreaterEqual(queue.drops, 4)
        self.assertEqual(s.stream_stats("B").frames_invalidated, 0)
        s.select_devices({"B": "GPU"})
        stats = s.stream_stats("B")
        self.assertEqual(stats.frames_invalidated, 2,
                         "queue_size is 2, so exactly the two queued frames are discarded")
        self.assertEqual(stats.state, "pending")

    def test_the_fourth_population_is_distinct_from_drops_and_refusals(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        queue = _private_stream(s, "A").queue
        for index in range(4):
            queue.put(_frame(index, stream="A"))
        s.select_devices({"A": "GPU"})
        stats = s.stream_stats("A")
        self.assertEqual(stats.frames_invalidated, 2, stats.to_json())
        self.assertEqual(stats.drops, 2, "queue-full drops are a different population")
        self.assertEqual(stats.frames_refused, 0, stats.to_json())
        self.assertEqual(stats.frames_analysed, 0)
        self.assertIn("frames_invalidated", stats.to_json())

    def test_only_the_affected_stream_records_an_invalidation(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        for stream in ("A", "C"):
            queue = _private_stream(s, stream).queue
            for index in range(2):
                queue.put(_frame(index, stream=stream))
        # B moves, so only B's queue is drained. A and C's two frames each stay put -
        # which is the point: a route change to Camera B must not cost Camera A a picture.
        s.select_devices({"B": "GPU"})
        by_id = {stats.stream_id: stats for stats in s.per_stream()}
        self.assertEqual(by_id["B"].frames_invalidated, 0, by_id["B"].to_json())
        self.assertEqual(by_id["A"].frames_invalidated, 0, by_id["A"].to_json())
        self.assertEqual(by_id["C"].frames_invalidated, 0, by_id["C"].to_json())
        self.assertEqual(len(_private_stream(s, "A").queue), 2,
                         "Camera A's two queued frames were discarded by a change that "
                         "did not touch Camera A")

    def test_the_frame_age_window_is_bounded(self):
        s = self._supervisor(device_ids=("CPU",))
        generation = s.stream_stats("A").generation
        for index in range(FRAME_AGE_WINDOW + 200):
            s.accept_result("A", generation, _result(fid=index, stream="A"))
        ages = s.frame_ages_ms()
        self.assertLessEqual(len(ages), FRAME_AGE_WINDOW,
                             f"{len(ages)} ages retained against a window of "
                             f"{FRAME_AGE_WINDOW}; the list must not grow without bound")

    def test_a_device_change_does_not_wipe_the_frame_age_history(self):
        # It used to, and it was wrong. The age is ingress-to-result through a pickling
        # queue: a measurement of the transport, not of the device. Clearing it meant the
        # displayed p50/p95 changed the instant a presenter pressed a toggle, with nothing
        # about latency having changed - the number on the card moved for a reason the card
        # could not show.
        s = self._supervisor(device_ids=("CPU", "CPU"))
        generation = s.stream_stats("A").generation
        for index in range(10):
            s.accept_result("A", generation, _result(fid=index, stream="A"))
        # Camera B needs its own history, or the second half of this test - the part that
        # actually kills mutation F8 - would be measuring A twice. A mutant that clears
        # only the *moved* stream's history passes a test that only watches the unmoved
        # one, which is exactly what the first version of this test did.
        s.accept_result("B", generation, _result(fid=0, stream="B"))
        self.assertEqual(len(s.frame_ages_ms()), 11)

        s.select_devices({"B": "GPU"})
        self.assertEqual(len(s.frame_ages_ms()), 11,
                         "the frame-age history was wiped by a device change; the "
                         "percentile would jump for no measurable reason")
        # And the stream that DID move keeps its history too, for the same reason: the
        # number is about the transport either way. This is the assertion F8 is aimed at.
        b_generation = s.stream_stats("B").generation
        self.assertNotEqual(b_generation, generation, "Camera B must have moved")
        self.assertEqual(len(_ages_of(s, "B")), 1,
                         "the stream that moved lost its lag history; a percentile that "
                         "resets on every toggle is not a latency measurement")

    def test_frame_ages_are_bounded_after_a_failed_stream_is_reselected(self):
        # The M1 single-stream path, which is the one `app/main.py` uses: the age list
        # must not accumulate without bound across repeated device selections.
        s = self._supervisor(device_ids=("CPU",))
        for round_index in range(6):
            generation = s.stream_stats("A").generation
            for index in range(40):
                s.accept_result("A", generation, _result(fid=index, stream="A"))
            s.select_device("CPU")
        self.assertLessEqual(len(s.frame_ages_ms()), FRAME_AGE_WINDOW)


class TransitionIdentity(unittest.TestCase):
    """M3: a transition's identity is monotonic per call, not its route generation."""

    def setUp(self):
        self._made: list[Supervisor] = []

    def tearDown(self):
        for supervisor in self._made:
            supervisor.stop()

    def _supervisor(self, **kwargs) -> Supervisor:
        supervisor = Supervisor(**kwargs)
        self._made.append(supervisor)
        return supervisor

    def test_two_restatements_in_one_generation_get_distinct_ids(self):
        # The defect: `_transitions` was keyed by the route generation, and a restatement
        # does not move it, so the second call overwrote the first's `started`. A late
        # `await_transition` on the first was then handed a deadline measured from the
        # second - a fresh 2 s for a transition that had already been running.
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        first = s.begin_transition({"B": "GPU"})
        self.assertEqual(s.route_plan().generation, 2)
        second = s.begin_transition({"B": "GPU"})
        self.assertEqual(s.route_plan().generation, 2,
                         "a restatement moves nothing, so the generation is unchanged")
        self.assertGreater(second, first,
                           "the ids must differ or the two records collided")

    def test_a_restatement_does_not_reset_an_earlier_transitions_deadline(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        first = s.begin_transition({"B": "GPU"})
        time.sleep(0.30)
        s.begin_transition({"B": "GPU"})
        outcome = s.await_transition(first, timeout_s=0.05)
        self.assertGreaterEqual(outcome.elapsed_s, 0.30,
                                "the first transition's deadline was measured from the "
                                "second one's `started`, which is a fresh deadline for a "
                                "transition already 300 ms old")

    def test_the_returned_id_is_awaitable_and_the_generation_is_reported_separately(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        s.accept_result("A", s.stream_stats("A").generation, _result(stream="A"))
        transition = s.begin_transition({"A": "CPU"})
        outcome = s.await_transition(transition, timeout_s=0.05)
        self.assertTrue(outcome.acknowledged, outcome.to_json())
        # The route generation is reported on the outcome, so nothing is lost by the
        # caller holding a transition id instead.
        self.assertEqual(outcome.generation, 1, outcome.to_json())
        self.assertIsNotNone(s.transition(transition))
        self.assertIsNone(s.transition(transition + 1000))

    def test_awaiting_an_id_that_was_never_handed_out_is_refused(self):
        s = self._supervisor()
        with self.assertRaises(ValueError):
            s.await_transition(999999, timeout_s=0.01)


class RetiringThreadHandover(unittest.TestCase):
    """M4: a thread that does not stop is a stated reason, not a silent overlap."""

    def setUp(self):
        self._made: list[Supervisor] = []

    def tearDown(self):
        for supervisor in self._made:
            supervisor.stop()

    def test_a_thread_that_will_not_stop_is_named_in_a_stated_reason(self):
        s = Supervisor(device_ids=("CPU",))
        self._made.append(s)
        published: list[str] = []
        s.error_raised.connect(published.append)
        # Stand in for a wedged feeder: alive, and it never exits.
        wedged = threading.Event()

        class _Stubborn(threading.Thread):
            def run(self):
                wedged.wait(10.0)

        thread = _Stubborn(name="stubborn-feeder", daemon=True)
        thread.start()
        _private_stream(s, "A").feeder = thread
        # `join` on a live thread with the real 5 s grace would cost 5 s of suite time, so
        # the grace is shortened for this test only - the branch under test is "still
        # alive after the join", which is exactly what a shorter grace makes certain.
        original = runner.STOP_GRACE_SECONDS
        runner.STOP_GRACE_SECONDS = 0.05
        try:
            s._halt_threads("A")
        finally:
            runner.STOP_GRACE_SECONDS = original
            wedged.set()
            thread.join(timeout=5.0)
        joined = " ".join(published)
        self.assertIn("did not exit", joined,
                      f"a thread that survived the join was not reported: {joined}")
        self.assertIn("feeder", joined, joined)
        self.assertIn("'A'", joined, joined)
        self.assertIn("overlap", joined, joined)
        # And it is recorded on the stream, so it survives into RoutePlan.errors.
        self.assertIn("did not exit", " ".join(s.errors()))

    def test_a_superseded_reader_retires_with_its_transport(self):
        """F15's victim, and the property is **a join duration, measured directly**.

        A reader used to loop until `self._stopping`, which only a real `stop()` sets - so a
        reader being replaced by a respawn sat on the **retired** queue for ever and
        `_halt_threads` waited out its full join grace for a thread that was never going to
        exit.

        **No `_wait_for` on `active`, and the elapsed time measures `_halt_threads` alone.**
        Both changes matter, and the first was a mistake I made:

        - The 180 s wait for the route to become `active` was **unnecessary**. `_spawn`
          assigns `stream.pid` (under the lock, `runner.py:1552-1557`) and starts the
          reader thread (`runner.py:1568-1569`) **before it returns**, so `start()` alone
          yields a valid **live** pid. The old docstring justified the wait by wanting
          "a live worker rather than a compiling one" - but the child is alive either way,
          and compiling is exactly what it is doing. It cost **182 s** of the suite's
          442 s (measured 2026-10-03: 181.979 s for this one test) for an assertion that
          needs no confirmation at all.
        - Timing `select_device()` measured far more than the join: process terminate and
          join, `_retire`, `_drain_queue`, then `_spawn` - a **process spawn** plus
          `_rebuild_collector` - against a 2.5 s bound. On a loaded booth machine that can
          exceed 2.5 s with no defect present, so it was a flaky assertion as well as a
          slow one. Timing `_halt_threads` is the property under test and is immune to
          spawn and collector latency.
        """
        s = Supervisor(device_ids=("CPU",))
        self._made.append(s)
        s.start()
        before = s.pid_for("A")
        self.assertIsNotNone(before,
                             "`start()` assigns the pid before it returns, so no wait for "
                             "`active` is needed for a live worker")

        # **The detach has to happen first, or the test measures the wrong thing.** A
        # feeder with nothing to be superseded by polls a live stream for ever, so it
        # correctly never exits on its own and every join would wait out its full grace.
        # `_apply_devices` detaches the process and queues under the lock *before* calling
        # `_halt_threads`, and that detach is exactly what lets the old threads notice
        # they are retired. So the test reproduces the handover in the same order and
        # times the join half of it - the part the assertion is about - rather than the
        # whole respawn (terminate, retire, drain, spawn a new process, rebuild the
        # telemetry collector), which is bounded by spawn latency on a loaded machine
        # rather than by anything this test is checking.
        stream = _private_stream(s, "A")
        with s._lock:
            stream.process = None
            stream.in_queue = None
            stream.out_queue = None

        started = time.perf_counter()
        s._halt_threads("A")
        elapsed = time.perf_counter() - started
        # STOP_GRACE_SECONDS is 5.0 and three threads are joined, so an unbounded reader
        # would cost at least one full 5 s grace here. A reader that retires with its
        # transport costs milliseconds. The bound is generous because the property is
        # "did not wait out the grace", not "was instant".
        self.assertLess(elapsed, runner.STOP_GRACE_SECONDS / 2.0,
                        f"halting one stream's threads took {elapsed:.2f} s, which is at "
                        f"least one thread's full {runner.STOP_GRACE_SECONDS} s join "
                        f"grace - a superseded reader is not retiring with its transport")
        self.assertEqual(s.errors(), (),
                         "and nothing should have been reported: every thread exited")

        # And the respawn still works, so the assertion is not passing because there was
        # nothing there to join. `select_device` on an already-detached stream re-spawns it.
        s.select_device("CPU")
        self.assertIsNotNone(s.pid_for("A"), "a replacement worker must be running")

    def test_a_thread_that_stops_cleanly_publishes_nothing(self):
        s = Supervisor(device_ids=("CPU",))
        self._made.append(s)
        published: list[str] = []
        s.error_raised.connect(published.append)
        thread = threading.Thread(target=lambda: None, name="well-behaved", daemon=True)
        thread.start()
        _private_stream(s, "A").feeder = thread
        s._halt_threads("A")
        self.assertEqual(published, [],
                         "the normal case must stay silent; a reason published on every "
                         "retire is a reason nobody reads")


class RouteTransition(unittest.TestCase):
    """The switching protocol of spec section 5, driven through the public API.

    The first four tests run against **unstarted** supervisors, which makes them about
    protocol rather than about hardware and takes microseconds. What cannot be faked -
    the 2 s warm-transition figure, the four real combinations, and a target that will
    not run - is in `SlowFourCombinationHardwareProof` and `TransitionUnderLoad` below.
    """

    def setUp(self):
        self._made: list[Supervisor] = []

    def tearDown(self):
        for supervisor in self._made:
            supervisor.stop()

    def _supervisor(self, **kwargs) -> Supervisor:
        supervisor = Supervisor(**kwargs)
        self._made.append(supervisor)
        return supervisor

    def test_begin_transition_asks_and_does_not_acknowledge(self):
        s = self._supervisor(device_ids=("CPU", "GPU", "NPU"))
        transition_id = s.begin_transition({"C": "CPU"})
        plan = s.route_plan()
        # The returned id is the transition's own identity and is monotonic per call; the
        # route generation is reported separately, on the plan and on the outcome. They are
        # equal here by coincidence (both counters started at 1 and both advanced once) and
        # that is not something to assert - it is exactly what M3 removed.
        self.assertGreater(transition_id, 0)
        self.assertEqual(plan.generation, 2)
        self.assertIsNotNone(s.transition(transition_id))
        # Asked, yes. Proven, no: a pending route with no confirmation claims nothing,
        # and that is the whole of "acknowledged only after the new route confirms".
        self.assertEqual(plan.requested_device("C"), "CPU")
        self.assertEqual(plan.confirmed_devices("C"), ())
        self.assertFalse(plan.is_confirmed("C", "CPU"))
        self.assertFalse(s.is_device("CPU", "C"), plan.to_json())
        self.assertEqual(plan.stream_state("C"), "pending")

    def test_await_transition_times_out_and_names_the_stream_that_did_not_confirm(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        generation = s.begin_transition({"B": "GPU", "C": "NPU"})
        outcome = s.await_transition(generation, timeout_s=0.1)
        self.assertEqual(outcome.bounded_by, "timeout")
        self.assertFalse(outcome.acknowledged)
        self.assertEqual(outcome.unconfirmed_streams(), ("B", "C"))
        self.assertEqual(dict(outcome.confirmed), {})
        self.assertGreaterEqual(outcome.elapsed_s, 0.1 - 0.02,
                                "the deadline is a real one, not a formality")
        self.assertLess(outcome.elapsed_s, 2.0,
                        "a 0.1 s timeout must not take 2 s to notice")

    def test_a_timed_out_transition_recovers_to_the_last_healthy_route_and_says_so(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        generation = s.begin_transition({"B": "GPU", "C": "NPU"})
        s.await_transition(generation, timeout_s=0.05)
        plan = s.route_plan()
        self.assertEqual(dict(plan.requested), {"A": "CPU", "B": "CPU", "C": "CPU"},
                         plan.to_json())
        reasons = " ".join(s.errors())
        self.assertIn("did not confirm", reasons)
        self.assertIn("'B'", reasons)
        self.assertIn("'C'", reasons)
        self.assertIn("Recovering", reasons)

    def test_the_recovery_reason_quotes_what_the_failing_route_actually_said(self):
        # `select_devices` clears a stream's errors, so the recovery has to read them
        # first or the operator gets "it went back to the CPU" with no word about the
        # device that refused. The text below is produced by the real failure path
        # (`accept_result` of an error result -> `_on_failure`), not written by hand, and
        # it is injected **after** the toggle because that is the order the real failure
        # happens in: the new target refuses, then the recovery runs.
        s = self._supervisor(device_ids=("CPU", "CPU"))
        generation = s.begin_transition({"B": "GPU"})
        current = s.stream_stats("B").generation
        s.accept_result("B", current,
                        _result(stream="B", error="the plugin said no", requested="GPU"))
        self.assertTrue(any("the plugin said no" in text for text in s.errors()),
                        s.errors())
        s.await_transition(generation, timeout_s=0.05)
        reasons = " ".join(s.errors())
        self.assertIn("the plugin said no", reasons,
                      f"the recovery erased the reason it was recovering from: {reasons}")

    def test_a_restart_is_not_attempted_on_a_supervisor_that_was_never_started(self):
        # `_on_failure` is reachable through `accept_result` without a `start()`, and a
        # restart there would spawn a real worker for a supervisor the caller never asked
        # to run - a real compile, on a real device, from a test.
        s = self._supervisor(device_ids=("CPU", "CPU"))
        generation = s.stream_stats("B").generation
        s.accept_result("B", generation,
                        _result(stream="B", error="the plugin said no", requested="CPU"))
        self.assertEqual(s.restart_count_for("B"), 1)
        self.assertIsNone(_private_stream(s, "B").process,
                          "a restart on a supervisor that was never started must not "
                          "spawn a worker nobody asked for")

    def test_a_late_caller_does_not_get_a_fresh_deadline(self):
        # `elapsed_s` is the age of the transition, not the time `await_transition`
        # blocked: the 2 s spec gate is a time from *toggle* to confirmed placement, and a
        # deadline that restarted whenever the caller got round to asking would make the
        # gate unmeasurable.
        #
        # **Both halves are asserted, and the second one is the one that bites.** A
        # mutation check replaced `transition.started + timeout` with `now() + timeout`
        # and this test survived while only checking `elapsed_s`, because the re-armed
        # deadline made `await_transition` block 50 ms *more* and then report an age that
        # was still over 200 ms. What distinguishes them is how long the call **blocked**:
        # against an already-expired deadline the correct code returns in one poll.
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        generation = s.begin_transition({"B": "GPU"})
        time.sleep(0.30)
        before_call = time.perf_counter()
        outcome = s.await_transition(generation, timeout_s=0.05)
        blocked = time.perf_counter() - before_call
        self.assertEqual(outcome.bounded_by, "timeout")
        self.assertGreaterEqual(outcome.elapsed_s, 0.30,
                                "the transition was already 300 ms old when the caller "
                                "asked; reporting 50 ms would be a fabricated number")
        self.assertLess(blocked, 0.025,
                        f"the deadline is 50 ms and was already 300 ms past; the call "
                        f"blocked {blocked * 1000:.1f} ms, so the deadline was re-armed")

    def test_a_coalesced_toggle_resolves_to_the_final_requested_state(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        first = s.begin_transition({"B": "GPU"})
        second = s.begin_transition({"B": "NPU"})
        self.assertGreater(second, first)
        # The superseded generation can never be acknowledged.
        stale = s.await_transition(first, timeout_s=0.05)
        self.assertEqual(stale.bounded_by, "timeout")
        self.assertFalse(stale.acknowledged)
        self.assertEqual(stale.unconfirmed_streams(), ("B",))
        # And what runs is the final request, not the one in between.
        self.assertEqual(s.route_plan().requested_device("B"), "NPU")
        self.assertEqual(s.await_transition(second, timeout_s=0.05).unconfirmed_streams(),
                         ("B",))

    def test_a_restated_routing_is_still_answered_by_confirmation_not_by_the_request(self):
        # Asking for the device a stream is already on moves nothing - and the
        # acknowledgement still has to come from a confirmed placement.
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        generation = s.stream_stats("A").generation
        s.accept_result("A", generation, _result(stream="A"))
        restated = s.begin_transition({"A": "CPU"})
        outcome = s.await_transition(restated, timeout_s=0.05)
        self.assertEqual(outcome.bounded_by, "confirmed")
        self.assertTrue(outcome.acknowledged)
        self.assertEqual(dict(outcome.confirmed), {"A": ("CPU",)})

    def test_awaiting_a_generation_that_was_never_opened_is_refused(self):
        # Kept alongside `TransitionIdentity.test_awaiting_an_id_that_was_never_handed_out_
        # is_refused`, which is the same assertion against the post-M3 name. Two tests for
        # one behaviour is redundancy on purpose here: the first pins the behaviour, the
        # second pins that the *name* moved with the parameter, and a rename that broke
        # the first would otherwise be invisible.
        s = self._supervisor()
        with self.assertRaises(ValueError):
            s.await_transition(99, timeout_s=0.01)

    def test_a_negative_timeout_is_refused(self):
        s = self._supervisor()
        with self.assertRaises(ValueError):
            s.await_transition(1, timeout_s=-1.0)

    def test_a_confirmation_that_names_a_different_device_reads_as_a_fallback(self):
        # Review focus #2, at the reporting layer: NPU asked for, `('CPU',)` reported,
        # and every reader of this plan must see a **fallback** - never a clean success
        # and never the UI claiming the NPU.
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        s.begin_transition({"C": "NPU"})
        current = s.stream_stats("C").generation
        s.accept_result("C", current,
                        _result(stream="C", devices=("CPU",), requested="NPU"))
        plan = s.route_plan()
        self.assertEqual(plan.requested_device("C"), "NPU")
        self.assertEqual(plan.confirmed_devices("C"), ("CPU",))
        self.assertTrue(plan.fell_back("C"), plan.to_json())
        self.assertFalse(plan.is_confirmed("C", "NPU"), plan.to_json())
        self.assertFalse(s.is_device("NPU", "C"), plan.to_json())
        self.assertTrue(s.is_device("CPU", "C"), plan.to_json())
        stats = s.stream_stats("C")
        self.assertTrue(stats.fell_back(), stats.to_json())
        # And it is still `active`: a fallback that actually happened is a working route
        # running somewhere else, and calling it `failed` would be a different lie.
        self.assertEqual(plan.stream_state("C"), "active", plan.to_json())

    def test_a_fallback_on_one_stream_does_not_spread_to_the_others(self):
        s = self._supervisor(device_ids=("CPU", "CPU", "CPU"))
        s.begin_transition({"B": "GPU"})
        current = s.stream_stats("B").generation
        s.accept_result("B", current,
                        _result(stream="B", devices=("CPU",), requested="GPU"))
        plan = s.route_plan()
        self.assertTrue(plan.fell_back("B"))
        self.assertFalse(plan.fell_back("A"))
        self.assertFalse(plan.fell_back("C"))
        self.assertEqual(plan.confirmed_devices("A"), ())


class TransitionUnderLoad(unittest.TestCase):
    """The protocol with real workers, real devices, real frames and a real failure."""

    def setUp(self):
        reason = _require_devices("CPU", "GPU", "NPU")
        if reason:
            self.skipTest(f"cannot run the live transition tests: {reason}")
        self._supervisors: list[Supervisor] = []
        self._feeders: list[_Feeder] = []

    def tearDown(self):
        for feeder in self._feeders:
            feeder.stop()
        for supervisor in self._supervisors:
            try:
                supervisor.stop()
            except Exception:  # noqa: BLE001 - teardown must not mask the real failure
                pass

    def _running(self, **kwargs) -> Supervisor:
        """A started supervisor with a 15 fps feed on every stream, as the booth runs it."""
        supervisor = Supervisor(**kwargs)
        self._supervisors.append(supervisor)
        supervisor.start()
        self._feeders.append(_Feeder(supervisor).start())
        return supervisor

    def test_a_transition_is_acknowledged_only_after_the_new_route_confirms(self):
        s = self._running(device_ids=("CPU", "CPU"))
        _wait_for(lambda: all(stats.state == "active" for stats in s.per_stream()),
                  seconds=180.0)
        self.assertEqual(s.stream_stats("B").confirmed_devices, ("CPU",))
        confirmed_at_request: list[tuple[str, ...]] = []
        generation = s.begin_transition({"B": "GPU"})
        # Immediately after the request, and before the GPU has compiled or inferred:
        # the plan must claim nothing. This is review focus #5 - "acknowledged before
        # the new route confirmed" - asserted at the only moment it can be wrong, with
        # the feed running throughout.
        confirmed_at_request.append(s.stream_stats("B").confirmed_devices)
        plan = s.route_plan()
        self.assertEqual(plan.requested_device("B"), "GPU")
        self.assertEqual(confirmed_at_request[0], ())
        self.assertFalse(s.is_device("GPU", "B"), plan.to_json())

        outcome = s.await_transition(generation, timeout_s=180.0)
        self.assertEqual(outcome.bounded_by, "confirmed", outcome.to_json())
        self.assertTrue(outcome.acknowledged)
        self.assertEqual(dict(outcome.confirmed), {"B": ("GPU.0",)}, outcome.to_json())
        self.assertEqual(outcome.unconfirmed_streams(), ())
        # And the acknowledgement is now true *because* an inference said so.
        self.assertTrue(s.is_device("GPU", "B"), s.route_plan().to_json())
        self.assertFalse(s.route_plan().fell_back("B"), s.route_plan().to_json())
        # The other stream was never touched by the transition: same generation, same
        # confirmation, and its result was never discarded.
        self.assertEqual(s.stream_stats("A").confirmed_devices, ("CPU",))
        self.assertEqual(s.stream_stats("A").generation, 1)
        self.assertGreater(s.stream_stats("A").frames_analysed, 0)

    def test_a_target_that_will_not_run_is_reported_and_the_route_recovers(self):
        # Not simulated. The worker is asked for a device this machine does not have, so
        # `run_worker` turns a genuine refusal into an error result and the supervisor's
        # restart-once-then-fail-visibly budget is spent for real. What must happen is
        # that Camera B ends up back on the CPU, **confirmed on the CPU**, with the
        # refusal stated - not claiming the device that was asked for.
        s = self._running(device_ids=("CPU", "CPU"))
        _wait_for(lambda: all(stats.state == "active" for stats in s.per_stream()),
                  seconds=180.0)
        generation = s.begin_transition({"B": "NOT_A_DEVICE"})
        outcome = s.await_transition(generation, timeout_s=180.0)
        self.assertEqual(outcome.bounded_by, "timeout", outcome.to_json())
        self.assertEqual(outcome.unconfirmed_streams(), ("B",))
        _wait_for(lambda: s.stream_stats("B").confirmed_devices == ("CPU",),
                  seconds=180.0)
        plan = s.route_plan()
        self.assertEqual(plan.requested_device("B"), "CPU", plan.to_json())
        self.assertEqual(plan.confirmed_devices("B"), ("CPU",), plan.to_json())
        self.assertFalse(plan.fell_back("B"), "it is running exactly where it was asked")
        self.assertFalse(s.is_device("NOT_A_DEVICE", "B"))
        reasons = " ".join(s.errors())
        self.assertIn("NOT_A_DEVICE", reasons)
        self.assertIn("Recovering", reasons)
        # The route that refused is not still claiming anything, and Camera A never
        # stopped.
        self.assertTrue(s.is_device("CPU", "A"), plan.to_json())
        self.assertEqual(plan.stream_state("A"), "active", plan.to_json())
        for stats in s.per_stream():
            self.assertTrue(stats.reconciles(), stats.to_json())
        # The recovered route is really working, not merely labelled: Camera B has
        # analysed frames again on the CPU.
        _wait_for(lambda: s.stream_stats("B").frames_analysed > 0, seconds=180.0)
        self.assertEqual(s.stream_stats("B").state, "active")

    def test_the_identity_still_holds_after_a_transition_invalidated_frames(self):
        # I3: `frames_invalidated` is a **fourth** population and must not break the
        # identity. The frames it counts were already counted once, at `submit` time, as
        # accepted - so adding the counter must not add a term to the identity, and this
        # asserts both halves: the identity holds, and the counter is non-zero on the
        # stream that actually lost frames.
        s = self._running(device_ids=("CPU", "CPU"))
        _wait_for(lambda: all(stats.state == "active" for stats in s.per_stream()),
                  seconds=180.0)
        generation = s.begin_transition({"B": "GPU"})
        s.await_transition(generation, timeout_s=180.0)
        for stats in s.per_stream():
            self.assertTrue(stats.reconciles(), stats.to_json())
        # Whether any frames were actually invalidated depends on how fast the feeder was
        # offering against a queue of 2, so the *identity* is the assertion and the count
        # is reported.
        by_id = {stats.stream_id: stats for stats in s.per_stream()}
        print(f"\n  transition invalidation: "
              + ", ".join(f"{sid}={stats.frames_invalidated}"
                          for sid, stats in by_id.items()))
        for stats in by_id.values():
            self.assertGreaterEqual(stats.frames_invalidated, 0)
            # Whatever was invalidated was accepted, so accepted is at least that many.
            self.assertGreaterEqual(stats.frames_accepted, stats.frames_invalidated)

    def test_a_transition_leaves_the_untouched_streams_proof_intact(self):
        # "Clears **only** those streams' proof" is half of the requirement and the half
        # that is easier to get wrong. Camera A is not named in the mapping, so its
        # confirmed CPU placement, its generation and its counters must all survive
        # untouched - a supervisor that cleared everything would satisfy the other half
        # of the test and leave two cards reading "moving..." after every toggle.
        #
        # **The proof is asserted at the end here; the structural claim ("only the moved
        # streams are cleared") is pinned deterministically in `PerStreamAccounting`,
        # because A re-confirms CPU about 20 ms after any false clear and a test that
        # only looked at the end state was shown by mutation checking to be blind to it.**
        s = self._running(device_ids=("CPU", "CPU"))
        _wait_for(lambda: all(stats.state == "active" for stats in s.per_stream()),
                  seconds=180.0)
        before = s.stream_stats("A")
        generation = s.begin_transition({"B": "GPU"})
        s.await_transition(generation, timeout_s=180.0)
        after = s.stream_stats("A")
        self.assertEqual(after.confirmed_devices, ("CPU",), after.to_json())
        self.assertEqual(after.generation, before.generation)
        self.assertEqual(after.generations_seen, before.generations_seen)
        self.assertGreaterEqual(after.frames_analysed, before.frames_analysed,
                                "Camera A's counters must keep running, not reset")
        self.assertTrue(s.is_device("CPU", "A"), s.route_plan().to_json())
        self.assertEqual(s.route_plan().stream_state("A"), "active")


class SlowFourCombinationHardwareProof(unittest.TestCase):
    """**This class dominates the suite's runtime. Read it as a hardware proof, not as a
    fast test.**

    ## Cost, stated so a runtime change is not read as a regression

    One test, **~37 s** measured on 2026-10-02, out of a ~130 s `tests.test_runner` run and
    a ~300 s full-suite run. It is **the slowest single test in the repository by a wide
    margin** and it accounts for the large majority of the difference between the two
    figures. Its cost is structural and irreducible:

    - **three cold `compile_model` calls** - one per device, the first time each runs
      (measured 3.5 s for all three together);
    - **three live transitions**, each respawning a worker and waiting for its compile
      (measured 1.0-1.8 s each, warm);
    - three spawned processes and nine supervisor threads alive throughout;
    - a 15 fps feed on all three streams, so the numbers are of real work.

    Measured standalone on 2026-10-02: **37 s**; on 2026-10-03 with the fix round's
    changes, **23 s**. The test prints its own `[cost]` line, so a later run carries its
    own figure rather than relying on these.

    **If the full suite's runtime moves, look here first and check the printed figure
    rather than assuming a regression.** It was renamed from `SpecSectionFiveRouting` for
    exactly that reason: "Spec" reads like a contract test and invites a runtime comparison
    that this class does not support.

    ## It is NOT the most expensive test in the file

    `RetiringThreadHandover.test_a_superseded_reader_retires_with_its_transport` measured
    **182 s** standalone - an order of magnitude more - because it waits on a CPU route
    becoming `active` before it can start measuring a join. Both are called out so neither
    is mistaken for a regression in the other.

    ## What it is actually for

    All four toggle combinations of spec section 5, executed **for real**: every one
    compiles, infers, and confirms a placement through `EXECUTION_DEVICES` read after its
    own inference on its own pid. A mapping-only assertion would pass on a supervisor
    whose workers never ran at all, which is the defect this exists to rule out.

    Camera A is on the CPU in all four rows and is asserted in all four - **and asserted to
    have never moved at all**, generation pinned at 1 - because that is what the table
    means: the toggles move inference jobs onto the CPU, they do not disable hardware.

    The transition times are **printed, not asserted against the 2 s target.** The target is
    a spec gate this milestone records; no timeout is adjusted to make it pass.
    """

    def setUp(self):
        reason = _require_devices("CPU", "GPU", "NPU")
        if reason:
            self.skipTest(f"cannot run the four-combination test: {reason}")
        self._supervisors: list[Supervisor] = []
        self._feeders: list[_Feeder] = []

    def tearDown(self):
        for feeder in self._feeders:
            feeder.stop()
        for supervisor in self._supervisors:
            try:
                supervisor.stop()
            except Exception:  # noqa: BLE001 - teardown must not mask the real failure
                pass

    def test_all_four_combinations_of_the_spec_five_table_execute(self):
        started = time.perf_counter()
        s = Supervisor(device_ids=("CPU", "GPU", "NPU"))
        self._supervisors.append(s)
        s.start()
        # The feed never pauses, exactly as in the booth: a toggle mid-feed puts the next
        # frames straight onto the new route.
        self._feeders.append(_Feeder(s).start())

        # Row 1 is the state the supervisor starts in, so it is **placed**, not
        # transitioned. It is confirmed here, live, before any toggle, and the time to
        # reach it is three cold compiles - which is not a warm-transition figure and is
        # reported separately below so the two cannot be confused.
        initial_started = time.perf_counter()
        _wait_for(lambda: all(stats.state == "active" for stats in s.per_stream()),
                  seconds=300.0)
        initial_s = time.perf_counter() - initial_started
        plan = s.route_plan()
        self.assertEqual(dict(plan.requested), SPEC_S5_TABLE[0][1], plan.to_json())
        self.assertEqual(plan.state, "active", plan.to_json())
        for stream, device in SPEC_S5_TABLE[0][1].items():
            self.assertTrue(plan.is_confirmed(stream, device), plan.to_json())
            self.assertFalse(plan.fell_back(stream), plan.to_json())

        elapsed: dict[str, float] = {}
        for (toggles, mapping) in SPEC_S5_TABLE[1:]:
            label = f"gpu={toggles[0]},npu={toggles[1]}"
            generation = s.begin_transition(mapping)
            outcome = s.await_transition(generation, timeout_s=300.0)
            elapsed[label] = outcome.elapsed_s
            plan = s.route_plan()
            self.assertEqual(dict(plan.requested), mapping,
                             f"{label}: {plan.to_json()} / {outcome.to_json()}")
            self.assertEqual(outcome.bounded_by, "confirmed",
                             f"{label}: {outcome.to_json()} / {s.errors()}")
            for stream, device in mapping.items():
                self.assertTrue(plan.is_confirmed(stream, device),
                                f"{label}: {plan.to_json()}")
                self.assertFalse(plan.fell_back(stream),
                                 f"{label}: {plan.to_json()}")
            # Camera A never leaves the CPU, in any row of the table. Not merely "still
            # on the CPU" - **never moved**, which is what makes it the control: its
            # generation has not advanced once through three toggles.
            self.assertEqual(plan.requested_device("A"), "CPU", f"{label}: {plan.to_json()}")
            self.assertEqual(s.stream_stats("A").generation, 1, f"{label}")
            self.assertEqual(s.stream_stats("A").generations_seen, (1,), f"{label}")
            self.assertTrue(s.is_device("CPU", "A"), f"{label}: {plan.to_json()}")
            for stats in s.per_stream():
                self.assertEqual(stats.state, "active", f"{label}: {stats.to_json()}")
                self.assertTrue(stats.reconciles(), f"{label}: {stats.to_json()}")
        # The measured figures are reported, not asserted against the 2 s target: the
        # target is a spec gate this milestone records, and no timeout is adjusted to
        # make it pass. Printing them here means a run's evidence is in the test output
        # rather than only in a report somebody has to remember to write.
        print(f"\n  [cost] this hardware proof took "
              f"{time.perf_counter() - started:.1f} s")
        print(f"  spec 5 initial three-device placement: {initial_s:.3f} s "
              f"(three cold compiles, not a transition)")
        print(f"  spec 5 warm transition times (s): {elapsed}")
        print(f"  spec 5 warm transitions over the 2 s target: "
              f"{sorted(k for k, v in elapsed.items() if v > 2.0) or 'none'}")
        # Nothing was ever misattributed: no result was read by a stream its frame did
        # not belong to.
        self.assertEqual(s.discarded_mismatched, 0,
                         "a result was read by a stream its frame did not belong to")
        # B and C each moved more than once, and their generations are recorded per
        # stream rather than collapsed into one verdict.
        for stream_id in ("B", "C"):
            self.assertGreater(len(s.stream_stats(stream_id).generations_seen), 1,
                               s.stream_stats(stream_id).to_json())


if __name__ == "__main__":  # pragma: no cover - convenience for a human at a booth
    unittest.main()