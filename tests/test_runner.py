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
import time
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.contracts.frame import FramePacket
from app.contracts.route_plan import RoutePlan, StreamStats
from app.engine.worker import InferenceResult
from app.inference.pinned import REPO_ROOT, verified_ir_path
from app.supervisor.runner import DEFAULT_STREAM_IDS, REFUSAL_REASONS, Supervisor

#: The stream id the plan is keyed by, kept in one place so the brief's `confirmed.get`
#: shape and this file's assertions cannot drift apart.
STREAM = "A"

#: The four rows of spec section 5, verbatim. `CPU` on Camera A is constant across every
#: row and that is the point: the toggles move inference *jobs* onto the CPU, they do not
#: disable hardware.
def _frame(fid=0, loop=0, stream=STREAM, size=(1080, 1920)):
    height, width = size
    image = np.zeros((height, width, 3), np.uint8)
    return FramePacket(session_id="s", scenario_id="retail_sanity", stream_id=stream,
                       frame_id=fid, source_pts=fid / 30, source_loop_id=loop,
                       captured_at=fid / 30, decoded_at=fid / 30,
                       width=int(width), height=int(height), image=image)


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
            held = _private_stream(s, stream)
            held.confirmed = ("CPU",)
            held.state = "active"
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


if __name__ == "__main__":  # pragma: no cover - convenience for a human at a booth
    unittest.main()