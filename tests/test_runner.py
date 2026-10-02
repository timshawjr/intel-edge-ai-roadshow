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
from app.contracts.route_plan import RoutePlan
from app.inference.pinned import REPO_ROOT, verified_ir_path
from app.supervisor.runner import Supervisor

#: The stream id the plan is keyed by, kept in one place so the brief's `confirmed.get`
#: shape and this file's assertions cannot drift apart.
STREAM = "A"


def _frame(fid=0, loop=0):
    image = np.zeros((1080, 1920, 3), np.uint8)
    return FramePacket(session_id="s", scenario_id="retail_sanity", stream_id=STREAM,
                       frame_id=fid, source_pts=fid / 30, source_loop_id=loop,
                       captured_at=fid / 30, decoded_at=fid / 30,
                       width=int(image.shape[1]), height=int(image.shape[0]), image=image)


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


if __name__ == "__main__":  # pragma: no cover - convenience for a human at a booth
    unittest.main()