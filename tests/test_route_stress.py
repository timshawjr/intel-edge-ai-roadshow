"""The route-change stress in the suite: six alternating warm toggles.

`tools/route_stress.py` runs the real 50. This runs the **same code** for six, so the
properties the tool reports are the properties the suite enforces - a stress whose test
reimplements it is a stress with two implementations and one of them is fiction.

**What is asserted, and what is only recorded.** The brief is explicit: "record the
warm-transition distribution rather than asserting a threshold - the 2 s target is a spec
gate to be *measured*". So the gate is **recorded**, `min/p50/p95/max` and the count over
2 s, and never asserted here. What *is* asserted is everything that has no excuse:

- no crash,
- zero stale overlays and zero mislabelled devices,
- `discarded_mismatched == 0` - a result never read by a stream its frame did not belong
  to,
- every acknowledged transition confirmed on the **live pid**, not the old worker,
- the leak trend, with the designed retention accounted for rather than explained away,
- `StressResult` refuses to be built from nonsense, because a record that cannot hold a
  bad number is a record nobody has to check.

## Cost

Six toggles with a warm-up before them. The whole class measured **~25 s** on
2026-10-03, out of a ~270 s suite, and it is skipped - loudly, with a reason - on a machine
that cannot run a real CPU worker, because a stress that quietly degrades to no streams is
not a stress.

The full 50-transition run is a tool, not a test: it takes minutes and belongs in
`logs/qualification/`, not in a red/green.
"""

from __future__ import annotations

import dataclasses
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

from app.contracts.frame import FramePacket  # noqa: E402
from app.engine.worker import InferenceResult  # noqa: E402
from tools import route_stress as rs  # noqa: E402
from tools.route_stress import (  # noqa: E402
    GATE_SECONDS,
    StressResult,
    TransitionSample,
    run_stress,
)


def _frame(fid=0, stream="A", size=(240, 320)):
    """A real `FramePacket`, because `InferenceResult`'s constructor is where the rules
    under test live and a stand-in object would bypass them."""
    height, width = size
    return FramePacket(session_id="stress", scenario_id="retail_sanity",
                       stream_id=stream, frame_id=fid, source_pts=fid / 30.0,
                       source_loop_id=0, captured_at=fid / 30.0,
                       decoded_at=fid / 30.0, width=width, height=height,
                       image=np.zeros((height, width, 3), np.uint8))


def _require_hardware() -> str | None:
    """Why this machine cannot run the stress, or `None` if it can.

    A skip must always say why. "No NPU" and "the pinned IR is unusable" call for
    different responses from whoever reads the result, and neither is the same as a pass -
    a skipped stress leaves the 2 s gate **unmeasured**, which has to be visible.
    """
    try:
        import openvino as ov
    except ImportError:
        return "openvino is not importable"
    from app.inference.pinned import verified_ir_path

    if not REPO_CLIP().exists():
        return "the pinned clip is not provisioned"
    present = tuple(ov.Core().available_devices)
    missing = [d for d in ("CPU", "GPU", "NPU") if d not in present]
    if missing:
        return f"this machine does not report {missing}; available_devices is {list(present)}"
    try:
        verified_ir_path()
    except Exception as exc:  # noqa: BLE001 - a skip reason, never a pass
        return f"the pinned IR is unusable: {exc}"
    return None


def REPO_CLIP():  # noqa: N802 - a function, named in caps to read as a constant path
    from app.inference.pinned import REPO_ROOT

    return REPO_ROOT.joinpath("assets", "media", "retail-checkout.mp4")


class RouteStressInTheSuite(unittest.TestCase):
    """Six alternating warm toggles, run through the tool's own code path.

    **One run, shared by every test in the class.** `run_stress` costs **~25 s**
    (measured 2026-10-03: 3.5 s cold placement, six toggles totalling ~20 s, 1.5 s
    settle, 0.3 s teardown), so four tests each running it would cost the suite 100 s for
    the same four observations. The record is immutable and every assertion below is a
    pure function of it, so running once loses nothing.
    """

    maxDiff = None
    result = None

    @classmethod
    def setUpClass(cls):
        reason = _require_hardware()
        if reason:
            raise unittest.SkipTest(
                f"cannot run the route stress: {reason}. The 2 s warm-transition gate is "
                f"therefore UNMEASURED by this suite run - not met, and not passed.")
        cls.result = run_stress(transitions=6, timeout_s=30.0, progress=False)

    def test_six_alternating_warm_toggles_leave_nothing_stale_mislabelled_or_leaking(self):
        result = self.result

        self.assertEqual(result.transitions, 6)
        self.assertEqual(result.crashes, 0,
                         f"the stress itself crashed: {result.samples!r}")
        # **The two headline booth properties.** Both counted from the live delivery path
        # by the tool, not asserted here from a counter the supervisor maintains.
        self.assertEqual(result.stale_overlays, 0,
                         f"results from a superseded generation reached the UI: "
                         f"{result.stale_overlay_detail!r}")
        self.assertEqual(result.mislabelled_devices, 0,
                         f"a delivered result disagreed with its stream's routing: "
                         f"{result.mislabelled_detail!r}")
        self.assertEqual(result.discarded_mismatched, 0,
                         "a result was read by a stream its frame did not belong to")
        # The run was not vacuous: it fed real frames and really got results back, so the
        # zero above is a measurement rather than an absence of traffic.
        self.assertGreater(result.results_delivered, 0,
                           "nothing was delivered, so 'no stale overlays' would be "
                           "vacuous")
        self.assertGreater(result.frames_submitted, 0,
                           "no frame was submitted, so the run proves nothing")
        # Placement was proved by the worker that is running now, not the one that was
        # replaced. Asserted per acknowledged transition so a failure names which.
        for sample in result.samples:
            if not sample.acknowledged:
                continue
            for stream, pid_at_confirm in sample.pids_at_confirm.items():
                self.assertIsNotNone(
                    pid_at_confirm,
                    f"transition {sample.index} to {sample.target} was acknowledged but "
                    f"no confirming delivery was observed for {stream}")
                self.assertEqual(
                    pid_at_confirm, sample.pids_after.get(stream),
                    f"transition {sample.index} to {sample.target}: {stream} was confirmed "
                    f"on pid {pid_at_confirm} but the live pid is "
                    f"{sample.pids_after.get(stream)} - the placement was proved by a "
                    f"superseded worker")
        self.assertEqual(result.placement_pids_matched, result.placements_confirmed,
                         "not every acknowledged transition matched its live pid")

    def test_the_record_counts_the_retained_records_the_run_keeps_on_purpose(self):
        result = self.result

        # **The retention is named, so a growth is not read as a leak.** `_transitions`
        # gains one record per toggle and grows by design: a cap would drop an open record
        # and turn a legitimate `await_transition` into `ValueError`.
        self.assertEqual(result.retained_transitions_end
                         - result.retained_transitions_start, 6,
                         "one transition record per toggle, kept")
        # **And the queues are NOT retained, which is the M4 Task 3 property.** They used
        # to be counted here as design growth because the ledger was append-only: the M2
        # run ended with 156-204 closed queues held for the life of the process, and on the
        # timeout path each held up to 2 s of frames addressed to a dead worker. After the
        # fix a retired queue is released, so the ledger holds exactly the live pair per
        # stream (two queues each) and at most one queue still inside the release grace.
        live_slots = 2 * len(result.per_stream)
        self.assertLessEqual(
            result.retained_queues_end, live_slots + 2,
            f"the run ended with {result.retained_queues_end} queues in the ledger for "
            f"{live_slots} live queue slots; a retired queue is released once its feeder "
            f"is gone, so a count that grows per toggle is the M2 leak coming back")
        self.assertLessEqual(
            result.retained_queues_end, result.retained_queues_start + 2,
            f"the ledger grew from {result.retained_queues_start} to "
            f"{result.retained_queues_end} queues across six toggles; retired queues are "
            f"released, not kept")

    def test_the_retained_records_were_actually_released(self):
        """The hard half of the retention claim, and the half that is measurable.

        RSS cannot show that records were released — Python's allocator keeps freed
        arenas — so "the retention explains the growth" would be an *explanation* and not a
        measurement. **Live object count is the measurement**: emptied, then collected, and
        the count must fall by more than one object.

        **The queues are no longer part of this**: a retired queue is released during the
        run (M4 Task 3), so at the moment the record is emptied the ledger holds only the
        live pairs. What the count releases is the transition records and their mappings,
        which is why this asserts a count rather than naming a queue.

        This is separate from the RSS direction because the two fail independently: the
        count can fall while RSS does not move, and RSS can fall by more than the count
        suggests. Asserting only the direction would have let this run unmeasured.
        """
        result = self.result

        self.assertLess(
            result.objects_after_release, result.objects_before_release,
            f"emptying {result.retained_transitions_end} transition records and "
            f"{result.retained_queues_end} live queue slots released no live objects "
            f"({result.objects_before_release} -> {result.objects_after_release}); the "
            f"records were not actually released")
        # And enough of them, not one: six toggles retain six transition records, each
        # carrying its own mappings and tuples.
        self.assertGreater(
            result.objects_before_release - result.objects_after_release, 100,
            f"only {result.objects_before_release - result.objects_after_release} objects "
            f"were released for six toggles' worth of retained transition records")

    def test_nothing_grows_that_the_retained_records_do_not_account_for(self):
        """The RSS half of the leak claim, stated so it can fail.

        Two statements, and both must hold for any value of `leak_bytes`:

        1. **Emptying the retained records and collecting never made RSS materially
           worse.** That is the direction that says the retention explains the growth. The
           tolerance is **8 MiB**, and it is a measurement of the instrument rather than a
           relaxation of the property:

           - What this asserts on is the **difference between two RSS readings**, and after
             M4 Task 3 the retention it empties is **seven `_Transition` records** — the
             retired queues are released during the run. Freeing seven small mappings cannot
             be expected to move RSS down, so at this scale the difference is page noise in
             either direction. Measured in this session: **0 inversions in 6 isolated runs**,
             and **1 inversion in 2 full-suite runs**, by **+1.4 MB**, with *both* readings
             negative (the process was giving memory back, and the "release" gave slightly
             less).
           - The instrument contributes to it: `len(gc.get_objects())` allocates a list of
             every tracked object. Measured: **176 KiB, retained**, for a census of 9 852
             objects; the suite's process tracks ~50 000. `run_stress` now reads RSS before
             that census, which removes that part, and this tolerance covers the rest.
           - **A real retention is tens of MiB, not one**: measured before the fix, the
             ledger held **154.1 MiB** that emptying it released none of. The tolerance is
             5% of that, so the regression this guards cannot hide inside it.

           Measured on 2026-10-03 with six toggles: `leak_bytes` 8192 and
           `leak_bytes_after_release` **-12345344** — the process gave back 12.3 MB, so
           the growth was not a leak at all and the retention was never the thing holding
           it. RSS here carries the OpenVINO runtime and the allocator's high-water mark,
           so it settles downwards as well as up, and an assertion that only handled growth
           would be asserting a direction this machine does not reliably have.
        2. **RSS did not grow without bound**, measured against the process's own size:
           six toggles must not double it. An absolute byte budget would be a claim about
           this machine's allocator rather than about the supervisor; the ratio is the
           property. **This half is unchanged and strict.**
        """
        result = self.result

        tolerance = 8 * 2**20
        self.assertLessEqual(
            result.leak_bytes_after_release, result.leak_bytes + tolerance,
            f"emptying the retained records and collecting took RSS from "
            f"{result.leak_bytes} B of growth to {result.leak_bytes_after_release} B, "
            f"which is backwards by more than the {tolerance} B this measurement can "
            f"resolve: releasing memory cannot cost memory beyond the allocator's noise")
        self.assertLess(
            result.leak_bytes, result.rss_start,
            f"RSS grew by {result.leak_bytes} B over {result.transitions} transitions on "
            f"a {result.rss_start} B process - that is not a bounded trend")

    def test_the_supervisor_itself_had_nothing_to_complain_about(self):
        # Not "the supervisor said nothing" on trust: the text is asserted to be empty,
        # so a surviving-thread reason or a failed route fails here.
        self.assertEqual(self.result.errors_at_end, (),
                         "the supervisor ended the run with reasons on record")

    def test_the_record_refuses_to_hold_a_number_that_would_make_the_report_a_lie(self):
        """`StressResult` is a report. A report that accepts nonsense is worse than none.

        Each of these is a field the whole exercise turns on, and each has one honest
        value: a negative leak is impossible, a `p95` below the `p50` is a mis-sorted
        percentile, a negative `discarded_results` is a counter that went backwards, and a
        `bounded_by` spelling the supervisor does not emit is not an observation.
        """
        good = dict(
            transitions=6, warm_elapsed_p50=1.5, warm_elapsed_p95=1.9,
            warm_elapsed_max=2.1, over_two_second=1, discarded_results=4,
            rss_start=1000, rss_end=1500, leak_bytes=500, stale_overlays=0,
            mislabelled_devices=0, crashes=0)
        StressResult(**good)  # the honest case must build

        with self.assertRaises(ValueError):
            StressResult(**{**good, "leak_bytes": -1})
        with self.assertRaises(ValueError):
            StressResult(**{**good, "warm_elapsed_p95": 0.5, "warm_elapsed_p50": 1.5})
        with self.assertRaises(ValueError):
            StressResult(**{**good, "discarded_results": -1})
        with self.assertRaises(ValueError):
            StressResult(**{**good, "over_two_second": -1})
        with self.assertRaises(ValueError):
            # A count over the gate cannot exceed the number of transitions measured.
            StressResult(**{**good, "over_two_second": 7})
        with self.assertRaises(ValueError):
            StressResult(**{**good, "transitions": 0})
        with self.assertRaises(ValueError):
            StressResult(**{**good, "stale_overlays": -1})
        with self.assertRaises(ValueError):
            StressResult(**{**good, "mislabelled_devices": -1})
        with self.assertRaises(ValueError):
            StressResult(**{**good, "crashes": -1})

    def test_a_sample_records_a_number_for_each_half_of_the_toggle(self):
        """The three columns, because the middle one hid the gate.

        `begin_transition` is synchronous - terminate/join/terminate the old worker,
        retire its queues, drain, spawn the replacement - and
        `TransitionOutcome.elapsed_s` is stamped after all of it. A record that only kept
        the outcome's figure would report a toggle of 5.5 s as 3.3 s, which is what
        M0-M2 did.
        """
        sample = TransitionSample(
            index=0, target={"A": "CPU"}, acknowledged=True, bounded_by="confirmed",
            generation=2, elapsed_s=5.5, transition_elapsed_s=3.3, setup_s=2.2,
            unconfirmed=(), confirmed={"A": ("CPU",)}, discarded_results=1,
            pids_at_confirm={"A": 11}, pids_after={"A": 11}, rss_before=100)
        self.assertAlmostEqual(sample.elapsed_s,
                               sample.setup_s + sample.transition_elapsed_s, places=6,
                               msg="the total must be the two halves, or the columns "
                                   "disagree with each other")
        # And the tool's own percentiles over a list of samples must bracket it.
        values = [s.elapsed_s for s in (sample, sample)]
        self.assertEqual(max(values), 5.5)
        self.assertEqual(sorted(values)[-1], 5.5)


class TheDetectorActuallyDetects(unittest.TestCase):
    """**A "zero stale overlays" assertion passes vacuously unless the detector is shown
    to fire.** Mutant S2 - which turns the stale check off entirely - survived the first
    version of this file, because `assertEqual(result.stale_overlays, 0)` cannot tell
    "nothing stale happened" from "nothing was ever checked". So the check is driven here
    with results whose answer is known.

    Both directions are pinned, and the second one is the interesting one: a frame
    submitted before a toggle, answered by a stream the toggle did **not** move, is an
    in-flight frame and not a stale overlay. Dropping the generation half of the condition
    (mutant S1) reported three of those in a four-transition run, all with
    `frame_id == mark` exactly, all perfectly current.
    """

    def _observer(self, *, generation=2, mark_generation=1, mark=40, pid=4242):
        observer = rs._Observer(_StubSupervisor(generation=generation, pid=pid))
        observer.set_marks({"A": mark}, {"A": mark_generation})
        return observer

    def _result(self, frame_id, devices=("GPU.0",), requested="GPU", error=None):
        frame = _frame(frame_id, stream="A")
        if error is not None:
            return InferenceResult.failure(frame, error, requested)
        return InferenceResult(frame=frame, detections=(), latency_ms=1.0,
                               execution_devices=tuple(devices), error=None,
                               requested_device=requested)

    def test_a_pre_toggle_frame_from_a_superseded_worker_is_a_stale_overlay(self):
        observer = self._observer()
        observer._inspect(self._result(40))
        self.assertEqual(len(observer.stale_overlays), 1,
                         f"frame 40 is at the mark and its stream's generation moved "
                         f"1 -> 2, so the worker that answered it has been replaced. "
                         f"Nothing was recorded: {observer.stale_overlays!r}")
        self.assertEqual(observer.stale_overlays[0]["frame_id"], 40)

    def test_a_pre_toggle_frame_from_an_unmoved_stream_is_not_a_stale_overlay(self):
        # Camera A is on the CPU in all four spec rows, so it never moves and every result
        # for it carries a pre-toggle frame id. Calling those stale would have made the
        # tool's headline number a count of in-flight frames.
        observer = self._observer(generation=1, mark_generation=1)
        observer._inspect(self._result(40, devices=("CPU",), requested="CPU"))
        self.assertEqual(observer.stale_overlays, [],
                         f"the stream's generation did not move, so this frame is in "
                         f"flight and current: {observer.stale_overlays!r}")

    def test_a_post_toggle_frame_from_a_moved_stream_is_not_a_stale_overlay(self):
        observer = self._observer()
        observer._inspect(self._result(41))
        self.assertEqual(observer.stale_overlays, [],
                         f"frame 41 is above the mark, so it was submitted after the "
                         f"toggle: {observer.stale_overlays!r}")

    def test_a_result_naming_another_device_is_a_mislabelled_device(self):
        observer = self._observer()
        observer._inspect(self._result(41, devices=("CPU",), requested="GPU"))
        self.assertEqual(len(observer.mislabelled), 1,
                         f"a GPU route answered with {('CPU',)} must be counted: "
                         f"{observer.mislabelled!r}")
        # And the same result on the right device is not, so the check is not a constant.
        observer._inspect(self._result(42))
        self.assertEqual(len(observer.mislabelled), 1,
                         "a correct placement was counted as mislabelled")

    def test_an_error_result_is_neither_stale_nor_mislabelled(self):
        # A failure carries no `EXECUTION_DEVICES`, so judging it by the device would
        # count every worker death as a mislabelled placement, and the count would stop
        # meaning anything.
        observer = self._observer()
        observer._inspect(self._result(40, error="the plugin said no"))
        self.assertEqual(observer.mislabelled, [])
        self.assertEqual(observer.stale_overlays, [])
        self.assertEqual(observer.error_results, 1)

    def test_the_first_delivery_that_reads_the_device_back_records_its_own_pid(self):
        observer = self._observer(pid=4242)
        observer._inspect(self._result(41))
        self.assertEqual(observer.first_confirm_pid("A", 2), 4242,
                         "the pid that carried EXECUTION_DEVICES for this generation")
        # A different generation has not been proved, so it must not borrow the record.
        self.assertIsNone(observer.first_confirm_pid("A", 1),
                          "generation 1's proof cannot stand in for generation 2's")
        self.assertIsNone(observer.first_confirm_pid("B", 2),
                          "another stream's proof is not this stream's")


class _StubSupervisor:
    """The four things `_Observer` reads off a supervisor, and nothing else.

    Admitted deliberately: the detector has to be driven with results whose answer is
    known, and a real supervisor's timing is not an answer anybody can write down. This
    stub makes "the generation is 2 and the pid is 4242" a statement rather than a hope.
    """

    def __init__(self, *, generation: int = 2, pid: int = 4242) -> None:
        self._generation = generation
        self._pid = pid

    @property
    def stream_ids(self):
        return ("A",)

    def stream_stats(self, stream_id):
        return _StubStats(self._generation)

    def pid_for(self, stream_id):
        return self._pid


class _StubStats:
    def __init__(self, generation: int) -> None:
        self.generation = generation
        self.confirmed_devices = ("GPU.0",)
        self.requested_device = "GPU"


class TransitionSampleRefusesNonsense(unittest.TestCase):
    """The per-sample record, checked the same way `StressResult` is.

    Separate because the two record different things: `StressResult` is the run and
    `TransitionSample` is one toggle. Mutants S9 and S10 removed two of the sample's
    refusals and survived until these existed.
    """

    def _sample(self, **overrides):
        base = dict(index=0, target={"A": "GPU"}, acknowledged=True,
                    bounded_by="confirmed", generation=2, elapsed_s=5.5,
                    transition_elapsed_s=3.3, setup_s=2.2, unconfirmed=(),
                    confirmed={"A": ("GPU.0",)}, discarded_results=1,
                    pids_at_confirm={"A": 11}, pids_after={"A": 11}, rss_before=100)
        base.update(overrides)
        return TransitionSample(**base)

    def test_the_honest_sample_builds(self):
        self._sample()

    def test_it_refuses_to_claim_acknowledged_without_confirming(self):
        with self.assertRaises(ValueError):
            self._sample(acknowledged=True, bounded_by="timeout")

    def test_it_refuses_acknowledged_with_no_confirmed_stream(self):
        with self.assertRaises(ValueError):
            self._sample(confirmed={})

    def test_it_refuses_a_negative_time_and_a_negative_count(self):
        for field, value in (("elapsed_s", -0.1), ("setup_s", -1.0),
                             ("transition_elapsed_s", -2.0), ("discarded_results", -1),
                             ("index", -1), ("rss_before", -5)):
            with self.assertRaises(ValueError, msg=field):
                self._sample(**{field: value})

    def test_it_refuses_a_bounded_by_the_supervisor_does_not_emit(self):
        for spelling in ("done", "confirmed ", "", None):
            with self.assertRaises(ValueError, msg=repr(spelling)):
                self._sample(bounded_by=spelling, acknowledged=spelling == "confirmed")

    def test_it_refuses_a_time_that_is_not_a_number(self):
        for bad in ("1.0", None, True):
            with self.assertRaises(TypeError, msg=repr(bad)):
                self._sample(elapsed_s=bad)


class StressRunIsSerializable(unittest.TestCase):
    """`to_json` is how the record reaches `logs/qualification/`.

    `json.dumps` is the consumer, so anything it cannot encode is a field that does not
    get published - and a field that does not get published is a field nobody checked.
    """

    def test_the_record_json_encodes_with_no_asdict_workarounds(self):
        import json

        payload = StressResult(
            transitions=1, warm_elapsed_p50=1.0, warm_elapsed_p95=1.0,
            warm_elapsed_max=1.0, over_two_second=0, discarded_results=0,
            rss_start=1, rss_end=2, leak_bytes=1, stale_overlays=0,
            mislabelled_devices=0, crashes=0,
            warm_elapsed_min=1.0, warm_elapsed_samples=(1.0,), acknowledged=1,
            unconfirmed_samples=(), timeout_s=2.0, censored_at_gate=False,
            retained_transitions_start=0, retained_transitions_end=1,
            retained_queues_start=0, retained_queues_end=2,
            leak_bytes_after_release=0, rss_peak=2, leak_bytes_per_transition=1.0,
            placements_confirmed=1, placement_pids_matched=1,
            discarded_mismatched=0, frames_submitted=1, frames_refused=0, drops=0,
            results_delivered=1, error_results=0, per_stream=(),
            frame_age_p50_ms=1.0, frame_age_p95_ms=2.0,
            samples=(TransitionSample(
                index=0, target={"A": "CPU"}, acknowledged=True,
                bounded_by="confirmed", generation=2, elapsed_s=1.0,
                transition_elapsed_s=1.0, setup_s=0.0, unconfirmed=(),
                confirmed={"A": ("CPU",)}, discarded_results=0,
                pids_at_confirm={"A": 1}, pids_after={"A": 1}, rss_before=1),))
        text = json.dumps(payload.to_json())
        self.assertIn('"transitions": 1', text)
        self.assertIn('"leak_after_retention"', text,
                      "the retention-adjusted figure must be in the published record")
        # Every field of the frozen dataclass must be published, or a number nobody reads
        # is a number nobody checks.
        published = set(payload.to_json())
        declared = {f.name for f in dataclasses.fields(StressResult)}
        self.assertEqual(declared - published, set(),
                         f"declared but not published: {sorted(declared - published)}")
        # Frozen, so a published record cannot be edited into a different run.
        with self.assertRaises(dataclasses.FrozenInstanceError):
            payload.transitions = 2  # type: ignore[misc]


if __name__ == "__main__":  # pragma: no cover - convenience for a human at a booth
    unittest.main()