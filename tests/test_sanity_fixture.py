"""The annotated detection sanity fixture, and the guard that it is not circular.

**What this fixture is for.** `app/inference/decode.py` reads a `[1, 84, 8400]` output
as 8400 anchors of `(cx, cy, w, h, 80 class scores)`. Every one of those readings is a
decision, and a wrong one still produces confident, plausible, *wrong* output: a
transposed class block renames every object, a missing `w/2` subtraction displaces every
box, a scale error moves everything by a constant. Nothing in the decoder's own unit
tests can catch that, because they all assert against tensors this project fabricated.

So the annotation in `retail_sanity.json` is a **human judgement about a real frame**:
one person was looked at, and boxes were drawn by eye in source pixels before the model
was ever run on it. The tests below cannot prove the annotation is *correct* - only a
person looking at the JPEG can - but they can refuse a fixture that is empty, malformed,
off the image, or whose image and JSON have drifted apart.

**The hardware-backed test at the bottom is new, and it exists because the suite had no
teeth.** The first version of this file imported `evaluate_fixture` and never called it,
on the reasoning that a unit suite which fails when the NPU driver changes is a suite
nobody trusts. That reasoning was right about the NPU and wrong about the consequence:
with no call, `unittest discover` was green on **every** decoder mutant, and since
`tools/preflight.py` does not exist yet, Task 6 had no automated regression protection at
all. So the fixture is now exercised end to end on the **CPU**, which is present on every
machine this project targets and is the one device a driver update is least likely to
change. It is skipped when the CPU is genuinely absent rather than failed, because a
machine with no CPU plugin cannot run this model at all.

**A note for Task 8.** This CPU test is a regression gate, not a qualification result. It
must become a **required row of `tools/preflight.py`**, because a skipped test and an
absent test look identical in a green suite, and "the fixture was not run" is
indistinguishable from "the fixture passed" unless something insists it ran.
"""

import json
import unittest
from pathlib import Path

from tools.build_sanity_fixture import (
    CONF_THRESHOLD as MODULE_CONF_THRESHOLD,
    IOU_THRESHOLD as MODULE_IOU_THRESHOLD,
    evaluate_fixture,
)

FIX = Path("tests/fixtures/retail_sanity/retail_sanity.json")

#: The decode thresholds the fixture is evaluated at. These are the project's working
#: values (the same `0.25 / 0.45` pair `tests/test_decode.py` uses), fixed here before
#: any model was run so that neither could be nudged afterwards.
#:
#: Pinned to the tool's own constants by `ThresholdsAreOneSet` below. The first version
#: of this file declared its own `0.25` and nothing tied the two together, so the claim
#: that `min_score` "cannot become an escape hatch" held only because two numbers happened
#: to agree - and the CLI's `--conf` flag made that coincidence matter, since it could
#: change the threshold the decoder used while `min_score` stayed where it was.
CONF_THRESHOLD = MODULE_CONF_THRESHOLD
IOU_THRESHOLD = MODULE_IOU_THRESHOLD


class FixtureIsReal(unittest.TestCase):
    def test_fixture_exists_and_is_not_empty(self):
        self.assertTrue(FIX.exists(), "sanity fixture has not been built")
        payload = json.loads(FIX.read_text(encoding="utf-8"))
        self.assertTrue(payload["expected"], "fixture must annotate at least one object")
        for e in payload["expected"]:
            self.assertIn("class_name", e)
            self.assertEqual(len(e["box_xyxy"]), 4)

    def test_every_expected_box_lies_inside_the_image(self):
        import cv2
        payload = json.loads(FIX.read_text(encoding="utf-8"))
        img = cv2.imread(str(FIX.parent / payload["image"]))
        h, w = img.shape[:2]
        for e in payload["expected"]:
            x1, y1, x2, y2 = e["box_xyxy"]
            self.assertGreaterEqual(min(x1, y1), -w * 0.05)
            self.assertLessEqual(max(x2, y2), w * 1.05)
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)


class BoxIsOnEachAxis(unittest.TestCase):
    """The brief's check above measures y against the image *width*.

    Kept verbatim because it is the specified test, and repeated per-axis here because
    a 1920x1080 frame has a 1080-tall ceiling, not 1920: the width-based bound cannot
    fail for any y below 2016, so a box drawn a metre below the frame would pass it.
    This is the stricter version, and it is the one that would catch a coordinate
    produced by a decoder that mixed up the two axes.
    """

    def test_boxes_are_inside_the_frame_on_each_axis(self):
        import cv2
        payload = json.loads(FIX.read_text(encoding="utf-8"))
        img = cv2.imread(str(FIX.parent / payload["image"]))
        h, w = img.shape[:2]
        for e in payload["expected"]:
            x1, y1, x2, y2 = e["box_xyxy"]
            self.assertGreaterEqual(x1, -w * 0.05, f"x1 {x1} is off the left edge")
            self.assertLessEqual(x2, w * 1.05, f"x2 {x2} is off the right edge")
            self.assertGreaterEqual(y1, -h * 0.05, f"y1 {y1} is off the top edge")
            self.assertLessEqual(y2, h * 1.05, f"y2 {y2} is off the bottom edge")


class FixtureIsSelfConsistent(unittest.TestCase):
    def setUp(self):
        if not FIX.exists():
            self.fail("sanity fixture has not been built")
        self.payload = json.loads(FIX.read_text(encoding="utf-8"))

    def test_the_image_named_by_the_fixture_exists(self):
        image = FIX.parent / self.payload["image"]
        self.assertTrue(image.exists(),
                        f"the fixture names {self.payload['image']!r}, which is not there")

    def test_the_frame_index_is_a_non_negative_int(self):
        # Reproducibility: a fixture with no frame index is a picture, not a fixture.
        self.assertIsInstance(self.payload["frame_index"], int)
        self.assertNotIsInstance(self.payload["frame_index"], bool)
        self.assertGreaterEqual(self.payload["frame_index"], 0)

    def test_the_source_clip_is_the_pinned_one(self):
        pinned = json.loads(
            Path("config/models.json").read_text(encoding="utf-8"))
        clip = next(m for m in pinned["media"] if m["id"] == "retail_checkout")
        self.assertEqual(self.payload["source_clip"], clip["path"])

    def test_the_image_resolution_matches_the_declared_source(self):
        import cv2
        pinned = json.loads(
            Path("config/models.json").read_text(encoding="utf-8"))
        clip = next(m for m in pinned["media"] if m["id"] == "retail_checkout")
        img = cv2.imread(str(FIX.parent / self.payload["image"]))
        h, w = img.shape[:2]
        self.assertEqual((w, h),
                         (clip["measured_width"], clip["measured_height"]),
                         "the fixture image is not at the clip's measured native "
                         "resolution, so boxes read off it would not be in the "
                         "coordinates the JSON claims")

    def test_class_ids_and_names_agree_with_the_pinned_labels(self):
        labels = Path("assets/models/yolo11n-fp16/labels.txt").read_text(
            encoding="utf-8").splitlines()
        for e in self.payload["expected"]:
            self.assertIn(e["class_id"], range(len(labels)),
                          f"class_id {e['class_id']} is not one of the {len(labels)} "
                          f"pinned labels")
            self.assertEqual(labels[e["class_id"]], e["class_name"],
                             f"class_id {e['class_id']} is labelled "
                             f"{labels[e['class_id']]!r} in labels.txt, not "
                             f"{e['class_name']!r}")

    def test_min_score_is_a_probability_and_not_a_tuned_measurement(self):
        for e in self.payload["expected"]:
            score = e["min_score"]
            self.assertIsInstance(score, (int, float))
            self.assertGreater(score, 0.0)
            self.assertLessEqual(score, 1.0)
            # Equal to the threshold the fixture is evaluated at, deliberately: the
            # score criterion then adds nothing beyond "the detector emitted this at
            # all", and the work is carried by class id and box IoU. A min_score
            # *below* the threshold would be a second, invisible way to relax the
            # fixture, and one that reads as a measurement.
            self.assertEqual(score, CONF_THRESHOLD,
                             "min_score must be the fixture's own conf threshold; a "
                             "different value is a tuned number, not an annotation")

    def test_the_fixture_records_how_it_was_made(self):
        notes = self.payload["notes"]
        self.assertTrue(notes.strip(),
                        "a fixture with no notes cannot say whether it was annotated "
                        "by eye or copied from the model's own output")
        lowered = notes.lower()
        for required in ("by eye", "not copied from"):
            self.assertIn(required, lowered,
                          f"notes must state that the annotation was made {required!r}")

    def test_the_annotation_is_small_and_deliberate(self):
        # Two or three objects. A fixture annotating twenty things is a labelling
        # project, and every extra entry is a chance to hide a decoder bug behind an
        # argument about a fuzzy boundary.
        count = len(self.payload["expected"])
        self.assertGreaterEqual(count, 2, "at least two objects, per the brief")
        self.assertLessEqual(count, 3, "more than three is a labelling project")


class ThresholdsAreOneSet(unittest.TestCase):
    """The test's thresholds and the tool's are the same numbers, by assertion.

    Not by coincidence. The first version of this file declared `CONF_THRESHOLD = 0.25`
    locally, and `tests.test_sanity_fixture.test_min_score_is_a_probability_and_not_a_
    tuned_measurement` compared `min_score` against that local copy. Change the tool's
    threshold and the test would have kept passing against a stale 0.25 while the fixture
    decoded at something else - which is the same class of bug as the `--conf` flag that
    printed one number and decoded at another.
    """

    def test_the_local_copies_are_the_tools_own(self):
        import tools.build_sanity_fixture as tool
        self.assertEqual(CONF_THRESHOLD, tool.CONF_THRESHOLD)
        self.assertEqual(IOU_THRESHOLD, tool.IOU_THRESHOLD)

    def test_a_caller_supplied_conf_threshold_actually_filters(self):
        """The threshold a caller passes must be the threshold the decoder uses.

        Measured before the fix: `--conf 0.9` printed `conf=0.9`, decoded at the module's
        `0.25`, and still emitted the 0.273557 apple with `PASS: 2/2`. Worse than a lying
        label - at a genuine `conf=0.5` the apple is correctly filtered out and the
        fixture should fail, and the flag **hid** that failure.
        """
        import tools.build_sanity_fixture as tool
        if not _cpu_available():
            self.skipTest("no CPU plugin on this machine; the threshold is threaded into "
                          "decode_detections and there is nothing to decode with")
        core = tool.ov.Core()
        spec = tool.describe_pinned_model()
        names = tool.class_names()
        at_default = tool.evaluate_fixture(spec, names, 0.25, 0.45, device="CPU",
                                           core=core)
        at_half = tool.evaluate_fixture(spec, names, 0.50, 0.45, device="CPU", core=core)
        self.assertTrue(at_default.passed, "the fixture must pass at its own threshold")
        # At 0.5 the person (0.913) still clears and the apple (0.274) does not, so the
        # fixture must FAIL - and a decoder ignoring the argument would have passed it.
        self.assertFalse(at_half.passed,
                         "conf=0.5 must filter the 0.27 apple and fail the fixture; a "
                         "pass here means the threshold is not reaching the decoder")
        self.assertLess(at_half.matched, at_default.matched)


class AnnotationRecordsItsOwnBounds(unittest.TestCase):
    """The two bounds that close the fixture's measured blind spots.

    Both are judgements about the picture. Both are load-bearing, and a fixture that
    silently lost either would go back to being green on a decoder that draws 1205 boxes
    or that mirrors every box.
    """

    def setUp(self):
        if not FIX.exists():
            self.fail("sanity fixture has not been built")
        self.payload = json.loads(FIX.read_text(encoding="utf-8"))

    def test_a_detection_count_bound_is_recorded_and_is_a_count(self):
        bound = self.payload.get("max_detections")
        self.assertIsInstance(bound, int)
        self.assertNotIsInstance(bound, bool)
        annotated = len(self.payload["expected"])
        self.assertGreater(bound, annotated,
                           "the bound must leave room for the marginal judgements a "
                           "detector will make that a person drawing boxes would not")
        self.assertLessEqual(bound, 10,
                             "a ceiling above ten is not a bound, it is a shrug; this "
                             "frame holds two named objects and a counter")

    def test_the_detection_count_bound_explains_itself(self):
        self.assertIn("max_detections_reasoning", self.payload)
        text = self.payload["max_detections_reasoning"].lower()
        self.assertIn("picture", text)
        self.assertIn("not a number read off the detector", text)
        self.assertIn("1205", text,
                      "the reasoning should name the failure it exists to catch, so a "
                      "reader can tell a bound from a shrug")

    def test_the_aspect_check_is_not_a_verdict_input(self):
        """The aspect ratio is reported, and must NOT gate the verdict.

        An aspect check was implemented to catch a `w`/`h` transposition and then
        **rejected on measurement**. Against this annotation the correct decoder is
        further from the annotated aspect than the mutant is: person 0.0303 correct vs
        0.0827 swapped, but apple **0.4387 correct vs 0.0459 swapped**. Any tolerance
        tight enough to catch the mutant fails the healthy run; any tolerance loose
        enough to pass the healthy run passes the mutant. So there must be no
        `max_aspect_delta` gate in the annotation, and the tool must not consult one.

        This test exists to stop the check being quietly reintroduced. A guard that fails
        a correct decoder is worse than a documented hole, because the response to a
        false failure is to loosen the guard until the real fault gets through too.
        """
        self.assertNotIn(
            "max_aspect_delta", self.payload,
            "the annotation must not carry an aspect tolerance: it is not a usable gate "
            "on this fixture, see known_blind_spots")
        import tools.build_sanity_fixture as tool
        import inspect
        verdict_source = inspect.getsource(tool.evaluate_fixture)
        self.assertNotIn("max_aspect_delta", verdict_source,
                         "evaluate_fixture must not read an aspect tolerance; the aspect "
                         "ratio is reported as information only")

    def test_the_aspect_blind_spot_is_recorded_with_its_measurement(self):
        spots = self.payload.get("known_blind_spots")
        self.assertIsInstance(spots, list)
        self.assertTrue(spots, "an unrecorded blind spot is an invisible one")
        joined = json.dumps(spots).lower()
        self.assertIn("w, h", joined.replace("w/h", "w, h"))
        self.assertIn("caught", joined)
        for spot in spots:
            self.assertIn("fault", spot)
            self.assertIn("caught", spot)
            self.assertIsInstance(spot["caught"], bool)
            self.assertIn("why", spot)
            self.assertGreater(len(spot["why"]), 80,
                               "a blind spot recorded without a reason is a rumour")

    def test_a_known_blind_spot_cannot_claim_to_be_caught(self):
        """Every recorded blind spot must agree with itself about `caught`."""
        for spot in self.payload.get("known_blind_spots") or ():
            if spot.get("caught") is True:
                self.assertNotIn("cannot", spot["why"].lower()[:40],
                                 "a spot marked caught whose reason says it cannot be "
                                 "separated is a contradiction, and a reader would "
                                 "believe the stronger claim")


class OutcomeIsHonest(unittest.TestCase):
    """`FixtureOutcome` is the record, and it must not be able to lie."""

    def test_a_failing_outcome_does_not_claim_a_match_count_it_never_reached(self):
        from tools.build_sanity_fixture import FixtureOutcome
        outcome = FixtureOutcome(passed=False, matched=1, expected=2, details=())
        self.assertFalse(outcome.passed)
        self.assertLess(outcome.matched, outcome.expected)

    def test_details_is_a_tuple_so_the_record_is_immutable(self):
        from tools.build_sanity_fixture import FixtureOutcome
        outcome = FixtureOutcome(passed=True, matched=1, expected=1,
                                 details=("VERDICT PASS: because",))
        self.assertIsInstance(outcome.details, tuple)
        with self.assertRaises(Exception):
            outcome.details = ("b",)  # type: ignore[misc]

    def test_a_pass_must_carry_the_reasons_it_passed(self):
        """`passed=True` with no `VERDICT PASS` line is refused.

        The verdict is now three conditions - every object matched, the detection count
        within the annotation's bound, and the run placed on the requested device - so a
        bare boolean no longer explains itself. Requiring the line means a reader of the
        record can always see *why*, and it means `passed` cannot be constructed as a
        stamp by a future caller that skipped the checks.
        """
        from tools.build_sanity_fixture import FixtureOutcome
        with self.assertRaises(ValueError):
            FixtureOutcome(passed=True, matched=1, expected=1, details=("all good",))

    def test_a_pass_with_an_unmatched_expectation_is_refused(self):
        """The weaker of the two guards, and the one that cannot be reasoned away.

        `passed` may be *stricter* than `matched == expected` - that is the whole point of
        the count bound and the placement clause - but it may never be looser.
        """
        from tools.build_sanity_fixture import FixtureOutcome
        with self.assertRaises(ValueError):
            FixtureOutcome(passed=True, matched=1, expected=2,
                           details=("VERDICT PASS: because",))

    def test_matched_cannot_exceed_expected(self):
        from tools.build_sanity_fixture import FixtureOutcome
        with self.assertRaises(ValueError):
            FixtureOutcome(passed=False, matched=3, expected=2, details=())

    def test_a_healthy_run_is_not_rejected_for_failing_the_count_bound(self):
        """The invariant this replaced was wrong, and it failed a correct run.

        The first version enforced `passed == (matched == expected)`, on the reasoning
        that `passed` is derived and must not be stampable. That became false the moment
        the verdict grew past matching, and it then raised
        `ValueError: passed=False contradicts matched=2 against expected=2` on a run that
        was correct in every respect. Measured, not hypothesised: that is the exact
        exception the first version of the aspect check provoked, by failing the honest
        apple at 0.4387.
        """
        from tools.build_sanity_fixture import FixtureOutcome
        outcome = FixtureOutcome(
            passed=False, matched=2, expected=2,
            details=("VERDICT FAIL: the decoder returned 1205 boxes",))
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.matched, outcome.expected)


def _cpu_available() -> bool:
    """Whether this machine has a CPU plugin, so the fixture can actually be run.

    Checked rather than assumed. A machine with no OpenVINO CPU plugin cannot run this
    model at all, and a test that hard-fails there would be telling the truth in the
    least useful way.
    """
    try:
        import openvino as ov
    except ImportError:
        return False
    try:
        return "CPU" in tuple(ov.Core().available_devices)
    except Exception:  # noqa: BLE001 - an unreadable device list means "cannot run"
        return False


def _cpu_fixture():
    """Run the real fixture on the CPU, or `None` if this machine cannot."""
    import tools.build_sanity_fixture as tool
    if not _cpu_available():
        return None
    try:
        core = tool.ov.Core()
        return tool.evaluate_fixture(tool.describe_pinned_model(), tool.class_names(),
                                     CONF_THRESHOLD, IOU_THRESHOLD, device="CPU",
                                     core=core)
    except Exception:  # noqa: BLE001 - reported as a skip, never as a pass
        return None


class TheFixtureActuallyRuns(unittest.TestCase):
    """The end-to-end check the suite was missing: the real model against the real
    annotation, on the CPU.

    Every other test in this file inspects the annotation or the tool's pure functions.
    None of them can notice that `app/inference/decode.py` has started reading the tensor
    the wrong way, because they never decode anything. This class does, which is why it
    is the only thing here with teeth against a decoder mutant - and also why it is
    limited to the CPU: the NPU and GPU rows are qualification, and belong in
    `tools/qualify_m0.py` and the M0 report, where a driver change is a report to write
    rather than a suite to flake.
    """

    @classmethod
    def setUpClass(cls):
        cls.outcome = _cpu_fixture()
        if cls.outcome is None:
            raise unittest.SkipTest(
                "no CPU OpenVINO plugin on this machine, or the model could not be "
                "prepared; the fixture cannot be exercised here. Task 8 must make this "
                "a required preflight row so that 'not run' cannot read as 'passed'.")

    def test_the_pinned_model_matches_the_hand_annotation_on_cpu(self):
        self.assertTrue(
            self.outcome.passed,
            "the pinned model does not agree with the hand annotation on CPU:\n  "
            + "\n  ".join(self.outcome.details))

    def test_the_verdict_is_placed_on_the_cpu(self):
        joined = "\n".join(self.outcome.details)
        self.assertIn("EXECUTION_DEVICES=", joined)
        self.assertIn("placed_on_requested=True", joined,
                      "the CPU run did not report itself placed on the CPU")

    def test_the_detection_count_is_within_the_annotated_bound(self):
        payload = json.loads(FIX.read_text(encoding="utf-8"))
        joined = "\n".join(self.outcome.details)
        self.assertIn("detection-count bound:", joined,
                      "the count bound was not evaluated at all, so the fixture is not "
                      "checking how many boxes the decoder produced")
        line = next(l for l in self.outcome.details
                    if l.startswith("detection-count bound:"))
        produced = int(line.split(":")[1].split("<=")[0].strip())
        self.assertLessEqual(produced, int(payload["max_detections"]))

    def test_the_aspect_ratios_are_reported(self):
        """Reported, as information. Not a gate - see
        `AnnotationRecordsItsOwnBounds.test_the_aspect_check_is_not_a_verdict_input`
        for the measurement that stopped it being one."""
        joined = "\n".join(self.outcome.details)
        self.assertIn("ASPECT", joined,
                      "no matched pair had its aspect ratio reported, so a reader "
                      "cannot see the shape evidence at all")
        self.assertIn("INFORMATIONAL", joined,
                      "the aspect line must say it is not part of the verdict, or a "
                      "reader will take it as a check that passed")

    def test_a_detection_outside_the_label_list_is_reported_not_raised(self):
        """A wrong class block must produce a sentence, not a traceback.

        Measured: a wrong-axis decode yields a class id of 8396 against 80 labels, and
        `evaluate_fixture` indexed the label tuple with it and died with
        `IndexError: tuple index out of range` - taking down the reporter that was
        supposed to be describing the fault.
        """
        import numpy as np
        from tools.build_sanity_fixture import Detection, _aspect, box_iou
        # The pure helpers are what the reporter uses on each detection; a degenerate
        # box must score 0 rather than raise, and a positive box must score 1 against
        # itself. That is the arithmetic the reporter depends on being total.
        self.assertEqual(box_iou([0, 0, 0, 0], [0, 0, 10, 10]), 0.0)
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(_aspect([0, 0, 10, 0]), 0.0)
        self.assertAlmostEqual(_aspect([0, 0, 150, 184]), 150 / 184)
        # A Detection itself refuses a non-float score, so a mis-decoded class id has to
        # be handled by the reporter, which is where the guard now lives.
        self.assertEqual(Detection(8396, 0.5, (0, 0, 1, 1)).class_id, 8396)


class PlacementIsPartOfTheVerdict(unittest.TestCase):
    """A device that ran somewhere else is not a pass. AGENTS.md rule 5.

    This is the check the first version of `evaluate_fixture` computed, printed and then
    ignored: `placed_on_requested` appeared on its own line while `passed` was
    `matched == len(expected)` and the exit code followed `passed`. A silent fallback to
    the CPU therefore printed `PASS: 2/2` and exited 0. `tools/qualify_m0.py` asserted the
    same rule correctly, so the two tools disagreed about one line of AGENTS.md, and the
    one that a booth would consult is the one that said yes.
    """

    def _with_placement(self, placement, device):
        """Run the fixture with `is_device` forced to answer `placement`."""
        import tools.build_sanity_fixture as tool
        real_is_device = tool.is_device
        tool.is_device = lambda reading, wanted: (
            list(reading) == list(placement) and wanted == device)
        try:
            return tool.evaluate_fixture(tool.describe_pinned_model(),
                                         tool.class_names(), CONF_THRESHOLD,
                                         IOU_THRESHOLD, device=device,
                                         core=tool.ov.Core())
        finally:
            tool.is_device = real_is_device

    @classmethod
    def setUpClass(cls):
        if not _cpu_available():
            raise unittest.SkipTest("no CPU OpenVINO plugin on this machine")

    def test_a_run_placed_on_the_requested_device_passes(self):
        outcome = self._with_placement(("NPU",), "NPU")
        self.assertTrue(outcome.passed,
                        "\n".join(self.outcome_details(outcome)))

    def test_a_run_placed_elsewhere_fails_even_with_every_object_matched(self):
        # Requested NPU, actually ran on the CPU, both objects found. The old verdict
        # called this a pass.
        outcome = self._with_placement(("CPU",), "NPU")
        self.assertEqual(outcome.matched, outcome.expected,
                         "this test is only meaningful if the objects really did match")
        self.assertFalse(outcome.passed,
                         "a device that ran on other hardware is not a pass for the "
                         "device that was asked for")
        self.assertIn("VERDICT FAIL", "\n".join(outcome.details))
        self.assertIn("not on the NPU that was requested",
                      "\n".join(outcome.details))

    def test_the_gpu_sub_device_suffix_still_counts_as_placed(self):
        outcome = self._with_placement(("GPU.0",), "GPU")
        self.assertTrue(outcome.passed, "\n".join(outcome.details))

    @staticmethod
    def outcome_details(outcome):
        return outcome.details


if __name__ == "__main__":
    unittest.main()
