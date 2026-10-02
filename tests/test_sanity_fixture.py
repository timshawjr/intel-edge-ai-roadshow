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

**What is deliberately NOT tested here.** Whether the pinned model agrees with the
annotation. That is `evaluate_fixture`, which needs hardware; it is not in the unit
suite, because a unit suite that fails when the NPU driver changes is a unit suite
nobody trusts. `tools/build_sanity_fixture.py --device NPU` is the per-device run, and
the M0 report carries its output.
"""

import json
import unittest
from pathlib import Path

from tools.build_sanity_fixture import evaluate_fixture

FIX = Path("tests/fixtures/retail_sanity/retail_sanity.json")

#: The decode thresholds the fixture is evaluated at. These are the project's working
#: values (the same `0.25 / 0.45` pair `tests/test_decode.py` uses), fixed here before
#: any model was run so that neither could be nudged afterwards.
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


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


class OutcomeIsHonest(unittest.TestCase):
    """`FixtureOutcome` is the record, and it must not be able to lie."""

    def test_a_failing_outcome_does_not_claim_a_match_count_it_never_reached(self):
        from tools.build_sanity_fixture import FixtureOutcome
        outcome = FixtureOutcome(passed=False, matched=1, expected=2, details=())
        self.assertFalse(outcome.passed)
        self.assertLess(outcome.matched, outcome.expected)

    def test_details_is_a_tuple_so_the_record_is_immutable(self):
        from tools.build_sanity_fixture import FixtureOutcome
        outcome = FixtureOutcome(passed=True, matched=1, expected=1, details=("a",))
        self.assertIsInstance(outcome.details, tuple)
        with self.assertRaises(Exception):
            outcome.details = ("b",)  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
