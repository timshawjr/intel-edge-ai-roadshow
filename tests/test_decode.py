"""The decoder: driven by observed model I/O, never by an assumed layout.

Review Focus #3: *"The export's output layout is not the assumed `(1, 84, 8400)`
convention. A wrong transpose yields plausible-looking boxes over the wrong objects."*
Review Focus #4 is Task 4's half of the same round trip: these tests are worthless if
the geometry they invert is itself untested, which is why
`app.inference.preprocess.Letterbox` is consumed rather than reimplemented here.

**Five corrections to the plan's tests, each because the plan's version could not pass
or could not fail. The evidence for each is in the Task 5 report; the short form:**

1. *The plan's synthetic tensor is self-colliding.* It writes
   `arr[0, ax+5, 0] = 320.0` for the box's `cy` and then `arr[0, ax+4+1, 0] = 0.9`,
   which is **the same index** - the score lands on the `cy` coordinate. It also
   fills row `k` with `k` via `np.arange`, so every class row reads 4..83, all far
   above a 0.25 threshold, and `argmax` returns class **79**, not the asserted class 0.
2. *The plan reads the first four values as `xyxy`.* The pinned model writes
   **`cx, cy, w, h`**. Measured first-hand: 15 of 15 above-threshold anchors are
   *inverted* under an `xyxy` reading, and rows 0/1 average 319/320 (coordinates)
   while rows 2/3 average 88/71 (sizes). `CwcywhIsNotXyxy` pins this absolutely,
   because an `xyxy` decoder produces plausible boxes over the wrong objects - the
   exact outcome Review Focus #3 exists to prevent.
3. *The plan keys the tensor by `"output0"`,* which contradicts the same brief's
   correction that the output port is unnamed and the tensor is keyed by **port
   index**. The helper here keys by `"0"`, and
   `TheTensorIsFetchedByPortIndexNotByName` pins that a name-based fetch is not a
   fallback.
4. *The plan's "pad_y 80" for `letterbox_params(1280, 720, 640, 640)` is 720-640,*
   not the leftover. The leftover is `640 - 360 = 280` and half of it is **140**; the
   same rule gives 140 for 1920x1080, measured first-hand.
5. *The plan passes `shapes=()`* to a field it declares as `dict[str, tuple[int, ...]]`.

**And the gap this file closes beyond the plan:** the plan's two-layout pair would pass
for an implementation that ignored `spec.layout` entirely, because its detection sits at
box index 0, which is a valid index on *both* axes. `BothLayoutsAreGenuinelyRead`
puts detections at late box indices with asymmetric coordinates, so a decoder that
does not transpose either raises or returns different boxes.
"""

import json
import unittest
from pathlib import Path

import numpy as np

from app.inference.decode import (
    ModelOutputSpec,
    UnsupportedLayoutError,
    describe_outputs,
    decode_detections,
    nms,
)
from app.inference.preprocess import letterbox_params

NUM_CLASSES = 80
NUM_BOXES = 8400
FEATURES = 4 + NUM_CLASSES  # 84, the 4 box values plus one score per class
MODELS_JSON = Path("config/models.json")
RECORDED_IO = Path("config/model_io/yolo11n-fp16.json")
IR_PATH = Path("assets/models/yolo11n-fp16/yolo11n.xml")


def _tensor(layout, detections, num_classes=NUM_CLASSES, num_boxes=NUM_BOXES):
    """An output tensor in `layout` holding exactly `detections`, and nothing else.

    `detections` is a sequence of `(box_index, cx, cy, w, h, class_id, score)`. The
    four box values are written **cx, cy, w, h** because that is what the pinned model
    writes - measured, not assumed; see the module docstring.

    Note that the two layouts put the box index on **different axes**: axis 2 for
    `yolo_v8_transposed` (`[1, 4+C, N]`) and axis 1 for `yolo_v8_passthrough`
    (`[1, N, 4+C]`). Writing one indexing for both is exactly the bug this module exists
    to prevent, and the first version of this helper did precisely that - caught by
    `test_passthrough_layout_yields_the_same_box` returning 0 detections.
    """
    if layout == "yolo_v8_transposed":
        arr = np.zeros((1, 4 + num_classes, num_boxes), dtype=np.float32)

        def put(feature, box_index, value):
            arr[0, feature, box_index] = value
    elif layout == "yolo_v8_passthrough":
        arr = np.zeros((1, num_boxes, 4 + num_classes), dtype=np.float32)

        def put(feature, box_index, value):
            arr[0, box_index, feature] = value
    else:
        raise ValueError(f"no synthetic tensor defined for layout {layout!r}")
    for box_index, cx, cy, w, h, class_id, score in detections:
        if not 0 <= box_index < num_boxes:
            raise ValueError(f"box_index {box_index} outside 0..{num_boxes - 1}")
        if not 0 <= class_id < num_classes:
            raise ValueError(f"class_id {class_id} outside 0..{num_classes - 1}")
        put(0, box_index, cx)
        put(1, box_index, cy)
        put(2, box_index, w)
        put(3, box_index, h)
        put(4 + class_id, box_index, score)
    return {"0": arr}


#: One confident class-0 detection, deliberately **off-centre and non-square** so that
#: an xyxy reading, a transposed read and a padless inverse all give different answers.
#: Model-space xyxy is (300-40, 260-20, 300+40, 260+20) = (260, 240, 340, 280).
ONE_DETECTION = (0, 300.0, 260.0, 80.0, 40.0, 0, 0.9)
ONE_DETECTION_MODEL_BOX = (260.0, 240.0, 340.0, 280.0)

#: Far apart, asymmetric, at **late** box indices, three different classes. Any two of
#: these being confused for the others changes a returned coordinate.
LATE_DETECTIONS = [
    (4096, 120.0, 480.0, 60.0, 100.0, 0, 0.91),
    (6001, 500.0, 200.0, 200.0, 40.0, 16, 0.62),
    (8399, 330.0, 330.0, 30.0, 70.0, 2, 0.35),
]


def _spec(layout, names=(), shapes=None, num_classes=NUM_CLASSES,
          has_embedded_nms=False, num_boxes=NUM_BOXES):
    """A spec whose recorded shape matches the tensor it will be handed.

    A helper rather than something inlined, so the two cannot drift: a spec whose
    recorded shape disagrees with its tensor is refused by `decode_detections`, which is
    the point, but it must not make every test in this file fail for that reason.
    """
    if shapes is None:
        if layout == "yolo_v8_transposed":
            shapes = {"0": (1, 4 + num_classes, num_boxes)}
        elif layout == "yolo_v8_passthrough":
            shapes = {"0": (1, num_boxes, 4 + num_classes)}
        else:
            shapes = {}
    return ModelOutputSpec(
        names=tuple(names),
        shapes=dict(shapes),
        layout=layout,
        num_classes=num_classes,
        has_embedded_nms=bool(has_embedded_nms),
    )


class _BoxAssertions(unittest.TestCase):
    def assertBoxAlmostEqual(self, got, want, places=4):
        got = tuple(round(float(v), places) for v in got)
        want = tuple(round(float(v), places) for v in want)
        self.assertEqual(len(got), len(want))
        for axis, (a, b) in enumerate(zip(got, want)):
            self.assertAlmostEqual(a, b, places=places, msg=f"axis {axis}: {got} != {want}")

    def assertDetectionsEqual(self, got, want):
        self.assertEqual(len(got), len(want), f"{len(got)} detections, expected {len(want)}")
        for a, b in zip(got, want):
            self.assertEqual(a.class_id, b.class_id)
            self.assertAlmostEqual(a.score, b.score, places=6)
            self.assertBoxAlmostEqual(a.box, b.box, places=3)


class CwcywhIsNotXyxy(_BoxAssertions):
    """The four box values are a centre and a size, and the difference is the whole
    drawing. Under an `xyxy` reading a 60x100 person at (120,480) becomes the
    quadrilateral (120,480)-(60,100): plausible, confident, and on nothing."""

    def test_the_centre_and_size_become_corners(self):
        spec = _spec("yolo_v8_transposed")
        lb = letterbox_params(640, 640, 640, 640)  # scale 1, no pad
        dets = decode_detections(spec, _tensor("yolo_v8_transposed", [ONE_DETECTION]),
                                 lb, 0.25, 0.45)
        self.assertEqual(len(dets), 1)
        self.assertBoxAlmostEqual(dets[0].box, ONE_DETECTION_MODEL_BOX)
        # The xyxy reading would have returned the raw four values unchanged.
        self.assertNotAlmostEqual(dets[0].box[0], 300.0, places=1)
        self.assertNotAlmostEqual(dets[0].box[2], 80.0, places=1)

    def test_a_negative_half_extent_reaches_past_the_centre(self):
        # cx=200, w=100 -> the box starts at 150, which is to the LEFT of the centre.
        # An xyxy reading would put x1 at 200 and x2 at 100, i.e. inverted.
        det = (0, 200.0, 500.0, 100.0, 20.0, 0, 0.8)
        spec = _spec("yolo_v8_transposed")
        dets = decode_detections(spec, _tensor("yolo_v8_transposed", [det]),
                                 letterbox_params(640, 640, 640, 640), 0.25, 0.45)
        self.assertBoxAlmostEqual(dets[0].box, (150.0, 490.0, 250.0, 510.0))

    def test_the_class_score_is_not_run_through_a_second_sigmoid(self):
        # Measured first-hand: rows 4..83 of the pinned output are already inside
        # [0, 1] (max 0.909073 over a real frame), so the export carries the
        # activation. A second sigmoid would compress 0.9 to 0.71 and quietly move
        # every score, and the recorded score is the number the booth must show.
        spec = _spec("yolo_v8_transposed")
        dets = decode_detections(
            spec, _tensor("yolo_v8_transposed", [ONE_DETECTION]),
            letterbox_params(640, 640, 640, 640), 0.25, 0.45)
        self.assertAlmostEqual(dets[0].score, 0.9, places=6)


class LayoutIsNotAssumed(CwcywhIsNotXyxy):
    def test_transposed_layout_yields_the_box(self):
        spec = _spec("yolo_v8_transposed")
        lb = letterbox_params(640, 640, 640, 640)
        dets = decode_detections(spec, _tensor("yolo_v8_transposed", [ONE_DETECTION]),
                                 lb, 0.25, 0.45)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].class_id, 0)
        self.assertBoxAlmostEqual(dets[0].box, ONE_DETECTION_MODEL_BOX)

    def test_passthrough_layout_yields_the_same_box(self):
        spec = _spec("yolo_v8_passthrough")
        lb = letterbox_params(640, 640, 640, 640)
        dets = decode_detections(spec, _tensor("yolo_v8_passthrough", [ONE_DETECTION]),
                                 lb, 0.25, 0.45)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].class_id, 0)
        self.assertBoxAlmostEqual(dets[0].box, ONE_DETECTION_MODEL_BOX)

    def test_unknown_layout_yields_nothing_rather_than_guessing(self):
        spec = _spec("unknown")
        lb = letterbox_params(640, 640, 640, 640)
        self.assertEqual(
            decode_detections(spec, _tensor("yolo_v8_passthrough", [ONE_DETECTION]),
                              lb, 0.25, 0.45),
            [],
        )

    def test_boxes_are_returned_in_source_pixels(self):
        # 1280x720 -> scale 0.5, 640x360 -> pad (0, 140). The detection's model-space
        # xyxy is (270, 260, 370, 340), so source is (270/0.5, (260-140)/0.5,
        # 370/0.5, (340-140)/0.5) = (540, 240, 740, 400). All four corners are checked
        # because the pad lives on the y axis only and a box that ignored it would
        # still get x right.
        spec = _spec("yolo_v8_transposed")
        det = (0, 320.0, 300.0, 100.0, 80.0, 0, 0.9)
        dets = decode_detections(spec, _tensor("yolo_v8_transposed", [det]),
                                 letterbox_params(1280, 720, 640, 640), 0.25, 0.45)
        self.assertEqual(len(dets), 1)
        self.assertBoxAlmostEqual(dets[0].box, (540.0, 240.0, 740.0, 400.0))

    def test_a_box_in_the_pad_band_comes_back_outside_the_frame_rather_than_clamped(self):
        # Model y 20 is in the 140-pixel top pad, so no real object is there. The
        # honest answer is a negative source y; a clamp to 0 would put the box on the
        # top edge of the frame and look like a detection at the ceiling.
        spec = _spec("yolo_v8_transposed")
        det = (0, 320.0, 20.0, 40.0, 40.0, 0, 0.9)
        dets = decode_detections(spec, _tensor("yolo_v8_transposed", [det]),
                                 letterbox_params(1280, 720, 640, 640), 0.25, 0.45)
        self.assertLess(dets[0].box[1], 0.0)


class BothLayoutsAreGenuinelyRead(_BoxAssertions):
    """The anti-symmetry guard.

    The plan's pair of layout tests cannot tell a correct decoder from one that
    hardcodes `arr[0, k, box]`, because its detection sits at box index 0, and index 0
    is in range on *both* axes of both layouts. These tests put detections at indices
    4096 / 6001 / 8399, where an always-transposed read of a passthrough tensor raises
    `IndexError` rather than agreeing by accident.
    """

    def test_both_layouts_decode_the_same_set_to_the_same_source_boxes(self):
        lb = letterbox_params(1920, 1080, 640, 640)  # scale 1/3, pad (0, 140)
        transposed = decode_detections(
            _spec("yolo_v8_transposed"), _tensor("yolo_v8_transposed", LATE_DETECTIONS),
            lb, 0.25, 0.45)
        passthrough = decode_detections(
            _spec("yolo_v8_passthrough"), _tensor("yolo_v8_passthrough", LATE_DETECTIONS),
            lb, 0.25, 0.45)
        self.assertEqual(len(transposed), 3)
        self.assertDetectionsEqual(transposed, passthrough)
        # And the answer is the one worked out on paper, not merely equal to itself.
        for det, (idx, cx, cy, w, h, cid, score) in zip(transposed, LATE_DETECTIONS):
            model_box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            self.assertEqual(det.class_id, cid)
            self.assertAlmostEqual(det.score, score, places=6)
            self.assertBoxAlmostEqual(det.box, lb.to_source(model_box), places=3)

    def test_a_late_box_index_alone_forces_the_layout_to_be_read(self):
        # One detection, one class, at index 8399 - the last anchor.
        det = (8399, 200.0, 100.0, 40.0, 80.0, 5, 0.77)
        for layout in ("yolo_v8_transposed", "yolo_v8_passthrough"):
            with self.subTest(layout=layout):
                dets = decode_detections(
                    _spec(layout), _tensor(layout, [det]),
                    letterbox_params(640, 640, 640, 640), 0.25, 0.45)
                self.assertEqual(len(dets), 1)
                self.assertEqual(dets[0].class_id, 5)
                self.assertBoxAlmostEqual(dets[0].box, (180.0, 60.0, 220.0, 140.0))

    def test_a_tensor_whose_extent_contradicts_the_layout_is_refused(self):
        # The transposed reading needs 84 features on axis 1; given 8400 it must not
        # quietly reinterpret the tensor, because that is the wrong-object failure.
        with self.assertRaises(ValueError):
            decode_detections(_spec("yolo_v8_transposed"),
                              _tensor("yolo_v8_passthrough", [ONE_DETECTION]),
                              letterbox_params(640, 640, 640, 640), 0.25, 0.45)
        with self.assertRaises(ValueError):
            decode_detections(_spec("yolo_v8_passthrough"),
                              _tensor("yolo_v8_transposed", [ONE_DETECTION]),
                              letterbox_params(640, 640, 640, 640), 0.25, 0.45)

    def test_an_empty_tensor_decodes_to_no_detections(self):
        for layout in ("yolo_v8_transposed", "yolo_v8_passthrough"):
            with self.subTest(layout=layout):
                self.assertEqual(
                    decode_detections(_spec(layout), _tensor(layout, []),
                                      letterbox_params(640, 640, 640, 640), 0.25, 0.45),
                    [])

    def test_two_anchors_on_the_same_object_collapse_to_one_detection(self):
        # Measured on frame 60 of retail-checkout.mp4: 9 of the 15 above-threshold
        # anchors are near-duplicates of the same person (box indices 8130-8171, boxes
        # within a few pixels of each other, scores 0.86-0.91). A decode path that never
        # calls nms would therefore draw nine boxes on one person - which is why
        # `Nms` testing the function in isolation is not enough, and the fact that
        # `decode_detections` actually suppresses is asserted here, for both layouts.
        duplicates = [
            (4096, 345.56, 252.28, 272.90, 221.96, 0, 0.9091),
            (6001, 345.05, 251.50, 274.13, 220.05, 0, 0.9048),
            (8399, 345.44, 252.08, 274.55, 220.52, 0, 0.8950),
        ]
        for layout in ("yolo_v8_transposed", "yolo_v8_passthrough"):
            with self.subTest(layout=layout):
                dets = decode_detections(
                    _spec(layout), _tensor(layout, duplicates),
                    letterbox_params(640, 640, 640, 640), 0.25, 0.45)
                self.assertEqual(len(dets), 1, f"{layout} returned {len(dets)} of 3")
                # The survivor is the highest scoring one, not merely the first.
                self.assertAlmostEqual(dets[0].score, 0.9091, places=4)
                self.assertEqual(dets[0].class_id, 0)

    def test_a_far_apart_set_is_left_alone_by_suppression(self):
        # The other half of the same property: suppression must not merge distinct
        # objects, or the booth would show one box where there are three people.
        dets = decode_detections(
            _spec("yolo_v8_transposed"), _tensor("yolo_v8_transposed", LATE_DETECTIONS),
            letterbox_params(640, 640, 640, 640), 0.25, 0.45)
        self.assertEqual(len(dets), 3)
        self.assertEqual([d.class_id for d in dets], [0, 16, 2])


class TheTensorIsFetchedByPortIndexNotByName(_BoxAssertions):
    def test_a_named_port_still_fetches_by_index(self):
        # The real output port is unnamed, so a spec carrying a name must not send the
        # decoder looking for one. This is the case the plan's `"output0"` key implied.
        outputs = _tensor("yolo_v8_transposed", [ONE_DETECTION])
        spec = _spec("yolo_v8_transposed", names=("output0",))
        dets = decode_detections(spec, outputs, letterbox_params(640, 640, 640, 640),
                                 0.25, 0.45)
        self.assertEqual(len(dets), 1)

    def test_there_is_no_name_based_fallback(self):
        outputs = _tensor("yolo_v8_transposed", [ONE_DETECTION])
        renamed = {"output0": outputs["0"]}
        spec = _spec("yolo_v8_transposed", names=("output0",))
        with self.assertRaises(ValueError):
            decode_detections(spec, renamed, letterbox_params(640, 640, 640, 640),
                              0.25, 0.45)

    def test_an_unnamed_spec_decodes_identically_to_a_named_one(self):
        lb = letterbox_params(640, 640, 640, 640)
        outputs = _tensor("yolo_v8_transposed", LATE_DETECTIONS)
        self.assertDetectionsEqual(
            decode_detections(_spec("yolo_v8_transposed", names=()), outputs, lb, 0.25, 0.45),
            decode_detections(_spec("yolo_v8_transposed", names=("output0",)),
                              outputs, lb, 0.25, 0.45),
        )


class UndecodableIsNotTheSameAsEmpty(unittest.TestCase):
    """"This layout is not supported" and "this frame has no detections" are different
    facts. Only the second one is an empty list."""

    def test_a_layout_outside_the_known_set_is_refused_at_construction(self):
        for typo in ["yolo_v8", "YOLO_V8_TRANSPOSED", "transposed", "", None]:
            with self.subTest(layout=typo), self.assertRaises(ValueError):
                ModelOutputSpec(names=(), shapes={}, layout=typo, num_classes=80,
                                has_embedded_nms=False)

    def test_ssd_is_a_known_layout_that_this_milestone_does_not_decode(self):
        spec = _spec("ssd")
        with self.assertRaises(UnsupportedLayoutError):
            decode_detections(spec, _tensor("yolo_v8_transposed", [ONE_DETECTION]),
                              letterbox_params(640, 640, 640, 640), 0.25, 0.45)
        self.assertTrue(issubclass(UnsupportedLayoutError, ValueError))

    def test_a_missing_port_key_is_refused_not_treated_as_no_detections(self):
        spec = _spec("yolo_v8_transposed")
        with self.assertRaises(ValueError):
            decode_detections(spec, {}, letterbox_params(640, 640, 640, 640), 0.25, 0.45)

    def test_a_feature_extent_that_is_not_four_plus_num_classes_is_refused(self):
        # 84 features against a spec that says 80 classes + 4 = 84 is fine; 90 is not,
        # and neither is 84 against a spec claiming 10 classes.
        good = _tensor("yolo_v8_transposed", [ONE_DETECTION])
        with self.assertRaises(ValueError):
            decode_detections(_spec("yolo_v8_transposed", num_classes=10), good,
                              letterbox_params(640, 640, 640, 640), 0.25, 0.45)
        wrong = _tensor("yolo_v8_transposed", [ONE_DETECTION], num_classes=86)
        with self.assertRaises(ValueError):
            decode_detections(_spec("yolo_v8_transposed"), wrong,
                              letterbox_params(640, 640, 640, 640), 0.25, 0.45)

    def test_a_recorded_shape_that_contradicts_the_tensor_is_refused(self):
        spec = _spec("yolo_v8_transposed", shapes={"0": (1, 84, 1234)})
        with self.assertRaises(ValueError):
            decode_detections(spec, _tensor("yolo_v8_transposed", [ONE_DETECTION]),
                              letterbox_params(640, 640, 640, 640), 0.25, 0.45)

    def test_an_unusable_threshold_is_refused(self):
        spec = _spec("yolo_v8_transposed")
        outputs = _tensor("yolo_v8_transposed", [ONE_DETECTION])
        lb = letterbox_params(640, 640, 640, 640)
        for conf, iou in [(-0.1, 0.45), (1.5, 0.45), (0.25, -0.1), (0.25, 1.5)]:
            with self.subTest(conf=conf, iou=iou), self.assertRaises(ValueError):
                decode_detections(spec, outputs, lb, conf, iou)


class Nms(unittest.TestCase):
    def test_overlapping_boxes_are_suppressed(self):
        b = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [200, 200, 300, 300]], dtype=np.float32)
        s = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        self.assertEqual(nms(b, s, 0.45), [0, 2])

    def test_low_confidence_is_filtered_before_nms(self):
        spec = _spec("yolo_v8_transposed")
        det = (0, 300.0, 260.0, 80.0, 40.0, 0, 0.01)
        self.assertEqual(
            decode_detections(spec, _tensor("yolo_v8_transposed", [det]),
                              letterbox_params(640, 640, 640, 640), 0.25, 0.45),
            [],
        )

    def test_the_lower_scoring_member_of_an_overlapping_pair_is_the_one_dropped(self):
        # Same pair as the plan's test with the scores swapped: suppression must follow
        # the score, not the array order.
        b = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], dtype=np.float32)
        self.assertEqual(nms(b, np.array([0.8, 0.9], dtype=np.float32), 0.45), [1])

    def test_indices_come_back_ascending_regardless_of_score_order(self):
        b = np.array([[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]],
                     dtype=np.float32)
        s = np.array([0.5, 0.9, 0.7], dtype=np.float32)
        self.assertEqual(nms(b, s, 0.45), [0, 1, 2])

    def test_nms_is_class_agnostic_over_the_whole_caller_filtered_list(self):
        # The plan specifies class-agnostic suppression. Two identical boxes of
        # different classes are therefore collapsed, and the survivor is the higher
        # score - recorded here so a later per-class change is a deliberate one.
        b = np.array([[0, 0, 50, 50], [0, 0, 50, 50]], dtype=np.float32)
        self.assertEqual(nms(b, np.array([0.9, 0.4], dtype=np.float32), 0.45), [0])

    def test_nms_on_nothing_is_nothing(self):
        self.assertEqual(nms(np.zeros((0, 4), dtype=np.float32),
                             np.zeros((0,), dtype=np.float32), 0.45), [])

    def test_a_box_pair_just_under_the_threshold_is_kept(self):
        # Two 100x100 boxes offset by 45 in x: intersection 55*100 = 5500, union 14500,
        # IoU 0.3793 - below 0.45, so both survive.
        b = np.array([[0, 0, 100, 100], [45, 0, 145, 100]], dtype=np.float32)
        s = np.array([0.9, 0.8], dtype=np.float32)
        self.assertEqual(nms(b, s, 0.45), [0, 1])


class DescribeOutputsAgainstThePinnedIR(unittest.TestCase):
    """Step 6: the recorded description must agree with the real model.

    The agreement is one-directional on purpose. If it ever fails, the recorded file is
    regenerated by `tools/dump_model_io.py` and the disagreement is reported; the
    recorded file is never edited to match the code, because that is how a wrong
    decoder becomes an accepted fact.
    """

    @classmethod
    def setUpClass(cls):
        import openvino as ov
        cls.core = ov.Core()
        cls.model = cls.core.read_model(str(IR_PATH))
        cls.compiled = cls.core.compile_model(cls.model, "CPU")
        cls.recorded = json.loads(RECORDED_IO.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MODELS_JSON.read_text(encoding="utf-8"))

    def _spec(self):
        return describe_outputs(self.model, num_classes=NUM_CLASSES)

    def test_the_pinned_output_port_is_really_unnamed(self):
        # If a future re-export names the port, this fails and says so, rather than the
        # decoder quietly depending on a name that came and went.
        port = self.model.outputs[0]
        self.assertEqual(port.get_names(), set())
        with self.assertRaises(RuntimeError):
            port.get_any_name()

    def test_describe_outputs_does_not_ask_for_a_name(self):
        # A guarded `get_any_name()` must be a non-event, so the recorded names are ()
        # and nothing raised.
        self.assertEqual(self._spec().names, ())
        self.assertEqual(self._spec().unnamed_ports, (0,))
        self.assertEqual(self._spec().port_keys, ("0",))

    def test_the_recorded_file_agrees_with_the_real_model(self):
        spec = self._spec()
        self.assertEqual(spec.layout, self.recorded["layout"])
        self.assertEqual(list(spec.names), self.recorded["output_names"])
        self.assertEqual(spec.num_classes, self.recorded["num_classes"])
        self.assertIs(spec.has_embedded_nms, self.recorded["has_embedded_nms"])
        self.assertEqual({k: list(v) for k, v in spec.shapes.items()},
                         self.recorded["shapes"])
        self.assertEqual(self.recorded["box_encoding"], "cxcywh")

    def test_the_recorded_file_says_the_output_port_has_no_name(self):
        # The whole reason `shapes` is keyed by index. If a re-export names the port,
        # this fails and the record is regenerated rather than the key quietly changing.
        self.assertEqual(self.recorded["output_names"], [])
        self.assertEqual(self.recorded["unnamed_ports"], [0])
        self.assertIn("RAISES", self.recorded["outputs"][0]["get_any_name"])

    def test_the_layout_is_classified_transposed_from_the_observed_shape(self):
        self.assertEqual(self._spec().layout, "yolo_v8_transposed")
        self.assertEqual(self.recorded["layout"], "yolo_v8_transposed")
        self.assertEqual(self.recorded["shapes"]["0"], [1, 84, 8400])

    def test_the_shape_dict_is_keyed_by_port_index_because_the_port_has_no_name(self):
        self.assertEqual(sorted(self._spec().shapes), ["0"])
        self.assertEqual(sorted(self.recorded["shapes"]), ["0"])

    def test_compilation_does_not_change_the_output_contract(self):
        # The brief asks for the recorded description to be checked "on the real
        # compiled model". Compilation cannot change the ports, and that is the claim
        # worth making - but a CompiledModel cannot be walked, so `describe_outputs`
        # refuses it rather than reporting has_embedded_nms from a graph it never read.
        compiled_ports = [list(p.get_partial_shape()) for p in self.compiled.outputs]
        read_ports = [list(p.get_partial_shape()) for p in self.model.outputs]
        self.assertEqual([[int(d.get_length()) for d in s] for s in compiled_ports],
                         [[int(d.get_length()) for d in s] for s in read_ports])
        for a, b in zip(self.compiled.outputs, self.model.outputs):
            self.assertEqual(a.get_names(), b.get_names())
            self.assertEqual(str(a.get_element_type().to_string()),
                             str(b.get_element_type().to_string()))
        with self.assertRaises(ValueError) as caught:
            describe_outputs(self.compiled, num_classes=NUM_CLASSES)
        self.assertIn("CompiledModel", str(caught.exception))

    def test_has_embedded_nms_is_false_and_the_file_says_why(self):
        self.assertIs(self._spec().has_embedded_nms, False)
        self.assertIs(self.recorded["has_embedded_nms"], False)
        self.assertIn("NonMaxSuppression", self.recorded["has_embedded_nms_evidence"])

    def test_the_recorded_num_classes_is_the_manifest_num_classes(self):
        model_entry = self.manifest["models"][0]
        self.assertEqual(model_entry["id"], "yolo11n-fp16")
        self.assertEqual(self.recorded["num_classes"], model_entry["num_classes"])
        self.assertEqual(self.recorded["ir_path"], model_entry["ir_path"])
        self.assertEqual(self.recorded["input_shape"], model_entry["input_shape"])

    def test_the_feature_count_is_four_box_values_plus_eighty_classes(self):
        self.assertEqual(4 + self.recorded["num_classes"],
                         self.recorded["shapes"]["0"][1])
        labels = (IR_PATH.parent / "labels.txt").read_text(encoding="utf-8").split()
        self.assertEqual(len(labels), self.recorded["num_classes"])


class _FakePort:
    """A port stub. `shape` may hold `None` for an open (dynamic) dimension."""

    def __init__(self, shape, element_type="f32", names=()):
        self._shape = shape
        self._type = element_type
        self._names = set(names)

    def get_names(self):
        return self._names

    def get_element_type(self):
        return self._type

    def get_partial_shape(self):
        return self._shape


class _FakeOp:
    def __init__(self, type_name, friendly_name=""):
        self._type_name = type_name
        self._friendly_name = friendly_name

    def get_type_name(self):
        return self._type_name

    def get_friendly_name(self):
        return self._friendly_name


class _FakeModel:
    def __init__(self, outputs, ops=()):
        self._outputs = list(outputs)
        self._ops = list(ops)

    @property
    def outputs(self):
        return list(self._outputs)

    def get_ordered_ops(self):
        return list(self._ops)


#: The op types measured in the pinned export, none of which suppresses. Used as the
#: default so `describe_outputs` is exercised against a graph that resembles the real one
#: rather than against an empty op list.
MEASURED_OP_TYPES = (
    "Constant", "Convert", "Add", "Convolution", "Swish", "Concat", "VariadicSplit",
    "Reshape", "GroupConvolution", "MaxPool", "MatMul", "Multiply", "Softmax",
    "Interpolate", "Divide", "Subtract", "Parameter", "Transpose", "ShapeOf", "Gather",
)


class DescribeOutputsClassification(unittest.TestCase):
    """The classifier, on synthetic ports. No model required, so every branch is pinned
    even where this machine has only one real model to point it at."""

    def _describe(self, shape, element_type="f32", names=(), num_classes=80, ops=None):
        port = _FakePort(shape, element_type, names)
        if ops is None:
            ops = [_FakeOp(t) for t in MEASURED_OP_TYPES]
        return describe_outputs(_FakeModel([port], ops), num_classes=num_classes)

    def test_84_features_on_axis_one_is_the_transposed_yolo_layout(self):
        self.assertEqual(self._describe([1, 84, 8400]).layout, "yolo_v8_transposed")

    def test_84_features_on_axis_two_is_the_passthrough_yolo_layout(self):
        self.assertEqual(self._describe([1, 8400, 84]).layout, "yolo_v8_passthrough")

    def test_a_shape_matching_neither_is_unknown_not_a_guess(self):
        for shape in ([1, 100, 100], [1, 84], [2, 84, 8400], [1, 2, 3, 4], []):
            with self.subTest(shape=shape):
                self.assertEqual(self._describe(shape).layout, "unknown")

    def test_a_non_float_output_is_unknown_because_it_would_decode_differently(self):
        self.assertEqual(self._describe([1, 84, 8400], element_type="i32").layout,
                         "unknown")

    def test_a_named_port_records_its_name_in_port_order(self):
        spec = self._describe([1, 84, 8400], names=("output0",))
        self.assertEqual(spec.names, ("output0",))
        self.assertEqual(spec.unnamed_ports, ())
        self.assertEqual(spec.port_keys, ("0",))

    def test_the_class_count_that_decides_the_layout_is_the_one_passed_in(self):
        # Same shape, different declared class count -> the 84 no longer means
        # 4 + num_classes, so the classifier must stop claiming a YOLO layout.
        self.assertEqual(self._describe([1, 84, 8400], num_classes=80).layout,
                         "yolo_v8_transposed")
        self.assertEqual(self._describe([1, 84, 8400], num_classes=79).layout, "unknown")

    def test_a_dynamic_extent_is_unknown_rather_than_assumed_static(self):
        # The NPU rejects dynamic shapes, so a port whose axis is open is a refusal,
        # not a layout to guess at. `None` stands for an open dimension here because
        # the real PartialShape object is not worth building in a unit test.
        self.assertEqual(self._describe([1, 84, None]).layout, "unknown")


class HasEmbeddedNmsIsDerivedFromTheOps(unittest.TestCase):
    """`has_embedded_nms` decides whether this module suppresses at all, so it is read
    off the graph rather than asserted. These pin both directions, so a future edit
    that hardcodes `False` is caught even on a machine that only has this one model."""

    def _spec(self, ops):
        return describe_outputs(
            _FakeModel([_FakePort([1, 84, 8400])], ops), num_classes=80)

    def test_a_graph_with_no_suppression_op_records_false(self):
        ops = [_FakeOp("Convolution"), _FakeOp("Softmax"), _FakeOp("Reshape")]
        self.assertIs(self._spec(ops).has_embedded_nms, False)

    def test_a_nonmaxsuppression_op_anywhere_records_true(self):
        for spelling in ("NonMaxSuppression", "NonMaxSuppressionV5",
                         "NonMaxSuppressionV9", "NonMaxSuppressionV10",
                         "NonMaxSuppressionV11"):
            with self.subTest(spelling=spelling):
                ops = [_FakeOp("Convolution"), _FakeOp(spelling)]
                self.assertIs(self._spec(ops).has_embedded_nms, True)

    def test_a_friendly_name_that_merely_looks_like_nms_is_not_evidence(self):
        # The op TYPE is the evidence. A friendly name is a developer-chosen label, so
        # `my_nms_wrapper` proves nothing - and believing it would record
        # has_embedded_nms=True for a model that does not suppress, which would then
        # make the decoder skip suppression it still has to do.
        ops = [_FakeOp("Convert", friendly_name="my_nms_wrapper")]
        self.assertIs(self._spec(ops).has_embedded_nms, False)
        ops = [_FakeOp("Convert", friendly_name="NonMaxSuppression")]
        self.assertIs(self._spec(ops).has_embedded_nms, False)

    def test_an_op_type_that_merely_starts_with_the_same_letters_is_not_suppression(self):
        for near_miss in ("NonMaxSuppressionScore", "NonMax", "TopK"):
            with self.subTest(type_name=near_miss):
                self.assertIs(self._spec([_FakeOp(near_miss)]).has_embedded_nms, False)

    def test_a_graph_with_no_ops_at_all_records_false(self):
        self.assertIs(self._spec([]).has_embedded_nms, False)

    def test_the_measured_pinned_graph_has_no_suppression_op(self):
        # The real census, stated as a test rather than only in the docstring: 24
        # distinct op types, none of which is a suppression op.
        for op_type in MEASURED_OP_TYPES:
            with self.subTest(op_type=op_type):
                self.assertIs(self._spec([_FakeOp(op_type)]).has_embedded_nms, False)

    def test_a_spec_that_says_nms_is_already_embedded_is_not_decoded_a_second_time(self):
        # Suppressing an already-suppressed list would drop real detections. The
        # decoder refuses rather than guessing which suppression was meant.
        spec = ModelOutputSpec(
            names=(), shapes={"0": (1, 84, 8400)}, layout="yolo_v8_transposed",
            num_classes=80, has_embedded_nms=True)
        outputs = _tensor("yolo_v8_transposed", [ONE_DETECTION])
        with self.assertRaises(UnsupportedLayoutError):
            decode_detections(spec, outputs, letterbox_params(640, 640, 640, 640),
                              0.25, 0.45)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
