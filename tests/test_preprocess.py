"""Letterbox geometry and the preprocessing tensor.

Review Focus #4: *"Letterbox coordinate recovery omits the inverse pad/scale, which is
invisible on a centred object and wrong on every other one."*

The first four tests below are the plan's, verbatim. The `AbsolutePlacement` and
`TheArrayAgreesWithTheLetterbox` classes are added because the plan's round-trip pair
(`test_round_trip_recovers_source_coordinates` and
`test_corner_detection_survives_the_inverse_transform`) **cannot** catch the failure it
names. Both assertions are `to_source(to_model(x)) == x`, and a `to_model`/`to_source`
pair that is *self-consistently* wrong - for instance one where both halves forget the
pad and apply only the scale - satisfies both tests exactly. The only assertions that
kill that mutant are the ones that pin a model-space coordinate **absolutely**, which
is what `AbsolutePlacement` does. See the Task 4 report for the mutation run.

`TheArrayAgreesWithTheLetterbox` closes the remaining gap: the geometry above and the
`float32` NCHW array are produced by the same function, so a test that only checks the
array's shape and dtype would not notice if the two disagreed about where the image
went. Task 5's decoder inverts `Letterbox.to_source` on boxes the model read out of
this array, so the two have to be pinned to the same numbers.
"""

import unittest

import numpy as np

from app.inference.preprocess import (
    DEFAULT_PAD_VALUE,
    Letterbox,
    letterbox_params,
    preprocess_bgr,
)


class LetterboxGeometry(unittest.TestCase):
    def test_asymmetric_source_pads_asymmetrically(self):
        # 200x100 into a square must pad vertically only.
        lb = letterbox_params(200, 100, 640, 640)
        self.assertAlmostEqual(lb.scale, 3.2, places=5)
        self.assertEqual(lb.pad_x, 0)
        self.assertEqual(lb.pad_y, 160)

    def test_portrait_source_pads_horizontally(self):
        lb = letterbox_params(100, 200, 640, 640)
        self.assertEqual(lb.pad_x, 160)
        self.assertEqual(lb.pad_y, 0)

    def test_round_trip_recovers_source_coordinates(self):
        # CORRECTED 2026-10-01. The plan's version of this test was
        #     for x1, y1, x2, y2 in [lb.to_model(src), lb.to_source(src)]:  assert == src
        # which is not a round trip: it asserts that BOTH directions return the source
        # box unchanged, and that holds only when scale == 1 and pad == 0. For this
        # letterbox (scale 3.2, pad_y 160) it demands to_model(10,20) == 10, which
        # contradicts `test_asymmetric_source_pads_asymmetrically` in this same class.
        # No implementation can satisfy both, so the test - not the geometry - is what
        # is wrong. The real round trip recovers the box exactly:
        #     to_source(to_model(src)) == src
        lb = letterbox_params(200, 100, 640, 640)
        src = (10.0, 20.0, 90.0, 80.0)
        for x1, y1, x2, y2 in [lb.to_source(lb.to_model(src))]:
            for a, b in zip((x1, y1, x2, y2), src):
                self.assertAlmostEqual(a, b, places=4)

    def test_the_identity_letterbox_is_the_one_where_each_direction_is_a_no_op(self):
        # The property the plan's test was actually reaching for, stated where it is
        # true: only a 1:1 letterbox makes to_model and to_source both the identity.
        lb = letterbox_params(200, 100, 200, 100)
        src = (10.0, 20.0, 90.0, 80.0)
        self.assertEqual(lb.to_model(src), src)
        self.assertEqual(lb.to_source(src), src)

    def test_corner_detection_survives_the_inverse_transform(self):
        # Review Focus #4: a centred object hides a missing pad/scale inverse.
        lb = letterbox_params(1920, 1080, 640, 640)
        corner_src = (1700.0, 980.0, 1900.0, 1070.0)
        back = lb.to_source(lb.to_model(corner_src))
        for a, b in zip(back, corner_src):
            self.assertAlmostEqual(a, b, places=3)


#: The exact `float32` value a `float32` NCHW tensor holds for a pad level of 114.
#: Comparing against Python's double `114 / 255.0` is wrong by 3.2e-9, because the array
#: rounds to float32 and the double does not. `float32` spacing at 0.447 is 3.0e-8, so
#: this is a representation difference, not a preprocessing defect.
PAD_114_AS_FLOAT32 = float(np.float32(114) / np.float32(255.0))


class _BoxAssertions(unittest.TestCase):
    """Boxes are compared coordinate-by-coordinate to 1e-9 rather than with `==`.

    `scale` is a float, so a product like `200 * 3.2` is not exactly 640.0 in binary.
    That is arithmetic, not a geometry defect, and pinning the transform to 1e-9 px is
    already seven orders of magnitude tighter than any pixel the model can produce."""

    def assertBoxAlmostEqual(self, got, want, places=9, msg=None):
        got = tuple(float(v) for v in got)
        want = tuple(float(v) for v in want)
        self.assertEqual(len(got), len(want), msg)
        for axis, (a, b) in enumerate(zip(got, want)):
            self.assertAlmostEqual(a, b, places=places, msg=f"{msg or ''} axis {axis}")


class AbsolutePlacement(_BoxAssertions):
    """Model-space coordinates pinned absolutely, so a self-consistent wrong transform
    cannot pass. Every expected value below is derived on paper, in the comment, from
    `scale = min(dst/src)` and a centred image - never from the implementation."""

    def test_the_whole_source_image_maps_onto_the_whole_padded_model_image(self):
        # 200x100 -> scale 3.2 -> 640x320 -> pad_y 160, pad_x 0. So source (0,0)-(200,100)
        # is model (0,160)-(640,480), and a source y of 0 is a model y of 160, not 0.
        lb = letterbox_params(200, 100, 640, 640)
        self.assertBoxAlmostEqual(lb.to_model((0.0, 0.0, 200.0, 100.0)),
                                  (0.0, 160.0, 640.0, 480.0))
        self.assertBoxAlmostEqual(lb.to_source((0.0, 160.0, 640.0, 480.0)),
                                  (0.0, 0.0, 200.0, 100.0))

    def test_the_pad_is_removed_by_the_inverse_not_carried_through(self):
        # 1280x720 -> scale 0.5 -> 640x360 -> pad_y 140, pad_x 0.
        # NOTE: the Task 5 brief annotated this case "pad_y 80". That is 720-640, i.e.
        # the difference between the two heights, which is not the leftover: the
        # leftover is 640 - 360 = 280 and half of it is 140. The same rule gives 140
        # for 1920x1080, which the probe measured first-hand on this machine.
        lb = letterbox_params(1280, 720, 640, 640)
        self.assertAlmostEqual(lb.scale, 0.5, places=6)
        self.assertEqual((lb.pad_x, lb.pad_y), (0, 140))
        # Model x 300 -> source 600; model y 300 -> source (300-140)/0.5 = 320.
        self.assertEqual(lb.to_source((300.0, 300.0, 300.0, 300.0)),
                         (600.0, 320.0, 600.0, 320.0))

    def test_a_corner_box_maps_to_its_own_model_coordinates(self):
        # 1920x1080 -> scale 1/3 -> 640x360 -> pad (0,140).
        # (1700,980) -> (1700/3, 980/3+140) = (566.666..., 466.666...)
        # (1900,1070) -> (1900/3, 1070/3+140) = (633.333..., 496.666...)
        lb = letterbox_params(1920, 1080, 640, 640)
        model = lb.to_model((1700.0, 980.0, 1900.0, 1070.0))
        for got, want in zip(model, (1700 / 3, 980 / 3 + 140, 1900 / 3, 1070 / 3 + 140)):
            self.assertAlmostEqual(got, want, places=4)

    def test_the_horizontal_pad_is_applied_to_x_and_only_to_x(self):
        # 100x200 -> scale 3.2 -> 320x640 -> pad_x 160, pad_y 0.
        lb = letterbox_params(100, 200, 640, 640)
        self.assertBoxAlmostEqual(lb.to_model((0.0, 0.0, 100.0, 200.0)),
                                  (160.0, 0.0, 480.0, 640.0))

    def test_a_landscape_source_into_a_wide_destination_pads_both_axes(self):
        # 200x100 -> 320x320: scale 1.6 -> 320x160 -> pad (0, 80).
        lb = letterbox_params(200, 100, 320, 320)
        self.assertEqual((lb.pad_x, lb.pad_y), (0, 80))
        # 100x200 -> 320x320: scale 1.6 -> 160x320 -> pad (80, 0).
        lb = letterbox_params(100, 200, 320, 320)
        self.assertEqual((lb.pad_x, lb.pad_y), (80, 0))

    def test_a_downscale_uses_the_same_rule(self):
        # 3840x2160 -> scale 1/6 -> 640x360 -> pad (0, 140).
        lb = letterbox_params(3840, 2160, 640, 640)
        self.assertAlmostEqual(lb.scale, 1 / 6, places=9)
        self.assertEqual((lb.pad_x, lb.pad_y), (0, 140))

    def test_an_already_square_source_has_no_pad_at_all(self):
        lb = letterbox_params(640, 640, 640, 640)
        self.assertEqual((lb.scale, lb.pad_x, lb.pad_y), (1.0, 0, 0))
        self.assertEqual(lb.to_model((12.0, 34.0, 56.0, 78.0)), (12.0, 34.0, 56.0, 78.0))

    def test_the_scaled_image_never_overflows_or_leaves_a_negative_pad(self):
        # `pad` is a floor of the leftover, not a round of it, precisely so that
        # `2*pad + scaled <= dst` holds for every aspect ratio. Prime and odd sizes are
        # here because that is where a round() would leave a one-pixel overflow.
        for src in [(200, 100), (100, 200), (1920, 1080), (1280, 720), (300, 133),
                    (7, 3), (641, 639), (3840, 2160), (1, 1), (999, 501)]:
            with self.subTest(src=src):
                lb = letterbox_params(*src, 640, 640)
                self.assertGreaterEqual(lb.pad_x, 0)
                self.assertGreaterEqual(lb.pad_y, 0)
                self.assertLessEqual(2 * lb.pad_x + lb.scaled_w, 640)
                self.assertLessEqual(2 * lb.pad_y + lb.scaled_h, 640)
                self.assertGreaterEqual(lb.scaled_w, 1)
                self.assertGreaterEqual(lb.scaled_h, 1)

    def test_letterbox_is_frozen(self):
        lb = letterbox_params(200, 100, 640, 640)
        with self.assertRaises(Exception):
            lb.pad_y = 0  # type: ignore[misc]


class LetterboxRefusesNonsense(unittest.TestCase):
    def test_a_zero_or_negative_extent_is_refused_rather_than_dividing_by_it(self):
        for args in [(0, 100, 640, 640), (100, 0, 640, 640),
                     (100, 100, 0, 640), (100, 100, 640, 0),
                     (-200, 100, 640, 640), (100, -200, 640, 640)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                letterbox_params(*args)

    def test_a_directly_constructed_letterbox_with_a_zero_scale_is_refused(self):
        # `to_source` divides by `scale`, so a zero scale must be caught at construction
        # rather than as a ZeroDivisionError on the first detection of the demo.
        with self.assertRaises(ValueError):
            Letterbox(scale=0.0, pad_x=0, pad_y=0, src_w=100, src_h=100,
                      dst_w=640, dst_h=640)


class PreprocessOutput(unittest.TestCase):
    def test_returns_float32_nchw_rgb_scaled_to_unit_range(self):
        bgr = np.full((100, 200, 3), 255, dtype=np.uint8)
        out, lb = preprocess_bgr(bgr, 640, 640)
        self.assertEqual(out.shape, (1, 3, 640, 640))
        self.assertEqual(out.dtype, np.float32)
        self.assertLessEqual(float(out.max()), 1.0)
        # BGR input of (0,0,255) is pure red, so channel 0 must be the max.
        blue = np.zeros((4, 4, 3), dtype=np.uint8)
        blue[..., 2] = 255
        out2, _ = preprocess_bgr(blue, 8, 8)
        self.assertGreater(float(out2[0, 0].max()), float(out2[0, 2].max()))

    def test_raises_on_a_three_dimensional_input(self):
        # (Named for the plan's wording. What it actually asserts is that an array which
        # is not HxWx3 is refused: this input is 2-dimensional, and a 4-channel array is
        # refused by the same rule - see the two cases below.)
        with self.assertRaises(ValueError):
            preprocess_bgr(np.zeros((10, 10), dtype=np.uint8), 640, 640)

    def test_raises_on_a_four_channel_or_one_dimensional_input(self):
        for bad in [np.zeros((10, 10, 4), dtype=np.uint8),
                    np.zeros((10, 10, 1), dtype=np.uint8),
                    np.zeros((3,), dtype=np.uint8),
                    np.zeros((2, 10, 10, 3), dtype=np.uint8)]:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                preprocess_bgr(bad, 640, 640)

    def test_a_non_uint8_array_is_refused_rather_than_rescaled_on_a_guess(self):
        # A float array could be in [0,1] or in [0,255]; guessing either would silently
        # produce a black or a blown-out image, so the dtype has to be stated.
        for bad in [np.zeros((10, 10, 3), dtype=np.float32),
                    np.zeros((10, 10, 3), dtype=np.int16)]:
            with self.subTest(dtype=bad.dtype), self.assertRaises(TypeError):
                preprocess_bgr(bad, 640, 640)

    def test_the_returned_letterbox_is_the_one_for_those_extents(self):
        out, lb = preprocess_bgr(np.zeros((100, 200, 3), dtype=np.uint8), 640, 640)
        self.assertEqual(lb, letterbox_params(200, 100, 640, 640))


class TheArrayAgreesWithTheLetterbox(_BoxAssertions):
    """The tensor and the `Letterbox` are two halves of one transform, and Task 5's
    decoder inverts the `Letterbox` on coordinates read out of this tensor. A test that
    only checked shape and dtype would not notice the two halves disagreeing."""

    def test_a_bright_quadrant_lands_exactly_where_the_letterbox_says(self):
        # 200x100 source, white only in its bottom-right quadrant: source rows 50..99
        # and cols 100..199. Under scale 3.2 that is resized rows 160..319 / cols
        # 320..639, and pad (0,160) puts it at model rows 320..479 / cols 320..639.
        # The 3-pixel margin is the bilinear ramp at the hard edge, nothing else.
        bgr = np.zeros((100, 200, 3), dtype=np.uint8)
        bgr[50:, 100:, :] = 255
        out, lb = preprocess_bgr(bgr, 640, 640)
        self.assertEqual((lb.pad_x, lb.pad_y), (0, 160))
        lit = out[0, 0] > 0.5
        rows = np.flatnonzero(lit.any(axis=1))
        cols = np.flatnonzero(lit.any(axis=0))
        self.assertGreaterEqual(int(rows[0]), 320 - 3)
        self.assertLessEqual(int(rows[-1]), 479 + 3)
        self.assertGreaterEqual(int(cols[0]), 320 - 3)
        self.assertLessEqual(int(cols[-1]), 639)
        # The quadrant's interior is fully white. Outside it there are now TWO distinct
        # values to tell apart, and conflating them is what this assertion is for:
        #   rows <320 and rows >480 include the 160px PAD band, which is 114 by default;
        #   cols <320 within rows 323..477 is the IMAGE's own black half, which is 0.
        # Under the previous zero-pad default all four read 0.0, so this could not have
        # distinguished "pad band" from "black part of the picture" at all.
        pad_level = PAD_114_AS_FLOAT32
        self.assertEqual(float(out[0, :, 323:477, 323:].min()), 1.0)
        self.assertEqual(float(out[0, :, :317, :].max()), pad_level)
        self.assertEqual(float(out[0, :, 483:, :].max()), pad_level)
        self.assertEqual(float(out[0, :, 323:477, :317].max()), 0.0)

    def test_the_pad_band_is_the_pad_value_and_the_image_band_is_not(self):
        # 200x100 -> 640x640 with pad_y 160, so rows 0..159 and 480..639 are pure pad.
        bgr = np.full((100, 200, 3), 255, dtype=np.uint8)
        out, lb = preprocess_bgr(bgr, 640, 640)
        self.assertEqual(lb.pad_y, 160)
        self.assertEqual(float(out[0, :, :160, :].max()), PAD_114_AS_FLOAT32)
        self.assertEqual(float(out[0, :, 480:, :].max()), PAD_114_AS_FLOAT32)
        self.assertEqual(float(out[0, :, 160:480, :].max()), 1.0)

    def test_a_corner_of_the_model_image_maps_back_to_a_corner_of_the_source(self):
        # Model (0,0) is pad, not source, so to_source must place it outside the source
        # frame rather than clamp it to 0. Clamping is a consumer's decision and is not
        # done here; this pins that the transform is affine and unclamped.
        lb = letterbox_params(1920, 1080, 640, 640)
        self.assertBoxAlmostEqual(lb.to_source((0.0, 0.0, 0.0, 0.0)),
                                  (0.0, -420.0, 0.0, -420.0))
        self.assertBoxAlmostEqual(lb.to_source((640.0, 640.0, 640.0, 640.0)),
                                  (1920.0, 1500.0, 1920.0, 1500.0))

    def test_preprocess_is_deterministic(self):
        bgr = (np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3))
        a, lb_a = preprocess_bgr(bgr, 640, 640)
        b, lb_b = preprocess_bgr(bgr, 640, 640)
        self.assertEqual(lb_a, lb_b)
        self.assertTrue(np.array_equal(a, b))


class ThePadValueIsADecisionThatWasMeasured(unittest.TestCase):
    """RULING 24, 2026-10-01: the pad band is mid-grey **114**, not zeros.

    The plan specified zeros and the first version of this module implemented that. It
    was then measured, and zeros turned out to be a train/serve distribution shift the
    export was converted *against*: this model came from the Ultralytics pipeline, which
    letterboxes with 114. On 1920x1080 into 640x640, 280 of the 640 rows are pad -
    **43.75% of the canvas**, 140 top and 140 bottom.

    The first version of this docstring said **21.88%**, which is 140/640: the *top band
    alone*, presented as the whole canvas. The reports have always said 43.75% and the
    M0 ledger ruled the ~22% reading wrong; this docstring was the copy that drifted.

    The number is now a decision with a measured consequence, so it is pinned here rather
    than left as a default that can drift silently. `0` stays reachable because the A/B
    must remain runnable - the M1 detection-set delta tool is what would settle whether
    the value changes results, and M0 does not claim that it does.
    """

    def test_the_default_pad_value_is_114(self):
        self.assertEqual(DEFAULT_PAD_VALUE, 114)
        bgr = np.full((100, 200, 3), 255, dtype=np.uint8)
        out, lb = preprocess_bgr(bgr, 640, 640)
        self.assertEqual(lb.pad_value, 114)
        self.assertEqual(float(out[0, :, :160, :].max()), PAD_114_AS_FLOAT32)
        self.assertEqual(float(out[0, 0, 0, 0]), PAD_114_AS_FLOAT32)

    def test_letterbox_params_defaults_to_the_same_value_as_preprocess(self):
        # The array and the record cannot disagree: both default to the one constant, and
        # `preprocess_bgr` builds the canvas from `letterbox.pad_value` rather than from
        # a literal, so there is no second place the value lives.
        self.assertEqual(letterbox_params(200, 100, 640, 640).pad_value,
                         DEFAULT_PAD_VALUE)
        _, lb = preprocess_bgr(np.zeros((100, 200, 3), dtype=np.uint8), 640, 640)
        self.assertEqual(lb, letterbox_params(200, 100, 640, 640))

    def test_zero_is_still_reachable_so_the_a_b_stays_runnable(self):
        bgr = np.full((100, 200, 3), 255, dtype=np.uint8)
        zero, lb_zero = preprocess_bgr(bgr, 640, 640, pad_value=0)
        grey, lb_grey = preprocess_bgr(bgr, 640, 640, pad_value=114)
        self.assertEqual(lb_zero.pad_value, 0)
        self.assertEqual(float(zero[0, :, :160, :].max()), 0.0)
        # The two differ by exactly the pad band and nowhere else.
        self.assertAlmostEqual(float(zero[0, :, 160:480, :].max()),
                               float(grey[0, :, 160:480, :].max()), places=6)
        self.assertGreater(float(np.abs(zero - grey).max()), 0.0)
        self.assertEqual(float(np.abs(zero - grey).max()), PAD_114_AS_FLOAT32)

    def test_only_the_pad_band_differs_between_pad_values(self):
        # Proves the parameter feeds the canvas fill and nothing else - a value that
        # also leaked into the image band would be a preprocessing bug, not a pad change.
        bgr = np.full((100, 200, 3), 255, dtype=np.uint8)
        zero, _ = preprocess_bgr(bgr, 640, 640, pad_value=0)
        grey, _ = preprocess_bgr(bgr, 640, 640, pad_value=114)
        self.assertEqual(float(np.abs(zero[0, :, 160:480, :] - grey[0, :, 160:480, :]).max()),
                         0.0)
        self.assertNotEqual(float(np.abs(zero[0, :, :160, :] - grey[0, :, :160, :]).max()),
                            0.0)

    def test_a_pad_value_that_is_not_a_uint8_level_is_refused(self):
        bgr = np.zeros((100, 200, 3), dtype=np.uint8)
        for bad in [-1, 256, 1000]:
            with self.subTest(pad_value=bad), self.assertRaises(ValueError):
                preprocess_bgr(bgr, 640, 640, pad_value=bad)
        # A bool is an int in Python, and `True` as a pixel level is a bug wearing a
        # valid-looking value; a float would be silently truncated.
        for bad in [True, 114.0, "114", None]:
            with self.subTest(pad_value=bad), self.assertRaises(TypeError):
                preprocess_bgr(bgr, 640, 640, pad_value=bad)
        with self.assertRaises(ValueError):
            letterbox_params(200, 100, 640, 640, pad_value=300)
        with self.assertRaises(TypeError):
            letterbox_params(200, 100, 640, 640, pad_value=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
