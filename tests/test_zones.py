"""`app/engine/zones.py` - normalised ROI geometry, hit tests, dwell, crossings.

Review Focus baked into this file: **a bad ROI must fail loudly.** A silently
mis-ordered or out-of-range zone does not crash; it produces a zone that is empty,
and an empty zone produces zero events, which on a booth screen reads as "nothing
is happening in this shop" - a false claim, and the most expensive kind.

What each test class is actually able to catch is recorded where a test is weaker
than its name suggests; two of the brief's own tests could not fail for the reason
they give, and they are replaced below with tests that can.

Pure: no clock, no I/O, no numpy.
"""

import unittest

from app.engine.zones import Zone, crossed_line, dwell_seconds_in_zone

SHELF = Zone("shelf", "shelf", (0.02, 0.22, 0.34, 0.55))
BASKET = Zone("basket", "shelf", (0.60, 0.30, 0.98, 0.70))
BELT = Zone(name="belt", kind="line", roi=(0.34, 0.18, 0.74, 0.58))


class Containment(unittest.TestCase):
    def test_zone_accepts_a_fully_contained_box(self):
        self.assertTrue(SHELF.contains((0.10, 0.30, 0.20, 0.40)))
        self.assertFalse(SHELF.contains((0.10, 0.30, 0.90, 0.40)))

    def test_a_box_touching_the_edge_is_inside(self):
        self.assertTrue(SHELF.contains((0.02, 0.22, 0.34, 0.55)))

    def test_pixels_round_to_the_source_frame(self):
        self.assertEqual(SHELF.to_pixels(1920, 1080), (38, 237, 652, 594))

    def test_roi_must_be_normalised_and_ordered(self):
        with self.assertRaises(ValueError):
            Zone("bad", "shelf", (0.9, 0.1, 0.2, 0.5))
        with self.assertRaises(ValueError):
            Zone("bad", "shelf", (0.1, 0.1, 1.2, 0.5))


class RejectedZones(unittest.TestCase):
    """Every way a config file can carry a zone that does not describe one.

    A zone is operator-authored JSON. Each entry below is a typo that would
    otherwise produce a plausible-looking rectangle in the wrong place - or none at
    all, which is worse.
    """

    def _refuse(self, **kw):
        base = dict(name="z", kind="shelf", roi=(0.1, 0.1, 0.2, 0.2))
        base.update(kw)
        with self.assertRaises(ValueError):
            Zone(**base)

    def test_x_out_of_range_is_refused_both_ways(self):
        self._refuse(roi=(-0.01, 0.1, 0.2, 0.2))   # left of the frame
        self._refuse(roi=(0.1, 0.1, 1.01, 0.2))    # right of the frame

    def test_y_out_of_range_is_refused_both_ways(self):
        self._refuse(roi=(0.1, -0.2, 0.2, 0.5))
        self._refuse(roi=(0.1, 0.1, 0.2, 2.0))

    def test_y_inversion_is_refused(self):
        # The brief tests an inverted x and an out-of-range x, but not an inverted
        # y: a transposed ROI, which is exactly what hand-editing a config does.
        self._refuse(roi=(0.1, 0.5, 0.2, 0.2))

    def test_a_zero_extent_zone_is_refused(self):
        # x1 == x2 is an empty rectangle. Accepted, it reads as "never inside",
        # which is a silent always-zero counter.
        self._refuse(roi=(0.1, 0.1, 0.1, 0.2))
        self._refuse(roi=(0.1, 0.1, 0.2, 0.1))

    def test_nan_coordinates_are_refused(self):
        # nan fails every comparison, so a range check written as `not 0 <= v <= 1`
        # lets it through. This is the specific hole worth pinning.
        self._refuse(roi=(0.1, float("nan"), 0.2, 0.5))
        self._refuse(roi=(float("inf"), 0.1, 0.2, 0.5))

    def test_a_short_or_long_roi_is_refused(self):
        self._refuse(roi=(0.1, 0.1, 0.2))
        self._refuse(roi=(0.1, 0.1, 0.2, 0.5, 0.9))

    def test_non_numeric_coordinates_are_refused(self):
        self._refuse(roi=(0.1, "0.1", 0.2, 0.5))
        self._refuse(roi=(0.1, 0.1, 0.2, None))

    def test_a_zone_needs_a_name(self):
        # The name is the key per-track state is held under; an unnamed zone is
        # unreachable rather than merely untidy.
        self._refuse(name="")
        self._refuse(name=None)

    def test_the_error_says_what_is_wrong(self):
        with self.assertRaises(ValueError) as caught:
            Zone("z", "shelf", (0.9, 0.1, 0.2, 0.5))
        message = str(caught.exception)
        self.assertIn("out of order", message)
        self.assertIn("z", message)

    def test_a_valid_zone_still_builds(self):
        self.assertEqual(SHELF.roi, (0.02, 0.22, 0.34, 0.55))


class Geometry(unittest.TestCase):
    def test_contains_is_full_containment_not_centre_in_zone(self):
        # A box that overlaps the zone but hangs outside it is NOT in the zone.
        # Counting a straddling object as present is how a shelf and a basket both
        # claim the same item and the counter doubles.
        self.assertFalse(SHELF.contains((0.30, 0.30, 0.50, 0.40)))
        self.assertFalse(SHELF.contains((0.10, 0.50, 0.20, 0.60)))

    def test_a_zone_contains_its_own_roi(self):
        self.assertTrue(SHELF.contains(SHELF.roi))

    def test_an_inverted_box_is_in_no_zone(self):
        # Not a raise: an inverted detection box is a fact about the decode, and
        # "is it in the zone?" has the answer no.
        self.assertFalse(SHELF.contains((0.20, 0.30, 0.10, 0.40)))

    def test_centroid_of_a_box(self):
        # `assertAlmostEqual`, not `assertEqual`: 0.10 + 0.20 is 0.30000000000000004
        # in binary floating point, so the midpoint comes back as 0.15000000000000002.
        # An exact comparison here would be testing float representation, not the
        # centroid.
        self.assertAlmostEqual(SHELF.centroid_of((0.10, 0.30, 0.20, 0.40))[0], 0.15)
        self.assertAlmostEqual(SHELF.centroid_of((0.10, 0.30, 0.20, 0.40))[1], 0.35)

    def test_to_pixels_is_truncated_not_rounded(self):
        # 0.22 * 1080 = 237.6 and 0.34 * 1920 = 652.8. Truncation gives the brief's
        # (38, 237, 652, 594); rounding gives (38, 238, 653, 594) and the test fails.
        # Pinned because the brief's expected value *is* the truncation, and the
        # reason it is truncation rather than rounding is that the far edge of a
        # drawn zone should not be reported one pixel outside the operator's ROI.
        self.assertEqual(SHELF.to_pixels(1920, 1080), (38, 237, 652, 594))
        self.assertEqual(BASKET.to_pixels(1920, 1080),
                         (1152, 324, 1881, 756))

    def test_to_pixels_stays_on_the_frame(self):
        # A full-frame ROI must not address pixel 1920 or row 1080.
        full = Zone("all", "shelf", (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(full.to_pixels(1920, 1080), (0, 0, 1919, 1079))

    def test_to_pixels_refuses_a_frame_with_no_pixels(self):
        # width=0 would be a ZeroDivisionError deep in a draw call; refused here,
        # with a name in the message.
        with self.assertRaises(ValueError):
            SHELF.to_pixels(0, 1080)
        with self.assertRaises(ValueError):
            SHELF.to_pixels(1920, -1)

    def test_centroid_of_a_short_box_is_refused(self):
        with self.assertRaises(ValueError):
            SHELF.centroid_of((0.1, 0.2, 0.3))


class ContainsPoint(unittest.TestCase):
    """`contains_point` is the presence primitive, and it is not `contains`.

    Measured on `retail_checkout`: a person's box is routinely clipped by the frame
    border (`x1 = -0.001`, `x2 = 0.999`), so **full containment is never true for a
    person in this footage**. A presence rule written on `contains` reports an empty
    shop, which is a false statement about a shop.
    """

    def test_a_point_inside_is_in(self):
        self.assertTrue(SHELF.contains_point((0.10, 0.30)))
        self.assertFalse(SHELF.contains_point((0.90, 0.30)))

    def test_a_point_on_the_edge_is_in(self):
        self.assertTrue(SHELF.contains_point((0.02, 0.22)))
        self.assertTrue(SHELF.contains_point((0.34, 0.55)))

    def test_a_box_clipped_by_the_frame_border_is_still_present(self):
        # The measured case: `x1` is negative and `x2` is at the frame edge.
        zone = Zone("band", "area", (0.0, 0.20, 1.0, 0.60))
        clipped = (-0.001, 0.004, 0.665, 0.613)
        self.assertFalse(zone.contains(clipped),
                         "the box overhangs the frame, so it is not contained")
        self.assertTrue(zone.contains_point(zone.centroid_of(clipped)),
                        "its centre is in the band, which is what presence means")

    def test_a_bad_point_is_refused(self):
        with self.assertRaises(ValueError):
            SHELF.contains_point((0.1,))
        with self.assertRaises(ValueError):
            SHELF.contains_point((float("nan"), 0.3))


class LineBandExtent(unittest.TestCase):
    """M2: `crossed_line` consults the band's `y` extent.

    It used to read `zy1`/`zy2` and discard them, while the docstring said they were
    "the extent along which a crossing counts". A horizontal movement at the bottom of
    the frame therefore counted as crossing a line drawn near the top.
    """

    def test_a_movement_outside_the_band_y_extent_is_not_a_crossing(self):
        self.assertFalse(crossed_line(BELT, (0.30, 0.90), (0.80, 0.92), direction=1))
        self.assertFalse(crossed_line(BELT, (0.30, 0.01), (0.80, 0.02), direction=1))

    def test_a_movement_starting_outside_and_ending_inside_is_not_a_crossing(self):
        # Documented strictness: the rule needs both endpoints within the extent, so an
        # object arriving from below the line does not fire. The looser
        # segment-intersection rule would fire for "arrived" as well as "crossed", which
        # is the one-event-two-reasons problem this function exists to avoid.
        self.assertFalse(crossed_line(BELT, (0.30, 0.90), (0.80, 0.42), direction=1))

    def test_a_movement_inside_the_band_y_extent_still_crosses(self):
        self.assertTrue(crossed_line(BELT, (0.30, 0.20), (0.80, 0.22), direction=1))
        self.assertTrue(crossed_line(BELT, (0.30, 0.56), (0.80, 0.58), direction=1))


class Dwell(unittest.TestCase):
    def test_dwell_accumulates_only_while_inside(self):
        # enters at t=1, still inside at t=3 -> 2 seconds
        self.assertAlmostEqual(dwell_seconds_in_zone(SHELF, (0.1, 0.3, 0.2, 0.4), 1.0, 3.0), 2.0)

    def test_dwell_never_goes_negative(self):
        self.assertEqual(dwell_seconds_in_zone(SHELF, (0.1, 0.3, 0.2, 0.4), 3.0, 1.0), 0.0)

    def test_a_box_outside_the_zone_contributes_no_dwell(self):
        # The brief tests the negative-clamp case but not this one, and this is the
        # case that decides a count: an object that has left the shelf must stop
        # accumulating, or its dwell keeps growing after it is gone.
        self.assertEqual(
            dwell_seconds_in_zone(SHELF, (0.70, 0.30, 0.80, 0.40), 1.0, 9.0), 0.0)

    def test_a_straddling_box_is_not_dwelling(self):
        self.assertEqual(
            dwell_seconds_in_zone(SHELF, (0.30, 0.30, 0.50, 0.40), 1.0, 9.0), 0.0)

    def test_dwell_needs_a_zone(self):
        with self.assertRaises(ValueError):
            dwell_seconds_in_zone("shelf", (0.1, 0.3, 0.2, 0.4), 1.0, 2.0)


class LineCrossing(unittest.TestCase):
    def test_a_left_to_right_crossing_is_detected(self):
        self.assertTrue(crossed_line(BELT, (0.30, 0.40), (0.80, 0.42), direction=1))

    def test_the_reverse_crossing_is_not_the_same_event(self):
        self.assertFalse(crossed_line(BELT, (0.30, 0.40), (0.80, 0.42), direction=-1))

    def test_a_movement_that_never_crosses_is_not_a_crossing(self):
        self.assertFalse(crossed_line(BELT, (0.40, 0.40), (0.50, 0.42), direction=1))

    def test_a_backwards_movement_is_detected_when_asked_for(self):
        # The brief checks only that a forward movement is *not* a backwards one.
        # That passes for an implementation that always returns False. This is the
        # positive case in the other direction.
        self.assertTrue(crossed_line(BELT, (0.80, 0.40), (0.30, 0.42), direction=-1))
        self.assertFalse(crossed_line(BELT, (0.80, 0.40), (0.30, 0.42), direction=1))

    def test_the_band_centre_is_the_line_not_its_near_edge(self):
        # BELT spans x 0.34..0.74, centre 0.54. A jump from 0.50 (inside, left of
        # centre) to 0.60 (inside, right of centre) has crossed the line even
        # though neither endpoint left the band. The centre is the decision because
        # the event is about the line the presenter sees drawn, not about the
        # rectangle's edge.
        self.assertTrue(crossed_line(BELT, (0.50, 0.40), (0.60, 0.42), direction=1))
        self.assertTrue(crossed_line(BELT, (0.60, 0.40), (0.50, 0.42), direction=-1))
        # Staying on the left of the centre is not a crossing, however far it moves.
        self.assertFalse(crossed_line(BELT, (0.35, 0.40), (0.53, 0.42), direction=1))
        self.assertFalse(crossed_line(BELT, (0.55, 0.40), (0.73, 0.42), direction=1))

    def test_a_point_exactly_on_the_centre_counts_as_the_right_hand_side(self):
        # Documented tie-break, pinned so it cannot change silently. The side test is
        # `x < centre`, so a centroid landing exactly on the line is on the far side
        # of it and arriving there counts as a crossing. This is an arbitrary choice
        # at a measure-zero boundary; the reason it is pinned rather than left
        # implicit is that changing it would change event counts, and a rule that
        # can change counts without anyone noticing is not a rule.
        self.assertTrue(crossed_line(BELT, (0.53, 0.40), (0.54, 0.42), direction=1))
        self.assertFalse(crossed_line(BELT, (0.54, 0.40), (0.54, 0.42), direction=1))

    def test_a_stationary_point_never_crosses(self):
        self.assertFalse(crossed_line(BELT, (0.30, 0.40), (0.30, 0.40), direction=1))
        self.assertFalse(crossed_line(BELT, (0.54, 0.40), (0.54, 0.40), direction=1))

    def test_direction_zero_is_refused_not_treated_as_either_way(self):
        # AGENTS.md rule 15: 0 must not stand in for a missing value. If 0 meant
        # "either direction", a caller that had no direction would silently get
        # both a transfer and a return for one movement.
        with self.assertRaises(ValueError):
            crossed_line(BELT, (0.30, 0.40), (0.80, 0.42), direction=0)

    def test_a_non_integer_direction_is_refused(self):
        for bad in (1.0, "1", None, 2):
            with self.assertRaises(ValueError):
                crossed_line(BELT, (0.30, 0.40), (0.80, 0.42), direction=bad)

    def test_crossed_line_refuses_an_area_zone(self):
        # An area zone has no line to cross; answering anyway would invent events
        # for any movement across a shelf rectangle.
        with self.assertRaises(ValueError):
            crossed_line(SHELF, (0.10, 0.30), (0.80, 0.42), direction=1)

    def test_a_bad_point_is_refused_rather_than_treated_as_a_position(self):
        with self.assertRaises(ValueError):
            crossed_line(BELT, (0.30,), (0.80, 0.42), direction=1)
        with self.assertRaises(ValueError):
            crossed_line(BELT, (float("nan"), 0.40), (0.80, 0.42), direction=1)


if __name__ == "__main__":
    unittest.main()