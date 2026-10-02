"""`app/engine/tracker.py` - identity manufactured from a flat list of boxes.

Review Focus #4: *"a track id is silently reused across classes, so a person
becomes a product."* A detector emits boxes with no identity; this module invents
one, and everything downstream - dwell, transfers, the on-screen event log -
inherits whatever this module got wrong. So the tests below are written to fail if
the identity rules are weakened, and three of the brief's own six tests are replaced
because they could not catch the fault they name (noted in place).

Fixtures are plain stand-in objects with `.class_id`, `.box` and `.score` rather
than `app.inference.decode.Detection`, so this module stays decoupled from the
decoder and the test does not need a model.
"""

import unittest

from app.engine.tracker import Track, Tracker, iou


def _d(x1, y1, x2, y2, cls=0):
    return type("D", (), {"class_id": cls, "score": 0.9, "box": (x1, y1, x2, y2)})()


class Tracking(unittest.TestCase):
    def test_a_stationary_object_keeps_one_id(self):
        t = Tracker()
        for f in range(4):
            t.update([_d(100, 100, 200, 200)], now=float(f) / 30)
        self.assertEqual(len(t.active_track_ids()), 1)

    def test_two_separate_objects_get_two_ids(self):
        t = Tracker()
        t.update([_d(0, 0, 50, 50), _d(400, 400, 450, 450)], now=0.0)
        t.update([_d(0, 0, 50, 50), _d(400, 400, 450, 450)], now=0.1)
        self.assertEqual(len(t.active_track_ids()), 2)

    def test_ids_survive_a_small_movement(self):
        t = Tracker()
        a = t.update([_d(100, 100, 200, 200)], now=0.0)[0]
        b = t.update([_d(105, 103, 205, 203)], now=0.1)[0]
        self.assertEqual(a.track_id, b.track_id)

    def test_a_track_is_lost_after_max_misses(self):
        t = Tracker(max_misses=2)
        t.update([_d(0, 0, 50, 50)], now=0.0)
        t.update([], now=0.1)
        t.update([], now=0.2)
        self.assertEqual(t.active_track_ids(), ())

    def test_reset_clears_every_id_so_a_loop_cannot_reuse_one(self):
        t = Tracker()
        t.update([_d(0, 0, 50, 50)], now=0.0)
        before = t.active_track_ids()
        t.reset()
        after = t.update([_d(0, 0, 50, 50)], now=1.0)
        self.assertNotEqual(after[0].track_id, before[0])

    def test_a_class_change_does_not_silently_reuse_the_id(self):
        t = Tracker()
        a = t.update([_d(0, 0, 50, 50, cls=0)], now=0.0)[0]
        b = t.update([_d(0, 0, 50, 50, cls=47)], now=0.1)[0]
        self.assertNotEqual(a.track_id, b.track_id)


class ClassStability(unittest.TestCase):
    """Review Focus #4, in the geometry that actually triggers it.

    The brief's version of this test compares `update(...)[0]` twice - and both
    calls return the lowest-id track, so an implementation that returned a single
    unchanging id would pass it. These below ask the question directly: after a class
    change on the *same box*, is the new track a different one, and did the old one
    survive as a person rather than being relabelled as an apple?
    """

    def test_the_new_class_gets_a_new_id_and_the_old_track_is_still_a_person(self):
        t = Tracker()
        first = t.update([_d(0, 0, 50, 50, cls=0)], now=0.0)[0]
        tracks = t.update([_d(0, 0, 50, 50, cls=47)], now=0.1)
        by_class = {tr.class_id: tr for tr in tracks}
        self.assertEqual(set(by_class), {0, 47})
        self.assertNotEqual(by_class[0].track_id, by_class[47].track_id)
        self.assertEqual(by_class[0].track_id, first.track_id)

    def test_identical_boxes_of_different_classes_never_match(self):
        t = Tracker()
        t.update([_d(0, 0, 100, 100, cls=0)], now=0.0)
        # A 100% overlapping apple. If class were not part of the matching key this
        # would be IoU 1.0 and would take the person's id.
        tracks = t.update([_d(0, 0, 100, 100, cls=47)], now=0.1)
        apples = [tr for tr in tracks if tr.class_id == 47]
        self.assertEqual(len(apples), 1)
        self.assertEqual(apples[0].hits, 1, "the apple inherited the person's hits")

    def test_a_person_standing_at_a_counter_does_not_become_the_counter(self):
        # The real geometry from retail_checkout: a large `dining_table` detection
        # overlapping a `person`. Without the class partition the table's larger box
        # wins on IoU and takes the person's id, and every later count is furniture.
        t = Tracker()
        person = t.update([_d(600, 200, 800, 700, cls=0)], now=0.0)[0]
        t.update([_d(0, 600, 1900, 900, cls=60), _d(600, 200, 800, 700, cls=0)], now=0.1)
        table = [tr for tr in t.tracks() if tr.class_id == 60]
        person_alive = [tr for tr in t.tracks() if tr.class_id == 0]
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0].hits, 1)
        self.assertEqual(len(person_alive), 1)
        self.assertEqual(person_alive[0].track_id, person.track_id)
        self.assertEqual(person_alive[0].hits, 2)


class Lifetime(unittest.TestCase):
    def test_an_unconfirmed_track_is_reported_but_flagged(self):
        # The event machine needs to *see* a tentative track to build dwell on it, so
        # a one-detection track must be listed - but never as evidence.
        t = Tracker(min_hits=2)
        track = t.update([_d(0, 0, 50, 50)], now=0.0)[0]
        self.assertEqual(t.active_track_ids(), (track.track_id,))
        self.assertFalse(track.confirmed)
        self.assertEqual(track.hits, 1)

    def test_confirmation_arrives_at_min_hits_and_stays(self):
        t = Tracker(min_hits=2)
        first = t.update([_d(0, 0, 50, 50)], now=0.0)[0]
        second = t.update([_d(0, 0, 50, 50)], now=0.1)[0]
        self.assertFalse(first.confirmed)
        self.assertTrue(second.confirmed)
        self.assertEqual(second.track_id, first.track_id)
        # and it is not revoked by further frames
        self.assertTrue(t.update([_d(0, 0, 50, 50)], now=0.2)[0].confirmed)

    def test_a_track_that_drifted_is_no_longer_confirmed(self):
        # The box in hand is the last one seen, not a statement about now.
        t = Tracker(min_hits=2, max_misses=3)
        t.update([_d(0, 0, 50, 50)], now=0.0)
        self.assertTrue(t.update([_d(0, 0, 50, 50)], now=0.1)[0].confirmed)
        coasted = t.update([], now=0.2)[0]
        self.assertEqual(coasted.misses, 1)
        self.assertFalse(coasted.confirmed)
        self.assertFalse(coasted.is_current)

    def test_a_coasted_track_is_still_listed_until_max_misses(self):
        t = Tracker(max_misses=3)
        t.update([_d(0, 0, 50, 50)], now=0.0)
        for i, expect_alive in ((1, True), (2, True), (3, False)):
            t.update([], now=0.1 * i)
            self.assertEqual(t.active_track_ids() != (), expect_alive,
                             f"after {i} consecutive misses")

    def test_reacquiring_before_max_misses_keeps_the_same_id(self):
        # A one-frame occlusion is not a new object. Losing the id here would make
        # every brief occlusion cancel a transfer that is still perfectly observable.
        t = Tracker(max_misses=3)
        first = t.update([_d(0, 0, 50, 50)], now=0.0)[0]
        t.update([], now=0.1)
        again = t.update([_d(0, 0, 50, 50)], now=0.2)[0]
        self.assertEqual(again.track_id, first.track_id)
        self.assertEqual(again.hits, 2)
        self.assertEqual(again.misses, 0)
        self.assertTrue(again.is_current)

    def test_ids_are_unique_for_the_whole_lifetime_of_one_tracker(self):
        # Review Focus #3: an id that repeats is an event dedup_key that collides.
        t = Tracker(max_misses=1)
        issued = []
        for frame in range(6):
            issued += [tr.track_id for tr in t.update([_d(0, 0, 50, 50)], now=float(frame))]
            t.reset()
        self.assertEqual(len(issued), len(set(issued)), issued)
        self.assertEqual(issued, sorted(issued))

    def test_reset_clears_tracks_and_the_frame_count_but_not_the_id_counter(self):
        t = Tracker()
        t.update([_d(0, 0, 50, 50)], now=0.0)
        t.update([_d(0, 0, 50, 50)], now=0.1)
        self.assertEqual(t.frames_seen, 2)
        t.reset()
        self.assertEqual(t.active_track_ids(), ())
        self.assertEqual(t.frames_seen, 0)
        self.assertIsNone(t.get(1))

    def test_get_answers_none_for_a_dead_track(self):
        # `on_track_lost` in the event machine is driven by this answer, so `None`
        # has to mean "no such track" and not "track with default values".
        t = Tracker(max_misses=1)
        track = t.update([_d(0, 0, 50, 50)], now=0.0)[0]
        self.assertIsNotNone(t.get(track.track_id))
        t.update([], now=0.1)
        t.update([], now=0.2)
        self.assertIsNone(t.get(track.track_id))


class Matching(unittest.TestCase):
    def test_the_best_overlap_wins_not_the_first_come(self):
        t = Tracker(iou_threshold=0.1)
        # Two overlapping-but-distinct tracks. Note the arrival order: the *worse*
        # match for the next frame comes first, so a first-come implementation would
        # take the lower id and the two tracks would swap objects.
        t.update([_d(40, 40, 140, 140), _d(0, 0, 100, 100)], now=0.0)
        tracks = t.update([_d(0, 0, 100, 100)], now=0.1)
        current = {tr.track_id: tr.hits for tr in tracks if tr.is_current}
        self.assertEqual(current, {2: 2}, "track 1 should have coasted, not matched")
        self.assertEqual(tracks[0].track_id, 2)

    def test_the_assignment_does_not_depend_on_detection_order(self):
        boxes = [_d(0, 0, 100, 100, cls=0), _d(300, 0, 400, 100, cls=1)]
        results = []
        for order in (boxes, list(reversed(boxes))):
            t = Tracker()
            t.update(boxes, now=0.0)
            t.update(boxes, now=0.1)
            results.append({tr.class_id: tr.track_id for tr in t.update(order, now=0.2)})
        self.assertEqual(results[0], results[1])

    def test_a_detection_matches_at_most_one_track(self):
        t = Tracker(iou_threshold=0.3)
        t.update([_d(0, 0, 100, 100), _d(2, 2, 102, 102)], now=0.0)
        tracks = t.update([_d(1, 1, 101, 101)], now=0.1)
        # One detection. It can have matched only one of them; the other must coast.
        self.assertEqual(sum(1 for tr in tracks if tr.is_current), 1)
        self.assertEqual(sum(1 for tr in tracks if tr.misses == 1), 1)

    def test_two_near_identical_detections_get_two_tracks(self):
        # NMS already suppressed duplicates upstream; if one still arrives, it is a
        # separate detection and deserves a separate id rather than being dropped.
        t = Tracker()
        tracks = t.update([_d(0, 0, 100, 100), _d(500, 500, 600, 600)], now=0.0)
        self.assertEqual(len(tracks), 2)

    def test_a_degenerate_detection_never_matches_anything(self):
        # Zero-area: IoU is undefined. A guard that divides anyway raises; one that
        # guards and returns 0.0 matches everything at threshold 0. Either way a
        # degenerate box must not become a track that follows the scene.
        t = Tracker(iou_threshold=0.0)
        tracks = t.update([_d(5, 5, 5, 5)], now=0.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].hits, 1)
        self.assertEqual(iou((5, 5, 5, 5), (0, 0, 100, 100)), 0.0)

    def test_iou_of_disjoint_boxes_is_zero(self):
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)


class OutputOrder(unittest.TestCase):
    def test_update_returns_live_tracks_by_id(self):
        t = Tracker()
        t.update([_d(400, 400, 450, 450), _d(0, 0, 50, 50)], now=0.0)
        ids = [tr.track_id for tr in t.update([_d(400, 400, 450, 450),
                                               _d(0, 0, 50, 50)], now=0.1)]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(ids, list(t.active_track_ids()))


class CurrentTracks(unittest.TestCase):
    """I3: the accessor an overlay or the event machine must use.

    The failure it avoids is silent and expensive. A coasted track's `box` is the last
    one *seen*; drawing it draws a plausible, correctly-placed rectangle from a recent
    frame on top of an object that has since left. Nothing about that rectangle looks
    wrong, which is why the filter has to be a named, tested call.
    """

    def test_current_tracks_excludes_coasted_and_unmatched_ones(self):
        t = Tracker(max_misses=3)
        t.update([_d(0, 0, 50, 50), _d(400, 400, 450, 450)], now=0.0)
        # One frame where only the first object is seen: the second coasts.
        current = t.update([_d(0, 0, 50, 50)], now=0.1)
        self.assertEqual(len(t.tracks()), 2, "the coasted track is still live")
        self.assertEqual(len(t.current_tracks()), 1)
        self.assertEqual([tr.track_id for tr in current if tr.is_current],
                         [tr.track_id for tr in t.current_tracks()])

    def test_a_track_returns_to_current_when_it_is_matched_again(self):
        t = Tracker(max_misses=3)
        t.update([_d(0, 0, 50, 50)], now=0.0)
        t.update([], now=0.1)
        self.assertEqual(t.current_tracks(), ())
        again = t.update([_d(0, 0, 50, 50)], now=0.2)
        self.assertEqual(len(t.current_tracks()), 1)
        self.assertTrue(again[0].is_current)

    def test_current_tracks_is_empty_when_nothing_was_seen_this_frame(self):
        t = Tracker(max_misses=3)
        t.update([_d(0, 0, 50, 50)], now=0.0)
        t.update([], now=0.1)
        self.assertEqual(t.current_tracks(), ())

    def test_it_is_ordered_by_track_id(self):
        t = Tracker()
        t.update([_d(400, 400, 450, 450), _d(0, 0, 50, 50)], now=0.0)
        t.update([_d(400, 400, 450, 450), _d(0, 0, 50, 50)], now=0.1)
        ids = [tr.track_id for tr in t.current_tracks()]
        self.assertEqual(ids, sorted(ids))


class Validation(unittest.TestCase):
    def _refuse_ctor(self, **kw):
        with self.assertRaises(ValueError):
            Tracker(**kw)

    def test_iou_threshold_must_be_a_real_number_in_range(self):
        self._refuse_ctor(iou_threshold=-0.1)
        self._refuse_ctor(iou_threshold=1.5)
        self._refuse_ctor(iou_threshold="0.3")
        self.assertEqual(Tracker(iou_threshold=0.45).iou_threshold, 0.45)

    def test_max_misses_of_zero_would_delete_every_track_on_creation(self):
        # `misses=0 >= max_misses=0` would drop a track on the very frame it was
        # born, so nothing could ever reach `min_hits`.
        self._refuse_ctor(max_misses=0)
        self._refuse_ctor(max_misses=-1)
        self._refuse_ctor(min_hits=0)

    def test_a_non_finite_now_is_refused(self):
        # A nan timestamp makes every later comparison False, which would silently
        # stop every dwell and every transition check rather than crash.
        t = Tracker()
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                t.update([_d(0, 0, 50, 50)], now=bad)

    def test_a_string_is_not_a_sequence_of_detections(self):
        with self.assertRaises(ValueError):
            Tracker().update("not detections", now=0.0)

    def test_an_empty_detection_list_coasts_everything(self):
        t = Tracker()
        self.assertEqual(t.update([], now=0.0), ())
        self.assertEqual(t.update([], now=0.1), ())
        self.assertEqual(t.active_track_ids(), ())


class TrackRecord(unittest.TestCase):
    def test_a_track_refuses_the_values_that_make_it_unusable(self):
        base = dict(track_id=1, class_id=0, box=(0, 0, 10, 10), first_seen=0.0,
                    last_seen=0.0, hits=1, misses=0, confirmed=False)
        for override, why in (
            (dict(track_id=0), "0 must never be a track id: an event dedup_key "
                               "containing it would look valid"),
            (dict(track_id=-1), "negative ids sort before every real one"),
            (dict(hits=0), "a track with no hits was never observed"),
            (dict(misses=-1), "misses counts frames since the last match"),
            (dict(confirmed="yes"), "confirmed is the event machine's counting gate, "
                                   "so a truthy non-bool must not read as evidence"),
            (dict(box=(0, 0, 10)), "a 3-tuple is not a box"),
        ):
            with self.assertRaises(ValueError, msg=why):
                Track(**{**base, **override})

    def test_centroid_of_a_track(self):
        track = Track(track_id=1, class_id=0, box=(0, 0, 10, 20), first_seen=0.0,
                      last_seen=0.0, hits=1, misses=0, confirmed=False)
        self.assertEqual(track.centroid(), (5.0, 10.0))

    def test_is_current_is_false_the_frame_after_a_miss(self):
        current = Track(track_id=1, class_id=0, box=(0, 0, 10, 10), first_seen=0.0,
                        last_seen=0.1, hits=2, misses=0, confirmed=True)
        coasted = Track(track_id=1, class_id=0, box=(0, 0, 10, 10), first_seen=0.0,
                        last_seen=0.1, hits=2, misses=1, confirmed=True)
        self.assertTrue(current.is_current)
        self.assertFalse(coasted.is_current)


if __name__ == "__main__":
    unittest.main()