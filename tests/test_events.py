"""`app/engine/events.py` - the retail event state machine. The product's claim.

Spec §4: `visible_on_shelf -> exited_shelf -> entered_basket -> candidate_transfer`,
with the same stable track, a minimum dwell, a bounded transition interval and a
dedup key. A reverse movement is a **separate** return event. Occlusion or a lost
track identity means **uncertain or not counted, never a completed transfer**. Gaps
are never filled with assumed detections.

This file is where Review Focus #2 and #3 are answered, so most of what follows is
about the ways the machine must *refuse* to produce an event:

- **A lost track cancels, it never completes.** `on_track_lost` discards that track's
  pending state outright. Re-observing the track id afterwards starts from nothing,
  because after an occlusion we do not know whether it is the same object. The
  cancellation is counted (`abandoned_transfers`) so the number that reaches the
  screen can say that a transfer was dropped rather than quietly being smaller.
- **Dwell is measured between two real observations.** `last_seen_in_shelf` minus
  `first_seen_in_shelf`. It is never interpolated and never carried across a gap.
- **The transition interval is bounded from the last shelf observation**, so a track
  seen in the shelf at t=0 and in the basket at t=99 does not transfer.
- **A return needs a transfer behind it.** `candidate_return` means "back where it
  came from"; without a counted transfer in this loop there is nothing to return
  from, so a shelf->basket->shelf jiggle in one second emits nothing.

**The machine is pure.** `now` is `frame.captured_at`; no clock is read, nothing is
opened, nothing is written. Every event carries the loop and frame it was observed on
so a caller can check it against what is on screen.

**On the zones, plainly.** `retail_checkout` contains **no shelf**: 72 frames were
reviewed and the clip is a self-checkout whose basket is already full at frame 0, and
the movement visible is basket -> scanner plate. See
`docs/qualification/M0-content-gate.md` §3-§4. So the slot names `"shelf"` and
`"basket"` in this test file are the **state machine's semantic fixtures** - they
exercise the geometry without claiming the footage contains them. The honest
retail configuration is `basket` -> `checkout`, and `test_a_basket_to_checkout_move_is_a_transfer`
below drives exactly that, so the machine is proven against the scenario that will
actually ship. Nothing here may be read as the shelf-to-basket claim being met.
"""

import unittest

import numpy as np

from app.contracts.frame import FramePacket
from app.contracts.scenario_manifest import ScenarioManifest
from app.engine.events import BusinessEvent, RetailEventMachine
from app.engine.tracker import Track
from app.engine.zones import Zone

# Normalised test-fixture slots. NOT a claim about the footage - see the module note.
SHELF = (0.02, 0.22, 0.34, 0.55)
BASKET = (0.60, 0.30, 0.98, 0.70)

# Boxes on a 1920x1080 frame, chosen to be **measured** inside the ROIs above:
#   (80, 300, 400, 500)   -> (0.0417, 0.2778, 0.2083, 0.4630)  inside SHELF
#   (1200, 380, 1700, 700) -> (0.6250, 0.3519, 0.8854, 0.6481)  inside BASKET
IN_SHELF = (80, 300, 400, 500)
IN_BASKET = (1200, 380, 1700, 700)

# The pinned label list has 80 entries and 'apple' is 47 (COCO). Supplied explicitly
# so this module's tests do not read the provisioned labels.txt.
CLASS_NAMES = tuple(f"class_{i}" for i in range(47)) + ("apple",) + tuple(
    f"class_{i}" for i in range(48, 80))
PERSON_ID = 0
APPLE_ID = 47


def _manifest(**kw):
    base = dict(
        id="retail", version=1, title="Retail / checkout zone",
        business_line="Items entering the checkout zone",
        class_allowlist=("person", "apple"),
        zones=(),
        event_rules={
            "min_dwell_seconds": 0.5, "max_transition_seconds": 12.0,
            "shelf": SHELF, "basket": BASKET, "tracked_classes": ("person", "apple"),
        },
        source_ids=("retail_checkout",), conf_threshold=0.25, iou_threshold=0.45,
    )
    base.update(kw)
    return ScenarioManifest(**base)


def _frame(fid, t, loop=0):
    return FramePacket(session_id="s", scenario_id="retail", stream_id="A",
                       frame_id=fid, source_pts=t, source_loop_id=loop,
                       captured_at=t, decoded_at=t, width=1920, height=1080,
                       image=np.zeros((1080, 1920, 3), np.uint8))


def _track(tid, box, t, cls=APPLE_ID, confirmed=True, hits=5, misses=0):
    return Track(track_id=tid, class_id=cls, box=box, first_seen=t, last_seen=t,
                 hits=hits, misses=misses, confirmed=confirmed)


class Transfer(unittest.TestCase):
    def setUp(self):
        self.m = RetailEventMachine(_manifest(), class_names=CLASS_NAMES)

    def _walk(self, machine, tid=1, loop=0, fid=0, cls=APPLE_ID, origin=IN_SHELF,
              destination=IN_BASKET):
        """origin -> destination, emitting events as we go."""
        events = []
        steps = [
            (0.0, origin),    # inside the origin zone
            (0.6, origin),    # dwell satisfied
            (1.0, destination),
            (1.4, destination),
        ]
        for t, box in steps:
            events += machine.observe(_track(tid, box, t, cls=cls), _frame(fid, t, loop))
            fid += 1
        return events

    def test_a_sustained_origin_to_destination_move_is_a_candidate_transfer(self):
        kinds = [e.kind for e in self._walk(self.m)]
        self.assertIn("candidate_transfer", kinds)

    def test_minimum_dwell_is_required(self):
        # The brief's version of this test observes origin, destination, destination
        # and asserts only on the **third** observation. That cannot catch an
        # implementation with no dwell check at all: such an implementation fires the
        # transfer on the *second* observation and is already `transferred` by the
        # third, so the third correctly returns nothing and the test passes. The
        # assertion that matters is on the frame the track first arrives.
        first = self.m.observe(_track(1, IN_SHELF, 0.0), _frame(0, 0.0))
        self.assertEqual([e.kind for e in first], [])
        second = self.m.observe(_track(1, IN_BASKET, 0.2), _frame(1, 0.2))
        self.assertEqual(
            [e.kind for e in second], [],
            "0.2s of dwell is under the 0.5s the manifest requires, so arriving in the "
            "destination is not yet a transfer")
        third = self.m.observe(_track(1, IN_BASKET, 0.4), _frame(2, 0.4))
        self.assertEqual([e.kind for e in third], [])

    def test_the_dwell_threshold_is_exactly_the_manifests(self):
        # Both sides of the boundary, so a machine that demanded *more* dwell than the
        # manifest states fails here as surely as one that demands none. Dwell is
        # measured between two observations in the origin, so the boundary is set by
        # the second origin frame.
        def run(origin_frames):
            m = RetailEventMachine(_manifest(), class_names=CLASS_NAMES)
            for i, t in enumerate(origin_frames):
                m.observe(_track(1, IN_SHELF, t), _frame(i, t))
            last = origin_frames[-1]
            return [e.kind for e in
                    m.observe(_track(1, IN_BASKET, last + 0.1), _frame(99, last + 0.1))]

        self.assertEqual(run([0.0, 0.49]), [], "0.49s of dwell is under the 0.5s required")
        self.assertEqual(run([0.0, 0.50]), ["candidate_transfer"],
                         "exactly 0.5s satisfies the manifest's requirement")

    def test_the_same_transfer_fires_once_not_once_per_frame(self):
        events = self._walk(self.m)
        self.assertEqual(len([e for e in events if e.kind == "candidate_transfer"]), 1)

    def test_a_transition_slower_than_the_maximum_does_not_count(self):
        self.m.observe(_track(1, IN_SHELF, 0.0), _frame(0, 0.0))
        self.m.observe(_track(1, IN_SHELF, 0.6), _frame(1, 0.6))
        late = self.m.observe(_track(1, IN_BASKET, 99.0), _frame(2, 99.0))
        self.assertEqual([e for e in late if e.kind == "candidate_transfer"], [])

    def test_losing_track_identity_cancels_the_transfer(self):
        # Review Focus #2: occlusion must not complete a transfer.
        self.m.observe(_track(1, IN_SHELF, 0.0), _frame(0, 0.0))
        self.m.observe(_track(1, IN_SHELF, 0.6), _frame(1, 0.6))
        self.m.on_track_lost(1, 0.8)
        self.m.observe(_track(1, IN_BASKET, 1.4), _frame(2, 1.4))
        self.assertEqual(self.m.events, [])

    def test_a_cancelled_transfer_is_counted_not_silently_dropped(self):
        # The screen must be able to say "one transfer was dropped by an occlusion"
        # rather than simply showing a smaller number.
        self.m.observe(_track(1, IN_SHELF, 0.0), _frame(0, 0.0))
        self.m.observe(_track(1, IN_SHELF, 0.6), _frame(1, 0.6))
        self.m.on_track_lost(1, 0.8)
        self.assertEqual(self.m.abandoned_transfers, 1)
        self.assertEqual(self.m.abandoned_reasons["track_lost"], 1)
        # Losing a track that had no pending transfer is not a cancellation.
        self.m.on_track_lost(99, 0.9)
        self.assertEqual(self.m.abandoned_transfers, 1)

    def test_a_return_movement_is_a_separate_event_kind(self):
        self._walk(self.m)
        more = []
        for t, box in [(2.0, IN_BASKET), (2.6, IN_BASKET),
                       (3.0, IN_SHELF), (3.4, IN_SHELF)]:
            more += self.m.observe(_track(1, box, t), _frame(20 + int(t), t))
        self.assertIn("candidate_return", [e.kind for e in more])

    def test_a_return_needs_a_transfer_behind_it(self):
        # origin -> destination -> origin with no dwell is a jiggle, not a return.
        for t, box in [(0.0, IN_SHELF), (0.1, IN_BASKET),
                       (0.2, IN_SHELF), (0.3, IN_SHELF)]:
            self.m.observe(_track(1, box, t), _frame(int(t * 10), t))
        self.assertEqual([e.kind for e in self.m.events if e.kind == "candidate_return"], [])

    def test_a_return_fires_once(self):
        self._walk(self.m)
        more = []
        for t, box in [(2.0, IN_BASKET), (2.6, IN_BASKET), (3.0, IN_SHELF),
                       (3.6, IN_SHELF), (4.0, IN_SHELF)]:
            more += self.m.observe(_track(1, box, t), _frame(100 + int(t), t))
        self.assertEqual(len([e for e in more if e.kind == "candidate_return"]), 1)

    def test_events_carry_a_dedup_key_and_their_frame(self):
        ev = [e for e in self._walk(self.m) if e.kind == "candidate_transfer"][0]
        self.assertTrue(ev.dedup_key)
        self.assertGreaterEqual(ev.at_frame_id, 0)
        self.assertFalse(ev.uncertain)

    def test_an_unconfirmed_track_is_never_counted(self):
        weak = _track(9, IN_SHELF, 0.0, confirmed=False, hits=1)
        self.m.observe(weak, _frame(0, 0.0))
        self.assertEqual(self.m.events, [])

    def test_an_unconfirmed_track_accrues_no_dwell_and_so_cannot_transfer(self):
        # The honest consequence of refusing unconfirmed observations: they leave no
        # state, so a track that was never confirmed in the origin has no dwell to
        # spend and cannot transfer on the strength of frames nobody vouched for.
        m = RetailEventMachine(_manifest(), class_names=CLASS_NAMES)
        for t in (0.0, 0.6):
            m.observe(_track(1, IN_SHELF, t, confirmed=False, hits=1), _frame(0, t))
        events = m.observe(_track(1, IN_BASKET, 1.0), _frame(1, 1.0))
        self.assertEqual([e.kind for e in events], [])
        self.assertEqual(m.state_of(1), "unseen")

    def test_a_track_confirmed_in_the_origin_transfers_normally(self):
        # The positive counterpart: once the tracker vouches for the track, its frames
        # count and the dwell requirement is met in the ordinary way.
        m = RetailEventMachine(_manifest(), class_names=CLASS_NAMES)
        for t in (0.0, 0.6):
            m.observe(_track(1, IN_SHELF, t), _frame(0, t))
        events = m.observe(_track(1, IN_BASKET, 1.0), _frame(1, 1.0))
        self.assertEqual([e.kind for e in events], ["candidate_transfer"])

    def test_a_coasted_track_is_not_observed_as_being_present(self):
        # `misses > 0` means the box in hand is the last one *seen*. Counting it as
        # present now would place the object somewhere nobody observed it, and would
        # keep accruing dwell for something that may have left the room.
        m = RetailEventMachine(_manifest(), class_names=CLASS_NAMES)
        m.observe(_track(1, IN_SHELF, 0.0), _frame(0, 0.0))
        m.observe(_track(1, IN_SHELF, 0.6, misses=1), _frame(1, 0.6))
        events = m.observe(_track(1, IN_BASKET, 1.0, misses=1), _frame(2, 1.0))
        self.assertEqual([e.kind for e in events], [])
        self.assertEqual(m.abandoned_transfers, 0)


class LoopBoundary(unittest.TestCase):
    """Review Focus #3: the same transfer must not fire twice across a loop."""

    def setUp(self):
        self.m = RetailEventMachine(_manifest(), class_names=CLASS_NAMES)

    def _walk(self, machine, loop=0, fid=0):
        events = []
        for t, box in [(0.0, IN_SHELF), (0.6, IN_SHELF),
                       (1.0, IN_BASKET), (1.4, IN_BASKET)]:
            events += machine.observe(_track(1, box, t), _frame(fid, t, loop))
            fid += 1
        return events

    def test_a_new_loop_resets_counters_and_clears_events(self):
        self._walk(self.m, loop=0)
        self.m.reset(loop_id=1)
        self.assertEqual(self.m.events, [])
        events = self._walk(self.m, loop=1, fid=1000)
        self.assertEqual(len([e for e in events if e.kind == "candidate_transfer"]), 1)
        self.assertEqual({e.source_loop_id for e in events}, {1})

    def test_reset_also_clears_the_dedup_set_and_the_abandon_count(self):
        self._walk(self.m, loop=0)
        self.m.on_track_lost(1, 2.0)
        self.m.reset(loop_id=1)
        self.assertEqual(self.m.abandoned_transfers, 0)
        self.assertEqual(self.m.abandoned_reasons, {})
        # A replay of loop 0's frames under loop 1 must still transfer exactly once,
        # which it would not if the dedup set had survived the reset.
        events = self._walk(self.m, loop=1, fid=0)
        self.assertEqual(len([e for e in events if e.kind == "candidate_transfer"]), 1)

    def test_a_frame_from_an_unannounced_loop_is_refused(self):
        # Spec §9 requires an explicit reset per loop. Silently adopting a new loop id
        # here would let one loop's pending state leak into the next, which is
        # Review Focus #3 in its other direction.
        self.m.observe(_track(1, IN_SHELF, 0.0), _frame(0, 0.0, loop=0))
        with self.assertRaises(ValueError):
            self.m.observe(_track(1, IN_SHELF, 0.6), _frame(1, 0.6, loop=1))


class DedupKey(unittest.TestCase):
    def setUp(self):
        self.m = RetailEventMachine(_manifest(), class_names=CLASS_NAMES)

    def _transfer(self, tid, loop=0, fid=0):
        for t, box in [(0.0, IN_SHELF), (0.6, IN_SHELF), (1.0, IN_BASKET)]:
            out = self.m.observe(_track(tid, box, t), _frame(fid, t, loop))
            fid += 1
        return [e for e in out if e.kind == "candidate_transfer"]

    def test_the_key_names_the_loop_the_track_and_the_class(self):
        ev = self._transfer(7, loop=2, fid=10)[0]
        self.assertEqual(ev.dedup_key, "2:7:apple:transfer")
        self.assertEqual(ev.track_id, 7)
        self.assertEqual(ev.class_name, "apple")
        self.assertEqual(ev.source_loop_id, 2)
        self.assertEqual(ev.at_frame_id, 12)

    def test_two_transfers_of_the_same_object_in_one_loop_have_different_keys(self):
        # Two apples, both basket -> checkout: distinct keys, so neither is deduped
        # against the other. Without the track id in the key one would be swallowed.
        keys = {self._transfer(1)[0].dedup_key, self._transfer(2)[0].dedup_key}
        self.assertEqual(len(keys), 2)

    def test_the_same_key_is_never_emitted_twice(self):
        first = self._transfer(1)
        self.assertEqual(len(first), 1)
        # Replay the identical sequence - same loop, same track, same frame ids. The
        # dedup key is what makes the second pass a no-op rather than a second count.
        replay = self._transfer(1)
        self.assertEqual(replay, [], "the replayed transfer was counted a second time")
        self.assertEqual(len(self.m.events), 1)
        self.assertEqual(self.m.events[0].dedup_key, first[0].dedup_key)

    def test_reset_clears_the_dedup_set_even_for_the_same_loop_id(self):
        # Because the dedup key embeds the loop id, a *new* loop's keys can never
        # collide with the old loop's, so clearing `_emitted` is only observable when
        # the same loop id is re-announced. That is a real case: a stream restarting
        # within one pass of the clip. `reset(loop_id)` means "start clean on this
        # loop", so the sequence is countable again - and a caller that wants it
        # counted only once across restarts keeps its own set of seen keys.
        self._transfer(1, loop=0)
        self.assertEqual(len(self.m.events), 1)
        self.m.reset(loop_id=0)
        self.assertEqual(self.m.events, [])
        again = self._transfer(1, loop=0)
        self.assertEqual(len(again), 1)
        self.assertEqual(again[0].dedup_key, "0:1:apple:transfer")

    def test_a_transfer_after_a_loop_reset_gets_a_new_key_for_the_new_loop(self):
        self._transfer(1, loop=0)
        self.m.reset(loop_id=1)
        # `reset` clears the event log as well as the dedup set (spec section 9), so
        # what remains is loop 1's work only. A caller wanting a running total across
        # loops must accumulate as it goes - the machine never presents a number it
        # has not counted in the loop being described.
        self.assertEqual(self.m.events, [])
        later = self._transfer(1, loop=1)
        self.assertEqual(len(later), 1)
        self.assertEqual(later[0].dedup_key, "1:1:apple:transfer")
        self.assertEqual(self.m.events[0].source_loop_id, 1)


class TrackedClasses(unittest.TestCase):
    def setUp(self):
        self.m = RetailEventMachine(_manifest(), class_names=CLASS_NAMES)

    def test_a_class_outside_tracked_classes_moves_without_producing_an_event(self):
        # A person walks the whole route. Nobody wants "person entered the basket" on
        # a retail loss screen, and the manifest is where that is decided.
        for t, box in [(0.0, IN_SHELF), (0.6, IN_SHELF),
                       (1.0, IN_BASKET), (1.4, IN_BASKET)]:
            self.m.observe(_track(1, box, t, cls=PERSON_ID), _frame(int(t * 10), t))
        self.assertEqual(self.m.events, [])

    def test_the_tracked_classes_come_from_the_manifest(self):
        self.assertEqual(self.m.rule.classes, ("person", "apple"))
        self.assertEqual(self.m.rule.min_dwell_seconds, 0.5)
        self.assertEqual(self.m.rule.max_transition_seconds, 12.0)

    def test_a_class_id_past_the_end_of_the_label_list_is_refused(self):
        # An out-of-range class id must not become the string "unknown" on a screen:
        # that is a fabricated class name presented as a measurement.
        for t, box in [(0.0, IN_SHELF), (0.6, IN_SHELF), (1.0, IN_BASKET)]:
            with self.assertRaises(ValueError):
                self.m.observe(_track(1, box, t, cls=9999), _frame(int(t * 10), t))

    def test_a_class_not_in_the_allowlist_is_refused_at_construction(self):
        # The allowlist and the tracked list have to agree, or the machine would
        # silently track something the pack says it does not show.
        bad = _manifest(event_rules={"min_dwell_seconds": 0.5,
                                     "max_transition_seconds": 12.0,
                                     "shelf": SHELF, "basket": BASKET,
                                     "tracked_classes": ("person", "banana")})
        with self.assertRaises(ValueError):
            RetailEventMachine(bad, class_names=CLASS_NAMES)


class TheFootageHonestConfiguration(unittest.TestCase):
    """`retail_checkout` is basket -> scanner plate. There is no shelf in it.

    This is the configuration that will actually ship, driven end to end, so the
    machine is proven against the reduced scope rather than against a fixture that
    presumes the unfilmed one. `docs/qualification/M0-content-gate.md` §3-§4.
    """

    # The basket is at the left of the frame; the scanner plate is centre-right.
    BASKET_ZONE = (0.00, 0.70, 0.22, 0.99)     # the black basket, bottom-left
    CHECKOUT_ZONE = (0.35, 0.30, 0.75, 0.62)   # the recessed scanner plate
    IN_BASKET_ZONE = (20, 780, 380, 1060)
    IN_CHECKOUT_ZONE = (700, 340, 1400, 660)

    def _manifest(self):
        return _manifest(event_rules={
            "min_dwell_seconds": 0.5, "max_transition_seconds": 12.0,
            "basket": self.BASKET_ZONE, "checkout": self.CHECKOUT_ZONE,
            # The slots are named in data. The machine's defaults are `shelf`/`basket`,
            # which this footage cannot honour: there is no shelf in it.
            "origin_slot": "basket", "destination_slot": "checkout",
            "tracked_classes": ("apple",),
        }, business_line="Items entering the checkout zone")

    def test_a_basket_to_checkout_move_is_a_candidate_transfer(self):
        m = RetailEventMachine(self._manifest(), class_names=CLASS_NAMES)
        for t, box in [(0.0, self.IN_BASKET_ZONE), (0.6, self.IN_BASKET_ZONE),
                       (1.0, self.IN_CHECKOUT_ZONE), (1.4, self.IN_CHECKOUT_ZONE)]:
            m.observe(_track(1, box, t), _frame(int(t * 10), t))
        kinds = [e.kind for e in m.events]
        self.assertIn("candidate_transfer", kinds)
        self.assertEqual(m.events[0].class_name, "apple")

    def test_the_business_line_is_the_reduced_scope_sentence(self):
        m = RetailEventMachine(self._manifest(), class_names=CLASS_NAMES)
        self.assertEqual(m.manifest.business_line, "Items entering the checkout zone")
        self.assertNotIn("shelf", m.manifest.business_line.lower())

    def test_the_origin_slot_is_whatever_the_manifest_names(self):
        # The machine has no hardcoded "shelf"; the slots come from event_rules, so a
        # pack whose footage has no shelf simply does not declare one.
        m = RetailEventMachine(self._manifest(), class_names=CLASS_NAMES)
        self.assertEqual(set(m.zones), {"basket", "checkout"})


class Purity(unittest.TestCase):
    def test_the_machine_reads_no_clock(self):
        # Two identically-driven machines must produce identical events, including
        # identical timestamps. If any clock were read, the second run would differ.
        def run():
            m = RetailEventMachine(_manifest(), class_names=CLASS_NAMES)
            for t, box in [(0.0, IN_SHELF), (0.6, IN_SHELF), (1.0, IN_BASKET)]:
                m.observe(_track(1, box, t), _frame(0, t))
            return m.events
        self.assertEqual(run(), run())

    def test_event_time_is_the_frame_capture_time(self):
        m = RetailEventMachine(_manifest(), class_names=CLASS_NAMES)
        for t, box in [(0.0, IN_SHELF), (0.6, IN_SHELF), (1.0, IN_BASKET)]:
            m.observe(_track(1, box, t), _frame(3, t + 1000.0))
        self.assertEqual(m.events[0].at_time, 1001.0)


class Records(unittest.TestCase):
    def test_an_event_is_immutable(self):
        ev = BusinessEvent(kind="candidate_transfer", class_name="apple", track_id=1,
                           source_loop_id=0, at_frame_id=1, at_time=1.0,
                           dedup_key="0:1:apple:transfer", uncertain=False)
        with self.assertRaises(Exception):
            ev.kind = "candidate_return"

    def test_a_business_event_needs_a_non_empty_dedup_key(self):
        with self.assertRaises(ValueError):
            BusinessEvent(kind="candidate_transfer", class_name="apple", track_id=1,
                          source_loop_id=0, at_frame_id=1, at_time=1.0,
                          dedup_key="", uncertain=False)

    def test_a_business_event_needs_a_known_kind(self):
        with self.assertRaises(ValueError):
            BusinessEvent(kind="definitely_happened", class_name="apple", track_id=1,
                          source_loop_id=0, at_frame_id=1, at_time=1.0,
                          dedup_key="0:1:apple:x", uncertain=False)

    def test_an_event_rejects_a_track_id_of_zero(self):
        # 0 must never be a track id: it would produce a dedup_key that looks real.
        with self.assertRaises(ValueError):
            BusinessEvent(kind="candidate_transfer", class_name="apple", track_id=0,
                          source_loop_id=0, at_frame_id=1, at_time=1.0,
                          dedup_key="0:0:apple:transfer", uncertain=False)

    def test_observed_events_are_never_uncertain(self):
        # This milestone's rule is "do not count it", never "count it as uncertain",
        # so the flag is pinned False rather than left to drift.
        m = RetailEventMachine(_manifest(), class_names=CLASS_NAMES)
        for t, box in [(0.0, IN_SHELF), (0.6, IN_SHELF), (1.0, IN_BASKET)]:
            m.observe(_track(1, box, t), _frame(int(t * 10), t))
        self.assertTrue(m.events)
        self.assertTrue(all(e.uncertain is False for e in m.events))

    def test_a_zone_is_needed_to_build_a_machine(self):
        with self.assertRaises(ValueError):
            RetailEventMachine(_manifest(event_rules={"min_dwell_seconds": 0.5,
                                                      "max_transition_seconds": 12.0,
                                                      "tracked_classes": ("apple",)}),
                               class_names=CLASS_NAMES)

    def test_the_machine_requires_the_origin_and_destination_slots(self):
        with self.assertRaises(ValueError):
            RetailEventMachine(_manifest(event_rules={"min_dwell_seconds": 0.5,
                                                      "max_transition_seconds": 12.0,
                                                      "shelf": SHELF,
                                                      "tracked_classes": ("apple",)}),
                               class_names=CLASS_NAMES)

    def test_the_manifest_must_be_a_manifest(self):
        with self.assertRaises(ValueError):
            RetailEventMachine({"id": "retail"}, class_names=CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()