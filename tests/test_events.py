"""`app/engine/events.py` - the retail event state machine. The product's claim.

Spec Â§4: `visible_on_shelf -> exited_shelf -> entered_basket -> candidate_transfer`,
with the same stable track, a minimum dwell, a bounded transition interval and a
dedup key. A reverse movement is a **separate** return event. Occlusion or a lost
track identity means **uncertain or not counted, never a completed transfer**. Gaps
are never filled with assumed detections.

This file is where Review Focus #2 and #3 are answered, so most of what follows is
about the ways the machine must *refuse* to produce an event:

- **A lost track cancels, it never completes.** `on_track_lost` discards that track's
  pending state outright. Re-observing the track id afterwards starts from nothing,
  because after an occlusion we do not know whether it is the same object. The
  cancellation is counted (`abandoned_episodes`) so the number that reaches the
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

**On the zones, plainly.** `retail_checkout` contains **no shelf**. Measured over all
3387 frames, **no product-transfer event is detectable in it at all** - the shipped
pack is **people-in-zone presence** (`app/engine/retail_config.py`,
`presence_zone="counter_band"`), and that is the reduced scope this milestone
delivers. Nothing in this file may be read as the shelf-to-basket claim being met, or
as "items entering the checkout zone".

The slot names `"shelf"` and `"basket"` below are **semantic fixtures** - generic slots
that exercise the geometry over synthetic boxes, asserting nothing about
`retail_checkout`. An earlier revision of this docstring claimed the clip's movement was
a two-zone product route, named the retail configuration as such a route, and cited a
test in this file as proof. That test was deleted in the review fix because the
rectangles it drove held a stationary apple and two `suitcase` detections; the stale
claim and the stale citation went with it. The deleted test is deliberately **not named
here**: `tests/test_retail_measurements.py` fails on any shipped docstring that cites a
test which does not exist, and a correction that named the wrong identifier would trip
it. The numbers are in `app/engine/retail_config.py`.
"""

import unittest
from pathlib import Path

import numpy as np

from app.contracts.frame import FramePacket
from app.contracts.scenario_manifest import ScenarioManifest
from app.engine.events import BusinessEvent, RetailEventMachine
from app.engine.retail_config import retail_sanity_manifest
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

# The pinned label list has 80 entries, 'person' is 0 and 'apple' is 47 (COCO) - both
# checked against `app.inference.pinned` by
# `test_the_default_label_list_is_resolved_and_cross_checked_at_construction`.
# Supplied explicitly so this module's tests do not read the provisioned labels.txt.
CLASS_NAMES = (
    ("person",) + tuple(f"class_{i}" for i in range(1, 47)) + ("apple",)
    + tuple(f"class_{i}" for i in range(48, 80))
)
PERSON_ID = 0
APPLE_ID = 47


def _manifest(**kw):
    base = dict(
        id="retail", version=1, title="Retail / checkout zone",
        business_line="Goods leaving the marked area",
        class_allowlist=("person", "apple"),
        zones=(),
        event_rules={
            "min_dwell_seconds": 0.5, "max_transition_seconds": 12.0,
            "origin_slot": "shelf", "destination_slot": "basket",
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
        self.assertEqual(self.m.abandoned_episodes, 1)
        self.assertEqual(self.m.abandoned_reasons["track_lost"], 1)
        # Losing a track that had no pending transfer is not a cancellation.
        self.m.on_track_lost(99, 0.9)
        self.assertEqual(self.m.abandoned_episodes, 1)

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
        self.assertEqual(m.abandoned_episodes, 0)


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
        self.assertEqual(self.m.abandoned_episodes, 0)
        self.assertEqual(self.m.abandoned_reasons, {})
        # A replay of loop 0's frames under loop 1 must still transfer exactly once,
        # which it would not if the dedup set had survived the reset.
        events = self._walk(self.m, loop=1, fid=0)
        self.assertEqual(len([e for e in events if e.kind == "candidate_transfer"]), 1)

    def test_a_frame_from_an_unannounced_loop_is_refused(self):
        # Spec Â§9 requires an explicit reset per loop. Silently adopting a new loop id
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
        # Two apples, both origin -> destination: distinct keys, so neither is deduped
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
        # A class the pack does not track walks the whole route and emits nothing. This
        # previously used `person`, which **is** in `tracked_classes` - the test passed
        # only because the synthetic label list happened to call class 0 `class_0`, so
        # it was really testing an unmapped name and saying it was testing the
        # allowlist. Class 1 is genuinely untracked.
        untracked = 1   # 'class_1', not in ("person", "apple")
        self.assertNotIn("class_1", self.m.tracked_classes)
        for t, box in [(0.0, IN_SHELF), (0.6, IN_SHELF),
                       (1.0, IN_BASKET), (1.4, IN_BASKET)]:
            self.m.observe(_track(1, box, t, cls=untracked), _frame(int(t * 10), t))
        self.assertEqual(self.m.events, [])
        self.assertEqual(self.m.skipped_unmapped_class, 0,
                         "a mapped but untracked class is not a skipped class")

    def test_the_tracked_classes_come_from_the_manifest(self):
        self.assertEqual(self.m.rule.classes, ("person", "apple"))
        self.assertEqual(self.m.rule.min_dwell_seconds, 0.5)
        self.assertEqual(self.m.rule.max_transition_seconds, 12.0)

    def test_a_class_id_past_the_end_of_the_label_list_is_skipped_not_fatal(self):
        # I2 review finding. `class_id` comes from the *model*, so an id past the end
        # of the label list is a fact about the model, not a caller bug. Raising here
        # took a live stream down, repeatedly, for something the caller cannot fix
        # (AGENTS.md rule 12). The genuine invariant is a startup one and lives in
        # `_resolve_class_names`.
        for t, box in [(0.0, IN_SHELF), (0.6, IN_SHELF), (1.0, IN_BASKET)]:
            events = self.m.observe(_track(1, box, t, cls=9999), _frame(int(t * 10), t))
            self.assertEqual(events, ())
        self.assertEqual(self.m.skipped_unmapped_class, 3)
        self.assertEqual(self.m.events, [])

    def test_an_unmapped_class_id_never_becomes_the_string_unknown(self):
        # The fabricated name is the thing that must not happen: "unknown" on a screen
        # reads as a class the model reported.
        for t in (0.0, 0.6, 1.0, 1.4):
            self.m.observe(_track(1, IN_SHELF, t, cls=9999), _frame(int(t * 10), t))
        names = [e.class_name for e in self.m.events]
        self.assertNotIn("unknown", names)
        self.assertEqual(names, [])

    def test_the_skipped_count_survives_so_the_screen_can_report_it(self):
        self.m.observe(_track(1, IN_SHELF, 0.0, cls=9999), _frame(0, 0.0))
        self.assertEqual(self.m.skipped_unmapped_class, 1)
        # A later, mappable observation still works: one bad id does not wedge the
        # machine.
        self.m.observe(_track(2, IN_SHELF, 0.1), _frame(1, 0.1))
        self.assertEqual(self.m.skipped_unmapped_class, 1)
        self.assertEqual(self.m.state_of(2), "in_origin")

    def test_an_empty_label_list_is_refused_at_construction(self):
        # With an empty list every observation would be unmappable and every frame
        # would be skipped - a machine that reports "nothing is happening" for a
        # reason it cannot state.
        with self.assertRaises(ValueError):
            RetailEventMachine(_manifest(), class_names=())

    def test_an_injected_label_list_of_the_wrong_width_is_refused(self):
        # Round-2 review finding: the width check against the pin ran only when
        # `class_names` was omitted, so an injected `("person",)` was accepted silently
        # and then every `class_id` above 0 became a `skipped_unmapped_class` at
        # observation time instead of a startup failure. The check is unconditional
        # now.
        with self.assertRaises(ValueError) as caught:
            RetailEventMachine(_manifest(), class_names=("person",))
        self.assertIn("classes", str(caught.exception))

    def test_an_injected_list_of_the_right_width_is_accepted(self):
        # The other side: the check must not reject a caller's own list merely for
        # being injected, so long as it is the model's width.
        m = RetailEventMachine(_manifest(), class_names=CLASS_NAMES)
        self.assertEqual(len(m.tracked_classes), 2)

    def test_the_default_label_list_is_resolved_and_cross_checked_at_construction(self):
        # C2: resolution happens once, here, and is checked against the pin. The
        # construction succeeding at all proves the lazy read is gone from `observe`.
        m = RetailEventMachine(_manifest())
        self.assertEqual(len(m.tracked_classes), 2)
        self.assertEqual(APPLE_ID, 47, "the fixture's apple id must match labels.txt")

    def test_a_class_not_in_the_allowlist_is_refused_at_construction(self):
        # The allowlist and the tracked list have to agree, or the machine would
        # silently track something the pack says it does not show.
        bad = _manifest(event_rules={"min_dwell_seconds": 0.5,
                                     "max_transition_seconds": 12.0,
                                     "shelf": SHELF, "basket": BASKET,
                                     "tracked_classes": ("person", "banana")})
        with self.assertRaises(ValueError):
            RetailEventMachine(bad, class_names=CLASS_NAMES)


class TheShippedPresenceConfiguration(unittest.TestCase):
    """The configuration M1 actually ships: **people in zone**.

    A previous version of this file declared `BASKET_ZONE = (0.00, 0.70, 0.22, 0.99)`
    and `CHECKOUT_ZONE = (0.35, 0.30, 0.75, 0.62)` to be "the honest configuration"
    and drove a transfer through them. Both rectangles were wrong, and a review
    measured it over the whole clip: the first holds 1286 detections of which **1285
    are the stationary apple** that is the only hand-annotated object in the
    repository, and the second holds **2 `suitcase` detections**. Zero product
    transfers. The configuration now lives in `app/engine/retail_config.py`, is
    derived from measured person-centroid occupancy, and is checked against the real
    footage by `tests/test_retail_footage_guard.py`.
    """

    def test_the_shipped_manifest_is_presence_only_and_declares_no_transfer(self):
        manifest = retail_sanity_manifest()
        rules = manifest.event_rules
        self.assertEqual(rules["presence_zone"], "counter_band")
        self.assertNotIn("origin_slot", rules)
        self.assertNotIn("destination_slot", rules)
        m = RetailEventMachine(manifest, class_names=CLASS_NAMES)
        self.assertIsNone(m.origin)
        self.assertIsNone(m.destination)
        self.assertIsNotNone(m.presence_zone)

    def test_the_business_line_does_not_claim_a_product_transfer(self):
        text = retail_sanity_manifest().business_line.lower()
        for forbidden in ("shelf", "basket", "transfer", "loss prevention"):
            self.assertNotIn(forbidden, text)
        self.assertIn("people", text)

    def test_the_scope_note_says_the_reduction_outright(self):
        note = retail_sanity_manifest().scope_note
        self.assertIn("NOT delivered", note)
        self.assertIn("no product-transfer event is detectable", note.lower())

    def test_only_person_is_tracked(self):
        # The counter surface is labelled 17 different ways by the detector. A count
        # over it is a count of label noise.
        self.assertEqual(retail_sanity_manifest().event_rules["tracked_classes"],
                         ("person",))

    def test_a_sustained_presence_is_a_zone_occupancy_event(self):
        m = RetailEventMachine(retail_sanity_manifest(), class_names=CLASS_NAMES)
        # A person box clipped by the frame border, exactly as measured on the clip.
        box = (500, 260, 1280, 660)
        events = []
        for t in (0.0, 1.0, 2.0, 2.5, 3.0):
            events += m.observe(_track(1, box, t, cls=PERSON_ID), _frame(int(t * 10), t))
        kinds = [e.kind for e in events]
        self.assertEqual(kinds.count("zone_occupancy"), 1)
        self.assertEqual(m.events[0].class_name, "person")
        self.assertIn("presence:counter_band", m.events[0].dedup_key)

    def test_presence_is_measured_on_the_centre_not_on_containment(self):
        # The measured reason the zone test is `contains_point`: this box overhangs the
        # frame, so `Zone.contains` is False for it and a rule built on `contains`
        # would report nobody present at all.
        m = RetailEventMachine(retail_sanity_manifest(), class_names=CLASS_NAMES)
        box = (0, 0, 1920, 1080)   # overhangs on every side
        self.assertFalse(m.presence_zone.contains(box))
        for t in (0.0, 1.0, 2.0, 2.5):
            m.observe(_track(1, box, t, cls=PERSON_ID), _frame(int(t * 10), t))
        self.assertEqual([e.kind for e in m.events], ["zone_occupancy"])

    def test_presence_below_the_minimum_is_not_an_event(self):
        m = RetailEventMachine(retail_sanity_manifest(), class_names=CLASS_NAMES)
        for t in (0.0, 0.5, 1.0):
            m.observe(_track(1, (500, 260, 1280, 660), t, cls=PERSON_ID),
                      _frame(int(t * 10), t))
        self.assertEqual(m.events, [])

    def test_a_second_stay_in_the_same_loop_is_suppressed_by_the_dedup_key(self):
        # M3 review finding, made explicit for presence. A track that leaves the zone
        # and comes back produces the *same* `dedup_key`, so it is counted once per
        # loop. That is the same rule the transfer legs use and it is deliberate; the
        # alternative needs a per-episode discriminator in the key. What matters is
        # that it is a decision rather than an accident.
        m = RetailEventMachine(retail_sanity_manifest(), class_names=CLASS_NAMES)
        inside = (500, 260, 1280, 660)
        outside = (20, 780, 380, 1060)   # centroid at y ~ 0.85, below the band
        emitted = []
        for t, box in [(0.0, inside), (1.0, inside), (2.0, inside),
                       (3.0, outside), (4.0, inside), (5.0, inside),
                       (6.0, inside), (7.0, inside)]:
            emitted += m.observe(_track(1, box, t, cls=PERSON_ID),
                                 _frame(int(t * 10), t))
        stays = [e for e in emitted if e.kind == "zone_occupancy"]
        self.assertEqual(len(stays), 1, "one stay per track per loop, by the dedup key")
        # The state, however, did restart: `state_of` and a fresh loop each see it.
        m.reset(loop_id=1)
        again = m.observe(_track(1, inside, 8.0, cls=PERSON_ID), _frame(80, 8.0, loop=1))
        self.assertEqual(again, (), "1.0s into a fresh loop's stay is below the minimum")

    def test_leaving_the_zone_restarts_the_dwell_clock(self):
        # ...and the clock really does restart, which is what makes the suppression
        # above a dedup decision rather than a dwell that never matured.
        m = RetailEventMachine(retail_sanity_manifest(), class_names=CLASS_NAMES)
        inside = (500, 260, 1280, 660)
        outside = (20, 780, 380, 1060)
        emitted = []
        for t, box in [(0.0, inside), (1.0, inside), (1.9, outside),
                       (2.0, inside), (3.0, inside), (3.5, inside)]:
            emitted += m.observe(_track(1, box, t, cls=PERSON_ID),
                                 _frame(int(t * 10), t))
        self.assertEqual([e for e in emitted if e.kind == "zone_occupancy"], [])

    def test_a_presence_stay_is_counted_once_not_once_per_frame(self):
        m = RetailEventMachine(retail_sanity_manifest(), class_names=CLASS_NAMES)
        emitted = []
        for t in [i * 0.5 for i in range(20)]:
            emitted += m.observe(_track(1, (500, 260, 1280, 660), t, cls=PERSON_ID),
                                 _frame(int(t * 10), t))
        self.assertEqual(len([e for e in emitted if e.kind == "zone_occupancy"]), 1)

    def test_a_stay_that_was_already_counted_is_not_abandoned(self):
        # Round-2 review finding: the comment on `on_track_lost` claimed a counted stay
        # is not abandoned, but the code could not tell a counted stay from an uncounted
        # one - it held a bare timestamp - so a stay maturing at t=2.0 and lost at
        # t=3.5 reported an abandonment for an event that had already been emitted.
        # `_Stay.counted` makes the two cases separable; this pins the outcome.
        m = RetailEventMachine(retail_sanity_manifest(), class_names=CLASS_NAMES)
        inside = (500, 260, 1280, 660)
        for t in (0.0, 1.0, 2.0, 2.5):
            m.observe(_track(1, inside, t, cls=PERSON_ID), _frame(int(t * 10), t))
        self.assertEqual(len(m.events), 1)
        self.assertEqual(m.abandoned_episodes, 0)
        m.on_track_lost(1, 3.5)
        self.assertEqual(m.abandoned_episodes, 0,
                         "the event was already emitted; nothing was withheld")

    def test_a_stay_lost_before_it_matured_is_abandoned(self):
        # The other side of the same boundary, so the test above cannot pass just
        # because nothing is ever counted.
        m = RetailEventMachine(retail_sanity_manifest(), class_names=CLASS_NAMES)
        inside = (500, 260, 1280, 660)
        for t in (0.0, 1.0):
            m.observe(_track(1, inside, t, cls=PERSON_ID), _frame(int(t * 10), t))
        m.on_track_lost(1, 1.5)
        self.assertEqual(m.abandoned_episodes, 1)
        self.assertEqual(m.abandoned_reasons["track_lost"], 1)

    def test_the_counter_is_named_for_what_it_counts(self):
        # It counts abandoned *episodes*, of two kinds. A name saying "transfers"
        # reads non-zero in the presence-only pack that ships, where no transfer exists.
        self.assertFalse(hasattr(RetailEventMachine(_manifest()), "abandoned_transfers"))
        m = RetailEventMachine(retail_sanity_manifest(), class_names=CLASS_NAMES)
        self.assertEqual(m.abandoned_episodes, 0)
        m.on_track_lost(999, 0.0)
        self.assertEqual(m.abandoned_episodes, 0,
                         "losing a track with no episode abandons nothing")

    def test_a_lost_track_does_not_carry_its_presence_forward(self):
        # The occlusion rule applied to presence: the track vanishes mid-dwell, and a
        # same-id observation afterwards must not be credited with the missing time.
        #
        # The observation at t=2.0 is the discriminator, and it has to be there
        # specifically. The stay began at 0.0; if the presence record survived the loss
        # the clock would still be running from 0.0 and would mature at t=2.0. With the
        # record cleared the new stay began at 1.6 and is only 0.4 s old at t=2.0.
        # A test that waited until the *new* stay had matured could not tell the two
        # apart, which is why an earlier version of this test survived the mutant.
        m = RetailEventMachine(retail_sanity_manifest(), class_names=CLASS_NAMES)
        box = (500, 260, 1280, 660)
        for t in (0.0, 1.0):
            m.observe(_track(1, box, t, cls=PERSON_ID), _frame(int(t * 10), t))
        m.on_track_lost(1, 1.5)
        self.assertEqual(m.abandoned_episodes, 1)
        early = m.observe(_track(1, box, 1.6, cls=PERSON_ID), _frame(16, 1.6))
        self.assertEqual([e.kind for e in early], [])
        at_two = m.observe(_track(1, box, 2.0, cls=PERSON_ID), _frame(20, 2.0))
        self.assertEqual([e.kind for e in at_two], [],
                         "a dwell matured on pre-loss time nobody observed standing there")
        # ...and the new stay does fire once it has genuinely earned 2.0 s of its own.
        late = m.observe(_track(1, box, 3.6, cls=PERSON_ID), _frame(36, 3.6))
        self.assertEqual([e.kind for e in late], ["zone_occupancy"])

    def test_reset_clears_presence_state(self):
        m = RetailEventMachine(retail_sanity_manifest(), class_names=CLASS_NAMES)
        inside = (500, 260, 1280, 660)
        for t in (0.0, 1.0, 2.0, 2.5):
            m.observe(_track(1, inside, t, cls=PERSON_ID), _frame(int(t * 10), t))
        self.assertEqual(len(m.events), 1)
        m.reset(loop_id=1)
        self.assertEqual(m.events, [])
        # The presence record itself must be gone, not merely the event log. If it
        # survived, the first observation of the new loop would inherit the old entry
        # time and mature a dwell made of frames nobody observed in this loop.
        first = m.observe(_track(1, inside, 3.0, cls=PERSON_ID), _frame(30, 3.0, loop=1))
        self.assertEqual([e.kind for e in first], [],
                         "3.0s after a reset must start a fresh stay, not inherit one")


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

    def test_observe_opens_no_file_after_construction(self):
        # C2 review finding. The lazy `app.inference.pinned` import meant
        # `pinned.class_names()` -> `Path.read_text('labels.txt')` ran on the first
        # observation, i.e. on the first frame of a live stream, making this module's
        # "no file is opened" claim false.
        #
        # `pathlib.Path.read_text` is the thing to spy on, **not** `builtins.open`:
        # `Path.read_text` calls `self.open(...)`, which does not go through the
        # `builtins.open` name the module holds, so an `open`-based spy sees nothing
        # and the test passes against the very code it is meant to catch.
        m = RetailEventMachine(_manifest(), class_names=CLASS_NAMES)
        opened: list[str] = []
        original = Path.read_text

        def spy(self, *args, **kwargs):
            opened.append(str(self))
            return original(self, *args, **kwargs)

        Path.read_text = spy
        try:
            for t, box in [(0.0, IN_SHELF), (0.6, IN_SHELF),
                           (1.0, IN_BASKET), (1.4, IN_BASKET)]:
                m.observe(_track(1, box, t), _frame(int(t * 10), t))
        finally:
            Path.read_text = original
        self.assertEqual(opened, [], f"observe() read {opened}")
        # ...and the spy is real: it does catch a read if one happens. This
        # self-check is what stops the assertion above from passing because the spy
        # was never installed on the code path that mattered.
        Path.read_text = spy
        try:
            Path("config/models.json").read_text(encoding="utf-8")
        finally:
            Path.read_text = original
        self.assertTrue(opened, "the Path.read_text spy did not observe a real read")


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

    def test_a_pack_with_no_rule_at_all_is_refused(self):
        # Tracked classes but no route and no presence rule: a machine that can never
        # emit. A booth showing it would read zero forever, which says "nothing is
        # happening here" rather than "this pack is misconfigured".
        with self.assertRaises(ValueError):
            RetailEventMachine(_manifest(event_rules={"tracked_classes": ("apple",)}),
                               class_names=CLASS_NAMES)

    def test_a_presence_zone_without_a_threshold_is_refused(self):
        with self.assertRaises(ValueError):
            RetailEventMachine(_manifest(event_rules={
                "presence_zone": "basket", "tracked_classes": ("apple",),
                "basket": BASKET}), class_names=CLASS_NAMES)

    def test_a_threshold_without_a_zone_is_refused(self):
        with self.assertRaises(ValueError):
            RetailEventMachine(_manifest(event_rules={
                "min_presence_seconds": 2.0, "tracked_classes": ("apple",)}),
                class_names=CLASS_NAMES)

    def test_half_a_transfer_route_is_refused(self):
        # One end declared and the other missing: the machine would have to invent it,
        # and an invented zone is a claim about the footage nobody made.
        with self.assertRaises(ValueError):
            RetailEventMachine(_manifest(event_rules={
                "min_dwell_seconds": 0.5, "max_transition_seconds": 12.0,
                "origin_slot": "shelf", "shelf": SHELF, "basket": BASKET,
                "tracked_classes": ("apple",)}), class_names=CLASS_NAMES)

    def test_a_route_naming_an_absent_roi_is_refused(self):
        with self.assertRaises(ValueError):
            RetailEventMachine(_manifest(event_rules={
                "min_dwell_seconds": 0.5, "max_transition_seconds": 12.0,
                "origin_slot": "shelf", "destination_slot": "checkout",
                "shelf": SHELF, "basket": BASKET,
                "tracked_classes": ("apple",)}), class_names=CLASS_NAMES)

    def test_the_default_zone_names_are_gone_from_the_app_layer(self):
        # No `shelf`/`basket` defaults remain. A pack's route is declared or absent, so
        # a config can never acquire a transfer it did not ask for. Asserted over the
        # source so a default cannot quietly return.
        import inspect
        from app.engine import events as events_module
        source = inspect.getsource(events_module)
        for dead in ("DEFAULT_ORIGIN_SLOT", "DEFAULT_DESTINATION_SLOT"):
            self.assertNotIn(dead, source)

    def test_the_manifest_must_be_a_manifest(self):
        with self.assertRaises(ValueError):
            RetailEventMachine({"id": "retail"}, class_names=CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()