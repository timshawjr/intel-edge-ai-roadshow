"""`app/contracts/scenario_manifest.py` - what a pack claims, and what it may claim.

The reason this contract exists at all: **M0 recorded that the retail footage does
not contain the scenario the original brief asked for.** Read
`docs/qualification/M0-content-gate.md` §3-§4 - `retail_checkout` is a
self-checkout station, there is no shelf anywhere in the clip, and measured over all
3387 frames **no product-transfer event is detectable in it at all**. The delivered
scope is therefore **people-in-zone presence**, and shelf-to-basket stays unmet on the
open-blockers list.

An earlier revision of this note claimed the visible movement was a two-zone product
route and recorded the reduced-scope deliverable as items entering the checkout zone.
Both claims were **withdrawn** when the rectangles were measured: they held a
stationary apple and two `suitcase` detections. The numbers, and the
`counter_band` presence zone that replaced them, are in
`app/engine/retail_config.py`.

The strings in the fixtures below are deliberately **neutral placeholders** rather
than the withdrawn wording, so that no file in the suite can be quoted as the
project's claim.

A manifest is where that fact has to live, so it cannot quietly stop being true. If
`business_line` is the sentence the booth shows, then it has to be the reduced-scope
sentence, and this module refuses a pack that leaves it empty - because an empty
title on a screen is indistinguishable from a title nobody wrote.

The tests here are therefore as much about the honesty of the claim as about the
shape of the record:

- `test_a_pack_must_state_its_business_line` - a reduced-scope label has to be
  present, not optional.
- `test_a_bad_roi_in_the_config_raises_at_load_naming_the_slot` - the review
  focus for this task. A malformed ROI must fail loudly at load, with the slot's
  name, rather than drawing boxes in the wrong place for a whole session.
- `test_event_rules_cannot_be_edited_after_construction` - `frozen=True` does not
  reach inside a dict, and a manifest whose dwell requirement can be changed after
  the event machine read it is not a contract.

Nothing here reads `config/models.json`, a clock, or the filesystem. The class
allowlist is checked for internal consistency only; asking the pinned model which
class ids exist is `app.inference.pinned`'s job.
"""

import unittest

from app.contracts.scenario_manifest import ScenarioManifest
from app.engine.zones import Zone

SHELF = (0.02, 0.22, 0.34, 0.55)
BASKET = (0.60, 0.30, 0.98, 0.70)


def _manifest(**kw):
    base = dict(
        id="retail", version=1, title="Retail / checkout zone",
        business_line="Goods leaving the marked area",
        class_allowlist=("person", "apple"),
        zones=(),
        event_rules={"min_dwell_seconds": 0.5, "max_transition_seconds": 12.0,
                     "shelf": SHELF, "basket": BASKET,
                     "tracked_classes": ("person", "apple")},
        source_ids=("retail_checkout",), conf_threshold=0.25, iou_threshold=0.45,
    )
    base.update(kw)
    return ScenarioManifest(**base)


class TheRecord(unittest.TestCase):
    def test_a_manifest_keeps_what_it_was_given(self):
        m = _manifest()
        self.assertEqual(m.id, "retail")
        self.assertEqual(m.conf_threshold, 0.25)
        self.assertEqual(m.source_ids, ("retail_checkout",))
        self.assertEqual(m.class_allowlist, ("person", "apple"))

    def test_sequences_are_normalised_to_tuples(self):
        # A JSON loader hands over lists. The record is frozen, so a list inside it
        # would still be mutable - the same hole as a dict.
        m = _manifest(class_allowlist=["person", "apple"], source_ids=["retail_checkout"])
        self.assertIsInstance(m.class_allowlist, tuple)
        self.assertIsInstance(m.source_ids, tuple)

    def test_to_json_round_trips_through_json_dumps(self):
        import json
        payload = _manifest(zones=(Zone("checkout", "area", (0.5, 0.2, 0.9, 0.8)),)).to_json()
        text = json.dumps(payload)
        self.assertIn('"business_line": "Goods leaving the marked area"', text)
        self.assertEqual(payload["zones"][0]["name"], "checkout")
        self.assertEqual(payload["zones"][0]["roi"], [0.5, 0.2, 0.9, 0.8])
        self.assertIsInstance(payload["event_rules"]["shelf"], list)

    def test_it_is_immutable(self):
        with self.assertRaises(Exception):
            _manifest().id = "other"


class TheClaim(unittest.TestCase):
    """The record must be able to say what a pack does not contain."""

    def test_a_pack_must_state_its_business_line(self):
        with self.assertRaises(ValueError):
            _manifest(business_line="")
        with self.assertRaises(ValueError):
            _manifest(business_line=None)

    def test_a_reduced_scope_note_rides_with_the_data(self):
        note = ("Shelf-to-basket is NOT delivered: no product-transfer event is "
                "detectable in the provisioned retail clip, which contains no shelf.")
        m = _manifest(scope_note=note)
        self.assertIn("NOT delivered", m.to_json()["scope_note"])

    def test_a_scope_note_is_optional_for_a_full_scope_pack(self):
        self.assertEqual(_manifest().scope_note, "")


class Validation(unittest.TestCase):
    def _refuse(self, **kw):
        with self.assertRaises(ValueError):
            _manifest(**kw)

    def test_identifiers_must_be_present(self):
        self._refuse(id="")
        self._refuse(title="  ")

    def test_version_must_be_a_positive_int(self):
        self._refuse(version=0)
        self._refuse(version=-1)
        self._refuse(version=1.0)

    def test_thresholds_must_be_within_zero_and_one(self):
        self._refuse(conf_threshold=1.5)
        self._refuse(conf_threshold=-0.1)
        self._refuse(iou_threshold=2.0)
        # 0 is a legal threshold; it is not a stand-in for "unset".
        self.assertEqual(_manifest(conf_threshold=0.0).conf_threshold, 0.0)

    def test_an_empty_allowlist_is_refused(self):
        # An empty allowlist is a detector with nothing to say, and it would render
        # as "no objects" forever.
        self._refuse(class_allowlist=())

    def test_a_pack_with_no_source_has_nothing_to_play(self):
        self._refuse(source_ids=())

    def test_zones_must_be_zone_objects(self):
        # A raw ROI tuple slipped into `zones` would skip Zone's validation, which
        # is the one thing `zones` is for.
        self._refuse(zones=(SHELF,))

    def test_duplicate_zone_names_are_refused(self):
        # Zone names key per-track event state. Two zones sharing a name share and
        # corrupt it, and the corruption is invisible until an event count is wrong.
        self._refuse(zones=(Zone("shelf", "area", SHELF),
                            Zone("shelf", "area", BASKET)))

    def test_event_rules_must_be_a_mapping(self):
        self._refuse(event_rules=[("min_dwell_seconds", 0.5)])

    def test_a_bad_roi_in_the_config_raises_at_load_naming_the_slot(self):
        # The review focus for this task, in the form a config file actually breaks:
        # a transposed ROI, silently accepted by a validator that only checks range.
        m = _manifest(event_rules={"min_dwell_seconds": 0.5,
                                   "basket": (0.9, 0.30, 0.60, 0.70)})
        with self.assertRaises(ValueError) as caught:
            m.zones_for_event_rules()
        self.assertIn("basket", str(caught.exception))
        self.assertIn("out of order", str(caught.exception))

    def test_an_out_of_range_roi_raises_rather_than_being_clamped(self):
        m = _manifest(event_rules={"min_dwell_seconds": 0.5, "shelf": (0.02, 0.22, 1.4, 0.55)})
        with self.assertRaises(ValueError) as caught:
            m.zones_for_event_rules()
        self.assertIn("shelf", str(caught.exception))
        self.assertIn("[0, 1]", str(caught.exception))


class Immutability(unittest.TestCase):
    def test_event_rules_cannot_be_edited_after_construction(self):
        # `frozen=True` does not freeze a dict. Without this, the dwell requirement
        # could be rewritten after the event machine read it and the two would
        # disagree about what counted - with nothing on screen to say so.
        m = _manifest()
        with self.assertRaises(TypeError):
            m.event_rules["min_dwell_seconds"] = 0.0

    def test_the_built_zones_are_real_zones(self):
        zones = _manifest().zones_for_event_rules()
        self.assertEqual(set(zones), {"shelf", "basket"})
        self.assertIsInstance(zones["shelf"], Zone)
        self.assertEqual(zones["basket"].roi, (0.60, 0.30, 0.98, 0.70))

    def test_an_event_rule_slot_overrides_a_declared_zone_of_the_same_name(self):
        # The event rule is the more specific statement of where that slot is; the
        # manifest's `zones` list is the presentation layer's copy.
        m = _manifest(zones=(Zone("basket", "area", (0.0, 0.0, 0.1, 0.1)),))
        self.assertEqual(m.zones_for_event_rules()["basket"].roi, BASKET)

    def test_scalar_rules_are_not_mistaken_for_rois(self):
        # min_dwell_seconds is a 4-tuple-shaped thing in neither sense, but a config
        # with `("a","b","c","d")` in a rule slot must not become a Zone either.
        m = _manifest(event_rules={"tracked_classes": ("person", "apple")})
        self.assertEqual(m.zones_for_event_rules(), {})


if __name__ == "__main__":
    unittest.main()