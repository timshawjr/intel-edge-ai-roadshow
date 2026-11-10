"""The four pack manifests, and the rules the engine cannot express.

The deliverable is four manifests with real clips, measured zones, an event rule the
machine can actually compute, a business metric and an explanation. Two failures are
possible and this module pins both:

- **a pack that names a rule the engine has no event kind for** (line crossing, a dwell
  event, an entry count) and then presents it as delivered. `app.engine.packs` refuses
  that at load; this suite proves the refusal fires, including the mutation.
- **a pack whose geometry was guessed.** Every zone here was measured from detected
  centroids; the manifest carries the range and this suite checks the ROI brackets it.
"""

from __future__ import annotations

import unittest

from app.contracts.scenario_manifest import ScenarioManifest
from app.engine.clips import load_catalogue
from app.engine.packs import (
    PackError,
    _check_delivery_words,
    _manifest_from_payload,
    available_packs,
    load_all_packs,
    load_pack,
    pack_ids,
)
from app.engine.retail_config import (
    COUNTER_BAND,
    MIN_PRESENCE_SECONDS,
    retail_sanity_manifest,
)

EXPECTED_COUNTS = {
    "retail_sanity": 4,
    "smart_city": 2,
    "medical_ops": 2,
    "depot_logistics": 1,
}

EXPECTED_UNEXPRESSIBLE = {
    "retail_sanity": {"product_transfer"},
    "smart_city": {"directional_line_crossing", "stopped_object_dwell"},
    "medical_ops": {"waiting_zone_dwell"},
    "depot_logistics": {"loading_zone_entry", "loading_zone_dwell"},
}


class TheFourPacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packs = {pack.id: pack for pack in load_all_packs()}
        cls.catalogue = {record.id: record for record in load_catalogue()}

    def test_all_four_manifests_are_on_disk(self):
        self.assertEqual(set(self.packs), set(EXPECTED_COUNTS))
        self.assertEqual(set(pack_ids()), set(EXPECTED_COUNTS))
        self.assertEqual(set(available_packs()),
                         {"depot_logistics", "medical_ops", "retail_sanity", "smart_city"})

    def test_the_clip_counts_are_four_two_two_and_one(self):
        for pack_id, expected in EXPECTED_COUNTS.items():
            self.assertEqual(len(self.packs[pack_id].clips), expected, pack_id)

    def test_every_clip_is_on_disk_and_assigned_to_that_pack(self):
        for pack_id, pack in self.packs.items():
            for record in pack.clips:
                self.assertTrue(record.exists(), f"{pack_id}: {record.path} missing")
                self.assertEqual(record.pack, pack_id)
                self.assertEqual(self.catalogue[record.id], record)

    def test_the_event_rule_is_one_the_machine_can_compute(self):
        for pack_id, pack in self.packs.items():
            rules = pack.manifest.event_rules
            self.assertIn("presence_zone", rules, pack_id)
            self.assertIn("min_presence_seconds", rules, pack_id)
            self.assertNotIn("origin_slot", rules, pack_id)
            self.assertNotIn("destination_slot", rules, pack_id)
            self.assertEqual(rules["tracked_classes"], tuple(pack.manifest.class_allowlist))

    def test_zones_validate_through_the_only_supported_path(self):
        for pack_id, pack in self.packs.items():
            built = pack.manifest.zones_for_event_rules()
            self.assertTrue(built, pack_id)
            self.assertIn(str(pack.manifest.event_rules["presence_zone"]), built, pack_id)
            for zone in built.values():
                self.assertGreaterEqual(zone.roi[0], 0.0)
                self.assertLessEqual(zone.roi[2], 1.0)

    def test_each_pack_names_the_rules_it_cannot_express(self):
        for pack_id, expected in EXPECTED_UNEXPRESSIBLE.items():
            self.assertEqual(set(self.packs[pack_id].unexpressible_rules), expected,
                             pack_id)

    def test_the_business_metric_and_explanation_are_present(self):
        for pack_id, pack in self.packs.items():
            self.assertTrue(pack.business_metric.strip(), pack_id)
            self.assertTrue(pack.explanation.strip(), pack_id)
            self.assertTrue(pack.delivered_event.strip(), pack_id)
            self.assertTrue(pack.manifest.business_line.strip(), pack_id)
            self.assertTrue(pack.manifest.scope_note.strip(), pack_id)

    def test_a_limitation_is_stated_in_the_same_words_on_the_screen(self):
        self.assertIn("NOT a line crossing",
                      self.packs["smart_city"].delivered_event)
        self.assertIn("dwell", self.packs["medical_ops"].delivered_event.lower())
        self.assertIn("NOT an entry count",
                      self.packs["depot_logistics"].delivered_event)
        self.assertIn("NOT a dwell measurement",
                      self.packs["retail_sanity"].delivered_event)
        self.assertIn("waiting-zone dwell is not delivered",
                      self.packs["medical_ops"].manifest.scope_note.lower())
        self.assertIn("loading-zone entry and dwell are not delivered",
                      self.packs["depot_logistics"].manifest.scope_note.lower())

    def test_the_retail_manifest_reproduces_the_measured_constants(self):
        manifest = retail_sanity_manifest()
        self.assertEqual(manifest.event_rules["counter_band"], COUNTER_BAND)
        self.assertEqual(manifest.event_rules["min_presence_seconds"],
                         MIN_PRESENCE_SECONDS)
        self.assertEqual(manifest.business_line,
                         "People present at the checkout counter")
        self.assertEqual(manifest.event_rules["tracked_classes"], ("person",))
        self.assertNotIn("origin_slot", manifest.event_rules)


class TheLoaderRefusesUnsupportedClaims(unittest.TestCase):
    def test_a_delivered_line_crossing_claim_is_refused(self):
        with self.assertRaises(PackError) as caught:
            _check_delivery_words(
                "smart_city",
                "Delivered event: line crossing at the stop line, counted.",
            )
        self.assertIn("line crossing", str(caught.exception))

    def test_a_claim_that_says_not_delivered_is_allowed(self):
        _check_delivery_words(
            "smart_city",
            "Directional line crossing is NOT delivered; presence is measured.",
        )

    def test_a_bad_roi_fails_at_load_naming_the_slot(self):
        catalogue = load_catalogue()
        payload = {
            "version": 1,
            "title": "bad",
            "business_line": "x",
            "business_metric": "x",
            "explanation": "x",
            "delivered_event": "x",
            "content_assignment": "x",
            "unexpressible_rules": ["x"],
            "scope_note": "x",
            "class_allowlist": ["person"],
            "zone_ids": [],
            "zones": [{"name": "road", "kind": "area", "roi": [0.9, 0.1, 0.2, 0.9]}],
            "event_rules": {"presence_zone": "road", "min_presence_seconds": 1.0,
                            "tracked_classes": ["person"],
                            "road": [0.9, 0.1, 0.2, 0.9]},
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
        }
        with self.assertRaises(ValueError) as caught:
            _manifest_from_payload("bad", payload, (catalogue[0],))
        self.assertIn("out of order", str(caught.exception))

    def test_load_pack_refuses_a_manifest_that_claims_line_crossing(self):
        # **The call site, not only the helper.** A `_check_delivery_words` unit test can
        # pass while `load_pack` never calls it, which is the shape of a guard that guards
        # nothing. This writes a real manifest whose delivered-event sentence claims a
        # line crossing and points the loader at it.
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock

        source = Path(__file__).resolve().parent.parent / "assets" / "scenarios" / \
            "smart_city" / "manifest.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["delivered_event"] = (
            "Delivered event: line crossing at the stop line, counted."
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "smart_city"
            target.mkdir()
            (target / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch("app.engine.packs.SCENARIOS_DIR", Path(tmp)):
                with self.assertRaises(PackError) as caught:
                    load_pack("smart_city")
        self.assertIn("line crossing", str(caught.exception))

    def test_a_pack_with_no_clip_is_refused(self):
        # The catalogue's assignment is authoritative: a pack that lists somebody else's
        # clip would run one pack's rule over another pack's footage.
        with self.assertRaises(PackError):
            load_pack("no_such_pack")


if __name__ == "__main__":
    unittest.main()
