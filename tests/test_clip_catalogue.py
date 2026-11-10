"""The clip catalogue is a measurement of the files, not a wish list.

`config/clips.json` is what `--clip` reads, so every field it carries is re-derived here
from the bytes on disk: SHA-256, byte count, resolution, fps, frame count, duration and
the 45-90 s band. A re-encoded clip with the same name fails this suite rather than
silently playing different footage under an old label.

Three clips have no licence sidecar and must stay `unverified`; six have a CC BY 4.0
sidecar and must carry it **without** being upgraded to `cleared`. That distinction is
the rights gate, and a catalogue that quietly cleared them would be the exact failure the
content gate exists to prevent.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import cv2

from app.engine.clips import (
    CLIPS_JSON,
    SPEC_MAX_SECONDS,
    SPEC_MIN_SECONDS,
    default_selection,
    load_catalogue,
    pack_ids,
    resolve_selection,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The nine provisioned clips, by catalogue id, in catalogue order. The four pack
#: assignments are the operator's content assignment, not a claim of scene fit.
EXPECTED = (
    ("retail-checkout", "retail_sanity", False, "unverified"),
    ("store-aisle-detection", "retail_sanity", True, "unverified"),
    ("bottle-detection", "retail_sanity", False, "CC BY 4.0"),
    ("one-by-one-person-detection", "retail_sanity", False, "unverified"),
    ("person-bicycle-car-detection", "smart_city", True, "CC BY 4.0"),
    ("car-detection", "smart_city", False, "CC BY 4.0"),
    ("people-detection", "medical_ops", True, "CC BY 4.0"),
    ("classroom", "medical_ops", False, "CC BY 4.0"),
    ("worker-zone-detection", "depot_logistics", True, "CC BY 4.0"),
)

SIDECAR_CLIPS = {
    "bottle-detection",
    "person-bicycle-car-detection",
    "car-detection",
    "people-detection",
    "classroom",
    "worker-zone-detection",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TheCatalogueMatchesTheFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_catalogue()

    def test_exactly_the_nine_provisioned_clips_are_catalogued(self):
        self.assertEqual(tuple(record.id for record in self.records),
                         tuple(expected[0] for expected in EXPECTED))
        self.assertEqual(len(self.records), 9)

    def test_every_record_names_a_file_that_exists(self):
        for record in self.records:
            self.assertTrue(record.exists(), f"{record.id}: {record.path} is not on disk")
            self.assertEqual(record.absolute_path.resolve(),
                             (REPO_ROOT / record.path).resolve())

    def test_every_measured_field_reproduces_from_the_file(self):
        for record in self.records:
            path = record.absolute_path
            self.assertEqual(record.sha256, _sha256(path), f"{record.id}: sha256")
            self.assertEqual(record.bytes, path.stat().st_size, f"{record.id}: bytes")
            capture = cv2.VideoCapture(str(path))
            try:
                self.assertTrue(capture.isOpened(), f"{record.id}: decoder cannot open")
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                capture.release()
            self.assertEqual((record.measured_width, record.measured_height), (width, height),
                             record.id)
            self.assertEqual(record.measured_fps, fps, record.id)
            self.assertEqual(record.measured_frames, frames, record.id)
            self.assertEqual(record.measured_duration_s, frames / fps, record.id)

    def test_the_duration_band_is_computed_from_the_measurement(self):
        for record in self.records:
            inside = SPEC_MIN_SECONDS <= record.measured_duration_s <= SPEC_MAX_SECONDS
            self.assertEqual(record.in_duration_band, inside,
                             f"{record.id}: {record.measured_duration_s} s")

    def test_the_operator_assigned_each_clip_to_the_named_pack(self):
        self.assertEqual(tuple((record.id, record.pack, record.in_duration_band,
                                record.license) for record in self.records),
                         EXPECTED)
        self.assertEqual(pack_ids(),
                         ("retail_sanity", "smart_city", "medical_ops", "depot_logistics"))

    def test_a_sidecar_is_recorded_as_cc_by_and_never_upgraded_to_cleared(self):
        for record in self.records:
            self.assertFalse(record.distribution_status.lower().startswith("cleared"),
                             record.id)
            if record.id in SIDECAR_CLIPS:
                self.assertEqual(record.license, "CC BY 4.0", record.id)
                self.assertIn("github.com/intel-iot-devkit/sample-videos",
                              record.license_source, record.id)
                self.assertIn("clearance not established",
                              record.distribution_status, record.id)
            else:
                self.assertEqual(record.license, "unverified", record.id)
                self.assertTrue(record.distribution_status.startswith("not cleared"),
                                record.id)

    def test_the_catalogue_file_is_committed_and_read_only_in_this_test(self):
        before = CLIPS_JSON.stat().st_mtime
        load_catalogue()
        self.assertEqual(CLIPS_JSON.stat().st_mtime, before)


class DefaultAndAliases(unittest.TestCase):
    def test_the_default_is_unchanged_at_retail_checkout_under_retail_sanity(self):
        selection = default_selection()
        self.assertEqual(selection.clip.id, "retail-checkout")
        self.assertEqual(selection.pack_id, "retail_sanity")

    def test_every_catalogue_id_resolves_to_its_own_file(self):
        ids = [expected[0] for expected in EXPECTED]
        for clip_id in ids:
            selection = resolve_selection(clip=clip_id, require_files=True)
            self.assertEqual(selection.clip.id, clip_id)
            self.assertTrue(selection.clip.exists())

    def test_the_underscore_media_id_is_accepted_as_an_alias(self):
        selection = resolve_selection(clip="retail_checkout", require_files=True)
        self.assertEqual(selection.clip.id, "retail-checkout")

    def test_a_pack_selects_its_first_clip(self):
        for pack, expected in (("retail_sanity", "retail-checkout"),
                               ("smart_city", "person-bicycle-car-detection"),
                               ("medical_ops", "people-detection"),
                               ("depot_logistics", "worker-zone-detection")):
            selection = resolve_selection(pack=pack, require_files=True)
            self.assertEqual(selection.clip.id, expected)
            self.assertEqual(selection.pack_id, pack)

    def test_a_clip_and_its_pack_together_are_accepted(self):
        selection = resolve_selection(clip="car-detection", pack="smart_city")
        self.assertEqual(selection.clip.id, "car-detection")
        self.assertEqual(selection.pack_id, "smart_city")

    def test_a_clip_outside_the_named_pack_is_refused(self):
        from app.engine.clips import SelectionError

        with self.assertRaises(SelectionError) as caught:
            resolve_selection(clip="retail-checkout", pack="smart_city")
        self.assertIn("belongs to pack", str(caught.exception))

    def test_an_unknown_clip_names_the_ids_that_exist(self):
        from app.engine.clips import SelectionError

        with self.assertRaises(SelectionError) as caught:
            resolve_selection(clip="not-a-clip")
        self.assertIn("retail-checkout", str(caught.exception))

    def test_an_unknown_pack_is_refused(self):
        from app.engine.clips import SelectionError

        with self.assertRaises(SelectionError):
            resolve_selection(pack="not_a_pack")


if __name__ == "__main__":
    unittest.main()
