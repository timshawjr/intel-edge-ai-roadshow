"""**M5 Task 5: the content gate, measured from the files.**

The requirement is spec section 4: **three 45-90 s clips per pack, four packs, twelve
clips**. The repository does not have that, and this suite's job is to make the shortfall
**unmissable** rather than to make the gate pass:

1. The inventory is measured from the video files with `cv2.VideoCapture`; the committed
   `DECLARED` table below is what those measurements are, and the test fails if a clip is
   added, removed or re-encoded without the table being updated.
2. Every gate verdict carries the requirement, the provisioned number and the **gap**.
   `met` is false for the clip count, the duration band, the pack count and rights.
3. Rights are **not** inferred from a filename: no clip carries a licence, and the
   inventory says `not cleared` rather than "Pexels (probably fine)".
4. Nothing acquires, generates or substitutes footage. The only generated file in this
   suite exists in a temporary directory to prove that an added clip changes the
   inventory, and is deleted with that directory.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools import content_gate


#: The committed inventory, measured with `cv2.VideoCapture` on 2026-10-30 and recorded at
#: full precision. **This is the assertion, not a cache**: re-running the measurement must
#: reproduce every field, and a clip added to or removed from `assets/media/` changes the
#: set of paths and fails here.
DECLARED = {
    "assets/media/one-by-one-person-detection.mp4": {
        "sha256": "a5964aa259099a482a8b360ffc2c57b5a30f84d5919236a4dad01f8e929ac07c",
        "bytes": 3291918,
        "width": 768, "height": 432, "fps": 10.0, "frames": 1394,
        "duration_s": 139.4, "inside_band": False,
    },
    "assets/media/retail-checkout.mp4": {
        "sha256": "e107db69155af4774f907abe8ddbf24493dfcf9d81d51f3e71ff25300e4f5906",
        "bytes": 64996307,
        "width": 1920, "height": 1080,
        "fps": 29.97002997002997, "frames": 3387,
        "duration_s": 113.0129, "inside_band": False,
    },
    "assets/media/store-aisle-detection.mp4": {
        "sha256": "3526fee39cac70d6366471e4324a6d63b337c8f8ab99c015521bee0d60ed6e04",
        "bytes": 9214573,
        "width": 720, "height": 404,
        "fps": 59.94005994005994, "frames": 3921,
        "duration_s": 65.41535, "inside_band": True,
    },
}


def _write_tiny_clip(path: Path, *, width: int = 64, height: int = 48) -> None:
    """A real, decodable one-frame clip **in a temporary directory**.

    Used only to prove that an added file changes the inventory. It is not, and must never
    become, an asset: the gate's whole point is that missing footage is a dependency, not
    something to generate.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0,
                             (width, height))
    if not writer.isOpened():
        raise RuntimeError("could not open a temporary VideoWriter")
    try:
        writer.write(np.zeros((height, width, 3), np.uint8))
    finally:
        writer.release()


class TheInventoryMatchesTheFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = content_gate.content_inventory("retail_sanity")

    def test_every_file_on_disk_is_inventoried_and_every_declared_file_is_on_disk(self):
        on_disk = {clip.path for clip in self.inventory.clips}
        self.assertEqual(on_disk, set(DECLARED),
                         "the files under assets/media/ and assets/scenarios/ do not match "
                         "the committed inventory; a clip was added, removed or renamed "
                         "without updating DECLARED")

    def test_every_measured_field_reproduces_the_committed_inventory(self):
        for clip in self.inventory.clips:
            expected = DECLARED[clip.path]
            self.assertEqual(clip.sha256, expected["sha256"],
                             f"{clip.path} is not the file the inventory measured")
            self.assertEqual(clip.bytes, expected["bytes"], clip.path)
            self.assertEqual((clip.width, clip.height),
                             (expected["width"], expected["height"]), clip.path)
            self.assertEqual(clip.fps, expected["fps"], clip.path)
            self.assertEqual(clip.frames, expected["frames"], clip.path)
            self.assertEqual(clip.duration_s, expected["duration_s"], clip.path)
            self.assertEqual(clip.inside_duration_band, expected["inside_band"], clip.path)

    def test_an_added_clip_fails_the_inventory_and_a_removed_one_does_too(self):
        # **The mutation evidence.** The inventory is a measurement, but the *declared*
        # table is the assertion; this proves both directions of drift are detectable
        # without touching the real assets.
        with tempfile.TemporaryDirectory(dir=content_gate.REPO_ROOT,
                                         prefix="content-gate-") as tmp:
            root = Path(tmp)
            media = root / "media"
            scenarios = root / "scenarios"
            media.mkdir()
            scenarios.mkdir()
            shutil.copy2(content_gate.MEDIA_DIR / "one-by-one-person-detection.mp4",
                         media / "one-by-one-person-detection.mp4")
            empty = content_gate.content_inventory(media_dir=media,
                                                   scenarios_dir=scenarios)
            self.assertEqual({clip.path for clip in empty.clips},
                             {content_gate._relative(
                                 media / "one-by-one-person-detection.mp4")},
                             "a one-clip directory did not inventory exactly that clip")

            _write_tiny_clip(media / "added-for-the-mutation.avi")
            added = content_gate.content_inventory(media_dir=media,
                                                   scenarios_dir=scenarios)
            self.assertEqual(len(added.clips), 2,
                             "adding a decodable clip did not change the inventory")
            self.assertNotEqual({clip.path for clip in added.clips}, set(DECLARED),
                                "an added clip left the declared inventory unchanged")

            (media / "added-for-the-mutation.avi").unlink()
            removed = content_gate.content_inventory(media_dir=media,
                                                     scenarios_dir=scenarios)
            self.assertEqual(len(removed.clips), 1,
                             "removing a clip did not change the inventory")

    def test_durations_are_measured_not_read_from_a_manifest(self):
        # A manifest in the temporary scenario tree claims 45 s. The decoder says 139.4 s.
        with tempfile.TemporaryDirectory(dir=content_gate.REPO_ROOT,
                                         prefix="content-gate-manifest-") as tmp:
            root = Path(tmp)
            media = root / "media"
            scenarios = root / "scenarios" / "pack"
            media.mkdir()
            scenarios.mkdir(parents=True)
            shutil.copy2(content_gate.MEDIA_DIR / "one-by-one-person-detection.mp4",
                         media / "one-by-one-person-detection.mp4")
            (scenarios / "manifest.json").write_text(
                '{"clips": [{"path": "x.mp4", "measured_duration_s": 45.0}]}',
                encoding="utf-8")
            measured = content_gate.content_inventory(media_dir=media,
                                                      scenarios_dir=root / "scenarios")
            clip = measured.clips[0]
            self.assertEqual(clip.duration_s, 139.4,
                             "the inventory took a duration from a manifest instead of "
                             "measuring the file")
            self.assertEqual(clip.frames / clip.fps, clip.duration_s)


class TheGateReportsTheShortfall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = content_gate.content_inventory("retail_sanity")
        cls.verdicts = {v.requirement: v for v in cls.inventory.verdicts()}

    def test_the_requirement_is_twelve_clips_across_four_packs_in_the_band(self):
        self.assertEqual(self.inventory.clips_required_total, 12)
        self.assertEqual(self.inventory.packs_required, 4)
        self.assertEqual(self.inventory.clips_required_per_pack, 3)
        self.assertEqual((self.inventory.min_duration_s,
                          self.inventory.max_duration_s), (45.0, 90.0))

    def test_the_gate_is_not_met_and_does_not_round_the_count_up(self):
        self.assertFalse(self.inventory.met,
                         "the content gate reports itself met; 3 clips is not 12")
        self.assertEqual(len(self.inventory.clips), 3)
        self.assertEqual(len(self.inventory.clips_in_band), 1)
        self.assertEqual(self.inventory.packs_provisioned, 1)

    def test_the_clip_count_verdict_names_the_nine_missing_clips(self):
        verdict = self.verdicts["three 45-90 s clips per pack"]
        self.assertFalse(verdict.met)
        self.assertIn("3 clip(s)", verdict.provisioned)
        self.assertIn("1 inside the band", verdict.provisioned)
        self.assertIn("missing 9 clip(s)", verdict.gap)
        self.assertIn("2 provisioned clip(s) outside the band", verdict.gap)

    def test_the_duration_verdict_states_how_far_each_clip_misses_by(self):
        verdict = self.verdicts["every clip inside the 45-90 s band"]
        self.assertFalse(verdict.met)
        self.assertIn("1 of 3", verdict.provisioned)
        # The measured excesses, not rounded: 113.0129 - 90 and 139.4 - 90.
        self.assertIn("23.0129 s over", verdict.gap)
        self.assertIn("49.4 s over", verdict.gap)

    def test_the_pack_verdict_names_the_three_missing_packs(self):
        verdict = self.verdicts["four scenario packs"]
        self.assertFalse(verdict.met)
        self.assertEqual(verdict.provisioned, "1 pack(s)")
        self.assertIn("missing 3 pack(s)", verdict.gap)

    def test_rights_are_recorded_as_not_cleared_rather_than_assumed(self):
        verdict = self.verdicts["rights-cleared footage"]
        self.assertFalse(verdict.met)
        self.assertEqual(verdict.provisioned, "0 cleared clip(s)")
        for clip in self.inventory.clips:
            self.assertTrue(clip.rights_status.startswith("not cleared"), clip)
            self.assertIn("no licence", clip.rights_status, clip)
            self.assertIn("unverified", clip.licence)

    def test_usage_is_read_from_the_application_not_from_a_manifest(self):
        used = {clip.path for clip in self.inventory.clips_used_by_app}
        self.assertEqual(used, {"assets/media/retail-checkout.mp4"},
                         "the clips the application actually opens changed; the inventory "
                         "must say which files the code names, not which a manifest lists")
        retail = next(clip for clip in self.inventory.clips
                      if clip.path == "assets/media/retail-checkout.mp4")
        self.assertIn("app/engine/retail_config.py", retail.used_in)

    def test_the_json_the_gate_serialises_carries_the_gap(self):
        payload = self.inventory.to_json()
        self.assertFalse(payload["met"])
        self.assertEqual(payload["requirement"]["clips_total"], 12)
        self.assertEqual(len(payload["clips"]), 3)
        self.assertEqual(len(payload["verdicts"]), 4)
        for verdict in payload["verdicts"]:
            self.assertTrue(verdict["gap"], verdict)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
