"""**M5 Task 5: the content gate, measured from the files.**

The requirement is spec section 4: **three 45-90 s clips per pack, four packs, twelve
clips**. The repository does not have that, and this suite's job is to make the shortfall
**unmissable** rather than to make the gate pass:

1. The inventory is measured from the video files with `cv2.VideoCapture`; the committed
   `DECLARED` table below is what those measurements are, and the test fails if a clip is
   added, removed or re-encoded without the table being updated.
2. Every gate verdict carries the requirement, the provisioned number and the **gap**.
   `met` is false for the clip count, the duration band, the pack count and rights.
3. Rights are **not** inferred from a filename: three clips have no licence sidecar and are
   recorded `not cleared`; six have a CC BY 4.0 sidecar and are recorded `licence file
   present; clearance not established`. A sidecar is never upgraded to `cleared`, and the
   gate still reports **0 rights-cleared**.
4. The three provisioned clips that measure under 45 s are an **operator-recorded
   deviation**, not an in-band result: the duration verdict counts them outside the band.
5. Nothing acquires, generates or substitutes footage. The only generated file in this
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


#: The committed inventory, measured with `cv2.VideoCapture` on 2026-10-31 and recorded at
#: full precision. **This is the assertion, not a cache**: re-running the measurement must
#: reproduce every field, and a clip added to or removed from `assets/media/` changes the
#: set of paths and fails here.
DECLARED = {
    "assets/media/bottle-detection.mp4": {
        "sha256": "d52ba94aedf8a923c342fe9ea1d2bd85f712c4cc0f49a6de1bac43eebe3a48ff",
        "bytes": 504961,
        "width": 640, "height": 360, "fps": 29.833333333333332, "frames": 1189,
        "duration_s": 39.85474860335196, "inside_band": False,
    },
    "assets/media/car-detection.mp4": {
        "sha256": "d31e0ebf194cc16e50eea784e7c3b0b1bb7976a4fb9b3fec26cb92a6704cca34",
        "bytes": 2811553,
        "width": 768, "height": 432, "fps": 12.5, "frames": 377,
        "duration_s": 30.16, "inside_band": False,
    },
    "assets/media/classroom.mp4": {
        "sha256": "a69bd5e39ff0e74286d13bcbfdadddd307b85decd2d9853db7518e0028785e31",
        "bytes": 13548133,
        "width": 1920, "height": 1080, "fps": 30.0, "frames": 984,
        "duration_s": 32.8, "inside_band": False,
    },
    "assets/media/one-by-one-person-detection.mp4": {
        "sha256": "a5964aa259099a482a8b360ffc2c57b5a30f84d5919236a4dad01f8e929ac07c",
        "bytes": 3291918,
        "width": 768, "height": 432, "fps": 10.0, "frames": 1394,
        "duration_s": 139.4, "inside_band": False,
    },
    "assets/media/people-detection.mp4": {
        "sha256": "18ffe8672d741e3e29c9d891d22c59d453720b086c25b35c88b393d55f92f693",
        "bytes": 5482579,
        "width": 768, "height": 432, "fps": 12.0, "frames": 596,
        "duration_s": 49.666666666666664, "inside_band": True,
    },
    "assets/media/person-bicycle-car-detection.mp4": {
        "sha256": "452b11b7e0efbd019f1d9570d0c790e90416ad4ad29eec6003872d08443140ef",
        "bytes": 6031199,
        "width": 768, "height": 432, "fps": 12.0, "frames": 647,
        "duration_s": 53.916666666666664, "inside_band": True,
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
    "assets/media/worker-zone-detection.mp4": {
        "sha256": "b8b58b8100a81600bc42ee7ee082ced777b12a32a3619139ef289d27cbdcd284",
        "bytes": 12596893,
        "width": 1920, "height": 1080, "fps": 59.94005994005994, "frames": 4548,
        "duration_s": 75.8758, "inside_band": True,
    },
}

#: The six clips added on 2026-10-31 from intel-iot-devkit/sample-videos, with a CC BY 4.0
#: sidecar each. The three under 45 s are a recorded deviation, not an in-band result.
SIDECAR_CLIPS = (
    "assets/media/bottle-detection.mp4",
    "assets/media/car-detection.mp4",
    "assets/media/classroom.mp4",
    "assets/media/people-detection.mp4",
    "assets/media/person-bicycle-car-detection.mp4",
    "assets/media/worker-zone-detection.mp4",
)


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

    def test_a_clip_scoped_sidecar_flips_only_its_own_clip(self):
        with tempfile.TemporaryDirectory(dir=content_gate.REPO_ROOT,
                                         prefix="content-gate-licence-") as tmp:
            root = Path(tmp)
            media = root / "media"
            scenarios = root / "scenarios"
            media.mkdir()
            scenarios.mkdir()
            names = ("one-by-one-person-detection.mp4", "store-aisle-detection.mp4")
            for name in names:
                shutil.copy2(content_gate.MEDIA_DIR / name, media / name)

            # A bare LICENSE in the folder cannot be attributed to one clip, so it must
            # not clear anything.
            (media / "LICENSE").write_text("Licence: bare directory licence\n",
                                           encoding="utf-8")
            bare = {clip.path: clip for clip in content_gate.content_inventory(
                media_dir=media, scenarios_dir=scenarios).clips}
            for clip in bare.values():
                self.assertTrue(clip.rights_status.startswith("not cleared"), clip)

            (media / "one-by-one-person-detection.mp4.LICENSE").write_text(
                "Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)\n",
                encoding="utf-8")
            scoped = {clip.path: clip for clip in content_gate.content_inventory(
                media_dir=media, scenarios_dir=scenarios).clips}
            licensed = scoped[content_gate._relative(
                media / "one-by-one-person-detection.mp4")]
            other = scoped[content_gate._relative(
                media / "store-aisle-detection.mp4")]
            self.assertEqual(licensed.rights_status,
                             "licence file present; clearance not established")
            self.assertIn("Creative Commons Attribution 4.0", licensed.licence)
            self.assertTrue(other.rights_status.startswith("not cleared"),
                            "a sidecar cleared a clip it is not named after")


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
                         "the content gate reports itself met; 9 clips is not 12")
        self.assertEqual(len(self.inventory.clips), 9)
        self.assertEqual(len(self.inventory.clips_in_band), 4)
        self.assertEqual(self.inventory.packs_provisioned, 1)

    def test_the_clip_count_verdict_names_the_three_missing_clips(self):
        verdict = self.verdicts["three 45-90 s clips per pack"]
        self.assertFalse(verdict.met)
        self.assertIn("9 clip(s)", verdict.provisioned)
        self.assertIn("4 inside the band", verdict.provisioned)
        self.assertIn("missing 3 clip(s)", verdict.gap)
        self.assertIn("5 provisioned clip(s) outside the band", verdict.gap)

    def test_the_duration_verdict_states_how_far_each_clip_misses_by(self):
        verdict = self.verdicts["every clip inside the 45-90 s band"]
        self.assertFalse(verdict.met)
        self.assertIn("4 of 9", verdict.provisioned)
        # The measured excesses and shortfalls, not rounded: 113.0129 - 90,
        # 139.4 - 90, 45 - 39.85474860335196, 45 - 30.16 and 45 - 32.8.
        self.assertIn("23.0129 s over", verdict.gap)
        self.assertIn("49.4 s over", verdict.gap)
        self.assertIn("5.14525 s short", verdict.gap)
        self.assertIn("14.84 s short", verdict.gap)
        self.assertIn("12.2 s short", verdict.gap)

    def test_the_sub_45s_clips_are_a_deviation_counted_outside_the_band(self):
        # The operator chose to use these three. That is a product decision, not a change
        # to the specification: the gate must still count them outside 45-90 s.
        short = {
            "assets/media/bottle-detection.mp4": 39.85474860335196,
            "assets/media/car-detection.mp4": 30.16,
            "assets/media/classroom.mp4": 32.8,
        }
        measured = {clip.path: clip for clip in self.inventory.clips}
        for path, duration in short.items():
            clip = measured[path]
            self.assertEqual(clip.duration_s, duration, path)
            self.assertFalse(clip.inside_duration_band,
                             f"{path} was counted inside a band it does not meet")
            self.assertAlmostEqual(clip.duration_shortfall_s, 45.0 - duration)

    def test_the_pack_verdict_names_the_three_missing_packs(self):
        verdict = self.verdicts["four scenario packs"]
        self.assertFalse(verdict.met)
        self.assertEqual(verdict.provisioned, "1 pack(s)")
        self.assertIn("missing 3 pack(s)", verdict.gap)

    def test_rights_record_a_sidecar_as_provenance_and_never_as_clearance(self):
        verdict = self.verdicts["rights-cleared footage"]
        self.assertFalse(verdict.met)
        self.assertEqual(verdict.provisioned, "0 cleared clip(s)",
                         "a licence sidecar was upgraded to a clearance")
        self.assertEqual(len(self.inventory.clips_with_unverified_rights), 9)

        measured = {clip.path: clip for clip in self.inventory.clips}
        for path in SIDECAR_CLIPS:
            clip = measured[path]
            self.assertEqual(clip.rights_status,
                             "licence file present; clearance not established", path)
            self.assertIn("Creative Commons Attribution 4.0", clip.licence, path)

        for path in set(DECLARED) - set(SIDECAR_CLIPS):
            clip = measured[path]
            self.assertTrue(clip.rights_status.startswith("not cleared"), path)
            self.assertIn("unverified", clip.licence, path)

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
        self.assertEqual(len(payload["clips"]), 9)
        self.assertEqual(len(payload["verdicts"]), 4)
        for verdict in payload["verdicts"]:
            self.assertTrue(verdict["gap"], verdict)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
