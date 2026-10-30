"""M6 Task 1: the reproducible offline bundle and its per-file SHA-256 manifest.

What this suite is for:

1. **The manifest is the bundle.** Every payload file must appear with a digest that is
   recomputed from the bytes on disk, not copied from `config/models.json` or from a
   builder's memory. A tampered byte has to fail verification.
2. **The wheelhouse is the pinned environment.** Every direct pin in `requirements.txt`
   must have a wheel, and no wheel may come from outside the pinned requirements and the
   dependencies those pins declare. `openvino-telemetry`, `PySide6-Essentials`,
   `PySide6-Addons` and `shiboken6` are in that closure because the pinned distributions
   declare them; a wheel such as `requests` is not.
3. **The bundle says it is unsigned.** There is no certificate on this machine and no
   signature is fabricated. The manifest carries `signed: False` and a note that says so.
4. **Nothing is silently omitted.** A source file that cannot be read or hashed aborts the
   build instead of disappearing from the manifest.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.engine.retail_config import RETAIL_CHECKOUT_CLIP
from tools import build_bundle as bb

REPO = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO / "requirements.txt"


def _wheel_name(distribution: str, version: str) -> str:
    """A PEP 427-shaped file name; the builder validates names, not wheel contents."""
    return f"{distribution.replace('-', '_')}-{version}-py3-none-any.whl"


def _make_wheelhouse(
    root: Path,
    *,
    omit: frozenset[str] = frozenset(),
    extra: tuple[tuple[str, str], ...] = (),
) -> Path:
    """A synthetic wheelhouse holding exactly the pinned closure, unless asked otherwise."""
    root.mkdir(parents=True)
    requirements = bb.parse_requirements(REQUIREMENTS)
    closure = bb.dependency_closure(requirements)
    for name, version in closure.items():
        if name in omit:
            continue
        (root / _wheel_name(name, version)).write_bytes(b"not a real wheel")
    for name, version in extra:
        (root / _wheel_name(name, version)).write_bytes(b"not a real wheel")
    return root


class TheBundleAndItsManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [
            path for path in (
                REPO / "assets" / "models" / "yolo11n-fp16" / "yolo11n.xml",
                REPO / "assets" / "models" / "yolo11n-fp16" / "yolo11n.bin",
                REPO / "assets" / "models" / "yolo11n-fp16" / "labels.txt",
                REPO / RETAIL_CHECKOUT_CLIP,
            )
            if not path.exists()
        ]
        if missing:
            raise unittest.SkipTest(
                "the bundle's models/media are not provisioned in this tree: "
                + ", ".join(str(path.relative_to(REPO)) for path in missing)
            )
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.wheelhouse = _make_wheelhouse(cls.root / "wheelhouse")
        cls.destination = cls.root / "bundle"
        cls.manifest = bb.build_bundle(cls.destination, cls.wheelhouse)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # -- 1. the manifest is computed from the files --------------------------------------

    def test_every_bundled_file_appears_with_its_sha256_read_from_the_file(self):
        on_disk = {}
        for path in self.destination.rglob("*"):
            if path.is_file() and path.name != bb.MANIFEST_NAME:
                on_disk[path.relative_to(self.destination).as_posix()] = (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                )
        self.assertEqual(
            on_disk,
            dict(self.manifest.files),
            "the manifest and the bytes on disk disagree; a file was omitted, added, or "
            "hashed from a value other than its own bytes",
        )
        self.assertNotIn(bb.MANIFEST_NAME, self.manifest.files,
                         "a manifest cannot contain its own digest; it is written after "
                         "the payload and excluded explicitly")

    def test_a_tampered_file_fails_verification(self):
        target = self.destination / "requirements.txt"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n# tampered after the manifest was written\n")
            report = bb.verify_bundle(self.destination)
            self.assertFalse(report.ok)
            self.assertIn("requirements.txt", report.mismatched)
        finally:
            target.write_bytes(original)
        self.assertTrue(bb.verify_bundle(self.destination).ok,
                        "restoring the original bytes must make verification pass again")

    def test_a_missing_or_added_file_fails_verification(self):
        missing_target = self.destination / "docs" / "install" / "SETUP.md"
        original = missing_target.read_bytes()
        try:
            missing_target.unlink()
            report = bb.verify_bundle(self.destination)
            self.assertFalse(report.ok)
            self.assertIn("docs/install/SETUP.md", report.missing)
        finally:
            missing_target.write_bytes(original)

        added = self.destination / "not-in-the-manifest.txt"
        added.write_text("surprise", encoding="utf-8")
        try:
            report = bb.verify_bundle(self.destination)
            self.assertFalse(report.ok)
            self.assertIn("not-in-the-manifest.txt", report.unexpected)
        finally:
            added.unlink()
        self.assertTrue(bb.verify_bundle(self.destination).ok)

    # -- 2. unsigned, and it says so ----------------------------------------------------

    def test_the_manifest_records_that_the_bundle_is_unsigned(self):
        self.assertIs(self.manifest.signed, False)
        note = self.manifest.signature_note.lower()
        self.assertIn("unsigned", note)
        self.assertIn("certificate", note,
                      "the note must say why it is unsigned, not merely that it is")
        reread = bb.read_manifest(self.destination)
        self.assertIs(reread.signed, False)
        self.assertEqual(reread.signature_note, self.manifest.signature_note)

    def test_the_manifest_names_the_python_and_platform_it_was_built_on(self):
        import platform
        self.assertEqual(self.manifest.python_version, platform.python_version())
        self.assertEqual(self.manifest.platform, platform.platform())

    # -- 3. the wheelhouse is the pinned environment ------------------------------------

    def test_the_wheelhouse_is_the_pinned_closure_and_no_pin_is_missing(self):
        requirements = bb.parse_requirements(REQUIREMENTS)
        wheels = {}
        for path in (self.destination / "wheelhouse").glob("*.whl"):
            name, version = bb.wheel_identity(path)
            self.assertNotIn(name, wheels, f"two wheels for {name}")
            wheels[name] = version

        for name, version in requirements.items():
            self.assertEqual(wheels.get(name), version,
                             f"{name}=={version} is pinned but its wheel is absent or is "
                             f"a different version")

        closure = bb.dependency_closure(requirements)
        self.assertEqual(
            set(wheels), set(closure),
            "the wheelhouse and the pinned dependency closure disagree; an unrelated "
            "wheel was added or a dependency of a pin was omitted",
        )
        # Spelled out so a resolver change cannot quietly widen the bundle.
        self.assertEqual(
            set(wheels) - set(requirements),
            {"openvino-telemetry", "pyside6-addons", "pyside6-essentials", "shiboken6"},
        )

    def test_a_wheel_outside_the_pinned_closure_fails_the_build(self):
        wheelhouse = _make_wheelhouse(
            self.root / "wheelhouse-extra", extra=(("requests", "2.31.0"),)
        )
        with self.assertRaises(bb.BundleError) as caught:
            bb.build_bundle(self.root / "bundle-extra", wheelhouse)
        self.assertIn("requests", str(caught.exception))

    def test_a_missing_pinned_wheel_fails_the_build(self):
        wheelhouse = _make_wheelhouse(
            self.root / "wheelhouse-missing", omit=frozenset({"numpy"})
        )
        with self.assertRaises(bb.BundleError) as caught:
            bb.build_bundle(self.root / "bundle-missing", wheelhouse)
        self.assertIn("numpy", str(caught.exception))

    # -- 4. the payload: models and the one clip the app uses ---------------------------

    def test_the_models_and_the_used_clip_are_in_the_bundle(self):
        config = json.loads((REPO / "config" / "models.json").read_text(encoding="utf-8"))
        expected = {RETAIL_CHECKOUT_CLIP}
        for model in config["models"]:
            expected.update(
                model[key]
                for key in ("ir_path", "weights_path", "labels_path")
                if model.get(key)
            )
        for relative in sorted(expected):
            self.assertIn(relative, self.manifest.files)
            self.assertTrue((self.destination / relative).is_file())
        self.assertIn("wheelhouse", {path.parts[0] for path in map(Path, self.manifest.files)})
        self.assertFalse((self.destination / "assets" / "media"
                          / "store-aisle-detection.mp4").exists(),
                         "only the clip the application uses belongs in the bundle")

    def test_no_bytecode_or_cache_is_bundled(self):
        for relative in self.manifest.files:
            self.assertNotIn("__pycache__", relative)
            self.assertFalse(relative.endswith(".pyc"))

    # -- 5. refusal instead of omission --------------------------------------------------

    def test_build_refuses_to_omit_a_file_it_cannot_hash(self):
        original = bb._hash_file

        def unreadable(path):
            if Path(path).name == "requirements.txt":
                raise OSError("injected unreadable source file")
            return original(path)

        destination = self.root / "bundle-refused"
        with mock.patch.object(bb, "_hash_file", side_effect=unreadable):
            with self.assertRaises(bb.BundleError) as caught:
                bb.build_bundle(destination, self.wheelhouse)
        self.assertIn("requirements.txt", str(caught.exception))
        self.assertFalse(destination.exists(),
                         "a failed build must not leave a partial tree that looks like a "
                         "bundle")
        self.assertFalse((destination / bb.MANIFEST_NAME).exists())

    # -- 6. reproducible ----------------------------------------------------------------

    def test_building_twice_produces_the_same_file_hashes(self):
        second = bb.build_bundle(self.root / "bundle-again", self.wheelhouse)
        self.assertEqual(dict(second.files), dict(self.manifest.files))
        self.assertEqual(second.python_version, self.manifest.python_version)
        self.assertEqual(second.platform, self.manifest.platform)
        self.assertIs(second.signed, False)


if __name__ == "__main__":
    unittest.main()
