"""M6 Task 4: the diagnostic export - gates, environment, and no camera imagery.

The export exists so a booth failure can be diagnosed away from the booth. That makes two
properties load-bearing:

1. **It carries the gates as they are, including the ones that fail.** A diagnostic bundle
   that silently drops a failing gate is worse than no bundle: it turns an unmet gate into
   an invisible one.
2. **It carries no camera imagery by default.** The check reads **bytes**, not member
   names, because a frame appended to a log or embedded in an error string is exactly the
   leak a name check misses. The export refuses to write a bundle whose bytes contain an
   image magic number, and the scan is shown to detect PNG, JPEG and WebP in constructed
   payloads so a zero is a measurement rather than a blind spot.

The export is also a tool in a repository where one tool rewrites `config/telemetry_map.json`
as a side effect. This one must not: the test hashes the file before and after a real export
and requires it unchanged.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from app import theme
from tools import diagnostic_export as de

REPO = Path(__file__).resolve().parents[1]
TELEMETRY_MAP = REPO / "config" / "telemetry_map.json"

FAKE_ENVIRONMENT = {
    "hostname": "FAKE-HOST-42",
    "python_version": "3.12.10",
    "ac_status": None,
    "sku_records": [{"source": "fake", "value": "FAKE-SKU"}],
}
FAKE_DISCOVERY = {
    "openvino_version": "fake-openvino",
    "available_devices": ["CPU", "GPU", "NPU"],
    "device_names": {"CPU": "Fake CPU"},
}
FAKE_SELFTEST = (
    "  ok   preprocess           letterbox 640x640 pad=114 scaled=640x360\n"
    "  ok   infer                1.00 ms on confirmed ['CPU']\n"
    "PASS\n"
)
FAKE_RECORD = b'{"format": "benchmark-record/1", "label": "FAKE-RECORD"}\n'


def _gates() -> tuple[de.GateResult, ...]:
    return (
        de.GateResult(gate="content", status="failed",
                      measured={"clips": 3, "required": 12, "rights_cleared": 0},
                      detail="3 of 12 clips provisioned, 0 rights-cleared",
                      source="fake content inventory"),
        de.GateResult(gate="legibility", status="failed",
                      measured={"cap_arcmin_at_3m": 11.2},
                      detail="fails at 3 m",
                      source="fake theme"),
        de.GateResult(gate="transition", status="failed",
                      measured={"over_two_second": 37, "transitions": 50},
                      detail="37 of 50 over 2 s",
                      source="fake M2 file"),
    )


def _export(destination: Path, **overrides):
    kwargs = dict(
        environment=FAKE_ENVIRONMENT,
        discovery=FAKE_DISCOVERY,
        selftest=FAKE_SELFTEST,
        gates=_gates(),
        benchmark_record=FAKE_RECORD,
    )
    kwargs.update(overrides)
    return de.diagnostic_bundle(destination, **kwargs)


def _members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


class TheZipCarriesWhatADiagnosisNeeds(unittest.TestCase):
    def test_the_zip_has_the_environment_discovery_selftest_gates_and_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _export(Path(tmp) / "diagnostic.zip")
            self.assertTrue(bundle.is_file())
            members = _members(bundle)

        self.assertEqual(
            set(members),
            {"manifest.json", "environment.json", "device_discovery.json",
             "selftest.txt", "gates.json", "benchmark_record.json", "README.txt"},
        )
        self.assertEqual(json.loads(members["environment.json"]), FAKE_ENVIRONMENT)
        self.assertEqual(json.loads(members["device_discovery.json"]), FAKE_DISCOVERY)
        self.assertIn("PASS", members["selftest.txt"].decode("utf-8"))
        self.assertEqual(json.loads(members["benchmark_record.json"]), json.loads(FAKE_RECORD))

    def test_the_manifest_hashes_every_member_from_its_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _export(Path(tmp) / "diagnostic.zip")
            members = _members(bundle)
        manifest = json.loads(members["manifest.json"])
        self.assertFalse(manifest["signed"])
        for name, payload in members.items():
            if name == "manifest.json":
                continue
            recorded = manifest["members"][name]
            self.assertEqual(recorded["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(recorded["bytes"], len(payload))

    def test_a_failing_gate_is_written_as_failing_not_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _export(Path(tmp) / "diagnostic.zip")
            payload = json.loads(_members(bundle)["gates.json"])
        self.assertEqual(len(payload["gates"]), 3)
        statuses = {gate["gate"]: gate["status"] for gate in payload["gates"]}
        self.assertEqual(statuses, {"content": "failed", "legibility": "failed",
                                    "transition": "failed"})
        self.assertFalse(payload["all_passed"])
        content = next(gate for gate in payload["gates"] if gate["gate"] == "content")
        self.assertEqual(content["measured"]["clips"], 3)
        self.assertEqual(content["measured"]["rights_cleared"], 0)
        self.assertIn("3 of 12", content["detail"])

    def test_a_missing_benchmark_record_is_recorded_as_unavailable(self):
        missing = REPO / "does" / "not" / "exist" / "record.json"
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _export(Path(tmp) / "diagnostic.zip",
                             benchmark_record=None, benchmark_record_path=missing)
            payload = json.loads(_members(bundle)["benchmark_record.json"])
        self.assertFalse(payload["available"])
        self.assertIn("record.json", payload["reason"])


class NoCameraImageryByDefault(unittest.TestCase):
    def test_no_member_carries_image_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _export(Path(tmp) / "diagnostic.zip")
            members = _members(bundle)
        hits = {name: magic for name, payload in members.items()
                if (magic := de.image_magic(payload)) is not None}
        self.assertEqual(hits, {}, "a diagnostic bundle carried camera imagery by default")

    def test_the_scan_can_actually_find_an_image(self):
        cases = {
            "png": b"\x89PNG\r\n\x1a\n" + b"fake",
            "jpeg": b"\xff\xd8\xff\xe0" + b"fake",
            "webp": b"RIFF\x10\x00\x00\x00WEBP" + b"fake",
        }
        for expected, payload in cases.items():
            with self.subTest(format=expected):
                self.assertEqual(de.image_magic(payload), expected)

    def test_the_export_refuses_a_member_that_would_carry_an_image(self):
        image = b"\x89PNG\r\n\x1a\n" + b"a frame appended to a log"
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "diagnostic.zip"
            with self.assertRaises(de.ImageryDetected) as caught:
                _export(destination, selftest="before\n".encode() + image + b"\nafter\n")
            self.assertFalse(destination.exists(),
                             "a refused export must not leave a partial bundle on disk")
        self.assertIn("selftest.txt", str(caught.exception))


class TheExportHasNoSideEffects(unittest.TestCase):
    def test_the_export_does_not_modify_the_telemetry_map(self):
        before = hashlib.sha256(TELEMETRY_MAP.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            _export(Path(tmp) / "diagnostic.zip")
        after = hashlib.sha256(TELEMETRY_MAP.read_bytes()).hexdigest()
        self.assertEqual(
            after, before,
            "the diagnostic export rewrote config/telemetry_map.json; that file is "
            "verified data and a diagnostic must not change the machine it describes",
        )


class TheGatesAreMeasuredNotHardcoded(unittest.TestCase):
    def test_the_content_gate_status_comes_from_the_inventory(self):
        failed = SimpleNamespace(
            clips=[object()] * 3, clips_required_total=12,
            clips_with_unverified_rights=[object()] * 3, clips_in_band=[object()],
            packs_provisioned=1, met=False,
        )
        passed = SimpleNamespace(
            clips=[object()] * 12, clips_required_total=12,
            clips_with_unverified_rights=[], clips_in_band=[object()] * 12,
            packs_provisioned=4, met=True,
        )
        failed_result = de.content_gate_result(failed)
        passed_result = de.content_gate_result(passed)
        self.assertEqual(failed_result.status, "failed")
        self.assertEqual(passed_result.status, "passed")
        self.assertIn("3 of 12", failed_result.detail)
        self.assertIn("0 rights-cleared", failed_result.detail)

    def test_the_legibility_gate_is_computed_from_the_geometry_it_is_given(self):
        attached = de.legibility_gate_result()
        required_panel = de.legibility_gate_result(
            mm_per_logical_px=theme.MM_PER_LOGICAL_PX)
        self.assertEqual(attached.status, "failed")
        self.assertEqual(required_panel.status, "passed",
                         "the 55-inch requirement must pass at 3 m; if it does not, the "
                         "angle is not being computed from the geometry")
        self.assertAlmostEqual(attached.measured["cap_arcmin"]["3.0"], 11.2, places=1)
        self.assertAlmostEqual(attached.measured["x_arcmin"]["2.0"], 12.2, places=1)

    def test_the_transition_gate_reports_the_committed_distribution(self):
        met = {"warm_elapsed_p50": 1.5, "over_two_second": 0, "transitions": 50,
               "timeout_s": 2.0, "acknowledged": 50}
        missed = {"warm_elapsed_p50": 2.628, "over_two_second": 37, "transitions": 50,
                  "timeout_s": 2.0, "acknowledged": 26}
        self.assertEqual(de.transition_gate_result(met).status, "passed")
        missed_result = de.transition_gate_result(missed)
        self.assertEqual(missed_result.status, "failed")
        self.assertEqual(missed_result.measured["over_two_second"], 37)
        self.assertIn("37 of 50", missed_result.detail)


class TheBundleSaysWhatItIs(unittest.TestCase):
    def test_the_readme_states_no_imagery_and_says_unsigned(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _export(Path(tmp) / "diagnostic.zip")
            readme = _members(bundle)["README.txt"].decode("utf-8").lower()
        self.assertIn("no camera imagery", readme)
        self.assertIn("unsigned", readme)

    def test_the_benchmark_record_is_copied_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _export(Path(tmp) / "diagnostic.zip")
            payload = _members(bundle)["benchmark_record.json"]
        self.assertEqual(payload, FAKE_RECORD)


if __name__ == "__main__":
    unittest.main()
