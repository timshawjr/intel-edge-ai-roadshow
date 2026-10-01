import json
import tempfile
import unittest
from pathlib import Path

from tools.verify_sources import (
    looks_like_xml,
    looks_like_mp4,
    looks_like_binary,
    sha256_of,
    collect_sources,
    verify,
    main,
    Source,
    Result,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "models.json"
PACK_MANIFEST = REPO_ROOT / "assets" / "scenarios" / "retail_sanity" / "manifest.json"


class ContentVerification(unittest.TestCase):
    def test_xml_must_declare_itself(self):
        self.assertTrue(looks_like_xml(b'<?xml version="1.0"?><net/>'))
        self.assertFalse(looks_like_xml(b'<net/>'))
        self.assertFalse(looks_like_xml(b'{"error": "not found"}'))

    def test_mp4_must_carry_an_ftyp_box(self):
        self.assertTrue(looks_like_mp4(b'\x00\x00\x00\x18ftypmp42' + b'\x00' * 8))
        self.assertFalse(looks_like_mp4(b'<html>404</html>'))

    def test_binary_payload_must_not_be_html_or_an_lfs_pointer(self):
        self.assertTrue(looks_like_binary(bytes(range(256)) * 4))
        self.assertFalse(looks_like_binary(b'<!DOCTYPE html><html>nope</html>'))
        self.assertFalse(looks_like_binary(b'version https://git-lfs.github.com/spec/v1\n'))


class XmlBomHandling(unittest.TestCase):
    """M-1: `bytes.lstrip(charset)` strips a *set* of bytes, not a prefix."""

    def test_a_real_utf8_bom_before_the_declaration_is_tolerated(self):
        self.assertTrue(looks_like_xml(b'\xef\xbb\xbf<?xml version="1.0"?><net/>'))

    def test_stray_high_bytes_are_not_mistaken_for_a_bom(self):
        self.assertFalse(looks_like_xml(b'\xbf\xbf\xbf<?xml version="1.0"?><net/>'))


class BinaryCheckRejectsTextErrorBodies(unittest.TestCase):
    """A 200 can carry a plain-text refusal instead of weights. Reject it.

    These are the bodies the previous three-shape heuristic actually let through.
    """

    def test_a_rate_limit_body_is_not_a_weight_payload(self):
        self.assertFalse(looks_like_binary(b"rate limit exceeded"))

    def test_a_blocked_access_body_is_not_a_weight_payload(self):
        self.assertFalse(looks_like_binary(b"Your access has been temporarily blocked."))

    def test_a_permission_denied_body_is_not_a_weight_payload(self):
        self.assertFalse(looks_like_binary(b"Permission denied for this resource"))

    def test_an_xml_body_served_where_weights_belongs_is_rejected(self):
        self.assertFalse(looks_like_binary(b'<?xml version="1.0"?><net/>'))

    def test_a_proportion_of_printable_text_means_it_is_text(self):
        self.assertFalse(looks_like_binary(b"A" * 4096))

    def test_the_real_weights_still_pass(self):
        weights = REPO_ROOT / "assets" / "models" / "yolo11n-fp16" / "yolo11n.bin"
        self.assertTrue(looks_like_binary(weights.read_bytes()[:65536]),
                        "the printable-ratio rule must not reject real FP16 weights")


class HashIsTheGate(unittest.TestCase):
    """The SHA-256 comparison is the only thing that proves which bytes we have.

    A mutation that made `verify` always return ok would pass every other test in
    this file. These tests exist to kill that mutation.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.path = self.dir / "weights.bin"
        self.payload = bytes(range(256)) * 8
        self.path.write_bytes(self.payload)
        self.real = sha256_of(self.path)
        self.source = Source("probe", "binary", self.path)

    def test_matching_bytes_and_matching_hash_is_ok(self):
        result = verify(self.source, self.real)
        self.assertTrue(result.ok, result.detail)
        self.assertIn(self.real, result.detail)

    def test_a_wrong_recorded_hash_is_not_ok(self):
        wrong = "0" * 64
        result = verify(self.source, wrong)
        self.assertFalse(result.ok)
        self.assertIn("sha256 mismatch", result.detail)

    def test_no_recorded_hash_fails_closed(self):
        result = verify(self.source, None)
        self.assertFalse(result.ok, "an unrecorded hash must never read as verified")
        self.assertIn("no sha256 is recorded", result.detail)

    def test_a_missing_file_is_not_ok(self):
        result = verify(Source("probe", "binary", self.dir / "absent.bin"), self.real)
        self.assertFalse(result.ok)
        self.assertIn("file not found", result.detail)

    def test_an_empty_file_is_not_ok(self):
        empty = self.dir / "empty.bin"
        empty.write_bytes(b"")
        result = verify(Source("probe", "binary", empty), sha256_of(empty))
        self.assertFalse(result.ok)
        self.assertIn("empty", result.detail)


def _write_config(directory: Path, recorded_sha256: str, path: Path) -> Path:
    config = {
        "models": [
            {
                "id": "probe",
                "ir_path": "ir.xml",
                "sha256": recorded_sha256,
            }
        ],
        "media": [],
    }
    cfg = directory / "models.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    return cfg


class ExitCodeReflectsFailure(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.payload = b'\x00\x00\x00\x18ftypmp42' + b"\x00" * 8
        # an mp4-shaped file, so only the hash can fail it
        self.media = self.dir / "clip.mp4"
        self.media.write_bytes(self.payload)
        self.real = sha256_of(self.media)

    def _config(self, recorded: str) -> Path:
        cfg = self.dir / f"cfg_{abs(hash(recorded))}.json"
        cfg.write_text(
            json.dumps(
                {
                    "models": [],
                    "media": [
                        {
                            "id": "probe_clip",
                            "path": str(self.media),
                            "sha256": recorded,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return cfg

    def test_matching_hash_exits_zero(self):
        self.assertEqual(main(["--config", str(self._config(self.real))]), 0)

    def test_a_wrong_hash_on_an_existing_file_exits_one(self):
        self.assertEqual(main(["--config", str(self._config("0" * 64))]), 1)

    def test_a_missing_file_exits_one(self):
        cfg = self.dir / "absent.json"
        cfg.write_text(
            json.dumps({"models": [], "media": [{"id": "gone", "path": "no/such.mp4",
                                                 "sha256": self.real}]}),
            encoding="utf-8",
        )
        self.assertEqual(main(["--config", str(cfg)]), 1)

    def test_an_unreadable_config_exits_one_without_a_traceback(self):
        missing = self.dir / "no_such_config.json"
        self.assertEqual(main(["--config", str(missing)]), 1)
        malformed = self.dir / "malformed.json"
        malformed.write_text("{not json", encoding="utf-8")
        self.assertEqual(main(["--config", str(malformed)]), 1)


class LabelsAreVerifiedToo(unittest.TestCase):
    """I-5: a recorded hash with no verification row is a hash nobody checks."""

    def test_collect_sources_emits_a_row_for_the_label_file(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        rows = collect_sources(config)
        label_rows = [r for r, _, _ in rows if r.kind == "text"]
        self.assertEqual(len(label_rows), 1, f"expected one text row, got {rows}")
        self.assertTrue(label_rows[0].path.name.endswith("labels.txt"))

    def test_the_label_file_must_decode_and_have_exactly_num_classes_lines(self):
        labels = REPO_ROOT / "assets" / "models" / "yolo11n-fp16" / "labels.txt"
        source = Source("yolo11n-fp16:labels", "text", labels)
        good = verify(source, sha256_of(labels), expected_lines=80)
        self.assertTrue(good.ok, good.detail)
        wrong_count = verify(source, sha256_of(labels), expected_lines=79)
        self.assertFalse(wrong_count.ok, "a 79-line label file must not verify")
        self.assertIn("79", wrong_count.detail)

    def test_every_recorded_artifact_in_the_shipped_config_is_checked(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        rows = collect_sources(config)
        # 1 xml + 1 weights + 1 labels + 3 media
        self.assertEqual(len(rows), 6)
        for source, recorded, _ in rows:
            self.assertIsNotNone(recorded, f"{source.id} has no recorded sha256")


class RecordedContentGateAgreesWithThePack(unittest.TestCase):
    """I-4: the two content-gate records must not drift apart."""

    def _gate_durations(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        pack = json.loads(PACK_MANIFEST.read_text(encoding="utf-8"))
        recorded = config["content_gate"]["measured_durations_s"]
        pack_durations = pack["content_gate"]["measured_durations_s"]
        self.assertEqual(
            sorted(recorded),
            sorted(pack_durations),
            "config/models.json and the pack manifest disagree on measured durations",
        )
        return sorted(recorded)

    def test_the_two_records_agree_on_the_numbers(self):
        self._gate_durations()

    def test_the_recorded_durations_are_the_measured_clip_durations(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        measured = sorted(m["measured_duration_s"] for m in config["media"])
        self.assertEqual(self._gate_durations(), measured)

    def test_both_records_agree_the_gate_is_unmet(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        pack = json.loads(PACK_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            config["content_gate"]["status"], pack["content_gate"]["status"]
        )
        self.assertIn("unmet", config["content_gate"]["status"])
