"""The export ZIP: manifest, transitions, metrics and logs - and **not** camera frames.

Review Focus #3 is *"an export that quietly includes imagery"*. A test that checks member
*names* is not enough: a frame can be written as `metrics_raw.jsonl`, or appended to
`errors.log`, and the names would still look clean. Every test here therefore reads every
member's **bytes** and looks for image magic numbers (`\\xFF\\xD8\\xFF` for JPEG,
`\\x89PNG` for PNG, `RIFF....WEBP` for WebP).

The other properties:

- the bundle contains every required member, and `metrics_raw.jsonl` is one JSON object
  per line that re-parses;
- `include_imagery=True` adds the frames **and** the README says so; the default README
  says they are excluded;
- a `recorded` record exports with `Recorded demonstration - metrics are not live`, so a
  bundle of a playback cannot be read as a live run;
- two exports have identical **contents** (the ZIP timestamp bytes may differ);
- the summary reports a median **and** a range from settled samples only, and an
  unavailable metric is `null` with its reason, never `0`.
"""

from __future__ import annotations

import json
import statistics
import tempfile
import unittest
import zipfile
from pathlib import Path

from app.contracts.benchmark_record import (
    BenchmarkRecord,
    EnvironmentManifest,
    ImageryFrame,
    RecordedSample,
    SamplePhase,
    TrialSummary,
    WorkloadFingerprint,
)
from app.contracts.device_status import Validity
from app.contracts.metric_sample import MetricSample
from app.export.bundle import export_bundle

REQUIRED_MEMBERS = {
    "manifest.json",
    "routing_transitions.json",
    "metrics_summary.json",
    "metrics_raw.jsonl",
    "errors.log",
    "README.txt",
}

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 40
WEBP = b"RIFF" + (48).to_bytes(4, "little") + b"WEBP" + b"\x00" * 40


def image_magic(blob: bytes) -> str | None:
    """The image format whose magic number appears in `blob`, or None.

    Scans the whole member, not a prefix: a frame appended to a text log is the leak
    this test exists to catch.
    """
    if b"\xff\xd8\xff" in blob:
        return "jpeg"
    if b"\x89PNG" in blob:
        return "png"
    index = blob.find(b"RIFF")
    while index >= 0:
        if blob[index + 8:index + 12] == b"WEBP":
            return "webp"
        index = blob.find(b"RIFF", index + 1)
    return None


def _fingerprint(**overrides) -> WorkloadFingerprint:
    base = dict(
        model_id="yolov8s-int8",
        model_sha256="a" * 64,
        input_shape=(1, 3, 640, 640),
        conf_threshold=0.25,
        iou_threshold=0.45,
        stream_count=3,
        target_fps=15.0,
        device_layout=("GPU", "NPU"),
        decode_path="msmf",
        openvino_version="2026.4.0-22959-99c81491cc3-releases/2026/4",
        numpy_version="2.5.3",
    )
    base.update(overrides)
    return WorkloadFingerprint(**base)


def _environment(**overrides) -> EnvironmentManifest:
    base = dict(
        machine_id="booth-01",
        cpu="Intel(R) Core(TM) Ultra 9 288V",
        gpu="Intel(R) Arc(TM) Graphics",
        npu="Intel(R) AI Boost",
        gpu_driver="32.0.101.7026",
        npu_driver="32.0.100.3104",
        windows_build="10.0.26100",
        ram_bytes=32 * 1024 ** 3,
        power_plan="Balanced",
        ac_status="AC",
        monitor="1920x1080 @ 2 m",
        python_version="3.12.10",
    )
    base.update(overrides)
    return EnvironmentManifest(**base)


def _sample(metric, value, *, at=10.0, phase=SamplePhase.SETTLED, unit="ms",
            device_scope="GPU", reason=None) -> RecordedSample:
    if value is None:
        sample = MetricSample(
            name=metric, value=None, unit=unit, provider="worker",
            device_scope=device_scope, process_scope=("worker",),
            window_seconds=0.0, sampled_at=at, validity=Validity.UNAVAILABLE,
            unavailable_reason=reason or "counter unreadable")
    else:
        sample = MetricSample(
            name=metric, value=value, unit=unit, provider="worker",
            device_scope=device_scope, process_scope=("worker",),
            window_seconds=1.0, sampled_at=at, validity=Validity.VALID)
    return RecordedSample(sample=sample, phase=phase)


def _record(*, mode="live", samples=None, imagery=(), errors=()):
    if samples is None:
        samples = (
            _sample("inference_ms", 5.2, at=1.0, phase=SamplePhase.COLD),
            _sample("inference_ms", 11.0, at=10.0),
            _sample("inference_ms", 12.0, at=11.0),
            _sample("npu_utilization_percent", None, at=12.0, unit="percent",
                    device_scope="NPU", reason="no PDH counter for this engine"),
        )
    unavailable = {
        s.metric: s.sample.unavailable_reason
        for s in samples if s.value is None
    }
    return BenchmarkRecord(
        mode=mode,
        started_at="2026-10-01T10:00:00",
        duration_s=60.0,
        fingerprint=_fingerprint(),
        environment=_environment(),
        samples=samples,
        trial_summary=(TrialSummary(condition="sustained", placement=("GPU",),
                                    settled_after_s=5.0),),
        exclusions=("first 5 s are warm-up",),
        comparison_eligible=mode == "live",
        unavailable_reasons=unavailable,
        warmup_s=5.0,
        routing_transitions=(
            {"at_s": 2.0, "from": "NPU", "to": "GPU", "reason": "placement fallback"},
        ),
        errors=errors,
        imagery=imagery,
    )


def _frames():
    return (
        ImageryFrame(name="frame-001.png", media_type="image/png", data=PNG,
                     captured_at=1.0),
        ImageryFrame(name="frame-002.jpg", media_type="image/jpeg", data=JPEG,
                     captured_at=2.0),
        ImageryFrame(name="frame-003.webp", media_type="image/webp", data=WEBP,
                     captured_at=3.0),
    )


class TheBundleContainsWhatItClaims(unittest.TestCase):
    def test_every_required_member_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(_record(), Path(tmp) / "bundle.zip")
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(set(archive.namelist()), REQUIRED_MEMBERS)

    def test_no_member_name_looks_like_imagery_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(_record(imagery=_frames()), Path(tmp) / "bundle.zip")
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    lowered = name.lower()
                    self.assertNotIn("imagery/", lowered, name)
                    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
                        self.assertFalse(lowered.endswith(suffix), name)

    def test_no_image_bytes_are_present_by_default(self):
        """Names are not the proof; the bytes are. The record holds frames in memory."""
        frame = _frames()[0]
        self.assertEqual(image_magic(frame.data), "png")
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(_record(imagery=_frames()), Path(tmp) / "bundle.zip")
            with zipfile.ZipFile(path) as archive:
                hits = {name: image_magic(archive.read(name))
                        for name in archive.namelist()}
            found = {name: kind for name, kind in hits.items() if kind}
            self.assertEqual(found, {}, f"camera imagery leaked into the default export: {found}")

    def test_the_scan_detects_imagery_when_it_is_included(self):
        """A scan that finds nothing because it cannot see anything proves nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(_record(imagery=_frames()), Path(tmp) / "bundle.zip",
                                 include_imagery=True)
            with zipfile.ZipFile(path) as archive:
                hits = {name: image_magic(archive.read(name))
                        for name in archive.namelist()}
            found = {name: kind for name, kind in hits.items() if kind}
            self.assertEqual(set(found.values()), {"png", "jpeg", "webp"})
            self.assertEqual(len(found), 3)

    def test_include_imagery_adds_the_frames_and_the_readme_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(_record(imagery=_frames()), Path(tmp) / "bundle.zip",
                                 include_imagery=True)
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                readme = archive.read("README.txt").decode("utf-8")
            self.assertEqual(names, REQUIRED_MEMBERS | {
                "imagery/frame-001.png", "imagery/frame-002.jpg",
                "imagery/frame-003.webp"})
            self.assertIn("Camera imagery IS included", readme)
            self.assertIn("3 camera frame", readme)

    def test_the_default_readme_says_imagery_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(_record(), Path(tmp) / "bundle.zip")
            with zipfile.ZipFile(path) as archive:
                readme = archive.read("README.txt").decode("utf-8")
            self.assertIn("Camera imagery is NOT included", readme)

    def test_raw_metrics_are_one_json_object_per_line_and_reparse(self):
        record = _record()
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(record, Path(tmp) / "bundle.zip")
            with zipfile.ZipFile(path) as archive:
                text = archive.read("metrics_raw.jsonl").decode("utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        self.assertEqual(len(lines), len(record.samples))
        parsed = [json.loads(line) for line in lines]
        self.assertEqual([p["phase"] for p in parsed],
                         [s.phase.value for s in record.samples])
        self.assertIsNone(parsed[3]["value"])
        self.assertNotEqual(parsed[3]["value"], 0)
        self.assertEqual(parsed[3]["unavailable_reason"],
                         "no PDH counter for this engine")

    def test_errors_log_carries_each_error(self):
        record = _record(errors=("stream 3 reconnect: timeout",
                                 "npu blob cache misses: 2"))
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(record, Path(tmp) / "bundle.zip")
            with zipfile.ZipFile(path) as archive:
                text = archive.read("errors.log").decode("utf-8")
        self.assertIn("stream 3 reconnect: timeout", text)
        self.assertIn("npu blob cache misses: 2", text)

    def test_routing_transitions_round_trip(self):
        record = _record()
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(record, Path(tmp) / "bundle.zip")
            with zipfile.ZipFile(path) as archive:
                payload = json.loads(archive.read("routing_transitions.json"))
        self.assertEqual(payload, [dict(t) for t in record.routing_transitions])

    def test_manifest_carries_the_fingerprint_hash_and_environment(self):
        record = _record()
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(record, Path(tmp) / "bundle.zip")
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(manifest["fingerprint_hash"], record.fingerprint_hash)
        self.assertEqual(manifest["fingerprint"], record.fingerprint.to_json())
        self.assertEqual(manifest["environment"]["machine_id"], "booth-01")
        self.assertEqual(manifest["mode"], "live")
        self.assertFalse(manifest["included_imagery"])
        self.assertEqual(manifest["sample_count"], len(record.samples))

    def test_a_destination_without_a_zip_suffix_gets_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(_record(), Path(tmp) / "bundle")
            self.assertEqual(path.name, "bundle.zip")
            self.assertTrue(path.is_file())

    def test_exporting_something_that_is_not_a_record_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                export_bundle({}, Path(tmp) / "bundle.zip")  # type: ignore[arg-type]


class ARecordedExportCannotLookLive(unittest.TestCase):
    def test_a_recorded_record_exports_with_the_recorded_notice(self):
        record = _record(mode="recorded")
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(record, Path(tmp) / "bundle.zip")
            with zipfile.ZipFile(path) as archive:
                readme = archive.read("README.txt").decode("utf-8")
                manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(manifest["mode"], "recorded")
        self.assertFalse(manifest["comparison_eligible"])
        self.assertIn("Recorded demonstration - metrics are not live", readme)
        self.assertNotIn("These metrics were measured live", readme)


class TheSummaryIsDerivedFromSettledSamples(unittest.TestCase):
    def test_summary_reports_a_median_and_a_range_not_just_a_median(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(_record(), Path(tmp) / "bundle.zip")
            with zipfile.ZipFile(path) as archive:
                summary = json.loads(archive.read("metrics_summary.json"))
        entry = summary["metrics"]["inference_ms"]
        settled = entry["settled"]
        self.assertEqual(settled["count"], 2)
        self.assertEqual(settled["median"], statistics.median([11.0, 12.0]))
        self.assertEqual(sorted(settled["values"]), [11.0, 12.0])
        self.assertEqual(settled["min"], 11.0)
        self.assertEqual(settled["max"], 12.0)
        self.assertEqual(entry["cold"]["count"], 1)
        self.assertEqual(entry["cold"]["values"], [5.2])

    def test_an_unavailable_metric_is_null_with_a_reason_never_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(_record(), Path(tmp) / "bundle.zip")
            with zipfile.ZipFile(path) as archive:
                summary = json.loads(archive.read("metrics_summary.json"))
        entry = summary["metrics"]["npu_utilization_percent"]
        self.assertIsNone(entry["settled"]["median"])
        self.assertIsNone(entry["settled"]["min"])
        self.assertIsNone(entry["settled"]["max"])
        self.assertIsNotNone(entry["unavailable_reason"])
        self.assertNotEqual(entry["settled"]["median"], 0)

    def test_the_cold_sample_is_not_reported_as_the_steady_state_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(_record(), Path(tmp) / "bundle.zip")
            with zipfile.ZipFile(path) as archive:
                summary = json.loads(archive.read("metrics_summary.json"))
        settled = summary["metrics"]["inference_ms"]["settled"]
        self.assertNotEqual(settled["median"], 5.2)
        self.assertNotIn(5.2, settled["values"])

    def test_manifest_and_summary_agree_on_the_settled_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export_bundle(_record(), Path(tmp) / "bundle.zip")
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                summary = json.loads(archive.read("metrics_summary.json"))
        self.assertEqual(manifest["warmup_s"], 5.0)
        self.assertEqual(summary["warmup_s"], 5.0)


class TheBundleIsReproducible(unittest.TestCase):
    def test_two_exports_have_the_same_contents(self):
        record = _record()
        with tempfile.TemporaryDirectory() as tmp:
            first = export_bundle(record, Path(tmp) / "first.zip")
            second = export_bundle(record, Path(tmp) / "second.zip")
            with zipfile.ZipFile(first) as left, zipfile.ZipFile(second) as right:
                self.assertEqual(set(left.namelist()), set(right.namelist()))
                for name in left.namelist():
                    self.assertEqual(left.read(name), right.read(name),
                                     f"{name} differed between two exports")

    def test_two_exports_are_byte_identical_because_timestamps_are_fixed(self):
        record = _record()
        with tempfile.TemporaryDirectory() as tmp:
            first = export_bundle(record, Path(tmp) / "first.zip")
            second = export_bundle(record, Path(tmp) / "second.zip")
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
