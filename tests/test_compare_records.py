"""The comparison runner: refuse mismatched conditions, and compare ranges, not vibes.

The failures these tests exist to prevent:

1. **Comparing across mismatched conditions.** A changed threshold, stream count, model
   artifact hash or device layout is refused, and the refusal **names the fields**. A
   comparison is not rebased onto the new setting: `per_metric` is empty, so there is no
   delta anyone could quote as though the baseline still applied.
2. **Reporting a median alone.** Every `MetricComparison` carries a median **and** the
   full min/max range for both records, because the spread is what tells a reader whether
   a delta is meaningful.
3. **A ramp-up sample reported as steady state.** Only `SETTLED` samples are used. The
   cold 5.2 ms in the baseline is counted and stored, but it never reaches the median or
   the range.
4. **`None` becoming `0`.** A metric missing from one side is `None` with a reason, and
   its delta is `None`, never `0`.
5. **An older record unlabelled.** Whichever record is older is called `previous run` and
   is displayed with its machine and timestamp.

`within_noise` is range overlap: if the two observed ranges intersect, the difference is
inside the variation the runs themselves showed. It is `None` when either side has no
settled measurement, because "no measurement" is not "within noise".
"""

from __future__ import annotations

import contextlib
import io
import json
import statistics
import tempfile
import unittest
from pathlib import Path

from app.contracts.benchmark_record import (
    BenchmarkRecord,
    EnvironmentManifest,
    RecordedSample,
    SamplePhase,
    TrialSummary,
    WorkloadFingerprint,
)
from app.contracts.device_status import Validity
from app.contracts.metric_sample import MetricSample
from tools.compare_records import compare, main


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


def _latency(values, *, start_at=10.0, extra=()):
    samples = [_sample("latency_ms", value, at=start_at + index,
                       phase=SamplePhase.SETTLED)
               for index, value in enumerate(values)]
    return tuple(samples) + tuple(extra)


def _record(*, samples, fingerprint=None, environment=None, started_at="2026-10-01T10:00:00",
            eligible=True, mode="live", warmup_s=5.0):
    unavailable = {
        s.metric: s.sample.unavailable_reason
        for s in samples if s.value is None
    }
    return BenchmarkRecord(
        mode=mode,
        started_at=started_at,
        duration_s=60.0,
        fingerprint=fingerprint or _fingerprint(),
        environment=environment or _environment(),
        samples=tuple(samples),
        trial_summary=(TrialSummary(condition="sustained", placement=("GPU",),
                                    settled_after_s=warmup_s),),
        exclusions=(),
        comparison_eligible=eligible,
        unavailable_reasons=unavailable,
        warmup_s=warmup_s,
    )


def _baseline(**overrides):
    overrides.setdefault("samples", _latency([10.0, 11.0, 12.0]))
    return _record(**overrides)


def _candidate(**overrides):
    overrides.setdefault("samples", _latency([12.0, 13.0, 14.0]))
    return _record(**overrides)


class MismatchedConditionsAreRefused(unittest.TestCase):
    def test_a_changed_threshold_and_stream_count_are_named(self):
        baseline = _baseline()
        candidate = _candidate(fingerprint=_fingerprint(conf_threshold=0.30,
                                                        stream_count=2))
        comparison = compare(baseline, candidate)
        self.assertFalse(comparison.eligible)
        self.assertEqual(comparison.differing_fields,
                         ("conf_threshold", "stream_count"))
        self.assertIn("conf_threshold", comparison.refusal)
        self.assertIn("stream_count", comparison.refusal)
        self.assertEqual(comparison.baseline_status, "invalidated")

    def test_the_model_hash_is_named_even_when_the_model_id_is_the_same(self):
        baseline = _baseline()
        candidate = _candidate(fingerprint=_fingerprint(model_sha256="b" * 64))
        comparison = compare(baseline, candidate)
        self.assertFalse(comparison.eligible)
        self.assertEqual(comparison.differing_fields, ("model_sha256",))
        self.assertIn("model_sha256", comparison.refusal)

    def test_a_changed_device_layout_invalidates_the_baseline_without_rebasing(self):
        baseline = _baseline()
        candidate = _candidate(fingerprint=_fingerprint(device_layout=("GPU",)))
        comparison = compare(baseline, candidate)
        self.assertFalse(comparison.eligible)
        self.assertEqual(comparison.differing_fields, ("device_layout",))
        # A refusal, not a rebase: there is no delta to quote.
        self.assertEqual(dict(comparison.per_metric), {})
        self.assertEqual(comparison.baseline_status, "invalidated")

    def test_a_changed_machine_condition_is_refused_and_named(self):
        baseline = _baseline()
        candidate = _candidate(environment=_environment(gpu_driver="32.0.101.9999"))
        comparison = compare(baseline, candidate)
        self.assertFalse(comparison.eligible)
        self.assertEqual(comparison.differing_fields,
                         ("environment.gpu_driver",))
        self.assertIn("environment.gpu_driver", comparison.refusal)

    def test_a_recorded_record_cannot_be_compared_to_a_live_baseline(self):
        baseline = _baseline()
        recorded = _candidate(mode="recorded", eligible=False)
        comparison = compare(baseline, recorded)
        self.assertFalse(comparison.eligible)
        self.assertIn("mode", comparison.differing_fields)
        self.assertIn("recorded", comparison.refusal.lower())
        self.assertEqual(dict(comparison.per_metric), {})

    def test_a_record_that_says_it_is_not_eligible_is_refused(self):
        baseline = _baseline()
        candidate = _candidate(eligible=False)
        comparison = compare(baseline, candidate)
        self.assertFalse(comparison.eligible)
        self.assertIn("comparison_eligible", comparison.differing_fields)
        self.assertEqual(dict(comparison.per_metric), {})

    def test_mapping_payloads_are_accepted_and_compared(self):
        baseline = _baseline()
        candidate = _candidate()
        comparison = compare(baseline.to_json(), candidate.to_json())
        self.assertTrue(comparison.eligible)
        self.assertEqual(comparison.per_metric["latency_ms"].median["baseline"], 11.0)

    def test_comparing_a_non_record_is_refused(self):
        with self.assertRaises(TypeError):
            compare({}, _candidate())
        with self.assertRaises(TypeError):
            compare(_baseline(), None)


class AMatchedPairReportsMediansAndRanges(unittest.TestCase):
    def test_both_medians_and_both_ranges_are_reported(self):
        comparison = compare(_baseline(), _candidate())
        self.assertTrue(comparison.eligible)
        self.assertEqual(comparison.differing_fields, ())
        self.assertIsNone(comparison.refusal)
        self.assertEqual(comparison.baseline_status, "valid")
        metric = comparison.per_metric["latency_ms"]
        self.assertEqual(metric.median["baseline"], statistics.median([10.0, 11.0, 12.0]))
        self.assertEqual(metric.median["candidate"], statistics.median([12.0, 13.0, 14.0]))
        self.assertEqual(metric.range["baseline"], (10.0, 12.0))
        self.assertEqual(metric.range["candidate"], (12.0, 14.0))
        self.assertEqual(metric.delta, 2.0)
        self.assertEqual(metric.sample_counts["baseline"], 3)
        self.assertEqual(metric.sample_counts["candidate"], 3)

    def test_ranges_drive_within_noise_not_medians_alone(self):
        overlapping = compare(_baseline(), _candidate())
        self.assertTrue(overlapping.per_metric["latency_ms"].within_noise)

        disjoint = compare(_baseline(), _candidate(samples=_latency([20.0, 21.0, 22.0])))
        self.assertFalse(disjoint.per_metric["latency_ms"].within_noise)
        self.assertEqual(disjoint.per_metric["latency_ms"].range["candidate"],
                         (20.0, 22.0))

    def test_a_single_sample_gives_a_median_and_a_degenerate_range(self):
        comparison = compare(_baseline(samples=_latency([10.0])),
                             _candidate(samples=_latency([12.0])))
        metric = comparison.per_metric["latency_ms"]
        self.assertEqual(metric.median["baseline"], 10.0)
        self.assertEqual(metric.range["baseline"], (10.0, 10.0))


class AColdSampleIsNeverTheSteadyStateFigure(unittest.TestCase):
    def test_cold_samples_are_counted_but_never_compared(self):
        baseline = _baseline(samples=_latency(
            [10.0, 11.0, 12.0],
            extra=(_sample("latency_ms", 5.2, at=1.0, phase=SamplePhase.COLD),)))
        candidate = _candidate(samples=_latency(
            [12.0, 13.0, 14.0],
            extra=(_sample("latency_ms", 5.9, at=1.0, phase=SamplePhase.COLD),)))
        metric = compare(baseline, candidate).per_metric["latency_ms"]
        self.assertEqual(metric.median["baseline"], 11.0)
        self.assertEqual(metric.median["candidate"], 13.0)
        self.assertEqual(metric.sample_counts["baseline"], 3)
        self.assertEqual(metric.sample_counts["candidate"], 3)
        self.assertNotIn(5.2, metric.range["baseline"])

    def test_a_record_with_only_cold_samples_has_no_measurement_to_compare(self):
        baseline = _baseline(samples=(
            _sample("latency_ms", 5.2, at=1.0, phase=SamplePhase.COLD),),
            eligible=False)
        candidate = _candidate(samples=_latency([12.0, 13.0, 14.0]))
        comparison = compare(baseline, candidate)
        # The baseline is not eligible, so the comparison is refused outright.
        self.assertFalse(comparison.eligible)
        self.assertEqual(dict(comparison.per_metric), {})


class AnOlderRecordIsLabelledPreviousRun(unittest.TestCase):
    def test_an_older_baseline_is_previous_run_with_machine_and_timestamp(self):
        baseline = _baseline(started_at="2026-10-01T09:00:00")
        candidate = _candidate(started_at="2026-10-01T11:00:00")
        comparison = compare(baseline, candidate)
        self.assertEqual(comparison.baseline_label, "previous run")
        self.assertEqual(comparison.candidate_label, "candidate")
        self.assertIn("previous run", comparison.baseline_display)
        self.assertIn("booth-01", comparison.baseline_display)
        self.assertIn("2026-10-01T09:00:00", comparison.baseline_display)

    def test_an_older_candidate_is_previous_run_with_machine_and_timestamp(self):
        baseline = _candidate(started_at="2026-10-01T11:00:00")
        candidate = _baseline(started_at="2026-10-01T09:00:00")
        comparison = compare(baseline, candidate)
        self.assertEqual(comparison.baseline_label, "baseline")
        self.assertEqual(comparison.candidate_label, "previous run")
        self.assertIn("previous run", comparison.candidate_display)
        self.assertIn("booth-01", comparison.candidate_display)
        self.assertIn("2026-10-01T09:00:00", comparison.candidate_display)


class AMissingMetricIsNoneNeverZero(unittest.TestCase):
    def test_a_metric_only_in_the_baseline_is_null_on_the_candidate_side(self):
        baseline = _baseline(samples=_latency(
            [10.0, 11.0, 12.0],
            extra=(_sample("throughput_fps", 300.0, at=10.0, unit="fps"),)))
        candidate = _candidate()
        metric = compare(baseline, candidate).per_metric["throughput_fps"]
        self.assertIsNone(metric.median["candidate"])
        self.assertIsNone(metric.range["candidate"])
        self.assertIsNone(metric.delta)
        self.assertIsNone(metric.within_noise)
        self.assertIsNotNone(metric.unavailable["candidate"])
        self.assertIn("candidate", metric.unavailable["candidate"])
        self.assertNotEqual(metric.median["candidate"], 0)

    def test_an_unavailable_sample_carries_its_reason_into_the_comparison(self):
        baseline = _baseline()
        candidate = _candidate(samples=_latency(
            [12.0, 13.0, 14.0],
            extra=(_sample("npu_utilization_percent", None, at=10.0, unit="percent",
                           device_scope="NPU", reason="no PDH counter"),)))
        metric = compare(baseline, candidate).per_metric["npu_utilization_percent"]
        self.assertIsNone(metric.median["baseline"])
        self.assertIsNone(metric.median["candidate"])
        self.assertEqual(metric.unavailable["candidate"], "no PDH counter")
        self.assertIsNone(metric.delta)

    def test_a_genuine_zero_is_compared_as_zero(self):
        baseline = _baseline(samples=_latency([0.0, 0.0, 0.0]))
        candidate = _candidate(samples=_latency([0.0, 0.0, 0.0]))
        metric = compare(baseline, candidate).per_metric["latency_ms"]
        self.assertEqual(metric.median["baseline"], 0.0)
        self.assertIsNotNone(metric.median["baseline"])
        self.assertEqual(metric.delta, 0.0)
        self.assertTrue(metric.within_noise)


class TheCommandLine(unittest.TestCase):
    def _write(self, directory: Path, name: str, record: BenchmarkRecord) -> Path:
        path = directory / name
        path.write_text(json.dumps(record.to_json()), encoding="utf-8")
        return path

    def test_a_matched_pair_prints_medians_ranges_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            left = self._write(directory, "baseline.json", _baseline())
            right = self._write(directory, "candidate.json", _candidate())
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main([str(left), str(right)])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("eligible: yes", text)
        self.assertIn("11", text)
        self.assertIn("10", text)
        self.assertIn("12", text)
        self.assertIn("13", text)

    def test_a_refused_pair_prints_the_differing_fields_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            left = self._write(directory, "baseline.json", _baseline())
            right = self._write(
                directory, "candidate.json",
                _candidate(fingerprint=_fingerprint(conf_threshold=0.30)))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main([str(left), str(right)])
        self.assertEqual(code, 1)
        text = out.getvalue()
        self.assertIn("eligible: no", text)
        self.assertIn("conf_threshold", text)
        self.assertIn("previous run", text)


if __name__ == "__main__":
    unittest.main()
