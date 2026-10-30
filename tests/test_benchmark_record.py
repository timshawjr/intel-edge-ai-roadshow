"""`BenchmarkRecord`: the unit of evidence, and the fingerprint that can refuse it.

The failure these tests exist to prevent is a record that *looks* reproducible. A
fingerprint that hashes only `model_id` is a plausible-looking object: it is stable,
it is a hash, and it changes when the model changes. It also survives a changed
threshold, a changed stream count and a changed device layout, which is exactly how two
records measured under different conditions get compared as though they were one.

So the fingerprint is tested **field by field**: for every covered field, a fingerprint
that differs only in that field must report `differing_fields() == (field,)` and a
different `fingerprint_hash()`. A model-id-only implementation fails that loop on ten of
its eleven entries, and the test names them.

The other structural claims:

- a `BenchmarkRecord` cannot be constructed without a `WorkloadFingerprint`; there is no
  warning path, only a `TypeError`;
- a `recorded` record cannot be `comparison_eligible` - the recording is not live
  evidence, and no field can make it say that it is;
- an unavailable metric is `MetricSample.value is None` **with a reason**, and the
  record's `unavailable_reasons` has to agree with its own samples, so `0` cannot stand
  in for "not measured";
- a sample inside the warm-up window cannot be labelled `SETTLED`, because the NPU's
  5.2-5.9 ms cold figure against 3.96-4.01 ms sustained is exactly the comparison that
  goes wrong when a ramp-up sample is reported as a steady-state one.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace

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

_HEX_A = "a" * 64
_HEX_B = "b" * 64


def _fingerprint(**overrides) -> WorkloadFingerprint:
    base = dict(
        model_id="yolov8s-int8",
        model_sha256=_HEX_A,
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


def _sample(metric="inference_ms", value=10.0, *, at=10.0,
            phase=SamplePhase.SETTLED, unit="ms", provider="worker",
            device_scope="GPU", reason=None) -> RecordedSample:
    if value is None:
        sample = MetricSample(
            name=metric, value=None, unit=unit, provider=provider,
            device_scope=device_scope, process_scope=("worker",),
            window_seconds=0.0, sampled_at=at, validity=Validity.UNAVAILABLE,
            unavailable_reason=reason or "counter unreadable",
        )
    else:
        sample = MetricSample(
            name=metric, value=value, unit=unit, provider=provider,
            device_scope=device_scope, process_scope=("worker",),
            window_seconds=1.0, sampled_at=at, validity=Validity.VALID,
        )
    return RecordedSample(sample=sample, phase=phase)


_UNSET = object()


def _record(*, samples=None, fingerprint=_UNSET, environment=_UNSET, mode="live",
            comparison_eligible=True, unavailable_reasons=None, warmup_s=5.0,
            duration_s=60.0, started_at="2026-10-01T10:00:00",
            trial_summary=None, exclusions=(), errors=(), routing_transitions=(),
            imagery=()):
    if fingerprint is _UNSET:
        fingerprint = _fingerprint()
    if environment is _UNSET:
        environment = _environment()
    if samples is None:
        samples = (_sample("latency_ms", 11.0, at=10.0),
                   _sample("latency_ms", 12.0, at=11.0))
    if unavailable_reasons is None:
        unavailable_reasons = {
            s.metric: s.sample.unavailable_reason
            for s in samples if s.value is None
        }
    if trial_summary is None:
        trial_summary = (
            TrialSummary(condition="sustained", placement=("GPU",),
                         settled_after_s=warmup_s),
        )
    return BenchmarkRecord(
        mode=mode,
        started_at=started_at,
        duration_s=duration_s,
        fingerprint=fingerprint,
        environment=environment,
        samples=samples,
        trial_summary=trial_summary,
        exclusions=exclusions,
        comparison_eligible=comparison_eligible,
        unavailable_reasons=unavailable_reasons,
        warmup_s=warmup_s,
        routing_transitions=routing_transitions,
        errors=errors,
        imagery=imagery,
    )


class ARecordNeedsAFingerprint(unittest.TestCase):
    """Structural, not a warning: no fingerprint means no record at all."""

    def test_a_record_without_a_fingerprint_cannot_be_constructed(self):
        with self.assertRaises(TypeError):
            _record(fingerprint=None)

    def test_a_plain_mapping_is_not_a_fingerprint(self):
        with self.assertRaises(TypeError):
            _record(fingerprint={"model_id": "yolov8s-int8"})

    def test_a_record_without_an_environment_cannot_be_constructed(self):
        with self.assertRaises(TypeError):
            _record(environment=None)


class TheFingerprintIsFieldByField(unittest.TestCase):
    """A model-id-only hash passes a naive test; this class is not naive."""

    ALTERNATIVES = {
        "model_id": "yolov8m-int8",
        "model_sha256": _HEX_B,
        "input_shape": (1, 3, 320, 320),
        "conf_threshold": 0.30,
        "iou_threshold": 0.50,
        "stream_count": 2,
        "target_fps": 30.0,
        "device_layout": ("GPU",),
        "decode_path": "opencv",
        "openvino_version": "2026.3.0-1-releases/2026/3",
        "numpy_version": "2.4.0",
    }

    def test_every_covered_field_is_covered_by_a_test(self):
        # Guards the loop below against a field being added to the dataclass and
        # silently escaping the per-field coverage check.
        self.assertEqual(set(self.ALTERNATIVES), set(WorkloadFingerprint.__dataclass_fields__))

    def test_hash_is_sha256_hex(self):
        value = _fingerprint().fingerprint_hash()
        self.assertRegex(value, r"^[0-9a-f]{64}$")

    def test_equal_fingerprints_have_equal_hashes(self):
        self.assertEqual(_fingerprint().fingerprint_hash(),
                         _fingerprint().fingerprint_hash())

    def test_a_changed_model_hash_differs_and_names_the_field(self):
        # The mutation a model-id-only hash survives: same model, different artifact.
        # An INT8 export and a re-quantised export share a model id and not a number.
        left = _fingerprint()
        right = _fingerprint(model_sha256=_HEX_B)
        self.assertEqual(left.model_id, right.model_id)
        self.assertEqual(left.differing_fields(right), ("model_sha256",))
        self.assertNotEqual(left.fingerprint_hash(), right.fingerprint_hash())

    def test_a_model_id_only_hash_would_not_be_enough(self):
        fingerprint = _fingerprint()
        model_id_only = hashlib.sha256(
            json.dumps({"model_id": fingerprint.model_id}).encode("utf-8")
        ).hexdigest()
        self.assertNotEqual(fingerprint.fingerprint_hash(), model_id_only)

    def test_a_changed_conf_threshold_differs_and_names_the_field(self):
        left = _fingerprint()
        right = _fingerprint(conf_threshold=0.30)
        self.assertEqual(left.differing_fields(right), ("conf_threshold",))
        self.assertNotEqual(left.fingerprint_hash(), right.fingerprint_hash())

    def test_a_changed_stream_count_differs_and_names_the_field(self):
        left = _fingerprint()
        right = _fingerprint(stream_count=2)
        self.assertEqual(left.differing_fields(right), ("stream_count",))
        self.assertNotEqual(left.fingerprint_hash(), right.fingerprint_hash())

    def test_a_changed_device_layout_differs_and_names_the_field(self):
        left = _fingerprint()
        right = _fingerprint(device_layout=("NPU", "GPU"))
        self.assertEqual(left.differing_fields(right), ("device_layout",))
        self.assertNotEqual(left.fingerprint_hash(), right.fingerprint_hash())

    def test_every_covered_field_changes_the_hash_and_is_named(self):
        base = _fingerprint()
        for name, alternative in self.ALTERNATIVES.items():
            with self.subTest(field=name):
                changed = replace(base, **{name: alternative})
                self.assertEqual(
                    base.differing_fields(changed), (name,),
                    f"{name} did not name itself as the only difference")
                self.assertNotEqual(
                    base.fingerprint_hash(), changed.fingerprint_hash(),
                    f"the fingerprint hash ignored {name}; a model-id-only hash is "
                    f"exactly the failure this test exists to catch")
                self.assertEqual(changed.differing_fields(base), (name,))
                self.assertEqual(base.differing_fields(_fingerprint()), ())

    def test_fingerprint_json_round_trips(self):
        base = _fingerprint()
        self.assertEqual(WorkloadFingerprint.from_json(base.to_json()), base)
        self.assertEqual(
            WorkloadFingerprint.from_json(base.to_json()).fingerprint_hash(),
            base.fingerprint_hash())

    def test_an_uppercase_or_short_hash_is_refused(self):
        with self.assertRaises(ValueError):
            _fingerprint(model_sha256="A" * 64)
        with self.assertRaises(ValueError):
            _fingerprint(model_sha256="abc")

    def test_an_unresolved_layout_is_refused(self):
        for layout in ((), ("AUTO",), ("HETERO",), ("",)):
            with self.subTest(layout=layout):
                with self.assertRaises(ValueError):
                    _fingerprint(device_layout=layout)

    def test_a_threshold_outside_the_unit_interval_is_refused(self):
        with self.assertRaises(ValueError):
            _fingerprint(conf_threshold=1.5)
        with self.assertRaises(ValueError):
            _fingerprint(iou_threshold=-0.1)

    def test_a_non_positive_stream_count_or_rate_is_refused(self):
        with self.assertRaises(ValueError):
            _fingerprint(stream_count=0)
        with self.assertRaises(ValueError):
            _fingerprint(target_fps=0.0)


class ARecordedRunCannotClaimToBeEligible(unittest.TestCase):
    """A playback is evidence of what was recorded, never of what the machine does now."""

    def test_a_recorded_record_cannot_be_comparison_eligible(self):
        with self.assertRaises(ValueError) as caught:
            _record(mode="recorded", comparison_eligible=True)
        self.assertIn("recorded", str(caught.exception).lower())
        self.assertIn("comparison_eligible", str(caught.exception))

    def test_a_recorded_record_with_eligible_false_is_fine(self):
        record = _record(mode="recorded", comparison_eligible=False)
        self.assertEqual(record.mode, "recorded")
        self.assertFalse(record.comparison_eligible)

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            _record(mode="replay", comparison_eligible=False)

    def test_a_live_eligible_record_needs_a_settled_measurement(self):
        cold_only = (_sample("latency_ms", 11.0, at=1.0, phase=SamplePhase.COLD),)
        with self.assertRaises(ValueError):
            _record(samples=cold_only, comparison_eligible=True)
        record = _record(samples=cold_only, comparison_eligible=False)
        self.assertFalse(record.comparison_eligible)


class UnavailableIsNoneWithAReasonNeverZero(unittest.TestCase):
    def test_an_unavailable_metric_is_none_and_carries_the_reason(self):
        record = _record(
            samples=(
                _sample("inference_ms", 4.0, at=10.0),
                _sample("npu_utilization_percent", None, at=10.0,
                        unit="percent", device_scope="NPU",
                        reason="no PDH counter for this engine"),
            ),
            unavailable_reasons={
                "npu_utilization_percent": "no PDH counter for this engine"},
        )
        sample = record.samples[1]
        self.assertIsNone(sample.value)
        self.assertNotEqual(sample.value, 0)
        self.assertEqual(sample.sample.unavailable_reason,
                         "no PDH counter for this engine")
        self.assertIsNone(
            json.loads(json.dumps(record.to_json()))["samples"][1]["value"])

    def test_a_zero_cannot_be_carried_as_unavailable(self):
        # The lie one layer in: a number beside "unavailable" renders as a reading.
        with self.assertRaises(ValueError):
            MetricSample(
                name="npu_utilization_percent", value=0.0, unit="percent",
                provider="pdh", device_scope="NPU", process_scope=("ui",),
                window_seconds=0.0, sampled_at=10.0,
                validity=Validity.UNAVAILABLE, unavailable_reason="no counter")

    def test_an_unavailable_sample_without_a_record_reason_is_refused(self):
        with self.assertRaises(ValueError):
            _record(
                samples=(_sample("npu_utilization_percent", None, at=10.0,
                                 unit="percent", reason="no counter"),),
                unavailable_reasons={},
                comparison_eligible=False,
            )

    def test_a_metric_cannot_be_both_measured_and_unavailable(self):
        with self.assertRaises(ValueError):
            _record(
                samples=(_sample("latency_ms", 11.0, at=10.0),),
                unavailable_reasons={"latency_ms": "not measured"},
            )

    def test_a_blank_reason_is_refused(self):
        with self.assertRaises(ValueError):
            _record(
                samples=(_sample("npu_utilization_percent", None, at=10.0,
                                 unit="percent", reason="no counter"),),
                unavailable_reasons={"npu_utilization_percent": "   "},
                comparison_eligible=False,
            )


class ColdIsNotSettled(unittest.TestCase):
    """The NPU's 5.2-5.9 ms is a cold figure; 3.96-4.01 ms is sustained."""

    def test_a_warmup_sample_cannot_be_labelled_settled(self):
        with self.assertRaises(ValueError) as caught:
            _record(samples=(_sample("inference_ms", 5.2, at=1.0,
                                     phase=SamplePhase.SETTLED),),
                    warmup_s=5.0)
        self.assertIn("settled", str(caught.exception).lower())
        self.assertIn("warm-up", str(caught.exception).lower().replace("warmup", "warm-up"))

    def test_a_warmup_sample_labelled_cold_is_accepted(self):
        record = _record(samples=(_sample("inference_ms", 5.2, at=1.0,
                                          phase=SamplePhase.COLD),
                                  _sample("inference_ms", 4.0, at=10.0)),
                         warmup_s=5.0)
        self.assertEqual([s.phase for s in record.samples],
                         [SamplePhase.COLD, SamplePhase.SETTLED])

    def test_a_sample_after_the_run_ended_is_refused(self):
        with self.assertRaises(ValueError):
            _record(samples=(_sample("inference_ms", 4.0, at=61.0),),
                    duration_s=60.0, warmup_s=5.0)

    def test_a_warmup_longer_than_the_run_is_refused(self):
        # A cold sample, so the phase rule cannot be what raises: only the
        # warm-up-versus-duration guard is left to refuse this record.
        with self.assertRaises(ValueError) as caught:
            _record(samples=(_sample("inference_ms", 5.2, at=1.0,
                                     phase=SamplePhase.COLD),),
                    warmup_s=61.0, duration_s=60.0)
        self.assertIn("warmup_s", str(caught.exception))

    def test_a_trial_summary_must_carry_its_execution_devices(self):
        with self.assertRaises(ValueError):
            TrialSummary(condition="sustained", placement=(), settled_after_s=5.0)

    def test_a_trial_summary_cannot_claim_auto_or_hetero(self):
        with self.assertRaises(ValueError):
            TrialSummary(condition="sustained", placement=("AUTO",),
                         settled_after_s=5.0)


class TheRecordRoundTrips(unittest.TestCase):
    def test_json_round_trip_preserves_samples_phases_and_none(self):
        record = _record(
            samples=(
                _sample("inference_ms", 5.2, at=1.0, phase=SamplePhase.COLD),
                _sample("inference_ms", 4.0, at=10.0),
                _sample("npu_utilization_percent", None, at=11.0,
                        unit="percent", device_scope="NPU",
                        reason="no counter"),
            ),
            unavailable_reasons={"npu_utilization_percent": "no counter"},
            exclusions=("first 5 s are warm-up",),
            errors=("stream 3 reconnect: timeout",),
            routing_transitions=({"at_s": 2.0, "from": "NPU", "to": "GPU"},),
            trial_summary=(
                TrialSummary(condition="sustained", placement=("GPU",),
                             settled_after_s=5.0),),
        )
        payload = json.loads(json.dumps(record.to_json()))
        rebuilt = BenchmarkRecord.from_json(payload)
        self.assertEqual(rebuilt, record)
        self.assertEqual(rebuilt.fingerprint_hash, record.fingerprint_hash)
        self.assertEqual(rebuilt.samples[0].phase, SamplePhase.COLD)
        self.assertIsNone(rebuilt.samples[2].value)

    def test_a_tampered_fingerprint_hash_is_refused_on_load(self):
        payload = _record().to_json()
        payload["fingerprint_hash"] = "0" * 64
        with self.assertRaises(ValueError) as caught:
            BenchmarkRecord.from_json(payload)
        self.assertIn("fingerprint", str(caught.exception).lower())

    def test_environment_round_trips(self):
        env = _environment()
        self.assertEqual(EnvironmentManifest.from_json(env.to_json()), env)

    def test_an_empty_environment_field_is_refused(self):
        with self.assertRaises(ValueError):
            _environment(machine_id="")

    def test_a_non_positive_ram_figure_is_refused(self):
        with self.assertRaises(ValueError):
            _environment(ram_bytes=0)

    def test_the_record_is_frozen(self):
        record = _record()
        with self.assertRaises(Exception):
            record.mode = "recorded"  # type: ignore[misc]

    def test_an_empty_record_is_refused(self):
        with self.assertRaises(ValueError):
            _record(samples=(), comparison_eligible=False)

    def test_imagery_cannot_escape_the_bundle_by_name(self):
        for name in ("../evil.png", "frames/../evil.png", "C:\\evil.png"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    ImageryFrame(name=name, media_type="image/png",
                                 data=b"\x89PNG\r\n\x1a\n", captured_at=1.0)


if __name__ == "__main__":
    unittest.main()
