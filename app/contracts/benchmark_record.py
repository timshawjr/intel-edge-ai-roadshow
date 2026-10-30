"""`BenchmarkRecord` - the unit of evidence, and the workload it is evidence *for*.

A number from a booth run is only useful if the conditions that produced it travel with
it. This module makes that a shape rather than a convention:

1. **A record cannot exist without a `WorkloadFingerprint`.** `fingerprint` is not
   optional and not a warning; `BenchmarkRecord(..., fingerprint=None)` raises. There is no
   path that produces evidence whose workload is unknown.
2. **The fingerprint covers every field that changes a number**: the model artifact
   **hash** (not just its id - two INT8 exports share an id and not a latency), the
   thresholds, the input shape, the stream count, the analysis rate, the device layout,
   the decode path, and the OpenVINO and NumPy versions. `fingerprint_hash()` is a SHA-256
   over all of them, and `differing_fields()` names every one that differs, so a
   comparison can be refused **by field**. A hash of `model_id` alone is the failure this
   module exists to prevent.
3. **A `recorded` record cannot be `comparison_eligible`.** A playback is evidence of
   what was recorded, never of what the machine does now. The refusal is enforced at
   construction, so no caller can set the flag and mean it.
4. **Unavailable is `None` with a reason.** Samples are `MetricSample`, which already
   makes `None` and `UNAVAILABLE` one fact and refuses a number beside `UNAVAILABLE`. The
   record additionally requires its `unavailable_reasons` mapping to agree with its own
   samples, so a metric cannot be reported as measured in one place and unavailable in
   another, and an unavailable metric can never be silently `0`.
5. **A label is not enough: a ramp-up sample cannot be relabelled steady-state.**
   Each sample carries `SamplePhase` (`COLD` or `SETTLED`), and `__post_init__` refuses a
   `SETTLED` sample whose `sampled_at` is inside the record's `warmup_s`. The NPU's
   published 5.2-5.9 ms is a cold figure against 3.96-4.01 ms sustained; this rule is what
   stops the first being compared as though it were the second.
6. **Every number traces to a sample.** `trial_summary` deliberately carries **no
   performance figure** - only the trial's identity, its `EXECUTION_DEVICES` placement
   (the proof of where it ran) and the warm-up boundary. Medians, ranges and deltas are
   derived by the comparison runner from the settled samples, never asserted here.

`sampled_at` in a `RecordedSample` is **seconds since the record started** (matching
`MetricSample`'s non-negative monotonic convention), which is what lets `warmup_s` and
`duration_s` constrain it. The exporter writes it verbatim; a reader can place it on the
record's timeline without guessing an epoch.

This module is pure: no clock, no device, no I/O. The telemetry producer supplies the
samples; the export and comparison tools consume the record.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from app.contracts.device_status import Validity
from app.contracts.metric_sample import MetricSample

__all__ = [
    "BenchmarkRecord",
    "EnvironmentManifest",
    "ImageryFrame",
    "RecordedSample",
    "SamplePhase",
    "TrialSummary",
    "WorkloadFingerprint",
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UNRESOLVED_DEVICES = frozenset({"AUTO", "HETERO", "MULTI"})


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string; got {value!r}. An empty value here is "
            f"indistinguishable from one nobody measured, and this record exists to tell "
            f"those apart."
        )
    return value


def _require_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number; got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return number


def _require_threshold(name: str, value: Any) -> float:
    number = _require_finite(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(
            f"{name} must be within [0, 1]; got {value!r}. Outside that range a threshold "
            f"suppresses everything or nothing, and both look like a working detector."
        )
    return number


def _require_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int; got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value!r}")
    return value


def _require_text_tuple(name: str, value: Any, *, non_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(
            f"{name} must be a sequence of strings; got {type(value).__name__}")
    items = tuple(_require_text(f"{name} entry", item) for item in value)
    if non_empty and not items:
        raise ValueError(
            f"{name} must not be empty. An empty list is the shape of a field nobody "
            f"filled in, and this record cannot carry one."
        )
    return items


def _canonical(value: Any) -> Any:
    """The JSON shape used for both hashing and equality of fingerprint fields."""
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        # `-0.0 == 0.0` is true but `json.dumps` spells them differently. A hash that
        # disagrees with `==` would refuse a comparison between equal workloads.
        return 0.0 if value == 0.0 else value
    return value


def _parse_timestamp(value: str) -> datetime:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"started_at must be an ISO-8601 timestamp; got {value!r} ({exc}). A record "
            f"whose start time cannot be ordered cannot be labelled 'previous run' "
            f"truthfully."
        ) from None


@dataclass(frozen=True)
class WorkloadFingerprint:
    """Everything about a run that changes one of its numbers.

    All eleven fields are covered by `fingerprint_hash()` and reported by
    `differing_fields()`. `model_sha256` is the **artifact** hash, so re-quantising a model
    without renaming it invalidates the baseline. `device_layout` is the resolved list of
    explicit execution devices, never `AUTO` or `HETERO`: an unresolved request does not
    identify what ran, and the fingerprint is a record of what ran.
    """

    model_id: str
    model_sha256: str
    input_shape: tuple[int, ...]
    conf_threshold: float
    iou_threshold: float
    stream_count: int
    target_fps: float
    device_layout: tuple[str, ...]
    decode_path: str
    openvino_version: str
    numpy_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _require_text("model_id", self.model_id))
        if not isinstance(self.model_sha256, str) or not _SHA256_RE.match(self.model_sha256):
            raise ValueError(
                f"model_sha256 must be 64 lowercase hex characters; got "
                f"{self.model_sha256!r}. A model id is not a fingerprint: two exports can "
                f"share an id and not a single number."
            )
        shape = self.input_shape
        if not isinstance(shape, (tuple, list)) or len(shape) != 4:
            raise ValueError(
                f"input_shape must be a 4-entry NCHW tuple; got {shape!r}. Static shapes "
                f"are required by the NPU, and a shape that is not carried is a shape a "
                f"comparison will get wrong."
            )
        normalised_shape = tuple(_require_int("input_shape entry", item)
                                 for item in shape)
        object.__setattr__(self, "input_shape", normalised_shape)
        object.__setattr__(self, "conf_threshold",
                           _require_threshold("conf_threshold", self.conf_threshold))
        object.__setattr__(self, "iou_threshold",
                           _require_threshold("iou_threshold", self.iou_threshold))
        object.__setattr__(self, "stream_count",
                           _require_int("stream_count", self.stream_count))
        fps = _require_finite("target_fps", self.target_fps)
        if fps <= 0.0:
            raise ValueError(f"target_fps must be > 0; got {self.target_fps!r}")
        object.__setattr__(self, "target_fps", fps)

        layout = _require_text_tuple("device_layout", self.device_layout)
        unresolved = sorted({d for d in layout if d.upper() in _UNRESOLVED_DEVICES})
        if unresolved:
            raise ValueError(
                f"device_layout must name the execution devices that ran, not "
                f"{unresolved}. AUTO/HETERO describe a request, and a request is not "
                f"proof of placement; EXECUTION_DEVICES is."
            )
        object.__setattr__(self, "device_layout", layout)
        object.__setattr__(self, "decode_path",
                           _require_text("decode_path", self.decode_path))
        object.__setattr__(self, "openvino_version",
                           _require_text("openvino_version", self.openvino_version))
        object.__setattr__(self, "numpy_version",
                           _require_text("numpy_version", self.numpy_version))

    def to_json(self) -> dict[str, Any]:
        """A `json.dumps`-acceptable payload; tuples become lists."""
        payload: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            payload[field.name] = _canonical(value)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "WorkloadFingerprint":
        known = {field.name for field in fields(cls)}
        missing = sorted(known - set(payload))
        if missing:
            raise ValueError(
                f"{cls.__name__}.from_json: missing required field(s) {missing}")
        kwargs = {name: payload[name] for name in known}
        return cls(**kwargs)

    def fingerprint_hash(self) -> str:
        """SHA-256 over **every** field, not over the model id.

        Canonical JSON with sorted keys and no whitespace, so equal fingerprints hash
        equally and a change anywhere changes the digest. The field-by-field tests in
        `tests/test_benchmark_record.py` mutate each field in turn; a hash that ignored
        one of them fails there by name.
        """
        canonical = json.dumps(
            self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def differing_fields(self, other: "WorkloadFingerprint") -> tuple[str, ...]:
        """The covered fields whose values differ, in declaration order.

        Empty means the two fingerprints are the same workload. The comparison runner
        refuses on a non-empty result and prints these names, rather than comparing two
        runs that were not measuring the same thing.
        """
        if not isinstance(other, WorkloadFingerprint):
            raise TypeError(
                f"differing_fields expects a WorkloadFingerprint; got "
                f"{type(other).__name__}")
        mine = self.to_json()
        theirs = other.to_json()
        return tuple(
            field.name for field in fields(self)
            if mine[field.name] != theirs[field.name]
        )


@dataclass(frozen=True)
class EnvironmentManifest:
    """The machine a record was measured on, as read from the machine.

    Every field is a measured string (or a measured byte count), never a default. The
    comparison runner treats a change here as a changed condition: a different driver or
    power plan can move a number without any workload field changing.
    """

    machine_id: str
    cpu: str
    gpu: str
    npu: str
    gpu_driver: str
    npu_driver: str
    windows_build: str
    ram_bytes: int
    power_plan: str
    ac_status: str
    monitor: str
    python_version: str

    def __post_init__(self) -> None:
        for name in ("machine_id", "cpu", "gpu", "npu", "gpu_driver", "npu_driver",
                     "windows_build", "power_plan", "ac_status", "monitor",
                     "python_version"):
            object.__setattr__(self, name, _require_text(name, getattr(self, name)))
        object.__setattr__(self, "ram_bytes",
                           _require_int("ram_bytes", self.ram_bytes, minimum=1))

    def to_json(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "EnvironmentManifest":
        known = {field.name for field in fields(cls)}
        missing = sorted(known - set(payload))
        if missing:
            raise ValueError(
                f"{cls.__name__}.from_json: missing required field(s) {missing}")
        return cls(**{name: payload[name] for name in known})

    def condition_fields(self) -> tuple[str, ...]:
        """The fields that can move a number when they change."""
        return ("machine_id", "cpu", "gpu", "npu", "gpu_driver", "npu_driver",
                "windows_build", "power_plan", "ac_status")


class SamplePhase(str, Enum):
    """Whether a sample is still ramping up or has settled.

    A `str` enum so it survives JSON as itself. `COLD` is not a lesser measurement - it
    is a different claim, and comparing it to a settled one is the mistake the NPU's
    published 5.2-5.9 ms against its sustained 3.96-4.01 ms demonstrates.
    """

    COLD = "cold"
    SETTLED = "settled"


@dataclass(frozen=True)
class RecordedSample:
    """One `MetricSample` plus the phase it belongs to.

    Composition, not inheritance: `MetricSample` already owns the rule that `None` and
    `UNAVAILABLE` are one fact, and re-declaring that rule here would create a second
    copy to drift. `metric` and `value` delegate, so consumers read naturally.
    """

    sample: MetricSample
    phase: SamplePhase

    def __post_init__(self) -> None:
        if not isinstance(self.sample, MetricSample):
            raise TypeError(
                f"sample must be a MetricSample; got {type(self.sample).__name__}")
        if not isinstance(self.phase, SamplePhase):
            raise TypeError(
                f"phase must be a SamplePhase; got {type(self.phase).__name__}. A free "
                f"string is how 'settled' and 'SETTLED' become two different states.")

    @property
    def metric(self) -> str:
        return self.sample.name

    @property
    def value(self) -> float | None:
        return self.sample.value

    @property
    def unit(self) -> str:
        return self.sample.unit

    def to_json(self) -> dict[str, Any]:
        payload = self.sample.to_json()
        payload["phase"] = self.phase.value
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "RecordedSample":
        data = dict(payload)
        if "phase" not in data:
            raise ValueError("RecordedSample.from_json: missing required field 'phase'")
        try:
            phase = SamplePhase(data.pop("phase"))
        except ValueError:
            raise ValueError(
                f"RecordedSample.from_json: unknown phase {payload.get('phase')!r}") from None
        known = {field.name for field in fields(MetricSample)}
        missing = sorted(known - set(data))
        if missing:
            raise ValueError(
                f"RecordedSample.from_json: missing sample field(s) {missing}")
        sample = MetricSample(
            name=data["name"],
            value=data["value"],
            unit=data["unit"],
            provider=data["provider"],
            device_scope=data["device_scope"],
            process_scope=tuple(data["process_scope"] or ()),
            window_seconds=data["window_seconds"],
            sampled_at=data["sampled_at"],
            validity=Validity(data["validity"]),
            unavailable_reason=data["unavailable_reason"],
        )
        return cls(sample=sample, phase=phase)


@dataclass(frozen=True)
class TrialSummary:
    """The identity of a trial; deliberately no performance figure.

    `placement` is the measured `EXECUTION_DEVICES` value, the only proof of where work
    ran. `settled_after_s` is the warm-up boundary this trial declared, not a measured
    latency. Any median or range a report prints is derived from `BenchmarkRecord.samples`
    by the comparison runner, so it can always be traced back to a sample.
    """

    condition: str
    placement: tuple[str, ...]
    settled_after_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition", _require_text("condition", self.condition))
        placement = _require_text_tuple("placement", self.placement)
        unresolved = sorted({d for d in placement if d.upper() in _UNRESOLVED_DEVICES})
        if unresolved:
            raise ValueError(
                f"placement must be the measured execution devices, not {unresolved}; "
                f"EXECUTION_DEVICES is the proof of placement.")
        object.__setattr__(self, "placement", placement)
        settled = _require_finite("settled_after_s", self.settled_after_s)
        if settled < 0.0:
            raise ValueError(f"settled_after_s must be >= 0; got {self.settled_after_s!r}")
        object.__setattr__(self, "settled_after_s", settled)

    def to_json(self) -> dict[str, Any]:
        return {"condition": self.condition, "placement": list(self.placement),
                "settled_after_s": self.settled_after_s}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "TrialSummary":
        return cls(condition=payload["condition"],
                   placement=tuple(payload["placement"]),
                   settled_after_s=payload["settled_after_s"])


@dataclass(frozen=True)
class ImageryFrame:
    """One camera frame held **in memory**, written only on explicit export opt-in.

    Never serialised into `manifest.json` and never written by `export_bundle` unless
    `include_imagery=True`. `name` is a single file name: it cannot contain a separator,
    so a frame cannot choose a path inside the ZIP.
    """

    name: str
    media_type: str
    data: bytes
    captured_at: float

    def __post_init__(self) -> None:
        name = _require_text("name", self.name)
        if name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(
                f"imagery name must be a single file name; got {name!r}. A frame that can "
                f"choose its own path is a frame that can escape the bundle.")
        object.__setattr__(self, "name", name)
        media_type = _require_text("media_type", self.media_type)
        if not media_type.startswith("image/"):
            raise ValueError(
                f"media_type must start with 'image/'; got {media_type!r}")
        object.__setattr__(self, "media_type", media_type)
        if not isinstance(self.data, (bytes, bytearray)) or not self.data:
            raise ValueError("imagery data must be non-empty bytes")
        object.__setattr__(self, "data", bytes(self.data))
        captured = _require_finite("captured_at", self.captured_at)
        if captured < 0.0:
            raise ValueError(f"captured_at must be >= 0; got {self.captured_at!r}")
        object.__setattr__(self, "captured_at", captured)


@dataclass(frozen=True)
class BenchmarkRecord:
    """One run of one workload, with the evidence needed to refuse a false comparison.

    `fingerprint` and `environment` are required and type-checked. `samples` are the
    record's only numbers: `trial_summary` carries identity and placement, and the export
    computes summaries from `samples`. `warmup_s` is the boundary that makes `COLD` and
    `SETTLED` checkable against `sampled_at`.

    `comparison_eligible` is the record's own claim that it may be compared. A `recorded`
    record **cannot** make it: mode is checked before the flag, so playback can never be
    presented as live evidence.
    """

    mode: str
    started_at: str
    duration_s: float
    fingerprint: WorkloadFingerprint
    environment: EnvironmentManifest
    samples: tuple[RecordedSample, ...]
    trial_summary: tuple[TrialSummary, ...]
    exclusions: tuple[str, ...]
    comparison_eligible: bool
    unavailable_reasons: Mapping[str, str]
    warmup_s: float = 0.0
    routing_transitions: tuple[Mapping[str, Any], ...] = ()
    errors: tuple[str, ...] = ()
    imagery: tuple[ImageryFrame, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("live", "recorded"):
            raise ValueError(
                f"mode must be 'live' or 'recorded'; got {self.mode!r}. A third spelling "
                f"is a third set of rules nobody has written yet.")
        object.__setattr__(self, "started_at", _require_text("started_at", self.started_at))
        _parse_timestamp(self.started_at)

        duration = _require_finite("duration_s", self.duration_s)
        if duration <= 0.0:
            raise ValueError(f"duration_s must be > 0; got {self.duration_s!r}")
        object.__setattr__(self, "duration_s", duration)
        warmup = _require_finite("warmup_s", self.warmup_s)
        if warmup < 0.0:
            raise ValueError(f"warmup_s must be >= 0; got {self.warmup_s!r}")
        if warmup >= duration:
            raise ValueError(
                f"warmup_s ({warmup}) must be shorter than duration_s ({duration}); a "
                f"record with no measured window carries no steady-state evidence.")
        object.__setattr__(self, "warmup_s", warmup)

        if not isinstance(self.fingerprint, WorkloadFingerprint):
            raise TypeError(
                f"a BenchmarkRecord requires a WorkloadFingerprint; got "
                f"{type(self.fingerprint).__name__}. A record without a fingerprint "
                f"cannot be reproduced and cannot be compared, so it cannot exist.")
        if not isinstance(self.environment, EnvironmentManifest):
            raise TypeError(
                f"a BenchmarkRecord requires an EnvironmentManifest; got "
                f"{type(self.environment).__name__}")

        samples = tuple(self.samples)
        if not samples:
            raise ValueError(
                "samples must not be empty. A record with no samples is a claim with no "
                "evidence, and every number in a record must trace to a sample.")
        for sample in samples:
            if not isinstance(sample, RecordedSample):
                raise TypeError(
                    f"samples must hold RecordedSample objects; got "
                    f"{type(sample).__name__}")
            at = sample.sample.sampled_at
            if at > duration:
                raise ValueError(
                    f"{sample.metric}: sampled_at {at} is after duration_s {duration}. A "
                    f"sample taken after the run is not a sample of the run.")
            if at < warmup and sample.phase is SamplePhase.SETTLED:
                raise ValueError(
                    f"{sample.metric}: a warm-up sample at {at}s (warm-up is {warmup}s) "
                    f"cannot be labelled settled. The NPU's cold 5.2-5.9 ms against its "
                    f"sustained 3.96-4.01 ms is exactly the comparison this refusal "
                    f"protects.")
        object.__setattr__(self, "samples", samples)

        trials = tuple(self.trial_summary)
        for trial in trials:
            if not isinstance(trial, TrialSummary):
                raise TypeError(
                    f"trial_summary must hold TrialSummary objects; got "
                    f"{type(trial).__name__}")
            if trial.settled_after_s > duration:
                raise ValueError(
                    f"{trial.condition}: settled_after_s {trial.settled_after_s} is after "
                    f"duration_s {duration}")
        object.__setattr__(self, "trial_summary", trials)

        exclusions = tuple(self.exclusions)
        for exclusion in exclusions:
            _require_text("exclusions entry", exclusion)
        object.__setattr__(self, "exclusions", exclusions)

        errors = tuple(self.errors)
        for error in errors:
            _require_text("errors entry", error)
        object.__setattr__(self, "errors", errors)

        transitions = tuple(self.routing_transitions)
        for transition in transitions:
            if not isinstance(transition, Mapping):
                raise TypeError(
                    f"routing_transitions must hold mappings; got "
                    f"{type(transition).__name__}")
        object.__setattr__(self, "routing_transitions",
                           tuple(MappingProxyType(dict(t)) for t in transitions))

        imagery = tuple(self.imagery)
        for frame in imagery:
            if not isinstance(frame, ImageryFrame):
                raise TypeError(
                    f"imagery must hold ImageryFrame objects; got "
                    f"{type(frame).__name__}")
        object.__setattr__(self, "imagery", imagery)

        reasons = self.unavailable_reasons
        if not isinstance(reasons, Mapping):
            raise TypeError(
                f"unavailable_reasons must be a mapping; got "
                f"{type(reasons).__name__}")
        normalised: dict[str, str] = {}
        for name, reason in reasons.items():
            normalised[_require_text("unavailable_reasons key", name)] = _require_text(
                "unavailable_reasons value", reason)
        available = {s.metric for s in samples if s.value is not None}
        for name in normalised:
            if name in available:
                raise ValueError(
                    f"{name}: listed as unavailable but also carries a measured sample. "
                    f"A metric cannot be both measured and not measured in one record.")
        for sample in samples:
            if sample.value is None:
                stated = normalised.get(sample.metric)
                if stated is None:
                    raise ValueError(
                        f"{sample.metric}: sample is unavailable with reason "
                        f"{sample.sample.unavailable_reason!r} but the record's "
                        f"unavailable_reasons does not carry it. The reason has to "
                        f"travel with the record or the absence becomes a zero.")
                if stated != sample.sample.unavailable_reason:
                    raise ValueError(
                        f"{sample.metric}: the record's reason {stated!r} disagrees with "
                        f"the sample's reason {sample.sample.unavailable_reason!r}")
        object.__setattr__(self, "unavailable_reasons", MappingProxyType(normalised))

        if not isinstance(self.comparison_eligible, bool):
            raise TypeError(
                f"comparison_eligible must be a bool; got "
                f"{type(self.comparison_eligible).__name__}")
        if self.mode == "recorded" and self.comparison_eligible:
            raise ValueError(
                "a recorded record cannot be comparison_eligible. Playback is evidence of "
                "what was recorded, never of what the machine does now, and no field may "
                "make it claim otherwise.")
        if self.comparison_eligible:
            settled = [s for s in samples
                       if s.phase is SamplePhase.SETTLED and s.value is not None]
            if not settled:
                raise ValueError(
                    "comparison_eligible=True requires at least one settled, measured "
                    "sample; a record with only warm-up or unavailable samples has no "
                    "steady-state evidence to compare.")

    @property
    def fingerprint_hash(self) -> str:
        """The digest of the workload, for a manifest and a quick refusal check."""
        return self.fingerprint.fingerprint_hash()

    @property
    def machine_id(self) -> str:
        return self.environment.machine_id

    def settled_samples(self) -> tuple[RecordedSample, ...]:
        """The samples a steady-state comparison may use, in record order."""
        return tuple(s for s in self.samples if s.phase is SamplePhase.SETTLED)

    def to_json(self) -> dict[str, Any]:
        """The exported record. **Imagery is deliberately absent.**

        Frames live only in memory and are written by explicit export opt-in; a manifest
        that carried them would defeat the default. `fingerprint_hash` is included so a
        reader can check the fingerprint against the fields without rehashing by hand,
        and `from_json` refuses a digest that does not match.
        """
        return {
            "format": "benchmark-record/1",
            "mode": self.mode,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "warmup_s": self.warmup_s,
            "fingerprint": self.fingerprint.to_json(),
            "fingerprint_hash": self.fingerprint_hash,
            "environment": self.environment.to_json(),
            "samples": [sample.to_json() for sample in self.samples],
            "trial_summary": [trial.to_json() for trial in self.trial_summary],
            "exclusions": list(self.exclusions),
            "comparison_eligible": self.comparison_eligible,
            "unavailable_reasons": dict(self.unavailable_reasons),
            "routing_transitions": [dict(t) for t in self.routing_transitions],
            "errors": list(self.errors),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "BenchmarkRecord":
        """Rebuild a record, refusing a missing field or a mismatched fingerprint hash."""
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"BenchmarkRecord.from_json expects a mapping; got "
                f"{type(payload).__name__}")
        required = ("mode", "started_at", "duration_s", "fingerprint", "environment",
                    "samples", "trial_summary", "exclusions", "comparison_eligible",
                    "unavailable_reasons")
        missing = sorted(name for name in required if name not in payload)
        if missing:
            raise ValueError(
                f"BenchmarkRecord.from_json: missing required field(s) {missing}")
        fingerprint = WorkloadFingerprint.from_json(payload["fingerprint"])
        stated = payload.get("fingerprint_hash")
        if stated is not None and stated != fingerprint.fingerprint_hash():
            raise ValueError(
                "fingerprint_hash does not match the fingerprint fields it claims to "
                "describe; the record was edited after the workload was measured.")
        record = cls(
            mode=payload["mode"],
            started_at=payload["started_at"],
            duration_s=payload["duration_s"],
            fingerprint=fingerprint,
            environment=EnvironmentManifest.from_json(payload["environment"]),
            samples=tuple(RecordedSample.from_json(item) for item in payload["samples"]),
            trial_summary=tuple(TrialSummary.from_json(item)
                                for item in payload["trial_summary"]),
            exclusions=tuple(payload["exclusions"]),
            comparison_eligible=payload["comparison_eligible"],
            unavailable_reasons=dict(payload["unavailable_reasons"]),
            warmup_s=payload.get("warmup_s", 0.0),
            routing_transitions=tuple(payload.get("routing_transitions") or ()),
            errors=tuple(payload.get("errors") or ()),
        )
        return record
