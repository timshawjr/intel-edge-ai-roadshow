"""`export_bundle` - a record as a ZIP, and **no camera imagery unless asked for**.

The failure this module exists to prevent is an export that quietly carries frames. Booth
footage is customer context; a bundle handed to a colleague, attached to a ticket or
committed to a repository must not contain it by default, and "the member names look like
JSON" is not proof. So:

- the default member set is fixed and textual: `manifest.json`,
  `routing_transitions.json`, `metrics_summary.json`, `metrics_raw.jsonl`, `errors.log`
  and `README.txt`;
- imagery on the record is held in memory and **never written** unless
  `include_imagery=True`;
- when it is written, the README **says so** in the bundle itself, so the opt-in travels
  with the file rather than living in the memory of whoever made it;
- the README states what is not included as well, because a bundle that lists only its
  contents reads as complete.

`metrics_summary.json` is derived from `metrics_raw.jsonl`: each metric carries the
settled samples' **median and full min/max range**, plus the cold samples listed
separately. A cold sample is never folded into the steady-state figure - the NPU's
published 5.2-5.9 ms against its sustained 3.96-4.01 ms is exactly the confusion that
would create. An unavailable metric is `null` with its reason, never `0`.

The archive is deterministic: fixed member order, fixed ZIP timestamps, canonical JSON.
Two exports of the same record are **byte-identical**, which makes a bundle diffable and
a reproducible artifact rather than a file whose hash changes every time it is written.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from app.contracts.benchmark_record import (
    BenchmarkRecord,
    RecordedSample,
    SamplePhase,
)

__all__ = ["export_bundle"]

BUNDLE_FORMAT = "benchmark-bundle/1"
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)
_REQUIRED_MEMBERS = (
    "manifest.json",
    "routing_transitions.json",
    "metrics_summary.json",
    "metrics_raw.jsonl",
    "errors.log",
    "README.txt",
)
_RECORDED_NOTICE = "Recorded demonstration - metrics are not live"


def _json_bytes(payload: Any) -> bytes:
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    return text.encode("utf-8")


def _jsonl_bytes(samples: tuple[RecordedSample, ...]) -> bytes:
    lines = [
        json.dumps(sample.to_json(), sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True)
        for sample in samples
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _metric_names(record: BenchmarkRecord) -> tuple[str, ...]:
    names: list[str] = []
    for sample in record.samples:
        if sample.metric not in names:
            names.append(sample.metric)
    # A metric that is wholly unavailable has no sample at all; its reason is in the
    # record and it belongs in the summary rather than vanishing from it.
    for name in record.unavailable_reasons:
        if name not in names:
            names.append(name)
    return tuple(names)


def _metric_entry(record: BenchmarkRecord, name: str) -> dict[str, Any]:
    samples = [s for s in record.samples if s.metric == name]
    settled = [s for s in samples if s.phase is SamplePhase.SETTLED]
    cold = [s for s in samples if s.phase is SamplePhase.COLD]
    settled_values = sorted(s.value for s in settled if s.value is not None)
    cold_values = sorted(s.value for s in cold if s.value is not None)
    first = samples[0] if samples else None
    return {
        "metric": name,
        "unit": first.unit if first is not None else None,
        "device_scope": first.sample.device_scope if first is not None else None,
        "provider": first.sample.provider if first is not None else None,
        "unavailable_reason": record.unavailable_reasons.get(name),
        "settled": {
            "count": len(settled_values),
            "values": settled_values,
            "median": (float(statistics.median(settled_values))
                       if settled_values else None),
            "min": min(settled_values) if settled_values else None,
            "max": max(settled_values) if settled_values else None,
        },
        "cold": {
            "count": len(cold_values),
            "values": cold_values,
        },
    }


def _summary(record: BenchmarkRecord) -> dict[str, Any]:
    return {
        "format": BUNDLE_FORMAT,
        "warmup_s": record.warmup_s,
        "duration_s": record.duration_s,
        "phase_basis": (
            "settled samples only; cold samples are listed but never summarised into "
            "the steady-state figure"),
        "metrics": {name: _metric_entry(record, name)
                    for name in _metric_names(record)},
        "traceability": (
            "every value above is computed from metrics_raw.jsonl; no figure is "
            "estimated or derived from another figure"),
    }


def _manifest(record: BenchmarkRecord, include_imagery: bool,
              imagery_names: tuple[str, ...]) -> dict[str, Any]:
    included = include_imagery and bool(imagery_names)
    return {
        "format": BUNDLE_FORMAT,
        "record_format": "benchmark-record/1",
        "mode": record.mode,
        "started_at": record.started_at,
        "duration_s": record.duration_s,
        "warmup_s": record.warmup_s,
        "fingerprint": record.fingerprint.to_json(),
        "fingerprint_hash": record.fingerprint_hash,
        "environment": record.environment.to_json(),
        "comparison_eligible": record.comparison_eligible,
        "unavailable_reasons": dict(record.unavailable_reasons),
        "exclusions": list(record.exclusions),
        "trial_summary": [trial.to_json() for trial in record.trial_summary],
        "sample_count": len(record.samples),
        "metric_names": list(_metric_names(record)),
        "imagery_requested": include_imagery,
        "included_imagery": included,
        "imagery_members": [f"imagery/{name}" for name in imagery_names]
                           if included else [],
        "notice": _RECORDED_NOTICE if record.mode == "recorded" else None,
    }


def _readme(record: BenchmarkRecord, include_imagery: bool,
            imagery_names: tuple[str, ...]) -> str:
    included = include_imagery and bool(imagery_names)
    lines = [
        "Benchmark bundle",
        "================",
        "",
        "Record format: benchmark-record/1",
        f"Bundle format: {BUNDLE_FORMAT}",
        f"Mode: {record.mode}",
        f"Started: {record.started_at}",
        f"Duration: {record.duration_s:g} s (warm-up {record.warmup_s:g} s)",
        f"Machine: {record.environment.machine_id}",
        f"Fingerprint: {record.fingerprint_hash}",
        "",
        "What is included in this bundle",
        "-------------------------------",
        "- manifest.json: mode, fingerprint, environment and the record's own claims.",
        "- routing_transitions.json: every placement transition the run recorded.",
        "- metrics_summary.json: settled median and min/max range per metric; cold "
        "samples listed separately.",
        "- metrics_raw.jsonl: one JSON object per sample, the source of every summary "
        "figure.",
        "- errors.log: one line per error, or empty when there were none.",
        "- README.txt: this file.",
    ]
    if included:
        lines += [
            "",
            "Camera imagery",
            "--------------",
            f"Camera imagery IS included in this bundle: {len(imagery_names)} camera "
            f"frame(s) under imagery/. This was an explicit operator request "
            f"(include_imagery=True); it is not the default export.",
        ]
    elif include_imagery:
        lines += [
            "",
            "Camera imagery",
            "--------------",
            "Camera imagery was requested (include_imagery=True) but this record carries "
            "no frames, so no image is present in this bundle.",
        ]
    else:
        lines += [
            "",
            "Camera imagery",
            "--------------",
            "Camera imagery is NOT included. No camera frame, thumbnail or video frame "
            "is present in this bundle; the default export is textual evidence only.",
        ]
    lines += [
        "",
        "What is not included",
        "--------------------",
        "- Camera imagery, unless the section above says it was explicitly included.",
        "- Any model artifact; the fingerprint identifies it by SHA-256.",
        "- Any figure that is not derivable from metrics_raw.jsonl.",
    ]
    if record.mode == "recorded":
        lines += [
            "",
            "Recorded demonstration",
            "----------------------",
            _RECORDED_NOTICE,
            "This bundle describes a playback of a recording. It is evidence of what was "
            "recorded, not of what the machine is doing now, and it cannot be compared "
            "against a live baseline.",
        ]
    else:
        lines += [
            "",
            "Live measurement",
            "----------------",
            "These metrics were measured live on the machine above, under the fingerprint "
            "above.",
        ]
    if record.exclusions:
        lines += ["", "Exclusions", "----------"]
        lines += [f"- {exclusion}" for exclusion in record.exclusions]
    lines.append("")
    return "\n".join(lines)


def _write_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=_FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 0
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def export_bundle(record: BenchmarkRecord, destination: Path | str,
                  include_imagery: bool = False) -> Path:
    """Write `record` to a ZIP at `destination` and return the path written.

    `destination` is treated as a file path; `.zip` is appended when it is missing, and
    parent directories are created. The record's imagery is written only when
    `include_imagery` is true, in which case `README.txt` says so.

    The write goes to a temporary file in the destination directory and is moved into
    place, so a failure part-way through cannot leave a half-written bundle that looks
    complete.
    """
    if not isinstance(record, BenchmarkRecord):
        raise TypeError(
            f"export_bundle expects a BenchmarkRecord; got {type(record).__name__}. A "
            f"dictionary cannot be exported as evidence because it has not passed the "
            f"record's own refusals.")
    if not isinstance(include_imagery, bool):
        raise TypeError("include_imagery must be a bool")

    path = Path(destination)
    if path.suffix.lower() != ".zip":
        path = path.with_name(path.name + ".zip")
    path.parent.mkdir(parents=True, exist_ok=True)

    imagery = tuple(record.imagery) if include_imagery else ()
    imagery_names = tuple(frame.name for frame in imagery)

    members: dict[str, bytes] = {
        "manifest.json": _json_bytes(_manifest(record, include_imagery, imagery_names)),
        "routing_transitions.json": _json_bytes(
            [dict(transition) for transition in record.routing_transitions]),
        "metrics_summary.json": _json_bytes(_summary(record)),
        "metrics_raw.jsonl": _jsonl_bytes(record.samples),
        "errors.log": ("\n".join(record.errors) + ("\n" if record.errors else ""))
                      .encode("utf-8"),
        "README.txt": _readme(record, include_imagery, imagery_names).encode("utf-8"),
    }
    for frame in imagery:
        members[f"imagery/{frame.name}"] = frame.data

    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".zip", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in _REQUIRED_MEMBERS:
                _write_member(archive, name, members[name])
            for name in sorted(name for name in members
                               if name not in _REQUIRED_MEMBERS):
                _write_member(archive, name, members[name])
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
