"""Compare two `BenchmarkRecord`s, or refuse and say why.

The one rule this tool enforces is that **a number from one set of conditions is not
evidence about another**. `compare()` checks the workload fingerprint field by field, the
machine conditions that can move a number, the record modes, and each record's own
`comparison_eligible` claim. If anything differs it returns `eligible=False` with
`differing_fields` naming every field, and `per_metric` is **empty**: refusing is not a
rebasing, and there is no delta for a reader to quote as though the baseline still
applied.

When the pair is eligible, every metric reports both sides' **median and full min/max
range**, plus `delta` (candidate median minus baseline median) and `within_noise` (the two
ranges overlap). A median with no range hides the variation that decides whether the
difference means anything.

Only `SETTLED` samples are used. A cold warm-up sample is retained in the record and
exported, but it cannot become the steady-state figure - the NPU's published 5.2-5.9 ms
against its sustained 3.96-4.01 ms is the case that makes this matter. A metric missing
from one record is `None` with a reason, never `0`.

Usage:

    python tools/compare_records.py baseline.json candidate.json

Each argument is a `BenchmarkRecord.to_json()` payload. The process exits 0 for an
eligible comparison and 1 for a refusal, so a CI step can gate on it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Sequence

from app.contracts.benchmark_record import BenchmarkRecord, SamplePhase

__all__ = ["Comparison", "MetricComparison", "compare", "format_comparison", "main"]

_ROLES = ("baseline", "candidate")


def _as_record(value: Any, role: str) -> BenchmarkRecord:
    if isinstance(value, BenchmarkRecord):
        return value
    if isinstance(value, Mapping):
        if "fingerprint" not in value or "samples" not in value:
            raise TypeError(
                f"{role} must be a BenchmarkRecord or a record JSON payload; the mapping "
                f"has no 'fingerprint' and 'samples', so it is not a record.")
        return BenchmarkRecord.from_json(value)
    raise TypeError(
        f"{role} must be a BenchmarkRecord or a record JSON payload; got "
        f"{type(value).__name__}")


def _parse_timestamp(value: str) -> datetime:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(text)


def _settled_values(record: BenchmarkRecord, metric: str) -> list[float]:
    return [
        sample.value
        for sample in record.samples
        if sample.metric == metric
        and sample.phase is SamplePhase.SETTLED
        and sample.value is not None
    ]


def _metric_names(baseline: BenchmarkRecord,
                  candidate: BenchmarkRecord) -> tuple[str, ...]:
    names: list[str] = []
    for record in (baseline, candidate):
        for sample in record.samples:
            if sample.metric not in names:
                names.append(sample.metric)
        for name in record.unavailable_reasons:
            if name not in names:
                names.append(name)
    return tuple(names)


def _unit(record: BenchmarkRecord, metric: str) -> str | None:
    for sample in record.samples:
        if sample.metric == metric:
            return sample.unit
    return None


def _reason(record: BenchmarkRecord, metric: str, role: str,
            values: Sequence[float]) -> str | None:
    if values:
        return None
    stated = record.unavailable_reasons.get(metric)
    if stated:
        return stated
    return f"no settled {metric!r} sample in {role} record"


@dataclass(frozen=True)
class MetricComparison:
    """One metric on both sides: medians, ranges, delta and whether they overlap.

    `median` and `range` are keyed by `"baseline"` and `"candidate"`. Both sides are
    always present; a side with no settled measurement is `None` and carries a reason in
    `unavailable`, never `0`.
    """

    metric: str
    unit: str | None
    median: Mapping[str, float | None]
    range: Mapping[str, tuple[float, float] | None]
    delta: float | None
    within_noise: bool | None
    sample_counts: Mapping[str, int]
    unavailable: Mapping[str, str | None]


@dataclass(frozen=True)
class Comparison:
    """The result of comparing two records: eligible, or refused by field.

    The older record is labelled `previous run`; the other keeps its role name. The
    display properties append its machine and timestamp, so a figure quoted from the
    older record can always be traced to the conditions it was measured under.
    """

    eligible: bool
    differing_fields: tuple[str, ...]
    baseline_label: str
    candidate_label: str
    per_metric: Mapping[str, MetricComparison]
    baseline_machine: str
    baseline_started_at: str
    candidate_machine: str
    candidate_started_at: str
    baseline_status: str
    refusal: str | None

    @property
    def baseline_display(self) -> str:
        return (f"{self.baseline_label} ({self.baseline_machine}, "
                f"{self.baseline_started_at})")

    @property
    def candidate_display(self) -> str:
        return (f"{self.candidate_label} ({self.candidate_machine}, "
                f"{self.candidate_started_at})")


def _labels(baseline: BenchmarkRecord,
            candidate: BenchmarkRecord) -> tuple[str, str]:
    """(baseline_label, candidate_label) - the older record is `previous run`."""
    baseline_time = _parse_timestamp(baseline.started_at)
    candidate_time = _parse_timestamp(candidate.started_at)
    if baseline_time <= candidate_time:
        return "previous run", "candidate"
    return "baseline", "previous run"


def _blockers(baseline: BenchmarkRecord,
              candidate: BenchmarkRecord) -> tuple[tuple[str, ...], str | None]:
    fingerprint_fields = baseline.fingerprint.differing_fields(candidate.fingerprint)
    environment_fields = tuple(
        f"environment.{name}"
        for name in baseline.environment.condition_fields()
        if getattr(baseline.environment, name) != getattr(candidate.environment, name)
    )
    mode_differs = baseline.mode != candidate.mode
    ineligible = (not baseline.comparison_eligible
                  or not candidate.comparison_eligible)

    ordered: list[str] = list(fingerprint_fields) + list(environment_fields)
    if mode_differs:
        ordered.append("mode")
    if ineligible:
        ordered.append("comparison_eligible")
    differing = tuple(dict.fromkeys(ordered))

    if not differing:
        return (), None

    parts: list[str] = []
    if fingerprint_fields:
        parts.append("fingerprints differ: " + ", ".join(fingerprint_fields))
    if environment_fields:
        parts.append("environment differs: " + ", ".join(environment_fields))
    if mode_differs:
        parts.append(f"mode differs: {baseline.mode} vs {candidate.mode}")
    if ineligible:
        who = [role for role, record in zip(_ROLES, (baseline, candidate))
               if not record.comparison_eligible]
        parts.append("not comparison_eligible: " + ", ".join(who))
    return differing, "; ".join(parts)


def _metric_comparison(baseline: BenchmarkRecord, candidate: BenchmarkRecord,
                       metric: str) -> MetricComparison:
    baseline_values = _settled_values(baseline, metric)
    candidate_values = _settled_values(candidate, metric)

    def median(values: Sequence[float]) -> float | None:
        return float(statistics.median(values)) if values else None

    def span(values: Sequence[float]) -> tuple[float, float] | None:
        return (float(min(values)), float(max(values))) if values else None

    baseline_median = median(baseline_values)
    candidate_median = median(candidate_values)
    baseline_range = span(baseline_values)
    candidate_range = span(candidate_values)

    delta: float | None
    if baseline_median is None or candidate_median is None:
        delta = None
    else:
        delta = candidate_median - baseline_median

    within_noise: bool | None
    if baseline_range is None or candidate_range is None:
        within_noise = None
    else:
        # Range overlap: the difference is inside what the runs themselves varied by.
        within_noise = not (baseline_range[1] < candidate_range[0]
                            or candidate_range[1] < baseline_range[0])

    return MetricComparison(
        metric=metric,
        unit=_unit(baseline, metric) or _unit(candidate, metric),
        median=MappingProxyType({"baseline": baseline_median,
                                 "candidate": candidate_median}),
        range=MappingProxyType({"baseline": baseline_range,
                                "candidate": candidate_range}),
        delta=delta,
        within_noise=within_noise,
        sample_counts=MappingProxyType({
            "baseline": len(baseline_values),
            "candidate": len(candidate_values),
        }),
        unavailable=MappingProxyType({
            "baseline": _reason(baseline, metric, "baseline", baseline_values),
            "candidate": _reason(candidate, metric, "candidate", candidate_values),
        }),
    )


def compare(baseline: BenchmarkRecord | Mapping[str, Any],
            candidate: BenchmarkRecord | Mapping[str, Any]) -> Comparison:
    """Compare two records, refusing any difference in conditions.

    A refusal returns every differing field name and **no per-metric results**. A
    matched pair reports every metric with both medians, both ranges, the delta and
    whether the ranges overlap.
    """
    left = _as_record(baseline, "baseline")
    right = _as_record(candidate, "candidate")
    differing, refusal = _blockers(left, right)
    baseline_label, candidate_label = _labels(left, right)
    if differing:
        return Comparison(
            eligible=False,
            differing_fields=differing,
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            per_metric=MappingProxyType({}),
            baseline_machine=left.machine_id,
            baseline_started_at=left.started_at,
            candidate_machine=right.machine_id,
            candidate_started_at=right.started_at,
            baseline_status="invalidated",
            refusal=refusal,
        )

    per_metric = {
        metric: _metric_comparison(left, right, metric)
        for metric in _metric_names(left, right)
    }
    return Comparison(
        eligible=True,
        differing_fields=(),
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        per_metric=MappingProxyType(per_metric),
        baseline_machine=left.machine_id,
        baseline_started_at=left.started_at,
        candidate_machine=right.machine_id,
        candidate_started_at=right.started_at,
        baseline_status="valid",
        refusal=None,
    )


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g}"


def _range(value: tuple[float, float] | None) -> str:
    if value is None:
        return "n/a"
    return f"[{value[0]:g}, {value[1]:g}]"


def format_comparison(comparison: Comparison) -> str:
    """A table with medians **and** ranges, or the refusal and the fields that caused it."""
    lines = [
        f"baseline:  {comparison.baseline_display}",
        f"candidate: {comparison.candidate_display}",
        f"eligible: {'yes' if comparison.eligible else 'no'}",
        f"baseline status: {comparison.baseline_status}",
    ]
    if not comparison.eligible:
        lines.append(f"refusal: {comparison.refusal}")
        lines.append("differing fields: "
                     + (", ".join(comparison.differing_fields) or "(none)"))
        return "\n".join(lines)

    lines += [
        "",
        "comparisons use settled samples only; range is [min, max]",
        f"{'metric':28} {'baseline median':>15} {'baseline range':>16} "
        f"{'candidate median':>16} {'candidate range':>17} {'delta':>8} "
        f"{'within noise':>12}",
    ]
    for metric in sorted(comparison.per_metric):
        item = comparison.per_metric[metric]
        within = ("n/a" if item.within_noise is None
                  else ("yes" if item.within_noise else "no"))
        lines.append(
            f"{metric:28} {_number(item.median['baseline']):>15} "
            f"{_range(item.range['baseline']):>16} "
            f"{_number(item.median['candidate']):>16} "
            f"{_range(item.range['candidate']):>17} "
            f"{_number(item.delta):>8} {within:>12}")
        for role in _ROLES:
            reason = item.unavailable[role]
            if reason:
                lines.append(f"  {role}: unavailable - {reason}")
    return "\n".join(lines)


def _load(path: Path) -> BenchmarkRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return BenchmarkRecord.from_json(payload)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{path}: not a benchmark record: {exc}") from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tools/compare_records.py",
        description="Compare two benchmark records; refuse a mismatched fingerprint.")
    parser.add_argument("baseline", type=Path, help="baseline record JSON")
    parser.add_argument("candidate", type=Path, help="candidate record JSON")
    args = parser.parse_args(argv)

    comparison = compare(_load(args.baseline), _load(args.candidate))
    print(format_comparison(comparison))
    return 0 if comparison.eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
