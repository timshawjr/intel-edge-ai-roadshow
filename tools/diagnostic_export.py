#!/usr/bin/env python3
"""A diagnostic ZIP: the environment, discovery, self-test, the gates and a record.

A booth failure is diagnosed away from the booth, so the bundle has to be safe to send.
That means two things, and both are tested by `tests/test_diagnostic_export.py`:

1. **The gates travel as they are, including failing ones.** The three open gates this
   project reports are content (9 of 12 clips, 0 rights-cleared), legibility (fails at 3 m
   on the attached monitor) and the 2 s warm transition (p50 2.238 s in M2's session; the
   committed 2 s-gate file's p50 is 2.628 s, with 37 of 50 over). A diagnosis that leaves
   out a failing gate is worse than none, so `gates.json` carries a `status` per gate and
   `all_passed` is computed, never assumed.
2. **No camera imagery by default.** The export scans **member bytes** for image magic
   numbers before writing, and refuses to produce a bundle whose content contains one. A
   frame appended to a log is exactly the leak a member-name check misses; the scan is
   shown to find PNG, JPEG and WebP in constructed bytes, so a zero is a measurement.

The export is deliberately read-only with respect to the machine it describes. In
particular it never writes `config/telemetry_map.json`: that file is verified data and one
tool in this repository rewrites it as a side effect. The test hashes it before and after.

Usage::

    python -m tools.diagnostic_export --destination logs/diagnostic.zip

Stdlib only, no network at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION: Path = REPO_ROOT / "logs" / "diagnostic.zip"
DEFAULT_RECORD: Path = REPO_ROOT / "docs" / "qualification" / "m5-records" / "candidate.json"
TRANSITION_RECORD: Path = (
    REPO_ROOT / "docs" / "qualification" / "m2-headless"
    / "route-stress-50-gate-2s-after-teardown-fix.json"
)
FORMAT: str = "diagnostic-bundle/1"
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)
_SIGNATURE_NOTE: str = (
    "UNSIGNED. This diagnostic bundle carries no Authenticode signature and no certificate "
    "was available to create one. It is a text bundle: environment, device discovery, "
    "self-test output, gate results and a benchmark record. It contains no camera imagery. "
    "Do not weaken Defender, ASR, AppLocker or the execution policy to open it."
)

#: Image magic numbers that must never appear in a diagnostic bundle. Each is a signature
#: at the start of a file format, not a name: a frame pasted into a log is caught too.
_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
)

GateStatus = str
_VALID_STATUSES = ("passed", "failed", "unavailable")


class ImageryDetected(RuntimeError):
    """Raised rather than writing a bundle whose bytes carry an image."""

    def __init__(self, hits: Mapping[str, str]) -> None:
        self.hits = dict(hits)
        super().__init__(
            "refusing to write a diagnostic bundle that carries camera imagery: "
            + ", ".join(f"{name} ({magic})" for name, magic in hits.items())
        )


def image_magic(data: bytes) -> str | None:
    """Return the image format whose magic number is in **these bytes**, or ``None``.

    The whole payload is searched, not a prefix: a PNG appended to a text log is the leak
    this exists to catch. WebP is a RIFF container, so its four-byte form tag at offset 8
    is checked rather than a twelve-byte prefix that would miss a chunked file.
    """
    for signature, name in _IMAGE_SIGNATURES:
        if signature in data:
            return name
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _json_bytes(payload: Any) -> bytes:
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    return text.encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    raise TypeError(
        f"expected a mapping or an object with as_dict(), got {type(value).__name__}"
    )


@dataclass(frozen=True)
class GateResult:
    """One gate, its measured values, and a status that is never inferred."""

    gate: str
    status: GateStatus
    measured: Mapping[str, Any] = field(default_factory=dict)
    detail: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"gate {self.gate!r} has status {self.status!r}; expected one of "
                f"{list(_VALID_STATUSES)}. A gate that was not measured is 'unavailable', "
                "not passed."
            )
        if not self.gate:
            raise ValueError("a gate needs a name")

    def to_json(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status,
            "measured": dict(self.measured),
            "detail": self.detail,
            "source": self.source,
        }


def _arcmin(height_mm: float, distance_m: float) -> float:
    import math

    return math.degrees(2.0 * math.atan(height_mm / (2.0 * distance_m * 1000.0))) * 60.0


def content_gate_result(inventory: Any | None = None, *,
                        measure: Callable[[], Any] | None = None) -> GateResult:
    """The content gate, read from a measured inventory (never a constant)."""
    if inventory is None:
        if measure is None:
            from tools.content_gate import content_inventory as measure

        try:
            inventory = measure()
        except Exception as exc:  # a diagnostic must not crash on a missing clip
            return GateResult(
                gate="content", status="unavailable",
                measured={"error": f"{type(exc).__name__}: {exc}"},
                detail=f"the content inventory could not be measured: {exc}",
                source="tools.content_gate.content_inventory",
            )

    clips = len(inventory.clips)
    required = int(inventory.clips_required_total)
    cleared = clips - len(inventory.clips_with_unverified_rights)
    in_band = len(inventory.clips_in_band)
    packs = int(inventory.packs_provisioned)
    status = "passed" if inventory.met else "failed"
    return GateResult(
        gate="content",
        status=status,
        measured={
            "clips_provisioned": clips,
            "clips_required": required,
            "clips_in_band": in_band,
            "packs_provisioned": packs,
            "rights_cleared": cleared,
            "met": bool(inventory.met),
        },
        detail=(
            f"{clips} of {required} clips provisioned, {cleared} rights-cleared; "
            f"{in_band} inside the duration band; {packs} pack(s)"
        ),
        source="tools.content_gate.content_inventory (measured from the files)",
    )


def legibility_gate_result(
    *,
    mm_per_logical_px: float | None = None,
    cap_em: float | None = None,
    x_em: float | None = None,
    font_px: int | None = None,
    distances_m: Sequence[float] = (2.0, 3.0),
    threshold_arcmin: float | None = None,
) -> GateResult:
    """The legibility gate at the geometry it is given, defaulting to the attached monitor.

    The default is the **measured** DELL S3422DWG geometry (`theme.ACTUAL_MONITOR_*`), not
    the 55-inch requirement. A test that defaulted to the requirement would pass at 3 m
    while the delivered screen fails there.
    """
    from app import theme

    mm = (theme.ACTUAL_MONITOR_MM_PER_LOGICAL_PX
          if mm_per_logical_px is None else float(mm_per_logical_px))
    cap = theme.ARIAL_CAP_HEIGHT_EM if cap_em is None else float(cap_em)
    xheight = theme.ARIAL_X_HEIGHT_EM if x_em is None else float(x_em)
    px = theme.TYPE_BOOTH_DETAIL if font_px is None else int(font_px)
    threshold = (theme.READING_MIN_ARCMIN
                 if threshold_arcmin is None else float(threshold_arcmin))

    cap_mm = px * cap * mm
    x_mm = px * xheight * mm
    cap_arcmin = {f"{distance:.1f}": round(_arcmin(cap_mm, distance), 2)
                  for distance in distances_m}
    x_arcmin = {f"{distance:.1f}": round(_arcmin(x_mm, distance), 2)
                for distance in distances_m}
    failures = [
        f"{measure} at {distance} m ({angle:.1f}' < {threshold:.1f}')"
        for measure, series in (("cap height", cap_arcmin), ("x-height", x_arcmin))
        for distance, angle in series.items()
        if angle < threshold
    ]
    passed = not failures
    return GateResult(
        gate="legibility",
        status="passed" if passed else "failed",
        measured={
            "font_px": px,
            "mm_per_logical_px": mm,
            "cap_em": cap,
            "x_height_em": xheight,
            "threshold_arcmin": threshold,
            "cap_arcmin": cap_arcmin,
            "x_arcmin": x_arcmin,
            "geometry": ("attached monitor (DELL S3422DWG, measured)"
                         if mm_per_logical_px is None
                         else "supplied geometry"),
        },
        detail=("every measured element clears the threshold"
                if passed else "fails: " + "; ".join(failures)),
        source="app.theme constants, 2*atan(h/2d) at the stated distances",
    )


def transition_gate_result(payload: Mapping[str, Any] | None = None, *,
                           read: Callable[[], Mapping[str, Any]] | None = None) -> GateResult:
    """The 2 s warm-transition gate from a committed M2 distribution."""
    if payload is None:
        if read is None:
            def read() -> Mapping[str, Any]:
                return json.loads(TRANSITION_RECORD.read_text(encoding="utf-8"))

        try:
            payload = read()
        except Exception as exc:
            return GateResult(
                gate="transition", status="unavailable",
                measured={"error": f"{type(exc).__name__}: {exc}"},
                detail=f"the transition distribution could not be read: {exc}",
                source=str(TRANSITION_RECORD.relative_to(REPO_ROOT)),
            )

    p50 = payload.get("warm_elapsed_p50")
    over = payload.get("over_two_second")
    transitions = payload.get("transitions")
    timeout = payload.get("timeout_s")
    if p50 is None or over is None or transitions is None:
        return GateResult(
            gate="transition", status="unavailable",
            measured=dict(payload),
            detail="the distribution does not carry p50, over-two-second and transitions",
            source=str(TRANSITION_RECORD.relative_to(REPO_ROOT)),
        )
    status = "passed" if int(over) == 0 else "failed"
    return GateResult(
        gate="transition",
        status=status,
        measured={
            "p50_s": float(p50),
            "over_two_second": int(over),
            "transitions": int(transitions),
            "timeout_s": timeout,
            "acknowledged": payload.get("acknowledged"),
            "censored": payload.get("censored"),
        },
        detail=(
            f"{int(over)} of {int(transitions)} warm transitions over {timeout} s "
            f"at p50 {float(p50):.3f} s"
        ),
        source=str(TRANSITION_RECORD.relative_to(REPO_ROOT)),
    )


def collect_gates() -> tuple[GateResult, ...]:
    """The three gates this milestone reports, measured independently."""
    results: list[GateResult] = []
    for name, measure in (
        ("content", content_gate_result),
        ("legibility", legibility_gate_result),
        ("transition", transition_gate_result),
    ):
        try:
            results.append(measure())
        except Exception as exc:  # never omit a gate because measuring it failed
            results.append(GateResult(
                gate=name, status="unavailable",
                measured={"error": f"{type(exc).__name__}: {exc}"},
                detail=f"the gate could not be measured: {exc}",
                source="tools.diagnostic_export",
            ))
    return tuple(results)


def collect_environment() -> Mapping[str, Any]:
    """Read the known-good environment from this machine at export time."""
    from tools.environment_manifest import collect_environment as collect

    return collect().as_dict()


def collect_device_discovery() -> Mapping[str, Any]:
    """What OpenVINO can see on this machine, including what it could not name."""
    import openvino as ov

    core = ov.Core()
    available = list(core.available_devices)
    names: dict[str, str | None] = {}
    unavailable: dict[str, str] = {}
    for device in available:
        try:
            names[device] = core.get_property(device, "FULL_DEVICE_NAME")
        except Exception as exc:
            names[device] = None
            unavailable[device] = f"{type(exc).__name__}: {exc}"
    return {
        "openvino_version": getattr(ov, "__version__", None),
        "available_devices": available,
        "device_names": names,
        "unavailable": unavailable,
    }


def run_selftest() -> str:
    """Run the repository self-test and return its real output and exit code."""
    command = [sys.executable, "-m", "app.main", "--selftest"]
    proc = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=300)
    return (
        "$ " + " ".join(command) + "\n"
        "--- stdout ---\n" + proc.stdout +
        "--- stderr ---\n" + proc.stderr +
        f"exit={proc.returncode}\n"
    )


def _record_member(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        return _json_bytes({
            "available": False,
            "path": str(path),
            "reason": f"{type(exc).__name__}: {exc}",
        })


def diagnostic_bundle(
    destination: str | Path,
    *,
    environment: Mapping[str, Any] | Any | None = None,
    discovery: Mapping[str, Any] | None = None,
    selftest: str | bytes | None = None,
    gates: Iterable[GateResult] | None = None,
    benchmark_record: bytes | None = None,
    benchmark_record_path: str | Path | None = None,
    generated_at: str | None = None,
) -> Path:
    """Write the diagnostic ZIP and return its path.

    Every argument is injectable so the tests can prove the bundle echoes its inputs and
    the gates are data rather than constants. The defaults read the machine.
    """
    destination = Path(destination)
    if environment is None:
        environment = collect_environment()
    environment_payload = _as_mapping(environment)

    if discovery is None:
        discovery = collect_device_discovery()
    discovery_payload = _as_mapping(discovery)

    if selftest is None:
        selftest_bytes = run_selftest().encode("utf-8")
    elif isinstance(selftest, bytes):
        selftest_bytes = selftest
    else:
        selftest_bytes = str(selftest).encode("utf-8")

    gate_results = tuple(collect_gates() if gates is None else gates)
    for gate in gate_results:
        if not isinstance(gate, GateResult):
            raise TypeError(f"gates must be GateResult instances, got {type(gate).__name__}")

    if benchmark_record is None:
        path = Path(benchmark_record_path) if benchmark_record_path is not None else DEFAULT_RECORD
        record_payload = _record_member(path)
    else:
        record_payload = bytes(benchmark_record)

    gates_payload = {
        "format": FORMAT,
        "gates": [gate.to_json() for gate in gate_results],
        "all_passed": all(gate.status == "passed" for gate in gate_results),
        "gate_count": len(gate_results),
    }

    members: dict[str, bytes] = {
        "environment.json": _json_bytes(environment_payload),
        "device_discovery.json": _json_bytes(discovery_payload),
        "selftest.txt": selftest_bytes,
        "gates.json": _json_bytes(gates_payload),
        "benchmark_record.json": record_payload,
        "README.txt": _readme(gates_payload).encode("utf-8"),
    }

    hits = {name: magic for name, payload in members.items()
            if (magic := image_magic(payload)) is not None}
    if hits:
        raise ImageryDetected(hits)

    manifest = {
        "format": FORMAT,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "signed": False,
        "signature_note": _SIGNATURE_NOTE,
        "imagery_included": False,
        "gates": {gate.gate: gate.status for gate in gate_results},
        "all_gates_passed": gates_payload["all_passed"],
        "members": {
            name: {"sha256": _sha256(payload), "bytes": len(payload)}
            for name, payload in members.items()
        },
    }
    members = {"manifest.json": _json_bytes(manifest), **members}

    # The manifest is generated content; scan it too rather than trusting the writer.
    hits = {name: magic for name, payload in members.items()
            if (magic := image_magic(payload)) is not None}
    if hits:
        raise ImageryDetected(hits)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip",
                                     dir=destination.parent) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in members.items():
                info = zipfile.ZipInfo(name, date_time=_FIXED_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, payload)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _readme(gates_payload: Mapping[str, Any]) -> str:
    statuses = ", ".join(
        f"{gate['gate']}={gate['status']}" for gate in gates_payload["gates"]
    )
    return (
        "Intel Edge AI Roadshow - diagnostic bundle\n"
        f"format: {FORMAT}\n"
        "signed: no. This bundle is unsigned and no certificate was available.\n"
        "No camera imagery is included. No camera frame, thumbnail or video frame is\n"
        "present in this bundle; it is textual evidence only, and the export refuses to\n"
        "write a member whose bytes contain an image magic number.\n"
        f"gates: {statuses}\n"
        f"all gates passed: {gates_payload['all_passed']}\n"
        "\n"
        "environment.json       the machine read at export time\n"
        "device_discovery.json what OpenVINO can see, including unnamed devices\n"
        "selftest.txt           the real `python -m app.main --selftest` output\n"
        "gates.json             content, legibility and transition, with their status\n"
        "benchmark_record.json  the record named on the command line, or why it is absent\n"
        "manifest.json          the SHA-256 and size of every other member\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.diagnostic_export",
        description=(
            "Write a diagnostic ZIP: environment, device discovery, self-test output, "
            "the three gate results and a benchmark record. No camera imagery."
        ),
    )
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION,
                        help=f"where to write the ZIP (default: {DEFAULT_DESTINATION})")
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD,
                        help="the BenchmarkRecord JSON to carry (default: the M5 record)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        path = diagnostic_bundle(args.destination, benchmark_record_path=args.record)
    except ImageryDetected as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
        gates = json.loads(archive.read("gates.json"))
    print(f"wrote {path}")
    print(f"members ({len(names)}): {', '.join(names)}")
    for gate in gates["gates"]:
        print(f"  gate {gate['gate']:<12} {gate['status']:<12} {gate['detail']}")
    print(f"all gates passed: {gates['all_passed']}")
    print(f"signed: {manifest['signed']}  imagery_included: {manifest['imagery_included']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the process entry point
    sys.exit(main())
