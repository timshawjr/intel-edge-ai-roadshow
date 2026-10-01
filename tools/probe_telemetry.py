r"""Probe the Windows `GPU Engine` counters once, at operator time, and record the answer.

Run this on the booth machine. Run it **while something is using the GPU and the NPU**, or
there will be no `GPU Engine` instances to enumerate and nothing can be concluded - the
probe says so and refuses to write a map rather than recording "no Neural engine exists"
from an empty observation.

    .venv\Scripts\python.exe tools\probe_telemetry.py
    .venv\Scripts\python.exe tools\probe_telemetry.py --allow-missing-npu

**Why this is a tool and not library code.** Resolving a LUID to a device needs Windows
management data, and the pinned dependency set has no `pywin32`. That is acceptable *here*
- this runs once, by an operator, before the demo - and is unacceptable on the 4 Hz
sampling path. So the resolution happens exactly once, here, and is written to
`config/telemetry_map.json`; `app/telemetry/pdh.py` and `app/telemetry/instance_map.py`
then only read that file and match strings, and never spawn a process. That split is the
design, and it is why the runtime can be pure.

**What is measured and what is resolved.** The instances, their engine types and their
utilisation percentages are all *measured*, read from
`Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine` through one CIM call. Which
*device* a LUID belongs to has to be resolved, and on this machine it cannot be read
directly - measured, see `DIRECT_LUID_LINK` in the output. The resolution therefore uses
the engine taxonomy Windows itself defines, and every attribution records the evidence it
rests on:

- a LUID exposing a `Neural` engine is the **NPU** - the neural engine is the NPU's engine
  and no other device here has one;
- a LUID exposing a `Compute`, `VideoDecode`, `VideoProcessing` or `Copy` engine is the
  **GPU**;
- a LUID exposing none of those is left **unattributed**. On this machine that is a real
  adapter, and assigning it to the nearest device would be exactly the real-but-wrong
  number this tool exists to prevent.

`--allow-missing-npu` exists because a negative result is a legitimate thing to record. It
is **not** a way to make a failing probe pass: with it, the map is still written with the
NPU absent and `npu_neural_counter: false`, and the runtime's NPU utilisation is `null`
because there is no rule to match against. The flag changes the exit code and nothing else.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inference.device_drivers import (  # noqa: E402
    POWERSHELL_EXE,
    POWERSHELL_TIMEOUT_S,
    WindowsDriverLookup,
)
from app.telemetry.instance_map import (  # noqa: E402
    ENGINE_TYPES,
    normalise_luid,
    parse_instance_name,
)

__all__ = [
    "CATEGORY",
    "COUNTER",
    "AdapterLuid",
    "EngineObservation",
    "main",
    "resolve_luid_devices",
    "run_cim",
]

#: The Windows performance object and counter this demo measures. Read from Windows rather
#: than assumed: `Get-Counter -ListSet 'GPU Engine'` on this machine reports exactly
#: `\GPU Engine(*)\Utilization Percentage` and `\GPU Engine(*)\Running Time`.
CATEGORY: Final[str] = "GPU Engine"
COUNTER: Final[str] = "Utilization Percentage"

ENGINE_WMI_CLASS: Final[str] = (
    "Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine"
)
ADAPTER_WMI_CLASS: Final[str] = (
    "Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory"
)

#: Engines that belong to the NPU. `Neural` is the NPU's engine; it appears on no GPU.
NPU_ENGINES: Final[frozenset[str]] = frozenset({"Neural"})

#: Engines that belong to a GPU compute/display adapter. `3D` is deliberately **excluded**:
#: measured on this machine a third adapter exposes `3D` and nothing else, and calling that
#: adapter the GPU on the strength of a `3D` engine would be a real number on the wrong
#: card.
GPU_ENGINES: Final[frozenset[str]] = frozenset(
    {"Compute", "VideoDecode", "VideoProcessing", "Copy", "GSC"}
)

#: One CIM call for everything. ~3-5 s cold on this machine, paid once, at operator time.
_SURVEY_SCRIPT: Final[str] = (
    "$ErrorActionPreference = 'Stop';"
    f"$engines = Get-CimInstance -ClassName {ENGINE_WMI_CLASS} "
    "| Select-Object Name,UtilizationPercentage,RunningTime;"
    f"$adapters = Get-CimInstance -ClassName {ADAPTER_WMI_CLASS} | Select-Object Name;"
    "$video = Get-CimInstance -ClassName Win32_VideoController "
    "| Select-Object Name,PNPDeviceID;"
    "$os = Get-CimInstance -ClassName Win32_OperatingSystem "
    "| Select-Object Caption,Version,BuildNumber;"
    "[pscustomobject]@{"
    "engines = $engines; adapters = $adapters; video = $video; os = $os} "
    "| ConvertTo-Json -Depth 4 -Compress"
)


def run_cim(script: str, timeout: float = POWERSHELL_TIMEOUT_S) -> str:
    """Run one inline PowerShell command and return stdout.

    The command is passed with `-Command` as a string. **No `.ps1` or `.bat` file is ever
    written or executed** - script execution is blocked by policy on the demo machine, and
    a wrapper script would be a convenience for a human operator, not the build path.
    """
    completed = subprocess.run(
        [POWERSHELL_EXE, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"powershell.exe exited {completed.returncode}: "
            f"{(completed.stderr or '').strip()[:400]}"
        )
    return completed.stdout


def _as_rows(payload: Any) -> list[dict[str, Any]]:
    """A `ConvertTo-Json` payload as a list of objects, whatever shape it arrived in."""
    if payload is None:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


@dataclass(frozen=True)
class AdapterLuid:
    """One adapter as the `GPU Adapter Memory` performance object reports it.

    That object has exactly **one instance per adapter**, so it is the authoritative
    adapter LUID set on this machine - measured: three of them,
    `luid_0x00000000_0x00010E9E_phys_0`, `luid_0x00000000_0x00011232_phys_0` and
    `luid_0x00000000_0x000112A0_phys_0`.
    """

    luid: str
    instance_name: str


@dataclass(frozen=True)
class EngineObservation:
    """One `GPU Engine` instance, exactly as Windows reported it.

    `utilization_percent` is the counter's own cooked value. `None` when the counter
    reported nothing - never `0.0`, because "no reading" and "no load" are different
    claims and only the first of them is safe to show.
    """

    instance: str
    pid: int | None
    luid: str
    engtype: str | None
    utilization_percent: float | None
    running_time: float | None


def _number(value: Any) -> float | None:
    """A CIM number, or `None` if it is absent or not a number. Never `0.0` for absent."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _engine_type(raw: str | None) -> str | None:
    """Windows spells the same engine both ways in different sources.

    Measured: `Get-Counter` prints `engtype_Compute` and `engtype_neural` in the same
    listing, and the CIM `Name` lower-cases everything. The canonical spelling is the one
    in `app.telemetry.instance_map.ENGINE_TYPES`, matched case-insensitively; an engine
    type outside that set keeps the spelling Windows used, because hiding an engine
    Windows reported would be the same mistake as inventing one.
    """
    text = str(raw or "").strip()
    if not text:
        return None
    for canonical in ENGINE_TYPES:
        if canonical.casefold() == text.casefold():
            return canonical
    return text


def survey() -> dict[str, Any]:
    """One CIM survey: engine instances, adapter LUIDs, display adapters, OS build."""
    payload = json.loads(run_cim(_SURVEY_SCRIPT))
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected CIM survey payload: {type(payload).__name__}")
    return payload


def parse_survey(payload: dict[str, Any]) -> tuple[
    tuple[EngineObservation, ...], tuple[AdapterLuid, ...], tuple[dict[str, Any], ...]
]:
    """The survey, split into engine observations, adapter LUIDs and display adapters."""
    engines: list[EngineObservation] = []
    for row in _as_rows(payload.get("engines")):
        name = str(row.get("Name") or "").strip()
        if not name:
            continue
        try:
            parsed = parse_instance_name(name)
        except ValueError:
            # Recorded by the caller as a parse failure rather than dropped silently.
            engines.append(EngineObservation(
                instance=name, pid=None, luid="", engtype=None,
                utilization_percent=_number(row.get("UtilizationPercentage")),
                running_time=_number(row.get("RunningTime")),
            ))
            continue
        engines.append(EngineObservation(
            instance=name,
            pid=parsed.pid,
            luid=parsed.luid,
            engtype=_engine_type(parsed.engtype),
            utilization_percent=_number(row.get("UtilizationPercentage")),
            running_time=_number(row.get("RunningTime")),
        ))

    adapters: list[AdapterLuid] = []
    for row in _as_rows(payload.get("adapters")):
        name = str(row.get("Name") or "").strip()
        # Measured: this object's instance names are `luid_0x<high>_0x<adapter>_phys_<n>`
        # with no `pid_` prefix, so the needle is `luid_` and not `_luid_`.
        if not name or "luid_" not in name.lower():
            continue
        try:
            parsed = parse_instance_name(
                name if name.lower().startswith("pid_") else f"pid_0_{name}"
            )
        except ValueError:
            continue
        adapters.append(AdapterLuid(luid=parsed.luid, instance_name=name))

    video = tuple(
        {
            "name": str(row.get("Name") or "").strip(),
            "pnp_device_id": str(row.get("PNPDeviceID") or "").strip(),
        }
        for row in _as_rows(payload.get("video"))
    )
    return tuple(engines), tuple(adapters), video


@dataclass
class LuidAttribution:
    """Why one LUID was assigned to one device, or why it was left unattributed."""

    luid: str
    device: str | None
    method: str
    reason: str
    engine_types: tuple[str, ...] = ()
    instance_count: int = 0
    max_utilization_percent: float | None = None


def resolve_luid_devices(
    engines: Sequence[EngineObservation],
) -> tuple[LuidAttribution, ...]:
    r"""Assign each observed LUID to a device, or leave it unattributed.

    The rule is stated rather than implied, and the evidence for each decision is kept:

    - **NPU** - the LUID exposes a `Neural` engine. The neural engine is the NPU's, and
      measured on this machine it is the only device that has one.
    - **GPU** - the LUID exposes a `Compute`, `VideoDecode`, `VideoProcessing`, `Copy` or
      `GSC` engine. `3D` alone is *not* enough: a third adapter on this machine exposes
      `3D` and nothing else, and calling it the GPU would be a real number on the wrong
      card.
    - **unattributed** - anything else. Recorded with its engine inventory so the report
      shows it exists and was not swept up.

    A LUID can only ever get one device here, and the result is refused outright if two
    different rules would claim the same LUID, which is the same invariant
    `app.telemetry.instance_map.InstanceMap` enforces on the recorded file.
    """
    by_luid: dict[str, list[EngineObservation]] = {}
    for observation in engines:
        if not observation.luid:
            continue
        by_luid.setdefault(observation.luid, []).append(observation)

    attributions: list[LuidAttribution] = []
    claimed: dict[str, str] = {}
    for luid in sorted(by_luid):
        rows = by_luid[luid]
        present = {r.engtype for r in rows if r.engtype}
        readings = [
            r.utilization_percent for r in rows if r.utilization_percent is not None
        ]
        npu_evidence = sorted(present & NPU_ENGINES)
        gpu_evidence = sorted(present & GPU_ENGINES)

        device: str | None = None
        method = "engine_taxonomy"
        if npu_evidence and gpu_evidence:
            # Refuse rather than pick. Two rules disagreeing is a data problem to report.
            raise ValueError(
                f"LUID {luid} exposes both NPU engine(s) {npu_evidence} and GPU "
                f"engine(s) {gpu_evidence}. One adapter cannot be both, so the engine "
                f"evidence is contradictory and the mapping is refused rather than "
                f"guessed."
            )
        elif npu_evidence:
            device = "NPU"
            reason = (
                f"exposes NPU engine(s) {npu_evidence}; the neural engine is the NPU's "
                f"engine and no other device on this machine has one"
            )
        elif gpu_evidence:
            device = "GPU"
            reason = (
                f"exposes GPU engine(s) {gpu_evidence}; '3D' is not in this set because "
                f"a third adapter on this machine exposes '3D' and nothing else"
            )
        else:
            method = "unattributed"
            reason = (
                f"exposes only {sorted(present) or ['no engine type']}, which identifies "
                f"neither the NPU nor the GPU; left unattributed rather than assigned to "
                f"the nearest device"
            )
        if device is not None:
            if luid in claimed and claimed[luid] != device:
                raise ValueError(
                    f"LUID {luid} is claimed by both {claimed[luid]!r} and {device!r}"
                )
            claimed[luid] = device
        attributions.append(LuidAttribution(
            luid=luid,
            device=device,
            method=method,
            reason=reason,
            engine_types=tuple(sorted(present)),
            instance_count=len(rows),
            max_utilization_percent=max(readings) if readings else None,
        ))
    return tuple(attributions)


def _windows_build(payload: dict[str, Any]) -> str:
    os_rows = _as_rows(payload.get("os"))
    if not os_rows:
        return "unknown"
    row = os_rows[0]
    return " ".join(
        str(row.get(key) or "").strip()
        for key in ("Caption", "Version", "BuildNumber")
        if str(row.get(key) or "").strip()
    ) or "unknown"


def _engine_rules(
    engines: Sequence[EngineObservation],
    attributions: Sequence[LuidAttribution],
) -> list[dict[str, Any]]:
    """The distinct `(luid, device, engtype)` triples, sorted.

    One row per *engine type per adapter*, not one row per instance: 456 instances
    collapsing to 7 rules is the useful record, and a repeated rule is noise that makes a
    reviewer stop reading. A LUID with no attribution contributes no rule, because a rule
    with no device would attribute an adapter to nothing.
    """
    device_of = {a.luid: a.device for a in attributions if a.device}
    triples = {
        (observation.luid, device_of[observation.luid], observation.engtype)
        for observation in engines
        if observation.luid and observation.engtype
        and observation.luid in device_of
    }
    return [
        {"luid": luid, "device": device, "engtype": engtype}
        for luid, device, engtype in sorted(triples, key=lambda row: (row[1], row[0], row[2]))
    ]


def _relevant_driver_rows() -> dict[str, Any]:
    """The PnP rows for the two demo devices, plus how many rows exist in total.

    Recorded from the one memoised enumeration Task 2 already owns, not a second PnP walk
    in this tool. Only the GPU and the NPU rows are kept because those are the two devices
    the map attributes; the other 25 rows are counted rather than copied, so the file
    stays readable and the count is still checkable.
    """
    rows = _driver_rows()
    wanted = ("Arc", "AI Boost")
    return {
        "rows_total": len(rows),
        "demo_devices": [row for row in rows if any(k in row["name"] for k in wanted)],
    }


def _driver_rows() -> list[dict[str, Any]]:
    """The PnP driver table, via the one memoised enumeration Task 2 already owns.

    Not re-implemented here on purpose: a second PnP enumeration would be a second source
    of truth for the same fact, and this tool must not become one.
    """
    lookup = WindowsDriverLookup()
    return [
        {
            "device_id": driver.device_id,
            "name": driver.name,
            "pci_device_id": driver.pci_device_id,
            "driver_version": driver.driver_version,
        }
        for driver in lookup.drivers
    ]


def build_map(
    engines: Sequence[EngineObservation],
    adapters: Sequence[AdapterLuid],
    video: Sequence[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """The recorded `config/telemetry_map.json` payload, built only from what was read."""
    attributions = resolve_luid_devices(engines)
    observed_luids = {a.luid for a in attributions}
    adapter_luids = {a.luid for a in adapters}

    npu_luids = {a.luid for a in attributions if a.device == "NPU"}
    gpu_luids = {a.luid for a in attributions if a.device == "GPU"}
    npu_neural = sorted({
        o.engtype for o in engines
        if o.luid in npu_luids and o.engtype in NPU_ENGINES and o.engtype
    })

    by_luid_devices: dict[str, set[str]] = {}
    for attribution in attributions:
        if attribution.device:
            by_luid_devices.setdefault(attribution.luid, set()).add(attribution.device)
    collisions = {luid: sorted(d) for luid, d in by_luid_devices.items() if len(d) > 1}
    if collisions:
        raise ValueError(
            f"one LUID mapped to two devices: {collisions}. Refusing to write a map that "
            f"would put a real but misattributed number on a device card."
        )

    return {
        "schema_version": 1,
        "probed_on": date.today().isoformat(),
        "windows_build": _windows_build(payload),
        "counter_source": {
            "category": CATEGORY,
            "counter": COUNTER,
            "read_via": f"CIM {ENGINE_WMI_CLASS}",
            "instances_observed": len(engines),
            "adapters_observed": len(adapters),
            "note": (
                "The runtime does not use this path. It reads "
                "config/telemetry_map.json and matches instance names by LUID; see "
                "app/telemetry/pdh.py for why the ctypes value read is not usable on "
                "this machine."
            ),
        },
        "direct_luid_link": {
            "available": False,
            "reason": (
                "Measured on this machine: no CIM or Win32 source exposes a LUID "
                "together with a PCI device id. Win32_PerfFormattedData_"
                "GPUPerformanceCounters_GPUAdapterMemory carries only Name and memory "
                "counters; Win32_VideoController carries PNPDeviceID but no LUID; "
                "EnumDisplayDevicesW returns PCI\\VEN_8086&DEV_64A0&SUBSYS_88F21043&"
                "REV_04 for all three display adapter entries and no LUID; "
                "Win32_LUID exposes no instances and MSFT_ComputeAccelerator is not a "
                "registered CIM class on this build. The mapping is therefore resolved "
                "from the engine taxonomy, with the evidence recorded per LUID."
            ),
        },
        "display_adapters": [dict(row) for row in video],
        "pnp_drivers": _relevant_driver_rows(),
        "luids": [
            {"luid": attribution.luid, "device": attribution.device}
            for attribution in attributions if attribution.device
        ],
        "engines": _engine_rules(engines, attributions),
        "unattributed_luids": [
            {
                "luid": attribution.luid,
                "engine_types": list(attribution.engine_types),
                "instance_count": attribution.instance_count,
                "reason": attribution.reason,
            }
            for attribution in attributions if attribution.device is None
        ],
        "engine_types_by_device": {
            device: sorted({
                o.engtype for o in engines
                if o.luid in luids and o.engtype
            })
            for device, luids in (("NPU", npu_luids), ("GPU", gpu_luids))
        },
        "npu_neural_counter": bool(npu_neural),
        "npu_neural_engine_types": npu_neural,
        "adapters_without_engine_instances": sorted(adapter_luids - observed_luids),
        "attributions": [
            {
                "luid": a.luid,
                "device": a.device,
                "method": a.method,
                "reason": a.reason,
                "engine_types": list(a.engine_types),
                "instance_count": a.instance_count,
                "max_observed_utilization_percent": a.max_utilization_percent,
            }
            for a in attributions
        ],
    }


def _print_instances(engines: Sequence[EngineObservation]) -> None:
    print("--- every GPU Engine instance enumerated, verbatim ---")
    print(f"{'INSTANCE':<78} {'UTIL%':>9}  {'RUNNING':>12}")
    for observation in engines:
        util = observation.utilization_percent
        util_text = f"{util:.4f}" if util is not None else "unavailable"
        run = observation.running_time
        run_text = f"{run:.4f}" if run is not None else "unavailable"
        print(f"{observation.instance:<78} {util_text:>9}  {run_text:>12}")
    print(f"--- {len(engines)} instances ---\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Probe, record, and fail closed. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate the Windows GPU Engine counters, resolve each adapter LUID to a "
            "device, and write config/telemetry_map.json."
        )
    )
    parser.add_argument(
        "--allow-missing-npu", action="store_true",
        help=(
            "Exit 0 even when no NPU 'Neural' engine counter was found. The map is still "
            "written with the NPU absent, npu_neural_counter false, and the runtime's "
            "NPU utilisation null. This flag changes the exit code and nothing else."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("config/telemetry_map.json"),
        help="where to write the recorded map (default: config/telemetry_map.json)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    print("=" * 100)
    print("PDH telemetry probe - GPU Engine / Utilization Percentage")
    print("=" * 100)
    try:
        payload = survey()
    except Exception as exc:  # noqa: BLE001 - the failure text is the deliverable
        print(f"FATAL: the CIM survey failed: {type(exc).__name__}: {exc}")
        print("Nothing was written. A probe that could not look cannot record an answer.")
        return 2

    engines, adapters, video = parse_survey(payload)
    unparsed = [o for o in engines if not o.luid]
    print(f"windows build        : {_windows_build(payload)}")
    print(f"counter source       : CIM {ENGINE_WMI_CLASS}")
    print(f"instances enumerated : {len(engines)}")
    print(f"adapter LUIDs        : {len(adapters)}")
    print(f"display adapters     : {[row['name'] for row in video]}")
    print(f"instance names that could not be parsed: {len(unparsed)}")
    for row in unparsed:
        print(f"    UNPARSED {row.instance!r}")
    print()

    if not engines:
        print("FATAL: no 'GPU Engine' instances existed at probe time.")
        print(
            "This is 'nothing was running', not 'the NPU has no Neural engine'. No map was "
            "written, because recording absence from an empty observation would be a "
            "fabrication. Start a GPU and NPU workload (for example tools/qualify_m0.py) "
            "and run this again."
        )
        return 2

    _print_instances(engines)

    print("--- adapter LUID -> device, with the evidence for each decision ---")
    try:
        attributions = resolve_luid_devices(engines)
    except ValueError as exc:
        print(f"FATAL: {exc}")
        print("Nothing was written.")
        return 2
    for attribution in attributions:
        util = attribution.max_utilization_percent
        util_text = f"{util:.2f}%" if util is not None else "no reading"
        print(f"  {attribution.luid}  ->  {attribution.device or 'UNATTRIBUTED'}")
        print(f"      method            : {attribution.method}")
        print(f"      engine types seen : {list(attribution.engine_types)}")
        print(f"      instances         : {attribution.instance_count}")
        print(f"      max utilisation   : {util_text}")
        print(f"      reason            : {attribution.reason}")
    print()

    print("--- engine types found per device ---")
    by_device: dict[str, set[str]] = {}
    for attribution in attributions:
        if attribution.device:
            by_device.setdefault(attribution.device, set()).update(attribution.engine_types)
    for device in sorted(by_device):
        print(f"  {device}: {sorted(by_device[device])}")
    unattributed = [a for a in attributions if not a.device]
    if unattributed:
        print("  (unattributed, deliberately not assigned):")
        for attribution in unattributed:
            print(f"    {attribution.luid}: {list(attribution.engine_types)}")
    print()

    try:
        recorded = build_map(engines, adapters, video, payload)
    except ValueError as exc:
        print(f"FATAL: {exc}")
        print("Nothing was written.")
        return 2

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(recorded, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {target}")
    print()

    npu_neural = recorded["npu_neural_counter"]
    print("--- NPU counter verdict ---")
    if npu_neural:
        print(
            f"  FOUND: an 'engtype {recorded['npu_neural_engine_types']}' engine exists "
            f"for the NPU LUID. The NPU card can show real utilisation."
        )
    else:
        print(
            "  NOT FOUND: no NPU 'Neural' engine counter was observed. This is recorded "
            "honestly rather than substituted: npu_percent stays null and the NPU card "
            "reads 'Utilization unavailable'. A booth visitor must never be shown "
            "'NPU 0%' when the truth is that nothing was measured."
        )
    if not args.allow_missing_npu and not npu_neural:
        print(
            "\nEXIT NON-ZERO: no NPU Neural engine counter, and --allow-missing-npu was "
            "not passed. Re-run with --allow-missing-npu to record the negative result "
            "deliberately."
        )
        return 1
    print()
    print("PROBE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
