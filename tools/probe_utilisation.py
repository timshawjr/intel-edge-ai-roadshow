"""The attribution control for M4 Task 5: does the GPU/NPU figure track *this* process?

Run it with the pinned interpreter:

    .venv\\Scripts\\python.exe tools\\probe_utilisation.py

It answers three questions, with measurements rather than argument:

1. **Idle, then busy, in the same process.** A worker process is spawned, left idle for a
   few seconds, then driven with OpenVINO inference on `GPU` and (separately) `NPU`. The
   provider samples that **worker's pid** throughout. A figure that is this process's must
   be absent (or near zero) while the worker is idle and high while it is busy.
2. **Another busy process is not attributed to us.** During the same busy phase, a second
   provider samples **this probe's own pid**, which has no engine instance. The snapshot's
   rows contain the worker's high instance; the probe's own reading must be absent. A
   machine-wide peak would read high here, and that is the failure this control catches.
3. **The machine was clean first.** A baseline snapshot is taken before any worker starts;
   the busiest engine on the machine must be below `CLEAN_PERCENT`, or the probe says the
   machine is not clean and exits non-zero rather than measuring through noise.

The JSON it writes is the evidence quoted in
`docs/qualification/M4-utilisation-attribution.md`. It writes no other file and changes no
configuration.

**No new dependency.** `openvino` is pinned (`requirements.txt`), and the model is the one
already provisioned under `assets/models/yolo11n-fp16`.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.inference.pinned import REPO_ROOT as PINNED_ROOT  # noqa: E402
from app.telemetry.accelerator import AcceleratorUtilisation  # noqa: E402
from app.telemetry.dxgi import adapters as dxgi_adapters  # noqa: E402
from app.telemetry.pdh import GPU_ENGINE_COUNTER, GPU_ENGINE_COUNTER_SET, PdhQuery  # noqa: E402

MODEL_XML = PINNED_ROOT / "assets" / "models" / "yolo11n-fp16" / "yolo11n.xml"
OUTPUT = PINNED_ROOT / "docs" / "qualification" / "m4-utilisation-control.json"

#: The busiest engine on an otherwise idle machine must be below this before the control
#: starts. Measured baseline on this machine before any worker: 0.00-3.51% (the desktop
#: compositor and this editor). 10% leaves room for a redraw without accepting a GPU
#: workload as "idle".
CLEAN_PERCENT = 10.0

#: A busy phase must read at least this much for the figure to be called attributable.
#: Measured on the committed control: GPU worker 97.06-100.0% (median 97.19), NPU worker
#: 63.76-66.08% (median 64.86). The threshold is deliberately well below both, because it
#: is a control, not a performance claim.
BUSY_FLOOR = {"GPU": 40.0, "NPU": 30.0}

#: Samples this close to a phase change are discarded: the counter's window straddles both
#: phases, and including them would blur exactly the movement being proved.
TRANSITION_SECONDS = 1.0


def _worker(device: str, idle_seconds: float, busy_seconds: float) -> int:
    """Child process: compile on `device`, stay idle, then run inference. Signals phases."""
    import numpy as np
    import openvino as ov

    print(f"PID={os.getpid()}", flush=True)
    core = ov.Core()
    started = time.perf_counter()
    model = core.compile_model(str(MODEL_XML), device)
    print(f"COMPILED {time.perf_counter() - started:.2f}s", flush=True)
    shape = [1 if (d is None or isinstance(d, str)) else int(d) for d in model.input(0).shape]
    tensor = np.zeros(shape, dtype=np.float32)

    print("IDLE", flush=True)
    time.sleep(max(0.0, idle_seconds))

    print("BUSY", flush=True)
    deadline = time.perf_counter() + max(0.0, busy_seconds)
    count = 0
    while time.perf_counter() < deadline:
        model(tensor)
        count += 1
    print(f"DONE inferences={count}", flush=True)
    return 0


def _read_rows() -> tuple[tuple[str, float], ...]:
    query = PdhQuery(GPU_ENGINE_COUNTER_SET, {"u": GPU_ENGINE_COUNTER})
    try:
        return tuple(query.read_array())
    finally:
        query.close()


def _engine_numbers(rows: tuple[tuple[str, float], ...]) -> dict[str, float]:
    """`pid_<n> -> that pid's busiest engine`, for naming the baseline's busy processes."""
    import psutil

    by_pid: dict[int, float] = {}
    for name, percent in rows:
        try:
            pid = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        by_pid[pid] = max(by_pid.get(pid, 0.0), percent)
    labelled: dict[str, float] = {}
    for pid, percent in by_pid.items():
        try:
            label = psutil.Process(pid).name()
        except Exception:  # noqa: BLE001 - a vanished process is not an error here
            label = "pid"
        labelled[f"{label}({pid})"] = round(percent, 3)
    return dict(sorted(labelled.items(), key=lambda kv: -kv[1])[:8])


def _machine_conditions() -> dict:
    import platform

    conditions = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu_driver": None,
        "npu_driver": None,
        "adapters": [
            {"description": a.description, "vendor_id": f"0x{a.vendor_id:04X}",
             "device_id": f"0x{a.device_id:04X}", "luid": a.luid, "software": a.software}
            for a in dxgi_adapters()
        ],
    }
    try:
        import openvino as ov

        from app.inference.devices import DeviceRegistry

        for status in DeviceRegistry(ov.Core()).describe():
            if status.device_id.upper().startswith("GPU"):
                conditions["gpu_driver"] = conditions["gpu_driver"] or status.driver_version
            elif status.device_id.upper().startswith("NPU"):
                conditions["npu_driver"] = status.driver_version
    except Exception as exc:  # noqa: BLE001 - a report, not a gate on this probe
        conditions["driver_error"] = f"{type(exc).__name__}: {exc}"
    return conditions


class _PhaseReader(threading.Thread):
    """Read the worker's stdout, recording when each phase line arrived (monotonic)."""

    def __init__(self, stream) -> None:
        super().__init__(daemon=True, name="phase-reader")
        self._stream = stream
        self.events: list[tuple[float, str]] = []
        self._lock = threading.Lock()
        self.pid: int | None = None

    def run(self) -> None:
        for line in self._stream:
            text = line.strip()
            if not text:
                continue
            with self._lock:
                self.events.append((time.monotonic(), text))
                if text.startswith("PID="):
                    self.pid = int(text[4:])

    def at(self, prefix: str) -> float | None:
        with self._lock:
            for when, text in self.events:
                if text.startswith(prefix):
                    return when
        return None

    def wait_for(self, prefix: str, timeout: float) -> float | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            when = self.at(prefix)
            if when is not None:
                return when
            time.sleep(0.05)
        return None


def _run_phase(device: str, idle_seconds: float, busy_seconds: float,
               machine: dict) -> dict:
    """One idle-then-busy worker, sampled by a provider on that worker's own pid."""
    started = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker", device,
         str(idle_seconds), str(busy_seconds)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    reader = _PhaseReader(process.stdout)
    reader.start()
    # Compilation is excluded from the phases: it is GPU/NPU work of its own and would
    # otherwise land in the "idle" window.
    if reader.wait_for("COMPILED", timeout=180.0) is None:
        process.kill()
        stderr = process.stderr.read() if process.stderr else ""
        return {"device": device, "error": f"worker did not compile: {stderr[-2000:]}"}
    pid = reader.pid
    if pid is None:
        pid = process.pid

    provider = AcceleratorUtilisation([pid], labels=("worker",))
    other = AcceleratorUtilisation([os.getpid()], labels=("probe",))
    samples: list[dict] = []
    rows_at_busy: tuple[tuple[str, float], ...] = ()
    try:
        while process.poll() is None:
            snapshot = provider.snapshot()
            mine = other.snapshot()
            at = time.monotonic()
            busy_at = reader.at("BUSY")
            idle_at = reader.at("IDLE")
            phase = ("compiling" if idle_at is None else
                     ("busy" if busy_at is not None and at >= busy_at else "idle"))
            if phase == "busy" and not rows_at_busy:
                import psutil

                reading = snapshot.gpu.percent if device == "GPU" else snapshot.npu.percent
                if (psutil.pid_exists(pid) and reading is not None
                        and reading >= BUSY_FLOOR[device]):
                    rows_at_busy = snapshot.rows
            samples.append({
                "at": round(at - started, 3),
                "phase": phase,
                "gpu_percent": snapshot.gpu.percent,
                "npu_percent": snapshot.npu.percent,
                "probe_pid_gpu_percent": mine.gpu.percent,
                "worker_rows": sum(
                    1 for name, _ in snapshot.rows if name.startswith(f"pid_{pid}_")),
            })
            time.sleep(0.5)
    finally:
        provider.close()
        other.close()
    process.wait(timeout=30)
    stderr = (process.stderr.read() if process.stderr else "").strip()
    busy_at = reader.at("BUSY")
    done_at = reader.at("DONE")
    if busy_at is None:
        return {"device": device, "error": f"worker never reported BUSY: {stderr[-2000:]}"}
    if process.returncode != 0:
        return {"device": device,
                "error": f"worker exited {process.returncode}: {stderr[-2000:]}"}

    device_key = "gpu_percent" if device == "GPU" else "npu_percent"
    idle = [s[device_key] for s in samples
            if s["phase"] == "idle" and s["at"] >= 0
            and busy_at - started - s["at"] > TRANSITION_SECONDS]
    busy = [s[device_key] for s in samples
            if s["phase"] == "busy" and s["at"] - (busy_at - started) > TRANSITION_SECONDS
            and (done_at is None or s["at"] < done_at - started - TRANSITION_SECONDS)]
    probe_busy = [s["probe_pid_gpu_percent"] for s in samples
                  if s["phase"] == "busy"
                  and s["at"] - (busy_at - started) > TRANSITION_SECONDS]
    present_idle = [v for v in idle if v is not None]
    present_busy = [v for v in busy if v is not None]
    present_probe = [v for v in probe_busy if v is not None]

    def summarise(values: list[float]) -> dict:
        return ({
            "samples": len(values),
            "min": round(min(values), 2) if values else None,
            "median": round(statistics.median(values), 2) if values else None,
            "max": round(max(values), 2) if values else None,
        })

    top_rows = sorted(rows_at_busy, key=lambda r: -r[1])[:5]
    return {
        "device": device,
        "worker_pid": pid,
        "idle_seconds": idle_seconds,
        "busy_seconds": busy_seconds,
        "idle": {**summarise(present_idle),
                 "absent_samples": len(idle) - len(present_idle),
                 "all": [round(v, 2) for v in present_idle]},
        "busy": {**summarise(present_busy),
                 "absent_samples": len(busy) - len(present_busy),
                 "all": [round(v, 2) for v in present_busy]},
        "probe_own_pid_while_worker_busy": {
            **summarise(present_probe),
            "absent_samples": len(probe_busy) - len(present_probe),
            "note": ("the probe's own reading while another process was at "
                     f"{round(statistics.median(present_busy), 2) if present_busy else '?'}%; "
                     "0.0 or absent means the busy worker was not attributed to the probe, "
                     "which is what a machine-wide peak would do"),
        },
        "busiest_rows_while_working": [
            {"instance": name, "percent": round(percent, 2)} for name, percent in top_rows
        ],
        "sample_count": len(samples),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=3, metavar=("DEVICE", "IDLE", "BUSY"))
    parser.add_argument("--idle", type=float, default=5.0)
    parser.add_argument("--busy", type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.worker:
        device, idle, busy = args.worker
        return _worker(device, float(idle), float(busy))

    machine = _machine_conditions()
    print(f"machine: {machine['hostname']}  {machine['platform']}")
    print(f"GPU driver: {machine['gpu_driver']}   NPU driver: {machine['npu_driver']}")
    for adapter in machine["adapters"]:
        print(f"  adapter {adapter['luid']}  {adapter['description']}  "
              f"PCI {adapter['vendor_id']}:{adapter['device_id']}"
              f"{'  [software]' if adapter['software'] else ''}")

    print("\nbaseline: no worker started; the machine must be clean")
    baseline_rows = _read_rows()
    baseline_top = dict(sorted(((name, round(p, 2)) for name, p in baseline_rows),
                               key=lambda kv: -kv[1])[:8])
    busy_by_process = _engine_numbers(baseline_rows)
    busiest = max((p for _n, p in baseline_rows), default=0.0)
    clean = busiest < CLEAN_PERCENT
    print(f"  {len(baseline_rows)} live instances, busiest {busiest:.2f}% "
          f"-> {'CLEAN' if clean else 'NOT CLEAN'}")
    for label, percent in busy_by_process.items():
        print(f"    {percent:7.2f}%  {label}")
    if not clean:
        print(f"\nRefusing to measure: an engine is above {CLEAN_PERCENT:.0f}% before any "
              f"worker started, so an idle reading could not be told from the machine's "
              f"own load.")
        return 3

    phases = []
    for device in ("GPU", "NPU"):
        print(f"\n{device}: worker idle {args.idle:.0f}s, then busy {args.busy:.0f}s")
        if not MODEL_XML.is_file():
            print(f"  model not provisioned at {MODEL_XML}; run tools/provision_venv.py")
            return 4
        result = _run_phase(device, args.idle, args.busy, machine)
        phases.append(result)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            return 2
        idle = result["idle"]
        busy = result["busy"]
        probe = result["probe_own_pid_while_worker_busy"]
        print(f"  idle  n={idle['samples']} min={idle['min']} median={idle['median']} "
              f"max={idle['max']} absent={idle['absent_samples']}")
        print(f"  busy  n={busy['samples']} min={busy['min']} median={busy['median']} "
              f"max={busy['max']} absent={busy['absent_samples']}")
        print(f"  probe's own pid while that worker was busy: n={probe['samples']} "
              f"max={probe['max']} absent={probe['absent_samples']}")
        for row in result["busiest_rows_while_working"]:
            print(f"    {row['percent']:7.2f}%  {row['instance']}")

    verdict = []
    for result in phases:
        device = result["device"]
        busy = result["busy"]
        idle = result["idle"]
        ok_busy = busy["median"] is not None and busy["median"] >= BUSY_FLOOR[device]
        # Idle: either no instance at all (absence) or a measured near-zero. Never a
        # machine-wide figure, which would show the *other* processes' load here.
        ok_idle = idle["max"] is None or idle["max"] < CLEAN_PERCENT
        probe_free = result["probe_own_pid_while_worker_busy"]["max"] is None \
            or result["probe_own_pid_while_worker_busy"]["max"] < CLEAN_PERCENT
        passed = bool(ok_busy and ok_idle and probe_free)
        verdict.append({
            "device": device,
            "busy_tracks_this_process": ok_busy,
            "idle_is_absent_or_near_zero": ok_idle,
            "busy_worker_not_attributed_to_probe": probe_free,
            "passed": passed,
        })
        print(f"  verdict {device}: busy={ok_busy} idle={ok_idle} "
              f"distractor={probe_free} -> {'PASS' if passed else 'FAIL'}")

    payload = {
        "probe": "tools/probe_utilisation.py",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine": machine,
        "clean_before_measurement": {"busiest_percent": round(busiest, 2),
                                     "threshold": CLEAN_PERCENT, "clean": clean},
        "baseline_top_instances": baseline_top,
        "baseline_busiest_by_process": busy_by_process,
        "phases": phases,
        "verdict": verdict,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT}")
    all_passed = all(item["passed"] for item in verdict)
    print(f"VERDICT: {'attributed - the figure tracks this process' if all_passed else 'FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
