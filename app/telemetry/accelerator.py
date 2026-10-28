"""Per-process GPU and NPU utilisation from the Windows `GPU Engine` counter set.

**The claim this module is allowed to make, and the one it refuses.**

Windows reports `\\GPU Engine(*)\\Utilization Percentage` once per **engine per process**.
The instance name carries the process and the adapter:

    pid_6588_luid_0x00000000_0x000113B4_phys_0_eng_5_engtype_Compute

so a figure filtered to this application's pids is a figure about this application, not
about the machine. Measured on this machine on 2026-10-27, with an OpenVINO worker doing
GPU inference in its own process: the worker's `engtype_Compute` instance read a median
**97.19%** (97.06-100.0%) while idle it read **0.0%**, while every other process on the
machine read below 3% (the busiest foreign rows measured 1.42% and 2.79%). With an NPU
worker: its `engtype_Neural` instance read a
median **64.86%** (63.76-66.08%), 0.0% when idle, and was the busiest instance on the
machine. Those are the controls that make the figure attributable; see
`docs/qualification/M4-utilisation-attribution.md`.

**What "100%" means here.** `Utilization Percentage` is the fraction of the interval
between two counter samples during which that engine was busy. It is **per engine**. This
provider reports the **busiest engine** of the device for the application's processes:
"at least one engine of this device was busy for this process for X% of the interval".
It never sums engines, because two engines can be busy at once and a sum would exceed 100
and stop meaning "time"; it never sums processes into a machine-wide figure. The scope
travels on the `MetricSample` and is stated on the card.

**The device is identified, not inferred.** The GPU is the DXGI hardware adapter
(`app.telemetry.dxgi`: Intel Arc, PCI `8086:64A0`, LUID `0x000113B4` on this boot), which
excludes Microsoft's Basic Render Driver - the software renderer that reports `3D` engine
instances for dozens of unrelated processes. The NPU is the one adapter LUID that exposes
an `engtype_Neural` engine. The two answers are independent: a GPU figure is never used to
claim anything about the NPU, and a machine where either is ambiguous reads `None` with the
reason, never a nearest match.

**`None` is never `0`.** A process that has no engine instance on a device has not been
observed using it, and the card says so; a process with an instance at 0.0% has a measured
zero and that is published as `0.0`. The distinction is the whole point of
`app.contracts.metric_sample`.

**No estimation, no interpolation, no history.** Each sample is one
`PdhCollectQueryData` + one `PdhGetFormattedCounterArrayW`. Nothing is carried forward: an
unreadable sample is `None` with a reason.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from app.contracts.metric_sample import MetricSample, Validity
from app.telemetry import dxgi
from app.telemetry.instance_map import parse_instance_name
from app.telemetry.pdh import (
    GPU_ENGINE_COUNTER,
    GPU_ENGINE_COUNTER_SET,
    PRIME_SECONDS,
    PdhQuery,
)

__all__ = [
    "GPU_METRIC",
    "NPU_METRIC",
    "PROVIDER",
    "AcceleratorSnapshot",
    "AcceleratorUtilisation",
    "DeviceSplit",
    "resolve_devices",
]

PROVIDER: Final[str] = "pdh_gpu_engine"
GPU_METRIC: Final[str] = "gpu_utilisation_percent"
NPU_METRIC: Final[str] = "npu_utilisation_percent"

#: The engine type that identifies the NPU. Measured on this machine: only the adapter
#: whose LUID ends `0x000117B6` exposes `engtype_Neural`; the Arc exposes
#: 3D/Compute/Copy/GSC/VideoDecode/VideoProcessing and the software renderer 3D only.
NEURAL_ENGINE: Final[str] = "Neural"

#: Card-sized reasons. Each names a *fact about this sample*, not a guess.
GPU_LUID_UNKNOWN_REASON = (
    "dxgi.dll did not report exactly one hardware display adapter, so this process's GPU "
    "engine instances cannot be attributed to the Arc. Utilization is unavailable, not 0."
)
NPU_LUID_UNKNOWN_REASON = (
    "no adapter exposes an engtype_Neural engine in this counter sample, so the NPU's LUID "
    "is unknown. Utilization is unavailable, not 0."
)
NPU_AMBIGUOUS_REASON = (
    "more than one adapter exposes an engtype_Neural engine in this sample, so the NPU "
    "cannot be identified without guessing. Utilization is unavailable, not 0."
)
NO_READING_REASON = (
    "the GPU Engine counter returned no reading for any instance in this sample: no "
    "process on the machine has an engine instance yet, or the query is unprimed. "
    "Unavailable, not 0."
)


def _absent_reason(device: str) -> str:
    return (
        f"this application has no {device} engine instance in this counter sample: Windows "
        f"creates an instance only for a process that has opened an engine on that "
        f"adapter. No instance is not a measured 0."
    )


@dataclass(frozen=True)
class DeviceSplit:
    """Which adapter LUID is the GPU, which is the NPU, and why when they are unknown."""

    gpu_luid: str | None
    npu_luid: str | None
    gpu_reason: str | None
    npu_reason: str | None
    adapters: tuple[dxgi.DxgiAdapter, ...] = ()


def resolve_devices(
    rows: Iterable[tuple[str, float]],
    adapters: Sequence[dxgi.DxgiAdapter] | None = None,
) -> DeviceSplit:
    """Resolve `(GPU LUID, NPU LUID)` from the live instance names and DXGI.

    GPU: exactly one **hardware** DXGI adapter. Zero or several is `None` with a reason;
    the software renderer is never a candidate.
    NPU: exactly one LUID exposing `engtype_Neural`. Zero or several is `None` with a
    reason. The NPU is not a DXGI adapter, so the two rules are independent.
    """
    pool = tuple(adapters) if adapters is not None else dxgi.adapters()
    hardware = dxgi.hardware_adapters(pool)
    gpu_luid: str | None = None
    gpu_reason: str | None = None
    if len(hardware) == 1:
        gpu_luid = hardware[0].luid
    else:
        gpu_reason = GPU_LUID_UNKNOWN_REASON

    neural_luids: list[str] = []
    for name, _value in rows:
        try:
            parsed = parse_instance_name(name)
        except ValueError:
            continue
        if parsed.engtype == NEURAL_ENGINE and parsed.luid not in neural_luids:
            neural_luids.append(parsed.luid)

    npu_luid: str | None = None
    npu_reason: str | None = None
    if len(neural_luids) == 1:
        npu_luid = neural_luids[0]
    elif len(neural_luids) > 1:
        npu_reason = NPU_AMBIGUOUS_REASON
    else:
        npu_reason = NPU_LUID_UNKNOWN_REASON
    return DeviceSplit(gpu_luid=gpu_luid, npu_luid=npu_luid,
                       gpu_reason=gpu_reason, npu_reason=npu_reason, adapters=pool)


@dataclass(frozen=True)
class DeviceReading:
    """One device's figure for this application, with the evidence behind it."""

    percent: float | None
    reason: str | None
    engine: str | None
    engines_considered: int = 0


def _read_device(rows: Sequence[tuple[str, float]], pids: frozenset[int],
                 luid: str | None, engtype: str | None) -> DeviceReading:
    """The **busiest** engine's percentage for these pids on this adapter.

    Percentages are already per-instance and in `[0, 100]` (`PdhQuery.read_array` refused
    anything else), so the maximum is a measured value, not a derived one.
    """
    if luid is None:
        return DeviceReading(None, None, None, 0)
    best: tuple[str, float] | None = None
    considered = 0
    for name, percent in rows:
        try:
            parsed = parse_instance_name(name)
        except ValueError:
            continue
        if parsed.luid != luid or parsed.pid not in pids:
            continue
        if engtype is not None and parsed.engtype != engtype:
            continue
        considered += 1
        if best is None or percent > best[1]:
            best = (name, percent)
    if best is None:
        return DeviceReading(None, None, None, 0)
    return DeviceReading(best[1], None, best[0], considered)


@dataclass(frozen=True)
class AcceleratorSnapshot:
    """One sample's rows and readings, kept for the probe and the report."""

    rows: tuple[tuple[str, float], ...]
    split: DeviceSplit
    gpu: DeviceReading
    npu: DeviceReading
    window_seconds: float


class AcceleratorUtilisation:
    """Samples this application's own GPU and NPU utilisation on the telemetry thread.

    `pids` is the application's process set - the UI process and the worker pids. A figure
    is "this application's" because the instance's own `pid_` field must be in that set:
    there is no code path that reads an instance belonging to another process.

    `query` and `adapters` are injectable so the decision logic is testable without a
    native call; the defaults are a real `PdhQuery` over `GPU Engine` and a real DXGI
    enumeration. Construction primes the rate counter once (`prime_seconds`), so the first
    `samples()` call has a window behind it rather than an unprimed `PDH_INVALID_DATA`.

    `samples()` never raises: a counter that cannot be read becomes two `UNAVAILABLE`
    samples with the measured reason, and the rest of the telemetry keeps running
    (AGENTS.md rule 12).
    """

    def __init__(
        self,
        pids: Sequence[int],
        *,
        labels: Sequence[str] | None = None,
        query: object | None = None,
        adapters: Sequence[dxgi.DxgiAdapter] | None = None,
        prime_seconds: float = PRIME_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        wanted = tuple(int(pid) for pid in pids)
        if not wanted:
            raise ValueError(
                "AcceleratorUtilisation needs at least one pid: a utilisation figure with "
                "no process to attribute it to is a machine-wide figure, which this "
                "provider must never produce."
            )
        if any(pid <= 0 for pid in wanted):
            raise ValueError(f"pids must be positive; got {wanted!r}")
        self._pids = frozenset(wanted)
        name_labels = (tuple(str(label) for label in labels)
                       if labels is not None else tuple(f"pid_{p}" for p in wanted))
        if len(name_labels) != len(wanted):
            raise ValueError(
                f"{len(wanted)} pids but {len(name_labels)} labels; every process this "
                f"figure covers has to be nameable"
            )
        self._labels = name_labels
        self._clock = clock
        self._query = (query if query is not None else PdhQuery(
            GPU_ENGINE_COUNTER_SET, {"u": GPU_ENGINE_COUNTER},
            prime_seconds=prime_seconds,
        ))
        self._adapters = tuple(adapters) if adapters is not None else dxgi.adapters()
        self._last_at = float(self._clock())
        self._lock = threading.Lock()
        self._last_snapshot: AcceleratorSnapshot | None = None

    # -- measurement ----------------------------------------------------------------

    @property
    def pids(self) -> frozenset[int]:
        return self._pids

    def snapshot(self) -> AcceleratorSnapshot:
        """One sample: rows, device split and both readings. **Call from one thread.**

        The lock makes two samplers safe rather than correct: `Utilization Percentage` is a
        rate over the interval between collects, so two callers at 4 Hz halve the window.
        The application drives this from the collector's single telemetry thread.
        """
        with self._lock:
            try:
                rows = tuple(self._query.read_array())  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - an absent reading is a result
                rows = ()
                failure = (
                    f"the GPU Engine counter could not be read: "
                    f"{type(exc).__name__}: {exc}. Utilization is unavailable, not 0."
                )
            else:
                failure = None
            now = self._clock()
            window = max(0.0, now - self._last_at)
            self._last_at = now
            if failure is not None:
                empty = DeviceReading(None, failure, None, 0)
                split = DeviceSplit(None, None, failure, failure, self._adapters)
                snapshot = AcceleratorSnapshot(rows, split, empty, empty, window)
            elif not rows:
                empty = DeviceReading(None, NO_READING_REASON, None, 0)
                split = DeviceSplit(None, None, NO_READING_REASON, NO_READING_REASON,
                                    self._adapters)
                snapshot = AcceleratorSnapshot(rows, split, empty, empty, window)
            else:
                split = resolve_devices(rows, self._adapters)
                gpu = _read_device(rows, self._pids, split.gpu_luid, None)
                npu = _read_device(rows, self._pids, split.npu_luid, NEURAL_ENGINE)
                if gpu.percent is None:
                    gpu = DeviceReading(None, split.gpu_reason or _absent_reason("GPU"),
                                        None, 0)
                if npu.percent is None:
                    npu = DeviceReading(None, split.npu_reason or _absent_reason("NPU"),
                                        None, 0)
                snapshot = AcceleratorSnapshot(rows, split, gpu, npu, window)
            self._last_snapshot = snapshot
            return snapshot

    @property
    def last_snapshot(self) -> AcceleratorSnapshot | None:
        return self._last_snapshot

    def samples(self) -> tuple[MetricSample, MetricSample]:
        """The GPU and NPU `MetricSample`s for one sample. Never raises."""
        snapshot = self.snapshot()
        return (
            self._sample(GPU_METRIC, "GPU", snapshot.gpu, snapshot.window_seconds),
            self._sample(NPU_METRIC, "NPU", snapshot.npu, snapshot.window_seconds),
        )

    def _sample(self, name: str, device: str, reading: DeviceReading,
                window: float) -> MetricSample:
        if reading.percent is not None and window <= 0.0:
            # Two samples inside the clock's resolution: there is a value but no interval
            # it was measured over, and `MetricSample` refuses a measured reading with a
            # zero window. Say so rather than inventing an interval.
            return MetricSample(
                name=name, value=None, unit="percent", provider=PROVIDER,
                device_scope=device, process_scope=self._labels,
                window_seconds=0.0, sampled_at=self._clock(),
                validity=Validity.UNAVAILABLE,
                unavailable_reason=(
                    "no measurable window elapsed since the previous accelerator sample, "
                    "so the percentage has no interval to be a percentage of. Measured "
                    "again on the next tick; this is not 0."
                ),
            )
        if reading.percent is None:
            return MetricSample(
                name=name, value=None, unit="percent", provider=PROVIDER,
                device_scope=device, process_scope=self._labels,
                window_seconds=0.0, sampled_at=self._clock(),
                validity=Validity.UNAVAILABLE,
                unavailable_reason=reading.reason or _absent_reason(device),
            )
        return MetricSample(
            name=name, value=float(reading.percent), unit="percent", provider=PROVIDER,
            device_scope=device, process_scope=self._labels,
            window_seconds=window, sampled_at=self._clock(),
            validity=Validity.VALID,
        )

    def close(self) -> None:
        close = getattr(self._query, "close", None)
        if callable(close):
            close()
