"""Sustained, paired device trials: the measurement a device comparison would need.

    .venv\\Scripts\\python.exe tools\\sustained_trials.py --seconds 20 --warmup 3 --repeats 3

Every device number this project published before M4 Task 4 was a **single-frame p50** -
CPU 20.99-21.16 ms, GPU 3.30-3.47 ms, NPU 5.92-5.99 ms, over 30 inferences each, as recorded
in `app/engine/worker.py`. Thirty inferences of one frame is a *burst*, not a throughput
claim, and it has been labelled as such everywhere it appears. This tool produces the
sustained measurement a device comparison needs - and, where the answer is that the
workload cannot discriminate, it says that instead of manufacturing a ranking.

## What is measured, and the three conditions

Each trial is one **fresh process** (a `Supervisor.stop()` leaves a worker behind once in a
while; a trial that inherited a previous trial's resident set would measure the leak Task 3
fixed rather than the device), on one device, for one condition:

| condition | feed | what a "frame" is | what it isolates |
|---|---|---|---|
| `booth` | 15.0 fps, the booth's analysis rate | one analysed clip frame through the real pipeline | whether the device delivers the booth's stream, with the frame age that implies |
| `source` | every source frame (~29.97 fps) | one analysed clip frame through the real pipeline | whether the device still delivers when asked for twice the booth's rate |
| `sustained` | as fast as the device will go | one `infer` call on the pinned model at 640x640, batch 1 | the accelerator's own sustained cost, with no decode and no queue |

`booth` and `source` run **the application's own pipeline** - `app.engine.source.ClipSource`,
`app.supervisor.runner.Supervisor`, `app.engine.tracker.Tracker` and
`app.engine.events.RetailEventMachine` - and fold results with `app.main._fold_result`, the
application's own counter fold. A trial whose tracker saw a different update sequence than
the booth would not be measuring the booth.

`sustained` is deliberately *not* the pipeline. At 640x640 the GPU does **307-310**
inferences a second and the NPU **246-249** on this machine (measured, 20 s each), while the
pipeline delivers at most ~30 (one per source frame), so a pipeline run at `source` rate is
capped by the *source* and a pipeline run at `booth` rate is capped by the 15 fps request.
**Neither can tell you what the device can do.** `sustained` feeds the device as fast as it
will accept work, which is the only condition on this machine where throughput is set by the
device rather than by the feed.

## The rules that make it a fair comparison

1. **Interleaved, with a rotating order.** Repeats are grouped into rounds; every device
   appears once per round; the order rotates by one each round
   (`CPU GPU NPU`, `GPU NPU CPU`, `NPU CPU GPU`). A drift in machine state - thermal,
   power, a background indexer - therefore lands on all three roughly equally instead of
   penalising whichever ran last. `interleaved_order` is the single definition and it is
   asserted, not described.
2. **The warm-up is identical and the compile is excluded.** The first frame is submitted
   alone and the tool waits for the first analysed result, so the compile is paid there and
   outside every window (measured: about 2 s more wall per NPU trial than per CPU or GPU one,
   with the same 20.01 s window and 3 s warm-up). Then the queue is drained until it is empty, and
   every trial feeds `--warmup` seconds before the measured window opens. **The measured
   window starts with an empty pipeline**, so the frame age it reports is this device's
   queue and not the compile's backlog.
3. **Placement is confirmed per trial, from `EXECUTION_DEVICES` read after inference.**
   The reading goes through `normalise_execution_devices`, because the NPU plugin reports
   the **bare string** `'NPU'` while the CPU reports `['CPU']` and the GPU `['GPU.0']`.
   Readings are collected for every frame of the window; if two different readings appear
   the trial is refused, because a route that moved mid-window is not one measurement. A
   fallback is recorded as what ran: `confirmed=('CPU',)` beside `requested='NPU'` and
   `fell_back=True`. **`TrialResult` refuses to be constructed with a fallback reported as
   its request** - see its `__post_init__` - so the wrong number cannot be built by hand
   either.
4. **Repeats, published as a range.** One run per device is an anecdote. The table carries
   every repeat; the report quotes the spread.
5. **The machine was clean before the run.** The census is `tools.input_ack_under_load`'s
   own `assert_machine_clean`/`machine_state` - one definition of "clean" for the whole
   project, and the one that caught 66 orphaned workers during M3. It is checked before the
   first trial **and polled every 2 s while each trial runs**, so a suite starting in
   another window is a refusal rather than a quietly contaminated figure.
   `ACKLD_ALLOW_DIRTY=1` still permits a deliberately dirty run, and the record says so.
6. **The distribution, not the p50.** Every trial reports throughput, p50/p95/max latency,
   p50/p95 frame age, and the sample counts they were computed from. A percentile over the
   300 samples of a `booth` window and one over the 6 208 of a `sustained` GPU window are both
   labelled with their `n`.
7. **Frame age and inference cost stay separate claims.** `latency_*` is the `infer` call
   (the same definition as `app.engine.worker.infer_once`, so it is comparable with M0's
   numbers). `frame_age_*` is how long the picture waited, measured from `captured_at`
   through `FramePacket.age_at`. On the NPU these have been seconds and milliseconds
   respectively at the same instant; they are never added, averaged or reconciled.

## What it refuses to do

- **It does not pick a winner.** It prints numbers and labels their conditions. The report
  (`task-4-report.md`) is where "supports" and "does not support" are decided, and that
  decision is bounded by this machine, `yolo11n-fp16`, 640x640, batch 1.
- **It does not treat a fallback as the device that was asked for.**
- **It does not report `0` for something it did not measure.** `drops` and `frame_age_*` are
  `None` in `sustained`, where there is no queue to drop from and no frame to be stale.
- **It does not touch `app/`.** Nothing in `app/` imports from `tools/`, and this tool adds
  no dependency: `openvino`, `numpy`, `psutil` and `PySide6` are the pinned set.

## Cost, measured on this machine

A campaign of 3 devices x 3 conditions x 3 repeats at `--seconds 20 --warmup 3` measured
**758.8 s (12 min 39 s)** of wall time on 2026-10-27, of which **540.33 s** was measured
window: 27 rows, 44 025 analysed frames or inferences. Per trial it measured 28.1 s for a
`booth` row on the CPU or the GPU and 30.1 s for the NPU's (the NPU's compile), 28.1 s for a
`source` row on the GPU and 30.1 s on the NPU and the CPU (compile, and the CPU's own backlog
drained), and 26.1 s for a `sustained` row. The record is
`docs/qualification/m4-sustained-trials.json`; `task-4-report.md` quotes it.

The suite's own test of this tool runs one **8 s** CPU trial (14.2 s for the class, three
assertions sharing it), and skips - with the reason - on a machine that cannot run one.

## Exit codes

`0` every trial ran and the record was written. `1` at least one trial failed (the failure
is in the record, and the others still ran). `3` refused before measuring - the machine was
not clean, or a provisioned asset is missing. `2` the tool itself failed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.inference.devices import (  # noqa: E402
    DEVICE_IDS,
    FORBIDDEN_TARGETS,
    is_device,
    normalise_execution_devices,
    plugin_class,
)

__all__ = [
    "CONDITIONS",
    "CONDITION_BOOTH",
    "CONDITION_SOURCE",
    "CONDITION_SUSTAINED",
    "DEFAULT_REPEATS",
    "DEFAULT_SECONDS",
    "DEFAULT_WARMUP",
    "TrialResult",
    "TrialSpec",
    "build_record",
    "format_table",
    "interleaved_order",
    "percentile",
    "placement_from_readings",
    "run_paired_trials",
    "run_trial",
]

# --------------------------------------------------------------------------------------
# Constants. Every number here is a condition of measurement, not a result.
# --------------------------------------------------------------------------------------

#: The booth's analysis rate: `app.main.run_headless` feeds `ClipSource(target_fps=15.0)`,
#: and a 29.97002997 fps clip quantises that to 14.985 (see `ClipSource.analysis_fps`).
#: The trial asks for 15.0 for the same reason the booth does, and reports the 14.985 that
#: the frame grid delivered.
BOOTH_FPS: Final[float] = 15.0

CONDITION_BOOTH: Final[str] = "booth"
CONDITION_SOURCE: Final[str] = "source"
CONDITION_SUSTAINED: Final[str] = "sustained"
CONDITIONS: Final[tuple[str, ...]] = (CONDITION_BOOTH, CONDITION_SOURCE, CONDITION_SUSTAINED)

#: Which conditions run `ClipSource` + `Supervisor` and which run the bare inference loop.
PIPELINE_CONDITIONS: Final[frozenset[str]] = frozenset({CONDITION_BOOTH, CONDITION_SOURCE})

#: 20 s at 15 fps is 300 analysed frames and at 29.97 fps is ~600. Long enough that a single
#: stalled inference cannot move a percentile, short enough that 27 trials fit in a sitting.
#: The report states the duration it actually used.
DEFAULT_SECONDS: Final[float] = 20.0

#: Identical for every device and every condition. The NPU's transient is the longest
#: measured here: in a fresh process its median over the **first 30** inferences was
#: 5.50-5.57 ms, against 3.96-4.01 ms over the 20 s window of the same trial (see
#: `docs/qualification/m4-sustained-trials.json`). 3 s is ~45 frames at the booth rate, ~90
#: at the source rate and ~740 at the sustained NPU rate this campaign measured, which is
#: long past that transient.
DEFAULT_WARMUP: Final[float] = 3.0

DEFAULT_REPEATS: Final[int] = 3

#: How long to wait for the very first analysed result, which is where the compile happens.
#: A 120 s ceiling turns a wedged worker into an error rather than a hung tool; the slowest
#: compile measured here is the NPU's, and the campaign's own wall times put it at about 2 s
#: (a `booth` row's 20.01 s window plus 3 s warm-up took 30.1 s per NPU trial against 28.1 s
#: per CPU or GPU trial).
SETTLE_TIMEOUT: Final[float] = 120.0

#: A measured window closes when it closes; whatever the worker was already given is
#: drained afterwards for the event record and for the counters, and never enters the
#: window's throughput or its percentiles.
DRAIN_SECONDS: Final[float] = 3.0

#: How long "nothing new has been analysed" must hold before the pipeline is called empty.
QUIESCE_SECONDS: Final[float] = 0.4

#: A trial's own ceiling. 20 s of window plus a 120 s settle plus a 3 s drain plus teardown
#: fits with room; a trial that exceeds this is killed and reported as a failure.
TRIAL_TIMEOUT: Final[float] = 300.0

#: Seconds between machine censuses while a trial runs. `tools.input_ack_under_load` polls
#: at 2 s for the same reason: cheap, and fast enough to catch a suite starting mid-window.
CENSUS_SECONDS: Final[float] = 2.0

#: The pinned artefact ids this run is about. Quoted in the record so a reader cannot read
#: these numbers as a claim about a different model, shape or clip.
MODEL_ID: Final[str] = "yolo11n-fp16"
INPUT_SHAPE: Final[tuple[int, int, int, int]] = (1, 3, 640, 640)

OUTPUT_JSON: Final[Path] = REPO_ROOT / "docs" / "qualification" / "m4-sustained-trials.json"

_CONDITION_NOTES: Final[Mapping[str, str]] = {
    CONDITION_BOOTH: "ClipSource at 15.0 fps (14.985 on this clip's frame grid) through the "
                     "real Supervisor pipeline, 640x640 detector, display copy 960 px wide",
    CONDITION_SOURCE: "every source frame (~29.97 fps) through the real Supervisor "
                      "pipeline - twice the booth's analysed rate, still capped by the clip",
    CONDITION_SUSTAINED: "one frame's letterboxed blob, repeated as fast as the device "
                         "accepts it: `infer` on the pinned IR only, no decode, no queue",
}


# --------------------------------------------------------------------------------------
# Ordering and statistics.
# --------------------------------------------------------------------------------------

def percentile(values: Sequence[float], fraction: float) -> float | None:
    """A nearest-rank percentile, or `None` for an empty sequence.

    **No interpolation**, matching `app.main._percentile` exactly: p95 of 20 samples is the
    19th of the 20 that were taken, not an average of two neighbours. A smoothed percentile
    is a number no measurement produced. `tests/test_sustained_trials.py` pins the agreement
    with the application's own function rather than trusting this paragraph.
    """
    if not values:
        return None
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction!r}")
    ordered = sorted(float(v) for v in values)
    rank = max(1, math.ceil(float(fraction) * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def interleaved_order(devices: Sequence[str],
                      repeats: int) -> tuple[tuple[int, str], ...]:
    """`(repeat, device)` pairs, round by round, with the device order rotated each round.

    This is the anti-drift rule, in one place. Round `r` is `devices` rotated left by `r`,
    so every device appears exactly once per round and the trial that runs first changes
    every round. A machine that warms up over the run then warms up *for every device*
    rather than for whichever ran first and last.

    Refuses an empty device list, a non-positive repeat count, and duplicate devices - a
    duplicate would measure the same device twice per round while the caller believed it
    had three.
    """
    labels = tuple(str(d) for d in devices)
    if not labels:
        raise ValueError("interleaved_order needs at least one device")
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate device in {labels!r}; each must appear once per round")
    count = int(repeats)
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise TypeError(
            f"repeats must be an int; got {repeats!r} ({type(repeats).__name__}). A float "
            f"would be silently truncated into a number of rounds nobody chose.")
    if count < 1:
        raise ValueError(f"repeats must be >= 1; got {count}")
    order: list[tuple[int, str]] = []
    for repeat in range(count):
        shift = repeat % len(labels)
        rotated = labels[shift:] + labels[:shift]
        order.extend((repeat, device) for device in rotated)
    return tuple(order)


def placement_from_readings(requested: str, readings: Sequence[Any]) -> tuple[tuple[str, ...], bool]:
    """`(confirmed, fell_back)` from every `EXECUTION_DEVICES` reading a trial saw.

    `readings` are the raw property values, one per inference (or per probe): the NPU's bare
    string, the CPU's `['CPU']`, the GPU's `['GPU.0']`. Each is normalised through
    `normalise_execution_devices`, so all three shapes compare.

    **Refused** rather than resolved:

    - an empty reading list, because nothing proved where the work ran;
    - two distinct readings, because a route that moved inside a measured window is not one
      measurement of one device;
    - a reading that normalises to nothing.

    `fell_back` is `not is_device(confirmed, requested)` - never a caller's opinion. A
    fallback is legitimate and is reported as what ran.
    """
    if not readings:
        raise ValueError(
            f"no EXECUTION_DEVICES reading for requested {requested!r}. Placement is only "
            f"proved by an inference that ran, so a trial with no reading has no device."
        )
    normalised = [normalise_execution_devices(value) for value in readings]
    empty = [item for item in normalised if not item]
    if empty:
        raise ValueError(
            f"EXECUTION_DEVICES came back empty for requested {requested!r} "
            f"({len(empty)} of {len(normalised)} readings). An empty reading is unproven "
            f"placement, not a device."
        )
    distinct = sorted(set(normalised))
    if len(distinct) > 1:
        raise ValueError(
            f"EXECUTION_DEVICES changed during one measured window for requested "
            f"{requested!r}: {[list(d) for d in distinct]}. A route that moved mid-window "
            f"is two devices' worth of work under one label, so the trial is refused."
        )
    confirmed = distinct[0]
    return confirmed, not is_device(confirmed, requested)


# --------------------------------------------------------------------------------------
# The trial: what was asked, and what was measured.
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialSpec:
    """One trial's conditions. Frozen, because a condition edited after the fact is not one.

    `feed_fps` is the **requested** analysis rate for a pipeline condition and ignored for
    `sustained`; `None` for `source` means "every source frame", resolved in the child from
    `app.engine.retail_config.clip_fps()` rather than guessed here.
    """

    condition: str
    device: str
    seconds: float
    warmup_s: float
    repeat: int
    order: int
    feed_fps: float | None = None

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(
                f"condition {self.condition!r} is not one of {list(CONDITIONS)}. The three "
                f"conditions measure three different things and cannot share a label.")
        if not str(self.device or "").strip():
            raise ValueError("device must be a non-empty explicit target (CPU, GPU or NPU)")
        if plugin_class(self.device) in FORBIDDEN_TARGETS:
            raise ValueError(
                f"device {self.device!r} is an implicit-placement plugin. This demo uses "
                f"explicit compile targets only, because EXECUTION_DEVICES read after the "
                f"first inference is the only proof of placement and these hide it.")
        for name in ("seconds", "warmup_s"):
            value = float(getattr(self, name))
            if not (value > 0.0) or value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"{name} must be a finite number > 0; got {value!r}")
        if self.condition in PIPELINE_CONDITIONS:
            if self.condition == CONDITION_BOOTH and self.feed_fps is None:
                raise ValueError(
                    f"condition {CONDITION_BOOTH!r} must state its feed rate; the booth's "
                    f"is {BOOTH_FPS} fps and leaving it out would make the label a guess.")
            if self.feed_fps is not None and not (float(self.feed_fps) > 0.0):
                raise ValueError(f"feed_fps must be > 0 when stated; got {self.feed_fps!r}")

    def as_payload(self) -> dict[str, Any]:
        """The JSON the child is handed. Every field, so the child inherits no defaults."""
        return {
            "condition": self.condition,
            "device": self.device,
            "seconds": float(self.seconds),
            "warmup_s": float(self.warmup_s),
            "repeat": int(self.repeat),
            "order": int(self.order),
            "feed_fps": None if self.feed_fps is None else float(self.feed_fps),
        }


#: The payload keys a `TrialResult` cannot be built without. Named here so `from_payload`
#: can refuse a partial payload by key rather than by a `TypeError` from a constructor.
_REQUIRED_PAYLOAD_KEYS: Final[tuple[str, ...]] = (
    "condition", "device", "requested", "confirmed", "fell_back", "repeat", "seconds",
    "frames", "throughput_fps", "latency_p50_ms", "latency_p95_ms", "latency_max_ms",
    "latency_samples", "frame_age_p50_ms", "frame_age_p95_ms", "frame_age_samples",
    "drops", "rss_start", "rss_end", "feed_fps", "warmup_s",
)


@dataclass(frozen=True)
class TrialResult:
    """One sustained trial: what was asked for, where it ran, and what it measured.

    ## The fields the brief names

    `device` is the trial's column label; `requested` is what the worker's own
    `WorkerConfig.device_id` said, read back from the child, so a trial cannot claim a
    request the worker never received. `confirmed` is the normalised `EXECUTION_DEVICES`
    reading, and `fell_back` is the comparison. **`__post_init__` enforces the pair**: a
    result claiming no fallback must have `confirmed` naming `requested`, and a result
    claiming a fallback must not. That is the "never reported as its requested target" rule
    as a construction-time refusal rather than a display convention.

    `seconds` is the **measured window** - the time the feed was actually open, or the timed
    span of the inference loop - not the trial's wall time. `frames` counts analysed frames
    (pipeline) or completed inferences (sustained). `throughput_fps` is `frames / seconds`.

    `latency_*` is the `infer` call only, the same definition `app.engine.worker.infer_once`
    uses for the number on the card. `frame_age_*` is queue wait, measured from
    `captured_at`; it is `None` in `sustained`, where there is no queue and no stale frame.
    `drops` is `None` there too: a zero drop count over zero submissions is a number that
    would be quoted as "no drops".

    `rss_start`/`rss_end` are **MiB of host resident set, summed over the trial's process
    tree** (the trial process and its spawned worker) at the window's two boundaries. They
    are host memory - OpenVINO's runtime, the model, the interpreter, the transport buffers
    - and **not accelerator memory**, which no property this milestone reads exposes. A
    trial that grew is recorded; nothing here asserts a threshold.

    `unmeasured` is set by the caller when a trial could not run at all (the child failed),
    and then every measurement is `None`: a failed trial must not carry a plausible zero.
    """

    condition: str
    device: str
    requested: str
    confirmed: tuple[str, ...]
    fell_back: bool | None
    repeat: int
    #: `None` only on an unmeasured row - see `unmeasured`. A measured row's figures are
    #: all present, and `__post_init__` refuses the row otherwise.
    seconds: float | None
    frames: int | None
    throughput_fps: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_max_ms: float | None
    latency_samples: int
    frame_age_p50_ms: float | None
    frame_age_p95_ms: float | None
    frame_age_samples: int
    drops: int | None
    rss_start: float | None
    rss_end: float | None
    feed_fps: float | None
    warmup_s: float
    cold30_p50_ms: float | None = None
    events: tuple[str, ...] = ()
    event_counts: tuple[tuple[str, int], ...] = ()
    errors: tuple[str, ...] = ()
    unmeasured: str | None = None

    # -- construction ----------------------------------------------------------------

    def __post_init__(self) -> None:
        for name in ("condition", "device", "requested"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string; got {getattr(self, name)!r}")
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition {self.condition!r} is not one of {list(CONDITIONS)}")
        if not isinstance(self.confirmed, tuple):
            raise TypeError(f"confirmed must be a tuple; got {type(self.confirmed).__name__}")
        object.__setattr__(self, "confirmed",
                           tuple(str(entry) for entry in self.confirmed))
        object.__setattr__(self, "events", tuple(str(entry) for entry in self.events))
        object.__setattr__(self, "errors", tuple(str(entry) for entry in self.errors))
        object.__setattr__(self, "event_counts",
                           tuple((str(kind), int(count)) for kind, count in self.event_counts))
        if not isinstance(self.repeat, int) or isinstance(self.repeat, bool) or self.repeat < 0:
            raise ValueError(f"repeat must be an int >= 0; got {self.repeat!r}")

        if self.unmeasured is not None:
            measured = (self.seconds, self.frames, self.throughput_fps, self.latency_p50_ms,
                        self.latency_p95_ms, self.latency_max_ms, self.frame_age_p50_ms,
                        self.frame_age_p95_ms, self.drops, self.rss_start, self.rss_end,
                        self.cold30_p50_ms)
            if self.confirmed or self.fell_back is not None:
                raise ValueError(
                    f"an unmeasured trial cannot carry a placement: confirmed="
                    f"{self.confirmed!r} fell_back={self.fell_back!r}. Nothing ran, so "
                    f"nothing was proved to run anywhere.")
            if any(value is not None for value in measured):
                raise ValueError(
                    "an unmeasured trial must carry no measurements; a failed row with a "
                    "plausible number in it is the row a reader quotes.")
            return

        if not self.confirmed:
            raise ValueError(
                f"a measured trial must carry the EXECUTION_DEVICES reading; requested "
                f"{self.requested!r} with confirmed=() is placement that was never proved.")
        if self.fell_back is None:
            raise ValueError(
                f"a measured trial with a confirmed placement must say whether it fell "
                f"back; {self.confirmed!r} beside requested {self.requested!r} is a "
                f"comparison the record has to make, not one a reader has to notice.")
        if not isinstance(self.fell_back, bool):
            raise TypeError(f"fell_back must be a bool or None; got {self.fell_back!r}")
        placed = is_device(self.confirmed, self.requested)
        if self.fell_back is False and not placed:
            raise ValueError(
                f"trial of {self.requested!r} says it did not fall back but "
                f"EXECUTION_DEVICES says {list(self.confirmed)}. A fallback reported as its "
                f"requested target is the one thing this record exists to prevent.")
        if self.fell_back is True and placed:
            raise ValueError(
                f"trial of {self.requested!r} says it fell back but EXECUTION_DEVICES says "
                f"{list(self.confirmed)}, which is the requested device. One of the two is "
                f"wrong and the record cannot hold both.")

        # Every figure a measured row publishes must be present. The alternative - a
        # default here - is the fabricated zero this project refuses everywhere else.
        required: dict[str, Any] = {
            "seconds": self.seconds, "frames": self.frames,
            "throughput_fps": self.throughput_fps, "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms, "latency_max_ms": self.latency_max_ms,
            "rss_start": self.rss_start, "rss_end": self.rss_end, "warmup_s": self.warmup_s,
        }
        if self.condition in PIPELINE_CONDITIONS:
            required["frame_age_p50_ms"] = self.frame_age_p50_ms
            required["frame_age_p95_ms"] = self.frame_age_p95_ms
            required["drops"] = self.drops
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(
                f"a measured trial must carry {missing}; a row with a hole in it is the row "
                f"a reader fills in from a neighbouring column.")

        for name in ("seconds", "throughput_fps", "warmup_s"):
            value = float(getattr(self, name))
            if value != value or value in (float("inf"), float("-inf")) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0; got {value!r}")
        if self.seconds <= 0.0:
            raise ValueError(f"seconds must be > 0 for a measured trial; got {self.seconds!r}")
        if isinstance(self.frames, bool) or not isinstance(self.frames, int) or self.frames < 0:
            raise ValueError(f"frames must be an int >= 0; got {self.frames!r}")
        for name in ("latency_samples", "frame_age_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an int >= 0; got {value!r}")
        if self.frames > 0 and self.latency_samples == 0:
            raise ValueError(
                f"{self.frames} frames were analysed but no latency sample was recorded. A "
                f"throughput without a latency distribution is half a result.")

        if self.condition == CONDITION_SUSTAINED:
            if self.frame_age_p50_ms is not None or self.frame_age_p95_ms is not None:
                raise ValueError(
                    "sustained has no queue and no frame age; a frame-age figure there "
                    "would be a number about nothing.")
            if self.drops is not None:
                raise ValueError("sustained has no submissions to drop; drops must be None")
        else:
            if self.drops is None:
                raise ValueError(f"condition {self.condition!r} drops frames from a queue; "
                                 f"a None drop count hides the lag it is there for")

        self._require_ordered_percentiles("latency", self.latency_p50_ms,
                                          self.latency_p95_ms, self.latency_max_ms)
        self._require_ordered_percentiles("frame age", self.frame_age_p50_ms,
                                          self.frame_age_p95_ms, None)
        for name, value in (("rss_start", self.rss_start), ("rss_end", self.rss_end)):
            if value is None or not isinstance(value, (int, float)) or value <= 0.0:
                raise ValueError(
                    f"{name} must be a positive number of MiB for a measured trial; got "
                    f"{value!r}. A zero resident set is not a measurement.")

    @staticmethod
    def _require_ordered_percentiles(label: str, p50: float | None, p95: float | None,
                                     maximum: float | None) -> None:
        if p50 is None and p95 is None:
            if maximum is not None:
                raise ValueError(f"{label} has a max but no percentiles")
            return
        if p50 is None or p95 is None:
            raise ValueError(f"{label} percentiles must be both present or both absent")
        values = [float(p50), float(p95)] + ([] if maximum is None else [float(maximum)])
        if any(v != v or v in (float("inf"), float("-inf")) for v in values):
            raise ValueError(f"{label} percentiles must be finite; got {values!r}")
        if not (p50 <= p95):
            raise ValueError(f"{label} p50 {p50!r} exceeds p95 {p95!r}")
        if maximum is not None and not (p95 <= maximum):
            raise ValueError(f"{label} p95 {p95!r} exceeds max {maximum!r}")

    # -- reading and writing ---------------------------------------------------------

    @property
    def counts(self) -> dict[str, int]:
        """`event_counts` as a dict, for printing and for the JSON."""
        return {kind: int(count) for kind, count in self.event_counts}

    @property
    def placement(self) -> str:
        """The placement as one string: what `EXECUTION_DEVICES` said, or `unmeasured`."""
        if self.unmeasured is not None:
            return "unmeasured"
        text = "/".join(self.confirmed) if self.confirmed else "unconfirmed"
        return f"{text} (fell back from {self.requested})" if self.fell_back else text

    @classmethod
    def unmeasured_result(cls, spec: "TrialSpec", reason: str) -> "TrialResult":
        """The row for a trial that could not run, with **no measurement and no placement**.

        Constructed from the spec, so the condition and the device it was asked for survive
        even though nothing was measured. `confirmed` stays empty and `fell_back` `None`:
        nothing ran, so nothing was proved to run anywhere, and a placement here would be a
        claim about hardware that was never touched.
        """
        return cls(
            condition=spec.condition, device=spec.device, requested=spec.device,
            confirmed=(), fell_back=None, repeat=spec.repeat, seconds=None, frames=None,
            throughput_fps=None, latency_p50_ms=None, latency_p95_ms=None,
            latency_max_ms=None, latency_samples=0, frame_age_p50_ms=None,
            frame_age_p95_ms=None, frame_age_samples=0, drops=None, rss_start=None,
            rss_end=None, feed_fps=spec.feed_fps, warmup_s=spec.warmup_s,
            unmeasured=str(reason),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TrialResult":
        """Build a result from a child's JSON, **refusing anything incomplete**.

        No field is defaulted except the optional extras, because a defaulted measurement
        is a fabricated one: `payload.get("frames", 0)` would report "0 analysed" for a
        payload that never carried a count, which is indistinguishable from a run that
        analysed nothing.
        """
        if not isinstance(payload, Mapping):
            raise TypeError(f"a trial payload must be a mapping; got {type(payload).__name__}")
        # **Unmeasured first.** A failed child writes a payload with a reason and no
        # measurements, so the completeness check below would refuse the very row that
        # exists to say nothing was measured.
        if payload.get("unmeasured") is not None:
            return cls(
                condition=str(payload.get("condition") or ""),
                device=str(payload.get("device") or ""),
                requested=str(payload.get("requested") or payload.get("device") or ""),
                confirmed=(), fell_back=None, repeat=int(payload.get("repeat") or 0),
                seconds=None, frames=None, throughput_fps=None,
                latency_p50_ms=None, latency_p95_ms=None, latency_max_ms=None,
                latency_samples=0, frame_age_p50_ms=None, frame_age_p95_ms=None,
                frame_age_samples=0, drops=None, rss_start=None, rss_end=None,
                feed_fps=None, warmup_s=float(payload.get("warmup_s") or 0.0),
                unmeasured=str(payload["unmeasured"]),
            )
        known = {field.name for field in fields(cls)}
        missing = [key for key in _REQUIRED_PAYLOAD_KEYS if key not in payload]
        if missing:
            raise ValueError(
                f"trial payload is missing {missing}. Every required figure is measured by "
                f"the child; a missing key means the child did not measure it, and no "
                f"default here can stand in for that.")
        kwargs = {key: value for key, value in payload.items() if key in known}
        kwargs["confirmed"] = tuple(payload["confirmed"])
        kwargs["events"] = tuple(payload.get("events") or ())
        kwargs["errors"] = tuple(payload.get("errors") or ())
        kwargs["event_counts"] = tuple(
            (str(kind), int(count))
            for kind, count in dict(payload.get("event_counts") or {}).items())
        return cls(**kwargs)

    def as_dict(self) -> dict[str, Any]:
        """The JSON-ready record. `None` stays `None`; nothing is rounded into a claim."""
        return {
            "condition": self.condition,
            "device": self.device,
            "requested": self.requested,
            "confirmed": list(self.confirmed),
            "fell_back": self.fell_back,
            "repeat": self.repeat,
            "seconds": self.seconds,
            "frames": self.frames,
            "throughput_fps": self.throughput_fps,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_max_ms": self.latency_max_ms,
            "latency_samples": self.latency_samples,
            "frame_age_p50_ms": self.frame_age_p50_ms,
            "frame_age_p95_ms": self.frame_age_p95_ms,
            "frame_age_samples": self.frame_age_samples,
            "drops": self.drops,
            "rss_start": self.rss_start,
            "rss_end": self.rss_end,
            "feed_fps": self.feed_fps,
            "warmup_s": self.warmup_s,
            "cold30_p50_ms": self.cold30_p50_ms,
            "events": list(self.events),
            "event_counts": self.counts,
            "errors": list(self.errors),
            "unmeasured": self.unmeasured,
        }


# --------------------------------------------------------------------------------------
# The child: one trial in its own process.
# --------------------------------------------------------------------------------------

def _tree_rss_mib() -> float:
    """Host resident set of this process **and its descendants**, in MiB.

    The worker is a spawned child, and on the NPU most of the resident set lives there; a
    measurement of this process alone would miss it. Not accelerator memory - see
    `TrialResult`.
    """
    import psutil

    process = psutil.Process()
    total = int(process.memory_info().rss)
    for child in process.children(recursive=True):
        try:
            total += int(child.memory_info().rss)
        except Exception:  # noqa: BLE001 - a worker that just exited is not an error here
            continue
    return total / (1024.0 * 1024.0)


class _ResultRecorder:
    """Collects each `InferenceResult` as the reader thread emits it.

    Connected with `Qt.DirectConnection`, so the slot runs in the emitting thread and no
    event loop is needed (a headless trial has none). It only appends under a lock: it must
    never do work that could slow the reader, or the measurement would include the
    recorder.

    `enabled` is the measured window. Samples collected during settle and warm-up are
    discarded, so a trial's percentiles describe the window its throughput does.
    """

    def __init__(self, clock: Callable[[], float]) -> None:
        import threading

        self._clock = clock
        self._lock = threading.Lock()
        self._latencies: list[float] = []
        self._ages: list[float] = []
        self._readings: list[Any] = []
        self._errors: list[str] = []
        self.enabled = False

    def on_result(self, result: Any) -> None:
        now = self._clock()
        with self._lock:
            self._readings.append(tuple(result.execution_devices))
            if result.error is not None:
                self._errors.append(str(result.error))
                return
            if not self.enabled:
                return
            if result.latency_ms is not None:
                self._latencies.append(float(result.latency_ms))
            self._ages.append(result.frame.age_at(now) * 1000.0)

    def open_window(self) -> None:
        with self._lock:
            self._latencies.clear()
            self._ages.clear()
            self._readings.clear()
            self._errors.clear()
            self.enabled = True

    def close_window(self) -> None:
        with self._lock:
            self.enabled = False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "latencies": list(self._latencies),
                "ages": list(self._ages),
                "readings": list(self._readings),
                "errors": list(self._errors),
            }


def _wait_for_first_result(supervisor: Any, fold: Callable[[], None],
                           timeout: float) -> float:
    """Wait until one frame has been analysed, and return how long that took.

    The compile happens inside the worker on its first frame, so this is where it is paid:
    measured on the task's own `booth` rows, an NPU trial took 30.1 s of wall where a CPU or
    GPU trial took 28.1 s, with the same 20.01 s window and 3 s warm-up. That difference is
    the compile, and it is **outside every window**: a comparison that charged one device's
    compile to its throughput would compare plugins, not devices.
    """
    started = time.perf_counter()
    deadline = started + float(timeout)
    while supervisor.frames_analysed == 0:
        fold()
        if time.perf_counter() > deadline:
            raise TimeoutError(
                f"no frame was analysed within {timeout:.0f} s of the first submission. The "
                f"worker either did not compile or never received the frame; a trial that "
                f"reports zero throughput here would be reporting a broken pipeline as a "
                f"slow device.")
        time.sleep(0.02)
    return time.perf_counter() - started


def _quiesce(supervisor: Any, fold: Callable[[], None],
             quiesce_s: float = QUIESCE_SECONDS, ceiling_s: float = DRAIN_SECONDS) -> float:
    """Fold until nothing new has been analysed for `quiesce_s`, or `ceiling_s` passes.

    This is what makes the measured window start from an **empty** pipeline. Without it,
    every trial inherits the frames it submitted while the worker was compiling, and the
    first device to run pays a backlog it did not create - or, worse, a device that cannot
    drain it is credited with the same throughput as one that can.
    """
    started = time.perf_counter()
    quiet_since = started
    last = supervisor.frames_analysed
    while time.perf_counter() - started < float(ceiling_s):
        fold()
        current = supervisor.frames_analysed
        if current != last:
            last = current
            quiet_since = time.perf_counter()
        elif time.perf_counter() - quiet_since >= float(quiesce_s):
            break
        time.sleep(0.02)
    return time.perf_counter() - started


def _feed_for(clip: Any, supervisor: Any, fold: Callable[[], None],
              seconds: float) -> tuple[float, int]:
    """Feed the clip into the pipeline for `seconds`. Returns `(span, submitted)`.

    The source paces itself (`ClipSource.read`), so this loop is the booth's own cadence and
    the loop's cost is part of the measurement - a device that saturates the CPU starves the
    decoder, and that is a real property of running the detector on the CPU.

    `submitted` counts the frames the pipeline **accepted**; `supervisor.drops` is read either
    side of the window for the ones a full queue refused, which is the lag counter this
    application keeps.
    """
    started = time.perf_counter()
    submitted = 0
    while time.perf_counter() - started < float(seconds):
        frame = clip.read()
        if frame is None:
            # The clip ran out inside the window. Every condition is far shorter than the
            # 113 s clip, so this means the source is broken, not that the window ended.
            raise RuntimeError(
                "the clip ended inside the measured window; the window would then be "
                "shorter than the one requested and its throughput would be divided by "
                "the wrong duration.")
        if supervisor.submit(frame):
            submitted += 1
        fold()
    return time.perf_counter() - started, submitted


def _run_pipeline_trial(spec: Mapping[str, Any]) -> dict[str, Any]:
    """`booth` and `source`: the application's own pipeline, for one measured window."""
    from PySide6 import QtCore

    from app.engine.events import RetailEventMachine
    from app.engine.retail_config import (
        RETAIL_CHECKOUT_CLIP,
        clip_fps,
        retail_sanity_manifest,
    )
    from app.engine.source import ClipSource
    from app.engine.tracker import Tracker
    from app.inference.pinned import class_names
    from app.main import _fold_result
    from app.supervisor.runner import Supervisor

    condition = str(spec["condition"])
    device = str(spec["device"])
    seconds = float(spec["seconds"])
    warmup_s = float(spec["warmup_s"])
    requested_fps = (clip_fps() if spec.get("feed_fps") is None
                     else float(spec["feed_fps"]))

    manifest = retail_sanity_manifest()
    tracker = Tracker(iou_threshold=manifest.iou_threshold)
    machine = RetailEventMachine(manifest, class_names=class_names())
    supervisor = Supervisor(device_id=device, stream_id="A", session_id="sustained-trials",
                            scenario_id="retail_sanity",
                            conf_threshold=manifest.conf_threshold,
                            iou_threshold=manifest.iou_threshold)
    clip = ClipSource(Path(RETAIL_CHECKOUT_CLIP), stream_id="A",
                      session_id="sustained-trials", scenario_id="retail_sanity",
                      target_fps=requested_fps)
    recorder = _ResultRecorder(time.perf_counter)
    supervisor.result_ready.connect(
        recorder.on_result, QtCore.Qt.ConnectionType.DirectConnection)
    folded: dict[str, Any] = {"identity": None}

    def fold() -> None:
        """The application's own fold, with one guard the application does not need.

        `run_headless` never rewinds the clip mid-run without also resetting the machine to
        the new loop, so `_fold_result` cannot meet a stale-loop result there. This trial
        *does* rewind - that is how the window starts at frame 0 - and a result for the last
        warm-up frame can still be in the reader thread's hands. `RetailEventMachine.observe`
        refuses such a frame by design ("adopting the new loop silently would carry one
        loop's pending transfers into the next"), and that refusal is right: the fold drops
        the frame instead of killing the trial.
        """
        latest = supervisor.latest_result()
        if (latest is not None and machine.loop_id is not None
                and not latest.frame.is_from(machine.loop_id)):
            return
        _fold_result(supervisor, tracker, machine, folded)

    payload: dict[str, Any] = {
        "condition": condition,
        "device": device,
        "requested": device,
        "repeat": int(spec["repeat"]),
        "feed_fps": clip.analysis_fps,
        "requested_feed_fps": requested_fps,
        "warmup_s": warmup_s,
        "analysis_fps": clip.analysis_fps,
        "source_fps": clip.source_fps,
        "clip": RETAIL_CHECKOUT_CLIP,
        "frames_refused": 0,
        "frames_submitted": 0,
        "unmeasured": None,
    }
    started = time.perf_counter()
    try:
        supervisor.start()
        # One frame alone, then wait: the compile is inside the worker's first inference.
        first = clip.read()
        if first is None:
            raise RuntimeError(f"{RETAIL_CHECKOUT_CLIP} produced no first frame")
        supervisor.submit(first)
        payload["settle_s"] = _wait_for_first_result(supervisor, fold, SETTLE_TIMEOUT)
        payload["quiesce_s"] = _quiesce(supervisor, fold)

        # Warm every device identically, then start the window from an empty pipeline.
        _feed_for(clip, supervisor, fold, warmup_s)
        payload["quiesce_before_window_s"] = _quiesce(supervisor, fold)

        clip.restart()
        tracker.reset()
        machine.reset(clip.loop_id)
        # The identity is set to the frame already folded, **not** cleared: clearing it makes
        # `_fold_result` re-fold the newest (previous-loop) result into the fresh tracker,
        # which is exactly the I2 defect the identity guard exists to prevent, in reverse.
        latest = supervisor.latest_result()
        folded["identity"] = None if latest is None else tuple(latest.frame.identity)
        # The queue was drained above, but a result for the warm-up's last frames can still
        # be in the reader thread's hands; `fold` drops it rather than letting the fresh
        # tracker see a frame from the previous loop.
        _quiesce(supervisor, fold)

        analysed_before = supervisor.frames_analysed
        drops_before = supervisor.drops
        refused_before = supervisor.frames_refused
        rss_start = _tree_rss_mib()
        recorder.open_window()
        span, submitted = _feed_for(clip, supervisor, fold, seconds)
        recorder.close_window()
        analysed_in_window = supervisor.frames_analysed - analysed_before
        drops_in_window = supervisor.drops - drops_before
        rss_end = _tree_rss_mib()
        payload.update({
            "seconds": span,
            "frames": analysed_in_window,
            "drops": drops_in_window,
            "rss_start": rss_start,
            "rss_end": rss_end,
            "frames_submitted": submitted,
            "frames_refused": supervisor.frames_refused - refused_before,
        })
        # Drain what the window left in flight, so the events and counters describe the
        # same instant. It never re-enters `frames` or the percentiles.
        payload["drain_s"] = _quiesce(supervisor, fold, DRAIN_SECONDS / 3.0, DRAIN_SECONDS)
    finally:
        try:
            supervisor.stop()
        finally:
            clip.close()

    samples = recorder.snapshot()
    latencies = samples["latencies"]
    ages = samples["ages"]
    payload.update({
        "throughput_fps": payload["frames"] / payload["seconds"],
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_max_ms": max(latencies) if latencies else None,
        "latency_samples": len(latencies),
        "frame_age_p50_ms": percentile(ages, 0.50),
        "frame_age_p95_ms": percentile(ages, 0.95),
        "frame_age_samples": len(ages),
        "cold30_p50_ms": None,
        "events": [event.dedup_key for event in machine.events],
        "event_counts": _count_events(machine.events),
        "errors": sorted(set(samples["errors"])),
        "wall_s": time.perf_counter() - started,
    })
    # The readings are every frame's, and `placement_from_readings` refuses a route that
    # moved inside the window rather than picking the last one.
    readings = [list(item) for item in samples["readings"]]
    confirmed, fell_back = placement_from_readings(device, readings)
    payload.update({
        "confirmed": list(confirmed),
        "fell_back": fell_back,
        "placement_readings": len(readings),
    })
    return payload


def _run_sustained_trial(spec: Mapping[str, Any]) -> dict[str, Any]:
    """`sustained`: the pinned model's `infer` at maximum rate, with no pipeline around it.

    The timed call is `request.infer({0: blob})` and nothing else - the same span
    `tools/qualify_m0.py` times and the same span `app.engine.worker.infer_once` puts around
    the `infer` call. Preprocessing, the letterbox and the placement read are outside it, so
    this latency is comparable with the project's recorded single-frame figures;
    `throughput_fps` divides the completed inferences by the timed span, including the
    Python loop's own cost, and is therefore a rate rather than `1 / p50`.
    """
    import statistics

    import openvino as ov

    from app.engine.retail_config import RETAIL_CHECKOUT_CLIP
    from app.engine.source import ClipSource
    from app.inference.pinned import verified_ir_path
    from app.inference.preprocess import preprocess_bgr

    condition = str(spec["condition"])
    device = str(spec["device"])
    seconds = float(spec["seconds"])
    warmup_s = float(spec["warmup_s"])
    width, height = INPUT_SHAPE[3], INPUT_SHAPE[2]

    # A real frame from the pinned clip, letterboxed exactly as the worker does it. A zero
    # image would time the same arithmetic, but this workload's input is a camera frame and
    # the blob's contents should be the ones the booth feeds.
    source = ClipSource(Path(RETAIL_CHECKOUT_CLIP), stream_id="A",
                        session_id="sustained-trials", scenario_id="retail_sanity")
    try:
        packet = source.read()
    finally:
        source.close()
    if packet is None:
        raise RuntimeError(f"{RETAIL_CHECKOUT_CLIP} produced no first frame")
    blob, _letterbox = preprocess_bgr(packet.image, width, height)

    ir = str(verified_ir_path())
    core = ov.Core()
    present = tuple(core.available_devices)
    if device not in present:
        raise RuntimeError(
            f"cannot measure {device!r}: ov.Core().available_devices() reports {list(present)}")
    model = core.read_model(ir)
    compiled = core.compile_model(model, device)
    request = compiled.create_infer_request()

    readings: list[Any] = []

    def read_placement() -> tuple[str, ...]:
        value = request.get_compiled_model().get_property("EXECUTION_DEVICES")
        reading = normalise_execution_devices(value)
        if not reading:
            raise RuntimeError(
                f"EXECUTION_DEVICES read back empty ({value!r}) after an inference; there is "
                f"no evidence this ran anywhere, so the trial is refused.")
        readings.append(reading)
        return reading

    payload: dict[str, Any] = {
        "condition": condition,
        "device": device,
        "requested": device,
        "repeat": int(spec["repeat"]),
        "feed_fps": None,
        "warmup_s": warmup_s,
        "frames_submitted": 0,
        "frames_refused": 0,
        "drops": None,
        "frame_age_p50_ms": None,
        "frame_age_p95_ms": None,
        "frame_age_samples": 0,
        "events": [],
        "event_counts": {},
        "errors": [],
        "unmeasured": None,
    }
    started = time.perf_counter()
    # One untimed inference first, exactly as `tools/qualify_m0.py` does, so the first timed
    # sample is steady-state and the placement read that follows is about a placement that
    # happened.
    request.infer({0: blob})
    read_placement()

    # The cold-30 figure: the sample size every earlier single-frame p50 in this project
    # used. Recording it here, from the same process and the same model, is what lets the
    # report compare "30 inferences" with "20 seconds" instead of asserting a difference.
    cold: list[float] = []
    for _ in range(30):
        one = time.perf_counter()
        request.infer({0: blob})
        cold.append((time.perf_counter() - one) * 1000.0)

    warmup_until = time.perf_counter() + warmup_s
    while time.perf_counter() < warmup_until:
        request.infer({0: blob})

    latencies: list[float] = []
    rss_start = _tree_rss_mib()
    window_started = time.perf_counter()
    while time.perf_counter() - window_started < seconds:
        one = time.perf_counter()
        request.infer({0: blob})
        latencies.append((time.perf_counter() - one) * 1000.0)
    span = time.perf_counter() - window_started
    rss_end = _tree_rss_mib()
    read_placement()

    payload.update({
        "seconds": span,
        "frames": len(latencies),
        "throughput_fps": (len(latencies) / span) if span > 0 else 0.0,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_max_ms": max(latencies) if latencies else None,
        "latency_samples": len(latencies),
        "cold30_p50_ms": statistics.median(cold) if cold else None,
        "rss_start": rss_start,
        "rss_end": rss_end,
        "wall_s": time.perf_counter() - started,
        "input_shape": list(INPUT_SHAPE),
    })
    confirmed, fell_back = placement_from_readings(device, readings)
    payload.update({
        "confirmed": list(confirmed),
        "fell_back": fell_back,
        "placement_readings": len(readings),
    })
    return payload


def _count_events(events: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    return counts


def run_trial_in_this_process(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The child's entry point: run one trial and return its payload.

    Never raises out to the operator: a failure becomes `{"unmeasured": reason}` so the
    parent records a failed row rather than losing the trials around it. That is the same
    "never crash in front of a customer" rule as the application's, applied to a
    measurement that is meant to complete.
    """
    base = {"condition": str(spec.get("condition") or ""),
            "device": str(spec.get("device") or ""),
            "requested": str(spec.get("device") or ""),
            "repeat": int(spec.get("repeat") or 0),
            "warmup_s": float(spec.get("warmup_s") or 0.0)}
    try:
        if spec["condition"] in PIPELINE_CONDITIONS:
            return _run_pipeline_trial(spec)
        return _run_sustained_trial(spec)
    except Exception as exc:  # noqa: BLE001 - a trial that cannot run is a recorded row
        detail = f"{type(exc).__name__}: {exc}"
        return {**base, "unmeasured": detail}


# --------------------------------------------------------------------------------------
# The parent: spawn, census, collect.
# --------------------------------------------------------------------------------------

def trial_command(spec_path: Path, out_path: Path) -> list[str]:
    """The exact command one trial runs. Built here so it can be printed and audited."""
    return [sys.executable, str(Path(__file__).resolve()), "--child",
            "--spec", str(spec_path), "--out", str(out_path)]


def run_trial(spec: TrialSpec, *, python: str | None = None,
              timeout: float = TRIAL_TIMEOUT,
              census: Callable[[], dict[str, Any]] | None = None) -> TrialResult:
    """Run one trial **in a fresh process**, polling the machine census while it runs.

    A fresh process per trial is deliberate: the leak Task 3 fixed only appeared at the real
    2 s deadline, and a trial that inherited a previous trial's resident set (or its
    compiled plugin state) would not be the fresh booth the comparison claims to measure.

    `census` is injected so the suite can test this without reading the process table; the
    default is `tools.input_ack_under_load.machine_state`, the project's one census.

    Raises `MachineNotClean` if another Python process appears while the trial runs - a
    competitor mid-window is a refusal, not a figure to average in.
    """
    from tools.input_ack_under_load import (
        CLEAN_OVERRIDE_ENV,
        MachineNotClean,
        machine_state,
    )

    poll = census if census is not None else machine_state
    with tempfile.TemporaryDirectory(prefix="sustained-trial-") as tmp:
        root = Path(tmp)
        spec_path = root / "spec.json"
        out_path = root / "result.json"
        stdout_path = root / "stdout.log"
        stderr_path = root / "stderr.log"
        spec_path.write_text(json.dumps(spec.as_payload(), indent=2), encoding="utf-8")
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        contaminated: list[dict[str, Any]] = []
        command = trial_command(spec_path, out_path)
        # stdout/stderr to files, not pipes: a pipe nobody drains can fill and block the
        # child, which would look exactly like a slow trial.
        with open(stdout_path, "w", encoding="utf-8") as out_file, \
                open(stderr_path, "w", encoding="utf-8") as err_file:
            process = subprocess.Popen(command, cwd=str(REPO_ROOT), env=env,
                                       stdout=out_file, stderr=err_file, text=True)
            deadline = time.perf_counter() + float(timeout)
            try:
                while process.poll() is None:
                    if time.perf_counter() > deadline:
                        process.kill()
                        process.wait()
                        raise RuntimeError(
                            f"trial {spec.condition}/{spec.device} did not finish within "
                            f"{timeout:.0f} s and was killed. See {stderr_path.name}.")
                    state = poll()
                    if not state.get("clean", True):
                        contaminated.append(state)
                    time.sleep(CENSUS_SECONDS)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        if contaminated and os.environ.get(CLEAN_OVERRIDE_ENV) != "1":
            raise MachineNotClean(
                f"{spec.condition}/{spec.device}: another Python process appeared while this "
                f"trial was running, so its figures are about the application and about "
                f"whatever else was on the machine. Censuses: "
                f"{json.dumps(contaminated)[:2000]}. Set {CLEAN_OVERRIDE_ENV}=1 to keep a "
                f"deliberately dirty figure.")
        if not out_path.is_file():
            raise RuntimeError(
                f"trial {spec.condition}/{spec.device} exited {process.returncode} without "
                f"writing a result. stdout:\n{stdout_text[-3000:]}\n"
                f"stderr:\n{stderr_text[-3000:]}")
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    result = TrialResult.from_payload(payload)
    if result.condition != spec.condition or result.device != spec.device:
        raise RuntimeError(
            f"the child measured {result.condition}/{result.device} but "
            f"{spec.condition}/{spec.device} was asked for; a row filed under the wrong "
            f"device is worse than a missing one.")
    if result.requested != spec.device:
        raise RuntimeError(
            f"the child's worker was configured for {result.requested!r} while the trial "
            f"asked for {spec.device!r}; the request is read back from the child so this "
            f"cannot be papered over.")
    return result


def run_paired_trials(*, devices: Sequence[str] = DEVICE_IDS,
                      conditions: Sequence[str] = CONDITIONS,
                      repeats: int = DEFAULT_REPEATS,
                      seconds: float = DEFAULT_SECONDS,
                      warmup_s: float = DEFAULT_WARMUP,
                      runner: Callable[[TrialSpec], TrialResult] | None = None,
                      progress: Callable[[str], None] | None = None,
                      collected: list[TrialResult] | None = None) -> list[TrialResult]:
    """Every condition, every repeat, devices interleaved and each in a fresh process.

    **Condition-major, round-major within a condition.** Each condition's rounds are
    adjacent, so the three devices in one round are seconds apart and share the machine's
    state; the rounding order rotates by one per round (see `interleaved_order`). The
    conditions themselves are not interleaved: their durations differ, so interleaving them
    would buy nothing and make the schedule harder to audit.

    A trial that fails is **recorded and the run continues** - `TrialResult(unmeasured=...)`
    - so one wedged device cannot cost the other devices' repeats. The count of failures is
    the caller's to report; nothing here quietly drops a row.

    `collected` is the caller's own list, appended to as each trial finishes. A campaign
    takes minutes, and a refusal mid-run (another process starting, the operator stopping
    it) must not cost the trials that already ran: the caller keeps them and writes what it
    has.
    """
    if runner is None:
        runner = run_trial
    say = progress if progress is not None else (lambda line: print(line, flush=True))
    results: list[TrialResult] = collected if collected is not None else []
    order = interleaved_order(devices, repeats)
    index = 0
    for condition in conditions:
        if condition not in CONDITIONS:
            raise ValueError(f"condition {condition!r} is not one of {list(CONDITIONS)}")
        say(f"\n[{condition}] {_CONDITION_NOTES[condition]}")
        for repeat, device in order:
            index += 1
            spec = TrialSpec(condition=condition, device=device, seconds=float(seconds),
                             warmup_s=float(warmup_s), repeat=repeat, order=index,
                             feed_fps=(BOOTH_FPS if condition == CONDITION_BOOTH else None))
            say(f"  run {index:02d}  rep {repeat}  {device:<4} ...")
            started = time.perf_counter()
            try:
                result = runner(spec)
            except Exception as exc:  # noqa: BLE001 - a failed trial is a recorded row
                detail = f"{type(exc).__name__}: {exc}"
                say(f"    FAILED after {time.perf_counter() - started:.1f}s: {detail}")
                results.append(TrialResult.unmeasured_result(spec, detail))
                continue
            elapsed = time.perf_counter() - started
            say(f"    {result.placement:<22} {result.frames:>5} frames  "
                f"{result.throughput_fps:>7.2f}/s  "
                f"lat p50 {_ms(result.latency_p50_ms)}  "
                f"age p50 {_ms(result.frame_age_p50_ms)}  "
                f"({elapsed:.1f}s)")
            results.append(result)
    return results


def _ms(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:6.2f}"


# --------------------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------------------

#: Column widths for `format_table`. A fixed-width table is the only way a reader can
#: compare two rows by eye; the report embeds this exact string.
_TABLE_COLUMNS: Final[tuple[tuple[str, str, int], ...]] = (
    ("condition", "cond", 9),
    ("repeat", "rep", 3),
    ("device", "device", 6),
    ("requested", "requested", 9),
    ("confirmed", "confirmed", 9),
    ("fell_back", "fell_back", 9),
    ("seconds", "seconds", 7),
    ("frames", "frames", 6),
    ("throughput_fps", "fps", 7),
    ("latency_p50_ms", "lat_p50", 7),
    ("latency_p95_ms", "lat_p95", 7),
    ("latency_max_ms", "lat_max", 7),
    ("frame_age_p50_ms", "age_p50", 7),
    ("frame_age_p95_ms", "age_p95", 7),
    ("drops", "drops", 5),
    ("rss_start", "rss_in", 7),
    ("rss_end", "rss_out", 7),
)


def _cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_table(results: Sequence[TrialResult]) -> str:
    """The full table: every repeat, every column the brief asks for.

    **A fallback prints its actual placement and `fell_back=yes`**, never the device it was
    asked for. An unmeasured row prints `n/a` in every measured column and its reason in the
    note, because a failed row carrying a plausible number is the row a reader quotes.
    """
    header = "  ".join(f"{label:>{width}}" for _name, label, width in _TABLE_COLUMNS)
    lines = [header, "-" * len(header)]
    for result in results:
        cells = []
        for name, _label, width in _TABLE_COLUMNS:
            if name == "confirmed":
                text = "unmeasured" if result.unmeasured is not None else (
                    "/".join(result.confirmed) if result.confirmed else "unconfirmed")
            elif name == "fell_back":
                text = _cell(result.fell_back)
            elif result.unmeasured is not None and name not in ("condition", "repeat",
                                                                "device", "requested",
                                                                "confirmed", "fell_back"):
                text = "n/a"
            else:
                text = _cell(getattr(result, name))
            cells.append(f"{text:>{width}}")
        line = "  ".join(cells)
        if result.unmeasured is not None:
            line += f"   ERROR {result.unmeasured}"
        elif result.errors:
            line += f"   errors={len(result.errors)}"
        elif result.fell_back:
            line += f"   FALLBACK from {result.requested}"
        lines.append(line)
    return "\n".join(lines)


def _spread(values: Sequence[float | None]) -> str:
    """`min .. max` over the measured values, or `unmeasured` when none were."""
    present = [float(v) for v in values if v is not None]
    if not present:
        return "unmeasured"
    low, high = min(present), max(present)
    return f"{low:.2f}" if low == high else f"{low:.2f}..{high:.2f}"


def _spread_count(values: Sequence[int | None]) -> str:
    """`min` or `min..max` over integer counts. A count is not a measurement to round."""
    present = sorted(int(v) for v in values if v is not None)
    if not present:
        return "unmeasured"
    return str(present[0]) if present[0] == present[-1] else f"{present[0]}..{present[-1]}"


def per_condition_rows(results: Sequence[TrialResult]
                       ) -> tuple[tuple[str, str, str], ...]:
    """`(condition, device, line)` summaries, as ranges across the repeats.

    This is the shape a report quotes: a published device figure is **a range over
    repeats with its conditions**, never one number from one run. A device whose every
    repeat fell back is summarised under the device that actually ran.
    """
    rows: list[tuple[str, str, str]] = []
    for condition in CONDITIONS:
        for device in DEVICE_IDS:
            group = [r for r in results
                     if r.condition == condition and r.device == device
                     and r.unmeasured is None]
            if not group:
                continue
            placements = sorted({"/".join(r.confirmed) for r in group})
            fell_back = any(r.fell_back for r in group)
            rows.append((
                condition,
                device,
                f"{_spread([r.throughput_fps for r in group])} fps over "
                f"{_spread([float(r.seconds) for r in group])} s, "
                f"{len(group)} repeat(s); lat p50 {_spread([r.latency_p50_ms for r in group])}"
                f" p95 {_spread([r.latency_p95_ms for r in group])}"
                f" max {_spread([r.latency_max_ms for r in group])} ms; "
                f"age p50 {_spread([r.frame_age_p50_ms for r in group])}"
                f" p95 {_spread([r.frame_age_p95_ms for r in group])} ms; "
                f"drops {_spread_count([r.drops for r in group])}; "
                f"placement {'/'.join(placements)}"
                f"{' FELL BACK' if fell_back else ''}",
            ))
    return tuple(rows)


def machine_conditions() -> dict[str, Any]:
    """The conditions these numbers belong to. Nothing here is a performance claim."""
    import psutil

    import openvino as ov

    conditions: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "openvino": str(ov.__version__),
        "cpu_count": os.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "memory_gib": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "model": MODEL_ID,
        "input_shape": list(INPUT_SHAPE),
        "batch": INPUT_SHAPE[0],
        "display_copy_max_width": 960,
    }
    try:
        conditions["available_devices"] = list(ov.Core().available_devices)
    except Exception as exc:  # noqa: BLE001 - a report, not a gate
        conditions["available_devices"] = f"unreadable: {type(exc).__name__}: {exc}"
    # Driver versions, so the numbers are bounded by the stack they were measured on.
    # `DeviceRegistry.describe()` costs 3.3-3.8 s cold on this machine (recorded in
    # `tools/probe_utilisation.py`); it is called once, after the measurement, and a failure
    # is recorded rather than raised.
    try:
        from app.inference.devices import DeviceRegistry

        conditions["drivers"] = {
            status.device_id: status.driver_version
            for status in DeviceRegistry(ov.Core()).describe()}
    except Exception as exc:  # noqa: BLE001 - a report, not a gate
        conditions["drivers"] = f"unreadable: {type(exc).__name__}: {exc}"
    return conditions


def build_record(results: Sequence[TrialResult], *, seconds: float, warmup_s: float,
                 repeats: int, census: Mapping[str, Any] | None = None,
                 wall_s: float | None = None) -> dict[str, Any]:
    """The JSON record: conditions, every row, and the ranges a report may quote."""
    return {
        "tool": "tools/sustained_trials.py",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "conditions": {
            "seconds": float(seconds),
            "warmup_s": float(warmup_s),
            "repeats": int(repeats),
            "interleaving": "condition-major; within a condition, rounds of all devices "
                            "with the order rotated one place per round",
            "order": [f"rep{repeat}:{device}"
                      for repeat, device in interleaved_order(DEVICE_IDS, repeats)],
            "notes": dict(_CONDITION_NOTES),
            "wall_s": None if wall_s is None else round(float(wall_s), 1),
        },
        "machine": machine_conditions(),
        "census": dict(census or {}),
        "results": [result.as_dict() for result in results],
        "ranges": [
            {"condition": condition, "device": device, "summary": line}
            for condition, device, line in per_condition_rows(results)
        ],
    }


# --------------------------------------------------------------------------------------
# The command line.
# --------------------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/sustained_trials.py",
        description="Sustained paired device trials: the basis for a device comparison.")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                        help=f"length of each measured window (default {DEFAULT_SECONDS}).")
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP,
                        help=f"identical warm-up before every window (default {DEFAULT_WARMUP}).")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"repeats per device per condition (default {DEFAULT_REPEATS}).")
    parser.add_argument("--devices", default=",".join(DEVICE_IDS),
                        help="comma-separated explicit targets. AUTO/HETERO are refused.")
    parser.add_argument("--conditions", default=",".join(CONDITIONS),
                        help=f"any of {','.join(CONDITIONS)}.")
    parser.add_argument("--json", dest="json_path", default=str(OUTPUT_JSON),
                        help="where to write the record.")
    parser.add_argument("--quiet", action="store_true", help="print only the table.")
    parser.add_argument("--table", action="store_true", help="print the table and exit "
                                                             "without running anything.")
    parser.add_argument("--from-json", dest="from_json", default=None,
                        help="print the table of an existing record instead of measuring.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--spec", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None, help=argparse.SUPPRESS)
    return parser


def _split(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def _results_from_record(path: Path) -> list[TrialResult]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{path} carries no results rows; nothing to print.")
    return [TrialResult.from_payload(row) for row in rows]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.child:
        if not args.spec or not args.out:
            print("--child requires --spec and --out", file=sys.stderr)
            return 2
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        payload = run_trial_in_this_process(spec)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0 if payload.get("unmeasured") is None else 3

    if args.from_json:
        print(format_table(_results_from_record(Path(args.from_json))))
        return 0

    if args.table:
        print(format_table(_results_from_record(OUTPUT_JSON)))
        return 0

    devices = _split(args.devices)
    conditions = _split(args.conditions)
    unknown = [d for d in devices if d not in DEVICE_IDS]
    if unknown:
        print(f"refusing {unknown}: this tool measures explicit targets only, one of "
              f"{list(DEVICE_IDS)}. AUTO/HETERO hide where the work ran.", file=sys.stderr)
        return 3
    try:
        from tools.input_ack_under_load import MachineNotClean, assert_machine_clean
    except Exception as exc:  # noqa: BLE001 - the census is not optional
        print(f"cannot load the machine census: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    try:
        census = assert_machine_clean()
    except MachineNotClean as exc:
        print(f"refusing to measure: {exc}", file=sys.stderr)
        return 3

    started = time.perf_counter()
    results: list[TrialResult] = []
    try:
        run_paired_trials(
            devices=devices, conditions=conditions, repeats=int(args.repeats),
            seconds=float(args.seconds), warmup_s=float(args.warmup),
            progress=None if args.quiet else (lambda line: print(line, flush=True)),
            collected=results)
    except MachineNotClean as exc:
        print(f"refusing the rest of the run: {exc}", file=sys.stderr)
        print(f"{len(results)} trial(s) had already run; their rows are written.",
              file=sys.stderr)
        return 3
    finally:
        if results:
            wall_s = time.perf_counter() - started
            record = build_record(results, seconds=float(args.seconds),
                                  warmup_s=float(args.warmup), repeats=int(args.repeats),
                                  census=census, wall_s=wall_s)
            target = Path(args.json_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(f"\n{format_table(results)}")
            print(f"\nwrote {target}  (wall {wall_s:.1f} s)")
    failures = sum(1 for result in results if result.unmeasured is not None)
    if failures:
        print(f"{failures} of {len(results)} trials did not run; their rows say why.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
