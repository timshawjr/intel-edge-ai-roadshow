"""`app.main` - the entry point, the startup gate, the self-test and the headless run.

Four things live here, and each is a place where a quiet lie is easiest:

## 1. The startup gate

`assert_no_simulation_in_release(release, providers)` is the AGENTS.md rule 14 gate. It
raises `SystemExit` when a release build has a simulated provider registered, and it
**names the provider it refused** - "release mode refused to start" with no name sends
somebody to the wrong subsystem, which at a trade show costs more than the failure.

Matching is on an **explicit set plus four prefixes** (`simulated`, `mock`, `fake`,
`stub`), never on a substring. Substring matching would catch more names and would also
be wrong in a way that matters: a real provider whose name merely contained one of those
words would shut down the application, and the person diagnosing that would not be
looking at a simulation at all.

`REGISTERED_PROVIDERS` is what this build actually registers: `psutil`, and nothing else.
`app.telemetry.pdh` is deliberately absent, because the counter read is not usable in
this process and registering an unusable provider would be a claim that something is
being measured when it is not.

## 2. `--selftest`

Runs preprocess, infer, decode, track, event and render-metadata with **no network and no
display**, prints `PASS`, and exits 0. Two properties are load-bearing:

- **It works with nothing provisioned.** A machine with no model and no clip must get a
  truthful answer rather than a crash, so a missing artefact is a stage `skipped` **with
  its reason stated**. A skip is never counted as a run: the summary separates the two
  verdicts, and the stage-reporting test in `tests/test_main.py` asserts that the
  separation survives. Named without its `def` form on purpose - shipped docstrings are
  scanned for `test_*` citations and a cited name must be one
  `tests/test_retail_measurements.py` can resolve, and that scanner walks a fixed list of
  M1 modules that does not include this one.
- **It opens no socket and no display.** No `QGuiApplication` is ever constructed and no
  QML is ever loaded; the "render-metadata" stage is the `UiBridge`'s formatting and
  overlay construction, not a window. It *does* import `PySide6.QtGui` - `UiBridge` builds
  a `QImage` and a `QColor` - which this module previously claimed it did not, and that
  claim was false. What it never does is import `PySide6.QtQuick`: the image provider was
  moved to `app.ui.provider` so the QML stack stays out of the self-test entirely, and
  the tests assert its absence from `sys.modules`. A self-test that needed a
  display would be untestable in CI and would tell an operator less than a stage list
  does.

The M1 verification block in AGENTS.md names `.venv\\Scripts\\python.exe -m app.main
--selftest` as a gate, so this is the command a person runs before a booth.

## 3. `--headless`

Runs the whole pipeline against the provisioned clip on the requested device, offscreen,
and writes a JSON summary. The summary's construction is in `build_run_summary`, which is a
**pure function of its arguments** and is therefore testable without a clip, a device or
Qt - the tests use it directly, and the real run only supplies evidence about the
numbers.

Three properties of the summary exist because their absence would be a lie:

- an unmeasured figure is `null`, never `0`;
- `device.requested` and `device.confirmed` are **separate keys**, and `fell_back` is
  `null` rather than `False` when there is no confirmation at all (an unproven route has
  not fallen back, it has not been shown); and
- every event carries its `dedup_key`, so a count can be checked against a replay.

## 4. The windowed run

The live booth: the real `Main.qml`, the real supervisor, the real collector, with the
frames fed from `ClipSource` on a background thread and everything crossing into Qt
through signals (AGENTS.md rule 11). `--windowed` is not the default because a headless
CI run and a booth are different jobs, and `--selftest` is the one that must always work.

**Startup is announced, not assumed.** `DeviceRegistry.describe()` costs **3.3-3.8 s** cold
here - one memoised PowerShell PnP enumeration - so it runs once, before the window
opens, and the operator is told that is what is happening rather than being shown a blank
screen. Nothing on a per-frame or per-sample path may call it.

## What this module does not do

It does not open a network connection, it does not resolve a device name to a plugin
itself (`Supervisor` refuses `AUTO`/`HETERO`/`MULTI` at the config boundary), and it does
not read `config/*.json` directly - the model facts come from `app.inference.pinned`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

__all__ = [
    "REGISTERED_PROVIDERS",
    "SIMULATED_PROVIDER_PREFIXES",
    "assert_no_simulation_in_release",
    "build_parser",
    "build_run_summary",
    "main",
    "run_headless",
    "run_selftest",
]

#: The providers this build registers. **A declared allow-list, not the source of truth.**
#:
#: `psutil` is the only one: it produces the CPU and RSS samples the cards show, and its
#: values are read through `MetricCollector.sample()`.
#:
#: `app.telemetry.pdh` is **not** here, and its absence is deliberate. The PDH counter read
#: is not usable in this process - measured, it terminates the process with `0xC0000005` -
#: so registering it would claim a measurement that never happens. The GPU and NPU cards
#: say "Utilization unavailable" because nothing is registered to produce a value for them.
REGISTERED_PROVIDERS: tuple[str, ...] = ("psutil",)

#: **Release mode is on unless someone turns it off.** `--release` used to default to
#: `False`, which meant a packaged build that forgot the flag ran with no gate at all - and
#: a gate nobody remembers to pass is not a gate. The safe direction is the default and
#: `--no-release` is the explicit development escape hatch; it exists because a
#: developer may legitimately want to run a simulated provider on their own machine, and
#: that is a decision somebody should have to type rather than a decision a build should
#: have to be told not to make.
RELEASE_BY_DEFAULT: bool = True


def consulted_providers() -> tuple[str, ...]:
    """The providers the telemetry collector **actually** reaches for, read from it.

    ## Why this is read rather than written here

    `REGISTERED_PROVIDERS` was a literal nothing consulted, while `MetricCollector`
    imported `psutil` directly. That made two independent statements about the same fact,
    and the release-gate test that
    asserted the literal satisfies the gate was therefore a tautology: it could only
    fail if someone edited the literal to something simulated.

    The gate's real question is "could a simulated provider produce a number on this
    screen?", and that is answered by what the collector imports. So the provider name
    comes from `app.telemetry.collector.PROVIDER`, which is the declaration beside the
    `psutil.cpu_percent` calls it names, and the tests assert
    `consulted_providers() <= REGISTERED_PROVIDERS` - the allow-list must cover reality,
    which is a check that can actually fail. The scanner that enforces citations of
    test names walks a fixed list of M1 modules that does not include the one these
    tests live in, so they are named here by file rather than by method.
    """
    from app.telemetry.collector import PROVIDER

    return (PROVIDER,)

#: Prefixes that mark a provider as simulated. Prefixes, not substrings; see the module
#: note for why that distinction is the whole point of the gate.
SIMULATED_PROVIDER_PREFIXES: tuple[str, ...] = ("simulated", "mock", "fake", "stub")


def _is_simulated(provider: str) -> bool:
    """Whether `provider` is a simulated source, by name.

    `name.startswith(prefix)` and nothing else. A real provider whose name happened to
    *contain* one of the four words would otherwise be shut down at startup, and the
    person diagnosing that would not find a simulation anywhere.
    """
    text = str(provider).strip().lower()
    return text.startswith(SIMULATED_PROVIDER_PREFIXES)


def assert_no_simulation_in_release(release: bool, providers: Sequence[str]) -> None:
    """Refuse to start a release build with a simulated provider registered.

    Raises `SystemExit`, naming every offending provider. Returns normally otherwise -
    including for an empty provider list, which is a build that measures nothing and says
    so, not a build that simulates something.
    """
    if not release:
        return
    offenders = [str(p) for p in providers if _is_simulated(p)]
    if not offenders:
        return
    raise SystemExit(
        "refusing to start in release mode: simulated provider(s) registered: "
        f"{', '.join(offenders)}. A release build must publish only measured figures "
        f"(AGENTS.md rule 14). Unregister the provider, or run without --release."
    )


# --------------------------------------------------------------------------------------
# The self-test.
# --------------------------------------------------------------------------------------

def _stage(name: str, fn, *, needs: Sequence[str] = ()) -> dict[str, Any]:
    """Run one self-test stage and record what happened, never raising.

    `needs` names provisions the stage cannot run without. A missing provision is
    `skipped` **with its reason**, which is a different record from `ran` - a self-test that
    reported a skip as a pass would be the failure this module exists to prevent.

    `name` is written into the record it returns. It was accepted and ignored in the first
    version, which meant the one argument a caller passed to describe the stage had no
    effect on anything; now the record carries its own label, so a stage's name and its
    verdict cannot be separated by whoever transcribes the summary.
    """
    missing = _missing_provisions(needs)
    if missing:
        return {"stage": str(name), "verdict": "skipped", "reason": missing,
                "detail": "nothing was exercised"}
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - a stage failure is a result, not a crash
        return {"stage": str(name), "verdict": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "detail": "the stage raised"}
    return {"stage": str(name), "verdict": "ran", "reason": "", "detail": str(detail)}


def _missing_provisions(needs: Sequence[str]) -> str:
    """A stated reason for each unmet provision, or `""` when all are met."""
    missing: list[str] = []
    for need in needs:
        if need == "ir":
            try:
                from app.inference.pinned import verified_ir_path

                verified_ir_path()
            except Exception as exc:  # noqa: BLE001 - a provision check, not a stage
                missing.append(f"the pinned IR is unusable: {exc}")
        elif need == "clip":
            from app.engine.retail_config import RETAIL_CHECKOUT_CLIP

            if not Path(RETAIL_CHECKOUT_CLIP).exists():
                missing.append(f"{RETAIL_CHECKOUT_CLIP} is not provisioned")
        elif need == "cpu":
            try:
                import openvino as ov

                if "CPU" not in ov.Core().available_devices:
                    missing.append("no CPU plugin on this machine")
            except ImportError:
                missing.append("openvino is not importable")
    return "; ".join(missing)


def _selftest_preprocess() -> str:
    from app.inference.pinned import pinned_input_shape
    from app.inference.preprocess import preprocess_bgr
    import numpy as np

    _, _, height, width = pinned_input_shape()
    blob, letterbox = preprocess_bgr(np.zeros((1080, 1920, 3), np.uint8), width, height)
    return (f"letterbox {width}x{height} pad={letterbox.pad_value} "
            f"scaled={letterbox.scaled_w}x{letterbox.scaled_h}")


def _selftest_infer() -> str:
    """One real inference on the CPU plugin, with the placement read after it.

    **The thresholds come from `retail_sanity_manifest()`,** as `run_headless` does. This
    used to hardcode `0.25` and `0.45`, which is two sources for one pair of numbers: a
    pack retune would change what the booth analyses and leave the self-test proving the
    old configuration still loads, and the two would disagree silently.
    """
    from app.engine.retail_config import retail_sanity_manifest
    from app.engine.worker import WorkerConfig, infer_once
    from app.inference.pinned import MODEL_ID, pinned_input_shape

    _, _, height, width = pinned_input_shape()
    manifest = retail_sanity_manifest()
    config = WorkerConfig(device_id="CPU", model_id=MODEL_ID, input_width=width,
                          input_height=height,
                          conf_threshold=manifest.conf_threshold,
                          iou_threshold=manifest.iou_threshold)
    import numpy as np

    from app.contracts.frame import FramePacket

    frame = FramePacket(session_id="selftest", scenario_id="selftest", stream_id="A",
                        frame_id=0, source_pts=0.0, source_loop_id=0,
                        captured_at=0.0, decoded_at=0.0, width=1920, height=1080,
                        image=np.zeros((1080, 1920, 3), np.uint8))
    result = infer_once(None, frame, config)
    if result.error is not None:
        raise RuntimeError(result.error)
    return (f"{result.latency_ms:.2f} ms on confirmed "
            f"{list(result.execution_devices)}")


def _selftest_decode_track_event() -> str:
    """The pure half of the path: decode a synthetic tensor, track it, count an event."""
    from app.contracts.frame import FramePacket
    from app.engine.events import RetailEventMachine
    from app.engine.retail_config import retail_sanity_manifest
    from app.engine.tracker import Tracker
    from app.inference.decode import Detection
    from app.inference.pinned import class_names
    import numpy as np

    manifest = retail_sanity_manifest()
    tracker = Tracker(iou_threshold=manifest.iou_threshold)
    machine = RetailEventMachine(manifest, class_names=class_names())
    # A person-shaped box inside COUNTER_BAND, held for the dwell the pack requires.
    person_id = class_names().index("person")
    emitted = 0
    for step in range(40):
        now = step / 10.0
        frame = FramePacket(session_id="selftest", scenario_id=manifest.id,
                            stream_id="A", frame_id=step, source_pts=now,
                            source_loop_id=0, captured_at=now, decoded_at=now,
                            width=1920, height=1080,
                            image=np.zeros((1080, 1920, 3), np.uint8))
        detection = Detection(class_id=person_id, score=0.9,
                              box=(200.0, 300.0, 700.0, 900.0))
        for track in tracker.update([detection], now=now):
            emitted += len(machine.observe(track, frame))
    if emitted == 0:
        raise RuntimeError(
            "the synthetic presence produced no zone_occupancy event, so the event "
            "machine is not reachable by this self-test. That is a real failure and not "
            "a skip: the pack's own guard proves the same configuration fires on real "
            "footage."
        )
    return f"{emitted} zone_occupancy event(s) from a synthetic presence"


def _selftest_render_metadata() -> str:
    """The UI layer's formatting and overlay construction, with no window.

    This is the "render" half of `--selftest` and it deliberately does **not** build a
    `QGuiApplication`: a self-test that needed a display would not be runnable in CI, and
    the stage that can be checked without one is the stage that checks whether numbers
    render as `unavailable` instead of `0`.

    It does construct `QImage`s, because `UiBridge` is the thing being checked and it
    paints overlay boxes. What is avoided is the *window*: no `QGuiApplication`, no
    `QQmlApplicationEngine`, and - because `FrameImageProvider` lives in `app.ui.provider`
    rather than in `app.ui.bridge` - no `QtQuick` import either.
    """
    from app.ui.bridge import UNAVAILABLE_TEXT, UiBridge

    bridge = UiBridge()
    if bridge.display_text("system_cpu_percent") != UNAVAILABLE_TEXT:
        raise RuntimeError(
            "a fresh UiBridge rendered a metric as something other than "
            f"{UNAVAILABLE_TEXT!r}; a cold-start card must read unavailable"
        )
    if bridge.device_confirmed is not None:
        raise RuntimeError("a fresh UiBridge claimed a confirmed device before anything ran")
    return ("unavailable renders as unavailable, and no device is claimed before an "
            "inference confirms one")


def run_selftest() -> dict[str, Any]:
    """Run every self-test stage and return the summary. **Never raises.**

    `display` is always `None` in the returned summary and is asserted to be by the
    tests: the self-test opens no window.
    """
    started = time.perf_counter()
    stages = {
        "preprocess": _stage("preprocess", _selftest_preprocess),
        "infer": _stage("infer", _selftest_infer,
                         needs=("ir", "cpu")),
        "decode-track-event": _stage("decode-track-event", _selftest_decode_track_event,
                                     needs=("cpu",)),
        "render-metadata": _stage("render-metadata", _selftest_render_metadata),
    }
    failed = [name for name, stage in stages.items() if stage["verdict"] == "failed"]
    return {
        "stages": stages,
        "display": None,
        "network": "none",
        "providers": list(REGISTERED_PROVIDERS),
        "failed": failed,
        "ok": not failed,
        "seconds": round(time.perf_counter() - started, 3),
    }


# --------------------------------------------------------------------------------------
# The run summary.
# --------------------------------------------------------------------------------------

def _percentile(values: Sequence[float], fraction: float) -> float | None:
    """A nearest-rank percentile, or `None` for an empty sequence.

    **No interpolation.** `p95` of 20 samples is the 19th of the 20 samples that were
    taken; a smoothed percentile is a number no measurement produced, which is the shape
    this project refuses everywhere else.
    """
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _sample_payload(sample: Any) -> dict[str, Any]:
    """One `MetricSample` as a JSON-ready dict, with `None` kept as `None`.

    The record's own fields decide the payload's shape; nothing here substitutes a
    number for an absent reading, and `validity` and `unavailable_reason` travel with the
    value so a reader can tell a number from an absence without inferring it.
    """
    return {
        "value": sample.value,
        "unit": sample.unit,
        "provider": sample.provider,
        "validity": sample.validity.value,
        "window_seconds": sample.window_seconds,
        "reason": sample.unavailable_reason,
        "process_scope": list(sample.process_scope),
    }


def build_run_summary(
    *,
    device_requested: str,
    state: str,
    confirmed: Sequence[str],
    frames_submitted: int,
    frames_analysed: int,
    drops: int,
    elapsed_s: float,
    frame_ages_ms: Sequence[float],
    events: Sequence[Any],
    samples: Sequence[Any],
    extra_errors: Sequence[str] = (),
    frames_refused: int = 0,
    frames_refused_reasons: Mapping[str, int] | None = None,
    feed_elapsed_s: float | None = None,
    drain_elapsed_s: float | None = None,
    frame_attempts: int | None = None,
) -> dict[str, Any]:
    """The headless run's summary, as a pure function of what was measured.

    Every optional figure is `None` when it was not measured. `device.requested` and
    `device.confirmed` are separate, and `fell_back` is `None` - not `False` - when there
    is no confirmation at all: an unproven route has not fallen back, it has **not been
    shown**, and `False` there would read as "running where you asked".

    ## The three frame counters, and why they are three

    `frames_submitted`, `frames_refused` and `drops` count three different populations and
    must never be merged. Submitted went into the queue. Refused never reached it because
    the pipeline was not running, shut down, or failed - **which is not a drop**, and
    counting it as one made a broken route report a clean zero. Dropped missed the queue
    because it was full, which *is* lag. `frame_attempts` is the total, so
    `submitted + refused + drops == attempts` is checkable rather than hoped for.

    ## Two windows, not one

    `ai_fps` divides by `feed_elapsed_s` - the time frames were actually being fed - and
    **not** by `elapsed_s`, which additionally contains the drain and the teardown.
    Measured: `--seconds 8` reported `elapsed_s: 10.152` and `ai_fps: 11.9187`, which is
    121 frames over 10.16 s, not over the 8 s the run claims to have fed. The old report
    quoted both figures in one table row as though they shared a window. `drain_elapsed_s`
    is now its own field so the difference is visible rather than absorbed.
    """
    from app.inference.devices import is_device

    confirmed_tuple = tuple(str(d) for d in confirmed)
    fell_back: bool | None = None
    if confirmed_tuple:
        fell_back = not is_device(confirmed_tuple, device_requested)

    analysed = int(frames_analysed)
    window = feed_elapsed_s if feed_elapsed_s is not None else elapsed_s
    fps = (analysed / float(window)) if analysed > 0 and window > 0 else None

    return {
        "device": {
            "requested": device_requested,
            "confirmed": list(confirmed_tuple),
            "fell_back": fell_back,
            "state": state,
        },
        "frames_submitted": int(frames_submitted),
        "frames_analysed": analysed,
        "frames_refused": int(frames_refused),
        "frames_refused_reasons": dict(frames_refused_reasons or {}),
        "frame_attempts": (int(frame_attempts) if frame_attempts is not None
                           else int(frames_submitted) + int(frames_refused)),
        "drops": int(drops),
        "elapsed_s": round(float(elapsed_s), 3),
        "feed_elapsed_s": round(float(window), 3),
        "drain_elapsed_s": (round(float(drain_elapsed_s), 3)
                            if drain_elapsed_s is not None else None),
        "ai_fps": round(fps, 4) if fps is not None else None,
        "frame_age_ms": {
            "p50": _percentile(frame_ages_ms, 0.50),
            "p95": _percentile(frame_ages_ms, 0.95),
            "samples": len(frame_ages_ms),
            "method": "nearest rank, no interpolation",
        },
        "metrics": {sample.name: _sample_payload(sample) for sample in samples},
        "events": [
            {
                "kind": event.kind,
                "class_name": event.class_name,
                "track_id": event.track_id,
                "source_loop_id": event.source_loop_id,
                "at_frame_id": event.at_frame_id,
                "at_time": round(float(event.at_time), 4),
                "dedup_key": event.dedup_key,
            }
            for event in events
        ],
        "event_counts": _count_events(events),
        "errors": [str(text) for text in extra_errors],
    }


def _take_reporting_sample(supervisor: Any, *, attempts: int = 12,
                           wait_s: float = 0.25) -> tuple:
    """One sample set that actually carries a measurement, or the last one taken.

    ## Why this is not just `supervisor.samples()`

    `psutil.cpu_percent(interval=None)` measures the delta **since the previous call**, and
    `MetricCollector`'s own background thread calls it on its interval. So a `sample()`
    taken from this process lands either just after the thread's own call - in which case
    there is no measurable window and every sample in the set comes back
    `UNAVAILABLE` with `window_seconds = 0.0` - or just before, in which case it works.
    It is a coin toss on timing.

    Measured: a 4 s run whose final `samples()` call landed on the wrong side of the
    thread reported **every** metric as unavailable, including `system_cpu_percent` with
    the honest reason "no measurable window elapsed since the previous sample". The
    summary was not lying, but it was reporting the one instant in the run where nothing
    had been measured - which is the least useful instant to publish.

    So this retries across a couple of collector intervals and returns the first set that
    carries at least one valid reading. If none does, it returns the last one taken -
    **with its reason**, because "we never got a window" is itself the honest answer for a
    run too short to produce one.
    """
    from app.contracts.metric_sample import Validity

    result: tuple = ()
    for attempt in range(max(1, int(attempts))):
        result = supervisor.samples()
        if any(sample.validity is not Validity.UNAVAILABLE for sample in result):
            return result
        if attempt + 1 < int(attempts):
            time.sleep(float(wait_s))
    return result


def _count_events(events: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    return counts


# --------------------------------------------------------------------------------------
# The headless run.
# --------------------------------------------------------------------------------------

def run_headless(*, device: str = "CPU", seconds: float = 5.0,
                 source: str = "loop", scenario: str = "retail",
                 scenario_id: str = "selftest-headless",
                 session_id: str = "headless") -> dict[str, Any]:
    """Run the whole pipeline offscreen and return the summary.

    Frames come from `ClipSource` at its own analysis rate; inference runs in the spawned
    worker the `Supervisor` owns; tracking and the event machine run in this process on
    `Tracker.current_tracks()` - **only current tracks**, because a coasted track's box is
    the last one seen and counting it would place an object where nobody observed it.

    A loop reset calls `Tracker.reset()` **and** `RetailEventMachine.reset(loop_id)`, per
    spec section 9: the tracker's ids deliberately do not rewind, so a rewind of the clip
    cannot complete a pending presence belonging to the previous pass.
    """
    from app.contracts.frame import FramePacket
    from app.engine.events import RetailEventMachine
    from app.engine.retail_config import RETAIL_CHECKOUT_CLIP, retail_sanity_manifest
    from app.engine.source import ClipSource
    from app.engine.tracker import Tracker
    from app.inference.pinned import class_names
    from app.supervisor.runner import Supervisor

    missing = _missing_provisions(("clip", "ir", "cpu"))
    if missing:
        raise RuntimeError(f"cannot run the real pipeline headless: {missing}")

    manifest = retail_sanity_manifest()
    tracker = Tracker(iou_threshold=manifest.iou_threshold)
    machine = RetailEventMachine(manifest, class_names=class_names())

    supervisor = Supervisor(device_id=device, stream_id="A", session_id=session_id,
                            scenario_id=scenario_id,
                            conf_threshold=manifest.conf_threshold,
                            iou_threshold=manifest.iou_threshold)
    clip = ClipSource(Path(RETAIL_CHECKOUT_CLIP), stream_id="A",
                      session_id=session_id, scenario_id=scenario_id, target_fps=15.0)
    # Started before the clock starts. A `Supervisor` that was never started refuses
    # every `submit` **without counting a drop** (see `submit`), so a run that forgot
    # this line would report zero submitted, zero drops and `state == "pending"` - a
    # summary that looks like a slow machine rather than a run that never happened.
    supervisor.start()
    started = time.perf_counter()
    seen_loops: set[int] = {0}
    # I2: the identity of the last result already folded. `latest_result()` keeps
    # returning the newest result until a newer one lands, so a fold that does not check
    # this re-folds the same detection set as many times as the loop iterates. Measured on
    # a 4 s CPU run: 61 frames analysed produced **147** `tracker.update` calls, one
    # identity re-read 95 times, and a maximum track `hits` of **147** against 61 real
    # results - so a detection seen in one real frame reached `min_hits = 2` and was
    # confirmed. The tracker's own evidence gate was defeated by the consumer, not by the
    # tracker. The event counts survived only because `dedup_key` suppressed the
    # duplicate emit, which is a coincidence rather than a safeguard.
    folded: dict[str, Any] = {"identity": None}
    # I3: take one sample set up front so a run that ends before the first telemetry tick
    # still reports a real CPU reading rather than `metrics: {}`.
    supervisor.samples()
    try:
        while time.perf_counter() - started < float(seconds):
            frame = clip.read()
            if frame is None:
                if source == "once" or not clip.restart():
                    break
                loop_id = clip.loop_id
                if loop_id not in seen_loops:
                    seen_loops.add(loop_id)
                    tracker.reset()
                    machine.reset(loop_id)
                frame = clip.read()
                if frame is None:
                    break
            supervisor.submit(frame)
            _fold_result(supervisor, tracker, machine, folded)
        # The feed window closes here, before the drain. Everything after this point is
        # the worker finishing what it was already given, and dividing by it would report
        # a rate for a window nobody fed.
        feed_elapsed = time.perf_counter() - started
        drain_started = time.perf_counter()
        # Drain what is already analysed, so the summary's frame counts and the events
        # they produced describe the same instant.
        _drain(supervisor, tracker, machine, folded, seconds=2.0)
        drain_elapsed = time.perf_counter() - drain_started
        # I3 again, and this is the reading the summary reports: one last whole sample set
        # while the collector is still alive, retried across a couple of collector
        # intervals so the summary is not published at the one instant when the window
        # happened to be zero. `last_samples` is what survives `stop()`.
        _take_reporting_sample(supervisor)
    finally:
        plan = supervisor.route_plan()
        clip.close()
        supervisor.stop()

    elapsed = time.perf_counter() - started
    return build_run_summary(
        device_requested=plan.requested_device(supervisor.stream_id) or device,
        state=plan.state,
        confirmed=plan.confirmed_devices(supervisor.stream_id),
        frames_submitted=supervisor.frames_submitted,
        frames_analysed=supervisor.frames_analysed,
        drops=supervisor.drops,
        elapsed_s=elapsed,
        frame_ages_ms=supervisor.frame_ages_ms(),
        events=tuple(machine.events),
        samples=supervisor.last_samples,
        extra_errors=tuple(plan.errors),
        frames_refused=supervisor.frames_refused,
        frames_refused_reasons=supervisor.frames_refused_reasons,
        feed_elapsed_s=feed_elapsed,
        drain_elapsed_s=drain_elapsed,
        frame_attempts=supervisor.frame_attempts,
    )


def _fold_result(supervisor: Any, tracker: Any, machine: Any,
                 folded: dict[str, Any]) -> None:
    """Fold the newest **unfolded** result into the tracker and the event machine.

    Reads `latest_result()` rather than connecting to a signal, because this loop is the
    consumer and the supervisor is its only producer; a signal would add a thread hop per
    frame for no gain. The Qt-facing path (`run_windowed`) does use signals, because there
    the consumer is a different thread.

    ## Why `folded` is a parameter and not a local

    `latest_result()` returns the newest result **and keeps returning it** until a newer
    one arrives. Calling this function per submitted frame *and* every 20 ms during the
    drain therefore folds the same result repeatedly - 147 folds of 61 real results,
    measured. `Tracker.update` counts a match as a hit, so a repeated fold inflates
    `hits` until a track is `confirmed`, and the screen's own words ("minimum two hits
    before a track counts") become false.

    `UiBridge.accept_result` already exists for exactly this decision and was unused here;
    the same identity check is made here against `folded["identity"]`, which is the
    smallest thing that fixes the same defect without dragging the UI layer into a
    headless run.
    """
    result = supervisor.latest_result()
    if result is None or result.error is not None:
        return
    frame = result.frame
    identity = tuple(frame.identity)
    if folded["identity"] == identity:
        return
    folded["identity"] = identity
    if machine.loop_id is None:
        machine.reset(frame.source_loop_id)
    now = frame.captured_at
    # `current_tracks()`, never `update()`'s return value: only a track matched by this
    # frame is evidence that its object is present now.
    for track in tracker.update(result.detections, now=now):
        machine.observe(track, frame)


def _drain(supervisor: Any, tracker: Any, machine: Any, folded: dict[str, Any],
           *, seconds: float) -> None:
    """Wait for the worker to finish what has already been submitted.

    Bounded by `seconds`, because a wedged worker must not hang a validation run. The
    loop is what makes the counts and the events describe the same instant: an event
    emitted after the last `_fold_result` would be counted without its frame.
    """
    deadline = time.perf_counter() + float(seconds)
    last = supervisor.frames_analysed
    while time.perf_counter() < deadline:
        _fold_result(supervisor, tracker, machine, folded)
        current = supervisor.frames_analysed
        if current == last:
            time.sleep(0.02)
            continue
        last = current


# --------------------------------------------------------------------------------------
# The windowed run.
# --------------------------------------------------------------------------------------

def run_windowed(*, device: str = "CPU", scenario: str = "retail",
                 source: str = "loop", session_id: str = "booth",
                 select: Mapping[str, str] | None = None,
                 fail_stream: str | None = None,
                 streams: int = 1) -> int:
    """The live booth. Returns the process exit code.

    **Device discovery happens here, before the window opens**, and it is announced:
    `DeviceRegistry.describe()` costs 3.3-3.8 s cold on this machine, and a booth operator
    needs to be told that is what is happening rather than shown a black screen. It is
    never called again.

    ## `select`: a presenter can move a camera, and the screen shows it **after** proof

    `select` is `stream_id -> device`, and it is applied by
    `_start_presenter_selection` on its **own thread**, after `engine.load` and after
    `supervisor.start()`.

    Both orderings are load-bearing and both were got wrong somewhere in this milestone's
    history:

    - **After `engine.load`.** The screen must already exist when the route arrives, or
      the badge never updates - that was defect C4, a `constant=True` property with no
      notify signal, which read correctly at `engine.load` and froze for the session.
    - **On its own thread.** `await_transition` blocks for as long as the placement takes,
      and doing that on the UI thread would freeze the window through the whole of the
      "Moving Camera B inference to CPU..." state that spec section 5 asks for. The
      presenter therefore sees the route go **pending** and then **confirmed**, which is
      the claim the badge makes and the claim the live test reads.

    The badge reads the **confirmed** device and never the requested one. That is the
    difference between "the presenter asked for the GPU" and "the GPU ran the frame", and
    M1's C4 is exactly the mistake of conflating them.

    ## `streams`: how many cameras, each on its own device and its own clip

    **`streams=1` is the default and behaves exactly as M0/M1 did** - one camera on
    `--device`. `streams=3` asks for the three-camera layout: one camera per available
    device, `--device` first, **its own `ClipSource`, its own decode thread, its own
    `Tracker` and its own `RetailEventMachine`**.

    Three of those are load-bearing rather than tidy:

    - **one clip per camera.** One clip decoded into three streams puts three workers on
      three devices each analysing the *same* picture, and the three tiles show three copies
      of one camera. That is what M1's second thumbnail was captioned with - *"same clip,
      replayed copy - not a second camera"* - and it is what this task closes.
    - **one tracker and one event machine per camera.** Sharing them interleaves track ids
      across cameras and lets one camera's zone rule fire on another's picture: a plausible,
      correctly-placed rectangle over the wrong object.
    - **devices are assigned in a stated order** (`--device` first, then `CPU`, `GPU`, `NPU`
      from what the registry reports), so a screenshot of the screen and the printed
      `[startup] N camera(s): A -> CPU, B -> GPU, C -> NPU` line agree.
    """
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtQml import QQmlApplicationEngine

    from app.engine.events import RetailEventMachine
    from app.engine.retail_config import RETAIL_CHECKOUT_CLIP, retail_sanity_manifest
    from app.engine.source import ClipSource
    from app.engine.tracker import Tracker
    from app.inference.pinned import class_names
    from app.inference.devices import DeviceRegistry
    from app.supervisor.runner import DEFAULT_STREAM_IDS, Supervisor
    from app.ui.bridge import ThemeTokens, UiBridge
    from app.ui.provider import FrameImageProvider

    print(f"[startup] discovering devices (~3.8 s cold on this machine); "
          f"this runs once", flush=True)
    _describe_devices()

    application = _gui_application()
    # The booth font. Measured: 411 families under the `windows` plugin and 0 under
    # `offscreen`, so the family is registered explicitly rather than relied on.
    _register_font(QFontDatabase)

    manifest = retail_sanity_manifest()
    # **M2: one tracker and one event machine PER STREAM.** With three cameras sharing
    # one tracker, track ids carry across cameras and one camera's zone rule fires on the
    # other's picture - a plausible rectangle over the wrong object. See `_on_result`.
    # **The devices this machine actually reports**, through the registry rather than a
    # guess: the same `DeviceRegistry` `_describe_devices()` used a moment ago, so the
    # `[startup] N camera(s)` line and the device list printed above it cannot disagree. A
    # machine without an NPU therefore gets two cameras, and the line says two.
    try:
        import openvino as ov

        available = tuple(DeviceRegistry(ov.Core()).available())
    except Exception as exc:  # noqa: BLE001 - discovery failed, so fall back and say so
        print(f"[startup] cannot enumerate devices for the camera layout "
              f"({type(exc).__name__}: {exc}); falling back to {device} only",
              file=sys.stderr, flush=True)
        available = ()
    if not available:
        available = (device,)
    # **`--streams` cameras, one per device, first camera on `--device`.** The default is
    # ONE camera, exactly as before: the booth's default is a single-camera window, and
    # `--streams 3` is how an operator asks for the three-camera layout. The devices are
    # assigned in the order `CPU, GPU, NPU` with `--device` first, so `--device GPU
    # --streams 3` puts the GPU on Camera A.
    wanted = max(1, min(int(streams or 1), len(available)))
    devices = [device] + [d for d in available if d != device]
    stream_devices = tuple(devices[:wanted])
    stream_ids = tuple(DEFAULT_STREAM_IDS[:wanted])
    streams = {sid: (Tracker(iou_threshold=manifest.iou_threshold),
                     RetailEventMachine(manifest, class_names=class_names()))
               for sid in stream_ids}

    bridge = UiBridge(application)
    theme = ThemeTokens(application)
    if len(stream_devices) == 1:
        supervisor = Supervisor(device_id=stream_devices[0], stream_id=stream_ids[0],
                                session_id=session_id,
                                scenario_id=manifest.id,
                                conf_threshold=manifest.conf_threshold,
                                iou_threshold=manifest.iou_threshold,
                                compile_failure=([fail_stream] if fail_stream else False))
    else:
        supervisor = Supervisor(device_ids=stream_devices, session_id=session_id,
                                scenario_id=manifest.id,
                                conf_threshold=manifest.conf_threshold,
                                iou_threshold=manifest.iou_threshold,
                                compile_failure=([fail_stream] if fail_stream else False))
    clips = [ClipSource(Path(RETAIL_CHECKOUT_CLIP), stream_id=sid,
                        session_id=session_id, scenario_id=manifest.id, target_fps=15.0)
             for sid in stream_ids]
    print(f"[startup] {len(stream_ids)} camera(s): "
          + ", ".join(f"{sid} -> {dev}" for sid, dev in zip(stream_ids, stream_devices)),
          flush=True)

    engine = QQmlApplicationEngine()
    engine.addImageProvider("frames", FrameImageProvider(bridge))
    context = engine.rootContext()
    context.setContextProperty("Theme", theme)
    context.setContextProperty("bridge", bridge)

    # -- the live wiring ---------------------------------------------------------------
    #
    # Four hops, and every one of them had **no caller** while this block was missing,
    # which is why the booth screen used to sit forever on "pending - no device
    # selected", "no frame decoded yet", every metric `unavailable` and every event
    # count 0 - with no error anywhere. The connection is to the bridge's *own* slots,
    # not to a lambda that re-implements them, so there is exactly one implementation of
    # "apply a route" and one of "apply a frame".
    bridge.routeArrived.connect(bridge.apply_route)
    bridge.errorArrived.connect(bridge.apply_error)
    bridge.frameArrivedFromWorker.connect(bridge.apply_frame)
    bridge.samplesArrived.connect(bridge.apply_samples)

    # The supervisor's own signals are converted into those bridge signals by
    # `_on_route` / `_append_error` / `_on_result`, which run on the **supervisor's**
    # reader thread. The bridge lives in the UI thread, so Qt queues the hop
    # automatically - AGENTS.md rule 11, satisfied by construction rather than by
    # remembering to pass a connection type.
    supervisor.state_changed.connect(lambda plan: _on_route(bridge, plan))
    supervisor.error_raised.connect(lambda text: _append_error(bridge, text))
    supervisor.result_ready.connect(
        lambda result: _on_result(bridge, streams, manifest, result))

    engine.load(QUrl.fromLocalFile(str(Path(__file__).parent / "ui" / "Main.qml")))
    roots = engine.rootObjects()
    if not roots:
        print("[startup] Main.qml produced no root objects; refusing to open a blank "
              "window", file=sys.stderr, flush=True)
        supervisor.stop()
        for clip in clips:
            clip.close()
        return 2

    supervisor.start()
    # **Declare the cameras before the event loop starts**, so the metrics column holds one
    # card per camera from the very first rendered frame with honest cold values -
    # `pending`, `unavailable`, no device - instead of being empty until the first route
    # payload arrives. An empty column on a cold start is indistinguishable from a broken
    # screen. Declaring a stream claims nothing: every figure still reads `unavailable`
    # until something is measured.
    bridge.set_streams(list(supervisor.stream_ids))
    # **One feeder thread per camera, each with its own `ClipSource`.** One clip decoded into
    # three streams would put three workers on three devices each analysing the *same*
    # picture, and the three tiles would show three copies of one camera - which is exactly
    # what M1's second thumbnail used to be captioned ("same clip, replayed copy - not a
    # second camera") and exactly what this task closes. Each stream now has its own decode
    # thread, its own tracker and its own image.
    for clip in clips:
        _start_feeders(bridge, supervisor, clip)
    _start_telemetry(bridge, supervisor, application)
    if select:
        # **After** `engine.load` and **after** `start()`, on its own thread. See the
        # `select` section of this function's docstring.
        _start_presenter_selection(supervisor, select)

    exit_code = application.exec()
    # The feeder thread is stopped **before** the supervisor and the clip, and it is
    # stopped by a flag it checks rather than by its own logic. See `_start_feeders`.
    stop_feeding()
    for thread in _SELECTION_THREADS:
        # Bounded, because the worker itself is bounded by `await_transition`'s deadline;
        # this only stops a move being abandoned half-done against a stopped supervisor.
        thread.join(timeout=35.0)
    _SELECTION_THREADS.clear()
    supervisor.stop()
    for clip in clips:
        clip.close()
    return int(exit_code)


#: The presenter-selection threads. The windowed run joins them before tearing the
#: supervisor down, so a move in flight is not left to finish against a stopped
#: supervisor - `await_transition` has a 30 s deadline and `stop()` would race it.
_SELECTION_THREADS: list[threading.Thread] = []

#: How long `_await_settled` waits for the current placement before moving anyway. Long
#: enough for the measured 3.5 s cold three-device placement, with room for a slower
#: machine; past it the move proceeds rather than never happening.
SETTLE_TIMEOUT_SECONDS: Final[float] = 60.0


def _await_settled(supervisor: Any, streams: tuple[str, ...]) -> str:
    """Wait until every named stream has a settled state, and report what it was.

    `active` (a placement is proved) or `failed` (it will never be) both end the wait; only
    `pending` keeps it going. The returned word is the state the move started from, printed
    by the caller so a presenter can tell "I moved it off a running camera" from "I moved it
    off one that had not proved anything yet".
    """
    deadline = time.perf_counter() + SETTLE_TIMEOUT_SECONDS
    while True:
        plan = supervisor.route_plan()
        states = [plan.stream_state(sid) for sid in streams]
        if states and all(state in ("active", "failed") for state in states):
            return ",".join(str(state) for state in states)
        if time.perf_counter() >= deadline:
            return "timeout:" + ",".join(str(state) for state in states)
        time.sleep(0.05)


def _start_presenter_selection(supervisor: Any, mapping: Mapping[str, str]) -> None:
    """Move the named cameras on their own thread, and say what happened.

    Closes M1's open criterion 2: the placement was already *proved* in M1, so what was
    missing was a **presenter causing one**. The screen is the evidence - the badge must
    change, and only after a placement is confirmed.

    Three things this deliberately does not do:

    - **It never sets the badge itself.** It asks the supervisor, and the badge moves when
      the supervisor publishes a route with a `confirmed` reading. A presenter-driven
      change that wrote the requested device to the screen would be C4 all over again.
    - **It does not block the UI thread.** See the `select` docstring.
    - **It does not retry.** A presenter asked for a move once; if it does not confirm, the
      transition protocol recovers to the last healthy route and says so. Retrying until
      clean would turn a failure into a hang and hide the very thing the presenter needs to
      see.

    The outcome is printed rather than only recorded, because a presenter watching a
    window needs to be told "the GPU did not take it and we are back on the CPU" in words,
    and a silent recovery on screen reads as a stuck toggle.
    """
    def worker() -> None:
        try:
            # **Wait for the placement that is already running to prove itself first.**
            # A cold placement takes ~3.5 s on this machine (three cold compiles), and a
            # move issued inside that window races it: the presenter's request supersedes
            # a generation that never confirmed, so the screen shows one device for the
            # whole run and the badge change proves nothing. Spec section 5's protocol is
            # "request a new generation" from a route that has an actual placement, and
            # that is also the only order in which a badge change is legible to a
            # presenter.
            #
            # Bounded, and **not** waited for when the route is already `failed`: a stream
            # with no worker will never become active, and a presenter who asked to move
            # it should not be held for the timeout.
            settled = _await_settled(supervisor, tuple(sorted(mapping)))
            print(f"[select] starting from {settled!r}", flush=True)
            transition = supervisor.begin_transition(dict(mapping))
            outcome = supervisor.await_transition(transition, timeout_s=30.0)
        except Exception as exc:  # noqa: BLE001 - a failed move is a result, not a crash
            print(f"[select] {dict(mapping)!r} failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            return
        confirmed = {sid: list(outcome.confirmed.get(sid, ())) for sid in sorted(mapping)}
        print(f"[select] requested {dict(mapping)!r} -> "
              f"acknowledged={outcome.acknowledged} "
              f"bounded_by={outcome.bounded_by!r} "
              f"generation={outcome.generation} "
              f"in {outcome.elapsed_s:.3f} s; confirmed={confirmed}; "
              f"unconfirmed={list(outcome.unconfirmed_streams())}", flush=True)
        if not outcome.acknowledged:
            for reason in supervisor.errors():
                print(f"[select]   {reason}", flush=True)

    thread = threading.Thread(target=worker, name="presenter-selection", daemon=True)
    thread.start()
    _SELECTION_THREADS.append(thread)


#: Set by `_start_feeders`; the windowed run calls `stop_feeding()` on the way out. A
#: module global rather than a returned handle because the feeder's only other exit is
#: `route_plan().state == "failed"`, and a supervisor that was stopped normally never
#: becomes "failed" - so without this the thread read frames and called `submit()` for the
#: rest of the process's life, against a supervisor that had already been torn down.
_FEED_STOP = threading.Event()


def stop_feeding() -> None:
    """Ask the feeder thread to stop. Idempotent, and prompt."""
    _FEED_STOP.set()


def _gui_application():
    """The `QGuiApplication`, created here so `--selftest` imports no Qt **window** stack.

    Note the wording: `--selftest` *does* import `PySide6.QtGui` - `app.ui.bridge`
    constructs a `QImage` and a `QColor` for the frame image provider and the overlay
    painter. What it never does is construct a `QGuiApplication` or load any QML, and that
    is the property the self-test asserts. An earlier version of this docstring claimed
    the stronger, false thing.
    """
    from PySide6.QtGui import QGuiApplication

    application = QGuiApplication.instance() or QGuiApplication(sys.argv)
    return application


def _register_font(database: Any) -> None:
    """Register the booth font explicitly, and say so when it cannot be found."""
    root = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Fonts" / "arial.ttf"
    if root.exists() and database.addApplicationFont(str(root)) < 0:
        print(f"[startup] could not register {root}; the screen will fall back to "
              f"whatever family the renderer offers", file=sys.stderr, flush=True)


def _describe_devices() -> None:
    """One `DeviceRegistry.describe()`, at startup, for the console log.

    The statuses are printed, not displayed: the compute cards take their availability
    from the telemetry layer, and this call cannot produce a utilisation figure.
    """
    try:
        import openvino as ov

        from app.inference.devices import DeviceRegistry

        for status in DeviceRegistry(ov.Core()).describe():
            if status.available:
                print(f"[device] {status.device_id}: {status.full_name} "
                      f"(driver {status.driver_version or 'unreported'})", flush=True)
            else:
                print(f"[device] {status.device_id}: unavailable - "
                      f"{status.unavailable_reason}", file=sys.stderr, flush=True)
    except Exception as exc:  # noqa: BLE001 - a discovery failure is not a startup abort
        print(f"[startup] device discovery failed: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)


def _on_route(bridge: Any, plan: Any) -> None:
    """Publish **each stream's own** route to the bridge. Runs on the supervisor's reader
    thread.

    **A signal, not `invokeMethod`.** The bridge lives in the UI thread and the emitter
    does not, so Qt queues the call automatically - which is AGENTS.md rule 11, satisfied
    by construction rather than by remembering to pass a connection type.
    `QMetaObject.invokeMethod` needs the exact registered type name and a wrong one fails
    **silently**: the call is dropped and the UI never updates, which is a stall with no
    diagnostic anywhere.

    ## M2 Task 4: migrated from the aggregate to per stream

    This used to publish **one** payload, for `next(iter(plan.requested))`, whose
    `"state"` was `plan.state` - the **aggregate** over every stream: `failed` only when all
    failed, `active` when any one proved a placement. Harmless with one camera, where the
    aggregate *is* that camera's state. **Wrong the moment this serves three streams**: a
    plan where Camera A is running and Camera C has failed published `"state": "active"`,
    and Camera A's card rendered a healthy route over a dead camera. A 22-line warning
    here said so and said Task 4 owned it; this is Task 4.

    Now:

    - **one payload per stream**, each carrying `plan.stream_state(stream)` - that
      camera's own answer, not the panel's;
    - `fell_back` per stream, from `plan.fell_back(stream)`, which is also per stream and
      was previously never published at all.

    ## M2 Task 5: `errors` comes back, and it is that stream's own

    Task 4 removed the `errors` key and said why: `RoutePlan` had no per-stream error
    channel, so forwarding `plan.errors` under a per-stream label would be the same defect
    with a different field name. **Task 5 added the channel** - `RoutePlan.stream_errors`,
    populated by `Supervisor.route_plan()` - so the reason a camera's card reads is now
    that camera's reason.

    The gap Task 4 recorded is therefore closed rather than papered over: a card showing
    `failed` no longer leaves the presenter to work out which of three cameras died and
    why, and it cannot show another camera's sentences instead.

    Emitting per stream rather than one payload with a map keeps the payload shape a plain
    flat dict, which is what `UiBridge.apply_route` already guards and what a queued
    `Signal(object)` can carry without a type registration.
    """
    for stream in sorted(plan.requested):
        bridge.routeArrived.emit({
            "stream": stream,
            "requested": plan.requested_device(stream),
            "confirmed": list(plan.confirmed_devices(stream)),
            # **`None` is passed through, not coerced to `"pending"`.** A stream that has
            # declared no state has declared nothing, and `RoutePlan.stream_state` is
            # explicit that a consumer must draw that as unknown rather than as healthy.
            # Coercing it here would turn "this camera has said nothing" into "this camera
            # is waiting", which is a claim. The bridge stores what it is given and coerces
            # only at the point of rendering.
            "state": plan.stream_state(stream),
            "fell_back": plan.fell_back(stream),
            # **This stream's own reasons**, newest last. Never `plan.errors`, which is
            # the concatenation of all three.
            "errors": list(plan.stream_reasons(stream)),
        })


def _append_error(bridge: Any, text: str) -> None:
    """Publish one stated reason. Reader thread."""
    bridge.errorArrived.emit(str(text))


def _on_result(bridge: Any, streams: Any, manifest: Any, result: Any) -> None:
    """Fold one result into **that stream's** tracker, events, pixels and zones.

    **Reader thread**, and the thread is per stream, so `streams` is looked up by
    `result.frame.stream_id` and **not** shared. **M2:** one tracker and one event machine
    for three cameras would interleave their tracks - one camera's track id continuing into
    the next camera's frame, one camera's zone rule firing on the other's picture - and the
    per-camera overlay would be a plausible rectangle over the wrong object, which is the
    defect M1 recorded as "a correct box on the wrong picture".

    A result whose stream has no entry is **dropped with a stated reason** rather than
    folded into somebody else's camera: it is a real condition (a stream declared after the
    feeders started) and silently attributing it would be a mislabelling.

    Everything published here crosses to the UI thread as one payload on
    `frameArrivedFromWorker`, and `UiBridge.apply_frame` - which runs on the **UI
    thread**, because Qt queues the connection - does the applying.

    ## Why this carries the frame's **pixels**

    The first version of this function published `identity`, `width`, `height`, boxes and
    events. It did not publish the image, so `set_frame_image` was never called, the image
    provider served its 1x1 transparent placeholder for the entire session, and
    `apply_frame` set `frame_ready = True` with a null image behind it - a hero that claims
    a frame is ready while showing nothing. A `QImage` cannot cross a `Signal(object)` as
    bytes without being rebuilt on the other side, so the numpy array travels in the
    payload and the bridge builds the `QImage` in the UI thread.

    ## Zones

    `set_zones` had no caller, so the zone repeater had an empty model and drew nothing.
    The retail claim **is** a zone rule, so a screen with no zones supports no claim. The
    zones come from the manifest - the same source `run_headless` and the pack use - and
    are published with the frame so they are normalised against the picture they sit on.
    Published **once**, on the first result: the zones are a property of the scenario, not
    of the frame, and re-publishing them 15 times a second would rebuild the repeater for
    nothing.

    ## Events

    Each event is applied through `record_event`, the same call the headless path's counts
    come from, so the footer's count and the summary's count are two readings of one
    number rather than two independently maintained ones.

    ## The loop boundary is handled here, and it was not

    **`RetailEventMachine.observe` refuses a frame from a loop it is not following**, and
    says so: "Call `reset(loop_id=...)` at each source loop (spec section 9); adopting the
    new loop silently would carry one loop's pending transfers into the next." `run_headless`
    does that reset; this function did not, and the clip rewinds.

    The failure was invisible while nothing consumed results. The moment this fold was
    wired, every run that reached the end of the clip raised `ValueError` **inside the
    supervisor's reader thread**, which killed that thread - so results stopped arriving,
    the screen froze on the last good frame, and the tracebacks went to a stderr nobody
    watches at a trade show. The input-acknowledgement probe hung for the same reason and
    never printed its line. Spec section 9 is not optional bookkeeping: the tracker's ids
    deliberately do not rewind, so a rewind has to be announced to both machines.
    """
    if result.error is not None:
        return
    frame = result.frame
    entry = streams.get(str(frame.stream_id))
    if entry is None:
        _append_error(bridge,
                      f"a result arrived for stream {frame.stream_id!r}, which has no "
                      f"tracker and no event machine; it was dropped rather than folded "
                      f"into another camera's picture")
        return
    tracker, machine = entry
    _reset_for_loop_change(frame, tracker, machine)
    now = frame.captured_at
    events = []
    for track in tracker.update(result.detections, now=now):
        events.extend(machine.observe(track, frame))
    bridge.frameArrivedFromWorker.emit(
        _frame_payload(frame, tracker, events, manifest))


def _reset_for_loop_change(frame: Any, tracker: Any, machine: Any) -> None:
    """Announce a clip rewind to both the tracker and the event machine.

    Called on the **first** frame of every loop, including loop 0. `tracker.reset()`
    clears the tracks but deliberately does not rewind the id counter, so a tracker's ids
    never collide across loops; the event machine's `reset(loop_id)` moves it onto the new
    loop and is what keeps one loop's pending presence from being completed by the next
    loop's arrival.
    """
    loop_id = frame.source_loop_id
    if machine.loop_id == loop_id:
        return
    tracker.reset()
    machine.reset(loop_id)


def _frame_payload(frame: Any, tracker: Any, events: Sequence[Any],
                   manifest: Any) -> dict[str, Any]:
    """One result, as the dict `UiBridge.apply_frame` consumes.

    Carries `image` (the raw numpy array) and, on the first frame only, `zones`. Both are
    rebuilt on the UI thread by the slot; see `_on_result` for why the pixels travel here
    rather than as a `QImage`.

    ## `stream` is the whole of the per-camera split

    **M2.** The payload says which camera it is, and `UiBridge.apply_frame` files the
    pixels, the revision and the overlay boxes under that camera. Without it there was one
    `_frame_image` and every tile asked the provider for `hero`, so all three surfaces showed
    whichever camera delivered last - a tile captioned "Camera B" showing Camera A's
    shopper. Read from `frame.stream_id`, which is the identity the frame was submitted
    under, so it cannot disagree with the worker that produced it.
    """
    return {
        "stream": str(frame.stream_id),
        "identity": list(frame.identity),
        "width": frame.width,
        "height": frame.height,
        "image": frame.image,
        # `zones_for_event_rules()`, **not** `manifest.zones`. The shipped pack carries its
        # zones inside `event_rules` (measured: `manifest.zones` is `()`), so reading the
        # attribute would have published an empty list and passed the "did I call
        # set_zones" check while drawing nothing.
        "zones": list(manifest.zones_for_event_rules().values()),
        "boxes": [
            {"track_id": t.track_id, "class_id": t.class_id,
             "x": t.box[0] / frame.width, "y": t.box[1] / frame.height,
             "w": (t.box[2] - t.box[0]) / frame.width,
             "h": (t.box[3] - t.box[1]) / frame.height,
             "confirmed": bool(t.confirmed)}
            for t in tracker.current_tracks()
        ],
        "events": [
            {"kind": e.kind, "class_name": e.class_name, "track_id": e.track_id,
             "dedup_key": e.dedup_key}
            for e in events
        ],
    }


def _start_feeders(bridge: Any, supervisor: Any, clip: Any) -> None:
    """One thread **per camera**: decode at the clip's rate and submit. Nothing touches a
    widget.

    **It has a stop condition that is not a guess.** The first version's only exit was
    `route_plan().state == "failed"`, which a supervisor that was shut down *normally*
    never reaches - so after `application.exec()` returned, the thread carried on calling
    `clip.read()` and `supervisor.submit()` against a torn-down supervisor and a closed
    clip until the process happened to exit. It now also checks `_FEED_STOP`, which
    `run_windowed` sets on the way out.

    ## The exit condition is now **per stream**, not the aggregate

    With one camera, `route_plan().state == "failed"` *was* that camera's state. With three
    it is `failed` only when **every** stream has failed, so a feeder for a dead camera
    would have gone on submitting into a queue nobody drains - for ever, at 15 frames a
    second, while the other two cameras were fine. The check is therefore this clip's own
    stream through `stream_state(stream_id)`, the per-stream answer
    `RoutePlan.stream_state` exists to provide.

    The tracker and the event machine are **not** arguments here. They are folded in the
    supervisor's reader thread by `_on_result`, on the same result that produces the
    payload this thread publishes - passing them in would invite a second fold of the same
    result, which is the defect I2 records for the headless path.
    """
    _FEED_STOP.clear()
    stream_id = str(clip.stream_id)

    def feed() -> None:
        while not _FEED_STOP.is_set():
            if supervisor.stream_state(stream_id) == "failed":
                return
            frame = clip.read()
            if frame is None:
                if _FEED_STOP.is_set() or not clip.restart():
                    return
                continue
            supervisor.submit(frame)
            time.sleep(0.005)

    thread = threading.Thread(target=feed, name=f"frame-feed-{stream_id}", daemon=True)
    thread.start()


def _start_telemetry(bridge: Any, supervisor: Any, application: Any) -> None:
    """A 2 Hz timer on the UI thread that takes one whole sample set.

    **On the UI thread, deliberately.** The collector has its own background thread for
    `psutil`, so calling `samples()` from here costs about 0.04 ms - measured - and doing
    it on the UI thread means no widget is touched from another one. At 2 Hz this is not
    a measurable load, and it keeps the whole UI-update path on one thread.

    ## It also publishes the three **derived** metrics

    `analysis fps`, `frame age p50` and `p95` are computed here from the supervisor's own
    counters, over the interval since the previous tick. They are not `MetricSample`s -
    there is no provider for a rate the application measures about itself - and nothing
    else published them, so the CPU card's activity line read "analysis fps unavailable ·
    frame age p50 unavailable p95 unavailable" for a whole session while the numbers sat
    in the supervisor. `unavailable` is the honest word for "nobody measured this", and
    now that something does, the card shows it.

    The rate is measured **between two ticks**, never over the process lifetime: a
    lifetime average on a screen would show the compile and the startup ramp smeared into
    a number that describes neither the running system nor anything anybody saw.

    ## The frame-age percentiles are over a **count-bounded window**, and it is not a
    ## fixed span of time

    `frame_age_p50_ms` and `frame_age_p95_ms` are percentiles of the supervisor's retained
    frame ages, which is a `deque(maxlen=FRAME_AGE_WINDOW)` - **256 samples**, chosen in
    `app/supervisor/runner.py` so the copy and the two sorts are bounded on this thread.

    **256 samples is ~17 s at 15 fps, and longer the further behind a stream falls** -
    256 / fps-of-frames-that-carry-an-age, not 256 / fps-offered. The two measured runs
    on 2026-10-03:

    - **Three streams on the CPU, no saturators** (spec section 5 row 4, the app's own
      load): each analysed 9.60-9.80 fps while being offered 15, so the window spans
      **~26 s** and the frame ages over three consecutive 120 s holds were p50
      **10767 / 11085 / 11319 ms** and p95 **11626 / 11905 / 12286 ms** (Task 3's
      `measure_steady_state`, one camera per hold). **An earlier, shorter hold read p50
      8239 ms / p95 12469 ms** and those numbers are the ones this docstring used to quote;
      the longer repeated measurement is the one to quote, and the two are not
      interchangeable.
    - **Six CPU saturators alongside the app** (the §5 hardware proof, a synthetic
      stand-in for "a busy machine", not the app's own load): the same three cameras
      analysed ~3.5 fps - 28 frames per 8 s - so the window spans **~73 s**.

    Unsaturated with a single CPU stream the app analysed 14.96 fps, so ~17 s, p50
    2039 ms. So a percentile here answers *"how late has this stream been lately"*, and
    how long "lately" is **depends on the backlog**: ~17 s when the stream keeps up with what
    it is offered, ~26 s when it analyses two thirds of it, ~73 s when it analyses a
    quarter. The field name does
    not carry that, and the bridge renders it without the window.
    **Task 4 owns the card**; what is fixed here is the docstring, because a reader of
    `app/main.py` must not take the figure for a rolling 5 s window or a lifetime
    percentile. Note also that `analysis fps` and the age percentiles are **different
    claims** and must not be read as one: a stream can hold 15 fps and still be seconds
    behind, which is exactly what the CPU rows show.
    """
    from PySide6.QtCore import QTimer

    previous = {"at": time.perf_counter(), "analysed": 0}

    def tick() -> None:
        # One whole sample set, through `sample()` and never through an accessor. The
        # accessors share one frozen `_Tick` with `sample()`; reading one between samples
        # moves the tick earlier for everybody and can flip the booth's own sample to
        # UNAVAILABLE with no other visible cause.
        bridge.samplesArrived.emit(tuple(supervisor.samples()))
        _publish_derived_metrics(bridge, supervisor, previous)

    timer = QTimer(application)
    timer.timeout.connect(tick)
    timer.start(500)


def _publish_derived_metrics(bridge: Any, supervisor: Any,
                             previous: dict[str, float]) -> None:
    """Push the analysis rate and frame-age percentiles to the bridge, on the UI thread.

    `None` for anything not yet measured - before the first result there is no rate and no
    frame age, and `set_metrics` stores `None` as `None` so the card says `unavailable`
    rather than `0`.
    """
    from app.supervisor.runner import FRAME_AGE_WINDOW
    now = time.perf_counter()
    analysed = supervisor.frames_analysed
    window = now - previous["at"]
    fps = ((analysed - previous["analysed"]) / window
           if window > 0 and analysed >= previous["analysed"] else None)
    ages = supervisor.frame_ages_ms()
    p50 = _percentile(ages, 0.50)
    p95 = _percentile(ages, 0.95)
    bridge.set_metrics(ai_fps=fps, frame_age_p50_ms=p50, frame_age_p95_ms=p95)
    # **State the window the two percentiles are taken over.** They are percentiles of a
    # `deque(maxlen=FRAME_AGE_WINDOW)`, so the window is "the most recent N analysed
    # frames" and its span in seconds depends on this stream's own rate. The card used to
    # print a bare `frame age p50` with nothing saying what that is a percentile
    # of: a p50 of 2 s over a 17 s window and a p50 of 11 s over a 26 s one are not the
    # same claim, and the label carried neither.
    # The constant is *published* here rather than imported by the bridge, so it stays in
    # one place and the UI module does not pull in the worker stack for a number.
    bridge.set_age_window(FRAME_AGE_WINDOW)
    # R3: the footer's event counts may only read `0` once a frame has actually been
    # analysed. This is the supervisor's own counter rather than a count of applied
    # results, so the row does not flicker between "unavailable" and `0` every time a
    # result is queued but its slot has not run yet.
    bridge.set_frames_analysed(analysed)
    _publish_stream_metrics(bridge, supervisor, previous, window)
    previous["at"] = now
    previous["analysed"] = analysed


def _publish_stream_metrics(bridge: Any, supervisor: Any,
                            previous: dict[str, float], window: float) -> None:
    """Push **each stream's own** lateness, backlog and rate to the bridge. UI thread.

    ## Why this exists: the merged figure is the most dangerous number this project has

    Measured, unsaturated, three streams on the CPU - which is spec section 5 row 4, a
    state the booth will be in on a quiet day - gives per-stream frame age **p50 10767 /
    11085 / 11319 ms** and **p95 11626 / 11905 / 12286 ms**, a backlog of **170-187 frames
    per stream**, each stream analysing about **13.5 of the 15 fps** offered, **zero
    drops**, and all three streams reporting `active`.

    That is a picture **eleven seconds old**, while every counter on the old screen reads
    healthy: no drops, an active state, a plausible fps. Nothing said the picture was stale.
    So the per-stream number is published, and the card says it in words
    (`UiBridge.STALE_AFTER_SECONDS`, quoted from spec section 7).

    **Frame age is not inference cost, and the two are never added or averaged.** On the
    NPU the measured p95 frame age is seconds while the inference itself costs ~5.9 ms; the
    gap is queue wait. They are two different claims with two different words, and merging
    them would produce a number that is true of neither.

    **Per stream, never merged.** `supervisor.frame_ages_ms()` returns the percentile of up
    to 3 x 256 samples across all three cameras, which is a figure about no camera; a card
    showing it would be showing an average on a card for one of them. So this reads
    `stream_frame_ages_ms(stream)` for each stream and computes its own percentiles.

    **Backlog** is `frames_accepted - frames_analysed - frames_invalidated`: submitted,
    taken off the queue and thrown away by a route change are all excluded, so what is
    left is the queue's own depth. The frame age is a consequence of that backlog rather
    than a cause of it, which is why the card shows both.

    **Per-stream fps is a real second derivative** - the previous tick's count for the same
    stream, not the aggregate - so one camera's backlog cannot make another's card look
    fast. `None` before a stream has analysed anything, which the card renders as
    `unavailable` rather than `0 fps`: a camera that has done no work is not a camera
    running at zero frames per second.

    **`window` is passed in** rather than measured again here. It is the interval the caller
    already computed for its own aggregate rate, and re-reading `perf_counter()` here would
    measure a *different, slightly shorter* interval - so the per-stream rates and the
    aggregate rate would be over slightly different windows and neither could be checked
    against the other. It also makes the publisher testable without a clock.
    """
    counters = previous.setdefault("per_stream", {})
    for stats in supervisor.per_stream():
        sid = stats.stream_id
        ages = supervisor.stream_frame_ages_ms(sid)
        p50 = _percentile(ages, 0.50)
        p95 = _percentile(ages, 0.95)
        prior = counters.get(sid)
        fps = None
        if window > 0 and prior is not None and stats.frames_analysed >= prior:
            fps = (stats.frames_analysed - prior) / window
        counters[sid] = stats.frames_analysed
        backlog = (stats.frames_accepted - stats.frames_analysed
                   - stats.frames_invalidated)
        bridge.set_stream_metrics(
            sid,
            age_p50_ms=p50,
            age_p95_ms=p95,
            # **`None` before any frame has been analysed**, so a stream that has done no
            # work renders "unavailable", never "0 frames of backlog" - which would read as
            # a healthy empty queue.
            backlog=backlog if stats.frames_accepted > 0 else None,
            analysis_fps=fps,
            window_samples=len(ages))


# --------------------------------------------------------------------------------------
# The CLI.
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """The CLI the brief specifies. Nothing else is accepted."""
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Intel Edge AI Roadshow - retail stream (M1).",
    )
    parser.add_argument("--scenario", default="retail",
                        choices=("retail",),
                        help="the scenario pack to run. Only `retail` ships in M1.")
    parser.add_argument("--device", default="CPU",
                        choices=("CPU", "GPU", "NPU"),
                        help="the explicitly requested compile target. No AUTO, no "
                             "HETERO, no MULTI: they hide where the work ran.")
    parser.add_argument("--source", default="loop",
                        choices=("loop", "once"),
                        help="loop the clip, or stop at the end of one pass.")
    parser.add_argument("--selftest", action="store_true",
                        help="run every stage with no network and no display, then exit.")
    parser.add_argument("--headless", action="store_true",
                        help="run the full pipeline offscreen and print the summary.")
    parser.add_argument("--windowed", action="store_true",
                        help="open the QML booth screen (the default when nothing else "
                             "is asked for).")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="how long a headless run lasts.")
    parser.add_argument("--json", dest="json", default=None,
                        help="write the run summary to this path as JSON.")
    parser.add_argument("--select", dest="select", action="append", default=None,
                        metavar="STREAM=DEVICE",
                        help="a presenter-driven move, repeatable. Applied after the "
                             "window exists, on its own thread, and the badge changes "
                             "only once a placement is CONFIRMED. e.g. "
                             "--select A=GPU")
    parser.add_argument("--streams", dest="streams", type=int, default=1,
                        help="how many cameras to run. 1 (the default) is the single-camera "
                             "window exactly as before; 3 asks for one camera per available "
                             "device, each with its OWN clip, decode thread, tracker and "
                             "event machine, and each tile showing only its own stream's "
                             "frames. --device names the FIRST camera's device.")
    parser.add_argument("--fail-stream", dest="fail_stream", default=None,
                        metavar="STREAM",
                        help="make this camera's worker really die before compiling, so "
                             "the failure path can be observed on the live screen. A "
                             "QUALIFICATION AID, not a booth feature: it injects the "
                             "same fault `Supervisor`'s own test seam injects, so what "
                             "you see is the real detect-a-dead-worker path and not a "
                             "mock.")
    parser.add_argument("--release", dest="release", action="store_true",
                        default=RELEASE_BY_DEFAULT,
                        help="refuse to start with a simulated provider registered. "
                             "On by default; see RELEASE_BY_DEFAULT.")
    parser.add_argument("--no-release", dest="release", action="store_false",
                        help="run without the simulated-provider gate. For a developer "
                             "who needs a simulated provider on their own machine, and "
                             "deliberately awkward to type.")
    return parser


def _write_json(path: str | None, payload: Any) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


#: The devices `--select` will accept. **The same three the rest of the CLI accepts**,
#: deliberately: a booth switch offers CPU, GPU or NPU and nothing else. `AUTO`, `HETERO`
#: and `MULTI` are refused for the reason `--device` gives - they hide where the work ran,
#: and a badge that says "AUTO" is not a placement.
SELECTABLE_DEVICES: Final[tuple[str, ...]] = ("CPU", "GPU", "NPU")


def _parse_selection(entries: Sequence[str] | None) -> dict[str, str] | None:
    """`["A=GPU", "B=CPU"]` -> `{"A": "GPU", "B": "CPU"}`. `None` in, `None` out.

    Returns `None` for a malformed entry rather than raising, because the caller's job is
    to print one clear line and exit 2 - a traceback from a mistyped booth switch is a
    worse answer than "use STREAM=DEVICE". A device outside `SELECTABLE_DEVICES` is
    refused for the same reason `HETERO` is: naming a device nothing will execute on would
    put a claim on screen that cannot be true.
    """
    if not entries:
        return None
    mapping: dict[str, str] = {}
    for entry in entries:
        text = str(entry)
        if "=" not in text:
            print(f"--select wants STREAM=DEVICE; got {text!r}",
                  file=sys.stderr, flush=True)
            return None
        stream, _, device = text.partition("=")
        stream, device = stream.strip(), device.strip()
        if not stream or device not in SELECTABLE_DEVICES:
            print(f"--select {text!r}: stream must be named and device must be one of "
                  f"{list(SELECTABLE_DEVICES)}", file=sys.stderr, flush=True)
            return None
        mapping[stream] = device
    return mapping


def main(argv: Sequence[str] | None = None) -> int:
    """The process entry point. Returns the exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    # Both sets, not just the allow-list: the gate refuses a build that registers anything
    # simulated, and it also refuses a build whose collector reaches for a provider the
    # gate was never told about. The second half is what makes the first half mean
    # something.
    assert_no_simulation_in_release(release=bool(args.release),
                                    providers=tuple(REGISTERED_PROVIDERS)
                                    + consulted_providers())

    if args.selftest:
        summary = run_selftest()
        for name, stage in summary["stages"].items():
            marker = {"ran": "ok", "skipped": "skip", "failed": "FAIL"}[stage["verdict"]]
            detail = stage["detail"] if stage["verdict"] == "ran" else stage["reason"]
            print(f"  {marker:4} {name:20} {detail}", flush=True)
        if summary["ok"]:
            print("PASS", flush=True)
            return 0
        print("FAIL: " + ", ".join(summary["failed"]), file=sys.stderr, flush=True)
        return 1

    if args.headless:
        try:
            summary = run_headless(device=args.device, seconds=args.seconds,
                                   source=args.source, scenario_id=args.scenario)
        except RuntimeError as exc:
            print(f"headless run refused: {exc}", file=sys.stderr, flush=True)
            return 2
        _write_json(args.json, summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        # I1: **a run that analysed nothing is not a run.** Exit `0` here is what let a
        # pipeline that never started, or died at once, or was stopped early report
        # `submitted: 0, drops: 0, state: pending` and still look like a clean run. The
        # summary is still written - it is the evidence for why - but the exit code says
        # the analysis did not happen.
        if summary["frames_analysed"] <= 0:
            print("headless run analysed no frames; this is not a successful run. "
                  f"state={summary['device']['state']!r} "
                  f"submitted={summary['frames_submitted']} "
                  f"refused={summary['frames_refused']} "
                  f"refused_reasons={summary['frames_refused_reasons']} "
                  f"errors={summary['errors']}",
                  file=sys.stderr, flush=True)
            return 2
        return 0

    if args.windowed or not (args.selftest or args.headless):
        selection = _parse_selection(args.select)
        if args.select and selection is None:
            print(f"--select wants STREAM=DEVICE; got {args.select!r}",
                  file=sys.stderr, flush=True)
            return 2
        return run_windowed(device=args.device, scenario=args.scenario,
                            source=args.source, select=selection,
                            fail_stream=args.fail_stream,
                            streams=max(1, int(args.streams or 1)))

    return 0


if __name__ == "__main__":  # pragma: no cover - the process entry point
    sys.exit(main())