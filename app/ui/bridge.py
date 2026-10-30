"""`UiBridge` - the one object QML talks to, and the only place numbers are formatted.

Every rule this module exists to enforce is about the difference between *measured*,
*absent* and *not yet known*. QML cannot enforce any of them: it has strings and it has
`undefined`. So the formatting happens here, in Python, next to the contract that
refuses the shapes, and QML renders what it is given.

## The five obligations, and where each is structural

1. **The overlay draws only current tracks.** `set_overlay` filters on `Track.is_current`
   and there is no code path that skips the filter. `Tracker.update()` returns coasted
   tracks *on purpose* - the event machine needs to know a track still exists - so a
   consumer that draws from that sequence without filtering draws a plausible,
   correctly-placed rectangle from a recent frame over an object that has since left.
   Nothing about that rectangle looks wrong, which is what makes it dangerous. The
   filter is here rather than in the QML because a QML `Repeater` cannot be trusted to
   hold a truth table about a track's history.

2. **Metrics arrive from `sample()`, never from the collector's accessors.** The
   accessors share one frozen `_Tick` with `sample()`, so calling one between samples
   moves the tick earlier for everybody and can flip the booth's own sample to
   `UNAVAILABLE`. `UiBridge.set_samples` takes a whole `MetricSample` tuple and
   `display_text` reads that. There is no accessor call anywhere in this module, and
   `tests/test_ui_contract.py` reads this file's source to keep it that way.

3. **The window is rendered only after `validity` is checked.** `window_text` branches on
   `validity` *first* and returns `"unavailable"` for `UNAVAILABLE` **before it looks at
   `window_seconds`**. That ordering is the whole point: `window_seconds == 0.0` is legal
   exactly when `validity is UNAVAILABLE`, and a formatter that formatted the window first
   and branched afterwards would be one line away from dividing by zero. There is no
   division anywhere in this module.

4. **`unavailable` is never `0`.** Every optional metric goes through `display_text`,
   which returns the literal string `"unavailable"` for `None` and a formatted number
   otherwise - including for a genuine `0.0`, which renders as `"0.0%"`. The refusal must
   not be indiscriminate: an idle-but-measurable process reads `0.0` with
   `validity=VALID`, and calling that unavailable is the same error pointing the other
   way.

5. **The displayed device is the CONFIRMED one.** `device_confirmed` comes from the
   confirmed `EXECUTION_DEVICES` tuple, normalised through
   `app.inference.devices.normalise_execution_devices`, and `set_route` refuses to report
   the requested device as confirmed. `device_fell_back` compares the two through
   `app.inference.devices.is_device` - `'NPU'` on this machine is the bare string, GPU
   reports `'GPU.0'`, and `'GPU' in 'GPU.0'` is a `True` substring test that would also
   pass on `'GP'`.

## GPU and NPU utilisation are measured, per process, or absent with a reason

`gpu_available` and `npu_available` are **derived from the last sample**, exactly as
`cpu_available` is: `True` only when this application's own per-process figure is in hand.
`app.telemetry.accelerator` produces it from `\\GPU Engine(*)\\Utilization Percentage`,
filtering each instance's own `pid_` to this application's processes and its adapter LUID
to the identified device. Measured on this machine on 2026-10-27: a GPU inference worker's
`engtype_Compute` instance read **97.19%** while every other process on the machine read
below 3% (the two busiest foreign rows measured 1.42% and 2.79%), and an NPU worker's
`engtype_Neural` instance read **64.86%**. That is the control
that makes the number attributable; it is in
`docs/qualification/M4-utilisation-attribution.md`.

The M0 record said the native PDH value read faulted with `0xC0000005`. That was real, and
it was a **three-argument binding of a four-argument function** in this repository, not a
PDH defect: with the documented signature the same call returns real values (see
`app.telemetry.pdh`). A machine-wide peak is still not attributable to this process - the
62%/68% figure was withdrawn for exactly that reason and is not reinstated - so the figure
here is per-process or nothing.

When there is no sample yet, or this application has no engine instance on that device, the
cards say **"Utilization unavailable"** with the sample's own reason. `ComputeCard` is never
given a number that no sample carried.

## The mandatory label

`RECORDED_FOOTAGE_LABEL` is `"Recorded footage • live inference"`, defined here in Python
and set on the QML label that is **always visible**. Its absence was a real defect in the
previous build, so `tests/test_ui_contract.py` pins it in the QML sources as well as
here - a QML file that dropped the binding would fail the test even though this constant
still existed.

**This module imports no OpenVINO, no `cv2` and no `psutil`.** It formats what the
engine and the collector hand it. That keeps the UI importable on a machine with no model
provisioned, which is what makes `--selftest` possible without a device.
"""

from __future__ import annotations

import math
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from PySide6.QtCore import Property, QObject, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter

import numpy as np

from app.contracts.benchmark_record import BenchmarkRecord
from app.contracts.metric_sample import MetricSample, Validity
from app.inference.devices import is_device, normalise_execution_devices


class _Extents:
    """The two numbers `set_zones` reads off a frame, and nothing else.

    `set_zones` takes a frame to prove a zone is being drawn over *a picture* - a zone
    with no frame behind it is a rectangle on nothing. When the zones arrive in an
    `apply_frame` payload they arrive alongside the frame's extents rather than the
    packet, so this carries them. Deliberately minimal: anything a zone could read off a
    frame beyond its size belongs in the packet, not here.
    """

    __slots__ = ("width", "height")

    def __init__(self, width: Any, height: Any) -> None:
        self.width = int(width or 0)
        self.height = int(height or 0)

__all__ = [
    "ATTRACT_HEADLINES",
    "ATTRACT_IDLE_SECONDS",
    "ATTRACT_TITLE",
    "CPU_UNAVAILABLE_TEXT",
    "DEFAULT_ROUTE_STREAM",
    "EXPLANATIONS",
    "FrameImageProvider",
    "RECORDED_CONTROL_NOTE",
    "RECORDED_DEMO_LABEL",
    "RECORDED_FOOTAGE_LABEL",
    "RECORDED_METRIC_LIMIT",
    "SCOPE_NOTE_LABEL",
    "SLOT_CALLS_SAFE_IN_A_QML_BINDING",
    "STALE_AFTER_SECONDS",
    "ThemeTokens",
    "UNAVAILABLE_TEXT",
    "UTILISATION_MEASURED_REASON",
    "UTILISATION_SHORT_REASON",
    "UTILISATION_UNAVAILABLE_TEXT",
    "UiBridge",
]

#: The stream the booth screen is bound to. **One camera, one card** - Task 5 adds the
#: other two. It is a named constant rather than a literal in three places so the bound
#: stream is one decision, and so a test can import it instead of repeating `"A"`.
DEFAULT_ROUTE_STREAM = "A"

#: Spec §4: displayed continuously. **Not optional and not per-screen.**
RECORDED_FOOTAGE_LABEL = "Recorded footage • live inference"

#: What an absent measurement renders as. A word, never a number (AGENTS.md rule 15).
UNAVAILABLE_TEXT = "unavailable"

#: What the GPU and NPU cards say instead of a utilisation figure. See the module note.
UTILISATION_UNAVAILABLE_TEXT = "Utilization unavailable"

#: The short, card-sized reason for a card with no accelerator sample yet.
#:
#: **Both halves matter.** "This application's own engine time" says what the figure
#: measures when it exists, and "unavailable, never 0" says what is shown when it does
#: not. The earlier version of this string asserted that the platform's counters fault
#: this process; that was a binding bug in this repository, corrected on 2026-10-27, and
#: repeating it here would be a claim the code no longer supports.
UTILISATION_SHORT_REASON = (
    "No accelerator sample has landed yet. GPU and NPU utilisation is this application's "
    "own engine time from the Windows GPU Engine counter, filtered to its processes; a "
    "device it has not used reads unavailable, never 0."
)

#: The long form, for the explanation panel: what the figure measures, how the device is
#: identified, and the control that makes it attributable.
UTILISATION_MEASURED_REASON = (
    "GPU and NPU utilisation is read from the Windows 'GPU Engine' counter set "
    "(\\GPU Engine(*)\\Utilization Percentage) through pdh.dll in this process, and "
    "filtered to this application's own pids - the UI process and its workers. Each "
    "instance name carries pid_<n>, the adapter LUID and the engine type, so a reading "
    "belongs to a process and an adapter by construction. 100% means the busiest engine of "
    "that adapter was busy for this application for the whole interval between two counter "
    "samples; engines are never summed, and processes are never summed into a machine-wide "
    "figure. The GPU is identified by DXGI as the one hardware adapter (on the demo "
    "machine Intel Arc 140V, PCI 8086:64A0); the NPU is the one adapter exposing an "
    "engtype_Neural engine. Measured on 2026-10-27 with an idle-then-busy control: a GPU "
    "worker's busiest engine read a median 97.19% (97.06-100.0%) while idle it read 0.0%, "
    "and an NPU worker's Neural engine read a median 64.86% (63.76-66.08%) while idle it "
    "read 0.0%. M0 withdrew a 62%/68% figure because it was a "
    "machine-wide peak - our pid appeared in none of the 469 GPU Engine instances M0 "
    "measured - and it is not reinstated. "
    "The control that proves this figure tracks this application is in "
    "docs/qualification/M4-utilisation-attribution.md."
)

#: What the **CPU** card says when a system CPU reading has not landed, and why it is not
#: the accelerator wording.
#:
#: "Utilization unavailable" is the *accelerator* story: the GPU and NPU cards show it
#: until a `pdh_gpu_engine` sample lands, and nothing is estimated in place of a
#: measurement. On the CPU card the same words would assert a failure to read a counter
#: that **does** work here - `psutil.cpu_percent` is called every 250 ms and returns a
#: figure. The true and much weaker fact is that no sample has landed yet, or that the last
#: delta window was too short to measure. Those are different claims and a presenter who is
#: asked "why does the CPU card say unavailable?" deserves the right one.
CPU_UNAVAILABLE_TEXT = "no CPU sample yet"

#: Calls a QML **binding** may make, because their result depends only on module-level
#: constants and therefore cannot go stale between evaluations. Qualified by owner, so a
#: QML helper and a Python Slot cannot be confused for one another.
#:
#: A QML binding that calls a method declares no property dependency, so Qt evaluates it
#: once - at `engine.load` - and never again. That is the defect class this milestone found
#: three times: the `constant=True` device badge, the metric cards over
#: `display_text(...)`, and the card's `windowNote` over `window_text(...)`. In each case
#: the bridge was right and the screen was frozen at its first evaluation.
#:
#: **This set is the standing guard.** A QML binding that calls anything else will not
#: re-evaluate, so `tests/test_ui_contract.py::SlotBindingAudit` fails on any such call
#: that is not named here. Adding a name is a deliberate act with a stated justification,
#: not something a binding acquires by existing.
#:
#: What qualifies:
#:
#: - `bridge.explanation` - reads `EXPLANATIONS`, a module constant, and its argument
#:   (`explainKey`) is fixed per card.
#: - `bridgeUi.explain` - a thin QML wrapper over the above; same constant, and its
#:   argument is a clicked key, so a click re-evaluates the binding anyway.
#:
#: What deliberately does **not** qualify:
#:
#: - `bridge.display_text`, `bridge.window_text`, `bridge.staleness_text` - all read live
#:   state. Bind to `metric_text`, `window_notes` and `staleness` instead; those are
#:   notified properties holding the same formatted values.
#: - `bridgeUi.confirms` - reads `device_confirmed`, which changes when the first result
#:   lands. Nothing binds to it; `ComputeCard` has its own `confirmedHere` property bound
#:   to `bridge.device_confirmed` directly, which does notify. Listed here because a
#:   binding to it would freeze, and saying so is cheaper than discovering it.
SLOT_CALLS_SAFE_IN_A_QML_BINDING: frozenset[str] = frozenset({
    "bridge.explanation",
    "bridgeUi.explain",
})

def _shipped_scope_note() -> str:
    """The retail pack's own `scope_note`, read from the pack.

    **Taken from the pack rather than restated here.** An earlier revision carried a
    hand-written summary in this module - "No product-transfer event is detectable on
    this footage; this is people-in-zone presence" - and it was *not* the pack's wording:
    the shipped `scope_note` says "on **retail_checkout**: the clip contains no shelf",
    names the counter-surface measurement, and ends by pointing at `retail_config.py`.
    Two sentences about one reduction, and a presenter reading the screen would be
    quoting wording that exists nowhere in the data. Reading the pack makes that
    impossible: there is one string and the screen shows it.
    """
    from app.engine.retail_config import retail_sanity_manifest

    return retail_sanity_manifest().scope_note


#: The reduction, in the shipped pack's own words. See `_shipped_scope_note`.
SCOPE_NOTE_LABEL = _shipped_scope_note()


# --------------------------------------------------------------------------------------
# Attract mode (M3 Task 4)
# --------------------------------------------------------------------------------------
#
# **Attract mode shows no number it cannot date.** The idle screen is a booth surface that
# nobody is driving, and the failure this section exists to prevent is a live figure that
# has quietly gone stale being read as current. So attract mode shows **recorded
# measurements only**, each carrying the date it was recorded and the committed file it
# came from, and the live screen is hidden while it is up - there is no live number behind
# the overlay to be misread, and there is no path here that can render a figure without a
# date.

#: The attract slide's title. Not a measurement and not a claim.
ATTRACT_TITLE = "Intel Edge AI Roadshow"

#: Spec section 9's wording for a recorded surface, quoted rather than paraphrased.
RECORDED_DEMO_LABEL = "Recorded demonstration \u2014 metrics are not live"

#: **M5 Task 4.** What a presenter is told when the live device controls are present but
#: dead. The wording is a statement about the screen, not about the machine: it says the
#: controls are disabled *while a recording plays*, which is the only time it is shown.
RECORDED_CONTROL_NOTE = (
    "Live device controls are disabled while a recording plays. "
    "Leave the recording to move a camera."
)

#: How many recorded metrics a playback puts on screen. The record can carry arbitrarily
#: many samples; the screen is 1080 logical pixels tall and every row is 44 px so it reads
#: from the aisle. Truncation is **stated** (`recorded_metrics_note`), never silent.
RECORDED_METRIC_LIMIT = 6

#: Spec section 9: attract starts after 90 s of no presenter input. A time, like
#: `STALE_AFTER_SECONDS`, so it lives in this module rather than in `app/theme.py`, which
#: holds layout tokens and no metrics.
ATTRACT_IDLE_SECONDS: float = 90.0

# **Rotation is not implemented, and there is no constant pretending otherwise.** The
# design's section 9 says the attract slides rotate every ~45 s. This implementation shows
# **every recorded claim at once**, on one screen, each with its own date and source - which
# is strictly stronger for the obligation this mode exists to discharge: a claim cannot be
# read without the line that dates it, because there is no transition in which the two could
# be separated, and no claim is off screen for 90 of every 135 s. Rotation is deferred until
# there are more claims than fit on one screen. Stated here rather than in a commit message
# so the next reader does not look for a timer that is not there.

#: **The recorded headline claims, each with the date it was recorded and the committed
#: file it is sourced from.** Every figure here is a measured result from a committed
#: artefact; the dates are the measurement dates recorded in
#: `docs/qualification/M2-qualification.md` (2026-10-03) and `M1-qualification.md`
#: (2026-10-02).
#:
#: **The numbers are pinned to the measurements by `tests/test_ui_contract.py`,** which
#: re-reads `docs/qualification/m2-headless/single-frame-latency-run1..4.json` and
#: `allcpu-steady-120s*.json` and fails if a range here has drifted from the files it
#: cites. A claim that is dated but whose number no longer matches its source is a
#: confident number with nothing behind it, which is the class this project keeps finding.
ATTRACT_HEADLINES: tuple[dict[str, str], ...] = (
    {
        "text": "Single-frame inference, 640x640, batch 1, 30 inferences: "
                "CPU 22.40-22.81 ms, GPU 3.40-3.54 ms, NPU 5.21-5.74 ms. "
                "Not a throughput claim.",
        "date": "2026-10-03",
        "source": "docs/qualification/M2-qualification.md",
    },
    {
        "text": "Three cameras on the CPU: frame age p50 11.3-14.7 s, p95 12.3-16.0 s, "
                "backlog 173-242 frames per stream, 0 drops, all three active.",
        "date": "2026-10-03",
        # **All three holds, named.** The figures are a range across three 120 s runs; a
        # source naming only `allcpu-steady-120s.json` would not contain the upper end of
        # its own range, so a visitor who opened the citation would find a number that
        # disagreed with the screen.
        "source": "docs/qualification/m2-headless/allcpu-steady-120s*.json (3 holds)",
    },
    {
        "text": "Delivered event: people-in-zone presence, 2 events per clip. "
                "NOT a people count. NOT a dwell measurement.",
        "date": "2026-10-02",
        "source": "docs/qualification/M1-qualification.md",
    },
)

#: The explanations the design's section 9 requires to be reachable from hover and from a
#: pinned panel. Values are the text; the QML shows them, this module owns them.
EXPLANATIONS: dict[str, str] = {
    "yolo": (
        "Detection: YOLO11n FP16 (yolo11n-fp16), a single convolutional detector run "
        "through OpenVINO on one explicitly chosen device. One 1920x1080 frame is "
        "letterboxed to the pinned 640x640 input with grey padding, and the output is "
        "decoded from cx,cy,w,h anchors with the model's own threshold. The export "
        "carries no NMS, so suppression runs here."
    ),
    "tracking": (
        "Tracking: greedy IoU matching within each class, minimum two hits before a track "
        "counts, two missed frames before it is dropped, and ids never reused. Ids are "
        "unique for the life of one tracker, so an event key built from a track id is a "
        "stable key. Only tracks matched by the current frame are drawn; a coasted "
        "track's box is the last one seen and would be drawn over an object that is no "
        "longer there."
    ),
    "zone": (
        "Zone rules: a normalised region of the frame. A person counts as present when "
        "the centre of their box is inside the band, because a person at a counter is "
        "routinely clipped by the frame edge and full containment would report an empty "
        "shop. Presence is sustained time in the zone, measured between two real "
        "observations. This is people-in-zone presence: no product-transfer event is "
        "detectable on this footage."
    ),
    "cpu": (
        "CPU: the always-available plugin. Its utilisation is measured by the operating "
        "system across all logical cores and normalised to the machine, so the two CPU "
        "figures on screen share one 0-100 scale."
    ),
    "gpu": (
        "GPU and NPU: Intel's integrated Arc GPU and the AI Boost NPU. Utilisation is "
        "read from the Windows GPU Engine counter, per process: each instance carries the "
        "owning pid, the adapter LUID and the engine type, so the figure on the card is "
        "this application's own engine time on the identified device - never a "
        "machine-wide peak. It is the busiest engine's busy fraction, so 100% means one "
        "engine was busy for the whole sampling interval; it is not a sum, and a device "
        "this application has not used reads unavailable, never 0. Also shown is "
        "inference activity: how many frames were analysed, how long each inference "
        "took, and which device the runtime confirmed the work ran on. Per-device "
        "single-frame latency on this machine measures CPU 20.99-21.16 ms, GPU "
        "3.30-3.47 ms, NPU 5.92-5.99 ms - single frame, 640x640, batch 1, 30 "
        "inferences, integrated Arc 140V sharing memory with the NPU. That is not a "
        "throughput claim."
    ),
    "utilisation": (
        "GPU and NPU utilisation is read from the Windows 'GPU Engine' counter set, per "
        "process and per identified adapter. The M0 record said the platform's counters "
        "faulted this process; that was a three-argument binding of a four-argument "
        "function in this repository, corrected on 2026-10-27, and the same call now "
        "returns real values. What remains true is that a machine-wide peak is not "
        "attributable to this process, so the provider filters each instance to this "
        "application's own pids and never reports a peak. When this application has no "
        "engine instance on a device, the card reads 'Utilization unavailable' with that "
        "reason. Nothing here is estimated and nothing is filled in."
    ),
    "metrics": (
        "Metrics: each figure names its provider, its scope and the window it was "
        "averaged over. System CPU is the whole machine. Application CPU is this "
        "process plus its worker, normalised to the machine. Frame age is how long a "
        "frame waited from capture to the moment its result arrived back at the "
        "supervisor - not to the moment it was painted, which this build does not "
        "measure. Anything not "
        "measurable reads 'unavailable', never 0, because 0 would claim a measurement "
        "that was not taken."
    ),
}

#: Metric names this bridge formats. `display_text` refuses anything outside this set, so
#: a typo in QML is a stated error rather than a card that silently reads "unavailable".
OPTIONAL_METRICS: tuple[str, ...] = (
    "ai_fps",
    "frame_age_p50_ms",
    "frame_age_p95_ms",
    "system_cpu_percent",
    "app_cpu_percent",
    "gpu_utilisation_percent",
    "npu_utilisation_percent",
)

#: The unit each optional metric is rendered in. `None` unit means the formatter appends
#: nothing and the metric is rendered as a bare number.
METRIC_UNITS: Mapping[str, str] = {
    "ai_fps": "fps",
    "frame_age_p50_ms": "ms",
    "frame_age_p95_ms": "ms",
    "system_cpu_percent": "%",
    "app_cpu_percent": "%",
    "gpu_utilisation_percent": "%",
    "npu_utilisation_percent": "%",
}


def _percent(value: float) -> str:
    return f"{value:.1f}%"


def _ms(value: float) -> str:
    return f"{value:.1f} ms"


def _fps(value: float) -> str:
    return f"{value:.2f} fps"


def _summary_line(*, fps: float | None, age_p50: float | None,
                  age_p95: float | None) -> str:
    """One camera's activity line: its rate and both of its ages.

    **Composed here rather than in QML** so the card cannot concatenate three different
    cards' figures into one line, and so the unit words (`fps`, `s`, `ms`) are decided in
    the same module as the formatters.

    **The window is deliberately NOT in here.** It has its own line on the card, labelled
    `window:`, because a percentile of a bounded deque is not comparable with a percentile
    of a different one - and the first render of this layout overflowed the card by one
    line with both on one string.

    **Frame age only. Never inference cost.** On the NPU the measured p95 frame age is
    seconds and the inference itself is about 5.9 ms; the difference is queue wait. Two
    claims, two words, and putting them in one number would produce a figure that is true
    of neither.
    """
    rate = UNAVAILABLE_TEXT if fps is None else _fps(fps)
    p50 = UNAVAILABLE_TEXT if age_p50 is None else _age(age_p50)
    p95 = UNAVAILABLE_TEXT if age_p95 is None else _age(age_p95)
    # "analysis fps" is kept verbatim from M1's line - "analysis fps 13.40 fps" is clumsy,
    # but a test in this repository asserts that prefix and renaming it here would change
    # the meaning of an assertion written against a different metric.
    return f"analysis fps {rate}  \u00b7  frame age p50 {p50}  p95 {p95}"


#: Spec section 7, verbatim: *"Values older than 2 s are `stale`."* The one threshold in
#: this project that comes from a document rather than from a measurement, so it is
#: quoted rather than approximated - and it is a **time**, unlike the frame-age window
#: below, which is a count.
STALE_AFTER_SECONDS: float = 2.0


def _age(value_ms: float) -> str:
    """A frame age, in **seconds** once it reaches a second, milliseconds below.

    A rendering decision, and the reason is legibility at a glance: the measured
    unsaturated three-stream condition is a p50 of **10.8-11.3 s**, and `11319.0 ms` is
    not something a presenter reads as "eleven seconds old" while a booth visitor is
    talking to them. `315.5 ms` is still the right precision below a second, so the
    switch happens at the unit boundary rather than at some chosen cutoff.
    """
    if value_ms >= 1000.0:
        return f"{value_ms / 1000.0:.1f} s"
    return _ms(value_ms)


#: name -> formatter. Chosen here rather than in QML so the rounding is one decision.
_FORMATTERS = {
    "ai_fps": _fps,
    "frame_age_p50_ms": _ms,
    "frame_age_p95_ms": _ms,
    "system_cpu_percent": _percent,
    "app_cpu_percent": _percent,
    "gpu_utilisation_percent": _percent,
    "npu_utilisation_percent": _percent,
}


def _token(name: str) -> Any:
    """One layout token out of `app.theme`, read by name.

    `ThemeTokens` is generated from this rather than written out by hand, so a token
    added to `app/theme.py` is reachable from QML without a second edit here, and a token
    renamed there cannot leave a stale alias behind in this file. The trade is that a typo
    in a QML binding reads `undefined` rather than a compile error; `Bridge.qml` guards
    the tokens it uses and `tests/test_ui_contract.py` asserts every token the QML
    references exists in `app/theme.py`.
    """
    from app import theme as theme_module

    return getattr(theme_module, name)


def _build_theme_class() -> type:
    """Build `ThemeTokens` with one constant property per token, by `type()`.

    **It has to be `type()`, and not `setattr` after the class body.** Measured here:
    attaching `Property` objects to an already-constructed `QObject` subclass produces a
    class whose attributes are `Property` *objects* rather than the descriptors Qt's
    meta-object system needs - `theme.CANVAS_WIDTH` then returned
    `<PySide6.QtCore.Property object at 0x...>` and every QML binding read `undefined`,
    with Qt reporting `Unable to assign [undefined] to int` and the window rendering at
    1x1. A hand-written list of properties avoids that trap and creates a worse one:
    twenty places to forget to update, where forgetting renders as `undefined` on a
    booth. `type()` with a generated namespace keeps `app/theme.py` the single source and
    cannot go stale.
    """
    namespace: dict[str, Any] = {
        "__doc__": "Layout tokens from `app.theme`, one constant property each.",
        "__init__": lambda self, parent=None: QObject.__init__(self, parent),
    }
    for name in _THEME_TOKEN_NAMES:
        namespace[name] = Property(_qml_type(name), _token_reader(name), constant=True)
    return type("ThemeTokens", (QObject,), namespace)


def _token_reader(name: str):
    """A getter closing over one token name. `getattr` at call time, not at build time."""

    def reader(self: Any) -> Any:
        return _token(name)

    reader.__name__ = f"read_{name.lower()}"
    return reader


def _qml_type(name: str) -> str:
    """The QML type a token of this name should be exposed as.

    `GRID` and `SPACING` are dicts and become `QVariantMap`; everything else is a `str`
    or an `int` in `app/theme.py`. **Read from the value, not from the name**, because a
    type derived from a token's *name* is a second convention to keep in step.
    """
    value = _token(name)
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if isinstance(value, Mapping):
        return "QVariantMap"
    return "QString"


def _theme_token_names() -> tuple[str, ...]:
    """Every public, upper-case token in `app.theme`.

    Filtered rather than listed, so a token added there is exposed here without a second
    edit. Lower-case names and anything starting with an underscore are skipped.
    """
    from app import theme as theme_module

    return tuple(sorted(
        name for name in dir(theme_module)
        if name.isupper() and not name.startswith("_")
    ))


_THEME_TOKEN_NAMES: tuple[str, ...] = _theme_token_names()

#: The concrete class. Built once at import; see `_build_theme_class`.
ThemeTokens: type = _build_theme_class()


#: The image provider lives in `app.ui.provider` so that importing this module does not
#: import `QtQuick`. `app.main --selftest` constructs a `UiBridge` for its render-metadata
#: stage and must stay clear of the QML stack; `tests.test_ui_contract` asserts that
#: `PySide6.QtQuick` is absent from `sys.modules` afterwards.
def __getattr__(name: str) -> Any:
    """Resolve `FrameImageProvider` on first use. See the note above.

    Present so `from app.ui.bridge import FrameImageProvider` keeps working for the
    windowed path and the offscreen render tests without making the QML import eager.
    """
    if name == "FrameImageProvider":
        from app.ui.provider import FrameImageProvider

        return FrameImageProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class UiBridge(QObject):
    """The QML-facing state object. One instance per window; lives in the UI thread.

    **Constructed with no arguments and it renders honestly**: every optional number is
    `None`, the route is `pending`, no frame has arrived, and the utilisation cards are
    disabled with their measured reason. A bridge that started with plausible numbers
    would make the first frame of a cold start indistinguishable from a running system.
    """

    #: A frame is ready to be displayed. The argument is the frame's identity tuple.
    frameArrived = Signal(object)
    #: An inference result arrived. The argument is the `InferenceResult`.
    resultArrived = Signal(object)
    #: The route changed. The argument is a `RoutePlan`-shaped dict.
    routeChanged = Signal(object)
    #: A telemetry sample set arrived. The argument is a tuple of `MetricSample`.
    metricUpdated = Signal(object)
    #: A business event was emitted. The argument is a `BusinessEvent`-shaped dict.
    eventEmitted = Signal(object)

    # -- cross-thread entry points -------------------------------------------------
    #
    # The supervisor's reader and feeder threads publish through these, and this object
    # lives in the UI thread. Qt turns that into a **queued** connection automatically,
    # which is exactly what AGENTS.md rule 11 requires: no thread touches a widget, and
    # the slot runs where the widgets are.
    #
    # `QMetaObject.invokeMethod` with `Q_ARG("QVariant", ...)` was the first mechanism and
    # it is not used: it needs the exact registered type name, and a wrong one fails
    # silently - the call is dropped and the UI simply never updates, which is a stall
    # with no error. A signal with a plain Python object argument cannot be wrong that
    # way.

    #: A route plan arrived from the supervisor's thread.
    routeArrived = Signal(object)
    #: An error text arrived from the supervisor's thread.
    errorArrived = Signal(str)
    #: A frame plus its already-filtered boxes arrived from the supervisor's thread.
    frameArrivedFromWorker = Signal(object)
    #: A whole telemetry sample set arrived.
    samplesArrived = Signal(object)

    @Slot(object)
    def apply_route(self, payload: object) -> None:
        """Apply **one stream's** route payload, on the UI thread. Guarded; see
        `apply_frame`.

        **The whole reason these four slots exist is that nothing called them.** Until the
        live path was wired, `run_windowed` connected supervisor lambdas that emitted these
        signals and then nothing consumed them, so the badge, the metrics, the frame and
        the event counts stayed at their cold-start values for the whole session with no
        error anywhere.

        ## M2 Task 4: the payload is per stream, and that is the whole change

        `_on_route` used to publish one payload for "the first stream" whose `state` was
        the **aggregate** `RoutePlan.state`. With one camera the aggregate is that camera's
        state, so it was harmless; with three it is a lie - a plan where Camera A is running
        and Camera C has failed publishes `"state": "active"`, and Camera A's card renders
        a healthy route over a dead camera.

        So each stream's payload is recorded separately in `_stream_routes`, and the card
        reads **only** the stream named by `route_stream`. Two consequences worth stating:

        - **A payload for another stream cannot move this card.** Its state, badge and
          fallback flag are stored and never consulted by `device_display`.

        ## M2 Task 5: the record also carries that stream's own stated reasons

        Task 4 dropped `errors` because `RoutePlan` had no per-stream channel. Task 5
        added `RoutePlan.stream_errors` and `Supervisor.route_plan()` populates it, so
        `errors` is back - and it is **that stream's** reasons, never the aggregate. A card
        reading `failed` with an empty `errors` list is still a card asking the presenter
        to guess, so `stream_card` renders the word "no reason was stated" rather than
        leaving the line blank. See `stream_reasons`.
        """
        try:
            data = dict(payload)  # type: ignore[arg-type]
            stream = str(data.get("stream") or self._route_stream)
            record = {
                "requested": data.get("requested"),
                "confirmed": tuple(data.get("confirmed") or ()),
                # `None` when the stream has declared no state, kept as `None` so
                # `stream_states` **omits** it rather than asserting a word for it. The
                # coercion to a renderable word happens below, for the card only.
                "state": data.get("state"),
                "fell_back": data.get("fell_back"),
                # This camera's own reasons. A `None`/absent entry means it has stated
                # none, which is different from stating none, so it is kept as a tuple and
                # never replaced with the aggregate.
                "errors": tuple(str(e) for e in (data.get("errors") or ())),
            }
            if self._stream_routes.get(stream) != record:
                self._stream_routes[stream] = record
                self.streamStatesChanged.emit()
            if stream == self._route_stream:
                # **The coercion is here and only here.** "Declared nothing" is drawn as
                # `pending`, which claims nothing about health; the stored record keeps the
                # `None` so anything reading `stream_states` sees the absence.
                self.set_route(requested=record["requested"],
                               confirmed=record["confirmed"],
                               state=str(record["state"] or "pending"),
                               fell_back=record["fell_back"])
            self.routeChanged.emit({"state": self._route_state,
                                    "confirmed": self._device_confirmed,
                                    "stream": stream})
        except Exception as exc:  # noqa: BLE001 - a slot failure is not a crash
            self._report_slot_failure("apply_route", exc)

    @Slot(str)
    def apply_error(self, text: str) -> None:
        """Show one stated reason. Additive: a later error does not erase an earlier one."""
        try:
            existing = self._error_message
            message = f"{existing}\n{text}" if existing else str(text)
            self.set_error(message)
        except Exception as exc:  # noqa: BLE001 - a slot failure is not a crash
            self._report_slot_failure("apply_error", exc)

    @Slot(object)
    def apply_frame(self, payload: object) -> None:
        """Record a frame, its **pixels**, its zones and its events, on the UI thread.

        Queued from the supervisor's reader thread by Qt, so everything here runs where
        the widgets are (AGENTS.md rule 11).

        ## The pixels are rebuilt here, not shipped as a `QImage`

        A `QImage` does not survive being carried in a `Signal(object)` payload, and
        building one on the worker thread would be a `QImage` touched from off the UI
        thread. So the payload carries the raw `numpy` array and the swap to RGB happens
        in `set_frame_image`, on this side, exactly once per frame.

        **This method had no caller** while the live path was unwired, which is why the
        image provider served its 1x1 transparent placeholder for a whole session while
        `frame_ready` went `True` - a hero claiming a frame it did not have.

        ## A failure here is reported, not swallowed

        **An exception raised inside a Qt slot does not propagate.** Qt catches it, prints
        the traceback to stderr - which on a booth machine goes nowhere - and carries on,
        so the screen keeps rendering a stale state and the operator sees nothing wrong.
        Measured here while wiring this method: a payload whose `zones` were zone *names*
        rather than zone objects produced 240 tracebacks in one 14-second run and the
        overlay, the zones and the event counts simply never updated.

        So the body is wrapped: the traceback is logged, and the **stated reason is put on
        the error line**, which is the one place an operator will actually see it. The
        stream is marked failed and the rest of the screen keeps running (rule 12).
        """
        try:
            self._apply_frame_payload(payload)
        except Exception as exc:  # noqa: BLE001 - a stream failure is not a crash
            self._report_slot_failure("apply_frame", exc)

    def _apply_frame_payload(self, payload: object) -> None:
        data = dict(payload)  # type: ignore[arg-type]
        # **M2: the payload is one STREAM's frame**, and it is stored under that stream's
        # key. Before this, every stream's frame landed on the single `_frame_image` and
        # every tile asked the provider for `hero`, so all three surfaces showed whichever
        # camera arrived last - which is the mislabelling this project keeps finding, in
        # its most visible form: a tile captioned "Camera B" showing Camera A's shopper.
        stream = str(data.get("stream") or self._route_stream)
        image = data.get("image")
        if image is not None:
            self._store_image(image, stream)
        zones = data.get("zones")
        if zones and not self._zones:
            # Once only. Zones are a property of the scenario pack, not of the frame, and
            # re-publishing them 15 times a second would rebuild the repeater for nothing.
            self._publish_zones(zones, _Extents(data.get("width"), data.get("height")))
        boxes = list(data.get("boxes") or ())
        # **The boxes belong to that stream too.** `bridge.overlay_boxes` stays the hero's,
        # because the hero draws the overlay and the hero is the bound stream; the record
        # is kept per stream so a hero switched to another camera draws *that* camera's
        # boxes and not the previous one's.
        self._stream_boxes[stream] = tuple(boxes)
        if stream == self._route_stream:
            self._overlay_boxes = tuple(boxes)
        self.overlayBoxesChanged.emit()
        self.streamFramesChanged.emit()
        for event in data.get("events") or ():
            self.record_event(event)
        if not self._frame_ready:
            self._frame_ready = True
            self.frameReadyChanged.emit()

    def _report_slot_failure(self, where: str, exc: BaseException) -> None:
        """Log a slot failure with its traceback and show the stated reason.

        The error line is **additive** - a later failure does not erase an earlier one -
        because a running screen that says one thing and fails at another is worse than
        one that says it failed. The reason is truncated because the line is read at
        booth distance, and the full traceback is in the log.
        """
        traceback.print_exc()
        reason = f"{where} failed: {type(exc).__name__}: {exc}"
        self.set_error(f"{self._error_message}\n{reason[:400]}" if self._error_message
                       else reason[:400])

    def _store_image(self, array: Any, stream: str) -> None:
        """Turn a BGR `numpy` array into this bridge's `QImage`, on the UI thread.

        The BGR-to-RGB swap happens here because `cv2` hands BGR and
        `QImage.Format_RGB888` wants RGB; getting it wrong renders a plausible-looking
        image with the red and blue channels exchanged, which nobody notices on a dark
        retail scene and everybody notices on an orange.

        **A copy, always.** A `QImage` over a numpy buffer would alias the clip's next
        decode and tear the frame being drawn - the same class of defect as a box drawn
        against the wrong frame, and just as invisible.
        """
        rgb = np.ascontiguousarray(array[:, :, ::-1])
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        qimage = QImage(rgb.tobytes(), width, height, width * 3,
                        QImage.Format_RGB888)
        stored = qimage.copy()
        # **M2: stored under this stream's key, and the bound stream's key is also the
        # hero's.** `frame_image` / `frame_revision` are the **bound stream's** frame and
        # stay exactly that - every M0/M1 reader of them keeps working, and they now mean
        # something precise: "the camera this screen is showing".
        record = self._stream_frames.setdefault(str(stream), {})
        record["image"] = stored
        record["revision"] = int(record.get("revision") or 0) + 1
        record["width"] = width
        record["height"] = height
        if str(stream) == self._route_stream:
            self._frame_image = stored
            self._frame_revision = int(record["revision"])
        self.frameRevisionChanged.emit()
        self.frameImageChanged.emit()
        # **And the per-stream record.** The tiles bind to `stream_frame_revisions`, whose
        # notify is `streamFramesChanged` and nothing else - so without this line a tile's
        # `?v=` never moves, it fetches once, and it shows frame 0 for the whole session.
        # Measured: the first per-stream render came back with the metrics column drawn and
        # the hero **empty**, with no QML warning, for exactly this reason.
        self.streamFramesChanged.emit()

    def _publish_zones(self, zones: Iterable[Any], frame: Any) -> None:
        """Publish zone rectangles once, against the frame they are drawn over."""
        self.set_zones(zones, frame)

    @Slot(object)
    def apply_samples(self, samples: object) -> None:
        """Take one whole telemetry sample set from the collector's reader. Guarded."""
        try:
            self.set_samples(tuple(samples))  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - a slot failure is not a crash
            self._report_slot_failure("apply_samples", exc)

    # -- every property raises its own change signal, so QML bindings re-evaluate only
    #    what actually moved. A single "anything changed" signal would re-evaluate every
    #    binding at 4 Hz and rebuild the overlay for a CPU reading.

    frameReadyChanged = Signal()
    overlayBoxesChanged = Signal()
    deviceConfirmedChanged = Signal()
    deviceRequestedChanged = Signal()
    routeStateChanged = Signal()
    aiFpsChanged = Signal()
    frameAgeP50Changed = Signal()
    frameAgeP95Changed = Signal()
    systemCpuPercentChanged = Signal()
    appCpuPercentChanged = Signal()
    gpuAvailableChanged = Signal()
    npuAvailableChanged = Signal()
    eventCountsChanged = Signal()
    statusMessageChanged = Signal()
    errorMessageChanged = Signal()
    #: The zone rectangles changed. Declared with the others so `zones` can name it in
    #: its `notify` argument; defining it beside the property would be a `NameError`.
    zonesChanged = Signal()
    #: The newest frame's pixels changed. QML's image `source` carries this as a query
    #: string so the hero is re-fetched; see `set_frame_image`.
    frameRevisionChanged = Signal()
    #: **The device badge's text changed.** Added because `device_display` and
    #: `device_fell_back` were `constant=True` with no notify signal, which is legal Qt and
    #: a silent lie here: the badge is the one string the specification names as
    #: load-bearing, and a `constant` property is evaluated **once**, when the QML binding
    #: that reads it is first created. The bridge's Python value changed and the screen
    #: never did, so a run that confirmed CPU on request left the badge reading
    #: "pending - no device selected" for the whole session. The offscreen render test did
    #: not catch it because it called `set_route` *before* `engine.load`.
    deviceDisplayChanged = Signal()
    #: The fallback flag changed. Same defect and same fix; the badge's colour and the
    #: "(fell back from ...)" suffix both read it.
    deviceFellBackChanged = Signal()
    #: The newest frame's `QImage` changed. The image provider reads the attribute
    #: directly from Python so it never needed this, but `frame_image` being `constant`
    #: meant a QML binding on it could never re-read either.
    frameImageChanged = Signal()
    #: The formatted metric strings changed. Emitted whenever any single metric's notify
    #: signal is, because the whole map is one property and QML binds to the map.
    metricsTextChanged = Signal()
    #: The staleness word changed, for the same reason.
    stalenessChanged = Signal()
    #: The per-device "why is there no figure" strings changed.
    unavailableNoticeChanged = Signal()
    #: The per-metric window strings changed, for the same reason as the metric map.
    windowNotesChanged = Signal()
    #: The number of analysed frames changed. See `frames_analysed`.
    framesAnalysedChanged = Signal()
    #: The **per-stream** route map changed. **M2 Task 4.**
    #:
    #: Declared separately from `routeStateChanged` because `route_state` is the state of
    #: the *one* stream this screen is bound to, and a three-stream plan must not move it.
    #: A screen with three cards (Task 5) will read `stream_states` per card; this signal
    #: is what lets it re-read.
    streamStatesChanged = Signal()
    #: The **per-stream card records** changed. **M2 Task 5.**
    #:
    #: Declared separately from `streamStatesChanged` because a card is more than a route:
    #: it also carries this stream's measured frame age, backlog and analysis rate, which
    #: move on the telemetry tick with no route change at all. One signal for both would
    #: mean a card re-reads the whole record set on every metric tick, and - worse - would
    #: make it impossible to tell which of the two caused a repaint.
    streamCardsChanged = Signal()
    #: A **per-stream frame** arrived, or the hero moved to another stream. **M2.**
    #:
    #: Separate from `frameRevisionChanged` because that one is the **hero's** revision and
    #: must not move when a background thumbnail's camera delivers a frame - otherwise
    #: every camera at 15 fps re-fetched the hero's image 45 times a second.
    streamFramesChanged = Signal()
    #: **Attract mode was entered or left. M3 Task 4.** `attract_active` is the one
    #: property the attract overlay binds to; without this signal the Loader would be
    #: evaluated once at `engine.load` and the idle screen would never appear.
    attractActiveChanged = Signal()
    #: **Recorded-demonstration mode was entered or left, or its record was replaced. M5.**
    #: Everything the recorded surface reads - `recorded_mode`, `recorded_metrics`,
    #: `recorded_source_text`, `recorded_metrics_note` - is notified by this one signal,
    #: because they are one fact: a record is (or is not) loaded.
    recordedModeChanged = Signal()
    #: The accepted live device request changed. See `deviceControl`.
    deviceControlChanged = Signal()
    #: A live device control asked to move a camera. `(stream, device)`.
    #:
    #: **Never emitted while recorded mode is active.** The windowed run connects this to
    #: the supervisor's existing presenter-selection path; a playback has no supervisor,
    #: and a control that still emitted here would be a live action behind a recorded
    #: screen.
    deviceRequested = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._frame_ready = False
        self._overlay_boxes: tuple[dict[str, Any], ...] = ()
        self._device_confirmed: str | None = None
        self._device_requested: str | None = None
        self._route_state = "pending"
        self._route_fell_back = False
        #: **Which stream this screen is showing.** The booth renders one camera, so one
        #: card, and this names whose route that card is allowed to read. Everything the
        #: card shows comes from this stream and nothing else - which is the point of the
        #: per-stream migration, because the aggregate would put Camera A's card on Camera
        #: C's health.
        self._route_stream = str(DEFAULT_ROUTE_STREAM)
        #: `stream_id -> {requested, confirmed, state, fell_back}`, kept for every stream
        #: the supervisor has published, whether or not this card shows it.
        self._stream_routes: dict[str, dict[str, Any]] = {}
        #: `stream_id -> {age_p50_ms, age_p95_ms, backlog, analysis_fps,
        #: window_samples}`. **Separate from the route record on purpose**: these arrive
        #: from the telemetry tick on a different clock, and merging them on arrival would
        #: let a route change look like a measurement. `stream_cards` merges them once, at
        #: read time.
        self._stream_metrics: dict[str, dict[str, Any]] = {}
        #: How many cameras this screen shows a card for, declared by `set_streams` rather
        #: than inferred from what has arrived. See `_get_stream_ids`.
        self._declared_streams: set[str] = set()
        self._metrics: dict[str, float | None] = {name: None for name in OPTIONAL_METRICS}
        #: How many analysed frames the `frame_age_*` percentiles are taken over. `0` means
        #: "nobody has said", and `age_window_note` then renders nothing rather than a
        #: number. See `_get_age_window_note`.
        self._age_window_samples = 0
        #: The `MetricSample` each optional metric came from. Held so `window_text` can
        #: branch on `validity` - the obligation is about the *ordering*, and that needs
        #: the record, not just its value.
        self._samples: dict[str, MetricSample] = {}
        self._zones: tuple[dict[str, Any], ...] = ()
        self._event_counts: dict[str, int] = {}
        self._status_message = ""
        self._error_message = ""
        self._displayed_identity: tuple[Any, ...] | None = None
        self._latest_identity: tuple[Any, ...] | None = None
        self._frame_image: QImage | None = None
        self._frame_revision = 0
        #: `stream_id -> {image, revision, width, height}`. **M2: one frame per camera.**
        #:
        #: This is what closes the disclosed gap "per-camera video surfaces are not
        #: built": before it there was one `_frame_image` and every tile asked the provider
        #: for `hero`, so all three surfaces showed whichever camera's frame arrived last.
        #: A tile now asks for **its own** stream and there is nothing for the provider to
        #: serve but that stream's pixels - so a tile cannot show another camera's picture
        #: even by accident.
        self._stream_frames: dict[str, dict[str, Any]] = {}
        #: `stream_id -> tuple of overlay boxes` for that camera's newest frame. Kept
        #: separately from `_overlay_boxes`, which is the **hero's**, so switching the hero
        #: to another camera draws that camera's boxes rather than the previous one's.
        self._stream_boxes: dict[str, tuple] = {}
        #: How many frames have been analysed. Set from the supervisor, not counted here:
        #: the bridge sees applied results, and "applied" is not the same word as
        #: "analysed". See `frames_analysed`.
        self._frames_analysed = 0
        #: **Attract mode (M3 Task 4).** The one piece of state attract mode owns. It is a
        #: display flag and nothing else: entering it hides the live screen and shows the
        #: recorded headlines, and leaving it restores the live screen **exactly** because
        #: no other field was touched. See `set_attract` and `tests/test_ui_contract.py`.
        self._attract_active = False
        #: The `perf_counter` reading of the last presenter input, for the idle timer. Set
        #: at construction so a window that is never touched enters attract after the idle
        #: interval rather than immediately.
        self._last_activity = time.perf_counter()
        #: **Recorded-demonstration mode (M5 Task 4).** The loaded record, or `None`. This
        #: is a *playback*, never a measurement: `_recorded_metrics` is derived from the
        #: record's own samples, and each row carries the record's timestamp. The live
        #: fields above are deliberately left untouched so leaving recorded mode restores
        #: the live screen exactly.
        self._recorded_record: BenchmarkRecord | None = None
        self._recorded_metrics: tuple[dict[str, str], ...] = ()
        self._recorded_metrics_hidden = 0
        self._recorded_source = ""
        self._recorded_source_name = ""
        self._recorded_refusal = ""
        #: Whether a live source is attached. Set by `run_windowed` when the supervisor
        #: starts and cleared when it stops. A playback cannot be entered while this is
        #: true, or while any live evidence exists - see `live_stream_running`.
        self._live_source_attached = False
        #: The last **accepted** live device request, empty before one. Reading back what
        #: was accepted (not what was clicked) is what keeps a disabled control from
        #: appearing to have done something.
        self._device_control = ""

        # Utilisation availability is **derived from the sample**, not asserted here: the
        # provider (`app.telemetry.accelerator`) measures this application's own engine
        # time and this module only formats what it is handed. A fresh bridge has no
        # sample, so both cards read unavailable - which is the truth before the first
        # tick, not a claim about the machine.

    # ------------------------------------------------------------------ properties

    def _get_frame_ready(self) -> bool:
        return self._frame_ready

    frame_ready = Property(bool, _get_frame_ready, notify=frameReadyChanged)

    def _get_overlay_boxes(self) -> list:
        return [dict(box) for box in self._overlay_boxes]

    overlay_boxes = Property("QVariantList", _get_overlay_boxes,
                             notify=overlayBoxesChanged)

    def _get_device_confirmed(self):
        return self._device_confirmed

    device_confirmed = Property(str, _get_device_confirmed,
                                notify=deviceConfirmedChanged)

    def _get_device_requested(self):
        return self._device_requested

    device_requested = Property(str, _get_device_requested,
                                notify=deviceRequestedChanged)

    def _get_route_state(self) -> str:
        return self._route_state

    route_state = Property(str, _get_route_state, notify=routeStateChanged)

    def _get_route_stream(self) -> str:
        """Which stream's route the single card is reading."""
        return self._route_stream

    route_stream = Property(str, _get_route_stream, notify=streamStatesChanged)

    def _get_stream_states(self) -> dict:
        """`stream_id -> that stream's own state`.

        **`RoutePlan.state` is the aggregate and is not in here.** With three streams a
        plan where Camera A is running and Camera C has failed publishes
        `state == "active"`, and a card reading that shows a healthy route over a dead
        camera. Each entry here is `RoutePlan.stream_state(stream)` - this camera's own
        answer - and a stream that has declared none is **absent**, which QML must read
        as unknown rather than as healthy.
        """
        return {sid: str(entry["state"]) for sid, entry in self._stream_routes.items()
                if entry.get("state")}

    stream_states = Property("QVariantMap", _get_stream_states,
                             notify=streamStatesChanged)

    def _get_stream_routes(self) -> dict:
        """Every stream's route record: requested, confirmed, state, fell_back."""
        return {sid: dict(entry) for sid, entry in self._stream_routes.items()}

    stream_routes = Property("QVariantMap", _get_stream_routes,
                             notify=streamStatesChanged)

    def _get_stream_confirmed_labels(self) -> dict:
        """`stream_id -> that camera's confirmed device as ONE STRING`, `""` if none.

        **M2, added because a caption printed `-1`.** `stream_routes[sid]["confirmed"]` is a
        Python tuple that reaches QML as a `QVariantList`, and in this PySide6 it arrives as
        something truthy with no usable `length` and no useful `toString`: the thumbnail
        caption rendered the literal string `-1` on every camera, on a render where the
        Python side held `confirmed: ('GPU.0',)` and the picture above the caption was real.

        So the joining happens **here**, in Python, where the value is a tuple and
        `"· ".join(...)` is a string. QML gets a plain string, and `""` means "no confirmed
        device", which the caption renders as words.
        """
        return {sid: ", ".join(str(d) for d in (entry.get("confirmed") or ()))
                for sid, entry in self._stream_routes.items()}

    stream_confirmed_labels = Property("QVariantMap", _get_stream_confirmed_labels,
                                      notify=streamStatesChanged)

    # -- per-camera frames ------------------------------------------------------------

    def _get_stream_frame_revisions(self) -> dict:
        """`stream_id -> that camera's own frame revision`. **The QML binding surface.**

        A tile's `Image.source` is `"image://frames/" + streamId + "?v=" + revision`, and it
        reads the revision out of this **notified** map. It does not call a method for it: a
        QML binding that calls a method declares no property dependency and is evaluated
        once, which is the defect this repository has now found four times.

        A stream with no frame is **absent from the map**, not present with a 0. `0` would
        be indistinguishable from "frame 0 has just been written", and the tile's
        `visible:` guard is `frames_ready[streamId] !== true` rather than a revision test.
        """
        return {sid: int(entry.get("revision") or 0)
                for sid, entry in self._stream_frames.items()}

    stream_frame_revisions = Property("QVariantMap", _get_stream_frame_revisions,
                                     notify=streamFramesChanged)

    def _get_stream_frames_ready(self) -> dict:
        """`stream_id -> whether that camera has delivered a frame at all`.

        **A camera with no frame is a distinct state from a camera with a stale one**, and
        the tile has to be able to say which. `True` only once a real `QImage` has been
        stored for that stream; absent from the map until then.
        """
        return {sid: True for sid, entry in self._stream_frames.items()
                if entry.get("image") is not None
                and not entry["image"].isNull()}

    stream_frames_ready = Property("QVariantMap", _get_stream_frames_ready,
                                  notify=streamFramesChanged)

    def frame_for(self, stream_id: str) -> QImage | None:
        """This stream's newest `QImage`, or `None`. **For the image provider only.**

        `None` for a stream with no frame, and **never a fallback to another stream's**:
        that is the whole guarantee. A tile whose camera has produced nothing gets nothing
        and says so in words, rather than holding whatever the hero is showing.
        """
        entry = self._stream_frames.get(str(stream_id))
        if entry is None:
            return None
        image = entry.get("image")
        return image if image is not None else None

    def overlay_boxes_for(self, stream_id: str) -> tuple:
        """This stream's newest overlay boxes. Hero only; thumbnails draw none."""
        return tuple(self._stream_boxes.get(str(stream_id), ()))

    def set_stream_frame(self, stream_id: str, image: Any) -> None:
        """Publish one camera's frame. **UI thread.** Used by the offscreen render tests.

        `set_frame_image` remains the single-stream entry point M0/M1 use, and forwards here
        with the bound stream - so there is one implementation of "turn BGR into a QImage and
        count the revision", not two that can drift.
        """
        if image is None:
            return
        self._store_image(image, str(stream_id))

    def stream_reasons(self, stream_id: str) -> str:
        """One stream's own stated reasons as **one renderable line**, or `""`.

        **Only that stream's.** With three streams the supervisor's `errors` is the
        concatenation of all three, and a card showing the concatenation reads as Camera
        C's failure with Camera A's and Camera B's sentences attached - a presenter cannot
        tell which sentence is about the camera in front of them.

        `""` when the stream has stated none. That is a real state - a stream that is
        `pending` has nothing to explain yet - and `stream_card` renders its own words for
        it rather than printing an empty line that reads like a rendering failure.
        """
        entry = self._stream_routes.get(str(stream_id)) or {}
        reasons = entry.get("errors") or ()
        if not reasons:
            return ""
        # Newest last, so the most recent cause is the last thing read, which is the one a
        # presenter needs. `_Stream.errors` is only ever appended to and is cleared by
        # that stream's own `clear_proof()`.
        return "  ".join(str(reason) for reason in reasons)

    # -- the per-stream card payload --------------------------------------------------
    #
    # **One owner for the route, one for the measurement, merged once.** The route record
    # (requested / confirmed / state / fell_back / errors) is owned by `apply_route`,
    # because it arrives from the supervisor. The measured numbers (frame ages, backlog,
    # this stream's own analysis rate) are owned by `set_stream_metrics`, because they
    # arrive from the telemetry tick. `stream_cards` merges them and formats **once**,
    # in Python, so no card string is composed in QML and there is nowhere for two cards to
    # disagree about one stream.

    def set_stream_metrics(self, stream_id: str, *, age_p50_ms: float | None,
                           age_p95_ms: float | None, backlog: int | None,
                           analysis_fps: float | None,
                           window_samples: int) -> None:
        """Record one stream's measured lateness and rate. UI thread; from the tick.

        Every argument is `float | None` and **`None` is preserved**. `None` means "not
        measured yet" and renders as `unavailable`; it never becomes `0`. A stream that has
        analysed nothing showing `age 0.0 ms · backlog 0` is the most dangerous card this
        project can draw, because it looks exactly like a healthy one.

        `backlog` is passed in rather than derived here because only the supervisor knows
        what a drop was and what an invalidation was; see `app.main`.
        """
        wanted = {
            "age_p50_ms": None if age_p50_ms is None else float(age_p50_ms),
            "age_p95_ms": None if age_p95_ms is None else float(age_p95_ms),
            "backlog": None if backlog is None else int(backlog),
            "analysis_fps": None if analysis_fps is None else float(analysis_fps),
            "window_samples": int(window_samples),
        }
        if self._stream_metrics.get(str(stream_id)) == wanted:
            return
        self._stream_metrics[str(stream_id)] = wanted
        self.streamCardsChanged.emit()

    def _card_device(self, route: dict[str, Any]) -> str:
        """`"CPU"` / `"GPU"` / `"NPU"` / `""` for a route record.

        From the **confirmed** reading first, because that is the placement claim, and
        from the requested device only when nothing has confirmed anything. `""` for a
        failed or unstarted stream, and the card then says so in words - a failed route
        claims no device and must not borrow the one that was asked for.
        """
        from app.inference.devices import plugin_class

        confirmed = tuple(route.get("confirmed") or ())
        if confirmed:
            # `'GPU.0'` -> `'GPU'`, through the module that owns that spelling. Not a
            # substring test here, which is the trap `plugin_class` exists to remove.
            return str(plugin_class(confirmed[0]))
        requested = route.get("requested")
        if route.get("state") == "failed":
            return ""
        return str(requested or "")

    def _stream_card(self, stream_id: str) -> dict[str, Any]:
        """One card's record. Every string is already formatted and already honest."""
        route = self._stream_routes.get(stream_id) or {}
        metrics = self._stream_metrics.get(stream_id) or {}
        state = route.get("state") or "pending"
        confirmed = tuple(route.get("confirmed") or ())
        requested = route.get("requested")
        device = self._card_device(route)

        age_p50 = metrics.get("age_p50_ms")
        age_p95 = metrics.get("age_p95_ms")
        backlog = metrics.get("backlog")
        fps = metrics.get("analysis_fps")
        window_samples = int(metrics.get("window_samples") or 0)

        # **Staleness, in words.** `stale` is only ever true from a measured p50, never
        # from an absent one, and the sentence carries the same number the age line shows
        # so the two cannot disagree.
        stale = age_p50 is not None and age_p50 > STALE_AFTER_SECONDS * 1000.0
        if stale:
            # **Short enough to fit two lines.** The first wording - "STALE - the picture on
            # screen is 10.8 s old (over the 2 s in spec section 7)" - wrapped to three
            # lines at this card's width and pushed the activity line **below the card's
            # bottom edge**, where a Repeater delegate draws it anyway because nothing
            # clips. The age is repeated deliberately: the numeral above is the figure and
            # this is the sentence, and the ruling asks for the word as well as the number.
            staleness = (f"STALE - this picture is {_age(age_p50)} old; "
                         f"the limit is {STALE_AFTER_SECONDS:.0f} s")
        else:
            staleness = ""

        reasons = tuple(route.get("errors") or ())
        if reasons:
            reason = "  ".join(str(entry) for entry in reasons)
            # **M3 Task 2: the cut is decided here, at a stated budget, and it says so.**
            #
            # The supervisor's stated reason for a dead worker is 2331 characters (measured,
            # `docs/qualification/m3-capture/100pct.json`) and the card has room for about
            # two lines of it. Before this, the card asked Qt to do the cutting
            # (`wrapMode: WordWrap`, `maximumLineCount: 3`, `elide: Text.ElideRight`), so the
            # screen showed `route transition to generation 1 was not acknowledged: stream(s)
            # ['C'] did not confir...` and **nothing on screen said that 2 291 of the 2 331
            # characters were missing.** A silently shortened error is a hidden error: a
            # presenter reads a sentence that ends in an ellipsis and has no way to know it is
            # 40 of 2 331 characters - **1.7%** of the message.
            #
            # So `reason_display` is the string the card shows, cut at `CARD_REASON_CHARS`
            # with the marker naming how much is not shown. `reason` stays complete, because
            # the full text is still the honest record and the harness can read it.
            budget = int(_token("CARD_REASON_CHARS"))
            if len(reason) > budget:
                hidden = len(reason) - budget
                reason_display = (reason[:budget].rstrip()
                                  + _token("CARD_REASON_MARKER").format(hidden=hidden))
            else:
                reason_display = reason
        elif state == "failed":
            # Reachable only if a caller built a failed route by hand. Printed rather than
            # left blank, because a blank reason line reads as a rendering failure and
            # "failed" with nothing after it asks the presenter to guess.
            reason = "failed, and no reason was stated for this stream"
            reason_display = reason
        else:
            reason = ""
            reason_display = ""

        badge = ", ".join(confirmed) if confirmed else "no device is running"
        if route.get("fell_back") and requested:
            badge = f"{badge} (fell back from {requested})"

        return {
            "stream": stream_id,
            "state": str(state),
            "device": device,
            "requested": str(requested or ""),
            "confirmed": ", ".join(confirmed),
            "fell_back": bool(route.get("fell_back")),
            "badge": badge,
            "age_p50_text": UNAVAILABLE_TEXT if age_p50 is None else _age(age_p50),
            "age_p95_text": UNAVAILABLE_TEXT if age_p95 is None else _age(age_p95),
            "fps_text": UNAVAILABLE_TEXT if fps is None else _fps(fps),
            "backlog_text": UNAVAILABLE_TEXT if backlog is None else f"{int(backlog)} frames",
            "window_text": (f"{window_samples} analysed frames"
                            if window_samples > 0 else UNAVAILABLE_TEXT),
            "stale": bool(stale),
            "staleness": staleness,
            # **The complete reason, and the string the card shows.** Both are published, so
            # "what the screen said" and "what was actually reported" are separately readable
            # and a test can assert that the second admits the cut in the first.
            "reason": reason,
            "reason_display": reason_display,
            "summary_text": _summary_line(fps=fps, age_p50=age_p50, age_p95=age_p95),
        }

    def _get_stream_cards_by_id(self) -> dict:
        """`stream_id -> that camera's card record`. **The QML binding surface.**

        A map rather than a list because a card knows *which camera it is* and has to
        index; finding a record in a `QVariantList` from QML needs a loop or a helper Slot,
        and a helper Slot in a binding is the defect the standing guard exists for.

        Every value is a complete record - there is no key whose value is `undefined` -
        so no card can read a property of nothing. A stream with no route and no metrics
        yet still gets a cold record: `pending`, every figure `unavailable`, and no device.
        """
        return {stream_id: self._stream_card(stream_id) for stream_id in self.stream_ids}

    stream_cards_by_id = Property("QVariantMap", _get_stream_cards_by_id,
                                  notify=streamCardsChanged)

    def _get_stream_ids(self) -> list:
        """Every stream id this screen shows a card for, sorted.

        **The declared set, not just what has been published.** `set_streams` is called by
        `run_windowed` immediately after the supervisor is built, so the three cards exist
        from the first rendered frame with honest cold values - `pending`, `unavailable`,
        no device - rather than the metrics column being empty until the first route
        payload arrives. An empty column on a cold start is indistinguishable from a
        broken screen, which is the same defect as a card frozen at its first evaluation.
        """
        return sorted(self._declared_streams | set(self._stream_routes)
                      | set(self._stream_metrics))

    stream_ids = Property("QStringList", _get_stream_ids, notify=streamStatesChanged)

    @Slot("QStringList")
    def set_streams(self, stream_ids: list) -> None:
        """Declare which cameras this screen shows a card for. UI thread.

        **A declaration, not a measurement**: it says how many cameras there are, not what
        any of them is doing. Every figure on those cards still reads `unavailable` until
        something is measured, so declaring a stream cannot make a card claim anything.
        """
        wanted = {str(entry) for entry in stream_ids if str(entry)}
        if wanted == self._declared_streams:
            return
        self._declared_streams = wanted
        self.streamStatesChanged.emit()
        self.streamCardsChanged.emit()


    def _get_stream_cards(self) -> list:
        """One record per stream this supervisor owns, in a stable order.

        Built fresh on every read rather than cached, because it is a merge of two dicts
        and a cached copy is one more thing that can go stale. The list is only read when
        a binding re-evaluates, which the telemetry tick and every route change cause.
        """
        return [self._stream_card(stream_id) for stream_id in self.stream_ids]

    stream_cards = Property("QVariantList", _get_stream_cards,
                            notify=streamCardsChanged)

    def stream_card(self, stream_id: str) -> dict:
        """One camera's card record, for a card that knows which camera it is.

        A plain method rather than a Slot, so it is **not** what a QML binding should
        call: a method call in a binding declares no property dependency and is evaluated
        once. `stream_cards` is the binding surface; this exists for Python-side reads and
        for the tests, which assert on the record rather than on the pixels.
        """
        return self._stream_card(str(stream_id))

    def set_route_stream(self, stream_id: str) -> None:
        """Point the single card at another camera's route.

        The **only** per-stream field the card reads. Until Task 5 renders three cards
        this is what makes "the screen is showing Camera A" a fact rather than an
        assumption, and it is deliberately not reachable from QML: a binding that wrote it
        would let the screen point itself at whichever stream answered last.
        """
        wanted = str(stream_id)
        if wanted == self._route_stream:
            return
        self._route_stream = wanted
        entry = self._stream_routes.get(wanted)
        self.set_route(
            requested=(entry or {}).get("requested"),
            confirmed=tuple((entry or {}).get("confirmed") or ()),
            state=str((entry or {}).get("state") or "pending"),
            fell_back=(entry or {}).get("fell_back"))
        # **M2: the hero's PICTURE and BOXES move with the badge.** Pointing the screen at
        # Camera C and leaving Camera A's pixels on it is the mislabelling this project keeps
        # finding, so `frame_image` and `overlay_boxes` are re-derived from the new stream's
        # own record rather than left behind.
        record = self._stream_frames.get(wanted) or {}
        self._frame_image = record.get("image")
        self._frame_revision = int(record.get("revision") or 0)
        self._overlay_boxes = tuple(self._stream_boxes.get(wanted, ()))
        self.overlayBoxesChanged.emit()
        self.frameRevisionChanged.emit()
        self.frameImageChanged.emit()
        self.streamFramesChanged.emit()
        self.streamStatesChanged.emit()

    def _get_ai_fps(self):
        return self._metrics["ai_fps"]

    ai_fps = Property(float, _get_ai_fps, notify=aiFpsChanged)

    def _get_p50(self):
        return self._metrics["frame_age_p50_ms"]

    frame_age_p50_ms = Property(float, _get_p50, notify=frameAgeP50Changed)

    def _get_p95(self):
        return self._metrics["frame_age_p95_ms"]

    frame_age_p95_ms = Property(float, _get_p95, notify=frameAgeP95Changed)

    def _get_system_cpu(self):
        return self._metrics["system_cpu_percent"]

    system_cpu_percent = Property(float, _get_system_cpu, notify=systemCpuPercentChanged)

    # The two accelerator utilisation properties exist so that a QML binding can name
    # them and get an honest `None` rather than a binding error. `set_samples` fills them
    # from the accelerator provider; a fresh bridge, and a device this application has not
    # used, are `None` with a reason. **No code path here invents a value.**
    def _get_gpu_utilisation(self):
        return self._metrics["gpu_utilisation_percent"]

    gpu_utilisation_percent = Property(float, _get_gpu_utilisation,
                                      notify=gpuAvailableChanged)

    def _get_npu_utilisation(self):
        return self._metrics["npu_utilisation_percent"]

    npu_utilisation_percent = Property(float, _get_npu_utilisation,
                                      notify=npuAvailableChanged)

    def _get_app_cpu(self):
        return self._metrics["app_cpu_percent"]

    app_cpu_percent = Property(float, _get_app_cpu, notify=appCpuPercentChanged)

    def _get_gpu_available(self) -> bool:
        """Whether a per-process GPU figure is in hand. **Derived from the value.**

        Same rule as `cpu_available`: availability cannot drift from the number it
        describes. A fresh bridge and a bridge whose last sample was unavailable both
        read `False`; only a sample this application's own engine time produced reads
        `True`.
        """
        return self._metrics["gpu_utilisation_percent"] is not None

    gpu_available = Property(bool, _get_gpu_available, notify=gpuAvailableChanged)

    def _get_npu_available(self) -> bool:
        """Whether a per-process NPU figure is in hand. See `_get_gpu_available`."""
        return self._metrics["npu_utilisation_percent"] is not None

    npu_available = Property(bool, _get_npu_available, notify=npuAvailableChanged)

    def _get_cpu_available(self) -> bool:
        """Whether a system CPU reading is actually in hand.

        **Not a constant.** `Main.qml` used to pass `utilisationAvailable: true` for the
        CPU card, hardcoded, which meant an unavailable CPU reading would render coloured
        and undimmed as though it were live - the exact defect the two accelerator cards
        are careful about, applied to the one card that is supposed to be measurable.
        `psutil` can fail, and `MetricCollector` reports that as an `UNAVAILABLE` sample
        rather than a zero, so the card has to be able to say so.

        Availability is derived from the **value**, not tracked separately, so it cannot
        drift from the number it describes.
        """
        return self._metrics["system_cpu_percent"] is not None

    cpu_available = Property(bool, _get_cpu_available, notify=systemCpuPercentChanged)

    def _get_event_counts(self) -> dict:
        return dict(self._event_counts)

    event_counts = Property("QVariantMap", _get_event_counts, notify=eventCountsChanged)

    def _get_status_message(self) -> str:
        return self._status_message

    status_message = Property(str, _get_status_message, notify=statusMessageChanged)

    def _get_error_message(self) -> str:
        return self._error_message

    error_message = Property(str, _get_error_message, notify=errorMessageChanged)

    # ------------------------------------------------------------------ constants

    @Property(str, constant=True)
    def recorded_footage_label(self) -> str:
        """The mandatory persistent label. See `RECORDED_FOOTAGE_LABEL`."""
        return RECORDED_FOOTAGE_LABEL

    @Property(str, constant=True)
    def unavailable_text(self) -> str:
        """The literal a card shows instead of a number it does not have."""
        return UNAVAILABLE_TEXT

    @Property(str, constant=True)
    def utilisation_unavailable_text(self) -> str:
        """What a GPU or NPU card says when it has no figure. Never a digit."""
        return UTILISATION_UNAVAILABLE_TEXT

    @Property(str, constant=True)
    def scope_note(self) -> str:
        """The reduction, verbatim from `app/engine/retail_config.py`."""
        return SCOPE_NOTE_LABEL

    @Property(str, constant=True)
    def gpu_reason(self) -> str:
        """The short, card-sized reason a GPU card has no figure yet."""
        return UTILISATION_SHORT_REASON

    @Property(str, constant=True)
    def npu_reason(self) -> str:
        """The short reason for the NPU. Same provider, same wording; the sample carries
        the device-specific detail when there is one."""
        return UTILISATION_SHORT_REASON

    @Property(str, constant=True)
    def utilisation_measured_reason(self) -> str:
        """The full measured method and the attribution control, for the panel.

        A module constant, not a string imported from the telemetry layer: the earlier
        version quoted `VALUE_READS_REFUSED_REASON`, which described a binding fault that
        has since been fixed. A quote is only as true as its source.
        """
        return UTILISATION_MEASURED_REASON

    @Property("QStringList", constant=True)
    def explanation_keys(self) -> list[str]:
        """The explanation keys, **the one definition both surfaces offer**.

        **M3 Task 5.** The drawer's first version bound its `Repeater` to
        `bridge.explanationKeys`. This bridge has `explanations` and `explanation()` and
        **no** `explanationKeys`, so the model was `undefined`, the `Repeater` produced
        **zero delegates**, and the drawer opened with no explanation buttons at all - the
        M1 dead-affordance shape, one surface over, on the surface whose job is explaining
        the boxes. (Measured while fixing it: a `Loader`-loaded file *can* read the ids of
        the file that loaded it, so binding the drawer to `bridgeUi.explanationKeys` would
        have worked too - but that is a second name for the list and it left `Bridge.qml`
        carrying a literal that could drift from `EXPLANATIONS`. The list lives here now
        and both surfaces read it.)
        """
        return list(EXPLANATIONS)

    @Property("QVariantMap", constant=True)
    def explanations(self) -> dict:
        """The six required explanations, reachable from hover and from a pinned panel."""
        return dict(EXPLANATIONS)

    @Property(bool, notify=deviceFellBackChanged)
    def device_fell_back(self) -> bool:
        """Whether the confirmed device differs from the requested one.

        `False` when either is absent - an unproven route has not fallen back, it has
        **not been shown**, and reporting `False` there would read as "running where you
        asked".

        **`notify=deviceFellBackChanged`, not `constant=True`.** A `constant` property is
        read exactly once per binding; the fallback is discovered from the worker's first
        result, which is seconds after the screen was built. See `deviceDisplayChanged`.
        """
        return self._route_fell_back

    # ------------------------------------------------------------------ setters

    @Slot(str, result=str)
    def explanation(self, key: str) -> str:
        """The explanation text for `key`, or `""` for a key that does not exist.

        An unknown key returns `""` rather than raising: a QML binding that reaches for a
        key which does not exist should render nothing, not take the window down at a
        trade show. `tests/test_ui_contract.py` pins the six required keys separately.
        """
        return EXPLANATIONS.get(str(key), "")

    def set_route(self, requested: str | None = None, confirmed: Iterable[str] | None = None,
                  state: str = "pending", fell_back: bool | None = None) -> None:
        """Record the route.

        `confirmed` is the **`EXECUTION_DEVICES` reading**, not a device id, and it is
        normalised here so the NPU's bare string cannot be filed as three characters.
        `confirmed` of `None` or `()` means *nothing has proved a placement*, and
        `device_confirmed` is then `None` - never the requested device, and never an
        empty string that a card might render as a name.
        """
        devices = normalise_execution_devices(confirmed if confirmed is not None else ())
        text = ", ".join(devices) if devices else None
        if requested != self._device_requested:
            self._device_requested = requested
            self.deviceRequestedChanged.emit()
        if text != self._device_confirmed:
            self._device_confirmed = text
            self.deviceConfirmedChanged.emit()
            # The badge is built from the confirmed device, so a confirmed-device change
            # is a badge change. Without this the binding would keep whatever it read at
            # `engine.load`.
            self.deviceDisplayChanged.emit()
        if fell_back is None:
            fell_back = bool(
                requested and devices and not is_device(devices, requested))
        if bool(fell_back) != self._route_fell_back:
            self._route_fell_back = bool(fell_back)
            self.deviceConfirmedChanged.emit()
            self.deviceFellBackChanged.emit()
            # The badge's suffix depends on the fallback, so the badge itself changed.
            self.deviceDisplayChanged.emit()
        if state != self._route_state:
            self._route_state = str(state)
            self.routeStateChanged.emit()
            # "failed - no device is running" and "pending - X not yet confirmed" are
            # both branches of `device_display`, so a state change moves the badge too.
            self.deviceDisplayChanged.emit()

    def _get_metric_text(self) -> dict:
        """Every optional metric, already formatted, as `name -> rendered string`.

        ## Why this exists, and why it is not a second formatter

        **A QML binding that calls a method does not re-evaluate when the method's inputs
        change.** Qt tracks property dependencies, and `bridge.display_text("system_cpu_percent")`
        declares none - the binding is evaluated once, at the first render, and never again.
        Every card in `ComputeCard.qml` and the whole activity line in `Main.qml` were
        written that way, which is why they rendered `unavailable` on a live run whose
        bridge held `system_cpu_percent: 58.0`: the bridge was right and the **screen** was
        frozen at its first evaluation. This is the same defect as the `constant=True`
        device badge, one layer out - a value that reaches QML without a notify signal -
        and the live-path test caught it because it read pixels, not just the bridge.

        So the formatted strings are published as **one notified property** instead. The
        formatting itself is still `display_text`, called once per change here - there is
        still exactly one place in this module that knows how a number is rendered, and a
        test can still check that place directly.
        """
        return {name: self.display_text(name) for name in self._metrics}

    metric_text = Property("QVariantMap", _get_metric_text, notify=metricsTextChanged)

    def _get_window_notes(self) -> dict:
        """Every metric's measurement window, already formatted, as `name -> text`.

        The `windowNote` binding used to call `window_text(name)` directly, which is a Slot
        call and therefore evaluated once at `engine.load` - so the "measured over N ms"
        line never appeared in a live session, with its `visible:` guard false throughout.
        See `_get_metric_text` for why a method call cannot work here.
        """
        return {name: self.window_text(name) for name in self._metrics}

    window_notes = Property("QVariantMap", _get_window_notes, notify=windowNotesChanged)

    def _get_age_window_note(self) -> str:
        """What the `frame_age_p50_ms`/`p95_ms` figures are percentiles **of**.

        ## Why this exists, and why it is a count and not a duration

        Those two are percentiles of the supervisor's retained frame ages, which is a
        `deque(maxlen=FRAME_AGE_WINDOW)`. They are **not** measured over a second, a tick,
        or the session. The window is *the most recent N analysed frames*, and how many
        seconds that is **depends on how fast the stream is being analysed** - which is the
        number nobody can read off a percentile.

        Measured on 2026-10-03, three streams on the CPU, 15 fps offered each:

        - a stream keeping up at ~15 fps analyses ~15 frames/s, so 256 ages span **~17 s**;
        - the same three cameras under six CPU saturators analysed ~3.5 fps (28 frames per
          8 s), so 256 ages span **~73 s**;
        - all three on the CPU with no saturators - spec section 5 row 4, the app's own
          steady-state load - analysed 9.60-9.80 fps, so ~26 s, with a measured frame age
          **p50 10.8-11.3 s over three consecutive 120 s holds** (Task 3). An earlier,
          shorter hold read p50 8239 ms; the longer, repeated measurement is the one to
          quote and the two are not interchangeable.

        So a p50 of 8 seconds is the p50 of a ~26 s window and a p50 of 2 seconds is the
        p50 of a ~17 s one. **Without the window the two are not comparable**, and the
        field name does not carry it.

        The count is **published, not imported**: `app.main` passes
        `FRAME_AGE_WINDOW` in, so the constant stays in one place
        (`app/supervisor/runner.py`) and this only formats what it is told. A bridge that
        imported the supervisor to read the constant would drag the worker stack into the
        UI module for a number.
        """
        samples = self._age_window_samples
        if samples <= 0:
            return ""
        text = f"most recent {samples} analysed frames"
        fps = self._metrics.get("ai_fps")
        if fps is not None and fps > 0:
            return f"{text} - about {samples / fps:.0f} s at {fps:.1f} fps"
        # Not a guess and not a placeholder: the span is genuinely unknown until the
        # stream's own rate has been measured, and saying so beats implying 15 fps.
        return f"{text} - span depends on this stream's own rate"

    age_window_note = Property(str, _get_age_window_note, notify=windowNotesChanged)

    def set_age_window(self, samples: int) -> None:
        """Record how many frame ages the percentiles are taken over. See the property."""
        wanted = int(samples)
        if wanted == self._age_window_samples:
            return
        self._age_window_samples = wanted
        self.windowNotesChanged.emit()

    def _get_unavailable_notice(self) -> dict:
        """`deviceId -> what to show when that card has no figure`.

        **Per card, because the reason is per card.** The GPU and NPU cards show
        `UTILISATION_UNAVAILABLE_TEXT` until a sample lands. The CPU card shows
        `CPU_UNAVAILABLE_TEXT`, because the true fact there is that no sample has landed
        yet - `psutil.cpu_percent` is being called every 250 ms and works.

        Sharing one literal made every run announce "Utilization unavailable" on the CPU
        card during startup, before the first telemetry tick, and asserted a failure to
        read a counter that is not failing. When a `MetricSample` carries an
        `unavailable_reason`, that reason is preferred over the default, because it is the
        measurement's own words - that is now true of the GPU and NPU samples as well, so a
        card whose device this application has not used says *that*, rather than the
        generic word.

        The three keys are fixed rather than derived from whatever the bridge happens to
        know, so a card asking for a device with no entry gets an empty string and QML's
        `!== undefined` guard falls through to `unavailable_text` - never to a blank.
        """
        notices = {
            "CPU": CPU_UNAVAILABLE_TEXT,
            "GPU": UTILISATION_UNAVAILABLE_TEXT,
            "NPU": UTILISATION_UNAVAILABLE_TEXT,
        }
        for device, metric in (("CPU", "system_cpu_percent"),
                               ("GPU", "gpu_utilisation_percent"),
                               ("NPU", "npu_utilisation_percent")):
            sample = self._samples.get(metric)
            if sample is not None and sample.validity is Validity.UNAVAILABLE:
                reason = (sample.unavailable_reason or "").strip()
                if reason:
                    notices[device] = reason
        return notices

    unavailable_notice = Property("QVariantMap", _get_unavailable_notice,
                                  notify=unavailableNoticeChanged)

    def _get_frames_analysed(self) -> int:
        """How many frames have been analysed. `0` before the first result.

        **Not the route state.** The event-count row used to print a measured `0` for
        every kind as soon as the device was confirmed - but a confirmed placement is
        established from the worker's route, which arrives *before* any frame has been
        analysed. So there was a real window, route confirmed and zero frames analysed, in
        which the footer printed three hard zeros for an event machine that had not run.
        A count of events may only read `0` once something could have counted one.
        """
        return self._frames_analysed

    frames_analysed = Property(int, _get_frames_analysed, notify=framesAnalysedChanged)

    # -- attract mode (M3 Task 4) ------------------------------------------------------
    #
    # **Attract mode owns exactly one field.** It is a display state: it hides the live
    # screen and shows recorded, dated headlines, and it touches nothing else. That is what
    # makes "exiting restores the state it interrupted, byte for byte" a property of the
    # code rather than a hope - there is no other field for it to have changed.

    def _get_attract_active(self) -> bool:
        return self._attract_active

    attract_active = Property(bool, _get_attract_active, notify=attractActiveChanged)

    @Property(str, constant=True)
    def attract_title(self) -> str:
        """The attract slide's title. Not a measurement and not a claim."""
        return ATTRACT_TITLE

    @Property(str, constant=True)
    def attract_recorded_label(self) -> str:
        """Spec section 9's recorded-surface label, shown on every attract slide."""
        return RECORDED_DEMO_LABEL

    @Property("QVariantList", constant=True)
    def attract_headlines(self) -> list:
        """The recorded headline claims, each with its date and its committed source.

        Constant on purpose: these are recorded measurements, not live state, so there is
        nothing here that could go stale between two evaluations. The overlay renders each
        claim **with its date**, which is the whole obligation of attract mode.
        """
        return [dict(headline) for headline in ATTRACT_HEADLINES]

    def set_attract(self, active: bool) -> None:
        """Enter (`True`) or leave (`False`) attract mode. **The only state it owns.**

        **M5: a playback is not yielded to the idle screen.** Attract and recorded mode are
        two different recorded surfaces with two different sets of dates; letting the idle
        timer replace a loaded record with the attract headlines would swap one record's
        numbers for another's without the presenter asking. Leaving attract is always
        allowed.
        """
        wanted = bool(active)
        if wanted and self._recorded_record is not None:
            return
        if wanted == self._attract_active:
            return
        self._attract_active = wanted
        self.attractActiveChanged.emit()

    def attract_due(self, *, now: float | None = None) -> bool:
        """Whether the idle interval has elapsed with no presenter input. Spec section 9.

        `now` is a parameter so the decision is testable without a clock; production
        passes nothing and the current `perf_counter` is used.
        """
        current = time.perf_counter() if now is None else float(now)
        return (current - self._last_activity) >= ATTRACT_IDLE_SECONDS

    def note_activity(self, *, now: float | None = None) -> None:
        """Record presenter input: reset the idle clock and leave attract mode.

        Spec section 9: "presenter (any input pauses rotation)". So any input both resets
        the idle countdown and dismisses the attract overlay.
        """
        self._last_activity = time.perf_counter() if now is None else float(now)
        self.set_attract(False)

    # -- recorded-demonstration mode (M5 Task 4) ---------------------------------------
    #
    # **The mirror of the mandatory live label.** A playback is not a measurement, so it
    # says so continuously and it cannot reach a live device control. Three properties
    # make that structural rather than conventional:
    #
    # - `recorded_mode` is `True` only while a `BenchmarkRecord` is loaded, and the screen
    #   binds the live label's `visible` to its inverse. The live label is not "hidden by
    #   the overlay"; it is `visible: false` in the scene graph, which is what the render
    #   reads.
    # - `recorded_metrics` is built **from the record's own samples**, so a playback cannot
    #   show a live number that was already on the screen, and every row carries the
    #   record's `started_at` rather than the wall clock.
    # - `deviceControl` refuses while `recorded_mode` is true, *and* the QML control's
    #   `enabled` is false, so a click has two independent reasons not to become an action.

    def _get_recorded_mode(self) -> bool:
        return self._recorded_record is not None

    recorded_mode = Property(bool, _get_recorded_mode, notify=recordedModeChanged)

    @Property(str, constant=True)
    def recorded_demo_label(self) -> str:
        """Spec section 9's wording, the same constant attract mode renders."""
        return RECORDED_DEMO_LABEL

    @Property(str, constant=True)
    def recorded_control_note(self) -> str:
        """Why the live device controls are disabled, shown only while recorded."""
        return RECORDED_CONTROL_NOTE

    def _get_recorded_metrics(self) -> list:
        return [dict(row) for row in self._recorded_metrics]

    recorded_metrics = Property("QVariantList", _get_recorded_metrics,
                               notify=recordedModeChanged)

    def _get_recorded_metrics_note(self) -> str:
        if not self._recorded_metrics_hidden:
            return ""
        return (f"{self._recorded_metrics_hidden} more measured metrics are in the "
                f"exported record; this screen shows {len(self._recorded_metrics)}.")

    recorded_metrics_note = Property(str, _get_recorded_metrics_note,
                                     notify=recordedModeChanged)

    def _get_recorded_source_text(self) -> str:
        return self._recorded_source

    recorded_source_text = Property(str, _get_recorded_source_text,
                                    notify=recordedModeChanged)

    def _get_recorded_refusal(self) -> str:
        """The stated reason the last `enter_recorded` refused, or `""`."""
        return self._recorded_refusal

    recorded_refusal = Property(str, _get_recorded_refusal,
                                notify=recordedModeChanged)

    def _get_live_stream_running(self) -> bool:
        return bool(self._live_stream_reason())

    live_stream_running = Property(bool, _get_live_stream_running,
                                   notify=recordedModeChanged)

    def _live_stream_reason(self) -> str:
        """Why a recording may not be entered now, or `""`.

        **Evidence, not a flag the caller sets.** A playback may not take over a screen
        that is showing live work: any analysed frame, any route that is running or
        pending, any decoded frame, or an attached live source is enough to refuse. The
        `_live_source_attached` field is set by `run_windowed` itself at the point the
        supervisor starts, so a caller that forgets to say "live" still cannot enter
        recorded mode after a frame has been analysed.
        """
        if self._live_source_attached:
            return "a live source is attached to this screen"
        if self._frames_analysed > 0:
            return f"{self._frames_analysed} frames have been analysed on this screen"
        running = sorted(
            stream for stream, route in self._stream_routes.items()
            if str(route.get("state") or "") in ("active", "pending"))
        if running:
            return f"camera(s) {', '.join(running)} have a live route"
        for stream, record in sorted(self._stream_frames.items()):
            if record.get("image") is not None:
                return f"camera {stream} has a decoded live frame"
        if self._frame_image is not None:
            return "the screen holds a live frame"
        return ""

    def attach_live_source(self) -> None:
        """Declare that `run_windowed` has started a supervisor behind this screen."""
        self._live_source_attached = True
        self.recordedModeChanged.emit()

    def detach_live_source(self) -> None:
        """Declare that the live source has stopped."""
        self._live_source_attached = False
        self.recordedModeChanged.emit()

    def set_recorded_source(self, name: str) -> None:
        """Name the file the record was read from, for the playback's own caption."""
        self._recorded_source_name = str(name)

    @staticmethod
    def _recorded_value_text(sample: MetricSample) -> str:
        """One sample's value, with its unit. Never a substitute for an absence."""
        if sample.value is None:
            reason = sample.unavailable_reason or "no reason was recorded"
            return f"unavailable - {reason}"
        value = float(sample.value)
        unit = str(sample.unit)
        if unit in ("percent", "%"):
            return f"{value:.1f}%"
        if unit == "fps":
            return f"{value:.1f} fps"
        if value.is_integer():
            return f"{int(value)} {unit}"
        return f"{value:g} {unit}"

    @classmethod
    def _recorded_rows(cls, record: BenchmarkRecord) -> tuple[tuple[dict[str, str], ...], int]:
        """The rows a playback may put on screen, and how many are not shown.

        **One row per metric, from the record's own samples, each stamped with the
        recording's own date.** The latest settled sample wins where one exists; a metric
        measured only during warm-up keeps its `cold` phase and its value, because showing
        it as steady state would be the NPU's 5.2 ms mistake. The row is deliberately plain
        data: the QML renders `name`, `text` and `timestamp` and invents nothing.
        """
        samples = record.settled_samples() or record.samples
        latest: dict[str, Any] = {}
        for sample in samples:
            current = latest.get(sample.metric)
            if current is None or sample.sample.sampled_at >= current.sample.sampled_at:
                latest[sample.metric] = sample
        rows = []
        for name in sorted(latest):
            sample = latest[name]
            rows.append({
                "name": name,
                "text": cls._recorded_value_text(sample.sample),
                "timestamp": (f"{record.started_at} "
                              f"(+{sample.sample.sampled_at:g} s into the recording)"),
                "phase": sample.phase.value,
            })
        shown = tuple(rows[:RECORDED_METRIC_LIMIT])
        return shown, max(0, len(rows) - len(shown))

    @Slot(object, result=bool)
    def enter_recorded(self, record: object) -> bool:
        """Load a `BenchmarkRecord` as a playback. **Refused while live work is running.**

        Returns `False` with `recorded_refusal` set rather than partially entering: a
        screen that showed half a playback and half a live run would be the exact confusion
        this mode exists to prevent.
        """
        try:
            if not isinstance(record, BenchmarkRecord):
                self._recorded_refusal = (
                    "recorded mode needs a BenchmarkRecord; got "
                    f"{type(record).__name__}")
                self.recordedModeChanged.emit()
                return False
            reason = self._live_stream_reason()
            if reason:
                self._recorded_refusal = (
                    "refusing to enter recorded mode while live work is running: "
                    f"{reason}")
                self.recordedModeChanged.emit()
                return False
            rows, hidden = self._recorded_rows(record)
            self._recorded_record = record
            self._recorded_metrics = rows
            self._recorded_metrics_hidden = hidden
            self._recorded_source = (
                f"recorded {record.started_at} on {record.machine_id}"
                + (f" - {self._recorded_source_name}" if self._recorded_source_name else ""))
            self._recorded_refusal = ""
            if self._attract_active:
                # Attract shows dated headlines; a playback shows a different record. Two
                # recorded surfaces at once would be two sets of dates over one screen.
                self._attract_active = False
                self.attractActiveChanged.emit()
            self.recordedModeChanged.emit()
            return True
        except Exception as exc:  # noqa: BLE001 - a slot failure is not a crash
            self._report_slot_failure("enter_recorded", exc)
            self._recorded_refusal = f"recorded mode failed: {exc}"
            self.recordedModeChanged.emit()
            return False

    @Slot(result=bool)
    def leave_recorded(self) -> bool:
        """Leave the playback. The live fields were never touched, so the screen returns."""
        if self._recorded_record is None:
            return False
        self._recorded_record = None
        self._recorded_metrics = ()
        self._recorded_metrics_hidden = 0
        self._recorded_source = ""
        self.recordedModeChanged.emit()
        return True

    def _get_device_control(self) -> str:
        return self._device_control

    def _set_device_control(self, value: str) -> None:
        """QML writes a device id here to ask for a move. **Refused while recorded.**

        A writable property rather than a Slot call, for two reasons: a QML binding over a
        Slot cannot re-evaluate (the standing guard), and an assignment is not a binding.
        The read-back is the **accepted** request, so a refused assignment leaves the
        screen's own record of "what I asked for" unchanged.
        """
        self.request_device(str(value))

    device_control = Property(str, _get_device_control, _set_device_control,
                              notify=deviceControlChanged)

    def request_device(self, device_id: str) -> bool:
        """Ask to move the bound camera to `device_id`. Returns whether it was accepted."""
        wanted = str(device_id).strip()
        if wanted not in ("CPU", "GPU", "NPU"):
            return False
        if self._recorded_record is not None:
            # **The second line of defence.** The QML control is disabled, so a click never
            # reaches here; if some other path writes the property, the playback still
            # does not become a live action.
            return False
        self._device_control = wanted
        self.deviceControlChanged.emit()
        self.deviceRequested.emit(str(self._route_stream), wanted)
        return True



    @Slot(str, result=str)
    def display_text(self, name: str) -> str:
        """The rendering of one optional metric. **The only path a number takes to QML.**

        `"unavailable"` when the value is `None`; a formatted number otherwise - and a
        genuine `0.0` renders as a number, because a measured zero is a reading and
        refusing it would be the same error pointing the other way.

        **An empty name returns `"unavailable"`** rather than raising, because that is the
        honest answer to "this card names no metric": a card whose `metricName` has not
        been assigned yet has nothing to show, and that is not an error. Measured: without
        it, QML evaluated the binding once at construction with `metricName == ""` and
        every card logged a `KeyError` on its first frame. A **non-empty** name outside
        `OPTIONAL_METRICS` still raises, because that is a typo and a typo must be loud.
        """
        key = str(name)
        if not key:
            return UNAVAILABLE_TEXT
        if key not in self._metrics:
            raise KeyError(
                f"{key!r} is not an optional metric this bridge formats; the set is "
                f"{list(OPTIONAL_METRICS)}. A typo here would render as 'unavailable' "
                f"on a card, which is indistinguishable from a measurement nobody took."
            )
        value = self._metrics[key]
        if value is None:
            return UNAVAILABLE_TEXT
        return _FORMATTERS[key](float(value))

    @Slot(str, result=str)
    def window_text(self, name: str) -> str:
        """The window a measurement was averaged over, **or why there is none**.

        **The `validity` branch comes first and returns before `window_seconds` is read.**
        That ordering is the requirement, not a style: `window_seconds == 0.0` is legal
        exactly when `validity is UNAVAILABLE`, so a formatter that rendered the window
        first would be one line from dividing by it. There is no division in this module
        at all, and a metric whose window has not been recorded reads as
        `unavailable` rather than as `0.00 s` - which would be a duration nobody measured.
        """
        key = str(name)
        if not key:
            # Same reasoning as `display_text`: a card with no metric named has nothing
            # to show, and that is not a failure.
            return UNAVAILABLE_TEXT
        if key not in self._metrics:
            raise KeyError(f"{key!r} is not an optional metric this bridge formats")
        sample = self._samples.get(key)
        if sample is None:
            return UNAVAILABLE_TEXT
        if sample.validity is Validity.UNAVAILABLE:
            # First branch, and it returns. `window_seconds` is not read on this path,
            # and its value is meaningless here by construction.
            return UNAVAILABLE_TEXT
        window = float(sample.window_seconds)
        if window <= 0.0:
            # A VALID reading with a non-positive window cannot be built - the contract
            # refuses it - so reaching here means the record is not what it claims.
            return UNAVAILABLE_TEXT
        if window >= 1.0:
            return f"over {window:.2f} s"
        return f"over {window * 1000.0:.0f} ms"

    @Slot(int)
    def set_frames_analysed(self, count: int) -> None:
        """Record how many frames the supervisor has analysed. UI thread.

        Distinct from counting applied results in `apply_frame`: that number would be 0
        whenever a result arrived but the queued slot had not run yet, and the event-count
        row must not flicker between "unavailable" and `0` on that basis. The supervisor's
        own counter is the one that says whether anything could have been counted.
        """
        value = int(count)
        if value == self._frames_analysed:
            return
        self._frames_analysed = value
        self.framesAnalysedChanged.emit()

    def _get_staleness(self) -> str:
        """`"fresh"`, `"stale"` or `""`, as a **notified property**.

        QML cannot re-evaluate a binding that calls `staleness_text()` - see
        `_get_metric_text`. The control bar binds to this instead. It is refreshed on the
        same events as the metrics, and `staleness_text()` stays as the callable form so
        there is still one implementation.
        """
        return self.staleness_text()

    staleness = Property(str, _get_staleness, notify=stalenessChanged)

    @Slot(result=str)
    def staleness_text(self) -> str:
        """One word about the freshness of the newest sample set, or `""` if there is none."""
        newest = max((s.sampled_at for s in self._samples.values()), default=None)
        if newest is None:
            return ""
        age = time.perf_counter() - newest
        if age > 2.0:
            return "stale"
        return "fresh"

    # -- samples -------------------------------------------------------------------

    def set_samples(self, samples: Sequence[MetricSample]) -> None:
        """Take one whole sample set from `MetricCollector.sample()`.

        **The collector's accessors are never called here or anywhere in this module.**
        See the module note; the source is read by `tests/test_ui_contract.py` to keep it
        that way.

        A sample whose `validity` is `UNAVAILABLE` sets the corresponding property to
        `None`, whatever its `value` field says - which is always `None`, because
        `MetricSample` refuses the other combination at construction. A GPU/NPU sample
        carrying a measured number sets that property and makes the card available; the
        card renders the number, not a claim about the hardware.
        """
        for sample in samples or ():
            name = sample.name
            if name not in self._metrics:
                continue
            effective = sample.effective_validity(time.perf_counter())
            value = None if effective is Validity.UNAVAILABLE else float(sample.value)
            self._samples[name] = sample
            if self._metrics[name] != value:
                self._metrics[name] = value
                self._emit_for(name)
        self.metricUpdated.emit(tuple(samples))

    def set_metrics(self, **values: float | None) -> None:
        """Set optional metrics directly, as floats or `None`.

        **This is a measured number or `None`. There is no simulated-provider path here,
        and none may be added** - AGENTS.md rule 14 says release mode refuses to start
        with a simulated provider registered, so a setter that fabricated values would be
        a way around a gate that is enforced elsewhere. `tests/test_ui_contract.py`
        asserts this module's source contains no `random`, no `math.sin` and no literal
        fallback.
        """
        for name, value in values.items():
            if name not in self._metrics:
                raise KeyError(
                    f"{name!r} is not an optional metric this bridge formats; the set is "
                    f"{list(OPTIONAL_METRICS)}"
                )
            number = None if value is None else float(value)
            if self._metrics[name] == number:
                continue
            self._metrics[name] = number
            self._emit_for(name)

    def _emit_for(self, name: str) -> None:
        {"ai_fps": self.aiFpsChanged,
         "frame_age_p50_ms": self.frameAgeP50Changed,
         "frame_age_p95_ms": self.frameAgeP95Changed,
         "system_cpu_percent": self.systemCpuPercentChanged,
         "app_cpu_percent": self.appCpuPercentChanged,
         "gpu_utilisation_percent": self.gpuAvailableChanged,
         "npu_utilisation_percent": self.npuAvailableChanged}[name].emit()
        # The formatted map is what QML binds to, and the staleness word is a property
        # rather than a method call for the same reason. See `_get_metric_text`.
        self.metricsTextChanged.emit()
        self.stalenessChanged.emit()
        self.windowNotesChanged.emit()
        self.unavailableNoticeChanged.emit()

    # -- frames, overlay and events -------------------------------------------------

    def set_frame(self, frame: Any) -> None:
        """Record that `frame` is the one on screen, and publish it.

        `frame` is a `FramePacket`. Its **identity tuple** is stored, not its index: two
        frames from different loops of the clip share a `frame_id`, so binding an overlay
        to `frame_id` paints one loop's boxes onto another loop's pixels the first time
        the clip rewinds. `accept_result` compares identities and drops an obsolete
        result rather than drawing it.
        """
        identity = tuple(frame.identity)
        self._displayed_identity = identity
        self._latest_identity = identity
        if not self._frame_ready:
            self._frame_ready = True
            self.frameReadyChanged.emit()
        self.frameArrived.emit(identity)

    def set_frame_image(self, frame: Any) -> None:
        """Hand the newest decoded frame's pixels to the image provider.

        `frame` is a `FramePacket`; its `image` is `uint8` BGR as `cv2` produces it. The
        swap to RGB happens here, and the array is **copied**: a `QImage` over a numpy
        buffer would alias the clip's next decode and tear the frame being drawn, which
        is the same class of defect as a box drawn against the wrong frame.

        Bumping `frame_revision` is what makes QML ask for the image again. An `Image`
        whose `source` is a constant `image://` URL is fetched once, so without a
        changing query string the hero would show frame 0 for the whole session.
        """
        self.set_frame(frame)
        array = frame.image
        # **M2: through `_store_image`, so there is ONE implementation of "BGR array to
        # QImage, keyed by stream, revision counted".** This method used to do the swap and
        # the store itself, which meant a second copy of the whole thing to keep in step -
        # and the per-stream store was the copy that got left behind, so the single-stream
        # entry point published a hero the tiles could not see.
        self._store_image(array, str(getattr(frame, "stream_id", None)
                                       or self._route_stream))
        image = self._frame_image
        # **Not `constant=True`.** The provider reads the attribute from Python so it
        # worked regardless, which is why this was harmless - but a QML binding on
        # `frame_image` would have captured the first frame for the whole session, and
        # `constant` on a property that changes 15 times a second is the kind of thing
        # that breaks the day someone binds to it.
        self.frameImageChanged.emit()

    def _get_frame_image(self):
        return self._frame_image

    frame_image = Property(QImage, _get_frame_image, notify=frameImageChanged)

    def _get_frame_revision(self) -> int:
        return self._frame_revision

    frame_revision = Property(int, _get_frame_revision, notify=frameRevisionChanged)

    def accept_result(self, result: Any) -> bool:
        """Whether `result` belongs to the frame currently on screen.

        `False` for a result whose frame identity is not the displayed one. This is
        review focus #1 made a named, tested call: a result that arrives after its frame
        has been replaced must be discarded, never drawn. It is **not** a check on how old
        the result is - age is a different question and has its own metrics.
        """
        identity = tuple(result.frame.identity)
        self._latest_identity = identity
        if self._displayed_identity is None or identity != self._displayed_identity:
            return False
        self.resultArrived.emit(result)
        return True

    def set_overlay(self, tracks: Iterable[Any], frame: Any = None) -> None:
        """Build the overlay boxes, **filtering on `is_current`**.

        The filter is the obligation this method exists for. `Tracker.update()` returns
        coasted tracks as well as current ones, deliberately, because the event machine
        needs to know a track still exists so it can decide whether a pending presence
        survives an occlusion. A coasted track's `box` is the last one **seen**, and
        drawing it puts a plausible, correctly-placed rectangle over an object that has
        since left - the mirror image of the frame-identity defect, and just as
        invisible.

        There is no `include_coasted` parameter and no branch that can skip the filter. A
        track with `misses > 0` produces no box at all.

        Boxes are normalised to `[0, 1]` against the frame's own extents, read from the
        packet and never from a config, so a box is in the right place on the picture it
        was measured on.
        """
        if frame is not None:
            width = float(frame.width)
            height = float(frame.height)
        else:
            width = height = 0.0
        boxes: list[dict[str, Any]] = []
        coasted = 0
        unplaceable = 0
        for track in tracks:
            if not getattr(track, "is_current", False):
                coasted += 1
                continue
            if width <= 0.0 or height <= 0.0:
                unplaceable += 1
                continue
            x1, y1, x2, y2 = (float(v) for v in track.box)
            boxes.append({
                "track_id": int(track.track_id),
                "class_id": int(track.class_id),
                "x": x1 / width,
                "y": y1 / height,
                "w": max(0.0, (x2 - x1) / width),
                "h": max(0.0, (y2 - y1) / height),
                "confirmed": bool(track.confirmed),
                "score": float(getattr(track, "score", 0.0) or 0.0),
            })
        if unplaceable and not boxes:
            # **No frame is a different fact from a coasted track.** Dividing by the
            # frame's extents is what makes a box land in the right place, and with no
            # frame there is no place. Saying "holding coasted tracks" here would blame
            # the tracker for the caller's missing argument - and this branch was
            # reachable precisely because `set_overlay` accepted `frame=None`.
            self._set_status("cannot place overlay boxes: no frame has been decoded yet")
        elif coasted and not boxes:
            # Coasted-only is a real state, and it is worth a status word: it means the
            # tracker is holding a track whose object has left. Rendering it as "no
            # detections" would be true and useless.
            self._set_status("holding coasted tracks: none of them are current")
        self._overlay_boxes = tuple(boxes)
        self.overlayBoxesChanged.emit()

    def set_zones(self, zones: Iterable[Any], frame: Any) -> None:
        """Publish zone rectangles in normalised coordinates, for the QML overlay.

        The ROI is already normalised - `Zone` validates all four values into `[0, 1]` -
        so it is passed through rather than rescaled against the frame. The frame is taken
        only to prove that a zone is being drawn over *a* picture: a zone published with
        no frame behind it is a rectangle on nothing.
        """
        width = float(frame.width)
        height = float(frame.height)
        self._zones = tuple(
            {"name": str(z.name),
             "x": float(z.roi[0]), "y": float(z.roi[1]),
             "w": float(z.roi[2]) - float(z.roi[0]),
             "h": float(z.roi[3]) - float(z.roi[1]),
             "pixel_w": width, "pixel_h": height}
            for z in zones
        )
        self.zonesChanged.emit()

    def _get_zones(self) -> list:
        return [dict(zone) for zone in self._zones]

    zones = Property("QVariantList", _get_zones, notify=zonesChanged)

    def record_event(self, event: Any) -> None:
        """Count one business event and publish it.

        The count is by `kind`, and the `dedup_key` is what makes it one: the event
        machine emits a key per track per loop, so a replayed loop produces a **new** key
        and a new count, which is what a booth wants, while re-seeing the same movement
        within one loop produces the same key and is not counted twice.

        ## Accepts an object **or** a mapping, and refuses to guess

        The windowed path's `apply_frame` receives events inside a `Signal(object)`
        payload, which crosses a thread boundary and therefore travels as a plain dict;
        the headless path and the tests hand it a `BusinessEvent`. Reading only by
        attribute would have counted every event the booth emitted under the kind
        `"unknown"` - measured: `{'unknown': 1}` where the machine had emitted
        `zone_occupancy` - which is the worst possible failure for an event counter,
        because it is right about the total and wrong about what happened.

        A payload that carries **neither** a `kind` attribute nor a `kind` key raises
        rather than counting as `unknown`. An unidentifiable event is a defect in the
        producer, and silently filing it under "unknown" would make the footer agree with
        a machine that said something else.
        """
        if isinstance(event, Mapping):
            fields = dict(event)
        else:
            if not hasattr(event, "kind"):
                raise TypeError(
                    f"record_event takes a BusinessEvent or a mapping carrying a 'kind'; "
                    f"got {type(event).__name__} with no kind. Counting it as 'unknown' "
                    f"would put a count on the screen that disagrees with the event "
                    f"machine that produced it.")
            fields = {
                "kind": getattr(event, "kind", ""),
                "class_name": getattr(event, "class_name", ""),
                "track_id": getattr(event, "track_id", 0),
                "dedup_key": getattr(event, "dedup_key", ""),
                "at_frame_id": getattr(event, "at_frame_id", 0),
            }
        kind = str(fields.get("kind", "") or "")
        if not kind:
            raise ValueError(
                "record_event was given an event with an empty kind. An event with no "
                "kind cannot be counted by kind, and 'unknown' would be a fabricated "
                "category.")
        self._event_counts[kind] = self._event_counts.get(kind, 0) + 1
        self.eventCountsChanged.emit()
        self.eventEmitted.emit({
            "kind": kind,
            "class_name": str(fields.get("class_name", "")),
            "track_id": int(fields.get("track_id", 0) or 0),
            "dedup_key": str(fields.get("dedup_key", "")),
            "at_frame_id": int(fields.get("at_frame_id", 0) or 0),
        })

    def reset_events(self) -> None:
        """Clear the event counts. Not the metrics, not the route, not the drivers."""
        self._event_counts = {}
        self.eventCountsChanged.emit()

    def _set_status(self, text: str) -> None:
        if text != self._status_message:
            self._status_message = text
            self.statusMessageChanged.emit()

    def set_status(self, text: str) -> None:
        """Set the status line."""
        self._set_status(str(text))

    def set_error(self, text: str | None) -> None:
        """Set the error line. `None` or `""` clears it.

        **Never cleared implicitly.** A worker failure produces a visible state here and
        it stays until something clears it deliberately, because an error that disappears
        on the next good frame is an error nobody saw.
        """
        value = "" if text is None else str(text)
        if value != self._error_message:
            self._error_message = value
            self.errorMessageChanged.emit()

    # ------------------------------------------------------------------ derived

    @Property(str, notify=deviceDisplayChanged)
    def device_display(self) -> str:
        """The one line the device badge shows.

        `"pending - no placement confirmed"` rather than the requested device. This is
        the string that must never read like a working device when nothing has run.

        ## Why this is `notify=deviceDisplayChanged` and was not `constant=True`

        **This property was `constant=True`, and that is a silent lie rather than a Qt
        detail.** Qt reads a `constant` property exactly once, when the binding that
        refers to it is first created - which is during `engine.load`. The device badge is
        built then, before any inference has happened, so the binding captured
        `"pending - no device selected"` and never re-read the property. When the worker
        then confirmed CPU, this method's Python value changed to `"CPU"` and the **screen
        did not**, leaving the booth displaying a pending state for a route that was
        running on the very device it said was unconfirmed.

        The offscreen render test in `tests/test_ui_contract.py` passed throughout,
        because it called `set_route(...)` *before* `engine.load(...)` - which is exactly
        when a `constant` property still updates. That test proved the badge can render a
        confirmed device; it could not prove the badge *updates*, because nothing ever set
        the route after the screen existed. the live-path test class sets a route
        after the engine has loaded and reads the **rendered** `Text` item, which is the
        only read that can tell a bridge that updated from a screen that did not.
        """
        if self._device_confirmed:
            if self._route_fell_back and self._device_requested:
                return (f"{self._device_confirmed} (fell back from "
                        f"{self._device_requested})")
            return str(self._device_confirmed)
        if self._state_failed():
            return "failed - no device is running"
        if self._device_requested:
            return f"pending - {self._device_requested} not yet confirmed"
        return "pending - no device selected"

    def _state_failed(self) -> bool:
        return self._route_state == "failed"