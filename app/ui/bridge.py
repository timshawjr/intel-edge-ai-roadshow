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

## GPU and NPU utilisation are `False`, and that is a measurement

`gpu_available` and `npu_available` are **False**, with the measured reason in
`gpu_reason` / `npu_reason`. Measured on this machine: the native PDH value read
(`PdhGetFormattedCounterValue`) terminates the process with `0xC0000005` for
`\\GPU Engine(*)\\Utilization Percentage` **and** for three controls with nothing to do
with the GPU, with the handle proven valid by `PdhGetCounterTimeBase`; and a machine-wide
peak is not attributable to this process (our pid appeared in 0 of 469 samples). So the
cards say **"Utilization unavailable"** and show inference activity - completions,
latency, confirmed placement - instead. `ComputeCard` is given the string
`UTILISATION_UNAVAILABLE_TEXT` and never a number. There is no code path here that can
produce a utilisation figure, and that is deliberate.

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
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from PySide6.QtCore import Property, QObject, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter

from app.contracts.metric_sample import MetricSample, Validity
from app.inference.devices import is_device, normalise_execution_devices

__all__ = [
    "EXPLANATIONS",
    "FrameImageProvider",
    "RECORDED_FOOTAGE_LABEL",
    "SCOPE_NOTE_LABEL",
    "ThemeTokens",
    "UNAVAILABLE_TEXT",
    "UTILISATION_UNAVAILABLE_TEXT",
    "UiBridge",
]

#: Spec §4: displayed continuously. **Not optional and not per-screen.**
RECORDED_FOOTAGE_LABEL = "Recorded footage • live inference"

#: What an absent measurement renders as. A word, never a number (AGENTS.md rule 15).
UNAVAILABLE_TEXT = "unavailable"

#: What the GPU and NPU cards say instead of a utilisation figure. See the module note.
UTILISATION_UNAVAILABLE_TEXT = "Utilization unavailable"

#: The short reason, for the card. The full measured text is 922 characters and is pinned
#: to the explanation panel where there is room for it; a card that carries the whole
#: measured reason elides to three lines and says nothing.
#:
#: **Both halves matter.** "The counters fault this process" says why there is no reading
#: from the obvious source, and "a machine-wide peak is not attributable to this process"
#: says why the obvious workaround - quote the machine's peak - would be a lie. Dropping
#: either half leaves a presenter with an obvious objection and no answer.
UTILISATION_SHORT_REASON = (
    "The platform's own performance counters fault this process when read, and a "
    "machine-wide peak is not attributable to this process. Inference activity is shown "
    "instead: frames analysed, latency, and the confirmed device."
)

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
        "NOT measurable here - the platform's own performance counters fault this process "
        "when read - so no utilisation number is shown. What is shown instead is "
        "inference activity: how many frames were analysed, how long each inference "
        "took, and which device the runtime confirmed the work ran on. Per-device "
        "single-frame latency on this machine measures CPU 20.99-21.16 ms, GPU "
        "3.30-3.47 ms, NPU 5.92-5.99 ms - single frame, 640x640, batch 1, 30 "
        "inferences, integrated Arc 140V sharing memory with the NPU. That is not a "
        "throughput claim."
    ),
    "utilisation": (
        "GPU and NPU utilisation is unavailable, and this is the measured reason: "
        "the platform's own performance counters terminate this process with an access "
        "violation when their values are read, for the GPU counter and for controls "
        "with nothing to do with the GPU alike; and a machine-wide peak is not "
        "attributable to this process. So the cards read 'Utilization unavailable' and "
        "show inference activity instead. Nothing here is estimated and nothing is "
        "filled in."
    ),
    "metrics": (
        "Metrics: each figure names its provider, its scope and the window it was "
        "averaged over. System CPU is the whole machine. Application CPU is this "
        "process plus its worker, normalised to the machine. Frame age is how long a "
        "frame waited from capture to the moment its result was shown. Anything not "
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
        """Apply a route plan payload from the supervisor's thread."""
        data = dict(payload)  # type: ignore[arg-type]
        self.set_route(requested=data.get("requested"),
                       confirmed=tuple(data.get("confirmed") or ()),
                       state=str(data.get("state") or "pending"))
        errors = tuple(str(e) for e in (data.get("errors") or ()))
        if errors:
            self.set_error(errors[-1])
        self.routeChanged.emit({"state": self._route_state,
                                "confirmed": self._device_confirmed})

    @Slot(str)
    def apply_error(self, text: str) -> None:
        """Show one stated reason. Additive: a later error does not erase an earlier one."""
        existing = self._error_message
        message = f"{existing}\n{text}" if existing else str(text)
        self.set_error(message)

    @Slot(object)
    def apply_frame(self, payload: object) -> None:
        """Record a frame and the boxes that belong to it, from the supervisor's thread."""
        data = dict(payload)  # type: ignore[arg-type]
        boxes = list(data.get("boxes") or ())
        self._overlay_boxes = tuple(boxes)
        self.overlayBoxesChanged.emit()
        for event in data.get("events") or ():
            self.eventEmitted.emit(dict(event))
        if not self._frame_ready:
            self._frame_ready = True
            self.frameReadyChanged.emit()

    @Slot(object)
    def apply_samples(self, samples: object) -> None:
        """Take one whole telemetry sample set from the collector's reader."""
        self.set_samples(tuple(samples))  # type: ignore[arg-type]

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

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._frame_ready = False
        self._overlay_boxes: tuple[dict[str, Any], ...] = ()
        self._device_confirmed: str | None = None
        self._device_requested: str | None = None
        self._route_state = "pending"
        self._route_fell_back = False
        self._metrics: dict[str, float | None] = {name: None for name in OPTIONAL_METRICS}
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

        # Utilisation is **unavailable**, and it is unavailable for a measured reason. The
        # reasons are the strings `app/telemetry/pdh.py` publishes as
        # VALUE_READS_REFUSED_REASON, quoted by reference rather than restated here so
        # this module cannot drift from the measurement that produced it.
        from app.telemetry.pdh import VALUE_READS_REFUSED_REASON

        self._utilisation_reason = VALUE_READS_REFUSED_REASON
        self._gpu_available = False
        self._npu_available = False

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
    # them and get an honest `None` rather than a binding error. **They are always
    # `None`** and `set_metrics` can set them - a machine where the counters *are*
    # readable would fill them, and nothing in this module decides whether they can be.
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
        return self._gpu_available

    gpu_available = Property(bool, _get_gpu_available, notify=gpuAvailableChanged)

    def _get_npu_available(self) -> bool:
        return self._npu_available

    npu_available = Property(bool, _get_npu_available, notify=npuAvailableChanged)

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
        """What the GPU and NPU cards say. They never show a utilisation number."""
        return UTILISATION_UNAVAILABLE_TEXT

    @Property(str, constant=True)
    def scope_note(self) -> str:
        """The reduction, verbatim from `app/engine/retail_config.py`."""
        return SCOPE_NOTE_LABEL

    @Property(str, constant=True)
    def gpu_reason(self) -> str:
        """The short, card-sized reason GPU utilisation is unavailable."""
        return UTILISATION_SHORT_REASON

    @Property(str, constant=True)
    def npu_reason(self) -> str:
        """The short reason for the NPU. Same provider, same fault, same absence."""
        return UTILISATION_SHORT_REASON

    @Property(str, constant=True)
    def utilisation_measured_reason(self) -> str:
        """The full measured text, quoted from `app.telemetry.pdh` by reference.

        This is the 922-character refusal the PDH module publishes verbatim, and it is
        reachable from the pinned explanation panel. **Quoted rather than restated**: a
        copy in this module would be a second version of a measured string that could
        disagree with the one the measurement produced.
        """
        return self._utilisation_reason

    @Property("QVariantMap", constant=True)
    def explanations(self) -> dict:
        """The six required explanations, reachable from hover and from a pinned panel."""
        return dict(EXPLANATIONS)

    @Property(bool, constant=True)
    def device_fell_back(self) -> bool:
        """Whether the confirmed device differs from the requested one.

        `False` when either is absent - an unproven route has not fallen back, it has
        **not been shown**, and reporting `False` there would read as "running where you
        asked".
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
        if fell_back is None:
            fell_back = bool(
                requested and devices and not is_device(devices, requested))
        if bool(fell_back) != self._route_fell_back:
            self._route_fell_back = bool(fell_back)
            self.deviceConfirmedChanged.emit()
        if state != self._route_state:
            self._route_state = str(state)
            self.routeStateChanged.emit()

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
        `MetricSample` refuses the other combination at construction.
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
        rgb = array[:, :, ::-1]
        height, width = int(array.shape[0]), int(array.shape[1])
        image = QImage(rgb.tobytes(), width, height, width * 3, QImage.Format_RGB888)
        self._frame_image = image.copy()
        self._frame_revision += 1
        self.frameRevisionChanged.emit()

    def _get_frame_image(self):
        return self._frame_image

    frame_image = Property(QImage, _get_frame_image, constant=True)

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
        skipped = 0
        for track in tracks:
            if not getattr(track, "is_current", False):
                skipped += 1
                continue
            if width <= 0.0 or height <= 0.0:
                skipped += 1
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
        if skipped and not boxes:
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
        """
        kind = str(getattr(event, "kind", "unknown"))
        self._event_counts[kind] = self._event_counts.get(kind, 0) + 1
        self.eventCountsChanged.emit()
        self.eventEmitted.emit({
            "kind": kind,
            "class_name": str(getattr(event, "class_name", "")),
            "track_id": int(getattr(event, "track_id", 0)),
            "dedup_key": str(getattr(event, "dedup_key", "")),
            "at_frame_id": int(getattr(event, "at_frame_id", 0)),
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

    @Property(str, constant=True)
    def device_display(self) -> str:
        """The one line the device badge shows.

        `"pending - no placement confirmed"` rather than the requested device. This is
        the string that must never read like a working device when nothing has run.
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


UiBridge._samples = {}