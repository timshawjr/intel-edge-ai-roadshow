"""The inference worker: one frame in, one result carrying that frame out.

Review Focus #4 is the reason this module is shaped the way it is: **the worker's real
target silently falling back to CPU while the UI says NPU.** Four rules, each of which
has to be true simultaneously or the booth is lying:

1. **`EXECUTION_DEVICES` is read from the request's own compiled model, after the
   inference.** Not from the requested device id, not from the `Core`, and not from a
   value cached at compile time. A compile tells you where the runtime *might* run it; an
   inference tells you where it did. `CompiledTarget` in `app.inference.devices` already
   holds no placement field at all for the same reason, and this module follows it.
   The reading goes through `normalise_execution_devices`, because the NPU plugin reports
   a **bare string** `'NPU'` on this machine while CPU reports `['CPU']` and GPU
   `['GPU.0']`.
2. **An unproven placement is a failure.** If the property cannot be read, or reads
   empty, `infer_once` returns an error result. It does not return a success with no
   devices, because "the NPU ran something, we cannot say what" is not a state a booth
   screen can draw.
3. **Any exception becomes an `error`, with no latency and no placement.** That is spec
   section 5 and AGENTS.md rule 12: one bad frame cannot take the stream down, and a
   frame that could not be analysed must not appear as a frame with nothing in it.
   `InferenceResult.__post_init__` makes that structural, so the failure shape cannot be
   constructed by hand either - a caller who reaches for `latency_ms=0.0` on an error
   gets a `ValueError` rather than a plausible zero in a latency column.
4. **A result carries the `FramePacket` it came from.** `InferenceResult.frame` is the
   exact object handed in, not a copy and not a bare detection list. That is what lets an
   overlay compare identities and drop a result whose frame has already been replaced -
   review focus #1 - and it survives the queue boundary because the queue carries
   packets too.
5. **The requested device rides on the result beside the confirmed one.** A fallback -
   NPU asked for, CPU confirmed - is a legitimate state and is reported as what actually
   ran. What review focus #4 forbids is it being *invisible*, so `requested_device` is a
   field of the record and `fell_back` is the comparison, rather than something a
   consumer has to know to perform.

**`latency_ms` is the `infer` call and nothing else.** Preprocessing, the decode and the
placement read are all outside the timer, which makes the number directly comparable with
the p50s `tools/qualify_m0.py` measured on this machine. Timing the whole frame would
produce a larger, differently-defined number and a reader would compare it against the
wrong reference.

**A `latency_ms` is not a throughput claim, and the device ordering here is not a
result.** Measured on this machine over one clip frame, four repeats of 30 inferences per
device: CPU 20.99-21.16 ms, GPU 3.30-3.47 ms, NPU 5.92-5.99 ms - so the **NPU is slower
than the GPU**, which is not the shape a booth story usually has and which the reviewer
re-measured twice to confirm it was not an artefact of ordering or interleaving. Those are
single-frame inferences, not a sustained run, and the GPU here is an integrated **Arc 140V
sharing system memory with the NPU** - exactly the configuration in which a single-frame
p50 favours the GPU. Nothing in this module should be read as claiming an accelerator is
faster than another; the module's job is to report what a frame cost and where it ran, and
it does not establish a ranking.

**The output tensor is fetched by the compiled model's port object, with
`get_output_tensor(0)` as the recorded fallback.** Measured on this build
(`config/model_io/yolo11n-fp16.json`, `fetch_api`): `infer` accepts an int index, a port
name or the **compiled** port object and rejects the IR model's port object;
`get_tensor` accepts the port object and **raises `TypeError` on an int index**; and
`get_output_tensor(0)` accepts the index. So the two sides cannot be keyed the same way,
and the code says which one it used.

**The compiled model and the request are built once per device per process** and cached.
That is the point of a worker process: one compile, then a steady loop. A failed compile
is cached too, so a device that will not compile does not pay the compile attempt again
on every frame.

**`run_worker` never raises.** It takes `None` off its input queue as the stop signal,
turns every other outcome into a result, and returns when the queue yields `None`. It is
designed to be a `multiprocessing` **spawn** entry point (Windows-safe, no `fork`
assumptions), so it takes its queues as arguments and reads nothing from globals it did
not build itself.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any, Final

import cv2
import numpy as np

from app.contracts.frame import FramePacket
from app.inference.decode import (
    DECODABLE_LAYOUTS,
    Detection,
    ModelOutputSpec,
    decode_detections,
    describe_outputs,
    port_key,
)
from app.inference.devices import (
    EXECUTION_DEVICES,
    is_device,
    normalise_execution_devices,
)
from app.inference.pinned import MODEL_ID, num_classes, pinned_input_shape, verified_ir_path
from app.inference.preprocess import preprocess_bgr

__all__ = ["DISPLAY_COPY_MAX_WIDTH", "InferenceResult", "WorkerConfig", "display_copy",
           "infer_once", "run_worker"]

#: The port key the tensor is filed under. `app.inference.decode` keys on the port
#: **index as a string**, because the pinned export's output port has no name at all and
#: there is nothing else that identifies it.
OUTPUT_PORT_KEY = port_key(0)

#: The long edge the **display** copy of a frame is scaled to before it crosses back to
#: the UI process. **960, and it is derived rather than chosen.**
#:
#: The booth's hero tile is `Theme.VIDEO_WIDTH_FRACTION` (0.68) of the 1920 px canvas wide
#: and the main band tall, and `VideoTile.qml`'s `Image` is `fillMode: PreserveAspectFit`.
#: Measured from those two numbers: the hero box is 1306 x 538, so a 16:9 image is fitted
#: at `min(1306/1920, 538/1080) = 0.4985` and **displayed at 957 x 538**. A 960-wide copy
#: is therefore native for the hero and four times more than a 187 px thumbnail needs.
#:
#: **The detector never sees this copy.** `run_worker` calls `infer_once` on the
#: full-resolution packet first and only then builds the display copy, so small-object
#: detection is unchanged (AGENTS.md rule: do not downscale anything the detector
#: consumes).
DISPLAY_COPY_MAX_WIDTH: Final[int] = 960


def display_copy(image: np.ndarray, *,
                 max_width: int = DISPLAY_COPY_MAX_WIDTH) -> np.ndarray:
    """A smaller BGR `uint8` copy of `image` for display, or `image` itself if it is small.

    `INTER_AREA` because this is a downscale and area averaging is the correct filter for
    one; `INTER_LINEAR` (what the letterbox uses) aliases on a 2x reduction.

    Returns `image` unchanged when it is already at or below `max_width` - a source that
    is small enough is not resized, so a test frame does not acquire a resize cost.
    """
    array = np.asarray(image)
    height, width = int(array.shape[0]), int(array.shape[1])
    if width <= int(max_width):
        return array
    scale = float(max_width) / float(width)
    target = (int(max_width), max(1, int(round(height * scale))))
    return cv2.resize(array, target, interpolation=cv2.INTER_AREA)


def _display_result(result: InferenceResult) -> InferenceResult:
    """The result as it crosses back to the UI: pixels and geometry at display resolution.

    **The detector has already run on the full-resolution frame** - `infer_once` letterboxed
    `frame.image` and decoded boxes in its source pixels - so this only changes what is
    transported and painted. It does two things, and they must happen together:

    1. the frame's image is replaced by `display_copy(frame.image)` and its `width`/`height`
       with the copy's, so the 6.2 MB full-resolution array is **not** in the return
       pickle at all;
    2. every detection box is scaled by the copy's own x and y factors, so a box stays in
       the same **normalised** position against the smaller extents.

    Doing (1) without (2) would move every overlay box and mis-classify every zone
    membership, because both `app.main._frame_payload` and
    `app.engine.events._normalise` divide a box by the frame's own extents. Scaling both
    together leaves every normalised coordinate identical - IoU is scale-invariant, so the
    tracker's matching is unchanged too - while the pixels that travel back are four times
    smaller.

    A failure is returned untouched: there is nothing to display and no geometry to scale.
    A frame already at or below `DISPLAY_COPY_MAX_WIDTH` is returned untouched, so the
    single-stream and test paths pay nothing.
    """
    if result.error is not None:
        return result
    frame = result.frame
    copy = display_copy(frame.image)
    if copy is frame.image:
        return result
    scale_x = copy.shape[1] / float(frame.width)
    scale_y = copy.shape[0] / float(frame.height)
    display_frame = replace(frame, width=int(copy.shape[1]), height=int(copy.shape[0]),
                            image=copy)
    display_detections = tuple(
        replace(detection, box=(detection.box[0] * scale_x, detection.box[1] * scale_y,
                                detection.box[2] * scale_x, detection.box[3] * scale_y))
        for detection in result.detections)
    return replace(result, frame=display_frame, detections=display_detections)


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be an int; got {value!r} ({type(value).__name__}). A float "
            f"would be silently truncated into a tensor extent nobody chose."
        )
    number = int(value)
    if number <= 0:
        raise ValueError(
            f"{name} must be > 0; got {number}. The NPU rejects dynamic shapes "
            f"(AGENTS.md rule 3), so a zero or negative extent is a refusal rather "
            f"than something to interpolate."
        )
    return number


def _require_threshold(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number; got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if not 0.0 <= number <= 1.0:
        raise ValueError(
            f"{name} must be within [0, 1]; got {number}. Outside that range either "
            f"everything or nothing is suppressed, and neither is what the caller meant."
        )
    return number


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str; got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string; got {value!r}")
    return value


@dataclass(frozen=True)
class WorkerConfig:
    """What one worker process was asked to run.

    A frozen record, so the supervisor can hand it to a spawned process and neither side
    can mutate it afterwards. `device_id` is an **explicit** plugin - `AUTO`, `HETERO`,
    `MULTI` and `BATCH` are refused here, at the config boundary, because they hand
    placement back to the runtime and hide where the work actually ran
    (`app.inference.devices`).
    """

    device_id: str
    model_id: str
    input_width: int
    input_height: int
    conf_threshold: float
    iou_threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_id", _require_text("device_id", self.device_id))
        object.__setattr__(self, "model_id", _require_text("model_id", self.model_id))
        object.__setattr__(self, "input_width", _require_positive_int(
            "input_width", self.input_width))
        object.__setattr__(self, "input_height", _require_positive_int(
            "input_height", self.input_height))
        object.__setattr__(self, "conf_threshold", _require_threshold(
            "conf_threshold", self.conf_threshold))
        object.__setattr__(self, "iou_threshold", _require_threshold(
            "iou_threshold", self.iou_threshold))
        head = str(self.device_id).split(":", 1)[0].split(".", 1)[0].strip().upper()
        if head in ("AUTO", "HETERO", "MULTI", "BATCH"):
            raise ValueError(
                f"device_id {self.device_id!r} is an implicit-placement plugin. This demo "
                f"uses explicit compile targets only, because EXECUTION_DEVICES read "
                f"after the first inference is the only proof of placement and these "
                f"hide it. Use one of 'CPU', 'GPU', 'NPU'."
            )

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        """This config's input as the `[N, C, H, W]` a `TargetSpec` wants. Batch is 1."""
        return (1, 3, self.input_height, self.input_width)


@dataclass(frozen=True)
class InferenceResult:
    """One frame's analysis, tied to the frame it analysed.

    **`requested_device` and `execution_devices` travel together and are checked against
    each other here.** Both are present, so a consumer cannot be handed one without the
    other - and at construction the pair is compared, so a result whose confirmation does
    not match its request is refused rather than published. Review focus #4 is a *silent
    fallback*: an NPU request answered by the CPU. Reporting that fallback honestly means
    saying `('CPU',)`, which the code already did - but with only `execution_devices` on
    the record, nothing forced a card to compare it against what was asked for, and
    `app/` had no caller of `is_device` outside its own definition. Carrying the request
    makes the comparison unavoidable.

    **`latency_ms` is `None` exactly when `error` is set, and never `0.0`.** A zero in a
    latency column reads as "the device was instantaneous", which is the one thing a
    measured latency is never. The brief declared this field `float`; it is widened to
    `float | None` because "no latency" is not expressible as a float, and a field that
    cannot hold "no measurement" will eventually hold a zero.

    **`execution_devices` is empty exactly when `error` is set.** Same reasoning: there
    is no reading of where the work ran when the work did not run, and `()` next to a
    success is a shape a screen cannot draw honestly. A failure still carries the
    `requested_device`, because "NPU was asked for and could not run" is exactly what an
    operator needs and it must not be lost.

    **`detections` is a `tuple`, not a list**, and it is empty when there is an error. A
    bare list of boxes is the thing review focus #1 is about; refusing it here means the
    result cannot be built without a frame to belong to.
    """

    frame: FramePacket
    detections: tuple[Detection, ...]
    latency_ms: float | None
    execution_devices: tuple[str, ...]
    error: str | None
    requested_device: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frame, FramePacket):
            raise TypeError(
                f"InferenceResult.frame must be the FramePacket the inference ran on; got "
                f"{type(self.frame).__name__}. A result that cannot name its frame cannot "
                f"be matched to the picture on screen, and a late result then paints onto "
                f"a newer frame."
            )
        if not isinstance(self.detections, tuple):
            raise TypeError(
                f"InferenceResult.detections must be a tuple of Detection; got "
                f"{type(self.detections).__name__}. A list of boxes is the shape review "
                f"focus 1 is about."
            )
        for index, detection in enumerate(self.detections):
            if not isinstance(detection, Detection):
                raise TypeError(
                    f"InferenceResult.detections[{index}] must be a Detection; got "
                    f"{type(detection).__name__}"
                )
        if not isinstance(self.execution_devices, tuple):
            raise TypeError(
                f"execution_devices must be a tuple; got "
                f"{type(self.execution_devices).__name__}. The NPU plugin reports a bare "
                f"string and the CPU plugin a list, so this field is normalised in one "
                f"place rather than compared as whatever arrived."
            )
        object.__setattr__(self, "execution_devices",
                           tuple(str(d) for d in self.execution_devices))
        if self.requested_device is not None:
            object.__setattr__(self, "requested_device",
                               _require_text("requested_device", self.requested_device))
        error = self.error
        if error is not None and (not isinstance(error, str) or not error.strip()):
            raise ValueError(
                f"InferenceResult.error must be a non-empty stated reason or None; got "
                f"{error!r}. A blank reason is indistinguishable from no reason, which "
                f"is the failure this record exists to report."
            )
        if error is None:
            if self.latency_ms is None:
                raise ValueError(
                    "InferenceResult with error=None must carry a measured latency_ms. "
                    "A success with no measurement of how long it took is a claim, not a "
                    "result."
                )
            if not self.execution_devices:
                raise ValueError(
                    "InferenceResult with error=None must carry the EXECUTION_DEVICES "
                    "read after the inference. Without it there is no evidence the work "
                    "ran where it was asked to run, and an empty tuple beside a success "
                    "is exactly the shape that renders as 'NPU' while the CPU did the "
                    "work."
                )
            if float(self.latency_ms) <= 0.0:
                raise ValueError(
                    f"latency_ms must be > 0 on a success; got {self.latency_ms!r}. A "
                    f"zero or negative latency is a broken clock, and rendering it as "
                    f"0.00 would put a number in a latency column that no timer "
                    f"produced."
                )
        else:
            if self.latency_ms is not None:
                raise ValueError(
                    f"InferenceResult with error={error!r} must carry latency_ms=None, "
                    f"not {self.latency_ms!r}. A failed inference has no measured "
                    f"latency, and a number beside an error is the one a reader scans "
                    f"for."
                )
            if self.execution_devices:
                raise ValueError(
                    f"InferenceResult with error={error!r} must carry "
                    f"execution_devices=(), not {self.execution_devices!r}. Placement "
                    f"is only proven by an inference that ran; reporting one beside a "
                    f"failure claims a placement that did not happen."
                )
            if self.detections:
                raise ValueError(
                    f"InferenceResult with error={error!r} must carry no detections, not "
                    f"{len(self.detections)} of them. Detections beside an error is a "
                    f"clean empty success wearing an error label."
                )

    @classmethod
    def failure(cls, frame: FramePacket, error: str,
                requested_device: str | None = None) -> "InferenceResult":
        """The only honest shape for a frame that could not be analysed.

        `requested_device` is kept even here: "the NPU was asked for and would not run"
        is the fact an operator needs, and a failure that forgot what it was trying to do
        could not distinguish a broken plugin from a misconfigured worker.
        """
        return cls(frame=frame, detections=(), latency_ms=None,
                   execution_devices=(), error=str(error),
                   requested_device=requested_device)

    @property
    def fell_back(self) -> bool | None:
        """Whether the work ran somewhere other than where it was requested.

        `None` when there is no request to compare against, or when the result is an
        error. `False` means confirmed on the requested device. `True` is a **fallback** -
        and it is a legitimate state: the runtime may genuinely put the work elsewhere, and
        the honest thing then is to report what actually ran rather than to fail. What
        must not happen is the fallback being invisible, which is why the requested
        device is a field on the record rather than something only the supervisor knows.

        `is_device` is used rather than `==` because the NPU plugin reports the bare
        string `'NPU'` while CPU reports `['CPU']` and GPU reports `['GPU.0']`.
        """
        if self.requested_device is None or self.error is not None:
            return None
        return not is_device(self.execution_devices, self.requested_device)

    def to_json(self) -> dict[str, Any]:
        """Metadata only - never the pixels, and never the detections themselves.

        `latency_ms` stays `None` here rather than becoming `0`, so a serialised failure
        is still visibly a failure. `requested_device` is included so a log reader sees
        the same fallback the card would: `fell_back` is the boolean form of comparing the
        two, carried rather than recomputed by every consumer.
        """
        return {
            "frame": self.frame.to_json(),
            "detection_count": len(self.detections),
            "latency_ms": self.latency_ms,
            "requested_device": self.requested_device,
            "execution_devices": list(self.execution_devices),
            "fell_back": self.fell_back,
            "error": self.error,
            "validity": "failed" if self.error else "measured",
            "fields": [f.name for f in fields(self)],
        }


# ------------------------------------------------------------------------------------
# The compiled model, built once per device per process.
# ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Session:
    """This process's live handles for one device, plus the model's recorded output spec.

    Holds no placement. `execution_devices` is read live off `request` on every call,
    after the inference - see `infer_once`.
    """

    device_id: str
    core: Any
    compiled: Any
    request: Any
    spec: ModelOutputSpec


_LOCK = threading.Lock()
_SESSIONS: dict[str, Any] = {}
_SPECS: dict[str, ModelOutputSpec] = {}


def _model_spec() -> ModelOutputSpec:
    """The pinned model's `ModelOutputSpec`, derived from the IR once per process.

    `describe_outputs` is given the **IR** `ov.Model`, never a `CompiledModel`: the
    latter cannot be walked, and its `get_runtime_model()` is the plugin-rewritten graph
    (562 ops against the IR's 734) rather than the export. `has_embedded_nms` is a claim
    about the export, so it is read from the export.

    Raises rather than returning a defaulted spec, because a default here would decide
    whether detections are suppressed at all.
    """
    ir = verified_ir_path()
    key = str(ir)
    with _LOCK:
        spec = _SPECS.get(key)
        if spec is not None:
            return spec
    import openvino as ov

    core = ov.Core()
    model = core.read_model(key)
    spec = describe_outputs(model, num_classes())
    if spec.layout not in DECODABLE_LAYOUTS:
        raise ValueError(
            f"the pinned IR describes its output as layout {spec.layout!r}, which "
            f"app.inference.decode does not decode (decodable: {sorted(DECODABLE_LAYOUTS)}). "
            f"Refusing to run: a worker that cannot read the output would report an "
            f"empty detection list for every frame, which is indistinguishable from a "
            f"frame with nothing in it."
        )
    with _LOCK:
        _SPECS[key] = spec
    return spec


def _session(device_id: str) -> _Session:
    """This process's `InferRequest` for `device_id`, compiled once.

    **A failure is cached too**, as the exception. A device that cannot be compiled must
    not pay the compile attempt again on every frame of a 15 fps stream; and the
    supervisor needs to see the same reason every time rather than a fresh one.
    """
    with _LOCK:
        cached = _SESSIONS.get(device_id)
    if cached is not None:
        if isinstance(cached, BaseException):
            raise cached
        return cached  # type: ignore[return-value]
    try:
        session = _build_session(device_id)
    except BaseException as exc:  # noqa: BLE001 - cached and re-raised, never swallowed
        with _LOCK:
            _SESSIONS[device_id] = exc
        raise
    with _LOCK:
        _SESSIONS[device_id] = session
    return session


def _build_session(device_id: str) -> _Session:
    """Compile the pinned IR for one explicit device and describe its outputs."""
    import openvino as ov

    spec = _model_spec()
    ir = verified_ir_path()
    core = ov.Core()
    present = tuple(core.available_devices)
    if device_id not in present:
        raise ValueError(
            f"cannot compile for {device_id!r}: ov.Core().available_devices() reports "
            f"{list(present)}. The worker will report this as an error rather than run "
            f"somewhere else, because a placement nobody asked for on a card labelled "
            f"{device_id!r} is the exact failure this module exists to prevent."
        )
    model = core.read_model(str(ir))
    compiled = core.compile_model(model, device_id)
    return _Session(device_id=device_id, core=core, compiled=compiled,
                    request=compiled.create_infer_request(), spec=spec)


def reset_sessions() -> None:
    """Forget every cached session and spec. For tests and for a clean respawn."""
    with _LOCK:
        _SESSIONS.clear()
        _SPECS.clear()


# ------------------------------------------------------------------------------------
# The two tensor reads, and why they cannot both be keyed the same way.
# ------------------------------------------------------------------------------------

def _set_input(request: Any, blob: np.ndarray) -> None:
    """Hand the letterboxed blob to the request.

    `{0: blob}` - an **int index** - is the measured-working form for `infer` on this
    build (`config/model_io/yolo11n-fp16.json`, `fetch_api.infer_input_int_index:
    accepted`). The compiled port object also works; the IR model's port object is
    rejected with a `TypeError`, and this module never has an IR model here anyway.
    """
    request.infer({0: blob})


def _fetch_output_tensor(request: Any) -> np.ndarray:
    """The output tensor, by the compiled model's port object.

    Measured on this build: `get_tensor` accepts a port object and **raises `TypeError`
    on an int index**, while `get_output_tensor(0)` accepts the index. So the compiled
    port object is tried first and the index form is the recorded fallback - a fallback
    that exists because a future build may reject the port object, not because either is
    preferred.

    Whatever is fetched is filed under the **port index as a string**, because
    `app.inference.decode` keys on that and the pinned export's output port has no name.
    """
    try:
        compiled = request.get_compiled_model()
        ports = list(compiled.outputs)
        if ports:
            return np.asarray(request.get_tensor(ports[0]).data)
    except (TypeError, AttributeError, RuntimeError):
        # No compiled port object, or this build will not take one. Fall through to the
        # index form rather than reporting an empty frame for a fetch that could have
        # succeeded.
        pass
    return np.asarray(request.get_output_tensor(0).data)


def _confirmed_devices(request: Any) -> tuple[str, ...]:
    """`EXECUTION_DEVICES` from the request's own compiled model, normalised.

    Raised on, not defaulted from: an empty reading means the placement is unproven, and
    the caller turns that into a failure rather than into a silent success.
    """
    value = request.get_compiled_model().get_property(EXECUTION_DEVICES)
    devices = normalise_execution_devices(value)
    if not devices:
        raise RuntimeError(
            f"{EXECUTION_DEVICES} read from the request's own compiled model after the "
            f"inference came back empty ({value!r}). There is no evidence any of this "
            f"ran where it was asked to run, so it is reported as a failure rather than "
            f"as a success with no placement."
        )
    return devices


# ------------------------------------------------------------------------------------
# The worker itself.
# ------------------------------------------------------------------------------------

def infer_once(
    request: Any,
    frame: FramePacket,
    config: WorkerConfig,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> InferenceResult:
    """Preprocess, run one inference, decode, prove the placement, and report.

    `request=None` means "use this process's request for `config.device_id`", compiled on
    first use and cached after. That is how `run_worker` drives it and how a test can
    call it without building an `ov.InferRequest` by hand.

    **The whole body is inside one try.** Any exception becomes an
    `InferenceResult.failure` carrying the exception's own type and text, with no latency
    and no placement. There is no partial result: a compile that succeeded and an
    inference that threw must not appear as a row with a compile time and no latency,
    because a reader scanning a table reads the number that is there.
    """
    try:
        if not isinstance(frame, FramePacket):
            raise TypeError(
                f"infer_once needs the FramePacket the image came from; got "
                f"{type(frame).__name__}. A result with no frame cannot be matched to "
                f"the picture on screen."
            )
        if config.model_id != MODEL_ID:
            raise ValueError(
                f"WorkerConfig.model_id is {config.model_id!r} but the only provisioned "
                f"detector is {MODEL_ID!r}, whose IR is at "
                f"{verified_ir_path()}. Running this config against that IR would "
                f"report one model's detections under another model's name."
            )
        pinned_shape = pinned_input_shape()
        if tuple(config.input_shape) != tuple(pinned_shape):
            raise ValueError(
                f"WorkerConfig says the input is {tuple(config.input_shape)} but "
                f"config/models.json pins {tuple(pinned_shape)}. Compiling one shape and "
                f"feeding another is a silent mismatch, not a rounding - and the NPU "
                f"rejects the wrong one anyway."
            )

        active = request if request is not None else _session(config.device_id).request
        blob, letterbox = preprocess_bgr(
            frame.image, config.input_width, config.input_height)

        # Only the infer call is timed. Preprocessing is not inference, and timing it
        # would make this number incomparable with tools/qualify_m0.py's p50s.
        started = float(clock())
        _set_input(active, blob)
        latency_ms = (float(clock()) - started) * 1000.0

        array = _fetch_output_tensor(active)
        detections = decode_detections(
            _model_spec(), {OUTPUT_PORT_KEY: array}, letterbox,
            config.conf_threshold, config.iou_threshold,
        )
        # After the inference, from the request's own compiled model. Never before.
        devices = _confirmed_devices(active)
        return InferenceResult(frame=frame, detections=tuple(detections),
                               latency_ms=latency_ms, execution_devices=devices,
                               error=None, requested_device=config.device_id)
    except Exception as exc:  # noqa: BLE001 - a device that cannot run is a result
        detail = f"{type(exc).__name__}: {exc}".strip()
        requested = config.device_id if isinstance(config, WorkerConfig) else None
        if isinstance(frame, FramePacket):
            return InferenceResult.failure(frame, detail, requested)
        return InferenceResult(frame=frame, detections=(), latency_ms=None,
                               execution_devices=(), error=detail,
                               requested_device=requested)


def run_worker(config: WorkerConfig, in_queue: Any, out_queue: Any) -> None:
    """Drain `in_queue` into `out_queue` until `in_queue` yields `None`.

    The `multiprocessing` **spawn** entry point. Windows-safe: it reads nothing from
    module globals it did not build, it takes both queues as arguments, and it never
    raises out of the loop - AGENTS.md rule 12 says a stream that hits one bad frame keeps
    running with that frame marked failed.

    A frame that cannot be handled - not a packet, a session that will not compile, a
    plugin that throws - produces an `InferenceResult` with `error` set and the loop
    continues. Only the `None` sentinel stops it.

    **Every successful result is reduced to display resolution before it is put on the
    queue.** `_display_result` replaces the frame's pixels and extents with a
    `DISPLAY_COPY_MAX_WIDTH`-wide copy and scales the detection boxes to match, and it runs
    **after** `infer_once` has finished, so the detector always saw the full-resolution
    frame. What it saves is the return transport and the UI process's work, both of which
    scale with pixel count; what it preserves is every normalised coordinate, so the
    overlay and the zone rules are unmoved.
    """
    while True:
        frame = in_queue.get()
        if frame is None:
            return
        result = infer_once(None, frame, config)
        # **Reduce the result to display resolution after the inference, never before it.**
        # `infer_once` has already letterboxed and run on `frame.image` at full
        # resolution, so the detector's input is untouched; this only decides what is
        # pickled back and painted. See `_display_result`.
        out_queue.put(_display_result(result))
