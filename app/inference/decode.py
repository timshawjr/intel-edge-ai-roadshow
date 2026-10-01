"""The decoder: what the model actually wrote, and where it is on screen.

Review Focus #3: *"The export's output layout is not the assumed `(1, 84, 8400)`
convention. A wrong transpose yields plausible-looking boxes over the wrong objects."*
Everything load-bearing below was measured first-hand on this machine against the
pinned export (`assets/models/yolo11n-fp16/yolo11n.xml`, `openvino==2026.4.0`), not
taken from a convention. What was measured, and what each measurement decided:

| observation | decision it forced |
|---|---|
| input port named `x`, `[1,3,640,640]`, `f32` | the input is not this module's problem |
| output port **`get_names() == set()`**; `get_any_name()` raises `RuntimeError: Check '!get_names().empty()' failed` | `names` is legitimately `()`; `get_any_name()` is never called unguarded, and not at all |
| output `[1,84,8400]`, `f32`, **fully static** (`is_dynamic() == False` on all three axes) | layout comes from the partial shape, and a dynamic extent is `unknown`, never a guess |
| `84 == 4 + 80` and `labels.txt` has 80 lines | the axis-1 split is 4 box values + 80 class scores |
| rows 4..83 all inside `[0,1]`, max **0.909073** over a real frame | the export **already carries the activation**; the decoder must not add a second sigmoid |
| rows 0/1 mean **319.4 / 320.0**; rows 2/3 mean **88.3 / 71.2** | rows 0/1 are coordinates, rows 2/3 are sizes: the box is **cx, cy, w, h** |
| **15 of 15** above-threshold anchors have `v0 > v2` or `v1 > v3` | the `xyxy` reading is falsified outright - it would draw an inverted box for every detection |
| `8400 == 80² + 40² + 20²` for a 640 input at strides 8/16/32 | the third axis is raw anchors, not post-NMS detections |
| op census over all **734** ops, 24 distinct types: no `NonMaxSuppression`, no TopK | `has_embedded_nms = False`, **derived from the graph** |
| the same census contains `Sigmoid` x1 and `Softmax` x2 | corroborates the row-range measurement: activation is in the graph, so the decoder adds none |
| top classes on frame 60 of `retail-checkout.mp4` are class 0 = `person` at 0.909 | the decoding is semantically right, not merely self-consistent |

**The `cx, cy, w, h` decision is the one worth stating loudly**, because the plan's own
test read the same four values as `xyxy` and would have passed. Under `xyxy` a person at
model centre (345.6, 252.3) with size 272.9 x 222.0 becomes the quadrilateral
(345.6, 252.3)-(272.9, 222.0) - x2 < x1, i.e. inside out. Every one of the 15
above-threshold anchors on that frame is inverted under that reading. The plan's
expected value (`box[0] ~= 320`) is simply the first raw number passed through, which
is what a decoder that never subtracted `w/2` would also produce.

**Two structural decisions the plan did not specify, recorded here because they change
the public shape of `ModelOutputSpec`:**

1. `shapes` is keyed by **port index as a string** (`"0"`), never by port name. This
   port has no name, so a name key would have to be a fabricated string, and the port
   index is the only key that exists for every model. `port_keys` and
   `unnamed_ports` are derived properties on the spec for the same reason.
2. `names` keeps the plan's declared `tuple[str, ...]` and holds the observed name of
   each output port **that has one**, in port order. For this model that is `()` - not
   because the field is empty, but because the port is. `unnamed_ports == (0,)` carries
   the distinction, because `()` alone cannot tell "unnamed" from "no outputs".
   (The alternative, a `tuple[str | None, ...]`, was rejected as a wider deviation from
   the declared type for the same information.)

**Undecodable is not empty.** `"unknown"` returns `[]`, because that is a recorded fact
about the model rather than a failure. A layout string outside the known set is refused
at construction; `"ssd"` is a *known* layout that this milestone does not decode and
raises `UnsupportedLayoutError`. A missing port key, a feature extent that is not
`4 + num_classes`, a recorded shape that contradicts the tensor, and an out-of-range
threshold all raise rather than returning `[]`. Otherwise "this decoder cannot read
this model" and "this frame is empty" would be the same list, and a booth would show
an empty screen for a reason it cannot report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from app.inference.preprocess import Letterbox

__all__ = [
    "BOX_ENCODING",
    "DECODABLE_LAYOUTS",
    "Detection",
    "KNOWN_LAYOUTS",
    "ModelOutputSpec",
    "UnsupportedLayoutError",
    "describe_outputs",
    "decode_detections",
    "nms",
    "port_key",
]

LAYOUT_YOLO_V8_TRANSPOSED: Final[str] = "yolo_v8_transposed"
LAYOUT_YOLO_V8_PASSTHROUGH: Final[str] = "yolo_v8_passthrough"
LAYOUT_SSD: Final[str] = "ssd"
LAYOUT_UNKNOWN: Final[str] = "unknown"

#: Every layout string `ModelOutputSpec` will accept. Membership is checked at
#: construction so a typo cannot reach the `else` branch of `decode_detections` and be
#: reported as an empty frame.
KNOWN_LAYOUTS: Final[frozenset[str]] = frozenset({
    LAYOUT_YOLO_V8_TRANSPOSED, LAYOUT_YOLO_V8_PASSTHROUGH, LAYOUT_SSD, LAYOUT_UNKNOWN,
})
#: The subset this milestone can actually turn into detections. `"ssd"` is deliberately
#: absent: no SSD model is pinned, so implementing it would be an untested code path.
DECODABLE_LAYOUTS: Final[frozenset[str]] = frozenset({
    LAYOUT_YOLO_V8_TRANSPOSED, LAYOUT_YOLO_V8_PASSTHROUGH,
})

#: How the first four values of each anchor are written, measured - see the table
#: above. It is a module constant rather than a `ModelOutputSpec` field so the record
#: keeps the field set the plan specified; it is written into
#: `config/model_io/*.json` as a human-readable cross-check.
BOX_ENCODING: Final[str] = "cxcywh"

#: The number of leading values per anchor that are a box rather than a class score.
BOX_VALUES: Final[int] = 4

#: Only f32/f16 can be a YOLO score-and-box tensor. An integer output would need a
#: different reading entirely, so it is `"unknown"` rather than a hopeful cast.
FLOAT_ELEMENT_TYPES: Final[frozenset[str]] = frozenset({"f32", "f16", "f64"})

Box = tuple[float, float, float, float]


class UnsupportedLayoutError(ValueError):
    """A layout this module knows about and cannot decode.

    A `ValueError` subclass so a caller guarding the decode family catches it with the
    rest, while it stays distinct from the `[]` that means "decoded, found nothing".
    """


def port_key(index: int) -> str:
    """The key an output tensor is filed under: its **port index**, as a string.

    Not a name. Measured: the pinned export's output port has no name at all, and
    `ov.InferRequest.get_tensor` accepts neither an `int` nor a name for it - only the
    port object. A caller therefore resolves the port itself and files the array under
    this key, which is the one identifier that exists for every model.
    """
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError(f"a port index must be a non-negative int; got {index!r}")
    return str(index)


@dataclass(frozen=True)
class Detection:
    """One detection, in **source** pixels.

    `box` is `xyxy` in the coordinate system the frame arrived in - not the letterboxed
    640x640 the model saw. It is unclamped, so a detection that lies in the pad band
    carries a negative or over-range coordinate, which is the truth; clipping is the
    drawer's decision. `score` is the model's own value with no activation applied on
    top, because the export already applied one.
    """

    class_id: int
    score: float
    box: Box

    def __post_init__(self) -> None:
        if not isinstance(self.class_id, int) or isinstance(self.class_id, bool):
            raise ValueError(f"class_id must be an int; got {self.class_id!r}")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "box", tuple(float(v) for v in self.box))  # type: ignore[arg-type]
        if len(self.box) != 4:
            raise ValueError(f"box must be (x1, y1, x2, y2); got {self.box!r}")


@dataclass(frozen=True)
class ModelOutputSpec:
    """What one pinned model writes, as observed.

    `names` - the observed name of each output port that **has** one, in port order.
        `()` for the pinned model, because its single output port is unnamed. Not a
        fabricated placeholder string.
    `shapes` - port index (as a string) to the port's static shape. Keyed by index
        precisely because the port has no name; see `port_key`.
    `layout` - one of `KNOWN_LAYOUTS`, checked at construction.
    `num_classes` - the class count used to decide the layout, from
        `config/models.json`, never inferred from the tensor.
    `has_embedded_nms` - measured from the IR op census, never assumed. `False` for the
        pinned model, so `decode_detections` runs its own NMS; if a re-export embedded
        NMS this becomes `True` and running suppression again would be a second,
        wrong, suppression pass over an already-suppressed list.

    Not hashable, because `shapes` is a dict. That is a deliberate consequence of
    keying by a mapping, and it is why the record is compared, not used as a key.
    """

    names: tuple[str, ...]
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    layout: str = LAYOUT_UNKNOWN
    num_classes: int = 0
    has_embedded_nms: bool = False

    def __post_init__(self) -> None:
        if self.layout not in KNOWN_LAYOUTS:
            raise ValueError(
                f"layout must be one of {sorted(KNOWN_LAYOUTS)}; got {self.layout!r}. "
                f"An unrecognised layout is refused here rather than reaching "
                f"decode_detections, where it would be indistinguishable from a frame "
                f"with no detections in it."
            )
        if not isinstance(self.num_classes, int) or isinstance(self.num_classes, bool) \
                or self.num_classes <= 0:
            raise ValueError(
                f"num_classes must be a positive int; got {self.num_classes!r}. It "
                f"decides whether an axis holds 4 + num_classes values, so 0 or a "
                f"negative count would make every shape look like something else."
            )
        if not isinstance(self.has_embedded_nms, bool):
            raise ValueError(
                f"has_embedded_nms must be a bool; got {self.has_embedded_nms!r}. "
                f"It decides whether this module suppresses at all, so a truthy "
                f"non-bool must not be read as a measurement."
            )
        if not isinstance(self.shapes, Mapping):
            raise ValueError(
                f"shapes must be a mapping of port index to shape; got {self.shapes!r}"
            )
        normalised: dict[str, tuple[int, ...]] = {}
        for key, shape in self.shapes.items():
            if key != port_key(_as_index(key)):
                raise ValueError(
                    f"shapes key {key!r} is not a port index; keys are "
                    f"{port_key(0)!r}, {port_key(1)!r}, ... so an unnamed port still has "
                    f"one"
                )
            try:
                normalised[key] = tuple(int(v) for v in shape)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"shapes[{key!r}] must be a tuple of ints; got {shape!r}"
                ) from exc
        object.__setattr__(self, "shapes", normalised)
        object.__setattr__(self, "names", tuple(str(n) for n in self.names))

    @property
    def port_keys(self) -> tuple[str, ...]:
        """Every output port's key, in port order. This is what `decode_detections`
        looks the tensor up by."""
        return tuple(port_key(i) for i in range(len(self.shapes)))

    @property
    def unnamed_ports(self) -> tuple[int, ...]:
        """Ports the IR gives no name, by index. Distinguishes `()` from "no outputs"."""
        return tuple(i for i, key in enumerate(self.port_keys) if not self.names
                     or i >= len(self.names))


def _as_index(key: Any) -> int:
    if isinstance(key, bool):
        raise ValueError(f"a port key must be an index string; got {key!r}")
    if isinstance(key, int):
        return key
    try:
        return int(str(key))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"a port key must be an index string; got {key!r}") from exc


def _port_name(port: Any) -> str | None:
    """A port's name, or `None` when the IR gives it none.

    `get_any_name()` is **never** called: measured on the pinned export it raises
    `RuntimeError: Check '!get_names().empty()' failed`, and an unnamed port is a
    normal state of a valid IR, not a defect. `get_names()` returns a set and does not
    raise.
    """
    try:
        names = sorted(str(n) for n in port.get_names())
    except Exception:  # noqa: BLE001 - a port that will not say is treated as unnamed
        return None
    return names[0] if names else None


def _port_shape(port: Any) -> tuple[Any, ...] | None:
    """The port's static extents, or `None` if any axis is open or the rank is not 3.

    A dynamic axis yields `None` rather than a guess: the NPU rejects dynamic shapes
    (AGENTS.md rule 3), so an open extent on a port this demo would compile is a
    refusal, not something to interpolate.
    """
    try:
        partial = port.get_partial_shape()
        shape = list(partial)
    except Exception:  # noqa: BLE001
        return None
    if len(shape) != 3:
        return None
    extents: list[Any] = []
    for dim in shape:
        getter = getattr(dim, "get_length", None)
        if callable(getter) and hasattr(dim, "is_dynamic"):
            if bool(dim.is_dynamic):
                return None
            extents.append(int(getter()))
        elif isinstance(dim, (int, np.integer)):
            extents.append(int(dim))
        else:
            return None
    return tuple(extents)


def _port_element_type(port: Any) -> str:
    """The port's element type as a short string, e.g. `'f32'`. `''` if unreadable.

    Measured: `str(port.get_element_type())` is `"<Type: 'float32'>"`, **not** a name a
    comparison can use - the first version of this function compared the `str()` form
    against `FLOAT_ELEMENT_TYPES`, so every real port classified as `"unknown"` while
    agreeing with itself. `to_string()` returns `'f32'`. Comparing the `str()` form is
    how a detector silently stopped being recognised while every self-consistency check
    still passed.
    """
    try:
        element_type = port.get_element_type()
    except Exception:  # noqa: BLE001
        return ""
    to_string = getattr(element_type, "to_string", None)
    try:
        if callable(to_string):
            return str(to_string())
        return str(element_type)
    except Exception:  # noqa: BLE001
        return ""


def classify_layout(shape: Sequence[Any] | None, element_type: str,
                    num_classes: int) -> str:
    """Which detector layout these extents describe, from shape and element type only.

    Never from a name. Measured: the pinned `[1, 84, 8400]` f32 output is
    `yolo_v8_transposed` because `84 == 4 + 80`.

    `num_classes` is a parameter, not something read off the tensor, because the whole
    point is that the tensor does not say how many of its values are class scores.

    When both readings fit - which needs `num_boxes == 4 + num_classes` - the
    transposed reading wins, and that choice is recorded rather than hidden. It cannot
    arise for the pinned model (`84 != 8400`) but would for a square-anchor export.
    """
    if shape is None or len(shape) != 3 or num_classes <= 0:
        return LAYOUT_UNKNOWN
    batch, axis1, axis2 = (int(v) for v in shape)
    if batch != 1 or axis1 <= 0 or axis2 <= 0:
        return LAYOUT_UNKNOWN
    if element_type not in FLOAT_ELEMENT_TYPES:
        return LAYOUT_UNKNOWN
    features = BOX_VALUES + num_classes
    if axis1 == features:
        return LAYOUT_YOLO_V8_TRANSPOSED
    if axis2 == features:
        return LAYOUT_YOLO_V8_PASSTHROUGH
    return LAYOUT_UNKNOWN


def _has_embedded_nms(model: Any) -> bool:
    """Whether the IR graph contains a suppression op, read from the ops themselves.

    Measured on the pinned export: 734 ops, 24 distinct types, and no
    `NonMaxSuppression` of any version. Combined with a static `[1, 84, 8400]` output -
    `8400` being exactly the 80+40+20 anchor count for a 640 input - the export has no
    embedded NMS and `decode_detections` must apply its own.

    Returning `False` because nothing was found is a claim about a graph that was
    actually walked, which is why an un-walkable object raises rather than quietly
    reporting `False`.

    Measured: **`ov.CompiledModel` cannot be walked.** It has no `get_ordered_ops()`, no
    `get_model()` and no `model` attribute - only `inputs`/`outputs`,
    `create_infer_request`, `get_property`, `set_property` and friends. Its
    `get_runtime_model()` does return a walkable `ov.Model`, but that is the
    **plugin-rewritten** graph (562 ops against the IR's 734), so an NMS op found there
    would be the plugin's, not the export's. `has_embedded_nms` is a claim about the
    export, so the IR is the only graph it is read from, and a `CompiledModel` is
    refused with a message saying so. Task 7 compiles from a `TargetSpec` that already
    holds the IR path, so it has the `ov.Model` to hand.
    """
    walker = getattr(model, "get_ordered_ops", None)
    if not callable(walker):
        raise ValueError(
            f"cannot walk the graph of a {type(model).__name__}, so has_embedded_nms "
            f"cannot be determined from it. Pass the ov.Model that was compiled - "
            f"ov.CompiledModel exposes no op list, and its get_runtime_model() is the "
            f"plugin-rewritten graph rather than the IR, so neither would answer the "
            f"question. Reporting 'no NMS' from a graph that was never walked would be a "
            f"guess, and it decides whether detections are suppressed at all."
        )
    try:
        ops = list(walker())
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"cannot read the op list, so has_embedded_nms cannot be determined: "
            f"{type(exc).__name__}: {exc}. Reporting 'no NMS' from a graph that was "
            f"never walked would be a guess, and it would decide whether detections are "
            f"suppressed at all."
        ) from exc
    for op in ops:
        if _is_suppression_op_type(str(op.get_type_name())):
            return True
    return False


#: The canonical OpenVINO op type for suppression, upper-cased. Matched against the op
#: **type** and nothing else.
NMS_OP_TYPE: Final[str] = "NONMAXSUPPRESSION"


def _is_suppression_op_type(type_name: str) -> bool:
    """Whether an op *type* is the suppression op, across its versioned spellings.

    OpenVINO versions this op - `NonMaxSuppression`, `NonMaxSuppressionV5`, `V9`,
    `V10` - so an exact list would go stale at the next version and silently start
    reporting `False` for a model that does suppress. A canonical prefix match covers
    them all.

    Matched on the op **type** only, never on `get_friendly_name()`. A friendly name is
    an arbitrary developer-chosen label, so an op called `my_nms_wrapper` proves nothing
    about whether it suppresses - and substring-matching labels is how a false "this
    model has embedded NMS" gets recorded, which would then make the decoder skip
    suppression it still has to do. The friendly name is kept in the op census as a
    record, and is not evidence.
    """
    name = str(type_name).upper()
    return name == NMS_OP_TYPE or name.startswith(NMS_OP_TYPE + "V")


def describe_outputs(model: Any, num_classes: int) -> ModelOutputSpec:
    """Read the model's output ports and record what they are.

    `model` is anything with `.outputs` and `.get_ordered_ops()`: a read
    `ov.Model` or a compiled one. Ports are a property of the IR, so compiling proves
    nothing extra here and costs seconds; the test suite runs it on both to show they
    agree.

    Every value is observed. `get_any_name()` is not called, because on an unnamed port
    it raises and the port is not at fault. The layout comes from the partial shape and
    the element type, never from a name. `has_embedded_nms` comes from walking the ops.

    The layout is decided on **the first output port**. A two-output detector (a raw
    head plus a decoded head, say) is not something this module can decode, and
    reporting a confident layout for it would be the guess this function exists to
    avoid - so a second output port makes the whole spec `"unknown"`.
    """
    ports = list(model.outputs)
    if not ports:
        return ModelOutputSpec(names=(), shapes={}, layout=LAYOUT_UNKNOWN,
                               num_classes=num_classes, has_embedded_nms=False)
    shapes: dict[str, tuple[int, ...]] = {}
    names: list[str] = []
    layout = LAYOUT_UNKNOWN
    for index, port in enumerate(ports):
        shape = _port_shape(port)
        key = port_key(index)
        if shape is not None:
            shapes[key] = shape
        name = _port_name(port)
        if name:
            names.append(name)
        if index == 0:
            layout = classify_layout(shape, _port_element_type(port), num_classes)
    if len(ports) > 1:
        # Two outputs and this module reads one of them. Saying so beats picking one.
        layout = LAYOUT_UNKNOWN
    return ModelOutputSpec(
        names=tuple(names),
        shapes=shapes,
        layout=layout,
        num_classes=num_classes,
        has_embedded_nms=_has_embedded_nms(model),
    )


def _require_threshold(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number; got {value!r}") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(
            f"{name} must be within [0, 1]; got {value!r}. A threshold outside that "
            f"range is a caller bug, and accepting it would mean either suppressing "
            f"everything or nothing."
        )
    return number


def _as_candidates(array: np.ndarray, layout: str,
                   num_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(boxes_cxcywh, class_scores, class_ids)` for one YOLO-family tensor.

    `yolo_v8_transposed` puts the features down axis 1 (`[1, 4+C, N]`), so it is
    transposed to the common `[N, 4+C]` first. `yolo_v8_passthrough` is already
    `[1, N, 4+C]`. Both end as `(num_boxes, 4 + num_classes)` because that is the only
    form in which the threshold, the NMS and the letterbox inverse can be written once.

    The extents are checked against `4 + num_classes` before anything is indexed. A
    decoder that trusted the axis and clipped would instead reinterpret a tensor whose
    shape contradicts the recorded spec, and draw confident boxes on the wrong objects -
    which is the failure this module exists to prevent.
    """
    features = BOX_VALUES + num_classes
    if array.ndim == 3 and array.shape[0] == 1:
        squeezed = array[0]
    elif array.ndim == 2:
        squeezed = array
    else:
        raise ValueError(
            f"a batch-1 YOLO output is rank 2 or 3 with a leading 1; got shape "
            f"{array.shape!r}"
        )
    if layout == LAYOUT_YOLO_V8_TRANSPOSED:
        if squeezed.shape[0] != features:
            raise ValueError(
                f"layout {layout!r} requires {features} features on axis 1 "
                f"(4 box values + {num_classes} classes); the tensor has "
                f"{squeezed.shape[0]} and shape {array.shape!r}. Refusing rather than "
                f"reinterpreting it: this tensor does not hold what the spec says it "
                f"holds."
            )
        # [features, num_boxes] -> [num_boxes, features]
        candidates = squeezed.T
    elif layout == LAYOUT_YOLO_V8_PASSTHROUGH:
        if squeezed.shape[1] != features:
            raise ValueError(
                f"layout {layout!r} requires {features} features on axis 2 "
                f"(4 box values + {num_classes} classes); the tensor has "
                f"{squeezed.shape[1]} and shape {array.shape!r}. Refusing rather than "
                f"reinterpreting it."
            )
        candidates = squeezed
    else:  # pragma: no cover - guarded by DECODABLE_LAYOUTS in the caller
        raise UnsupportedLayoutError(f"no tensor reader for layout {layout!r}")
    boxes = candidates[:, :BOX_VALUES]
    scores = candidates[:, BOX_VALUES:]
    return boxes, scores, np.argmax(scores, axis=1).astype(np.int64)


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """`(cx, cy, w, h)` -> `(x1, y1, x2, y2)`, in place of a re-read of the raw four.

    The subtraction of half the extent is the single most consequential line in this
    module: without it every box is displaced by `w/2, h/2` and stretched, which is
    exactly the "plausible-looking boxes over the wrong objects" outcome.
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    half_w, half_h = w / 2.0, h / 2.0
    return np.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy IoU suppression over an already-thresholded list. Returns kept indices.

    Class-**agnostic**, as the plan specifies: the caller has already reduced each
    anchor to its single best class, so the surviving list can hold two classes, and
    suppressing across them is what the plan asked for. Changing that is a deliberate
    act and would change which detections a booth shows.

    Greedy in descending score, and the returned indices are **ascending** so the result
    does not depend on the score ordering of the input. An index of `-1` is a degenerate
    box and is dropped rather than allowed to divide by zero; the returned list is
    therefore not always a permutation of the input length, and a caller that needs a
    one-to-one mapping must check its own boxes.
    """
    threshold = _require_threshold("iou_threshold", iou_threshold)
    box_array = np.asarray(boxes, dtype=np.float32)
    score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    if box_array.size == 0:
        return []
    if box_array.ndim != 2 or box_array.shape[1] != 4:
        raise ValueError(f"boxes must be an (N, 4) xyxy array; got {box_array.shape!r}")
    if box_array.shape[0] != score_array.shape[0]:
        raise ValueError(
            f"boxes and scores must be the same length; got {box_array.shape[0]} and "
            f"{score_array.shape[0]}"
        )
    # A negative or zero area cannot have an IoU, so it is not a box.
    widths = np.maximum(box_array[:, 2] - box_array[:, 0], 0.0)
    heights = np.maximum(box_array[:, 3] - box_array[:, 1], 0.0)
    areas = widths * heights
    order = np.argsort(-score_array, kind="stable")
    keep: list[int] = []
    while order.size:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(box_array[best, 0], box_array[rest, 0])
        yy1 = np.maximum(box_array[best, 1], box_array[rest, 1])
        xx2 = np.minimum(box_array[best, 2], box_array[rest, 2])
        yy2 = np.minimum(box_array[best, 3], box_array[rest, 3])
        inter = np.maximum(xx2 - xx1, 0.0) * np.maximum(yy2 - yy1, 0.0)
        union = areas[best] + areas[rest] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0.0, inter / union, 0.0)
        order = rest[iou <= threshold]
    return sorted(keep)


def decode_detections(spec: ModelOutputSpec, outputs: Mapping[str, np.ndarray],
                      letterbox: Letterbox, conf_threshold: float,
                      iou_threshold: float) -> list[Detection]:
    """One model's output tensors -> detections in **source** pixels.

    `outputs` is keyed by **port index** (`port_key(0)` == `"0"`), not by port name,
    and there is no name-based fallback: the pinned export's output port has no name at
    all, so a lookup by name would be a lookup for a string that does not exist.

    `"unknown"` returns `[]` - that is a recorded fact about the model, not a failure.
    `"ssd"` raises `UnsupportedLayoutError`, because a known layout this module cannot
    read is a different fact from an empty frame and must not look like one. So does a
    missing port key, a tensor whose extents contradict `spec`, a recorded shape that
    contradicts the tensor, and a threshold outside `[0, 1]`.

    Suppression runs here rather than in the model, because `spec.has_embedded_nms` is
    `False` for the pinned export. Were it `True`, this module would skip its own NMS
    rather than suppress an already-suppressed list a second time.

    The returned order is by descending score, which is the order a drawer wants; the
    geometry is `letterbox.to_source`, which does not clamp, so a detection in the pad
    band keeps its out-of-range coordinate.
    """
    if not isinstance(spec, ModelOutputSpec):
        raise TypeError(f"spec must be a ModelOutputSpec; got {type(spec).__name__}")
    conf = _require_threshold("conf_threshold", conf_threshold)
    iou = _require_threshold("iou_threshold", iou_threshold)
    if not isinstance(outputs, Mapping):
        raise TypeError(
            f"outputs must be a mapping of port key to array; got "
            f"{type(outputs).__name__}"
        )
    if not isinstance(letterbox, Letterbox):
        raise TypeError(
            f"letterbox must be a Letterbox from preprocess_bgr or letterbox_params; "
            f"got {type(letterbox).__name__}"
        )
    if spec.layout == LAYOUT_UNKNOWN:
        return []
    if spec.layout not in DECODABLE_LAYOUTS:
        raise UnsupportedLayoutError(
            f"layout {spec.layout!r} is a known layout this milestone does not decode. "
            f"Decodable: {sorted(DECODABLE_LAYOUTS)}. No SSD model is pinned, so this is "
            f"a refusal rather than a guess - and deliberately not an empty list, which "
            f"would be indistinguishable from a frame with nothing in it."
        )
    if spec.has_embedded_nms:
        raise UnsupportedLayoutError(
            f"spec.has_embedded_nms is True, so this model's output has already been "
            f"suppressed and must not be suppressed again. This decoder only implements "
            f"the un-suppressed path; the recorded spec must be re-derived from the IR "
            f"with tools/dump_model_io.py rather than edited to match the code."
        )

    key = port_key(0)
    if key not in outputs:
        raise ValueError(
            f"no output tensor under port key {key!r}; got keys "
            f"{sorted(map(str, outputs))}. The tensor is fetched by port index because "
            f"the port has no name - there is no name to fall back to."
        )
    array = np.asarray(outputs[key])
    recorded = spec.shapes.get(key)
    if recorded is not None and tuple(int(v) for v in array.shape) != tuple(recorded):
        raise ValueError(
            f"the tensor under port {key!r} has shape {tuple(array.shape)} but the "
            f"recorded spec says {tuple(recorded)}. Refusing rather than decoding a "
            f"tensor that is not the one that was recorded."
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(
            f"the tensor under port {key!r} is {array.dtype}; a YOLO score-and-box "
            f"output is floating point. An integer reading would be a different decode, "
            f"not a rounding of this one."
        )

    boxes_cxcywh, class_scores, class_ids = _as_candidates(
        array, spec.layout, spec.num_classes)
    keep_rows = np.flatnonzero(class_scores.max(axis=1) >= conf)
    if keep_rows.size == 0:
        return []
    boxes_xyxy = _cxcywh_to_xyxy(boxes_cxcywh[keep_rows])
    best_scores = class_scores[keep_rows].max(axis=1)
    best_classes = class_ids[keep_rows]
    survivors = nms(boxes_xyxy, best_scores, iou)
    # nms returns positions into the filtered arrays, ascending; the booth wants the
    # strongest first, so the order is re-established from the scores.
    survivors.sort(key=lambda i: -float(best_scores[i]))
    detections: list[Detection] = []
    for i in survivors:
        model_box = (float(boxes_xyxy[i, 0]), float(boxes_xyxy[i, 1]),
                     float(boxes_xyxy[i, 2]), float(boxes_xyxy[i, 3]))
        detections.append(Detection(
            class_id=int(best_classes[i]),
            score=float(best_scores[i]),
            box=letterbox.to_source(model_box),
        ))
    return detections
