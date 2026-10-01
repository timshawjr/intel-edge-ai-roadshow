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
from types import MappingProxyType
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
        score = float(self.score)
        box = tuple(float(v) for v in self.box)
        # Arity is checked BEFORE anything is stored, so a bad box leaves no half-built
        # record behind. Setting first and validating after would have meant a 2-tuple
        # was already installed on `self` when the refusal was raised.
        if len(box) != 4:
            raise ValueError(f"box must be (x1, y1, x2, y2); got {self.box!r}")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "box", box)


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
    `named_port_keys` - the port key of each entry of `names`, in the same order. This
        exists because `names` is **compacted** (a port with no name contributes
        nothing), so `names[j]` is NOT port `j` and the two cannot be indexed together.
        A two-output model with port 0 unnamed and port 1 named `b` has
        `names == ("b",)`, and without this field `unnamed_ports` would report port 1 -
        the named one - as the unnamed one. `describe_outputs` always fills it, and a
        hand-built spec whose `names` and `named_port_keys` disagree in length is
        refused rather than silently mis-aligned.

    **Immutable, and unhashable.** `shapes` is wrapped in a read-only
    `MappingProxyType`, so `spec.shapes["0"] = ...` raises instead of silently changing
    what `decode_detections` will read: the record is built by `describe_outputs` and
    read later by the decoder, and a mutation in between would change the decode with no
    trace. The `frozen=True` dataclass alone did not give this, because a dict is
    mutable however frozen the holder is. Being unhashable follows from the mapping.
    """

    names: tuple[str, ...]
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    layout: str = LAYOUT_UNKNOWN
    num_classes: int = 0
    has_embedded_nms: bool = False
    named_port_keys: tuple[str, ...] = ()

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
        names = tuple(str(n) for n in self.names)
        named_port_keys = tuple(str(k) for k in self.named_port_keys)
        if len(names) != len(named_port_keys):
            raise ValueError(
                f"names has {len(names)} entr(ies) but named_port_keys has "
                f"{len(named_port_keys)}: {names!r} against {named_port_keys!r}. names is "
                f"compacted - a port with no name contributes nothing - so the two must be "
                f"built together, and unnamed_ports cannot be derived without this pairing. "
                f"Use describe_outputs, which fills both."
            )
        for key in named_port_keys:
            if key not in normalised:
                raise ValueError(
                    f"named_port_keys names port {key!r}, which is not in shapes "
                    f"({sorted(normalised)}); a name for a port with no recorded shape "
                    f"cannot be lined up with one that has a shape."
                )
        # Read-only, so a caller cannot change the decode after the describe. The
        # `frozen=True` dataclass does not do this: the attribute is frozen, the dict it
        # points at is not.
        object.__setattr__(self, "shapes", MappingProxyType(normalised))
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "named_port_keys", named_port_keys)

    @property
    def port_keys(self) -> tuple[str, ...]:
        """Every output port's key, in port order. This is what `decode_detections`
        looks the tensor up by."""
        return tuple(port_key(i) for i in range(len(self.shapes)))

    @property
    def unnamed_ports(self) -> tuple[int, ...]:
        """Ports the IR gives no name, by index. Distinguishes `()` from "no outputs".

        Derived from `named_port_keys`, never from the position of an entry in `names`:
        `names` is compacted, so a two-output model with port 0 unnamed and port 1 named
        has `names == ("b",)` and would otherwise report port **1** - the named one - as
        unnamed. Measured, and the reason `named_port_keys` exists.
        """
        named = set(self.named_port_keys)
        return tuple(i for i, key in enumerate(self.port_keys) if key not in named)


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

    `model` is anything with `.outputs` and `.get_ordered_ops()`: a read `ov.Model` or a
    runtime model. **Not** a `CompiledModel`, which cannot be walked - see
    `_has_embedded_nms`. Ports are a property of the IR, so compiling proves nothing
    extra here and costs seconds.

    Every value is observed. `get_any_name()` is not called, because on an unnamed port
    it raises and the port is not at fault. The layout comes from the partial shape and
    the element type, never from a name. `has_embedded_nms` comes from walking the ops.

    **The graph is walked before any early return.** A zero-output model is a real IR
    state, and the honest answer for one is "this graph contains no suppression op",
    not "there is nothing to suppress" - the first is a measurement, the second is an
    assumption about a graph that was never opened. So `has_embedded_nms` is computed
    from the ops first and the zero-port case only *reports* it.

    The layout is decided on **the first output port**. A two-output detector (a raw
    head plus a decoded head, say) is not something this module can decode, and
    reporting a confident layout for it would be the guess this function exists to
    avoid - so a second output port makes the whole spec `"unknown"`.
    """
    # First, and unconditionally: has_embedded_nms is a claim about the graph.
    has_embedded_nms = _has_embedded_nms(model)
    ports = list(model.outputs)
    if not ports:
        return ModelOutputSpec(names=(), shapes={}, named_port_keys=(),
                               layout=LAYOUT_UNKNOWN, num_classes=num_classes,
                               has_embedded_nms=has_embedded_nms)
    shapes: dict[str, tuple[int, ...]] = {}
    names: list[str] = []
    named_keys: list[str] = []
    layout = LAYOUT_UNKNOWN
    for index, port in enumerate(ports):
        shape = _port_shape(port)
        key = port_key(index)
        if shape is not None:
            shapes[key] = shape
        name = _port_name(port)
        if name:
            # The key is recorded with the name, because `names` alone cannot say which
            # port an entry belongs to. See ModelOutputSpec.named_port_keys.
            names.append(name)
            named_keys.append(key)
        if index == 0:
            layout = classify_layout(shape, _port_element_type(port), num_classes)
    if len(ports) > 1:
        # Two outputs and this module reads one of them. Saying so beats picking one.
        layout = LAYOUT_UNKNOWN
    return ModelOutputSpec(
        names=tuple(names),
        shapes=shapes,
        named_port_keys=tuple(named_keys),
        layout=layout,
        num_classes=num_classes,
        has_embedded_nms=has_embedded_nms,
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
    else:
        # Defence in depth, not a branch the decode path can reach: `decode_detections`
        # refuses anything outside DECODABLE_LAYOUTS before it gets here, and that
        # refusal is what the tests pin. This backstop exists so that a future caller
        # who reaches `_as_candidates` directly still gets an explicit refusal rather
        # than a mis-read tensor.
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
    does not depend on the score ordering of the input.

    **Degenerate boxes are dropped, not kept.** A box with zero or negative area - one
    where `x2 <= x1` or `y2 <= y1` - cannot have an IoU, and the first version of this
    function let it through: zero area makes `union` zero, `iou` was then defined as
    `0.0`, and `0.0 <= threshold` *kept* it. Measured:

        nms([[5, 5, 5, 5]], [0.9], 0.45)                  -> [0]     (was)
        nms([[100, 100, 0, 0]], [0.9], 0.45)              -> [0]     (inside-out)
        nms([[5,5,5,5],[0,0,10,10]], [0.9, 0.8], 0.45)    -> [0, 1]  (degenerate survived)

    That is not a theoretical concern. `_cxcywh_to_xyxy` on a `w` or `h` of zero - which
    a YOLO head can emit for a degenerate anchor - produces exactly a zero-size box, and
    it would have reached the drawer as a zero-area or inside-out rectangle drawn at
    whatever confidence the anchor carried.

    So degenerate indices are removed before the greedy loop, and the returned list is a
    **subset of the input indices** in which every entry has positive area. It is not a
    permutation of the input, and a caller that needs a one-to-one mapping with its own
    arrays must account for that.
    """
    threshold = _require_threshold("iou_threshold", iou_threshold)
    box_array = np.asarray(boxes, dtype=np.float32)
    score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    # Arity is checked BEFORE the empty short-circuit. A (0, 3) or (0,) array is empty
    # and malformed at once, and the order below decides which complaint it gets: the
    # first version returned [] for both, so a caller passing the wrong width got a
    # plausible empty result instead of being told the array was malformed.
    if box_array.ndim != 2 or box_array.shape[1:] != (4,):
        raise ValueError(f"boxes must be an (N, 4) xyxy array; got {box_array.shape!r}")
    if box_array.shape[0] == 0:
        return []
    if box_array.shape[0] != score_array.shape[0]:
        raise ValueError(
            f"boxes and scores must be the same length; got {box_array.shape[0]} and "
            f"{score_array.shape[0]}"
        )
    # A negative or zero area cannot have an IoU, so it is not a box: drop those rows
    # before suppression rather than letting them survive as iou 0.0.
    widths = box_array[:, 2] - box_array[:, 0]
    heights = box_array[:, 3] - box_array[:, 1]
    usable = np.flatnonzero((widths > 0.0) & (heights > 0.0))
    if usable.size == 0:
        return []
    box_array = box_array[usable]
    score_array = score_array[usable]
    areas = widths[usable] * heights[usable]
    order = np.argsort(-score_array, kind="stable")
    keep: list[int] = []
    while order.size:
        best = int(order[0])
        keep.append(int(usable[best]))
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
