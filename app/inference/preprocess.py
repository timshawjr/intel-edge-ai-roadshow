"""Letterbox preprocessing and the coordinate recovery that goes with it.

This module exists because of one specific, invisible failure: **a letterbox that pads
and scales the image but does not undo both when mapping a detection back to the
frame.** On a centred object the error is a few pixels and nobody can see it. On
anything near an edge it is tens or hundreds of pixels, and the booth draws a
confident box on the wrong object. Every geometry value here is therefore derived once,
stored on a frozen record, and inverted by an exact algebraic inverse - never by
re-deriving a scale, and never by clamping.

The transform, in full, with nothing implicit:

    scale  = min(dst_w / src_w, dst_h / src_h)      # uniform, so no distortion
    scaled = (round(src_w * scale), round(src_h * scale))
    pad    = (dst_w - scaled_w) // 2, (dst_h - scaled_h) // 2   # top-left offset
    to_model(x, y)  = (x * scale + pad_x, y * scale + pad_y)
    to_source(x, y) = ((x - pad_x) / scale, (y - pad_y) / scale)

`pad` is a **floor** of the leftover, not the `int(round(...))` the plan's prose
suggests, so that `2 * pad + scaled <= dst` holds for every aspect ratio. Rounding can
hand back half a pixel too much and push the right or bottom edge of the image off the
canvas. Floor division cannot, and the remainder is simply pad at the bottom/right,
which is invisible for a uniform pad. Every value the plan's tests pin is unchanged by
this: 320/2 = 160 and 280/2 = 140 either way.

Measured first-hand on this machine, through this module's own rule:

| source | destination | scale | scaled | pad (x, y) |
|---|---|---|---|---|
| 1920x1080 | 640x640 | 0.333333 | 640x360 | (0, 140) |
| 1280x720  | 640x640 | 0.5      | 640x360 | (0, 140) |
| 200x100   | 640x640 | 3.2      | 640x320 | (0, 160) |
| 100x200   | 640x640 | 3.2      | 320x640 | (160, 0) |

**RULING 24, 2026-10-01: the pad band is filled with mid-grey `114`, not zeros.**

The plan specified zeros, and the first version of this module implemented that. It was
then measured, and the zeros were a train/serve distribution shift that the export was
converted *against*: this model came from the Ultralytics pipeline, which letterboxes
with 114. On a 1920x1080 frame into 640x640, 21.88% of the canvas is pad, and filling it
with 0 rather than 114 moves the tensor by up to 0.447 in `float32` NCHW (114/255 =
0.447; the reviewer's uint8-domain figure of 368.11 is the same difference unscaled).

`pad_value` is therefore a parameter defaulting to `114`, and it is recorded **on the
`Letterbox`** so the pixel buffer and the record cannot disagree - the same discipline
as `scaled_w`/`scaled_h`. `0` stays reachable so the A/B remains runnable, because the
honest summary of what was measured is narrow:

- The raw score array moves materially (max abs diff 0.806 across four frames).
- The number of anchors above 0.25 moves (15 -> 20 on one frame).
- The **decoded post-NMS detections were identical on all four frames tested.**

So the zeros demonstrably change what the model computes, and demonstrably did not
change the top detections on the frames that were checked. That second fact is luck of
those frames - large, central, high-contrast objects - and it is **not** evidence that
zeros are safe. What the change removes is the distribution shift the export was
converted under. **No claim is made here that 114 improves detection quality**; there is
no measurement of that, and the A/B detection-set delta tool is M1 work.

**Two more things this module does not do, deliberately.**

- *It does not clamp.* `to_source` is affine and unclamped, so a box in the pad band
  comes back as a negative or over-range source coordinate, which is the truth. Silently
  clamping here would destroy the information a consumer needs to decide whether a
  detection is genuinely at the frame edge. Clipping is a display decision and belongs
  to whoever draws the box.
- *It does not use OpenVINO.* Nothing here needs a device, so the whole module is
  unit-testable on a machine with no NPU. The resize uses `cv2`, which is already a
  pinned dependency (`opencv-python==5.0.0.93`) and is the same library the frame
  reader uses; a hand-rolled bilinear resampler would add a second, subtler class of
  off-by-one indexing bug to a module whose entire job is to be exactly right about
  geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

__all__ = ["DEFAULT_PAD_VALUE", "Letterbox", "letterbox_params", "preprocess_bgr"]

Box = tuple[float, float, float, float]

#: The value the pad band is filled with. 114 is what the Ultralytics training and
#: export pipeline letterboxes with, and this export came from that pipeline - see
#: RULING 24 in the module docstring. It is a named constant rather than a bare literal
#: in a signature because it is a decision with a measured consequence, and a test pins
#: that `preprocess_bgr` actually defaults to it.
DEFAULT_PAD_VALUE: Final[int] = 114


def _check_pad_value(pad_value: int) -> int:
    """A pad value that is one uint8 pixel level, or a refusal naming the range.

    A `bool` is refused because `True` is an `int` in Python and `True` as a pixel level
    is a bug wearing a valid-looking value.
    """
    if not isinstance(pad_value, int) or isinstance(pad_value, bool):
        raise TypeError(
            f"pad_value must be an int pixel level in [0, 255]; got {pad_value!r}. A "
            f"float would be silently truncated and a bool is an int in Python, so both "
            f"are refused rather than coerced."
        )
    if not 0 <= pad_value <= 255:
        raise ValueError(
            f"pad_value must be a uint8 pixel level in [0, 255]; got {pad_value!r}. The "
            f"canvas is uint8, so anything outside that range cannot be written to it."
        )
    return pad_value



@dataclass(frozen=True)
class Letterbox:
    """One letterbox operation, and the exact inverse of its coordinate mapping.

    `pad_x`/`pad_y` are the **top-left** offsets at which the scaled image was placed,
    not a per-side margin. That is the only reading under which `to_model` and
    `to_source` are each other's inverse, and it is what `preprocess_bgr` actually
    does to the pixel buffer.

    `scaled_w`/`scaled_h` are derived, not stored: they are `int(round(src * scale))`
    clamped to the destination, and they are the one place that expression exists. If
    the geometry and the resize were computed separately they could disagree by a
    pixel, and a one-pixel disagreement is a one-pixel error on every drawn box.

    `pad_value` **is** stored, for the same reason: it is a fact about the pixel buffer
    rather than about the geometry, and a record that did not carry it would let a
    consumer reconstruct the transform and get a different tensor than the one the
    model was shown. Ruling 24 - it is 114, not 0.
    """

    scale: float
    pad_x: int
    pad_y: int
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    pad_value: int = DEFAULT_PAD_VALUE

    def __post_init__(self) -> None:
        if not (self.scale > 0.0) or not np.isfinite(self.scale):
            raise ValueError(
                f"Letterbox.scale must be a positive finite number; got {self.scale!r}. "
                f"to_source divides by it, so zero would be a ZeroDivisionError on the "
                f"first detection instead of a clear refusal here."
            )
        for name in ("src_w", "src_h", "dst_w", "dst_h"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"Letterbox.{name} must be a positive int; got {value!r}"
                )
        _check_pad_value(self.pad_value)

    @property
    def scaled_w(self) -> int:
        """Width the source is resized to, before padding."""
        return _scaled_extent(self.src_w, self.scale, self.dst_w)

    @property
    def scaled_h(self) -> int:
        """Height the source is resized to, before padding."""
        return _scaled_extent(self.src_h, self.scale, self.dst_h)

    @property
    def box_coordinates(self) -> str:
        """How this transform expects a box to be written, for diagnostics only."""
        return "xyxy in model pixels; to_model/to_source take and return that form"

    def to_model(self, box: Box) -> Box:
        """Source-pixel box -> model-pixel box. `x * scale + pad_x`, axis by axis."""
        x1, y1, x2, y2 = _as_box(box)
        return (x1 * self.scale + self.pad_x, y1 * self.scale + self.pad_y,
                x2 * self.scale + self.pad_x, y2 * self.scale + self.pad_y)

    def to_source(self, box: Box) -> Box:
        """Model-pixel box -> source-pixel box. The exact algebraic inverse.

        Unclamped, on purpose - see the module docstring.
        """
        x1, y1, x2, y2 = _as_box(box)
        return ((x1 - self.pad_x) / self.scale, (y1 - self.pad_y) / self.scale,
                (x2 - self.pad_x) / self.scale, (y2 - self.pad_y) / self.scale)


def _scaled_extent(src: int, scale: float, dst: int) -> int:
    """`src * scale` as a whole number of pixels, never 0 and never past `dst`."""
    return max(1, min(int(dst), int(round(int(src) * float(scale)))))


def _as_box(box: Box) -> Box:
    """Four floats out, with the arity checked so a bad call fails here and not later."""
    try:
        values = tuple(float(v) for v in box)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"a box must be four numbers, got {box!r}") from exc
    if len(values) != 4:
        raise ValueError(f"a box must be (x1, y1, x2, y2), got {box!r}")
    return values  # type: ignore[return-value]


def letterbox_params(src_w: int, src_h: int, dst_w: int, dst_h: int,
                     pad_value: int = DEFAULT_PAD_VALUE) -> Letterbox:
    """The letterbox for one source/destination pair: uniform scale, centred, recorded.

    The scale is `min` of the two axis ratios, so the whole image fits and the aspect
    ratio is preserved. The leftover on each axis is split evenly and the **top-left**
    half is the recorded pad; the image is placed there, which is what makes the
    single-value pad a true inverse rather than an approximation of one.

    `pad_value` is recorded rather than applied here - this function describes the
    transform, and `preprocess_bgr` is what writes pixels - so that a caller can build
    the record without a frame and get exactly the record the array was built with.
    """
    for name, value in (("src_w", src_w), ("src_h", src_h),
                        ("dst_w", dst_w), ("dst_h", dst_h)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"{name} must be a positive int; got {value!r}. A non-positive extent "
                f"has no letterbox, and a zero source extent is a ZeroDivisionError "
                f"rather than a refusal."
            )
    _check_pad_value(pad_value)
    scale = min(dst_w / src_w, dst_h / src_h)
    scaled_w = _scaled_extent(src_w, scale, dst_w)
    scaled_h = _scaled_extent(src_h, scale, dst_h)
    return Letterbox(
        scale=scale,
        pad_x=(dst_w - scaled_w) // 2,
        pad_y=(dst_h - scaled_h) // 2,
        src_w=src_w,
        src_h=src_h,
        dst_w=dst_w,
        dst_h=dst_h,
        pad_value=pad_value,
    )


def preprocess_bgr(bgr: np.ndarray, dst_w: int, dst_h: int,
                   pad_value: int = DEFAULT_PAD_VALUE) -> tuple[np.ndarray, Letterbox]:
    """One BGR `uint8` frame -> one `float32` NCHW RGB tensor in `[0, 1]`, plus the record.

    Batch size is 1 and is not a parameter: the NPU rejects dynamic shapes (AGENTS.md
    rule 3), so the static leading 1 is the whole of the contract and a caller that
    wants more must loop.

    BGR in, RGB out, because that is the order the pinned model was converted with;
    `np.ascontiguousarray` after the transpose is not decoration - the transpose of a
    sliced array is a view with non-standard strides, and some inference backends read
    it as if it were dense.

    The pad band is filled with `pad_value`, which defaults to **114** (Ruling 24, module
    docstring) rather than the zeros the plan specified. `0` is still reachable so the
    A/B stays runnable, and the value that was used is on the returned `Letterbox`.

    Fails closed on anything it cannot interpret without guessing: a non-`HxWx3` shape,
    a dtype that does not say which of `[0,1]` and `[0,255]` the numbers are in, or a
    `pad_value` that is not a uint8 pixel level.
    """
    array = np.asarray(bgr)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"preprocess_bgr expects an HxWx3 BGR frame; got shape {array.shape!r} "
            f"(ndim={array.ndim}). A grayscale, BGRA, batched or flattened array is "
            f"refused rather than coerced, because the coercion that would make it run "
            f"is the one that would put the channels in the wrong order."
        )
    if array.dtype != np.uint8:
        raise TypeError(
            f"preprocess_bgr expects uint8 pixel values; got {array.dtype!r}. A float "
            f"frame could be in [0,1] or in [0,255] and there is no measurement here "
            f"that would tell the two apart, so the caller has to say which."
        )
    _check_pad_value(pad_value)
    src_h, src_w = int(array.shape[0]), int(array.shape[1])
    letterbox = letterbox_params(src_w, src_h, dst_w, dst_h, pad_value=pad_value)

    # INTER_LINEAR, not INTER_NEAREST: this is an upscale on this machine (1920x1080 ->
    # 640 wide) and nearest-neighbour blockiness is visible on a booth screen.
    resized = cv2.resize(
        bgr,
        (letterbox.scaled_w, letterbox.scaled_h),
        interpolation=cv2.INTER_LINEAR,
    )
    # `np.full`, not `np.zeros`: the band is 114 by default. Filling it through the
    # record's own field is deliberate - a literal here would be a second place the
    # value lives, which is how the array and the record come to disagree.
    canvas = np.full((dst_h, dst_w, 3), letterbox.pad_value, dtype=np.uint8)
    canvas[letterbox.pad_y:letterbox.pad_y + letterbox.scaled_h,
           letterbox.pad_x:letterbox.pad_x + letterbox.scaled_w] = resized

    rgb = canvas[:, :, ::-1]                       # BGR -> RGB
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return chw[None, ...], letterbox
