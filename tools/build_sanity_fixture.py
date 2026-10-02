"""The annotated detection sanity fixture: build it, and check the model against it.

Run from the repo root:

    .venv\\Scripts\\python.exe tools\\build_sanity_fixture.py --extract-frame 0
    .venv\\Scripts\\python.exe tools\\build_sanity_fixture.py --device CPU
    .venv\\Scripts\\python.exe tools\\build_sanity_fixture.py --device CPU --device GPU --device NPU

**Why the annotation is a human judgement and not a recorded detection.**
`app/inference/decode.py` makes four load-bearing readings of the pinned export's
`[1, 84, 8400]` output: the features are on axis 1, the first four values per anchor are
`cx, cy, w, h` rather than `xyxy`, the class scores are already activated, and the
letterbox inverse maps a model box back to source pixels. Every one of those can be wrong
and still return confident, plausible, wrong detections - a transposed class block renames
every object, a missing `w/2` shifts every box, a missed inverse displaces the edges. The
decoder's own unit tests cannot catch any of it, because every tensor they decode is one
this project fabricated.

**Which of those four this fixture actually checks, measured by injecting mutants.**
A first version of this docstring claimed all four. That was an over-claim, and a mutant
sweep found two of the four were not covered at all:

| reading | mutant injected | caught? |
|---|---|---|
| features on axis 1 | slice axis 1, never transpose | **yes** |
| `cx, cy, w, h` not `xyxy` | no `w/2, h/2` subtraction; read as `xyxy` | **yes** |
| letterbox inverse applied | `to_source` made the identity | **yes** |
| letterbox pad band at 114 | pad 255 | **NO - see below** | (pad 0 IS caught) |
| scores already activated | a second sigmoid | **was not checked; NOW IS** |
| `w` and `h` in the right order | swap them | **NO - not separable here, see `_aspect`** |

**The activation hole, and the one fix for it.** A second activation pushes every score
into `(0.5, 0.73)`, so all 8400 anchors clear the conf threshold and the decoder returns
**1205 boxes** where the frame holds four objects. The original fixture went green,
because IoU-matching only ever looks at the best-matching box per annotation and never
asks how many boxes arrived - and 1205 rectangles on a booth screen is a catastrophic,
silent failure. A disabled NMS (19 boxes) survived the same way.

`max_detections` closes both. It is how many boxes the frame can possibly justify - two
named objects, the basket, the counter, the trolley's lower half, a second shopper's
feet - and it is a judgement about **the picture**, recorded in the annotation, which is
what keeps the fixture non-circular. It also makes the `oven` false positive on the
counter visible as a box rather than something a reader has to notice.

**The `w`/`h` hole is real and is NOT closed here.** A transposition mirrors each box,
both annotated objects are near-square, and a mirrored box overlaps the original well
above any sane IoU. An aspect-ratio check was implemented, measured, and **rejected**:
against this annotation the correct decoder is *further* from the annotated aspect than
the mutant is, so the check failed a healthy run and passed the fault. The full numbers
and the algebra are in `_aspect`. A check that cries wolf is worse than a documented hole,
because the response to a false failure is to loosen the check until the real fault gets
through. This fixture does not catch a transposed `w`/`h`, and says so.

**A third hole the sweep found, which is about the frame rather than the decoder:** the
pad-band mutant at `255` survives on frame 0 - filling the band with `255` instead of
`114` still yields 2/2 here, because a white band is closer to the scene than a black one
and both annotated objects sit well inside the image. **The `0` mutant does NOT survive:**
measured, `pad_value=0` returns **1/2** with the apple **absent from the frame entirely**,
and this fixture **FAILS**. An earlier version of this docstring claimed that `0` and `255`
both returned 2/2; that was wrong for `0`, and it understated this fixture's sensitivity to
the pad decision. The correction is recorded here because it is the point: frame 0 *is* one
of the frames where the pad value changes a detection, which makes it better evidence for
the `114` default than the four benign frames Task 4 happened to try. Only the `255` variant
needs a different frame to catch, which is M1's job to find.

The only thing that can catch a decoder fault is an annotation made by somebody **looking
at the frame**. So the workflow here is deliberately ordered:

1. `extract_frame` writes `retail_000.jpg` from the pinned clip at native resolution.
2. A human opens that JPEG and writes the boxes in `retail_sanity.json` **by eye**, in
   source pixels, with a `notes` field recording the reasoning.
3. Only then is the model run, and the annotation is compared against it.

**A disagreement is a finding, not a nuisance.** If the model does not agree with a box
somebody drew by looking at a real frame, the first thing to suspect is the decoder. The
annotation is not adjusted to make the model agree; that would turn the fixture into
"the model agrees with itself", which is the one thing it exists to disprove. This tool
therefore never writes the fixture JSON. `evaluate_fixture` opens it read-only and returns
a record; `--extract-frame` writes the JPEG and refuses to touch the annotation.

**Measured on this machine while writing this, and the reason the fetch calls look the
way they do** (`openvino==2026.4.0-22959-99c81491cc3-releases/2026/4`, CPU plugin):

| call | key | result |
|---|---|---|
| `req.infer({0: blob})` | int port index | **accepted** |
| `req.infer({"x": blob})` | input port name | **accepted** |
| `req.infer({model.inputs[0]: blob})`, `model` = the **IR** | IR input port object | `TypeError: Incompatible key type for input` |
| `req.infer({model.input(0): blob})`, `model` = the **IR** | IR `ConstOutput` | `TypeError: Incompatible key type for input` |
| `req.infer({compiled.inputs[0]: blob})` | **compiled** input port object | **accepted** |
| `req.get_output_tensor(0)` | int port index | **accepted**, `[1, 84, 8400]` |
| `req.get_tensor(model.outputs[0])` | output port object | **accepted**, `[1, 84, 8400]` |
| `req.get_tensor(0)` | int port index | `TypeError: incompatible function arguments` |

**The distinction is which model, not which key type.** A first version of this note said
the port object "raises" and that the int index was the only key type that works; both
were wrong in the same way, and the difference matters because the obvious reading
("never pass a port object") is false. Passing the port object works when it comes from
the **`ov.CompiledModel`**; it is the **IR** model's port object that is refused, with
`Incompatible key type for input`. This module holds the IR (that is the only graph
`describe_outputs` can walk), so using the int index is the choice that works from either.
`config/model_io/yolo11n-fp16.json`'s measured `fetch_api` block records the IR-model
results and does not distinguish the two models, so it reads as a blanket prohibition on
port objects; amending it is a Task 5 file and is out of scope here.

The output port of this export has no name at all (`get_names() == set()`), so a
name-based call is not available on the output side, and the tensors are handed to the
decoder under `port_key(0)` - the string `"0"` - which is this project's own key, assigned
because it is the one identifier that exists for an unnamed port.

`describe_outputs` is given the **`ov.Model` read from the IR**, never the
`ov.CompiledModel`: a compiled model cannot be walked, so `has_embedded_nms` would be a
guess, and a guess there decides whether detections are suppressed at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openvino as ov  # noqa: E402

from app.inference.decode import (  # noqa: E402
    Detection,
    ModelOutputSpec,
    decode_detections,
    describe_outputs,
    port_key,
)
from app.inference.devices import DEVICE_IDS, is_device, normalise_execution_devices  # noqa: E402
from app.inference.preprocess import preprocess_bgr  # noqa: E402

MODELS_JSON = REPO_ROOT / "config" / "models.json"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "retail_sanity"
FIXTURE_JSON = FIXTURE_DIR / "retail_sanity.json"
FIXTURE_IMAGE = "retail_000.jpg"

MODEL_ID = "yolo11n-fp16"
CLIP_ID = "retail_checkout"

#: The decode thresholds the fixture is evaluated at, and the same pair
#: `tests/test_decode.py` uses. Fixed before any model was run, so that neither the
#: threshold nor `min_score` could afterwards be nudged to whatever the model happened
#: to produce.
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

#: The IoU a detection must reach against the hand-drawn box to count as a match. The
#: brief specifies 0.5. It is a real tolerance and not a formality: a person drawn by
#: eye at 1920x1080 and the same person drawn by the model in 640x640 model space and
#: mapped back will not agree to the pixel, and demanding more would make the fixture
#: fail on annotation jitter rather than on a decoder fault.
MATCH_IOU = 0.5

#: JPEG quality for the stored frame. Lossy, and recorded, because the fixture's boxes
#: were drawn on *this* file: a different quality is a different picture.
JPEG_QUALITY = 95


@dataclass(frozen=True)
class FixtureOutcome:
    """The result of checking one device against the hand annotation.

    `passed` is the **verdict**, and it is stricter than `matched == expected`. It
    requires all four of: every expectation matched, the detection count within the
    annotation's `max_detections` bound, the matched pairs' aspect ratios within the
    annotation's `max_aspect_delta`, and the run placed on the device that was asked for.

    That last clause was missing from the first version. A device that silently fell back
    to the CPU was printing `PASS: 2/2` and exiting 0, with `placed_on_requested=False`
    two lines above it, while `tools/qualify_m0.py` - reading the same AGENTS.md rule 5 -
    correctly failed. Two tools disagreeing about the same rule is how the rule stops
    meaning anything.

    The first version also enforced `passed == (matched == expected)` **in this
    constructor**, on the reasoning that `passed` is derived and must not be stampable.
    That invariant became false the moment the verdict grew past matching, and it then
    raised `ValueError: passed=False contradicts matched=2 against expected=2` on a
    correct, honest run. `matched <= expected` is still enforced - a record claiming more
    matches than annotations is a bug in the matcher and would render as a better result
    than the annotation allows. The rest of the verdict is `evaluate_fixture`'s to
    compute, and its reasons are written into `details` so the verdict is never a bare
    boolean.

    `details` carries the evidence - every detection the decoder produced, every
    expectation and what matched it, the `EXECUTION_DEVICES` reading the run was placed
    by, and the `VERDICT` line naming every reason it failed. It is a tuple so the
    outcome cannot be edited after the fact.
    """

    passed: bool
    matched: int
    expected: int
    details: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", tuple(str(d) for d in self.details))
        if not isinstance(self.passed, bool):
            raise TypeError(f"passed must be a bool; got {self.passed!r}")
        if self.matched > self.expected:
            raise ValueError(
                f"matched ({self.matched}) cannot exceed expected ({self.expected}); a "
                f"record claiming more matches than annotations is a bug in the "
                f"matcher, and it would render as a better result than the annotation "
                f"allows."
            )
        if self.passed and self.matched != self.expected:
            raise ValueError(
                f"passed=True with matched={self.matched} against expected="
                f"{self.expected}. A pass must at least have matched every annotated "
                f"object; the remaining verdict conditions can only make it stricter."
            )
        if self.passed and not any(d.startswith("VERDICT PASS") for d in self.details):
            raise ValueError(
                "passed=True with no 'VERDICT PASS' line in details. A pass a reader "
                "cannot check the reasons for is the kind of record this project exists "
                "not to produce."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "matched": self.matched,
            "expected": self.expected,
            "details": list(self.details),
        }


# --------------------------------------------------------------------------- config


def _models_config() -> dict[str, Any]:
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))


def pinned_model_entry() -> dict[str, Any]:
    entry = next(m for m in _models_config()["models"] if m["id"] == MODEL_ID)
    if entry["role"] != "detector":
        raise ValueError(f"{MODEL_ID} has role {entry['role']!r}, not 'detector'")
    return entry


def pinned_clip_entry() -> dict[str, Any]:
    entry = next(m for m in _models_config()["media"] if m["id"] == CLIP_ID)
    return entry


def ir_path() -> Path:
    return REPO_ROOT / pinned_model_entry()["ir_path"]


def class_names() -> tuple[str, ...]:
    """The pinned label list, in the order the model indexes its class block.

    Read from `labels.txt` and never from a hardcoded tuple, so a re-export with a
    different label set cannot be decoded against the old names.
    """
    path = REPO_ROOT / pinned_model_entry()["labels_path"]
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return tuple(name for name in lines if name)


def read_pinned_model() -> ov.Model:
    """The IR as an `ov.Model`, which is the only thing `describe_outputs` accepts.

    Read from the recorded `ir_path`, and the file's bytes are checked against the
    SHA-256 `config/models.json` records, so a swapped model is refused here rather
    than decoded as though it were the pinned one.
    """
    entry = pinned_model_entry()
    path = REPO_ROOT / entry["ir_path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(
            f"{path} hashes to {digest}, not the {entry['sha256']} recorded in "
            f"config/models.json. Refusing to describe or decode a model that is not "
            f"the pinned one; run tools/verify_sources.py."
        )
    return ov.Core().read_model(str(path))


def describe_pinned_model() -> ModelOutputSpec:
    """`describe_outputs` over the IR, with the pinned class count.

    The class count comes from `config/models.json`, never from the tensor: the tensor
    does not say how many of its 84 values are class scores, which is the entire
    question `classify_layout` is being asked.
    """
    model = read_pinned_model()
    return describe_outputs(model, num_classes=int(pinned_model_entry()["num_classes"]))


# ------------------------------------------------------------------------- fixture


def load_fixture() -> dict[str, Any]:
    """The annotation, read. Never written by this module."""
    if not FIXTURE_JSON.exists():
        raise FileNotFoundError(
            f"{FIXTURE_JSON} does not exist. Build the frame with "
            f"`tools/build_sanity_fixture.py --extract-frame N`, then annotate it by "
            f"eye. This tool will not create the annotation: an annotation it wrote "
            f"would be its own output checked against itself."
        )
    return json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))


def fixture_image_path(fixture: dict[str, Any]) -> Path:
    return FIXTURE_DIR / fixture["image"]


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two `xyxy` boxes. `0.0` for a degenerate box, which cannot have one.

    The same refusal `app.inference.decode.nms` makes: a zero-area box would make the
    union zero, and `0.0 <= threshold` would *pass* it. So a degenerate annotation
    cannot quietly satisfy a match.
    """
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _aspect(box: Sequence[float]) -> float:
    """A box's width / height. `0.0` for a degenerate one, which has no aspect.

    **Reported by `evaluate_fixture`; deliberately NOT part of the verdict.** The first
    version of this module's author proposed an aspect check precisely to catch a decoder
    that read the four values as `cx, cy, h, w`, and it does not work on this fixture.
    Measured, baseline against a `w`/`h`-swap mutant, relative difference from the
    annotated box's aspect:

    | object | annotated | baseline decode | swap mutant |
    |---|---|---|---|
    | person | 0.9468 | 0.9754 (delta **0.0303**) | 1.0252 (delta **0.0827**) |
    | apple | 0.8152 | 1.1728 (delta **0.4387**) | 0.8527 (delta **0.0459**) |

    The apple is **inverted**: the correct decoder is 0.4387 away from the annotation and
    the mutant is only 0.0459 away, so any threshold tight enough to catch the mutant
    fails the correct run, and any threshold loose enough to pass the correct run passes
    the mutant. The reason is that this annotation's apple box is deliberately *tighter*
    than the model's on three of four sides (independently confirmed against the pixels:
    y1 878 vs 915, x2 150 vs 176), and a tight box is a poor estimate of the object's
    aspect - so the honest disagreement about the fruit's extent is larger than the
    transposition's own effect.

    Nor can anything else separate them. A pure `w`/`h` swap preserves `cx`, `cy` and
    `w * h` exactly - measured: 422304.4 px^2 both ways for the person, 26352.4 px^2 both
    ways for the apple - so every centre- and area-based statistic is identical to the
    correct decode. Aspect against an independent estimate of the object's true aspect is
    the *only* candidate, and this fixture cannot supply one.

    So the transposition is a **known, uncaught blind spot** and the honest thing is to
    say so rather than ship a check that fails a healthy run. The reasoning is recorded
    here, in the annotation's `known_blind_spots`, and in the M0 report so the next person
    does not re-derive it. A fixture whose guard cries wolf on a correct decoder is worse
    than one with a documented hole, because the response to a false failure is to loosen
    the guard until the real fault passes too.

    Closing it needs a different frame, not a different threshold: an object with an
    aspect far from 1, annotated at the same tightness as the model draws it.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if height <= 0.0:
        return 0.0
    return width / height


# ---------------------------------------------------------------------- frame build


def extract_frame(frame_index: int, out_path: Path | None = None) -> Path:
    """Write one frame of the pinned clip at native resolution, and record how.

    `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, n)` then `read()`. That is a **seek**, and
    a seek into a compressed video is not guaranteed to land on `n`, so the counter is
    captured either side of the read and the decoded frame's own SHA-256 printed.
    **Measured on this clip: the seek is exact, and the counter after a read is `n + 1`,
    not `n`.** `CAP_PROP_POS_FRAMES` is the index of the *next* frame to be decoded, not
    the one just decoded:

        sequential, fresh handle:   before=0 read -> after=1
                                    before=1 read -> after=2
                                    before=2 read -> after=3
        seek to 0, then read:       before=0 read -> after=1, and the decoded frame is
                                    **byte-identical** to the sequential read of frame 0
                                    (sha256 of the raw BGR buffer starts 20dd0662)
        the same for seeks to 1 and 2, again byte-identical to sequential; no read
        failed at 0, 1, 2, 60, 300, 1200 or 3386 - the last of 3387 frames.

    The off-by-one is worth stating because the obvious check is wrong. Asserting
    `counter_after == frame_index` reports **every** frame as a seek failure, and the
    natural reaction to that - "the seek is broken, fall back to a sequential read" -
    would be a repair for a bug that is not there, on a 3387-frame clip. So the counter
    is read either side of the read, the frame's own SHA-256 is printed, and a
    violation of `before == n and after == n + 1` is reported rather than absorbed.

    Writes **only the JPEG**. It does not create, touch or read-modify-write the
    annotation: see the module docstring for why.
    """
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise ValueError(f"frame_index must be a non-negative int; got {frame_index!r}")
    clip = pinned_clip_entry()
    path = REPO_ROOT / clip["path"]
    target = Path(out_path) if out_path is not None else FIXTURE_DIR / FIXTURE_IMAGE
    target.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index >= total:
            raise ValueError(
                f"frame_index {frame_index} is past the end of {clip['path']}, which "
                f"cv2 reports as {total} frames (config/models.json records "
                f"{clip['measured_frames']})."
            )
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        before = int(round(float(capture.get(cv2.CAP_PROP_POS_FRAMES))))
        ok, frame = capture.read()
        after = int(round(float(capture.get(cv2.CAP_PROP_POS_FRAMES))))
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(
            f"cv2.VideoCapture.read() returned {ok!r} for frame {frame_index} of "
            f"{clip['path']}. A frame that will not decode is not a fixture."
        )
    expected = (clip["measured_height"], clip["measured_width"])
    if frame.shape[:2] != expected:
        raise RuntimeError(
            f"frame {frame_index} decoded at {frame.shape[1]}x{frame.shape[0]}, but "
            f"config/models.json records this clip as "
            f"{clip['measured_width']}x{clip['measured_height']}. The fixture's boxes "
            f"are in source pixels, so the frame must be at native resolution."
        )
    # The frame's own content hash, so a re-extraction can be checked for *equality*
    # rather than for "it ran without raising". A fixture whose picture has drifted is
    # a fixture whose boxes no longer describe anything.
    frame_sha256 = hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
    if not cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY),
                                            JPEG_QUALITY]):
        raise RuntimeError(f"cv2.imwrite failed for {target}")
    print(f"wrote {target}")
    print(f"  requested frame_index   {frame_index}")
    print(f"  counter before / after  {before} / {after}   "
          f"(after is the NEXT frame; expected {frame_index + 1})")
    print(f"  decoded shape           {frame.shape[1]}x{frame.shape[0]} "
          f"(native, checked against config/models.json)")
    print(f"  raw BGR sha256          {frame_sha256}")
    print(f"  jpeg quality            {JPEG_QUALITY}")
    if before != frame_index or after != frame_index + 1:
        print(f"  WARNING: the seek did not land where it was aimed "
              f"(before={before}, after={after}, asked for {frame_index}). Record all "
              f"three in the fixture's provenance; the picture may not be frame "
              f"{frame_index}.")
    print(f"  copy these into {FIXTURE_JSON.name}'s provenance block.")
    return target


# ------------------------------------------------------------------------ inference


def pinned_input_shape() -> tuple[int, int, int, int]:
    """The pinned model's `[N, C, H, W]`, from `config/models.json`.

    **Not** from `spec.shapes`, and the first version of this function got that wrong in
    a way worth recording. `ModelOutputSpec.shapes` holds the **output** ports, and this
    model's output is `[1, 84, 8400]` - so `spec.shapes[port_key(0)][-1]` is `8400`, and
    letterboxing the frame to 8400x8400 produced a blob the runtime refused on all three
    plugins with `The input tensor size is not equal to the model input type: got
    [1,3,8400,8400] expecting [1,3,640,640]`. The model input is `[1, 3, 640, 640]` in
    `config/models.json` and the spec does not carry it, so that is where it is read from.
    The error was legible and the device reported as failed rather than as zero, which is
    the behaviour that made the bug findable at all.
    """
    shape = tuple(int(v) for v in pinned_model_entry()["input_shape"])
    if len(shape) != 4:
        raise ValueError(f"config/models.json input_shape must be [N,C,H,W]; got {shape!r}")
    return shape  # type: ignore[return-value]


def detect(core: ov.Core, spec: ModelOutputSpec, model: ov.Model, image: np.ndarray,
           device: str, conf_threshold: float = CONF_THRESHOLD,
           iou_threshold: float = IOU_THRESHOLD) -> tuple[list[Detection], tuple[str, ...]]:
    """Compile on `device`, infer once, and decode. Returns detections and placement.

    **One** inference: this is a correctness check, not a benchmark, and a loop here
    would make the fixture's cost scale with nothing. `tools/qualify_m0.py` is the thing
    that runs a hundred requests.

    The letterbox destination is the model's **input** extent - see `pinned_input_shape`
    for why it does not come from the spec.

    `conf_threshold` and `iou_threshold` are **parameters and are used**. The first
    version of this function hardcoded the module constants while `evaluate_fixture`
    printed the caller's values, so `--conf 0.9` on the CLI displayed `conf=0.9`, decoded
    at 0.25, and still emitted the 0.273557 apple with `PASS: 2/2`. A number on screen
    that is not the number in effect is the exact failure this project exists to prevent,
    and it was worse than cosmetic: at a genuine `conf=0.5` the apple is correctly filtered
    out and the fixture should fail, and the flag **hid** that failure. Thresholds are now
    threaded from the caller all the way into `decode_detections`, and
    `test_a_caller_supplied_conf_threshold_actually_filters` pins it.

    `EXECUTION_DEVICES` is read from the request's **own** compiled model, after the
    inference, and normalised through `normalise_execution_devices` - the NPU plugin
    reports a bare string `'NPU'` where CPU and GPU report a list, so the raw property
    is never rendered directly.
    """
    _, _, dst_h, dst_w = pinned_input_shape()
    blob, letterbox = preprocess_bgr(image, dst_w, dst_h)
    compiled = core.compile_model(model, device)
    request = compiled.create_infer_request()
    # Measured on this runtime, and the distinction is WHICH MODEL, not which key type:
    #   infer key = int port index 0      -> accepted, from either the IR or the compiled
    #   infer key = ir_model.inputs[0]    -> TypeError: Incompatible key type for input
    #   infer key = compiled.inputs[0]    -> accepted
    # So the int index is the one key that works from either model, which is what this
    # uses. The port object is fine when it comes from the CompiledModel; it is the IR
    # model's port object that is refused. `config/model_io/yolo11n-fp16.json`'s
    # `fetch_api` block records the IR-model results and does not distinguish the two.
    request.infer({0: blob})
    # ...and takes the int index here too, which `get_tensor(0)` does not.
    tensor = request.get_output_tensor(0)
    outputs = {port_key(0): np.array(tensor.data)}
    placement = normalise_execution_devices(
        request.get_compiled_model().get_property("EXECUTION_DEVICES"))
    detections = decode_detections(spec, outputs, letterbox, conf_threshold,
                                   iou_threshold)
    return detections, placement


def _format_box(box: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(v):.1f}" for v in box) + ")"


def evaluate_fixture(spec: ModelOutputSpec, class_names: Sequence[str],
                     conf_threshold: float, iou_threshold: float, *,
                     device: str = "CPU", core: ov.Core | None = None,
                     match_iou: float = MATCH_IOU) -> FixtureOutcome:
    """Check one device's detections against the hand annotation.

    `spec` is the recorded `ModelOutputSpec` for the pinned model; it is passed in
    rather than rebuilt so a caller can hand over a spec it read from
    `config/model_io/yolo11n-fp16.json` and see the disagreement if the two differ.
    `class_names` indexes the class block - `class_names[class_id]` must be what the
    annotation's `class_name` says, or the fixture is comparing names to ids.

    **A failure is a result, not an exception.** An unavailable device, a compile
    refusal, an inference error: each returns `passed=False` with the runtime's own
    message in `details` and every number left at its unmeasured state. A device that
    could not run is never rendered as a device that ran and found nothing.

    **`passed` requires three things, not one.** Every expectation matched, the
    detection count within the annotation's `max_detections` bound, **and** the run
    placed on the device that was asked for. The placement clause was missing and is the
    more serious of the two additions: a device that silently fell back to the CPU was
    printing `PASS: 2/2` and exiting 0, with `placed_on_requested=False` visible two
    lines above it. That is AGENTS.md rule 5 - `EXECUTION_DEVICES` is the only proof of
    placement - and `tools/qualify_m0.py` already asserted it, so the two tools disagreed
    about the same rule. A fixture that reports success for work the NPU never did is
    worse than one that fails, because a booth will not read the line above the verdict.

    Matching is greedy on IoU, one detection per expectation. Each expectation needs a
    detection of the **same class id**, scoring at least the annotation's `min_score`,
    with IoU at least `match_iou` against the hand-drawn box. The best-IoU pair is taken
    first so the result does not depend on the order the two lists happen to be in.

    Two further things are recorded in the annotation, and they are not the same kind of
    thing. `max_detections` **gates** the verdict: it bounds how many boxes the decoder
    may return, because a second activation applied to scores the export has already
    activated pushes every score into `(0.5, 0.73)`, all 8400 anchors clear the conf
    threshold, and the decoder returns **1205** boxes where there are 4 objects on screen.
    A booth drawing 1205 rectangles is a catastrophic, silent failure, and the original
    fixture went green on it because IoU-matching only ever looks at the best box per
    annotation and never asks how many boxes arrived.

    The matched pairs' **aspect ratios are reported and do not gate anything.** An aspect
    check was written to catch a transposed `w`/`h`, measured, and rejected: against this
    annotation the correct decode is *further* from the annotated aspect than the mutant
    is, so it failed a healthy run and passed the fault. `matched_pairs` therefore appear
    in `details` as an `ASPECT` line marked INFORMATIONAL, and a `w`/`h` transposition is
    a known uncaught blind spot of this fixture. `_aspect` has the numbers and the algebra.
    """
    names = tuple(str(n) for n in class_names)
    if spec.num_classes != len(names):
        raise ValueError(
            f"spec.num_classes is {spec.num_classes} but {len(names)} class names "
            f"were supplied. The class block is indexed by these, so a length "
            f"mismatch would compare an id against the wrong name."
        )
    try:
        fixture = load_fixture()
    except FileNotFoundError as exc:
        return FixtureOutcome(passed=False, matched=0, expected=0,
                              details=(f"no fixture: {exc}",))
    expected_entries = list(fixture.get("expected") or ())
    details: list[str] = [f"device={device} conf={conf_threshold} iou={iou_threshold} "
                         f"match_iou={match_iou}",
                         f"spec: layout={spec.layout!r} shapes={dict(spec.shapes)} "
                         f"has_embedded_nms={spec.has_embedded_nms}"]
    if not expected_entries:
        return FixtureOutcome(passed=False, matched=0, expected=0,
                              details=tuple(details + ["the annotation is empty"]))

    # The recorded class_name must be the one this label list gives that id. Checked
    # before the model runs, so a stale annotation fails as a stale annotation.
    for entry in expected_entries:
        class_id = int(entry["class_id"])
        if not 0 <= class_id < len(names):
            return FixtureOutcome(
                passed=False, matched=0, expected=len(expected_entries),
                details=tuple(details + [f"annotation class_id {class_id} is outside "
                                         f"the {len(names)} pinned labels"]))
        if names[class_id] != entry["class_name"]:
            return FixtureOutcome(
                passed=False, matched=0, expected=len(expected_entries),
                details=tuple(details + [
                    f"annotation says class_id {class_id} is "
                    f"{entry['class_name']!r} but labels.txt says "
                    f"{names[class_id]!r}. The annotation is stale; the model has not "
                    f"been run."]))

    image = cv2.imread(str(fixture_image_path(fixture)))
    if image is None:
        return FixtureOutcome(
            passed=False, matched=0, expected=len(expected_entries),
            details=tuple(details + [f"cv2.imread returned None for "
                                     f"{fixture_image_path(fixture)}"]))
    model = read_pinned_model()
    live_core = core if core is not None else ov.Core()
    try:
        detections, placement = detect(live_core, spec, model, image, device,
                                       conf_threshold, iou_threshold)
    except Exception as exc:  # noqa: BLE001 - a device that cannot run is a result
        return FixtureOutcome(
            passed=False, matched=0, expected=len(expected_entries),
            details=tuple(details + [
                f"FAILED on {device}: {type(exc).__name__}: {exc}",
                "no detections, no placement and no score were produced, and none are "
                "reported above; this device did not run."]))
    placed = is_device(placement, device)
    details.append(f"EXECUTION_DEVICES={list(placement)} "
                   f"placed_on_requested={placed}")
    details.append(f"decoder returned {len(detections)} detection(s)")
    for det in detections:
        # A class id outside the label list is what a mis-read class block produces, and
        # indexing `names` with it raised IndexError and took the whole reporter down
        # with a traceback instead of reporting the mis-decode. A wrong answer deserves a
        # sentence, not a crash.
        label = (names[det.class_id] if 0 <= det.class_id < len(names)
                 else f"<OUT OF RANGE, only {len(names)} labels>")
        details.append(
            f"  detected class_id={det.class_id} name={label!r} "
            f"score={det.score:.6f} box={_format_box(det.box)}")

    # Greedy best-IoU-first pairing. Built from every admissible (expectation,
    # detection) pair, sorted by IoU descending, so the outcome cannot depend on the
    # order the detector happened to return.
    pairs: list[tuple[float, int, int]] = []
    for ei, entry in enumerate(expected_entries):
        for di, det in enumerate(detections):
            if det.class_id != int(entry["class_id"]):
                continue
            if det.score < float(entry["min_score"]):
                continue
            iou = box_iou(entry["box_xyxy"], det.box)
            if iou >= match_iou:
                pairs.append((iou, ei, di))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    taken_e: dict[int, tuple[int, float]] = {}
    taken_d: set[int] = set()
    for iou, ei, di in pairs:
        if ei in taken_e or di in taken_d:
            continue
        taken_e[ei] = (di, iou)
        taken_d.add(di)

    for ei, entry in enumerate(expected_entries):
        label = (f"expected class_id={entry['class_id']} "
                 f"name={entry['class_name']!r} min_score={entry['min_score']} "
                 f"box={_format_box(entry['box_xyxy'])}")
        if ei in taken_e:
            di, iou = taken_e[ei]
            det = detections[di]
            details.append(f"  MATCH {label} <- detection {di} "
                           f"iou={iou:.4f} score={det.score:.6f}")
        else:
            details.append(f"  NO MATCH {label}")
            # Say why, with the closest thing the decoder actually produced. "Not
            # found" and "found, but under a different class" are different bugs.
            best: tuple[float, Detection] | None = None
            for det in detections:
                iou = box_iou(entry["box_xyxy"], det.box)
                if best is None or iou > best[0]:
                    best = (iou, det)
            if best is None:
                details.append("    the decoder returned no detection on this frame "
                               "that overlaps this box at all")
            else:
                det = best[1]
                same = det.class_id == int(entry["class_id"])
                enough = det.score >= float(entry["min_score"])
                label = (names[det.class_id] if 0 <= det.class_id < len(names)
                         else f"<OUT OF RANGE, only {len(names)} labels>")
                details.append(
                    f"    closest detection: class_id={det.class_id} "
                    f"name={label!r} score={det.score:.6f} "
                    f"iou={best[0]:.4f}"
                    f"{'' if same else '  (WRONG CLASS)'}"
                    f"{'' if enough else '  (BELOW min_score)'}"
                    f"{'' if best[0] >= match_iou else '  (IOU BELOW THRESHOLD)'}")

    matched = len(taken_e)
    verdict_reasons: list[str] = []

    # 1. Every annotated object found.
    if matched != len(expected_entries):
        verdict_reasons.append(f"{len(expected_entries) - matched} annotated object(s) "
                               f"not matched")

    # 2. Not a swarm of boxes. A judgement about the picture, recorded in the annotation.
    max_detections = fixture.get("max_detections")
    if max_detections is None:
        verdict_reasons.append(
            "the annotation records no max_detections bound, so a decoder returning a "
            "thousand boxes would pass; the fixture cannot check what it was not told")
    else:
        limit = int(max_detections)
        details.append(f"detection-count bound: {len(detections)} <= {limit} "
                       f"(the annotation says at most {limit} objects on this frame)")
        if len(detections) > limit:
            verdict_reasons.append(
                f"the decoder returned {len(detections)} boxes where the annotation "
                f"says this frame holds at most {limit}")

    # 3. Placed on the device that was asked for. AGENTS.md rule 5.
    if not placed:
        verdict_reasons.append(
            f"the run was placed on {list(placement)}, not on the {device} that was "
            f"requested, so its detections are not {device} detections")

    # 4. Aspect ratio of each matched pair: REPORTED, and deliberately not a verdict
    #    input. See `_aspect` and the module docstring for the measurement that stopped
    #    this from being a gate.
    for ei, entry in enumerate(expected_entries):
        if ei not in taken_e:
            continue
        det = detections[taken_e[ei][0]]
        annotated = _aspect(entry["box_xyxy"])
        found = _aspect(det.box)
        if annotated <= 0.0 or found <= 0.0:
            continue
        details.append(
            f"  ASPECT {entry['class_name']}: annotated {annotated:.4f} vs detected "
            f"{found:.4f}, relative difference {abs(found / annotated - 1.0):.4f} "
            f"(INFORMATIONAL - not part of the verdict; see the module docstring)")

    details.append(f"RESULT matched {matched} of {len(expected_entries)}")
    if verdict_reasons:
        details.append("VERDICT FAIL: " + "; ".join(verdict_reasons))
    else:
        details.append("VERDICT PASS: every annotated object matched, the detection "
                       "count is within the annotation's bound, and the run was placed "
                       "on the requested device. KNOWN BLIND SPOT, not checked here: a "
                       "transposed w/h. See the module docstring - it is not separable "
                       "on this fixture by any measurement.")
    return FixtureOutcome(passed=not verdict_reasons, matched=matched,
                          expected=len(expected_entries), details=tuple(details))


# ----------------------------------------------------------------------------- CLI


def run_devices(devices: Sequence[str], conf: float, iou: float) -> list[dict[str, Any]]:
    spec = describe_pinned_model()
    names = class_names()
    core = ov.Core()
    results: list[dict[str, Any]] = []
    for device in devices:
        print(f"\n=== {device} ===")
        outcome = evaluate_fixture(spec, names, conf, iou, device=device, core=core)
        for line in outcome.details:
            print(f"  {line}" if not line.startswith("  ") else line)
        print(f"  {'PASS' if outcome.passed else 'FAIL'}: "
              f"{outcome.matched}/{outcome.expected}")
        results.append({"device": device, **outcome.as_dict()})
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and check the hand-annotated detection sanity fixture.")
    parser.add_argument("--extract-frame", type=int, metavar="N",
                        help="write frame N of the pinned clip to the fixture image")
    parser.add_argument("--device", action="append", dest="devices",
                        help="device to check; repeatable. Default: all three.")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--json-out", type=Path, default=None,
                        help="also write the per-device outcomes here")
    args = parser.parse_args(argv)

    if args.extract_frame is not None:
        extract_frame(args.extract_frame)
        if not FIXTURE_JSON.exists():
            print(f"\nNext: open {FIXTURE_DIR / FIXTURE_IMAGE}, look at it, and write "
                  f"{FIXTURE_JSON} by hand. Do not run the model first - see the module "
                  f"docstring.")
        return 0

    devices = tuple(args.devices) if args.devices else DEVICE_IDS
    unknown = [d for d in devices if d not in DEVICE_IDS]
    if unknown:
        parser.error(f"unknown device(s) {unknown}; this milestone qualifies "
                     f"{list(DEVICE_IDS)}")
    results = run_devices(devices, args.conf, args.iou)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
