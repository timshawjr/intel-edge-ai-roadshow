"""`tools.capture_ui` - the offscreen capture harness for the M3 DPI gate.

## Why this module exists at all

The demo machine's desktop is **2293x960**, so **a 1920x1080 canvas cannot be displayed on it
at all** - M0 measured the usable height at 960 px. The M3 gate is "the booth screen at
100 / 125 / 150 / 200 %", and on this machine that gate **cannot be looked at**. This module is
what makes it existable: it renders the **real** scene graph offscreen at a requested geometry and
scale, saves a PNG, and **reads every string back out of the running scene**.

A harness that returns a plausible-looking empty result is worse than no harness, because four
tasks will then report numbers it invented. Every design decision below is against one of the
three ways this project has measured itself inventing a number:

| the defect | what this module does instead |
|---|---|
| A `Repeater` delegate is invisible to `QObject.findChild` and `QObject.children()`. M2's first three-card reader returned `MISSING` for every item **on a render whose PNG plainly showed all three cards.** | [`walk_items`](#walking-the-scene) starts at `contentItem()` and descends `childItems()`. The walk **reports whether it completed**, and `read_scene` names every unreachable item rather than treating it as absent. |
| `Math.max` passes `NaN` through, which produced an entirely empty metrics column **with no QML warning**. | Geometry is read from the item. A non-finite coordinate puts the item in `geometry_unavailable` - **never** in the "no overflow" bucket. |
| A rendered string can be wrong while everything around it is right. M2 shipped a thumbnail caption reading `Camera B · -1` because a `QVariantList` was concatenated with a string. | Every reported string comes from `item.property("text")` **on the live item**. No `.qml` file is opened during the read, and `READ_AUDIT` proves it. |

## Which strings, and from where

`strings` is `objectName -> rendered text` for every **named** item the walk reached. Three states
are kept apart, because collapsing them is the defect:

- **reached** - the name is in `strings`. Its value may legitimately be `""`.
- **absent** - the name is not in `strings` and the walk completed. It is listed in `absent`.
- **unreachable** - the walk **could not descend** somewhere, so a missing name proves nothing.
  Those names are listed in `unreachable`, and `walk_complete` is `False`.

A fourth state is per-field: an item with **no `text` property at all** (an `Item`, a `Rectangle`)
is reached but has `text is None`, which is not `""`.

## Truncation and overflow are different faults

- **`truncated`** is `Text.truncated` read off the item: `True`, `False`, or `None` for an item
  that is not a `Text`. `Qt`'s `text` property is the **whole** string even when the layout elides
  it, so a truncated string is reported **in full** *and* named in `truncated`. Nothing is ever
  silently shortened.
- **`overflow`** is the item's **actual rendered extent** against **its clip**. The clip is the
  nearest ancestor item with `clip: true`; where there is none, the clip is the **window's own
  rect**, and `clip_sources` records which of the two applied. It is never estimated from a font
  metric and never computed with `Math.max` over a possibly-`NaN` height.

**No item in the shipped QML sets `clip: true`** (measured - `grep` finds no `clip:` in
`app/ui/*.qml`, and the live read returns `clip=False` for every named item), so on the real booth
screen `overflow` is measured against the window rect. That is a real result and it is reported as
one; a card's text spilling past its own card is invisible to this measure, and the harness says so
through `clip_sources` rather than pretending otherwise.

## Where the seeded numbers come from

No number is typed in here. `load_measured_seed()` reads the **committed** measurement files:

- `docs/qualification/m2-headless/allcpu-steady-120s.json` - one 120 s hold of spec section 5
  row 4, three streams on the CPU, and the hold whose figures M2's report led with. Two
  further holds are committed beside it as `-hold2` and `-hold3`; M2's report gives the
  **range** across all three (p50 10.7-14.7 s) and this harness seeds the first, so the
  captures are reproducible from exactly the numbers named here.
- `docs/qualification/m2-headless/route-stress-fault-visible-fallback.json` - the **committed**
  fault run, for the failed route's own stated reason.

If either file is missing, `load_measured_seed()` **raises** with the path it wanted. It never
substitutes a plausible number, because a layout measured against invented figures is the exact
failure this module exists to prevent. `CaptureResult.seed_provenance` records the file each field
came from, so a reader can check.

## Live mode versus seed mode

- **Seed mode** (default, `device=None`): the **real** `Main.qml`, the **real** `UiBridge`, the
  **real** `FrameImageProvider` and the **real** `ThemeTokens`, constructed the way
  `app.main.run_windowed` constructs them, with the measured figures seeded in. **No inference
  worker is started**, which is why the capture tests need no hardware and finish in seconds.
- **Live mode** (`device="CPU"`): drives **`app.main.main(["--windowed", ...])`** exactly the way
  `tests/test_main.py::LiveWindowedPath` does - same seams (`_gui_application`, the engine
  constructor), same "grab then quit" timer - and reads the strings out of **that** window. This
  needs the provisioned clip and IR; without them it raises `NotProvisioned` so the caller can skip
  **with a stated reason** rather than pass silently.

Both modes share one reader, one grab, one audit. Only how the window comes to exist differs.

## The scale is a process-level fact, so a capture is a child process

`QT_SCALE_FACTOR` is read once, when the platform integration starts, and `QGuiApplication` is a
singleton - and the rest of this suite constructs one too. So `capture()` writes a child program and
runs it, the same reason `LiveWindowedPath._drive` does.

**Measured on this machine, Qt 6.11.2, `QT_QPA_PLATFORM=offscreen`:**

| `QT_SCALE_FACTOR` | screen DPR | `grabWindow()` pixels |
|---|---|---|
| 1 | 1.0 | 1920x1080 |
| 1.25 | 1.25 | 2400x1350 |
| 1.5 | 1.5 | 2880x1620 |
| 2 | 2.0 | **3840x2160** |

**`QQuickWindow.grabWindow()` returns device pixels, and `QImage.devicePixelRatio()` on the result
is `1.0` at every scale** - so the only correct way to read the PNG's size is
`image.width()` / `image.height()`, and `CaptureResult.pixel_width` / `pixel_height` are read from
the image rather than computed from the scale. Asserting on `devicePixelRatio()` would have
passed at 100 % and failed at 200 %, which is the opposite of what this gate is for.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "CaptureResult",
    "NotProvisioned",
    "READ_AUDIT",
    "SCALES",
    "SceneRead",
    "capture",
    "load_measured_seed",
    "main",
    "read_scene",
    "walk_items",
]

#: The repository root, resolved from this file rather than from the working directory, so the
#: CLI works from anywhere and so the committed measurement files are found where they are.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The scales the M3 gate names, and the only ones `main()` accepts by default.
SCALES: tuple[int, ...] = (100, 125, 150, 200)

#: Slack when comparing an extent against a clip, in logical pixels.
#:
#: **Measured, not guessed.** Over the 48 named extents of a 1920x1080 booth capture, **41 have
#: at least one fractional coordinate and the largest distance from an integer is 0.5 px** (for
#: example `1635.1399999999999`). A comparison with no slack would therefore report overflow on a
#: screen that fits, which would make the whole measure noise; a slack much larger than 0.5 would
#: hide a real 1 px spill. The comparison is strict `>`, so an error of exactly the tolerance does
#: not trigger it.
#:
#: **The consequence, stated rather than hidden: an overflow smaller than half a logical pixel is
#: not reported.** At 200% a logical pixel is two physical pixels, so half a logical pixel is one
#: physical pixel - and every overflow that matters for a layout is tens of pixels.
TOLERANCE_PX = 0.5

#: Where the committed measurements live, relative to the repository root.
MEASURED_STEADY = pathlib.Path("docs/qualification/m2-headless/allcpu-steady-120s.json")
MEASURED_FAULT = pathlib.Path("docs/qualification/m2-headless/route-stress-fault-visible-fallback.json")

#: The child program's report prefix. One line, JSON, so a parent never parses prose.
REPORT_PREFIX = "CAPTURE_REPORT "
AUDIT_PREFIX = "CAPTURE_OPENED "


class NotProvisioned(RuntimeError):
    """Raised when a **live** capture is asked for and the assets are not provisioned.

    A distinct type so a test can `skipTest` **with the reason printed**, rather than failing
    with a stack trace or - worse - passing because the capture silently produced nothing.
    """


@dataclasses.dataclass(frozen=True)
class SceneRead:
    """What one pass over the scene graph found. Frozen; it is a measurement."""

    strings: Mapping[str, str]
    text_property: Mapping[str, bool]
    visible: Mapping[str, bool]
    #: `objectName -> font.pixelSize`, **only for items that have a font** (a `Text`). A
    #: `Rectangle` is absent rather than `0`, because `0` is not a font size.
    #:
    #: Read through `property("font").pixelSize()`, measured to work on this PySide6 - `None`
    #: for an item with no font, and `property("font.pixelSize")` returns `None` for **every**
    #: item including a `Text`, which is why the `QFont` is asked for the number instead.
    #: **This is what lets a layout claim "no font size changed" be checked rather than
    #: asserted**, and it is the measurement M3 Task 6's readability review needs.
    font_pixels: Mapping[str, int]
    truncated: tuple[str, ...]
    overflow: tuple[str, ...]
    geometry_unavailable: tuple[str, ...]
    clip_sources: Mapping[str, str]
    extents: Mapping[str, tuple[float, float, float, float]]
    absent: tuple[str, ...]
    unreachable: tuple[str, ...]
    walk_complete: bool
    walk_failures: tuple[str, ...]
    items_seen: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "strings": dict(self.strings),
            "text_property": dict(self.text_property),
            "visible": dict(self.visible),
            "font_pixels": dict(self.font_pixels),
            "truncated": list(self.truncated),
            "overflow": list(self.overflow),
            "geometry_unavailable": list(self.geometry_unavailable),
            "clip_sources": dict(self.clip_sources),
            "extents": {k: list(v) for k, v in self.extents.items()},
            "absent": list(self.absent),
            "unreachable": list(self.unreachable),
            "walk_complete": self.walk_complete,
            "walk_failures": list(self.walk_failures),
            "items_seen": self.items_seen,
        }


@dataclasses.dataclass(frozen=True)
class CaptureResult:
    """One capture. `width`/`height` are what was asked for; `pixel_*` is what was saved."""

    png_path: pathlib.Path
    width: int
    height: int
    scale_percent: int
    dpr: float
    pixel_width: int
    pixel_height: int
    #: **`QImage.devicePixelRatio()` on the grabbed image, read and reported.** Measured
    #: **1.0 at every scale**, including a 3840x2160 grab - so it is not the size
    #: signal and reading it would pass at 100% and fail at 200%. Reported so a reader
    #: can check that claim against the record instead of taking it.
    image_dpr: float
    strings: Mapping[str, str]
    font_pixels: Mapping[str, int]
    truncated: tuple[str, ...]
    overflow: tuple[str, ...]
    absent: tuple[str, ...]
    unreachable: tuple[str, ...]
    geometry_unavailable: tuple[str, ...]
    clip_sources: Mapping[str, str]
    extents: Mapping[str, tuple[float, float, float, float]]
    visible: Mapping[str, bool]
    walk_complete: bool
    items_seen: int
    settled: bool
    settle_missed: tuple[str, ...]
    frames_settled: int
    qobject_hits: tuple[str, ...]
    qobject_missed: tuple[str, ...]
    qobject_child_count: int | None
    glyph_pixels: Mapping[str, int]
    elapsed_s: float
    mode: str
    seed_provenance: Mapping[str, str]
    opened_during_read: tuple[str, ...]
    read_window_closed: bool
    json_path: pathlib.Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "png_path": str(self.png_path),
            "json_path": None if self.json_path is None else str(self.json_path),
            "width": self.width,
            "height": self.height,
            "scale_percent": self.scale_percent,
            "dpr": self.dpr,
            "pixel_width": self.pixel_width,
            "pixel_height": self.pixel_height,
            "image_dpr": self.image_dpr,
            "strings": dict(self.strings),
            "font_pixels": dict(self.font_pixels),
            "visible": dict(self.visible),
            "truncated": list(self.truncated),
            "overflow": list(self.overflow),
            "absent": list(self.absent),
            "unreachable": list(self.unreachable),
            "geometry_unavailable": list(self.geometry_unavailable),
            "clip_sources": dict(self.clip_sources),
            "extents": {k: list(v) for k, v in self.extents.items()},
            "walk_complete": self.walk_complete,
            "items_seen": self.items_seen,
            "settled": self.settled,
            "settle_missed": list(self.settle_missed),
            "frames_settled": self.frames_settled,
            "qobject_hits": list(self.qobject_hits),
            "qobject_missed": list(self.qobject_missed),
            "qobject_child_count": self.qobject_child_count,
            "glyph_pixels": dict(self.glyph_pixels),
            "elapsed_s": self.elapsed_s,
            "mode": self.mode,
            "seed_provenance": dict(self.seed_provenance),
            "opened_during_read": list(self.opened_during_read),
            "read_window_closed": self.read_window_closed,
        }


# --------------------------------------------------------------------------------------
# Walking the scene
# --------------------------------------------------------------------------------------

def walk_items(root: Any, *, limit: int = 20000):
    """Yield `(objectName, item)` for every named item under `root`, depth-first.

    ## Why `childItems()` and not `findChild` / `children()`

    A `QQuickItem` delegate created by a `Repeater` is a **visual** child - `setParentItem`, not
    `setParent` - so `QObject.findChild(name)` and `QObject.children()` **cannot see it**.
    `QQuickItem.childItems()` can. Measured in M2: the first three-card reader used `findChild`
    and reported `MISSING` for *every* item on a render whose PNG plainly showed all three cards,
    and an assertion on `""` passes - so that reader would have been a vacuous pass across a
    working screen.

    ## Why the walk starts at `contentItem()`

    A `Window`'s own `childItems()` is **empty**, because its content item is *assigned*
    (`setContentItem`) rather than added. Measured in M2 as well; the reader that started at the
    window found nothing.

    ## Why it yields rather than raising

    A dead C++ object under a live parent raises `RuntimeError` from `childItems()`. That must
    not abort a measurement - it must be **recorded**, because an item whose subtree could not be
    walked is an item whose existence the harness cannot vouch for. `read_scene` turns those
    paths into `unreachable` names.
    """
    if root is None:
        return
    stack: list[Any] = [root]
    seen = 0
    while stack:
        item = stack.pop()
        seen += 1
        if seen > limit:
            raise RuntimeError(
                f"the scene walk exceeded {limit} items; a capture that cannot finish its "
                f"walk must report no strings rather than a truncated set")
        try:
            name = str(item.objectName())
        except (AttributeError, RuntimeError):
            name = ""
        if name:
            yield name, item
        try:
            kids = list(item.childItems())
        except (AttributeError, RuntimeError):
            # **Yielded as `(path, None)`, and that `None` is load-bearing.** The first version
            # yielded a private marker object in the *name* slot and read the failure test off
            # the *item* slot, so a walk failure was filed as a named item whose geometry could
            # not be read: it appeared in `geometry_unavailable` under a marker object as its
            # key, `walk_complete` stayed `True`, and a broken walk reported itself complete.
            # Measured by a test that failed for exactly that reason.
            yield _label(item), None
            continue
        stack.extend(kids)


def _label(item: Any) -> str:
    """The best available name for an item, for reporting a path the walk could not enter."""
    try:
        return str(item.objectName())
    except Exception:  # noqa: BLE001 - a label is best-effort by definition
        return "<unnamed>"


def _geometry(item: Any) -> tuple[float, float, float, float] | None:
    """`(x, y, w, h)` of `item` **in scene coordinates**, or `None` if it cannot be had.

    `None` means *unavailable*, which is a different answer from `(0, 0, 0, 0)`. A `NaN`
    coordinate - the `Math.max`/`NaN` failure this whole module is written against - must never
    be allowed to read as "fits".
    """
    try:
        from PySide6.QtCore import QPointF

        point = item.mapToScene(QPointF(0.0, 0.0))
        values = (float(point.x()), float(point.y()),
                  float(item.width()), float(item.height()))
    except Exception:  # noqa: BLE001 - any failure here is "unavailable", not "fits"
        return None
    for value in values:
        if value != value or value in (float("inf"), float("-inf")):
            return None
    return values


def _clip_rect(item: Any, window_rect: tuple[float, float, float, float]) -> tuple[
        tuple[float, float, float, float], str]:
    """The rect `item` is clipped against, and **which** clip that was.

    Walks `parentItem()` upward to the nearest ancestor with `clip: true`. Where no ancestor
    clips - which is the case for **every** named item on the shipped booth screen, measured -
    the clip is the window's own rect and the source says `"window"`. A caller that wants to know
    whether a card's text spilled past its own card can read `clip_sources` and find out.
    """
    try:
        ancestor = item.parentItem()
    except Exception:  # noqa: BLE001
        ancestor = None
    depth = 0
    while ancestor is not None and depth < 64:
        depth += 1
        try:
            if bool(ancestor.clip()):
                rect = _geometry(ancestor)
                if rect is not None:
                    return rect, "ancestor:" + _label(ancestor)
        except Exception:  # noqa: BLE001
            pass
        try:
            ancestor = ancestor.parentItem()
        except Exception:  # noqa: BLE001
            ancestor = None
    return window_rect, "window"


def read_scene(root: Any, *,
               window_rect: tuple[float, float, float, float] | None = None,
               expect: Iterable[str] = ()) -> SceneRead:
    """Read every named item under `root`. **The only place strings come from.**

    `root` is a `QQuickItem` - normally `window.contentItem()`. It is a parameter rather than
    hardcoded so this function can be pointed at a small purpose-built QML scene by a test,
    which is how overflow and truncation are proved end to end with real Qt rather than with
    hand-written fakes.

    `expect` names items the caller believes must exist. Each is classified as **absent** (the
    walk completed and it was not there) or **unreachable** (the walk could not descend
    somewhere, so its absence proves nothing). Nothing else is inferred.
    """
    rect = window_rect if window_rect is not None else (0.0, 0.0, 0.0, 0.0)
    strings: dict[str, str] = {}
    text_property: dict[str, bool] = {}
    visible: dict[str, bool] = {}
    truncated: list[str] = []
    overflow: list[str] = []
    geometry_unavailable: list[str] = []
    clip_sources: dict[str, str] = {}
    extents: dict[str, tuple[float, float, float, float]] = {}
    font_pixels: dict[str, int] = {}
    failures: list[str] = []
    items_seen = 0

    for name, item in walk_items(root):
        if item is None:
            # **A path the walk could not enter.** Its whole subtree is unknown, so any
            # expected name under it is `unreachable` rather than `absent`.
            failures.append(name)
            continue
        items_seen += 1
        try:
            raw = item.property("text")
        except (AttributeError, RuntimeError):
            raw = None
        text_property[name] = raw is not None
        if raw is not None:
            # **The whole string, never the elided one.** Qt's `text` is the source string; the
            # elide is a layout consequence, reported separately in `truncated`.
            strings[name] = str(raw)
        try:
            visible[name] = bool(item.property("visible"))
        except (AttributeError, RuntimeError):
            visible[name] = False
        try:
            flag = item.property("truncated")
        except (AttributeError, RuntimeError):
            flag = None
        if flag is True:
            truncated.append(name)
        # **The font size, for every named item that has one.** Absent from the map rather
        # than `0` for an item with no font: a `Rectangle` has no font size and `0` would be
        # a number a layout claim could be checked against by accident.
        try:
            font = item.property("font")
            if font is not None:
                font_pixels[name] = int(font.pixelSize())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        box = _geometry(item)
        if box is None:
            geometry_unavailable.append(name)
            continue
        extents[name] = box
        clip, source = _clip_rect(item, rect)
        clip_sources[name] = source
        if (box[0] < clip[0] - TOLERANCE_PX or box[1] < clip[1] - TOLERANCE_PX
                or box[0] + box[2] > clip[0] + clip[2] + TOLERANCE_PX
                or box[1] + box[3] > clip[1] + clip[3] + TOLERANCE_PX):
            overflow.append(name)

    walk_complete = not failures
    wanted = list(dict.fromkeys(expect))
    absent = tuple(name for name in wanted
                   if name not in strings and name not in text_property
                   and walk_complete)
    unreachable = tuple(name for name in wanted
                        if name not in strings and name not in text_property
                        and not walk_complete)
    return SceneRead(
        strings=strings,
        text_property=text_property,
        visible=visible,
        font_pixels=font_pixels,
        truncated=tuple(sorted(truncated)),
        overflow=tuple(sorted(overflow)),
        geometry_unavailable=tuple(sorted(geometry_unavailable)),
        clip_sources=clip_sources,
        extents=extents,
        absent=absent,
        unreachable=unreachable,
        walk_complete=walk_complete,
        walk_failures=tuple(failures),
        items_seen=items_seen,
    )


#: Names a booth capture is expected to show. **A list to be checked, not a list to be
#: believed**: anything missing lands in `absent` and a test asserts on that.
#:
#: Chosen to span the three ways M2 has shipped a wrong screen, and each one is here for a
#: reason a reader can check:
#:
#: - **a `Repeater` delegate** - `card_A`, `card_B`, `card_C` and every `stream*Text_*`. **All
#:   fifteen** of the nineteen names `QObject.findChild` misses on this screen are in this list;
#:   the four it does reach are `deviceBadgeText`, `heroTile`, `tileNoFrameText_A` and
#:   `eventCountRow`. Measured on the committed `100pct.json`: `qobject_hits` 4,
#:   `qobject_missed` 15.
#: - **a binding that read the wrong thing** - `streamWindowText_A`, the window line, which is
#:   the one M2's `Window`-note guard exists for.
#: - **a list concatenated with a string** - `thumb_B` and `thumb_C`, the captions that
#:   rendered `Camera B · -1`.
#:
#: `card_A` and `thumb_B` have **no `text` property** and are here precisely so their absence
#: from `strings` can be asserted: a box must not be reported as a blank label.
BOOTH_EXPECTED: tuple[str, ...] = (
    "deviceBadgeText",
    "card_A", "card_B", "card_C",
    "streamAgeText_A", "streamAgeText_B", "streamAgeText_C",
    "streamStaleText_A", "streamStaleText_C",
    "streamWindowText_A",
    "streamBacklogText_A",
    "streamRateText_A",
    "streamBadgeText_A",
    "streamReasonText_C",
    "heroTile", "thumb_B", "thumb_C",
    "tileNoFrameText_A",
    "eventCountRow",
)

#: Audit hook flag name. The child flips this immediately before `read_scene`, so the audit
#: records exactly the window in which strings are produced - and a `.qml` file opened inside
#: that window would mean the harness had gone to the source instead of the scene.
READ_AUDIT = "_capture_reading"


def load_measured_seed(steady: pathlib.Path | None = None,
                       fault: pathlib.Path | None = None) -> dict[str, Any]:
    """The measured figures a capture is seeded with, and where each came from.

    **Reads committed files. Raises rather than inventing.** If a measurement is missing, a
    layout measured against a made-up number is worse than no capture, so this is an error the
    caller has to handle - and the test suite handles it by skipping with the path printed.
    """
    steady_path = steady or (REPO_ROOT / MEASURED_STEADY)
    fault_path = fault or (REPO_ROOT / MEASURED_FAULT)
    if not steady_path.is_file():
        raise NotProvisioned(
            f"the committed steady-state measurement is missing: {steady_path}. A capture "
            f"seeded with invented figures is the failure this module exists to prevent, so "
            f"this is raised, not defaulted.")
    if not fault_path.is_file():
        raise NotProvisioned(
            f"the committed fault-run measurement is missing: {fault_path}. Its stated reason "
            f"is the string the failed card has to be able to show, so it is not invented here.")
    steady_data = json.loads(steady_path.read_text(encoding="utf-8"))
    fault_data = json.loads(fault_path.read_text(encoding="utf-8"))
    streams = []
    for entry in steady_data["per_stream"]:
        streams.append({
            "stream_id": entry["stream_id"],
            "requested_device": entry["requested_device"],
            "confirmed_devices": list(entry["confirmed_devices"]),
            "state": entry["state"],
            "age_p50_ms": entry["frame_age_p50_ms"],
            "age_p95_ms": entry["frame_age_p95_ms"],
            "backlog": entry["backlog"],
            "analysis_fps": round(entry["frames_analysed"] / steady_data["seconds"], 2),
            "window_samples": entry["window_samples"],
        })
    reason = ""
    recovery = fault_data.get("fault_recovery") or {}
    for text in recovery.get("errors") or ():
        if "COMPILE FAILURE" in text or "compile" in text.lower():
            reason = text
            break
    if not reason:
        raise NotProvisioned(
            f"the committed fault run at {fault_path} carries no stated compile-failure "
            f"reason, so the failed card's string cannot be seeded from it. Raised, not "
            f"invented.")
    return {
        "streams": streams,
        "failed_stream": steady_data["per_stream"][-1]["stream_id"],
        "failed_reason": reason,
        "provenance": {
            "age_p50_ms": str(MEASURED_STEADY),
            "age_p95_ms": str(MEASURED_STEADY),
            "backlog": str(MEASURED_STEADY),
            "analysis_fps": str(MEASURED_STEADY) + " (frames_analysed / seconds)",
            "window_samples": str(MEASURED_STEADY),
            "failed_reason": str(MEASURED_FAULT),
            "failed_state": "seeded, not measured: this is a fault injected so the failed "
                            "card's own string is on screen to be measured",
        },
    }


# --------------------------------------------------------------------------------------
# The seed -> route rule, written ONCE and shared with the child
# --------------------------------------------------------------------------------------
#
# **It was inline in the child, and that is where the bug was.** The first version of this task
# added `errors` and `keep_confirmed` and, in the same edit, changed the failed stream's state to
# `entry.get("state", "failed")`. `entry` **always** has a `state`, and the committed
# steady-state measurement records all three cameras as `active`, so the failed camera was
# rendered **`active` with no confirmed device** - a state the booth cannot be in and the state
# that makes the card look healthy. The capture reported `items 57, strings 32` instead of
# `56, 33` and still exited 0, because a capture has no opinion about whether the state it was
# asked for is the state it should be in.
#
# So the rule is a **module-level function with no Qt dependency**, `exec`'d into this module's
# namespace and pasted into the child unchanged. That gives three things: it is unit-testable
# without spawning anything, the parent and the child cannot drift, and `seed_state` is the only
# way to override a state - spelled differently from `state` so it cannot be hit by accident.

_ROUTES_FROM_SEED_SOURCE = r'''
def routes_from_seed(seed):
    """The bridge routes a seed describes, one dict per stream.

    ## The default, and what each optional key changes

    The stream named by `seed["failed_stream"]` is **failed**, claims **no** device, and carries
    `seed["failed_reason"]`. Every other stream keeps its measured `state`, its measured
    `confirmed_devices`, and has no reason.

    Three optional per-stream keys exist because **the busiest layout is not the one the default
    seed produces**, and this is the reasoning behind each of them:

    - `errors` - a stated reason on a camera that is *not* the failed one. The failed camera has
      no confirmed device, so its right column shows four rows with **67.3 px** of slack
      (measured). The case that stresses a card is a camera that **fell back successfully**: it
      has a confirmed device, so `cardText_` and `cardWindowNote_` appear, *and* a reason from
      the transition - six rows in one column, the reason last, **43.3 px** of slack (measured).
    - `keep_confirmed` - the failed stream keeps its measured `confirmed_devices`, so the same
      six-row shape can be built on the failed stream itself.
    - `seed_state` - overrides the state **for any stream**, including the failed one. Named
      differently from `state` on purpose: `state` is what was measured, `seed_state` is what a
      caller is choosing, and the two must never be confused.

    With none of the three present the result is byte-identical to what M3 Task 1 seeded.
    """
    routes = []
    for entry in seed["streams"]:
        failed = entry["stream_id"] == seed["failed_stream"]
        errors = list(entry.get("errors") or [])
        if failed and not errors:
            errors = [seed["failed_reason"]]
        confirmed = list(entry["confirmed_devices"])
        if failed and not entry.get("keep_confirmed"):
            confirmed = []
        routes.append({
            "stream": entry["stream_id"],
            "requested": entry["requested_device"],
            "confirmed": confirmed,
            "state": entry.get("seed_state", "failed" if failed else entry["state"]),
            "fell_back": bool(entry.get("fell_back")),
            "errors": errors,
        })
    return routes
'''

# One definition, live in this module's namespace. `exec` rather than a copy, because a copy is
# two pieces of code that can disagree and the disagreement is invisible.
exec(compile(_ROUTES_FROM_SEED_SOURCE, "<routes_from_seed>", "exec"), globals())


# --------------------------------------------------------------------------------------
# The child program
# --------------------------------------------------------------------------------------

_CHILD = r'''
# **The seed -> route rule is pasted in above, not reimplemented here.** It is
# `_ROUTES_FROM_SEED_SOURCE` verbatim, so this child and the parent module share one definition
# and cannot drift. An earlier version had the rule inline in this string, and it silently
# rendered the failed camera as `active` with no confirmed device - a state the booth cannot be
# in. The comment on `_ROUTES_FROM_SEED_SOURCE` is the whole story.
''' + _ROUTES_FROM_SEED_SOURCE + r'''

"""Build (or drive) the real booth scene, settle it, grab it, and read every string back.

Run as a **file**, never with `python -c`: `QT_SCALE_FACTOR` is read by the platform plugin at
import time and a spawned inference worker re-imports `__main__`.

Prints exactly two lines the parent reads: `CAPTURE_OPENED {json}` - every file opened while the
strings were being produced - and `CAPTURE_REPORT {json}`.

## **Everything that starts anything is behind `if __name__ == "__main__"`**

Windows `multiprocessing` uses the **spawn** context, and a spawned worker re-imports this file
as `__mp_main__`. The first version of the live path called `app.main.main(["--windowed", ...])`
at module scope, so the worker's re-import **re-ran the whole application from inside the
spawning import**. The inner run tried to spawn a worker of its own and
`spawn._check_not_importing_main` raised; the worker exited 1; the route went `failed`; and the
capture faithfully reported `device: failed - no device is running`.

**That is the harness working**: it read the screen and the screen was wrong. But it is also a
harness bug, and the only thing that distinguishes the two is the guard. A harness whose own
failures are indistinguishable from its subject's is a harness that will be believed.
"""
import json
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, @@REPO@@)

# -- the audit: every file opened inside the read window ------------------------------
# Module level, and harmless on a re-import: installing an audit hook costs nothing and opens
# nothing. Only the reader's own `open` events matter, and the reader does not run here.
_OPENED = []
_PHASE = {"reading": False}
READ_WINDOW_CLOSED = False


def _hook(event, args):
    if _PHASE["reading"] and event == "open":
        try:
            _OPENED.append(str(args[0]))
        except Exception:
            pass


def read_scene(window, expect, window_rect):
    import tools.capture_ui as harness
    _PHASE["reading"] = True
    try:
        return harness.read_scene(window.contentItem(), window_rect=window_rect,
                                  expect=expect)
    finally:
        _PHASE["reading"] = False
        # **Recorded, because an audit window that is opened and never closed is an audit that
        # proves nothing.** Nothing opens a file after the read in this child, so leaving the
        # hook armed would still report an empty list - measured as an equivalent mutant - and
        # an assertion on `opened_during_read` alone could not tell the two apart. The report
        # says the window was closed, and a test asserts it.
        globals()["READ_WINDOW_CLOSED"] = True


def grab_and_read(window, started, expect, out, elapsed_note=""):
    """Wait for the scene to reach the state it was asked to capture, grab, then read.

    ## Readiness is a **scene-graph** fact, not a frame count

    The first version waited on `frameSwapped` and reported `frames_settled: 0` on every scale -
    measured, with the offscreen platform - because a window that is not on a real screen does
    not emit it reliably. It then sat out the whole deadline before grabbing. That is the
    "capture that settles on nothing" case, and it must never be silent.

    So readiness is **every expected name being present in the walk**, polled on a `QTimer` with
    the same reader the report uses. Both facts are reported: `settled` (bool) and
    `frames_settled` (the frame-swap count, which may honestly be 0).
    """
    from PySide6.QtCore import QEventLoop, QTimer
    import tools.capture_ui as harness

    deadline_s = float(ARGS["seconds"])
    loop = QEventLoop()
    state = {"frames": 0, "settled": False, "missed": list(expect), "polls": 0}
    started_at = time.perf_counter()

    def on_frame():
        state["frames"] += 1

    window.frameSwapped.connect(on_frame)

    def poll():
        state["polls"] += 1
        present = set()
        try:
            for name, item in harness.walk_items(window.contentItem()):
                if item is not None:
                    present.add(name)
        except Exception:
            pass
        state["missed"] = [name for name in expect if name not in present]
        if not state["missed"]:
            state["settled"] = True
            loop.quit()
        elif time.perf_counter() - started_at > deadline_s:
            loop.quit()

    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(25)
    loop.exec()
    timer.stop()
    try:
        window.frameSwapped.disconnect(on_frame)
    except Exception:
        pass

    image = window.grabWindow()
    saved = image.save(out + ".png")

    read = read_scene(window, list(expect) + list(ARGS.get("absent_expected") or ()),
                      (0.0, 0.0, float(window.width()), float(window.height())))
    report = {
        "ok": bool(saved) and read.walk_complete and state["settled"],
        "png_saved": bool(saved),
        "settled": state["settled"],
        "settle_polls": state["polls"],
        "settle_missed": state["missed"],
        "mode": ARGS["mode"],
        "dpr": float(window.devicePixelRatio()),
        "logical_width": int(window.width()),
        "logical_height": int(window.height()),
        "pixel_width": int(image.width()),
        "pixel_height": int(image.height()),
        "image_dpr": float(image.devicePixelRatio()),
        "frames_settled": state["frames"],
        "elapsed_s": time.perf_counter() - started,
        "note": elapsed_note,
        "scene": read.as_dict(),
        "opened_during_read": sorted(set(_OPENED)),
        "read_window_closed": bool(READ_WINDOW_CLOSED),
        "seed_provenance": ARGS.get("seed_provenance", {}),
    }
    report.update(_qobject_and_pixel_evidence(window, read, image, expect))
    print("CAPTURE_OPENED " + json.dumps(report["opened_during_read"]))
    print("CAPTURE_REPORT " + json.dumps(report))
    sys.stdout.flush()
    app.quit()
    return 0


def _qobject_and_pixel_evidence(window, read, image, expect):
    """The M2 trap, quantified on **this** render, plus what the PNG says about it.

    `findChild` / `children()` / `findChildren` are asked for the same names the walk found, and
    the counts go in the report side by side. A harness built on them would return nothing for a
    delegate; this makes the difference a number rather than a story.

    The pixel count is the other half of the same claim: for a named item it is the number of
    pixels **inside its reported extent** that differ from the image's modal colour. A rendered
    word is thousands and an empty rectangle is zero, so it is the PNG confirming the item is
    drawn - which is what stops the `findChild` comparison from being two pieces of Python
    agreeing with each other.
    """
    evidence = {}
    qobject_hits, qobject_missed = [], []
    try:
        from PySide6.QtCore import QObject
        for name in expect:
            found = (window.findChild(QObject, name)
                     or window.contentItem().findChild(QObject, name))
            (qobject_hits if found is not None else qobject_missed).append(name)
    except Exception as exc:  # noqa: BLE001 - a probe failure is reported, not hidden
        qobject_missed = ["<findChild unavailable: %s>" % type(exc).__name__]
    evidence["qobject_hits"] = qobject_hits
    evidence["qobject_missed"] = qobject_missed
    try:
        from PySide6.QtCore import QObject
        evidence["qobject_child_count"] = len(window.findChildren(QObject))
    except Exception:
        evidence["qobject_child_count"] = None
    painted = {}
    try:
        from collections import Counter
        from PySide6.QtGui import QImage
        rgb = image.convertToFormat(QImage.Format_RGB888)
        w, h, stride = rgb.width(), rgb.height(), rgb.bytesPerLine()
        raw = bytes(rgb.constBits())
        modal = Counter(raw[i:i + 3] for i in range(0, len(raw), 3)).most_common(1)[0][0]
        for name, box in read.extents.items():
            count = 0
            for yy in range(max(0, int(box[1])), min(h - 1, int(box[1] + box[3]))):
                base = yy * stride
                for xx in range(max(0, int(box[0])),
                                min(w - 1, int(box[0] + box[2]))):
                    off = base + xx * 3
                    if raw[off:off + 3] != modal:
                        count += 1
            painted[name] = count
    except Exception as exc:  # noqa: BLE001
        painted = {"<error>": str(exc)}
    evidence["glyph_pixels"] = painted
    return evidence


def run_seed(font):
    """The real `Main.qml`, the real bridge, the real provider, **no inference worker**."""
    import shutil
    import tempfile
    from PySide6.QtCore import QTimer, QUrl
    from PySide6.QtGui import QFontDatabase, QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuick import QQuickWindow
    import tools.capture_ui as harness
    from app.ui.bridge import FrameImageProvider, ThemeTokens, UiBridge

    out = str(ARGS["out"])
    # **The QML copy goes in a temporary directory, not beside the PNG.** The first version made
    # it `<scale>pct_ui/`, and a `git add -A` committed four stale copies of every QML file and
    # of `bridge.py` and `provider.py` alongside the evidence. A copy of the application that
    # can drift from the application is worse than no copy at all, and it is scratch either way.
    work = tempfile.mkdtemp(prefix="capture-ui-")
    shutil.rmtree(work, ignore_errors=True)
    # **The repository's `app/ui`, reached through the harness module - not
    # `Path(__file__).parent`.** This file lives in a temporary directory, so `__file__`'s
    # parent is the capture's own scratch space and the copy failed with
    # `FileNotFoundError: '...\cap6\app\ui'`.
    shutil.copytree(str(harness.REPO_ROOT / "app" / "ui"), work)
    globals()["app"] = QGuiApplication(sys.argv[:1])
    QFontDatabase.addApplicationFont(font)
    bridge = UiBridge(globals()["app"])
    theme = ThemeTokens(globals()["app"])
    engine = QQmlApplicationEngine()
    engine.addImageProvider("frames", FrameImageProvider(bridge))
    ctx = engine.rootContext()
    ctx.setContextProperty("Theme", theme)
    ctx.setContextProperty("bridge", bridge)

    seed = ARGS["seed"]
    bridge.set_streams([entry["stream_id"] for entry in seed["streams"]])
    for entry, route in zip(seed["streams"], routes_from_seed(seed)):
        bridge.apply_route(route)
        bridge.set_stream_metrics(entry["stream_id"], age_p50_ms=entry["age_p50_ms"],
                                  age_p95_ms=entry["age_p95_ms"],
                                  backlog=entry["backlog"],
                                  analysis_fps=entry["analysis_fps"],
                                  window_samples=entry["window_samples"])

    engine.load(QUrl.fromLocalFile(work + "/Main.qml"))
    window = next((r for r in engine.rootObjects() if isinstance(r, QQuickWindow)), None)
    if window is None:
        print("CAPTURE_REPORT " + json.dumps({"ok": False, "error": "Main.qml produced no "
                                                                 "QQuickWindow"}))
        sys.stdout.flush()
        return 2
    window.setWidth(float(ARGS["width"]))
    window.setHeight(float(ARGS["height"]))
    started = time.perf_counter()
    QTimer.singleShot(120, lambda: grab_and_read(window, started, ARGS["expect"], out))
    return globals()["app"].exec()


def run_live(font):
    """Drive the real application and read **its** window - the `LiveWindowedPath` pattern."""
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QFontDatabase, QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuick import QQuickWindow
    import app.main as main_module

    globals()["app"] = QGuiApplication(sys.argv[:1])
    QFontDatabase.addApplicationFont(font)
    out = str(ARGS["out"])
    _real_gui = main_module._gui_application
    _engine_holder = {}
    _real_engine_init = QQmlApplicationEngine.__init__

    def engine_init(self, *a, **k):
        _real_engine_init(self, *a, **k)
        _engine_holder["engine"] = self

    def spy_gui():
        qobject = _real_gui()
        QTimer.singleShot(int(float(ARGS["seconds"]) * 1000), capture_the_live_window)
        return qobject

    def capture_the_live_window():
        engine = _engine_holder.get("engine")
        roots = engine.rootObjects() if engine is not None else []
        window = next((r for r in roots if isinstance(r, QQuickWindow)), None)
        if window is None:
            print("CAPTURE_REPORT " + json.dumps(
                {"ok": False, "error": "the live application produced no QQuickWindow"}))
            sys.stdout.flush()
            globals()["app"].quit()
            return
        grab_and_read(window, started, ARGS["expect"], out)

    main_module._gui_application = spy_gui
    QQmlApplicationEngine.__init__ = engine_init
    started = time.perf_counter()
    return main_module.main(["--windowed", "--device", str(ARGS["device"])]
                            + list(ARGS.get("extra_args") or []))


if __name__ == "__main__":
    sys.addaudithook(_hook)
    ARGS = json.loads(sys.argv[2])
    mode = ARGS["mode"]
    code = run_seed(ARGS["font"]) if mode == "seed" else run_live(ARGS["font"])
    raise SystemExit(code)
'''


def capture(width: int | None = None, height: int | None = None,
            scale_percent: int = 100, seconds: float = 6.0,
            device: str | None = None,
            seed_frames: Mapping[str, Any] | None = None,
            out_dir: pathlib.Path | str | None = None,
            expect: Sequence[str] = BOOTH_EXPECTED,
            absent_expected: Sequence[str] = (),
            font: str | None = None,
            extra_args: Sequence[str] = ()) -> CaptureResult:
    """Render the real booth scene offscreen at `scale_percent` and read every string back.

    `width`/`height` default to the shipped canvas, **read from `app.theme`** rather than typed
    in here - a tool that restates the canvas is a tool that can disagree with it.

    `scale_percent` is one of `SCALES`. **The PNG's pixel size is measured from the image**, and
    `pixel_width`/`pixel_height` are never computed from the scale: at 200% a 1920x1080 logical
    canvas is 3840x2160 pixels, and `QImage.devicePixelRatio()` is `1.0` at every scale, so
    reading that instead would pass at 100% and fail at 200%.

    `device=None` is seed mode: no inference worker. `device="CPU"` drives the real application
    through `app.main.main(["--windowed", ...])` and needs the provisioned assets, raising
    `NotProvisioned` when they are absent.

    `expect` is what the screen is asked to show: **readiness means all of it is present**, and
    anything missing is accounted for as `absent`. `absent_expected` is for names the caller
    knows will **not** be there - a one-camera live run has no `card_B`, and asserting that
    requires naming it. They are accounted for exactly like `expect` but excluded from
    readiness, because a capture that waits for something that will never exist would time
    out on a correct screen.
    """
    if scale_percent not in SCALES:
        raise ValueError(
            f"scale_percent must be one of {list(SCALES)}; got {scale_percent!r}. The M3 gate "
            f"names exactly these four scales and inventing a fifth would be a measurement of "
            f"nothing.")
    from app.theme import CANVAS_HEIGHT, CANVAS_WIDTH

    logical_w = int(CANVAS_WIDTH if width is None else width)
    logical_h = int(CANVAS_HEIGHT if height is None else height)
    resolved_font = font or os.path.join(
        os.environ.get("SystemRoot", r"C:\Windows"), "Fonts", "arial.ttf")
    if not os.path.isfile(resolved_font):
        raise NotProvisioned(
            f"no font to register: {resolved_font}. Measured on this machine, the offscreen "
            f"platform reports **0 font families** with none registered, so a capture would "
            f"save a PNG with no glyphs in it and every string in it would be unreadable. "
            f"Raised, not defaulted.")

    mode = "live" if device else "seed"
    if mode == "seed" and seed_frames is None:
        seed_frames = load_measured_seed()
    provenance = dict((seed_frames or {}).get("provenance", {}))

    target = pathlib.Path(out_dir) if out_dir is not None else pathlib.Path(
        tempfile.mkdtemp(prefix="capture-"))
    target.mkdir(parents=True, exist_ok=True)
    stem = target / f"{scale_percent}pct"
    # **The child program is scratch too, and lives in its own temporary directory.** Writing it
    # beside the PNG put a `_capture_child.py` in the evidence directory, where `git add -A`
    # committed it.
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="capture-child-"))

    args = {
        "out": str(stem),
        "width": logical_w,
        "height": logical_h,
        "seconds": float(seconds),
        "device": device or "CPU",
        "mode": mode,
        "expect": list(expect),
        "absent_expected": list(absent_expected),
        "font": resolved_font,
        "seed": dict(seed_frames or {}),
        "seed_provenance": provenance,
        "extra_args": list(extra_args),
    }
    script = scratch / "_capture_child.py"
    # **`str.replace`, not `%`-formatting.** The child is full of literal `%` - percentages in
    # prose, `"%s" % exc` inside the probes - and `%`-formatting it raises
    # `TypeError: not enough arguments for format string` at the first one. Measured, and the
    # failure is at run time rather than at import, which is the worst place for it.
    script.write_text(
        _CHILD
        .replace("@@REPO@@", repr(str(REPO_ROOT)))
        .replace("@@AUDIT_FLAG@@", repr(READ_AUDIT)),
        encoding="utf-8")

    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["QT_SCALE_FACTOR"] = str(scale_percent / 100.0)
    env["PYTHONPATH"] = str(REPO_ROOT)
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(script), str(stem), json.dumps(args)],
        capture_output=True, text=True, timeout=max(120.0, float(seconds) * 8.0 + 90.0),
        cwd=str(REPO_ROOT), env=env)
    elapsed = time.perf_counter() - started

    report_line = next((line for line in completed.stdout.splitlines()
                        if line.startswith(REPORT_PREFIX)), None)
    if report_line is None:
        raise RuntimeError(
            "the capture child produced no report, so every number this module could return "
            "would be invented.\nSTDOUT:\n" + completed.stdout[-2500:]
            + "\nSTDERR:\n" + completed.stderr[-2500:])
    report = json.loads(report_line[len(REPORT_PREFIX):])
    if not report.get("ok"):
        raise RuntimeError(
            f"the capture child reported a failure. "
            f"settled={report.get('settled')} "
            f"settle_missed={report.get('settle_missed')} "
            f"png_saved={report.get('png_saved')} "
            f"walk_complete={report.get('scene', {}).get('walk_complete')} "
            f"error={report.get('error') or '-'}\nSTDOUT:\n" + completed.stdout[-1500:]
            + "\nSTDERR:\n" + completed.stderr[-1500:])
    scene = report["scene"]
    png_path = pathlib.Path(str(stem) + ".png")
    if not png_path.is_file():
        raise RuntimeError(
            f"the capture reported success but saved no PNG at {png_path}. A capture with no "
            f"image is worse than no capture.")
    return CaptureResult(
        png_path=png_path,
        width=logical_w,
        height=logical_h,
        scale_percent=scale_percent,
        dpr=float(report["dpr"]),
        pixel_width=int(report["pixel_width"]),
        pixel_height=int(report["pixel_height"]),
        image_dpr=float(report["image_dpr"]),
        strings=dict(scene["strings"]),
        font_pixels=dict(scene.get("font_pixels") or {}),
        truncated=tuple(scene["truncated"]),
        overflow=tuple(scene["overflow"]),
        absent=tuple(scene["absent"]),
        unreachable=tuple(scene["unreachable"]),
        geometry_unavailable=tuple(scene["geometry_unavailable"]),
        clip_sources=dict(scene["clip_sources"]),
        extents={k: tuple(v) for k, v in scene["extents"].items()},
        visible=dict(scene["visible"]),
        walk_complete=bool(scene["walk_complete"]),
        items_seen=int(scene["items_seen"]),
        settled=bool(report.get("settled")),
        settle_missed=tuple(report.get("settle_missed") or ()),
        frames_settled=int(report["frames_settled"]),
        qobject_hits=tuple(report.get("qobject_hits") or ()),
        qobject_missed=tuple(report.get("qobject_missed") or ()),
        qobject_child_count=report.get("qobject_child_count"),
        glyph_pixels=dict(report.get("glyph_pixels") or {}),
        elapsed_s=float(report["elapsed_s"]),
        mode=mode,
        seed_provenance=provenance,
        opened_during_read=tuple(scene.get("opened_during_read") or ()),
        read_window_closed=bool(report.get("read_window_closed")),
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture the booth screen offscreen at each DPI scale and read every "
                    "string out of the running scene.")
    parser.add_argument("--scale", default=",".join(str(s) for s in SCALES),
                        help="comma-separated scales to capture (default: 100,125,150,200)")
    parser.add_argument("--width", type=int, default=None,
                        help="logical width (default: the shipped canvas, from app.theme)")
    parser.add_argument("--height", type=int, default=None,
                        help="logical height (default: the shipped canvas, from app.theme)")
    parser.add_argument("--seconds", type=float, default=6.0,
                        help="wall-clock bound on letting the scene settle")
    parser.add_argument("--device", default=None,
                        help="drive the real application on this device instead of seeding; "
                             "needs the provisioned clip and IR")
    parser.add_argument("--out", default="docs/qualification/m3-capture",
                        help="output directory (default: docs/qualification/m3-capture)")
    args = parser.parse_args(argv)

    scales = [int(part) for part in args.scale.split(",") if part.strip()]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for scale in scales:
        result = capture(width=args.width, height=args.height, scale_percent=scale,
                         seconds=args.seconds, device=args.device, out_dir=out_dir)
        json_path = out_dir / f"{scale}pct.json"
        json_path.write_text(
            json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[{scale}%] {result.png_path}  {result.pixel_width}x{result.pixel_height} px "
              f"(logical {result.width}x{result.height}, dpr {result.dpr})  "
              f"items {result.items_seen}  frames {result.frames_settled}  "
              f"strings {len(result.strings)}  truncated {list(result.truncated)}  "
              f"overflow {list(result.overflow)}  absent {list(result.absent)}  "
              f"-> {json_path}")
        if result.absent or result.unreachable or result.geometry_unavailable:
            failures += 1
    if failures:
        print(f"\n{failures} of {len(scales)} capture(s) had items the harness could not "
              f"account for. The JSON says which and how.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())