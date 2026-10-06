"""`tests.test_capture_ui` - what the M3 capture harness must be able to tell.

## The instrument, not the instrument's user

Four later tasks will report DPI-gate numbers from `tools/capture_ui.py`. **A harness that
returns a plausible-looking empty result is worse than no harness**, because every one of those
numbers would then be a number it invented. So this file is mostly about the harness's own
failure modes, and each test here was checked by introducing the fault it is supposed to catch
(see the report's mutation table).

The three ways this project has measured itself inventing a number, and what each test pins:

| measured defect | test |
|---|---|
| M2's first three-card reader used `QObject.findChild`, which **cannot see a `Repeater` delegate**, and reported `MISSING` for every item on a render whose PNG plainly showed all three cards. An assertion on `""` passes, so that reader would have been a **vacuous pass across a working screen**. | `test_findchild_cannot_see_the_delegates_the_png_proves_are_there` |
| `Math.max` passes `NaN` through, which produced an entirely empty metrics column **with no QML warning**. An overflow measure built on `Math.max` over a possibly-`NaN` height reports "no overflow" for a card that is not there. | `test_a_nan_height_is_geometry_unavailable_and_never_fits` |
| M2 shipped a thumbnail caption reading `Camera B · -1`: a `QVariantList` concatenated with a string. The Python side held `confirmed: ('GPU.0',)` and the picture above the caption was real - **nothing about a screenshot of that caption says it is wrong**. | `test_the_strings_come_from_the_scene_and_not_from_the_qml_source` |

Three states are kept apart throughout, because collapsing them is the defect: **reached** (in
`strings`, possibly `""`), **absent** (the walk completed, it was not there) and **unreachable**
(the walk could not descend somewhere, so its absence proves nothing).
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools import capture_ui  # noqa: E402


# --------------------------------------------------------------------------------------
# Reading the saved PNG back, independently of the harness
# --------------------------------------------------------------------------------------
#
# **These two helpers exist so a number in a JSON can be checked against the file it came from.**
# The harness counts pixels in the child process and writes the count into the JSON; nothing in
# the test suite ever opened the PNG. So the JSON could have been wrong about its own image and
# nothing would have noticed - and it was: `glyph_pixels` was `0` for 35 of 54 named items at
# 150 % because the **device**-pixel image was indexed with **logical** extents.

def _load_png(path):
    """Open a saved capture. `QImage` reads a PNG without a `QGuiApplication`."""
    from PySide6.QtGui import QImage

    image = QImage(str(path))
    if image.isNull():
        raise AssertionError(f"{path} did not load as an image; it is "
                             f"{path.stat().st_size if path.is_file() else 'absent'} bytes on disk")
    return image


def _count_non_modal(image, box, scale_x, scale_y):
    """Non-modal pixels inside `box`, **scaled into device pixels first**.

    **Deliberately not a copy of the harness's loop.** This one uses ordinary half-open
    `[x0, x1) x [y0, y1)` bounds, and the harness's excludes the final row and column. So the two
    differ by **at most one row plus one column**, and that difference is what
    `test_the_pixel_count_in_the_json_matches_the_png_it_claims_to_describe` asserts. A harness
    that indexed the device image with logical coordinates would be off by a factor of the scale -
    at 150 % that is 274 750 against 48 389 for a card - which no edge-row allowance covers.
    """
    from collections import Counter

    from PySide6.QtGui import QImage

    rgb = image.convertToFormat(QImage.Format_RGB888)
    width, height = rgb.width(), rgb.height()
    stride = rgb.bytesPerLine()
    raw = bytes(rgb.constBits())
    modal = Counter(raw[i:i + 3] for i in range(0, len(raw), 3)).most_common(1)[0][0]
    x0 = max(0, min(width, int(box[0] * scale_x)))
    x1 = max(0, min(width, int((box[0] + box[2]) * scale_x)))
    y0 = max(0, min(height, int(box[1] * scale_y)))
    y1 = max(0, min(height, int((box[1] + box[3]) * scale_y)))
    count = 0
    for yy in range(y0, y1):
        base = yy * stride
        for xx in range(x0, x1):
            off = base + xx * 3
            if raw[off:off + 3] != modal:
                count += 1
    return count, (x1 - x0) + (y1 - y0)


# --------------------------------------------------------------------------------------
# Fakes, for the reader's own logic
# --------------------------------------------------------------------------------------

class _FakeItem:
    """The minimum a `QQuickItem` offers the reader. **Only for the reader's own rules.**

    Nothing here fabricates a *measurement*: the real measurements come from
    `RealQmlProbeScene`, which loads actual QML through an actual engine. These fakes exist so
    the two states a real scene cannot easily be made to produce - a `NaN` coordinate and an
    un-descendable subtree - can be produced at all.
    """

    def __init__(self, name="", *, text=None, visible=True, truncated=None,
                 x=0.0, y=0.0, w=10.0, h=10.0, clip=False, children=(), parent=None):
        self._name = name
        self._props = {"text": text, "visible": visible, "truncated": truncated}
        self._pos = (x, y)
        self._size = (w, h)
        self._clip = clip
        self._children = list(children)
        self._parent = parent
        self.raise_on_children = False

    # -- QQuickItem surface ---------------------------------------------------------
    def objectName(self):
        return self._name

    def property(self, key):
        return self._props.get(key)

    def childItems(self):
        if self.raise_on_children:
            raise RuntimeError("simulated dead C++ object")
        return list(self._children)

    def parentItem(self):
        return self._parent

    def clip(self):
        return self._clip

    def width(self):
        return self._size[0]

    def height(self):
        return self._size[1]

    def mapToScene(self, point):
        class _P:
            def __init__(self, x, y):
                self._x, self._y = x, y

            def x(self):
                return self._x

            def y(self):
                return self._y
        return _P(self._pos[0] + point.x(), self._pos[1] + point.y())


def _tree():
    """A small tree: a clipping panel with one child inside it and one hanging out of it."""
    panel = _FakeItem("panel", x=0.0, y=0.0, w=100.0, h=100.0, clip=True)
    inside = _FakeItem("insideText", text="fits", x=5.0, y=5.0, w=50.0, h=20.0, parent=panel)
    outside = _FakeItem("outsideText", text="spills", x=50.0, y=90.0, w=200.0, h=40.0,
                        parent=panel)
    panel._children = [inside, outside]
    return panel, inside, outside


class SceneReaderRules(unittest.TestCase):
    """The reader's own rules, with no Qt window involved."""

    WINDOW = (0.0, 0.0, 1920.0, 1080.0)

    def test_overflow_is_measured_against_the_nearest_clipping_ancestor(self):
        panel, inside, outside = _tree()
        read = capture_ui.read_scene(panel, window_rect=self.WINDOW)
        self.assertEqual(read.strings["insideText"], "fits")
        self.assertNotIn("insideText", read.overflow,
                         f"an item inside its clip is not overflowing: {read.overflow}")
        self.assertIn("outsideText", read.overflow,
                      f"a 200x40 item at y=90 in a 100x100 clipping panel extends to y=130, "
                      f"which is 30 px past the panel's own clip. Measured from the rendered "
                      f"extent, not from a font metric: {read.extents.get('outsideText')}")
        self.assertEqual(read.clip_sources["outsideText"], "ancestor:panel",
                         f"the clip must be named so a reader can tell an item clipped by "
                         f"its own panel from one only clipped by the window: "
                         f"{read.clip_sources}")

    def test_with_nothing_clipping_the_clip_is_the_window_and_that_is_recorded(self):
        loose = _FakeItem("looseText", text="x", x=0.0, y=0.0, w=100.0, h=100.0)
        wide = _FakeItem("wideText", text="y", x=0.0, y=1000.0, w=100.0, h=100.0,
                         parent=loose)
        loose._children = [wide]
        read = capture_ui.read_scene(loose, window_rect=self.WINDOW)
        self.assertEqual(read.clip_sources["wideText"], "window")
        self.assertIn("wideText", read.overflow,
                      f"y 1000+100 = 1100 exceeds a 1080 window: {read.extents.get('wideText')}")

    def test_a_nan_height_is_geometry_unavailable_and_never_fits(self):
        # **The `Math.max`/`NaN` failure this module is written against.** A `NaN` height has
        # to be *unavailable*, and an unavailable geometry must not be counted as fitting -
        # otherwise an overflow measure over a `NaN` height reports "no overflow" for a card
        # that is not there, which is what an empty metrics column looks like from here.
        ghost = _FakeItem("ghostText", text="nowhere", x=0.0, y=0.0, w=100.0,
                          h=float("nan"))
        read = capture_ui.read_scene(ghost, window_rect=self.WINDOW)
        self.assertIn("ghostText", read.geometry_unavailable,
                      f"a NaN height must be reported as unavailable: {read.geometry_unavailable}")
        self.assertNotIn("ghostText", read.extents,
                         "an item with no usable geometry must not appear in extents")
        self.assertNotIn("ghostText", read.overflow,
                         f"an item whose geometry could not be read is NOT known to fit: "
                         f"{read.overflow}")
        # And it is still reached: "cannot measure it" is not "not there".
        self.assertEqual(read.strings["ghostText"], "nowhere")
        self.assertNotIn("ghostText", read.absent)

    def test_absent_empty_and_unreachable_are_three_different_answers(self):
        empty = _FakeItem("emptyText", text="")
        read = capture_ui.read_scene(empty, window_rect=self.WINDOW,
                                     expect=["emptyText", "neverBuilt"])
        # reached, and genuinely empty
        self.assertIn("emptyText", read.strings)
        self.assertEqual(read.strings["emptyText"], "")
        self.assertNotIn("emptyText", read.absent,
                         "an item that is present and empty is not absent")
        # never built, and the walk completed, so its absence is informative
        self.assertEqual(read.absent, ("neverBuilt",))
        self.assertEqual(read.unreachable, ())
        self.assertTrue(read.walk_complete)

        # ...and the same name on a walk that could not descend is **unreachable**
        broken = _FakeItem("brokenRoot")
        broken.raise_on_children = True
        broken._children = [empty]
        read2 = capture_ui.read_scene(broken, window_rect=self.WINDOW,
                                      expect=["emptyText", "neverBuilt"])
        self.assertFalse(read2.walk_complete)
        self.assertEqual(read2.unreachable, ("emptyText", "neverBuilt"))
        self.assertEqual(read2.absent, (),
                         "a walk that could not descend must not call anything absent")

    def test_an_item_with_no_text_property_is_not_reported_as_an_empty_string(self):
        # A `Rectangle` has no `text`. Reporting `""` for it would put a real empty string in
        # the map, and a caller asking "is this label blank?" would be told the label is blank.
        plain = _FakeItem("plainBox", text=None)
        read = capture_ui.read_scene(plain, window_rect=self.WINDOW)
        self.assertNotIn("plainBox", read.strings,
                         f"an item with no `text` property must not appear in strings: "
                         f"{read.strings}")
        self.assertFalse(read.text_property["plainBox"],
                         "and text_property must say so, so 'absent' is not confused with 'blank'")

    def test_a_truncated_string_is_reported_whole_and_also_named(self):
        long_text = "route transition to generation 1 was not acknowledged: " + ("x" * 400)
        item = _FakeItem("reasonText", text=long_text, truncated=True)
        read = capture_ui.read_scene(item, window_rect=self.WINDOW)
        self.assertEqual(read.truncated, ("reasonText",))
        self.assertEqual(read.strings["reasonText"], long_text,
                         "the whole string, never the elided prefix: a harness that reported "
                         "the visible fragment would be reporting a string the screen never "
                         "held")
        self.assertEqual(len(read.strings["reasonText"]), len(long_text))

    def test_a_false_truncated_flag_is_not_reported_as_truncated(self):
        item = _FakeItem("plain", text="short", truncated=False)
        read = capture_ui.read_scene(item, window_rect=self.WINDOW)
        self.assertEqual(read.truncated, ())

    def test_walk_items_yields_delegates_that_qobject_cannot_reach(self):
        # The reader reaches a delegate because `childItems()` is a **visual**-child walk. A
        # Repeater delegate is parented with `setParentItem`, so it is invisible to
        # `QObject.children()` - which is the whole reason this walk exists. Asserted on the
        # fake because the real proof is `RealQmlProbeScene` and `BoothCapture`.
        delegate = _FakeItem("delegateText", text="from a delegate")
        holder = _FakeItem("holder")
        holder._children = [delegate]
        names = [name for name, item in capture_ui.walk_items(holder)]
        self.assertIn("delegateText", names)
        # The QObject-shaped alternative: `objectName()` is reachable, but `children()` is not
        # consulted at all - the walk had to descend to find it.
        self.assertEqual(getattr(holder, "children", None), None,
                         "this fake deliberately has no `children()`; if a future reader "
                         "reached for one it would fail here rather than in production")


# --------------------------------------------------------------------------------------
# A real Qt scene, small and purpose-built, to prove overflow and truncation end to end
# --------------------------------------------------------------------------------------

#: The long string the probe scene elides. **A constant, not a literal in the QML**, so the
#: assertion can compare against the source of truth instead of a typed character count. The
#: first version asserted a hardcoded `96` and the real string was 89 - a number in a test
#: that was about nothing.
PROBE_LONG_TEXT = ("route transition to generation 1 was not acknowledged: "
                   "stream(s) ['C'] did not confirm a placement")

#: A real QML scene with: a clipping panel, a child that spills out of it, a long elided
#: `Text`, an empty `Text`, and a `Repeater` whose delegate `findChild` cannot see.
#:
#: **Real Qt, not fakes.** `truncated` is a layout consequence, so only a laid-out `Text` can
#: produce it; and `overflow` is about what was actually drawn, which only a real scene has.
PROBE_QML = """
import QtQuick

Window {
    id: probe
    width: 800
    height: 600
    visible: true
    color: "#101418"

    Rectangle {
        id: panel
        objectName: "clippingPanel"
        x: 20; y: 20; width: 300; height: 200
        color: "#1b2430"
        border.color: "#3a7bd5"
        border.width: 1
        clip: true

        Text {
            objectName: "fitsText"
            x: 5; y: 5; width: parent.width - 10; height: 20
            text: "fits"
            color: "#e8eef5"
            font.pixelSize: 16
        }

        Text {
            objectName: "spillsText"
            x: 5; y: 190; width: 600; height: 40
            text: "this text is wider than the panel that clips it"
            color: "#ffd166"
            font.pixelSize: 16
        }
    }

    Text {
        objectName: "longElidedText"
        x: 20; y: 240; width: 200; height: 20
        text: "@@LONG@@"
        color: "#e8eef5"
        font.pixelSize: 16
        elide: Text.ElideRight
    }

    Text {
        objectName: "blankText"
        x: 20; y: 270; width: 200; height: 20
        text: ""
        color: "#e8eef5"
        font.pixelSize: 16
    }

    Row {
        id: delegateRow
        x: 20; y: 300; width: 600; height: 40
        spacing: 10

        Repeater {
            model: 3
            delegate: Text {
                required property int index
                objectName: "rowDelegate_" + index
                width: 150; height: 30
                text: "delegate " + index
                color: "#7ee081"
                font.pixelSize: 16
            }
        }
    }

    Text {
        objectName: "outsideWindowText"
        x: 700; y: 590; width: 400; height: 40
        text: "this one hangs off the window"
        color: "#ff6b6b"
        font.pixelSize: 16
    }
}
"""

#: A child program that loads `PROBE_QML` through a real engine and hands its `contentItem()`
#: to the harness's reader. Its own process, because `QGuiApplication` is a singleton.
_PROBE_CHILD = r'''
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, @@REPO@@)
from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtGui import QFontDatabase, QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickWindow
import tools.capture_ui as harness

qml = sys.argv[1]
out = sys.argv[2]
app = QGuiApplication(sys.argv[:1])
QFontDatabase.addApplicationFont(r"C:\Windows\Fonts\arial.ttf")
engine = QQmlApplicationEngine()
engine.load(QUrl.fromLocalFile(qml))
roots = engine.rootObjects()
window = next((r for r in roots if isinstance(r, QQuickWindow)), None)
if window is None:
    print("PROBE " + json.dumps({"ok": False}))
    sys.exit(2)
loop = QEventLoop()
state = {"done": False}


def read():
    state["done"] = True
    qobj_hits, qobj_missed = [], []
    try:
        from PySide6.QtCore import QObject
        for name in ("rowDelegate_0", "rowDelegate_1", "fitsText", "clippingPanel"):
            found = (window.findChild(QObject, name)
                     or window.contentItem().findChild(QObject, name))
            (qobj_hits if found is not None else qobj_missed).append(name)
    except Exception as exc:
        qobj_missed = ["<error %s>" % type(exc).__name__]
    scene = harness.read_scene(
        window.contentItem(),
        window_rect=(0.0, 0.0, float(window.width()), float(window.height())),
        expect=("clippingPanel", "fitsText", "spillsText", "longElidedText", "blankText",
                "rowDelegate_0", "rowDelegate_1", "outsideWindowText", "neverBuilt"))
    image = window.grabWindow()
    image.save(out)
    painted = {}
    try:
        from collections import Counter
        rgb = image.convertToFormat(__import__("PySide6.QtGui", fromlist=["QImage"]).QImage.Format_RGB888)
        w, h, stride = rgb.width(), rgb.height(), rgb.bytesPerLine()
        raw = bytes(rgb.constBits())
        modal = Counter(raw[i:i + 3] for i in range(0, len(raw), 3)).most_common(1)[0][0]
        for name, box in scene.extents.items():
            n = 0
            for yy in range(max(0, int(box[1])), min(h - 1, int(box[1] + box[3]))):
                base = yy * stride
                for xx in range(max(0, int(box[0])), min(w - 1, int(box[0] + box[2]))):
                    off = base + xx * 3
                    if raw[off:off + 3] != modal:
                        n += 1
            painted[name] = n
    except Exception as exc:
        painted = {"<error>": str(exc)}
    print("PROBE " + json.dumps({"ok": True, "scene": scene.as_dict(),
                                "qobj_hits": qobj_hits, "qobj_missed": qobj_missed,
                                "painted": painted,
                                "pixels": [image.width(), image.height()]}))
    loop.quit()


QTimer.singleShot(500, read)
loop.exec()
if not state["done"]:
    sys.exit(3)
'''


class RealQmlProbeScene(unittest.TestCase):
    """Overflow and truncation proved on a **real** Qt scene, not on fakes."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="capture-probe-")
        qml = pathlib.Path(cls._tmp.name) / "probe.qml"
        qml.write_text(PROBE_QML.replace("@@LONG@@", PROBE_LONG_TEXT), encoding="utf-8")
        script = pathlib.Path(cls._tmp.name) / "probe_child.py"
        script.write_text(_PROBE_CHILD.replace("@@REPO@@", repr(str(REPO_ROOT))),
                          encoding="utf-8")
        png = pathlib.Path(cls._tmp.name) / "probe.png"
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(REPO_ROOT))
        completed = subprocess.run(
            [sys.executable, str(script), str(qml), str(png)],
            capture_output=True, text=True, timeout=300, env=env)
        line = next((ln for ln in completed.stdout.splitlines() if ln.startswith("PROBE ")), None)
        if line is None:
            raise AssertionError(
                "the probe scene produced no report.\nSTDOUT:\n" + completed.stdout[-2000:]
                + "\nSTDERR:\n" + completed.stderr[-2000:])
        cls.report = json.loads(line[len("PROBE "):])
        cls.scene = capture_ui.SceneRead(**{
            field: (tuple(value) if isinstance(value, list) else value)
            for field, value in _coerce(cls.report["scene"]).items()})
        cls.png = png

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_probe_scene_really_rendered(self):
        # **The guard that stops every assertion below being about nothing.** The rest of this
        # class reads geometry and truncation out of this scene, so the scene has to be real:
        # its PNG must exist, be the requested size, and contain more than one colour.
        self.assertEqual(self.report["pixels"], [800, 600])
        self.assertTrue(self.png.is_file() and self.png.stat().st_size > 0)
        self.assertTrue(self.scene.walk_complete,
                        f"the probe walk did not complete: {self.scene.walk_failures}")

    def test_a_child_that_spills_past_its_clipping_panel_is_reported_as_overflow(self):
        self.assertIn("clippingPanel", self.scene.extents,
                      f"the panel must be measured for this to mean anything: "
                      f"{sorted(self.scene.extents)}")
        self.assertIn("spillsText", self.scene.overflow,
                      f"a 600 px wide Text at x=5 inside a 300 px panel with clip: true "
                      f"extends 305 px past the panel. Reported from the rendered extent: "
                      f"panel {self.scene.extents.get('clippingPanel')}, "
                      f"text {self.scene.extents.get('spillsText')}")
        self.assertEqual(self.scene.clip_sources["spillsText"], "ancestor:clippingPanel")
        self.assertNotIn("fitsText", self.scene.overflow,
                         f"the child that fits must not be reported: {self.scene.overflow}")

    def test_an_item_off_the_window_is_reported_against_the_window(self):
        self.assertIn("outsideWindowText", self.scene.overflow,
                      f"x=700 width=400 in an 800 px window runs 300 px past the edge: "
                      f"{self.scene.extents.get('outsideWindowText')}")
        self.assertEqual(self.scene.clip_sources["outsideWindowText"], "window")

    def test_a_long_elided_text_is_named_and_reported_whole(self):
        self.assertIn("longElidedText", self.scene.truncated,
                      f"a 200 px wide Text with elide: Text.ElideRight holding "
                      f"{len(PROBE_LONG_TEXT)} characters is elided by the real layout: "
                      f"{self.scene.truncated}")
        self.assertEqual(self.scene.strings["longElidedText"], PROBE_LONG_TEXT,
                         f"and the harness reports all {len(PROBE_LONG_TEXT)} characters, not "
                         f"the elided prefix")

    def test_a_blank_text_is_present_and_blank(self):
        self.assertIn("blankText", self.scene.strings)
        self.assertEqual(self.scene.strings["blankText"], "")
        self.assertNotIn("blankText", self.scene.absent)

    def test_a_name_the_scene_never_built_is_absent_and_not_unreachable(self):
        self.assertIn("neverBuilt", self.scene.absent)
        self.assertEqual(self.scene.unreachable, (),
                         f"the probe walk completed, so nothing is unreachable: "
                         f"{self.scene.unreachable}")

    def test_a_repeater_delegate_is_reached_though_findchild_cannot(self):
        for index in (0, 1, 2):
            name = f"rowDelegate_{index}"
            self.assertIn(name, self.scene.strings,
                          f"the walk must reach every delegate: {sorted(self.scene.strings)}")
            self.assertEqual(self.scene.strings[name], f"delegate {index}")
        self.assertEqual(self.scene.qobject_missed if hasattr(self.scene, "qobject_missed")
                         else self.report["qobj_missed"], ["rowDelegate_0", "rowDelegate_1"],
                         f"the delegates are the names `findChild` must miss, and `fitsText` "
                         f"and `clippingPanel` are names it must find - otherwise this test "
                         f"would pass for the wrong reason")


def _coerce(raw: dict) -> dict:
    """Turn the JSON report back into the field types `SceneRead` expects."""
    out = {}
    for key, value in raw.items():
        if key in ("strings", "text_property", "visible", "clip_sources"):
            out[key] = dict(value)
        elif key == "extents":
            out[key] = {k: tuple(v) for k, v in value.items()}
        else:
            out[key] = tuple(value) if isinstance(value, list) else value
    return out


# --------------------------------------------------------------------------------------
# The booth screen itself
# --------------------------------------------------------------------------------------

class BoothCapture(unittest.TestCase):
    """The real `Main.qml`, the real bridge, the real provider, at each of the four scales.

    **No inference worker**: the bridge is seeded from the committed M2 measurements, so these
    capture the state that actually breaks layouts and need no hardware.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="capture-booth-")
        cls.results = {}
        for scale in capture_ui.SCALES:
            cls.results[scale] = capture_ui.capture(
                scale_percent=scale, seconds=10.0, out_dir=pathlib.Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def result(self, scale):
        return self.results[scale]

    def test_the_png_is_the_requested_pixel_size_and_not_the_logical_size(self):
        # **This distinction is the whole DPI gate.** The logical canvas never changes with the
        # scale; the pixels must. Asserting the logical size at every scale would pass
        # unconditionally and prove nothing.
        expected = {100: (1920, 1080), 125: (2400, 1350), 150: (2880, 1620), 200: (3840, 2160)}
        for scale, (want_w, want_h) in expected.items():
            result = self.result(scale)
            self.assertEqual((result.width, result.height), (1920, 1080),
                             f"the logical canvas must not change with the scale: "
                             f"{result.width}x{result.height}")
            self.assertEqual((result.pixel_width, result.pixel_height), (want_w, want_h),
                             f"at {scale}% the PNG must be {want_w}x{want_h} pixels; it is "
                             f"{result.pixel_width}x{result.pixel_height}. "
                             f"logical {result.width}x{result.height}, dpr {result.dpr}")
            self.assertEqual(round(result.dpr * 100), scale,
                             f"the reported dpr must be the scale asked for: {result.dpr}")
        self.assertNotEqual(self.result(200).pixel_width, self.result(200).width,
                            "at 200% the pixels and the logical width are different numbers; if "
                            "they are equal the gate is measuring nothing")

    def test_the_image_device_pixel_ratio_is_not_the_size_signal_and_is_reported_anyway(self):
        # **Measured on this machine: `QImage.devicePixelRatio()` is 1.0 at every scale**, even
        # when the grab is 3840x2160. A harness that read it instead of `width()`/`height()`
        # would pass at 100% and fail at 200% - the opposite of what this gate is for. It is
        # reported in every JSON so the claim is checkable against the record.
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            self.assertEqual(result.image_dpr, 1.0,
                             f"at {scale}% the grabbed image reports dpr "
                             f"{result.image_dpr}; this is the claim under test, and if it ever "
                             f"stopped being 1.0 the size would have to be read differently")
            self.assertAlmostEqual(result.pixel_width / result.width, scale / 100.0, places=6,
                                   msg=f"at {scale}% the pixel width is exactly the logical "
                                       f"width times the scale: {result.pixel_width} / "
                                       f"{result.width}")
        self.assertTrue(all(self.result(s).png_path.is_file() for s in capture_ui.SCALES))

    def test_every_scale_settled_and_the_png_is_not_blank(self):
        """**The PNG is loaded here.** It was not, and the test's name was a claim it never made.

        The first version of this test asserted `settled`, `settle_missed`, `walk_complete` and
        `items_seen` - all *construction* facts - and nothing about the image it saved. Meanwhile
        `glyph_pixels`, the measure that is supposed to say an item **is drawn**, was reported as
        `0` for items that are plainly on screen at three of the four scales (see
        `test_every_expected_name_has_painted_pixels_at_every_scale`). A test called
        `..._and_the_png_is_not_blank` that never opens the PNG is a test whose name outruns it.
        So this one opens the file, checks its size, and checks it has more than one colour.
        """
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            self.assertTrue(result.settled,
                            f"the scene never reached the state it was asked to capture; "
                            f"missing {result.settle_missed}. A capture that settled on "
                            f"nothing would save a plausible-looking blank PNG.")
            self.assertEqual(result.settle_missed, ())
            self.assertTrue(result.walk_complete,
                            f"at {scale}% the walk did not complete somewhere: "
                            f"{result.walk_failures}")
            self.assertGreater(result.items_seen, 40,
                               f"only {result.items_seen} named items were reached at "
                               f"{scale}%; the booth screen has more than that")
            self.assertTrue(result.png_path.is_file(),
                            f"the capture reported a PNG at {result.png_path} and there is "
                            f"nothing there")
            # **Actually load it.** `QImage` reads a PNG without a `QGuiApplication`, so this does
            # not disturb any other test's application instance.
            image = _load_png(result.png_path)
            self.assertEqual((image.width(), image.height()),
                             (result.pixel_width, result.pixel_height),
                             f"at {scale}% the file on disk must be the size the capture "
                             f"reported: {result.pixel_width}x{result.pixel_height}, the file "
                             f"is {image.width()}x{image.height()}")
            colours = len({image.pixel(x, y)
                           for x in range(0, image.width(), max(1, image.width() // 64))
                           for y in range(0, image.height(), max(1, image.height() // 64))})
            self.assertGreater(colours, 3,
                               f"at {scale}% the PNG holds {colours} sampled colour(s). A "
                               f"single-colour image is a blank image, and a blank image plus "
                               f"`settled: true` is exactly the result that would have been "
                               f"believed.")

    def test_every_expected_name_has_painted_pixels_at_every_scale(self):
        """**C1. The anti-vacuity evidence, checked where it was wrong.**

        `glyph_pixels[name]` is the number of non-modal pixels inside the item's reported extent,
        in **device** pixels. `read.extents` is in **logical** pixels, so the box has to be scaled
        by `image.width() / window.width()` before it indexes the buffer - the harness does that,
        and `glyph_scale` in each JSON is the factor it used.

        **What this assertion is worth depends entirely on running it at every scale.** At 100 %
        the factor is `1.0` and a harness that forgot to scale still passes. At 150 % the old
        code reported `0` for **35 of 54** named items, including `streamStaleText_A` and
        `eventCountRow` - items that are on the screen in front of the reader. A self-check that
        only runs at the one scale where the bug cannot show is not a self-check.
        """
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            self.assertEqual(tuple(result.glyph_scale), (scale / 100.0, scale / 100.0),
                             f"at {scale}% the pixel measure must have scaled the logical "
                             f"extents by the scale factor; it used {result.glyph_scale}. If "
                             f"this is 1.0 at 125% or more the box is being indexed in the "
                             f"wrong space and the counts below are meaningless.")
            missing = []
            for name in capture_ui.BOOTH_EXPECTED:
                painted = result.glyph_pixels.get(name)
                if painted is None or painted <= 0:
                    missing.append((name, painted))
            self.assertEqual(missing, [],
                             f"at {scale}% these expected items have **no painted pixels** "
                             f"inside their own reported extent: {missing}. The walk found "
                             f"them and the image says nothing is drawn there - which is the "
                             f"shape of a capture that renders nothing and exits 0.")
            # **And the count must grow with the scale.** A box that is not scaled reports the
            # same number at every scale, so this is the assertion the previous test was missing.
            if scale > 100:
                base = self.result(100).glyph_pixels
                smaller = [name for name in capture_ui.BOOTH_EXPECTED
                           if result.glyph_pixels.get(name, 0) <= base.get(name, 0)]
                self.assertEqual(smaller, [],
                                 f"at {scale}% every expected item must account for *more* "
                                 f"device pixels than at 100%; these did not: {smaller}")

    def test_the_pixel_count_in_the_json_matches_the_png_it_claims_to_describe(self):
        """**The JSON checked against the file, at every scale.** See `_count_non_modal`.

        This is the assertion that would have caught C1 from the outside. The harness reported
        `0` for `eventCountRow` at 150 % and 200 %; the JSON said `0`; no test opened the PNG, so
        nothing contradicted it. Here the count is recomputed from the **file on disk** and
        compared against the **reported** number, and the only slack allowed is the one row and
        one column the harness's loop bounds exclude by construction.
        """
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            image = _load_png(result.png_path)
            scale_x = result.glyph_scale[0]
            scale_y = result.glyph_scale[1]
            self.assertEqual((scale_x, scale_y), (scale / 100.0, scale / 100.0),
                             f"at {scale}% the harness must have scaled by the scale factor, "
                             f"not 1.0: {result.glyph_scale}")
            for name in capture_ui.BOOTH_EXPECTED:
                box = result.extents.get(name)
                self.assertIsNotNone(box, f"{scale}%: {name} must be measured for this to "
                                          f"mean anything")
                mine, edge = _count_non_modal(image, box, scale_x, scale_y)
                theirs = result.glyph_pixels.get(name, 0)
                self.assertGreater(theirs, 0,
                                   f"at {scale}% the JSON says {name} has {theirs} painted "
                                   f"pixels and the PNG has {mine}. A zero is the C1 shape: "
                                   f"the walk found an item the image does not show.")
                self.assertLessEqual(theirs, mine,
                                     f"at {scale}% {name}: the JSON claims {theirs} pixels but "
                                     f"the box holds only {mine}. The reported count cannot "
                                     f"exceed what is in the box.")
                self.assertLessEqual(mine - theirs, edge,
                                     f"at {scale}% {name}: the JSON says {theirs} and the PNG "
                                     f"says {mine}, a gap of {mine - theirs} against an "
                                     f"allowance of {edge} for the loop's excluded edge. A gap "
                                     f"this size is the box being indexed in the wrong space - "
                                     f"at {scale}% a logical box covers 1/{scale / 100.0:g}th "
                                     f"of the buffer.")

    def test_every_camera_caption_is_named_read_and_correct(self):
        """**I1. The caption was invisible to the instrument that exists to check it.**

        M2 shipped `Camera B · -1`: a `QVariantList` concatenated with a string, over a real
        picture. The `Text` carrying it had **no `objectName`**, so `walk_items` never yielded it
        and it appeared in none of the four committed JSONs - and `BOOTH_EXPECTED`'s own comment
        claimed `thumb_B`/`thumb_C` were listed *because* they are the captions that rendered it,
        which was false. **A regression here would have left all four JSONs byte-identical and the
        CLI exiting 0.**

        So the caption is named `tileCaptionText_<streamId>`, it is in `BOOTH_EXPECTED` and
        `BOOTH_NONEMPTY`, and its value is asserted here **per camera** - including that it names
        the camera that is on the tile and the device that camera actually confirmed.
        """
        from tools.capture_ui import load_measured_seed

        seed = load_measured_seed()
        # **The route's confirmed device, not the seed's measured one.** The seed->route rule
        # renders the failed camera with **no** confirmed device, so its caption is
        # `Camera C · not confirmed` even though the committed steady-state measurement records
        # a device for it. Reading `entry["confirmed_devices"]` directly here expected `CPU` and
        # failed on a correct screen: the expectation has to come from the rule the capture
        # actually applies, or the assertion is about a different render than the one on screen.
        routes = {route["stream"]: route for route in capture_ui.routes_from_seed(seed)}
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            for sid, name in (("A", "tileCaptionText_A"), ("B", "tileCaptionText_B"),
                              ("C", "tileCaptionText_C")):
                self.assertIn(name, result.extents,
                              f"{scale}%: {name} must be a **named, measured** item; if it is "
                              f"not named the harness cannot read it at all")
            # A and B are the thumbnails; C is the failed camera. The hero shares the component
            # and its caption Text exists with `""` because `Main.qml` gives the hero no caption -
            # so it is deliberately not in `BOOTH_EXPECTED` and is not required to be non-empty.
            for sid in ("B", "C"):
                name = f"tileCaptionText_{sid}"
                want = ", ".join(routes[sid]["confirmed"]) or "not confirmed"
                want = f"Camera {sid} \u00b7 {want}"
                self.assertEqual(result.strings.get(name), want,
                                 f"{scale}%: {name} reads "
                                 f"{result.strings.get(name)!r}, expected {want!r}. This is the "
                                 f"M2 defect: a caption that is present, plausible and wrong. "
                                 f"Nothing about a screenshot of it says so.")
                self.assertNotIn("-1", result.strings.get(name, ""),
                                 f"{scale}%: {name} reads -1, which is the literal M2 symptom: "
                                 f"a QVariantList stringified and concatenated")
                self.assertIn(f"Camera {sid}", result.strings.get(name, ""),
                              f"{scale}%: the caption must name the camera on its own tile")
            # **The hero's caption exists and is empty, and both are facts worth keeping apart.**
            self.assertEqual(result.strings["tileCaptionText_A"], "",
                             "the hero tile is given no caption by Main.qml, so its caption Text "
                             "is present and empty - reached, not absent, and not a blank label "
                             "invented for a box")

    def test_a_capture_waits_for_those_items_to_show_something_not_just_to_exist(self):
        """**I2. `settled` is a construction check, and this is the other half.**

        `settled` says every expected name is in the walk. Under the mutant that broke
        `stream_confirmed_labels`, **every expected name was still present** - so `settled` was
        `True`, `opened_during_read` was `[]`, `read_window_closed` was `True` and `items_seen`
        was normal - **on a wrong screen.** A capture in which every binding yielded `""` is the
        M2 shape, so `expect_nonempty` now joins the same poll.
        """
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            self.assertEqual(result.empty_nonempty, (),
                             f"at {scale}% these were present but showed nothing: "
                             f"{list(result.empty_nonempty)}. Present-and-empty is the shape of "
                             f"a capture that renders an empty screen and calls it settled.")
            for name in capture_ui.BOOTH_NONEMPTY:
                self.assertTrue(result.strings.get(name),
                                f"at {scale}% {name} is required to be non-empty and was "
                                f"{result.strings.get(name)!r}")

    def test_the_committed_records_are_what_this_code_produces_today(self):
        """**The four committed captures, checked against a fresh render.**

        I1 was possible *because* nothing compared them: if `stream_confirmed_labels` regressed,
        all four JSONs would have come out identical and the CLI would still have exited 0.
        So the committed files are not left as evidence of a past run - they are compared here,
        field by field, against the captures this class just made.

        **Which fields are compared is stated, because "identical" would be false.** `elapsed_s`,
        `json_path` and `png_path` are per-run by construction, and `seed_provenance` is a
        constant but is listed anyway. Everything else - every string, every extent, every glyph
        count - has to match exactly.
        """
        comparable = ("pixel_width", "pixel_height", "dpr", "image_dpr", "width", "height",
                      "scale_percent", "strings", "text_property", "font_pixels", "visible",
                      "truncated", "overflow", "absent", "unreachable",
                      "geometry_unavailable", "clip_sources", "extents", "walk_complete",
                      "walk_failures", "items_seen", "settled", "settle_missed",
                      "empty_nonempty", "qobject_hits", "qobject_missed",
                      "qobject_child_count", "glyph_pixels", "glyph_scale",
                      "seed_provenance", "opened_during_read", "read_window_closed", "mode")
        for scale in capture_ui.SCALES:
            committed_path = (REPO_ROOT / "docs" / "qualification" / "m3-capture"
                              / f"{scale}pct.json")
            self.assertTrue(committed_path.is_file(),
                            f"the committed record {committed_path} is missing; a capture that "
                            f"is only in this run's temporary directory is not evidence")
            committed = json.loads(committed_path.read_text(encoding="utf-8"))
            fresh = self.result(scale).as_dict()
            for field in comparable:
                self.assertEqual(
                    fresh[field], committed.get(field),
                    f"{scale}%: the committed {committed_path.name} no longer matches what "
                    f"this code produces for {field}. The record is the evidence; a record that "
                    f"has drifted from the instrument is worse than none. Re-render with "
                    f"`python -m tools.capture_ui --out docs/qualification/m3-capture` and "
                    f"commit by explicit path.")
            # And the two things a byte comparison would have hidden, checked explicitly.
            self.assertNotIn("elapsed_s", comparable,
                             "elapsed_s is wall-clock and differs run to run")
            self.assertNotIn("json_path", comparable,
                             "json_path is written by the CLI after the result is built")

    def test_the_record_carries_the_fourth_per_field_state_and_the_walk_failures(self):
        """**I4. Three fields the docstring claimed and the JSON did not carry.**

        `text_property` is how "reached but has no `text` property" is told apart from "reached
        and blank"; `walk_failures` is *why* `walk_complete` might be false; and `settle_polls` is
        how many times readiness was polled before the scene qualified. All three were in the
        code and in no committed record, which made the docstring's claim about them
        undiscernible from the evidence.
        """
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            payload = json.loads((REPO_ROOT / "docs" / "qualification" / "m3-capture"
                                  / f"{scale}pct.json").read_text(encoding="utf-8"))
            for field in ("text_property", "walk_failures", "settle_polls"):
                self.assertIn(field, payload,
                              f"{scale}%: `{field}` is claimed in capture_ui's docstring, so it "
                              f"has to be in the committed record")
            # **And they say something.** `text_property` separates a box from a blank label:
            # `card_A` is reached, has no `text` property, and is absent from `strings`.
            self.assertIn("card_A", result.text_property)
            self.assertFalse(result.text_property["card_A"])
            self.assertIn("deviceBadgeText", result.text_property)
            self.assertTrue(result.text_property["deviceBadgeText"])
            self.assertEqual(result.walk_failures, (),
                             f"at {scale}% the walk completed, so it names no failures: "
                             f"{result.walk_failures}")
            self.assertGreaterEqual(result.settle_polls, 1,
                                    f"at {scale}% readiness was polled at least once, and the "
                                    f"count is what says a capture settled on the first try")

    def test_the_strings_come_from_the_scene_and_not_from_the_qml_source(self):
        # **The audit is the assertion.** An `open` audit hook is armed immediately before the
        # reader runs and disarmed immediately after, so `opened_during_read` is exactly the
        # files the strings could have come from. A `.qml` file opened in that window would
        # mean the harness had gone to the source.
        for scale in capture_ui.SCALES:
            opened = self.result(scale).opened_during_read
            self.assertEqual(list(opened), [],
                             f"at {scale}% the reader opened {list(opened)}. Strings must come "
                             f"from the running scene and nowhere else.")
            # **And the audit window was closed.** An audit hook left armed after the read
            # reports the same empty list, because nothing opens a file afterwards - measured
            # as a surviving mutant - so without this the assertion could not tell a scoped
            # audit from an unscoped one, which is the difference between a measurement and a
            # coincidence.
            self.assertTrue(self.result(scale).read_window_closed,
                            f"at {scale}% the read audit window was never closed, so an empty "
                            f"list of opened files proves nothing")
        # And the reported strings are ones that exist in **no** QML file: the measured frame
        # ages are computed by Python and pushed through the bridge, so if the harness were
        # parsing source it could not produce them at all.
        from tools.capture_ui import load_measured_seed
        seed = load_measured_seed()
        by_id = {entry["stream_id"]: entry for entry in seed["streams"]}
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            expected_a = f"{by_id['A']['age_p50_ms'] / 1000:.1f} s"
            self.assertEqual(result.strings["streamAgeText_A"], expected_a,
                             f"the age headline must be the measured figure {expected_a}, "
                             f"which appears in no QML source")

    def test_findchild_cannot_see_the_delegates_the_png_proves_are_there(self):
        # **The M2 trap, quantified on this render.** `QObject.findChild` is asked for every
        # name the walk found. It cannot see the `Repeater` delegates - so a harness built on
        # it would report `""` for the three cards and every assertion on `""` would pass.
        result = self.result(100)
        missed = set(result.qobject_missed)
        self.assertTrue(missed, "findChild found every expected name, which means this render "
                                "has no delegates in it and the proof is vacuous")
        # **`streamAgeText_*` and `streamReasonText_C` are the delegates that carry a string**,
        # which is what makes them comparable with what `findChild` found. `card_A` is a
        # `Rectangle` delegate with no `text` property, so it cannot be in `strings` at all -
        # asking for it there would be asking for a blank string, and an assertion on a blank
        # string is the vacuous shape this whole module is written against.
        for delegate in ("streamAgeText_A", "streamAgeText_B", "streamAgeText_C",
                         "streamReasonText_C"):
            self.assertIn(delegate, missed,
                          f"{delegate} is a Repeater delegate and findChild must not reach it; "
                          f"missed = {sorted(missed)}")
            self.assertIn(delegate, result.strings,
                          f"...and the walk must reach it and read it: {sorted(result.strings)}")
        # **The PNG's own pixels are what makes this a proof and not a claim.** Each of those
        # delegates has non-background pixels inside its reported extent, so they are drawn.
        # findChild says they are not there; the image says they are.
        for delegate in ("streamAgeText_A", "streamAgeText_B", "streamAgeText_C",
                         "streamReasonText_C", "card_A"):
            painted = result.glyph_pixels.get(delegate, 0)
            self.assertGreater(painted, 0,
                               f"{delegate} is reached by the walk and missed by findChild, "
                               f"and if it has no painted pixels then nothing was drawn and "
                               f"the comparison is about nothing")
        # **And a box with no `text` property is reached without being reported as a blank
        # string**, so "the walk reached it" and "it has a string" stay separate facts.
        self.assertNotIn("card_A", result.strings)
        self.assertNotIn("card_A", result.absent)

    def test_a_repeater_delegates_own_string_is_the_measured_figure(self):
        result = self.result(100)
        from tools.capture_ui import load_measured_seed
        seed = load_measured_seed()
        by_id = {entry["stream_id"]: entry for entry in seed["streams"]}
        for sid in ("A", "B", "C"):
            name = f"streamAgeText_{sid}"
            want = f"{by_id[sid]['age_p50_ms'] / 1000:.1f} s"
            self.assertEqual(result.strings.get(name), want,
                             f"{name}: three cameras, three of their own figures. The value "
                             f"for {sid} must not be another's: {result.strings.get(name)}")
        self.assertNotEqual(result.strings["streamAgeText_A"],
                            result.strings["streamStaleText_A"],
                            "the headline and the sentence are different items and this "
                            "assertion would catch a reader that returned one for both")

    def test_the_staleness_sentence_is_present_for_every_camera(self):
        result = self.result(100)
        for sid in ("A", "B", "C"):
            self.assertEqual(
                result.strings[f"streamStaleText_{sid}"],
                f"STALE - this picture is {result.strings[f'streamAgeText_{sid}']} old; "
                f"the limit is 2 s",
                f"camera {sid} must say in words that its picture is stale, naming the age "
                f"and the limit: {result.strings[f'streamStaleText_{sid}']!r}")

    def test_the_failed_streams_badge_and_reason_are_on_the_screen(self):
        result = self.result(100)
        self.assertEqual(result.strings["streamBadgeText_C"], "no device is running",
                         f"a failed camera must claim no device: "
                         f"{result.strings.get('streamBadgeText_C')!r}")
        from tools.capture_ui import load_measured_seed
        reason = load_measured_seed()["failed_reason"]
        # **M3 Task 2: the card shows `reason_display`, not the complete reason.** The
        # complete 2331-character message is still what the supervisor stated; the screen
        # shows the part that fits plus a marker naming what it is not showing. Asserting
        # equality with the whole message here is exactly what this task removed.
        shown = result.strings["streamReasonText_C"]
        self.assertLess(len(shown), len(reason),
                        f"the screen must not be claiming to show all {len(reason)} "
                        f"characters; it shows {len(shown)}")
        self.assertTrue(shown.startswith(reason[:40].rstrip()),
                        f"the visible reason must begin with the reason itself, unedited: "
                        f"{shown[:60]!r}")
        self.assertIn("COMPILE FAILURE", reason,
                      f"the seeded reason is the committed supervisor message: {reason[:120]}")

    def test_the_failure_reason_is_cut_deliberately_and_says_how_much_is_missing(self):
        """**M3 Task 2 changed this, and the change is the point.**

        Task 1 found the failed camera's reason `truncated` with only an ellipsis glyph on
        screen. The cut is now made **in Python at a declared budget**
        (`Theme.CARD_REASON_CHARS`) and the marker names the loss, so `truncated` is empty -
        Qt has nothing left to elide, and the screen says what it is not showing.
        """
        from app.theme import CARD_REASON_CHARS, CARD_REASON_MARKER
        from tools.capture_ui import load_measured_seed

        reason = load_measured_seed()["failed_reason"]
        hidden = len(reason) - CARD_REASON_CHARS
        marker = CARD_REASON_MARKER.format(hidden=hidden)
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            shown = result.strings["streamReasonText_C"]
            self.assertTrue(shown.endswith(marker),
                            f"at {scale}% the reason must end by admitting what it is not "
                            f"showing. Expected the marker {marker!r}; the string ends "
                            f"{shown[-40:]!r}")
            self.assertEqual(shown[:CARD_REASON_CHARS].rstrip(), reason[:CARD_REASON_CHARS].rstrip(),
                             f"at {scale}% the first {CARD_REASON_CHARS} characters must be "
                             f"the reason itself, verbatim and unedited")
            self.assertLess(len(shown), len(reason),
                            f"at {scale}% the screen must be showing less than the whole "
                            f"reason: {len(shown)} of {len(reason)}")
            # **And Qt elided nothing**, so there is no second, silent cut behind the first.
            self.assertNotIn("streamReasonText_C", result.truncated,
                             f"at {scale}% Qt elided the reason as well, so there are two "
                             f"cuts and only one of them is visible: {result.truncated}")

    def test_overflow_is_empty_at_every_scale_and_every_clip_is_named(self):
        """**M3 Task 2 made this a per-card measure.**

        Task 1 could only measure against the window, because `grep -c 'clip: true'` over
        `app/ui/*.qml` was **0**. The card, its two columns and the three tiles now clip, so
        the nearest clipping ancestor is the card for most items and the overflow measure has
        teeth it did not have.
        """
        from app.theme import GRID
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            self.assertEqual(result.overflow, (),
                             f"at {scale}% nothing spills out of its own card or tile: "
                             f"{list(result.overflow)}")
            self.assertEqual(result.geometry_unavailable, (),
                             f"every item's geometry must be readable: "
                             f"{list(result.geometry_unavailable)}")
            # **Every clip is named.** An unnamed clipping ancestor would report as the bare
            # string `ancestor:`, and a reader could not tell the two card columns from the
            # card from the window. The two columns are named precisely so this can be said.
            for name, source in result.clip_sources.items():
                self.assertTrue(source == "window" or source.startswith("ancestor:")
                                and len(source) > len("ancestor:"),
                                f"{name} is clipped by {source!r}, which names nothing; a "
                                f"report that cannot say which box overflowed cannot be acted on")
            # **And the card clips are actually in use**, which is the difference between
            # "nothing overflowed" and "nothing was measured".
            sources = set(result.clip_sources.values())
            self.assertTrue(any(s.startswith("ancestor:card_") for s in sources),
                            f"at {scale}% no card was used as a clip, so the measure is still "
                            f"window-only: {sorted(sources)}")
            self.assertTrue(GRID["main"] > 0, "sanity: the grid still has a main band")

    def test_no_expected_item_is_absent_or_unreachable(self):
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            self.assertEqual(result.absent, (),
                             f"at {scale}% these expected items were not found: "
                             f"{list(result.absent)}. The harness would otherwise report a "
                             f"screen it had not seen.")
            self.assertEqual(result.unreachable, (),
                             f"at {scale}% the walk did not complete somewhere: "
                             f"{list(result.unreachable)}")

    def test_a_non_text_item_is_not_reported_as_a_blank_string(self):
        result = self.result(100)
        for name in ("heroTile", "thumb_B", "card_A"):
            self.assertNotIn(name, result.strings,
                             f"{name} is an Item/Rectangle with no `text` property; putting "
                             f"'' in strings for it would make 'this label is blank' true "
                             f"about a box: {sorted(result.strings)}")

    def test_the_seed_provenance_names_the_committed_file_for_every_figure(self):
        result = self.result(100)
        for key in ("age_p50_ms", "age_p95_ms", "backlog", "analysis_fps",
                    "window_samples", "failed_reason"):
            self.assertIn(key, result.seed_provenance,
                          f"every seeded figure must name where it came from: "
                          f"{sorted(result.seed_provenance)}")
            self.assertIn("m2-headless", result.seed_provenance[key],
                          f"{key} must trace to a committed measurement, not to this file: "
                          f"{result.seed_provenance[key]!r}")
        self.assertIn("not measured", result.seed_provenance["failed_state"],
                      "the failed route is seeded, and the provenance has to say so")


# --------------------------------------------------------------------------------------
# The harness's own guarantees about itself
# --------------------------------------------------------------------------------------

class HarnessSelfChecks(unittest.TestCase):

    def test_a_fifth_scale_is_refused_rather_than_measured(self):
        with self.assertRaises(ValueError) as caught:
            capture_ui.capture(scale_percent=175)
        self.assertIn("scale_percent must be one of", str(caught.exception))
        self.assertEqual(capture_ui.SCALES, (100, 125, 150, 200),
                         "the M3 gate names exactly these four")

    def test_a_required_nonempty_string_that_is_empty_stops_the_capture(self):
        """**I2, proved on a seed capture and not only on provisioned hardware.**

        `expect_nonempty` is what separates "the item was built" from "the item shows
        something". On the four committed captures every required string happens to be
        non-empty, so a capture that ignored `expect_nonempty` would look identical there.
        This asks for a name that is **legitimately blank on a correct screen** - camera A's
        reason line - and requires it to be non-empty, which no correct screen satisfies.

        A poll that ignored `expect_nonempty` would settle on the first look and return a
        result, so the capture would **not** raise. The assertion below is therefore the
        difference between a readiness check and a construction check, and it needs no
        inference worker: it is the `stream_confirmed_labels`-class hole, reached without the
        live path.
        """
        with tempfile.TemporaryDirectory(prefix="capture-empty-") as tmp:
            with self.assertRaises(RuntimeError) as caught:
                capture_ui.capture(
                    scale_percent=100, seconds=1.0, out_dir=pathlib.Path(tmp),
                    expect=("streamReasonText_A",),
                    expect_nonempty=("streamReasonText_A",))
        message = str(caught.exception)
        self.assertIn("empty_nonempty", message,
                      f"the failure must name the empty required string, not merely time out: "
                      f"{message[-400:]}")
        self.assertIn("streamReasonText_A", message,
                      f"and it must be that string that was empty: {message[-400:]}")

    def test_a_missing_measurement_raises_rather_than_defaulting(self):
        # **The honesty test.** A capture seeded with invented figures measures nothing, so a
        # missing committed measurement must be an error the caller has to handle - not a
        # silent fallback to a plausible number.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(capture_ui.NotProvisioned) as caught:
                capture_ui.load_measured_seed(
                    steady=pathlib.Path(tmp) / "nope.json",
                    fault=REPO_ROOT / capture_ui.MEASURED_FAULT)
        self.assertIn("nope.json", str(caught.exception))
        self.assertIn("invented", str(caught.exception).lower() + " ")

    def test_the_seed_is_the_committed_measurement_and_nothing_else(self):
        seed = capture_ui.load_measured_seed()
        steady = json.loads(
            (REPO_ROOT / capture_ui.MEASURED_STEADY).read_text(encoding="utf-8"))
        self.assertEqual(len(seed["streams"]), len(steady["per_stream"]))
        for entry, source in zip(seed["streams"], steady["per_stream"]):
            self.assertEqual(entry["stream_id"], source["stream_id"])
            self.assertEqual(entry["age_p50_ms"], source["frame_age_p50_ms"])
            self.assertEqual(entry["age_p95_ms"], source["frame_age_p95_ms"])
            self.assertEqual(entry["backlog"], source["backlog"])
            self.assertEqual(entry["window_samples"], source["window_samples"])
            self.assertAlmostEqual(
                entry["analysis_fps"],
                source["frames_analysed"] / steady["seconds"], places=2)
        self.assertEqual(seed["failed_stream"], steady["per_stream"][-1]["stream_id"])
        # The figures the M3 report will quote, so a reader can see them without the JSON.
        ages = sorted(round(e["age_p50_ms"] / 1000, 1) for e in seed["streams"])
        self.assertEqual(ages, [10.7, 11.1, 11.7],
                         f"the committed hold's per-stream p50 ages in seconds: {ages}")

    def test_nothing_in_app_imports_tools(self):
        # The dependency runs one way: `tools/` may import `app/`, never the reverse.
        offenders = []
        for path in sorted((REPO_ROOT / "app").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(name == "tools" or name.startswith("tools.") for name in names):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [],
                         f"app/ must not import tools/: {offenders}. The harness in tools/ "
                         f"reads the scene graph; the application must not know it exists.")

    def test_the_harness_does_import_the_application_and_only_from_the_allowed_direction(self):
        source = (REPO_ROOT / "tools" / "capture_ui.py").read_text(encoding="utf-8")
        self.assertIn("from app.theme import CANVAS_HEIGHT, CANVAS_WIDTH", source,
                      "the canvas defaults must come from app.theme, so the tool cannot "
                      "disagree with the screen about how big the screen is")
        self.assertNotIn("import pytest", source)
        self.assertNotIn("import numpy", source,
                         "the harness needs no dependency beyond the pinned venv")

    def test_the_cli_writes_a_png_and_a_json_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = capture_ui.main(["--scale", "100", "--out", tmp, "--seconds", "10"])
            self.assertEqual(code, 0)
            png = pathlib.Path(tmp) / "100pct.png"
            data = pathlib.Path(tmp) / "100pct.json"
            self.assertTrue(png.is_file())
            self.assertTrue(data.is_file())
            payload = json.loads(data.read_text(encoding="utf-8"))
            for key in ("pixel_width", "pixel_height", "strings", "text_property",
                        "truncated", "overflow", "absent", "unreachable", "opened_during_read",
                        "seed_provenance", "walk_failures", "settle_polls", "glyph_scale",
                        "glyph_pixels"):
                self.assertIn(key, payload, f"the JSON is the record; {key} must be in it")
            self.assertEqual((payload["pixel_width"], payload["pixel_height"]), (1920, 1080))
            self.assertEqual(payload["strings"]["deviceBadgeText"], "device: CPU")
            self.assertEqual(payload["glyph_scale"], [1.0, 1.0],
                             "at 100% the device-to-logical ratio is 1.0, and the record says "
                             "so - which is the scale factor the pixel count used")

    def test_the_caption_the_instrument_checks_is_named_in_the_qml(self):
        """**The `objectName`, pinned, because an unnamed label is a label nobody can read.**

        The caption `Text` in `VideoTile.qml` had no `objectName`, so `walk_items` never yielded
        it, it was absent from `strings` in all four committed captures, and
        `BOOTH_EXPECTED`'s comment claimed `thumb_B`/`thumb_C` were listed *because* they are the
        captions that rendered `Camera B · -1` - which was false. **If `stream_confirmed_labels`
        had regressed, all four committed JSONs would have come out byte-identical and the CLI
        would still have exited 0.** The end-to-end assertions are in `BoothCapture`; this is the
        guard that names the mechanism, so removing the name is a failure with a message rather
        than a silent hole in the instrument.
        """
        tile = (REPO_ROOT / "app" / "ui" / "VideoTile.qml").read_text(encoding="utf-8")
        self.assertIn('objectName: "tileCaptionText_" + tile.streamId', tile,
                      "the caption Text must be named per stream, or the harness cannot read "
                      "the caption and the defect it was built to catch is invisible to it")
        self.assertEqual(tile.count('objectName: "tileCaptionText_"'), 1,
                         "one caption Text, one name; a second would mean the string is "
                         "reported under a name nothing else agrees with")
        # **And it is one of the names the harness waits for and asserts on**, so removing it from
        # either list would fail rather than quietly reduce coverage.
        self.assertIn("tileCaptionText_B", capture_ui.BOOTH_EXPECTED)
        self.assertIn("tileCaptionText_C", capture_ui.BOOTH_EXPECTED)
        self.assertIn("tileCaptionText_B", capture_ui.BOOTH_NONEMPTY)
        self.assertIn("tileCaptionText_C", capture_ui.BOOTH_NONEMPTY)

    def test_the_cli_fails_loudly_when_it_could_not_account_for_an_item(self):
        # The CLI's exit code is a claim about the captures, so it must be driven by the
        # account and not by whether a file appeared.
        source = (REPO_ROOT / "tools" / "capture_ui.py").read_text(encoding="utf-8")
        self.assertIn("if result.absent or result.unreachable or result.geometry_unavailable:",
                      source,
                      "a capture with an item the harness cannot account for must set the "
                      "exit code, or the CLI's zero would mean only 'a file appeared'")


class LiveCaptureOnTheRealApplication(unittest.TestCase):
    """The harness reading the **real application**, not a seeded bridge.

    `tests/test_main.py::LiveWindowedPath` drives `app.main.main(["--windowed"])` and reads the
    live scene graph; that is the pattern. This asks the harness to do the same, so the
    harness is proved on the application's own window and not only on a scene it built.

    Needs the provisioned clip and IR. Without them it **skips with the reason printed** -
    passing silently is the failure this module exists to prevent.
    """

    @classmethod
    def setUpClass(cls):
        missing = []
        for candidate in ("assets/media/retail-checkout.mp4", "app/main.py"):
            if not (REPO_ROOT / candidate).exists():
                missing.append(candidate)
        from app.main import _missing_provisions
        reason = _missing_provisions(("clip", "ir", "cpu")) or (
            ", ".join(missing) if missing else "")
        if reason:
            raise unittest.SkipTest(
                f"cannot run the real application: {reason}. No claim is made about the live "
                f"path on this machine, and none is passed off as one.")
        cls._tmp = tempfile.TemporaryDirectory(prefix="capture-live-")
        try:
            cls.result = capture_ui.capture(
                scale_percent=100, seconds=14.0, device="CPU",
                out_dir=pathlib.Path(cls._tmp.name),
                expect=("deviceBadgeText", "heroTile", "card_A",
                        "streamAgeText_A", "streamBadgeText_A", "tileNoFrameText_A",
                        "eventCountRow"),
                # **The two names whose emptiness would be a rendering failure.** `expect` says
                # "was it built"; these say "does it show something". `expect_nonempty` is what
                # refuses a capture in which a required binding came out `""`, and it is proved
                # deterministically (with no hardware) by
                # `HarnessSelfChecks.test_a_required_nonempty_string_that_is_empty_stops_the_capture`.
                expect_nonempty=("deviceBadgeText", "streamBadgeText_A",
                                 "streamAgeText_A"),
                # **Named as expected-to-be-missing.** The live run has one camera, so
                # `card_B` does not exist and never will; waiting for it would time the
                # capture out on a correct screen.
                absent_expected=("card_B",))
        except capture_ui.NotProvisioned as exc:
            raise unittest.SkipTest(f"the live capture needs the provisioned assets: {exc}")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_the_live_mode_reports_that_it_ran_the_real_application(self):
        self.assertEqual(self.result.mode, "live")

    def test_the_harness_reads_the_applications_own_scene_graph(self):
        result = self.result
        self.assertTrue(result.settled, f"missing {result.settle_missed}")
        self.assertEqual(result.settle_missed, ())
        # **I2, and a correction to the review's diagnosis.** The review asked this test to
        # assert `settle_polls > 1`, on the reasoning that the live path "reads the screen before
        # telemetry arrives". **Measured on this machine: it does not.** `run_live` schedules the
        # grab at `seconds * 1000` ms (14 000 ms here) and only then begins polling, so by the
        # first poll every required string is already non-empty - measured `settle_polls == 1`,
        # `elapsed_s` 17.76 s, `streamBadgeText_A == "CPU"`, `streamAgeText_A == "53.2 ms"`.
        # Asserting `> 1` therefore fails on a **correct** screen (it did, in a 1317-test suite
        # run), which is the "assert a timing accident" defect this project has already recorded.
        # So the count is asserted as what this capture actually produces, and the property the
        # review wanted - a capture whose bindings are empty must **not** settle - is proved
        # deterministically, with no hardware and no timing, by
        # `HarnessSelfChecks.test_a_required_nonempty_string_that_is_empty_stops_the_capture`.
        self.assertGreaterEqual(result.settle_polls, 1,
                                f"the live capture must have polled at least once; it polled "
                                f"{result.settle_polls}. `settle_polls` is reported so this is a "
                                f"checkable value and not a construction truth.")
        self.assertEqual(result.empty_nonempty, (),
                         f"these were present but showed nothing: "
                         f"{list(result.empty_nonempty)}")
        self.assertEqual([name for name in result.absent if name != "card_B"], [],
                         f"only the camera that was deliberately not built may be absent: "
                         f"{list(result.absent)}")
        self.assertEqual(result.strings["deviceBadgeText"], "device: CPU",
                         "the badge is read off the rendered Text, which is the only way to "
                         "tell a bridge that updated from a screen that did not")
        self.assertEqual(result.strings["streamBadgeText_A"], "CPU")
        self.assertEqual((result.pixel_width, result.pixel_height), (1920, 1080),
                         "a live capture at 100% is 1920x1080 pixels")
        self.assertTrue(result.png_path.is_file())

    def test_a_camera_that_was_never_built_is_absent_and_not_unreachable(self):
        # The live run has **one** camera, so `card_B` genuinely does not exist. The harness
        # must call that **absent** - not silently omit it, and not call it unreachable.
        # Those are different claims: "absent" says one camera was built; "unreachable" says
        # the walk could not tell; and a harness that reported neither would be a harness that
        # reports what it happens to have found.
        result = self.result
        self.assertEqual(result.absent, ("card_B",),
                         f"the one camera that was not built: {list(result.absent)}")
        self.assertEqual(result.unreachable, ())
        self.assertTrue(result.walk_complete)
        self.assertNotIn("card_B", result.strings)


if __name__ == "__main__":
    unittest.main()