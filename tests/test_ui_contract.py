"""The QML booth screen: the mandatory copy, null-not-zero, and a real offscreen render.

Three of these tests are the brief's, and two of those three are kept verbatim because
they say what they mean. The rest exist because two milestones in this project have
produced tests that passed without the thing they claimed to test, and this file is
pinned against that.

**What is verified here, and how:**

- The mandatory label exists **in the QML sources**, not merely in Python. Spec §4 says it
  is displayed continuously; its absence was a real defect in the previous build, so a
  Python constant alone would not catch a QML file that dropped the binding.
- An absent optional metric renders as the word `unavailable`, and a **genuine `0.0`
  renders as a number**. Both halves. A bridge that rendered `0.0` as "unavailable" would
  pass the first test and be wrong in the other direction.
- The displayed device is the **confirmed** one, and a fallback is visible as a fallback.
- The window formatter **branches on `validity` before touching `window_seconds`**, which
  is the obligation that arrives with Ruling 23: `window_seconds == 0.0` is legal exactly
  on an `UNAVAILABLE` sample, and the only in-tree consumer that would divide by it was
  going to be this one.
- A coasted track produces **no box**. `Tracker.update()` returns coasted tracks on
  purpose; the filter lives in `UiBridge.set_overlay`. See `OverlayContract` below for
  the mutation that proves the test can fail.
- **The QML loads and renders offscreen at exactly 1920x1080**, and the render is checked
  for content rather than for having happened: a dark background, glyphs inside the
  label's rectangle, the video frame's own pixels on screen, and a populated metrics
  column.

## The offscreen render, and the font trap

`QT_QPA_PLATFORM=offscreen` gives **no fonts at all** on this machine. Measured:
`QFontDatabase.families()` returns **0 families under `offscreen`** and **411 under the
`windows` plugin**. An offscreen render with no font registered draws no glyphs, so a
"the label is on screen" assertion passes on a screen with no text on it - a vacuous pass
of exactly the kind this repository keeps finding. So:

- `add_application_font` registers `C:/Windows/Fonts/arial.ttf` before the render, and
- the render assertion checks for **light-on-dark pixels inside the label's rectangle**,
  which is impossible without glyphs.

## Why the render runs in a subprocess

`QGuiApplication` is a process-wide singleton and the offscreen platform plugin is chosen
at construction. Running the render in-process would make the rest of the suite depend on
which order the tests happen to run in. `_render_png` therefore runs a child process with
the platform pinned and returns the PNG path, so the assertion is on a file and this test
module needs no event loop of its own.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = REPO_ROOT / "app" / "ui"

#: The mandatory copy, exactly. Pinned here *and* in `app/ui/bridge.py` *and* in the
#: QML, and `test_the_label_constant_agrees_with_the_qml` asserts the three agree.
RECORDED_FOOTAGE_COPY = ("Recorded footage", "live inference")

#: The font registered for the offscreen render. See the module note on the font trap.
RENDER_FONT = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Fonts" / "arial.ttf"


def _qml_sources() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(UI_DIR.glob("*.qml")))


def _module_source(name: str) -> str:
    return (REPO_ROOT / "app" / name).read_text(encoding="utf-8")


def _bridge():
    from app.ui.bridge import UiBridge

    return UiBridge()


def _frame(fid: int = 0, loop: int = 0, width: int = 1920, height: int = 1080):
    import numpy as np

    from app.contracts.frame import FramePacket

    image = np.zeros((height, width, 3), np.uint8)
    return FramePacket(session_id="s", scenario_id="retail_sanity", stream_id="A",
                       frame_id=fid, source_pts=fid / 30, source_loop_id=loop,
                       captured_at=fid / 30, decoded_at=fid / 30,
                       width=width, height=height, image=image)


def _track(track_id: int = 1, box=(100.0, 100.0, 400.0, 800.0),
           confirmed: bool = True, misses: int = 0):
    """A real `Track`, not a stand-in.

    A stand-in with an `is_current` attribute would make the coasted-track test a test of
    the stand-in rather than of the filter. `Tracker` is the object whose contract this
    is, so the object is the real one.
    """
    from app.engine.tracker import Track

    return Track(track_id=track_id, class_id=0, box=tuple(float(v) for v in box),
                 first_seen=0.0, last_seen=0.0, hits=2, misses=misses,
                 confirmed=confirmed)


def _sample(name: str, value, *, validity, window: float = 0.25, now: float = 1.0):
    from app.contracts.metric_sample import MetricSample

    return MetricSample(
        name=name, value=value, unit="percent", provider="test",
        device_scope=None, process_scope=(), window_seconds=window,
        sampled_at=now, validity=validity,
        unavailable_reason=("nothing was measured" if value is None else None),
    )


#: The seed M1's render has always used: one stream, one confirmed CPU placement with a
#: fallback off the NPU, and telemetry that is all measured.
_SINGLE_STREAM_SEED = '''bridge.set_route(requested="NPU", confirmed=("CPU",), state="active")
bridge.set_metrics(system_cpu_percent=18.4, app_cpu_percent=12.0, ai_fps=14.9,
                   frame_age_p50_ms=63.0, frame_age_p95_ms=88.0)
bridge.set_streams(["A"])
'''

#: **M2 Task 5: the measured unsaturated `allcpu` condition**, spec section 5 row 4.
#:
#: Task 3's three consecutive 120 s holds, one stream each: frame age p50 **10767 /
#: 11085 / 11319 ms**, p95 **11626 / 11905 / 12286 ms**, backlog **170-187 frames**, about
#: **13.5 of the 15 fps** offered analysed, **zero drops**, and all three streams
#: `active`.
#:
#: That is a picture **eleven seconds old** while every counter reads healthy, and it is
#: the condition this screen has to make legible. Camera C is additionally given a real
#: failure with the wording Task 4 proved correct, so the per-stream reason channel is on
#: screen in the same render - the case the brief calls a card asking the presenter to
#: guess.
_ALLCPU_SEED = '''bridge.set_streams(["A", "B", "C"])
MEASURED_ALLCPU = [
    ("A", 10767.0, 11626.0, 170, 13.5, "active", []),
    ("B", 11085.0, 11905.0, 179, 13.4, "active", []),
    ("C", 11319.0, 12286.0, 187, 13.6, "failed",
     ["The restart budget for stream 'C' at generation 1 is spent, so this route is "
      "failed and no device is being claimed for it. The requested device 'CPU' is "
      "recorded; re-select one to try again."]),
]
for _sid, _p50, _p95, _backlog, _fps, _state, _errors in MEASURED_ALLCPU:
    bridge.apply_route({"stream": _sid, "requested": "CPU",
                        "confirmed": ["CPU"] if _state == "active" else [],
                        "state": _state, "fell_back": False, "errors": _errors})
    bridge.set_stream_metrics(_sid, age_p50_ms=_p50, age_p95_ms=_p95,
                              backlog=_backlog, analysis_fps=_fps,
                              window_samples=256)
'''


def _render_png(tmpdir: str, seed: str = _SINGLE_STREAM_SEED) -> str:
    """Render `app/ui/Main.qml` offscreen at 1920x1080 in a child process.

    Returns the PNG path. Raises on a non-zero exit so a QML syntax error fails loudly
    instead of leaving a stale or absent image for the assertions to read.
    """
    return _render(tmpdir, seed)[0]


def _render(tmpdir: str, seed: str = _SINGLE_STREAM_SEED,
            drawer: bool = False) -> tuple[str, dict, dict]:
    """`(png_path, {objectName: [text, visible, ...]}, per_camera_probe)`.

    The strings come out of the child's **scene graph**, read from the named items the QML
    declares - not from the bridge's Python properties, and not from the PNG. A bridge can
    hold a correct value while the screen is frozen at its first evaluation, and that is the
    defect class this repository has now found three times (the `constant=True` device
    badge, the `display_text()` card bindings, the `window_text()` line).

    `seed` chooses what state exists before the scene loads; see `_SINGLE_STREAM_SEED` and
    `_ALLCPU_SEED`. **`drawer` opens the presenter drawer after the load**, which is how the
    figures M4 moved off the always-visible screen (backlog, rate, window, activity,
    utilisation wording) are still read **as rendered**, not as bridge properties.
    """
    script = tmpdir.replace("\\", "/") + "/render.py"
    output = tmpdir.replace("\\", "/") + "/main.png"
    Path(script).write_text(_render_script(seed, drawer=drawer), encoding="utf-8")
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, script, output],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT), env=env,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"the offscreen render failed (exit {completed.returncode}).\n"
            f"stdout: {completed.stdout[-4000:]}\nstderr: {completed.stderr[-4000:]}"
        )
    strings: dict = {}
    for line in completed.stdout.splitlines():
        if line.startswith("SCENESTRINGS "):
            strings = json.loads(line[len("SCENESTRINGS "):])
        elif line.startswith("FRAMEPROBE "):
            probe = json.loads(line[len("FRAMEPROBE "):])
            probe["requests"] = [(raw, bool(served)) for raw, served in probe["requests"]]
    if not strings:
        raise AssertionError(
            "the render published no scene strings, so every string assertion would be "
            "reading nothing - which is exactly how the M1 card-text assertion proved "
            "two of three keys.\nstdout: " + completed.stdout[-3000:])
    return output, strings, probe


#: **Three cameras on three devices**, which is the layout the per-device wording is really
#: about. `AUTO`/`HETERO` are not involved and the confirmed readings are the real ones this
#: machine reports: CPU, `GPU.0`, NPU.
#:
#: Rendered as a **synthetic route**, not measured: the point is which wording reaches which
#: card, and that does not need the hardware. What it does need is a camera on each device,
#: which is exactly what the booth's single-stream run cannot provide.
_THREE_DEVICE_SEED = '''bridge.set_streams(["A", "B", "C"])
THREE_DEVICES = [
    ("A", "CPU", ["CPU"]),
    ("B", "GPU", ["GPU.0"]),
    ("C", "NPU", ["NPU"]),
]
for _sid, _requested, _confirmed in THREE_DEVICES:
    bridge.apply_route({"stream": _sid, "requested": _requested,
                        "confirmed": _confirmed, "state": "active",
                        "fell_back": False, "errors": []})
    bridge.set_stream_metrics(_sid, age_p50_ms=315.5, age_p95_ms=902.4, backlog=3,
                              analysis_fps=14.9, window_samples=256)
'''


def _render_script(seed: str = _SINGLE_STREAM_SEED, drawer: bool = False) -> str:
    """The child-process program, built with the repo's real paths substituted in.

    Written as one piece with `%`-style placeholders rather than a separate `.py` file in
    `app/ui/`, because a `.py` under `app/ui/` would be imported by nothing and would be a
    second copy of the UI wiring that could drift from `app.main`'s.

    `seed` is the block that publishes route and metric state **before** the scene loads.
    It is a parameter rather than a second template so that the one-stream M1 render and
    the three-stream M2 render exercise the *same* QML, the same font registration, the same
    grab and the same string dump - a second template would be a second thing that can
    drift, which is how M0 and M1 each produced a vacuous assertion.

    `drawer` opens the presenter drawer after the scene exists, so a test can read the
    figures M4 moved off the always-visible screen **as rendered**.
    """
    template = _RENDER_TEMPLATE
    # The paths go in as **repr'd literals**, not raw text. The first version substituted
    # them straight into the source and the render died with
    # `(unicode error) truncated \uXXXX escape` at `shutil.copytree("C:\\dev\\...")`: a
    # Windows path contains `\u` and `\d` sequences that Python's own parser tries to
    # decode. `repr()` escapes every backslash, which is the whole fix.
    return (template
            .replace("@REPO@", repr(str(REPO_ROOT)))
            .replace("@FONT@", repr(str(RENDER_FONT)))
            .replace("@UI@", repr(str(UI_DIR)))
            .replace("@DRAWER@", "window.setProperty('drawerOpen', True)"
                    if drawer else "pass")
            .replace("@SEED@", seed))


_RENDER_TEMPLATE = '''
"""Render app/ui/Main.qml offscreen and save a PNG. Run in its own process."""
import json
import os
import shutil
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, @REPO@)

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtGui import QFontDatabase, QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickWindow

from app.engine.tracker import Track
from app.ui.bridge import FrameImageProvider, ThemeTokens, UiBridge

out = sys.argv[1]
work = out + "_ui"

app = QGuiApplication(sys.argv)

# Measured on this machine: 0 font families exist under the offscreen plugin and 411
# under the windows plugin, so without this every glyph renders as nothing and a
# "the label is visible" assertion passes on a screen with no text on it.
font_id = QFontDatabase.addApplicationFont(@FONT@)
if font_id < 0:
    print("could not register the render font", file=sys.stderr)
    sys.exit(3)

bridge = UiBridge(app)
theme = ThemeTokens(app)
# **M2 Task 5: declare the three cameras before the scene loads.** The metrics column is a
# `Repeater` over `bridge.stream_ids`, and `stream_ids` is a *declaration* rather than
# something inferred from what has arrived - so with no declaration the column is empty and
# this render is a render of nothing. That is the same cold-start defect a booth screen
# would have, which is why `run_windowed` declares too, and it is declared **here** rather
# than defaulted in the bridge: a default would hide a caller that forgets.
bridge.set_streams(["A", "B", "C"])


class FakeFrame:
    """A minimal stand-in for a `FramePacket`.

    Only the attributes `UiBridge` reads are provided: identity, extents, image and - **M2,
    per-camera frames** - the stream it belongs to. The identity is the real four-tuple
    shape, because that is what `accept_result` compares.
    """

    identity = ("selftest", "A", 0, 0)
    width = 1920
    height = 1080
    stream_id = "A"


def _image():
    image = np.zeros((1080, 1920, 3), np.uint8)
    image[:, :, 1] = 90                       # green cast over the whole frame
    image[240:820, 620:1010, 0] = 200         # a CYAN block, for the overlay to sit on
    image[240:820, 620:1010, 2] = 200         # B and R high, G low
    return image


FakeFrame.image = _image()


class FakeZone:
    name = "counter_band"
    roi = (0.0, 0.20, 1.00, 0.60)


def track(track_id, box, misses):
    return Track(track_id=track_id, class_id=0, box=tuple(float(v) for v in box),
                 first_seen=0.0, last_seen=0.0, hits=2, misses=misses, confirmed=True)


frame = FakeFrame()
bridge.set_frame_image(frame)
bridge.set_zones([FakeZone()], frame)
# One current track and one coasted track. Only the current one may be drawn.
bridge.set_overlay([track(1, (620, 240, 1010, 820), 0),
                    track(2, (1250, 260, 1600, 900), 1)], frame)
@SEED@
bridge.set_status("running the retail_checkout clip")

# **M2 Task 5: the render also reports what it drew.** A PNG proves pixels happened; the
# scene graph's own strings are what the card tests assert on, so the render publishes them
# rather than only saving an image. The walk starts at `window.contentItem()` and uses
# `childItems()` because a `Repeater` delegate is a **visual** child: `QObject.findChild`
# and `QObject.children()` both fail to reach it, which is how a test that reads a
# delegate's named items by `findChild` silently gets `""` for all of them.
NAMES = {}


def walk_items(item):
    yield item
    for child in item.childItems():
        yield from walk_items(child)


def dump_strings():
    for item in walk_items(window.contentItem()):
        try:
            name = str(item.objectName())
        except Exception:
            continue
        if not name:
            continue
        NAMES[name] = [str(item.property("text")), bool(item.property("visible"))]
        if name == "heroTile":
            # **M2: the hero's own `streamId`.** A hero hardcoded to camera A and a hero
            # following `route_stream` leave exactly the same *set of named items* on the
            # screen, so the difference is invisible unless the hero has a name and its own
            # read. It was a surviving mutant until this existed.
            NAMES[name] = [str(item.property("streamId"))]
            continue
        if name.startswith("thumb_"):
            # **M2: a thumbnail's caption**, because the caption is a claim about which
            # camera is on the surface and it is only checkable by reading it.
            NAMES[name] = [str(item.property("caption"))]
            continue
        if name.startswith("tile"):
            # **M2: a tile's source, source size and painted width**, because "the hero is
            # empty" and "the hero is bound to the wrong URL" look identical in pixels and
            # the difference is one property away.
            _size = item.property("sourceSize")
            # `sourceSize` is `None` on an `Item` that is not an image - the dump reads
            # every named item, and a bare `Item` has no size at all. `None` is written as
            # `""` rather than raising, because a missing size on a non-image is not a
            # failure of anything.
            _size_text = ("" if _size is None
                          else "%dx%d" % (_size.width(), _size.height()))
            NAMES[name] = NAMES[name] + [str(item.property("source")),
                                          _size_text,
                                          str(item.property("paintedWidth"))]
    print("SCENESTRINGS " + json.dumps(NAMES, sort_keys=True))
    # **M2: the per-camera probe.** Which camera has a frame, which has not, and **every
    # id the image provider was actually asked for** - the last of those is the only place
    # a cross-wired tile shows up, and it is what the three-surface tests assert on.
    _probe = bridge.frame_for("A")
    print("FRAMEPROBE " + json.dumps({
        "frame": None if _probe is None else "%dx%d" % (_probe.width(), _probe.height()),
        "ready": dict(bridge.stream_frames_ready),
        "revisions": dict(bridge.stream_frame_revisions),
        "requests": [[str(raw), bool(served)] for raw, served in provider.requests[:16]],
    }))


shutil.rmtree(work, ignore_errors=True)
shutil.copytree(@UI@, work)

engine = QQmlApplicationEngine()
provider = FrameImageProvider(bridge)
engine.addImageProvider("frames", provider)
ctx = engine.rootContext()
ctx.setContextProperty("Theme", theme)
ctx.setContextProperty("bridge", bridge)
engine.load(QUrl.fromLocalFile(work + "/Main.qml"))

roots = engine.rootObjects()
if not roots:
    print("Main.qml produced no root objects", file=sys.stderr)
    sys.exit(4)
window = roots[0]
if not isinstance(window, QQuickWindow):
    print("root object is not a QQuickWindow", file=sys.stderr)
    sys.exit(5)

# **@DRAWER@: the presenter drawer, opened after the scene exists.** The figures M4 moved
# off the always-visible screen are read *as rendered* here rather than as bridge
# properties - a bridge can hold a correct backlog while the screen shows nothing.
@DRAWER@

loop = QEventLoop()
QTimer.singleShot(500, loop.quit)
loop.exec()

grab = window.grabWindow()
dump_strings()
if grab.isNull() or grab.width() != 1920 or grab.height() != 1080:
    print("grab is %sx%s" % (grab.width(), grab.height()), file=sys.stderr)
    sys.exit(6)
if not grab.save(out):
    print("could not save the render", file=sys.stderr)
    sys.exit(7)
print("rendered %dx%d with %d overlay boxes"
      % (grab.width(), grab.height(), len(bridge.overlay_boxes)))
shutil.rmtree(work, ignore_errors=True)
sys.exit(0)
'''


class ThreeStreamBoothLayout(unittest.TestCase):
    """**M2 Task 5.** Three cameras, three cards, and the one thing the screen must say.

    Every assertion here reads the **rendered strings out of the scene graph**, produced by
    one offscreen render seeded with the *measured* unsaturated `allcpu` condition - spec
    section 5 row 4, three streams on the CPU. That condition is: frame age p50
    **10767 / 11085 / 11319 ms**, p95 **11626 / 11905 / 12286 ms**, backlog **170-187**,
    **13.5 of 15 fps** analysed, **zero drops**, all three streams `active`.

    So the picture on screen is **eleven seconds old** and every other counter says the
    system is healthy. Nothing on the M1 screen said so. These tests are what says so.

    ## Two traps this class exists to avoid

    - **A named item that is not there reads as `""`.** The M1 card-text test asserted on
      `cardText_ai_fps`, which no item was ever named, and `"" != "unavailable"` passed for
      two of its three keys. So `test_every_item_this_class_reads_is_on_screen` comes
      **first** and asserts the items exist.
    - **A `Repeater` delegate is invisible to `QObject.findChild` and to
      `QObject.children()`.** It is a *visual* child, so it has to be reached with
      `window.contentItem()` and `childItems()`. The first version of the three-card reader
      used `findChild` and got `MISSING` for every card item while the PNG plainly showed
      all three cards drawn - a test that would have "passed" on nothing.
    """

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.mkdtemp(prefix="roadshow-allcpu-")
        cls._tmp = tmp
        cls.png, cls.strings, cls.probe = _render(tmp, _ALLCPU_SEED)
        # **M4: the figures that left the always-visible screen, read as rendered.** A
        # second render with the presenter drawer open, from the same seed, because the
        # drawer covers the metrics column and the warning-border test below is a check on
        # the *live* screen's pixels.
        cls._tmp_detail = tempfile.mkdtemp(prefix="roadshow-allcpu-detail-")
        _png, cls.detail_strings, _probe = _render(cls._tmp_detail, _ALLCPU_SEED,
                                                   drawer=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)
        shutil.rmtree(cls._tmp_detail, ignore_errors=True)

    def text(self, name: str) -> str:
        self.assertIn(name, self.strings,
                      f"{name} is not on screen; the items that are: "
                      f"{sorted(self.strings)}")
        return self.strings[name][0]

    def visible(self, name: str) -> bool:
        self.assertIn(name, self.strings, f"{name} is not on screen at all")
        return self.strings[name][1]

    def detail(self, name: str) -> str:
        """`name`'s rendered text **on the presenter drawer**, which M4 opened for this."""
        self.assertIn(name, self.detail_strings,
                      f"{name} is not on the drawer; the items that are: "
                      f"{sorted(self.detail_strings)}")
        return self.detail_strings[name][0]

    def test_every_item_this_class_reads_is_on_screen(self):
        # **First, and on purpose.** A missing item yields `""`, and every "is this not
        # empty / not zero" assertion on `""` passes. Naming the item first is the only
        # thing that stops that.
        #
        # **M4: two surfaces, two lists.** The ages, states, badges and verdict lines are on
        # the always-visible screen; the six detail figures are on the presenter drawer, so
        # each list is asserted against the render that has it.
        for sid in ("A", "B", "C"):
            for prefix in ("streamAgeText_", "streamStaleText_", "streamBadgeText_",
                           "streamReasonText_", "cameraName_", "cameraState_",
                           "verdictRow_"):
                self.assertIn(prefix + sid, self.strings,
                              f"camera {sid}'s {prefix} item is missing, so any "
                              f"assertion about its contents would be asserting about "
                              f"nothing: {sorted(self.strings)}")
            for prefix in ("streamWindowText_", "streamBacklogText_", "streamRateText_",
                           "streamActivityText_", "drawerCameraHeading_"):
                self.assertIn(prefix + sid, self.detail_strings,
                              f"camera {sid}'s {prefix} item is missing from the presenter "
                              f"drawer: {sorted(self.detail_strings)}")
            self.assertIn("card_" + sid, self.strings,
                          f"there is no card item for camera {sid}")
        self.assertEqual(len([n for n in self.strings if n.startswith("card_")
                              and not n.startswith(("cardColumn_", "cardExplainMouse_"))]), 3,
                         f"exactly three cameras, three cards: {sorted(self.strings)}")

    def test_the_eleven_second_old_picture_is_legible_as_stale_in_words(self):
        # The ruling. Each camera states its own age, **in words as well as a number**, and
        # the age in the sentence is the age in the numeral.
        #
        # **M4: exactly one verdict line per camera.** A and B are stale with nothing else
        # to say, so they carry the sentence; C has a stated failure reason, and the reason
        # is the more specific statement, so it takes the line. Both are asserted, and so is
        # the exclusivity - a camera cannot show two verdicts in a one-line budget, and
        # cannot show none.
        for sid, p50, p95, backlog in (("A", "10.8 s", "11.6 s", "170 frames"),
                                       ("B", "11.1 s", "11.9 s", "179 frames"),
                                       ("C", "11.3 s", "12.3 s", "187 frames")):
            self.assertEqual(self.text(f"streamAgeText_{sid}"), p50,
                             f"camera {sid}'s measured p50 frame age is not on its card")
            has_reason = self.visible(f"streamReasonText_{sid}")
            self.assertNotEqual(has_reason, self.visible(f"streamStaleText_{sid}"),
                                f"camera {sid} shows both verdicts or neither")
            if has_reason:
                reason = self.text(f"streamReasonText_{sid}")
                self.assertTrue(reason.startswith(f"Camera {sid}: "),
                                f"camera {sid}'s reason line must name the camera: {reason!r}")
            else:
                stale = self.text(f"streamStaleText_{sid}")
                self.assertIn("STALE", stale,
                              f"camera {sid}: eleven seconds old must be a WORD, not only a "
                              f"number. It reads {stale!r}")
                self.assertIn(p50, stale,
                              f"camera {sid}: the sentence must carry the age, so the number "
                              f"and the word cannot disagree: {stale!r}")
                self.assertIn("2 s", stale,
                              f"camera {sid}: the sentence must carry the limit it exceeds: "
                              f"{stale!r}")
            # **The window, the backlog, the rate and the p95 - on the presenter drawer
            # (M4).** All three measured, all three on the surface that says which camera
            # they are about, and all three read as rendered strings rather than as bridge
            # properties.
            self.assertEqual(self.detail(f"streamWindowText_{sid}"),
                             "window: 256 analysed frames",
                             f"camera {sid}: the percentiles are of 256 frames and the "
                             f"drawer must say so")
            self.assertEqual(self.detail(f"streamBacklogText_{sid}"),
                             f"backlog {backlog}",
                             f"camera {sid}'s backlog is not on its own drawer block")
            self.assertIn("analysis", self.detail(f"streamRateText_{sid}"))
            self.assertIn(p95, self.detail(f"streamActivityText_{sid}"),
                          f"camera {sid}'s p95 is not in its drawer block")

    def test_a_stale_camera_is_bordered_in_the_warning_colour(self):
        # The warning has to be visible across a room, not only to a reader of small text.
        # **Pixels, not a property**: a colour that is not on the screen cannot be asserted
        # by reading the value that would have produced it, which is the frozen-binding
        # shape this repository has found three times.
        import numpy as np

        from PySide6.QtGui import QImage

        image = QImage(self.png)
        buf = np.frombuffer(bytes(image.constBits()),
                            dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
        arr = buf[:, : image.width() * 4].reshape(
            image.height(), image.width(), 4)[:, :, :3].astype(int)
        # `Theme.COLOR_WARN` is #f0b429: high R, high G, low B. `QImage.Format_RGB32` is
        # stored **BGRA** on a little-endian host, so channel 0 is blue - getting that
        # backwards is why M1's first accent mask counted zero pixels on a render with
        # 13987 of them.
        warn = ((arr[:, :, 2] > 200) & (arr[:, :, 1] > 140) & (arr[:, :, 1] < 210)
                & (arr[:, :, 0] < 120))
        # The cards' left edges. Measured on this render: x 1313-1314 carries **408**
        # warning-coloured rows, which is **two** card borders - the two stale cameras. The
        # third is `failed`, and a failed card is outlined in the bad colour, not the
        # warning one, so three borders would be a bug.
        band = warn[:, 1313:1315].any(axis=1)
        rows = int(band.sum())
        self.assertGreater(
            rows, 300,
            f"only {rows} warning-coloured rows on the cards' left edges; the stale "
            f"outline is not being drawn. Measured 408 for two stale cards in this "
            f"layout, so the floor is well below the real figure and well above nothing.")

    def test_a_failed_camera_claims_no_device_and_states_its_own_reason(self):
        # The case the brief calls a card asking the presenter to guess.
        self.assertEqual(self.text("streamBadgeText_C"), "no device is running",
                         "a failed route must claim no device; it certainly must not "
                         "borrow the one that was requested")
        reason = self.text("streamReasonText_C")
        self.assertTrue(self.visible("streamReasonText_C"))
        self.assertIn("restart budget for stream 'C'", reason,
                      f"camera C's own reason must be on camera C's card: {reason!r}")
        # **M3 Task 2 moved this sentence off the card, deliberately, and the badge carries it.**
        #
        # The reason is cut in Python at `Theme.CARD_REASON_CHARS` with a marker naming what
        # is not shown, so "no device is being claimed for it" - **33 characters, first at
        # index 88 of this 190-character message** - is no longer on screen. What replaced it
        # is the **badge** three lines above, reading `no device is running`: four words
        # instead of six, and the assertion at the top of this test is the one that pins it.
        #
        # So the claim "a failed camera claims no device" is **still on the screen and still
        # asserted**, and what the card no longer shows is a *redundant restatement* of it. This
        # is recorded in the M3 Task 2 report as a change to what a visitor can read, because
        # trimming a line to make a layout fit without saying so is the failure this project
        # keeps finding.
        self.assertNotIn("no device is being claimed for it", reason,
                         "the reason is cut at a declared budget now; if the whole sentence "
                         "fits again, CARD_REASON_CHARS is larger than this test assumes and "
                         "the card is wasting the space")
        # **And the other two cards must not carry it.** This is the whole reason the
        # per-stream error channel exists.
        for sid in ("A", "B"):
            self.assertEqual(self.text(f"streamReasonText_{sid}"), "",
                             f"camera {sid} is active and healthy and must not display "
                             f"camera C's failure")
            self.assertFalse(self.visible(f"streamReasonText_{sid}"))
            self.assertEqual(self.text(f"streamBadgeText_{sid}"), "CPU",
                             f"camera {sid}'s confirmed placement is on its own card")

    def test_each_camera_carries_its_own_device_and_no_other_cameras(self):
        for sid in ("A", "B"):
            self.assertEqual(self.text(f"streamBadgeText_{sid}"), "CPU")
        self.assertEqual(self.text("streamBadgeText_C"), "no device is running")
        # The confirmed device drives the device's own utilisation wording. With three
        # cameras on three devices there is one of each; here all three are on the CPU, so
        # all three must carry the **CPU** wording - "no CPU sample yet", never the
        # accelerators' "Utilization unavailable". **M4: that line is in the presenter
        # drawer**, on the surface that names the device and its sampling window.
        self.assertEqual(self.detail("cardText_system_cpu_percent"), "no CPU sample yet")
        self.assertNotIn("cardText_gpu_utilisation_percent", self.detail_strings,
                         "no camera is on the GPU, so no line may claim a GPU metric")
        self.assertNotIn("cardText_", self.detail_strings,
                         f"a half-named cardText_ item exists, which is the exact shape "
                         f"the M1 test was rewritten to stop reaching for: "
                         f"{sorted(self.detail_strings)}")

    def test_a_camera_with_nothing_measured_reads_unavailable_and_never_zero(self):
        # A **second render**, seeded with three declared streams and nothing else: no
        # route, no metrics, no frames. This is the screen's first frame.
        tmp = tempfile.mkdtemp(prefix="roadshow-cold-")
        try:
            _png, strings, _probe = _render(tmp, 'bridge.set_streams(["A", "B", "C"])\n')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        for sid in ("A", "B", "C"):
            self.assertIn(f"streamAgeText_{sid}", strings,
                          "a cold card must still be a card; an empty metrics column is "
                          "indistinguishable from a broken screen")
            age, age_visible = strings[f"streamAgeText_{sid}"]
            self.assertEqual(age, "unavailable",
                             f"camera {sid} has analysed nothing, so its age must be the "
                             f"word and not 0.0 ms - it reads {age!r}")
            self.assertTrue(age_visible)
            self.assertEqual(strings[f"streamBadgeText_{sid}"][0],
                             "no device is running",
                             "nothing has confirmed anything, so no camera claims a "
                             "device")
            self.assertEqual(strings[f"streamReasonText_{sid}"][1], False,
                             "a pending camera has nothing to explain")
        # **The moved figures, on the record that renders them.** The cold state's drawer
        # block says `unavailable` for every optional figure; reading the record here rather
        # than the drawer is the same claim at one remove, and it is the only way to assert
        # it without a fourth child render - the record is what the drawer binds to, and
        # `test_the_record_reads_unavailable_and_never_zero` (below) proves the binding is
        # live.
        bridge = _bridge()
        bridge.set_streams(["A", "B", "C"])
        for sid in ("A", "B", "C"):
            record = bridge.stream_cards_by_id[sid]
            for field in ("backlog_text", "fps_text", "window_text"):
                self.assertEqual(record[field], "unavailable",
                                 f"camera {sid}'s {field} reads {record[field]!r}; "
                                 f"'0 frames of backlog' looks like a healthy empty queue")

    def test_the_overlay_geometry_is_recomputed_and_a_square_frame_would_not_catch_it(self):
        # **Obligation: overlay geometry must be checked against an independent
        # recomputation, and the test must not be vacuous on a square frame.**
        #
        # `app.main._frame_payload` normalises a track box by dividing x and w by
        # `frame.width` and y and h by `frame.height`. Getting either wrong still produces a
        # box inside [0, 1], so a bounds check cannot catch it - which is exactly what M1
        # recorded. The live test in `tests/test_main.py` catches it by agreeing with the
        # tracker's own boxes, and it runs on a 1920x1080 frame.
        #
        # This one runs with **no hardware at all**, on a deliberately lopsided frame, and
        # includes the negative control that proves the control has teeth.
        import numpy as np

        from app.engine.tracker import Track
        from app.main import _frame_payload

        class _Frame:
            # **640x360**, so the two divisors differ by a factor of 1.78 and a swap is
            # unmissable. A square frame would make them identical, which is why a
            # square-frame version of this test proves nothing.
            width = 640
            height = 360
            # The payload carries the pixels on to the bridge; a stand-in frame with no
            # `image` raises before the boxes are even built, which is the first version's
            # only failure.
            image = np.zeros((360, 640, 3), np.uint8)

            def __init__(self):
                self.identity = ("s", "A", 1, 0)
                self.stream_id = "A"

        class _Manifest:
            def zones_for_event_rules(self):
                return {}

        class _Tracker:
            def current_tracks(self):
                return [Track(track_id=7, class_id=0, box=(64.0, 36.0, 320.0, 180.0),
                              first_seen=0.0, last_seen=0.0, hits=2, misses=0,
                              confirmed=True)]

        payload = _frame_payload(_Frame(), _Tracker(), [], _Manifest())
        self.assertEqual(len(payload["boxes"]), 1, payload)
        box = payload["boxes"][0]
        # The correct normalisation, written out longhand.
        self.assertAlmostEqual(box["x"], 64.0 / 640.0, places=9)
        self.assertAlmostEqual(box["y"], 36.0 / 360.0, places=9)
        self.assertAlmostEqual(box["w"], (320.0 - 64.0) / 640.0, places=9)
        self.assertAlmostEqual(box["h"], (180.0 - 36.0) / 360.0, places=9)
        # **The negative control.** Swapping the divisors produces numbers that are all
        # inside [0, 1] and would pass every bounds check - and that differ from the
        # correct ones, so the agreement assertion above is doing the work.
        wrong_x, wrong_y = 64.0 / 360.0, 36.0 / 640.0
        for value in (wrong_x, wrong_y):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0, "a swapped divisor is still legal, which is "
                                           "the trap")
        self.assertNotAlmostEqual(wrong_x, box["x"], places=3)
        self.assertNotAlmostEqual(wrong_y, box["y"], places=3)


class PerStreamErrorChannel(unittest.TestCase):
    """**M2 Task 5's first job: a card showing `failed` with no reason is a card asking
    the presenter to guess.**

    Task 4 removed `errors` from the per-stream route payload and said why: `RoutePlan` had
    no per-stream error channel, so forwarding `plan.errors` under a per-stream label would
    show Camera C's failure carrying Camera A's and Camera B's sentences. This class is
    the channel itself, and it is tested on the contract rather than on the screen because
    the screen test (`ThreeStreamBoothLayout`) can only check what the bridge was given.
    """

    def _plan(self, **kwargs):
        from app.contracts.route_plan import RoutePlan

        base = dict(generation=1,
                    requested={"A": "CPU", "B": "CPU", "C": "CPU"},
                    confirmed={"A": ("CPU",), "B": ("CPU",)},
                    state="active",
                    stream_states={"A": "active", "B": "active", "C": "failed"})
        base.update(kwargs)
        return RoutePlan(**base)

    def test_a_failed_stream_must_carry_its_own_stated_reason(self):
        # The refusal this field exists to enforce. The aggregate `errors` may well hold the
        # reason - and putting it only there is what the card cannot use.
        from app.contracts.route_plan import RoutePlan

        with self.assertRaises(ValueError) as caught:
            RoutePlan(generation=1,
                      requested={"A": "CPU", "C": "CPU"},
                      confirmed={"A": ("CPU",)},
                      state="active",
                      errors=("something about a camera",),
                      stream_states={"A": "active", "C": "failed"})
        self.assertIn("C", str(caught.exception))
        self.assertIn("per-stream", str(caught.exception))

    def test_the_plan_is_accepted_once_each_failed_stream_has_its_reason(self):
        plan = self._plan(
            errors=("A note about stream 'C'",),
            stream_errors={"C": ("The restart budget for stream 'C' at generation 1 is "
                                 "spent, so this route is failed and no device is being "
                                 "claimed for it.",)})
        self.assertEqual(
            plan.stream_reasons("C"),
            ("The restart budget for stream 'C' at generation 1 is spent, so this route is "
             "failed and no device is being claimed for it.",))
        # **And the two healthy cameras have none of their own.**
        for sid in ("A", "B"):
            self.assertEqual(plan.stream_reasons(sid), (),
                             f"camera {sid} is active; it must not have borrowed C's "
                             f"reason from the aggregate")

    def test_the_aggregate_still_exists_and_is_still_the_concatenation(self):
        # Not replaced - removed would break M1 and M0's readers. It is simply no longer
        # what a card reads, and this test says which is which.
        plan = self._plan(
            errors=("first about A", "second about C"),
            stream_errors={"A": ("first about A",),
                           "C": ("second about C",)})
        self.assertEqual(plan.errors, ("first about A", "second about C"))
        self.assertEqual(plan.stream_reasons("A"), ("first about A",))
        self.assertEqual(plan.stream_reasons("C"), ("second about C",))

    def test_a_stream_that_is_not_failed_needs_no_reason(self):
        plan = self._plan(state="active",
                          stream_states={"A": "active", "B": "pending", "C": "pending"})
        self.assertEqual(dict(plan.stream_errors), {})

    def test_a_reason_for_a_stream_nobody_requested_is_refused(self):
        from app.contracts.route_plan import RoutePlan

        with self.assertRaises(ValueError) as caught:
            RoutePlan(generation=1, requested={"A": "CPU"},
                      state="active",
                      stream_errors={"Z": ("about a camera nobody asked for",)})
        self.assertIn("Z", str(caught.exception))

    def test_the_field_survives_validation_as_tuples_and_json(self):
        import json

        plan = self._plan(stream_errors={"C": ("reason one", "reason two")})
        self.assertEqual(plan.stream_errors["C"], ("reason one", "reason two"))
        self.assertIsInstance(plan.stream_errors["C"], tuple)
        # A mutable list in a frozen dataclass would be a shared-mutation bug.
        with self.assertRaises(Exception):
            plan.stream_errors["C"] = ("other",)
        payload = json.dumps(plan.to_json())
        self.assertIn('"stream_errors"', payload)


class StreamCardRecord(unittest.TestCase):
    """The card record, asserted in Python. No window, no hardware.

    The screen test proves the strings reach the screen; this proves each string is the
    honest one, including the cases the screen cannot easily be driven into - a stream that
    has measured nothing, and a route that failed after it had.
    """

    def _bridge(self, sid="A", device="CPU", state="active", confirmed=("CPU",),
                errors=(), p50=10767.0, p95=11626.0, backlog=170, fps=13.5,
                window=256, requested="CPU", fell_back=False):
        bridge = _bridge()
        bridge.set_streams([sid])
        bridge.apply_route({"stream": sid, "requested": requested,
                            "confirmed": list(confirmed), "state": state,
                            "fell_back": fell_back, "errors": list(errors)})
        bridge.set_stream_metrics(sid, age_p50_ms=p50, age_p95_ms=p95, backlog=backlog,
                                  analysis_fps=fps, window_samples=window)
        return bridge

    def test_a_measured_eleven_second_card_says_so_in_words(self):
        card = self._bridge().stream_card("A")
        self.assertEqual(card["age_p50_text"], "10.8 s", card)
        self.assertEqual(card["age_p95_text"], "11.6 s", card)
        self.assertEqual(card["backlog_text"], "170 frames", card)
        self.assertEqual(card["window_text"], "256 analysed frames", card)
        self.assertTrue(card["stale"], card)
        self.assertIn("STALE", card["staleness"])
        self.assertIn("10.8 s", card["staleness"])
        self.assertIn("2 s", card["staleness"])
        self.assertEqual(card["badge"], "CPU", card)
        self.assertEqual(card["device"], "CPU", card)

    def test_a_fresh_picture_is_not_called_stale(self):
        card = self._bridge(p50=315.5, p95=902.4).stream_card("A")
        self.assertEqual(card["age_p50_text"], "315.5 ms", card)
        self.assertFalse(card["stale"], card)
        self.assertEqual(card["staleness"], "",
                         "a fresh stream must not carry a STALE sentence with an empty "
                         "figure in it")

    def test_the_threshold_is_spec_section_sevens_two_seconds_and_it_is_exclusive(self):
        from app.ui.bridge import STALE_AFTER_SECONDS

        self.assertEqual(STALE_AFTER_SECONDS, 2.0)
        # **Strictly older, which is what the spec says**: "Values older than 2 s are
        # `stale`". So exactly 2.000 s is not stale and 2.001 s is. The first version of
        # this test asserted the inclusive reading and failed; the spec's word is "older",
        # and picking `>=` to make a test pass would be changing the document's meaning
        # rather than the code.
        self.assertFalse(self._bridge(p50=1999.0).stream_card("A")["stale"])
        self.assertFalse(self._bridge(p50=2000.0).stream_card("A")["stale"],
                         "2.000 s is not *older than* 2 s")
        self.assertTrue(self._bridge(p50=2001.0).stream_card("A")["stale"])

    def test_an_unmeasured_stream_reads_unavailable_on_every_figure(self):
        bridge = _bridge()
        bridge.set_streams(["A"])
        bridge.apply_route({"stream": "A", "requested": "CPU", "confirmed": [],
                            "state": "pending", "fell_back": False, "errors": []})
        # **No `set_stream_metrics` at all** - the state of the screen's first frame.
        card = bridge.stream_card("A")
        for field in ("age_p50_text", "age_p95_text", "fps_text", "backlog_text",
                      "window_text"):
            self.assertEqual(card[field], "unavailable",
                             f"{field} reads {card[field]!r} for a stream that has "
                             f"measured nothing; '0' here looks exactly like health")
        self.assertFalse(card["stale"], "an absent age is not a stale age")
        self.assertEqual(card["staleness"], "")
        self.assertEqual(card["badge"], "no device is running",
                         "nothing has confirmed a placement, so the badge claims none")
        # **The device line is a different question from the badge.** A pending camera that
        # asked for the CPU legitimately shows the CPU's own utilisation wording - that
        # counter exists and it is the CPU's - while claiming no placement. The *failed*
        # case below is the one that must name no device at all, because a camera with no
        # worker has no device whose utilisation could be relevant.
        self.assertEqual(card["device"], "CPU", card)
        self.assertEqual(card["state"], "pending")

    def test_a_failed_route_names_no_device_even_though_one_was_requested(self):
        card = self._bridge(state="failed", confirmed=(),
                            errors=("the reason",)).stream_card("A")
        self.assertEqual(card["badge"], "no device is running", card)
        self.assertEqual(card["device"], "", "a failed route must not borrow the "
                                             "requested device for its utilisation line")
        self.assertEqual(card["requested"], "CPU", card)
        self.assertEqual(card["reason"], "the reason", card)

    def test_a_failed_route_with_no_reason_says_so_rather_than_leaving_it_blank(self):
        card = self._bridge(state="failed", confirmed=()).stream_card("A")
        self.assertIn("no reason was stated", card["reason"],
                      "a blank reason line reads as a rendering failure and 'failed' with "
                      "nothing after it asks the presenter to guess")

    def test_a_fallback_shows_the_requested_device_alongside_the_confirmed_one(self):
        # `fell_back` is **published by the plan**, not computed by the bridge - the
        # confirmed readings are `GPU.0` and the request is `NPU`, and only
        # `app.inference.devices.is_device` knows that a mismatch is a real one rather than
        # a spelling difference. So the bridge stores what it is told, and this test
        # supplies the truth.
        card = self._bridge(requested="NPU", confirmed=("CPU",),
                            fell_back=True).stream_card("A")
        self.assertTrue(card["fell_back"], card)
        self.assertEqual(card["badge"], "CPU (fell back from NPU)", card)

    def test_the_dotted_gpu_reading_is_normalised_for_the_device_line(self):
        # `'GPU.0'` is what the GPU reports. A card that keyed its utilisation wording off
        # the raw string would find nothing and print "unavailable" for a camera whose
        # utilisation genuinely has no counter - the right answer for the wrong reason.
        card = self._bridge(device="GPU", requested="GPU",
                            confirmed=("GPU.0",)).stream_card("A")
        self.assertEqual(card["device"], "GPU", card)
        self.assertEqual(card["badge"], "GPU.0", card)

    def test_frame_age_and_inference_cost_are_never_merged_into_one_number(self):
        # The standing rule. On the NPU the measured p95 frame age is seconds and the
        # inference is about 5.9 ms; the gap is queue wait. Nothing here adds them, so this
        # test asserts the *shape*: the only age on the record is a frame age, in the age
        # fields, and no field mixes the two units.
        card = self._bridge().stream_card("A")
        age_fields = {k: v for k, v in card.items() if "age" in k}
        self.assertTrue(age_fields, card)
        for name, value in age_fields.items():
            self.assertNotIn("fps", str(value),
                             f"{name} reads {value!r}, which mixes a rate into an age")

    def test_declaring_a_cameras_notifies_the_scene_on_its_own(self):
        # `set_streams` must emit. A `Repeater` bound to `bridge.stream_ids` has **no other
        # way** to learn the set changed, and the whole point of declaring the cameras is
        # that the cards exist from the first rendered frame - a declaration that updates
        # Python and never tells QML produces an empty metrics column forever, which is
        # indistinguishable from a broken screen.
        #
        # Asserted on the signal itself with **no** route payload and **no** metrics, so it
        # cannot be satisfied by another path emitting the same signal.
        bridge = _bridge()
        route_ticks = []
        card_ticks = []
        bridge.streamStatesChanged.connect(lambda: route_ticks.append(1))
        bridge.streamCardsChanged.connect(lambda: card_ticks.append(1))
        bridge.set_streams(["A", "B", "C"])
        self.assertEqual(len(route_ticks), 1,
                         "`stream_ids` is notified by `streamStatesChanged`, and nothing "
                         "else emits it here")
        self.assertEqual(len(card_ticks), 1,
                         "and the cards themselves are notified, or the QML card bindings "
                         "would read a cold record")
        # Declaring the same set again changes nothing, so a 4 Hz caller cannot repaint.
        bridge.set_streams(["C", "B", "A"])
        self.assertEqual(route_ticks, [1], "re-declaring the same set is not a change")

    def test_declaring_a_stream_creates_a_card_that_claims_nothing(self):
        # `set_streams` is a declaration, not a measurement. A card for a camera that has
        # said nothing must say nothing.
        bridge = _bridge()
        bridge.set_streams(["A", "B", "C"])
        self.assertEqual(list(bridge.stream_ids), ["A", "B", "C"])
        for sid in ("A", "B", "C"):
            card = bridge.stream_card(sid)
            self.assertEqual(card["state"], "pending", card)
            self.assertEqual(card["age_p50_text"], "unavailable", card)
            self.assertEqual(card["badge"], "no device is running", card)

    def test_three_cameras_never_share_one_figures(self):
        bridge = _bridge()
        bridge.set_streams(["A", "B", "C"])
        for sid, p50, backlog in (("A", 10767.0, 170), ("B", 11085.0, 179),
                                  ("C", 11319.0, 187)):
            bridge.apply_route({"stream": sid, "requested": "CPU",
                                "confirmed": ["CPU"], "state": "active",
                                "fell_back": False, "errors": []})
            bridge.set_stream_metrics(sid, age_p50_ms=p50, age_p95_ms=p50 + 500.0,
                                      backlog=backlog, analysis_fps=13.5,
                                      window_samples=256)
        cards = {c["stream"]: c for c in bridge.stream_cards}
        self.assertEqual(len(cards), 3, cards)
        self.assertEqual([cards[s]["age_p50_text"] for s in "ABC"],
                         ["10.8 s", "11.1 s", "11.3 s"])
        self.assertEqual([cards[s]["backlog_text"] for s in "ABC"],
                         ["170 frames", "179 frames", "187 frames"])

    def test_publishing_the_window_notifies_so_a_card_binding_re_reads(self):
        bridge = _bridge()
        ticks = []
        bridge.streamCardsChanged.connect(lambda: ticks.append(1))
        bridge.set_stream_metrics("A", age_p50_ms=None, age_p95_ms=None, backlog=None,
                                  analysis_fps=None, window_samples=256)
        self.assertEqual(len(ticks), 1,
                         "the card's whole record comes from this one signal; without it "
                         "a QML binding reads the cold card for the whole session")

    def test_publishing_identical_metrics_does_not_notify(self):
        # A telemetry tick at 4 Hz must not repaint three cards 4 times a second when
        # nothing moved. Bounded by "only when a value changed", which is the same rule
        # `_metrics` follows.
        bridge = self._bridge()
        ticks = []
        bridge.streamCardsChanged.connect(lambda: ticks.append(1))
        bridge.set_stream_metrics("A", age_p50_ms=10767.0, age_p95_ms=11626.0,
                                  backlog=170, analysis_fps=13.5, window_samples=256)
        self.assertEqual(ticks, [])


    def test_each_camera_carries_its_own_device_wording_on_a_three_device_render(self):
        # **Obligation: the CPU camera must not wear the accelerator's wording.** M1 checked
        # this by asserting `unavailable_notice["GPU"] == "Utilization unavailable"` on a
        # screen that had one camera on the CPU - i.e. on cards that were not there. Now
        # there really is one camera per device, so each wording is read off the line that
        # carries it. **M4: that line is in the presenter drawer** (the always-visible
        # screen shows the device class, not a utilisation figure), so the render is taken
        # with the drawer open - still as rendered strings, never as bridge properties.
        tmp = tempfile.mkdtemp(prefix="roadshow-devices-")
        try:
            _png, strings, _probe = _render(tmp, _THREE_DEVICE_SEED, drawer=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        expected = {"system_cpu_percent": "no CPU sample yet",
                    "gpu_utilisation_percent": "Utilization unavailable",
                    "npu_utilisation_percent": "Utilization unavailable"}
        for metric, wording in expected.items():
            name = "cardText_" + metric
            self.assertIn(name, strings,
                          f"there must be a camera figure named for the {metric}: "
                          f"{sorted(strings)}")
            self.assertEqual(strings[name][0], wording,
                             f"{metric} renders {strings[name][0]!r}; a machine-wide "
                             f"GPU peak is NOT attributable to this process, so "
                             f"'Utilization unavailable' is the only honest wording and "
                             f"the CPU must never borrow it")
        # And the CPU is genuinely different - the assertion above is not satisfied by all
        # three saying the same thing.
        self.assertNotEqual(expected["system_cpu_percent"],
                            expected["gpu_utilisation_percent"])
        # Each camera's badge names its own confirmed reading, the dotted form included.
        self.assertEqual(strings["streamBadgeText_A"][0], "CPU")
        self.assertEqual(strings["streamBadgeText_B"][0], "GPU.0",
                         "the GPU's own EXECUTION_DEVICES spelling, not a tidied 'GPU'")
        self.assertEqual(strings["streamBadgeText_C"][0], "NPU")
        # None of these three cameras is stale, so none may carry the word.
        for sid in ("A", "B", "C"):
            self.assertEqual(strings[f"streamStaleText_{sid}"][1], False,
                             f"camera {sid} is at 315 ms; it must not be called stale")


#: **M2, closing the disclosed gap**: three declared cameras, **two with frames and one
#: without**, and the two frames are different **sizes** as well as colours. Different
#: sizes matter: a cross-wired tile that showed the other camera's pixels would still be
#: 1920x1080, so only the size makes the swap visible in `sourceSize` without decoding an
#: image. 960x540 and 480x270 against a 1x1 blank is unambiguous.
_PER_CAMERA_SEED = '''bridge.set_streams(["A", "B", "C"])


def _frame(width, height, channel, level):
    import numpy as _np
    img = _np.zeros((height, width, 3), _np.uint8)
    img[:, :, channel] = level
    return img


bridge.apply_route({"stream": "A", "requested": "CPU", "confirmed": ["CPU"],
                    "state": "active", "fell_back": False, "errors": []})
bridge.apply_route({"stream": "B", "requested": "GPU", "confirmed": [],
                    "state": "pending", "fell_back": False, "errors": []})
bridge.apply_route({"stream": "C", "requested": "NPU", "confirmed": ["NPU"],
                    "state": "active", "fell_back": False, "errors": []})
bridge.apply_frame({"stream": "A", "image": _frame(960, 540, 1, 200),
                    "width": 960, "height": 540, "zones": [], "boxes": [], "events": []})
bridge.apply_frame({"stream": "C", "image": _frame(480, 270, 2, 200),
                    "width": 480, "height": 270, "zones": [], "boxes": [], "events": []})
'''


class PerCameraVideoSurfaces(unittest.TestCase):
    """**M2: each tile serves its own camera's frames, or says it has none.**

    This closes the gap Task 5 disclosed: *"per-camera video surfaces are not built - the
    hero and M1's two honestly-labelled thumbnails are unchanged, and three live pictures
    need per-stream frames through `FrameImageProvider` and a feeder per stream."*

    Two offscreen renders, both seeded with three declared cameras:

    - **Camera A has a 960x540 frame, Camera B has none, Camera C has a 480x270 frame.**
    Different sizes, so a cross-wired tile is visible in the `sourceSize` the `Image`
    reports without decoding anything, and B's absence is visible as a 1x1.
    - **A second render with the hero pointed at Camera C**, to prove the hero follows the
      selection rather than always being Camera A.

    ## Every assertion is on a named item or the provider's own audit trail

    Three separate traps make a weaker version of this test pass: an absent item reads
    `""`, a `Repeater` delegate is invisible to `findChild`, and a provider that returns a
    1x1 transparent image draws **something** rather than nothing. So the tests read
    `objectName`s that must exist, and the provider's `requests` list, which records every
    id it was asked for and whether it served anything.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="roadshow-percam-")
        cls.png, cls.strings, cls.probe = _render(cls._tmp, _PER_CAMERA_SEED)
        cls._tmp2 = tempfile.mkdtemp(prefix="roadshow-herosel-")
        _png, cls.hero_strings, cls.hero_probe = _render(
            cls._tmp2, _PER_CAMERA_SEED + 'bridge.set_route_stream("C")\n')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)
        shutil.rmtree(cls._tmp2, ignore_errors=True)

    def tile(self, strings, sid):
        """`(source, sourceSize)` for one camera's `Image`, asserting the item exists."""
        name = f"tileImage_{sid}"
        self.assertIn(name, strings,
                      f"there is no {name} in the scene, so any assertion about which "
                      f"picture it shows would be asserting about nothing: "
                      f"{sorted(strings)}")
        return strings[name][2], strings[name][3]

    def test_each_tile_asks_for_its_own_stream_and_no_other_name(self):
        # **The loud part.** Every id the provider was asked for must be one of the three
        # camera ids - never `"hero"`, never `"thumb0"`, never another camera's. That was the
        # whole defect: every tile asked for `"hero"`, and all three surfaces showed
        # whichever camera delivered last.
        asked = [raw.split("?")[0] for raw, _served in self.probe["requests"]]
        self.assertTrue(asked, "the provider was asked for nothing at all, so this test "
                               "would pass over an empty scene")
        self.assertEqual(sorted(set(asked)), ["A", "B", "C"],
                         f"the provider must only ever be asked for a camera id; it was "
                         f"asked for {sorted(set(asked))}")
        for legacy in ("hero", "thumb0", "thumb1", ""):
            self.assertNotIn(legacy, asked,
                             f"'{legacy}' is the pre-M2 shared path. Every tile asking for "
                             f"one shared path is exactly how three cameras ended up "
                             f"showing one camera.")

    @staticmethod
    def _stream_of(url_text):
        """`"image://frames/A?v=1"` -> `"A"`.

        **The id is read the same way the provider reads it**, out of the `QUrl`'s last path
        segment with the query dropped. The first version of these tests did a
        `.split("?")[0]` on `str(QUrl)` and compared it against a hand-written string, which
        did not match `"PySide6.QtCore.QUrl('image://frames/A"` - so the assertion was
        comparing the wrong two things and would have failed for the wrong reason on any
        render that was actually correct.
        """
        text = str(url_text)
        start = text.find("image://")
        if start >= 0:
            text = text[start + len("image://"):]
        return text.split("?")[0].strip("') ").split("/")[-1].strip("') ")

    def test_a_tile_with_a_frame_serves_that_frame_and_only_that_frame(self):
        # The two sizes make a swap unmissable: A's is 960x540 and C's is 480x270, so a tile
        # showing the other's pixels reports the other's size.
        source, size = self.tile(self.strings, "A")
        self.assertEqual(self._stream_of(source), "A",
                         f"camera A's tile asks for {source!r}")
        self.assertIn("960x540", size,
                      f"camera A's tile must show A's 960x540 frame; it reports {size}")
        source, size = self.tile(self.strings, "C")
        self.assertEqual(self._stream_of(source), "C",
                         f"camera C's tile asks for {source!r}")
        self.assertIn("480x270", size,
                      f"camera C's tile must show C's 480x270 frame; it reports {size}")
        # And the two are served, so this is not a tile pair that happens to look right.
        served = {raw.split("?")[0] for raw, ok in self.probe["requests"] if ok}
        self.assertEqual(served, {"A", "C"},
                         f"the provider must serve exactly the two cameras that have "
                         f"frames: {sorted(served)}")

    def test_a_camera_with_no_frame_says_so_and_shows_no_picture(self):
        # **B has produced nothing.** The tile must say so in words, must not be painted,
        # and above all must not be showing A's or C's picture.
        self.assertNotIn("B", self.probe["ready"],
                         f"camera B is declared and has no frame, so it must be absent "
                         f"from the ready map, not present-and-false: "
                         f"{self.probe['ready']}")
        self.assertEqual(self.probe["revisions"].get("B"), None,
                         f"a camera with no frame has no revision; 0 would be "
                         f"indistinguishable from frame 0: {self.probe['revisions']}")
        source, size = self.tile(self.strings, "B")
        self.assertEqual(self._stream_of(source), "B",
                         f"camera B's tile asks for {source!r}")
        self.assertIn("1x1", size,
                      f"camera B has no frame, so its image must be the 1x1 blank and not "
                      f"another camera's pixels: it reports {size}")
        not_shown = f"tileNoFrame_{'B'}"
        self.assertIn(not_shown, self.strings, sorted(self.strings))
        self.assertTrue(self.strings[not_shown][1],
                        "camera B has no frame, so its 'no frame' panel must be VISIBLE - a "
                        "hidden panel with a blank picture is the same as nothing")
        words = self.strings[f"tileNoFrameText_B"][0]
        self.assertEqual(words, "no frame from Camera B yet",
                         f"the tile must say which camera has no frame, in words: {words!r}")
        # **And the reason it is blank is not a rendering failure.**
        not_served = [raw for raw, ok in self.probe["requests"] if not ok]
        self.assertTrue(any(raw.startswith("B") for raw in not_served),
                        f"the provider must have refused to serve something rather than "
                        f"silently substituting another camera's pixels: {not_served}")

    def test_the_hero_serves_the_bound_stream_and_follows_a_change_of_selection(self):
        # **The hero is Camera A** in the first render, and **Camera C** after
        # `set_route_stream("C")`. Three separate things move together - the picture, the
        # badge and the overlay boxes - and this is the assertion that they move *together*.
        #
        # **The hero's own `streamId` is read from the named hero item**, not inferred from
        # which tile happens to be largest. A hero hardcoded to `"A"` and a hero following
        # `route_stream` leave exactly the same set of named items on the screen - the first
        # version of this test passed against the hardcoded hero, because every assertion
        # was about a *tile* and the tiles are the same either way.
        self.assertIn("heroTile", self.strings, sorted(self.strings))
        self.assertEqual(self.strings["heroTile"][0], "A",
                         f"the hero must be bound to the default stream A; the item says "
                         f"{self.strings['heroTile'][0]!r}")
        self.assertEqual(self.hero_strings["heroTile"][0], "C",
                         f"and after set_route_stream('C') it must say C; it says "
                         f"{self.hero_strings['heroTile'][0]!r}")
        self.assertEqual(self._stream_of(self.strings["tileImage_A"][2]), "A",
                         "the default bound stream is Camera A")
        self.assertEqual(self._stream_of(self.hero_strings["tileImage_A"][2]), "A",
                         "the thumbnail for camera A is still camera A's own stream after "
                         "the hero moved")
        # Camera C's tile is the hero now, and it shows **C's** frame.
        hero_source, hero_size = self.tile(self.hero_strings, "C")
        self.assertEqual(self._stream_of(hero_source), "C", hero_source)
        self.assertIn("480x270", hero_size,
                      f"the hero must show the newly bound camera's own frame; it reports "
                      f"{hero_size}")
        # **The badge moved with it**, which is the "nothing describes a different camera"
        # property. `hero_strings['streamBadgeText_C']` is C's card; the hero's own badge is
        # the header's `deviceBadgeText`.
        self.assertIn("device: NPU", self.hero_strings["deviceBadgeText"][0],
                      f"the header badge must describe the newly bound camera: "
                      f"{self.hero_strings['deviceBadgeText'][0]!r}")
        self.assertIn("device: CPU", self.strings["deviceBadgeText"][0],
                      f"and in the default render it described camera A: "
                      f"{self.strings['deviceBadgeText'][0]!r}")

    def test_a_thumbnail_is_captioned_with_its_own_camera_and_its_own_device(self):
        # **The caption is a claim about the picture, and it is read, not assumed.**
        #
        # The first version of this binding was
        # `"Camera " + modelData + " · " + bridge.stream_routes[modelData].confirmed`,
        # and `confirmed` is a Python tuple crossing into QML as a `QVariantList`.
        # Concatenating a list with a string rendered the literal **`-1`** on both
        # thumbnails - on a render where the Python side held `confirmed: ('GPU.0',)` for
        # camera B and the picture above the caption was real. It was visible only by
        # reading the scene graph, which is why it is a test and not a review comment.
        #
        # A camera with no confirmed device must say so in words, must not print a number,
        # and must not borrow another camera's device.
        self.assertEqual(self.strings.get("thumb_B", [""])[0], "Camera B · not confirmed",
                         f"camera B is pending with no confirmed device in this render: "
                         f"{self.strings.get('thumb_B')!r}")
        self.assertEqual(self.strings.get("thumb_C", [""])[0], "Camera C · NPU",
                         f"camera C is confirmed on NPU in this render: "
                         f"{self.strings.get('thumb_C')!r}")
        self.assertNotIn("-1", self.strings.get("thumb_B", [""])[0],
                         f"the literal -1 is what a QVariantList stringifies to when it "
                         f"is concatenated instead of joined: "
                         f"{self.strings.get('thumb_B')!r}")
        # **And the negative control**: take camera C's confirmation away and its caption
        # must say so in words rather than printing a device or a number.
        empty = _render(tempfile.mkdtemp(prefix="roadshow-nocap-"),
                        _PER_CAMERA_SEED.replace('"confirmed": ["NPU"]',
                                                 '"confirmed": []'))
        self.assertIn("thumb_C", empty[1], sorted(empty[1]))
        self.assertEqual(empty[1]["thumb_C"][0], "Camera C · not confirmed",
                         f"an unconfirmed camera must say so in words: "
                         f"{empty[1]['thumb_C']!r}")

    def test_a_thumbnail_caption_follows_a_confirmation_that_arrives_after_the_load(self):
        # **The notify wiring, and this is not the same test as the one above.** That one
        # seeds the routes *before* `engine.load`, so a property with the **wrong**
        # `notify=` still reads correctly and the caption is right by accident - which is
        # how a non-notified map survives a green suite. Here the routes arrive **after**
        # the scene exists, which is what a live booth does: the card reads
        # `pending - not yet confirmed` and only later names the device. A caption frozen
        # at its first value is a presenter telling a visitor the wrong device for the
        # rest of the session, and nothing on screen would show it.
        seed = ('bridge.set_streams(["A", "B", "C"])\n'
                'bridge.apply_route({"stream": "A", "requested": "CPU",\n'
                '                    "confirmed": [], "state": "pending",\n'
                '                    "fell_back": False, "errors": []})\n')
        # **The late routes are published from a `QTimer` inside the render process**, so
        # the change the caption must notice is a property *notification*, not a rebuild.
        late = (
            "\nimport PySide6.QtCore as _qt\n"
            "def _confirm():\n"
            '    bridge.apply_route({"stream": "A", "requested": "CPU",\n'
            '                        "confirmed": ["CPU"], "state": "active",\n'
            '                        "fell_back": False, "errors": []})\n'
            '    bridge.apply_route({"stream": "B", "requested": "NPU",\n'
            '                        "confirmed": ["NPU"], "state": "active",\n'
            '                        "fell_back": False, "errors": []})\n'
            "_qt.QTimer.singleShot(1, _confirm)\n")
        before = _render(tempfile.mkdtemp(prefix="roadshow-cap0-"), seed)
        after = _render(tempfile.mkdtemp(prefix="roadshow-cap1-"), seed + late)
        self.assertIn("thumb_B", before[1], sorted(before[1]))
        self.assertEqual(before[1]["thumb_B"][0], "Camera B · not confirmed",
                         f"before any confirmation: {before[1]['thumb_B']!r}")
        self.assertEqual(after[1]["thumb_B"][0], "Camera B · NPU",
                         f"after the confirmation is published, the caption must name the "
                         f"device - a caption frozen at its first value tells a visitor "
                         f"the wrong device for the rest of the session: "
                         f"{after[1]['thumb_B']!r}")

    def test_the_provider_never_falls_back_to_another_streams_pixels(self):
        """The guarantee itself, at the provider's own level, with no window involved.

        `UiBridge.frame_for` returns `None` for a stream with no frame and the provider
        answers 1x1 transparent. It must **not** answer with the hero's frame, and it must
        not answer with the most recent frame from any camera - either would put another
        camera's shopper under this camera's caption.
        """
        import numpy as np
        from PySide6.QtGui import QImage
        from PySide6.QtCore import QCoreApplication

        from app.ui.bridge import UiBridge
        from app.ui.provider import FrameImageProvider

        app = QCoreApplication.instance() or QCoreApplication([])
        bridge = UiBridge()
        bridge.set_streams(["A", "B", "C"])
        red = np.zeros((64, 48, 3), np.uint8)
        red[:, :, 2] = 200
        bridge.apply_frame({"stream": "A", "image": red, "width": 48, "height": 64,
                            "zones": [], "boxes": [], "events": []})
        provider = FrameImageProvider(bridge)

        served = provider.requestImage("A?v=1")
        self.assertEqual((served.width(), served.height()), (48, 64))
        for sid in ("B", "C", "hero", "", "Z"):
            blank = provider.requestImage(f"{sid}?v=1")
            self.assertEqual((blank.width(), blank.height()), (1, 1),
                             f"a request for {sid!r} returned "
                             f"{blank.width()}x{blank.height()}; it must be the 1x1 "
                             f"blank and never another camera's pixels")
        # And with nothing at all stored, even camera A gets nothing.
        empty = UiBridge()
        empty.set_streams(["A"])
        blank_provider = FrameImageProvider(empty)
        self.assertEqual(blank_provider.requestImage("A?v=1").width(), 1,
                         "a camera with no frame must not be given the last frame any "
                         "camera produced")


class FrameAgeWindowAndFallbackRendering(unittest.TestCase):
    """**M2 Task 4.** Two rendering questions, neither of which needs a window.

    ## The frame-age percentiles must say what they are percentiles of

    `frame_age_p50_ms` is a percentile of a `deque(maxlen=FRAME_AGE_WINDOW)` - *the most
    recent N analysed frames* - and how many seconds that spans depends on the stream's own
    rate. The card used to print a bare `frame age p50 315.5 ms` with no window anywhere,
    and `app/main.py:1113` labelled the field `frame_age_p50_ms` with nothing beside it. A
    p50 of 2 s over a 17 s window and a p50 of 8 s over a 26 s one are different claims and
    the label carried neither.

    ## The fallback rendering, proved as rendering and **not** as an observation

    M0's unmet gate 9 needs a *silent* fallback: work that executes somewhere other than
    where it was asked to go. On this machine every device executes where it is asked, so
    that does not occur, and Task 4 could not observe it. What **is** provable here is that
    the screen would say so if it did - the badge's `(fell back from ...)` suffix and the
    `device_fell_back` flag. This class proves the rendering; it does **not** claim the
    hardware ever produced one, and the live-path test in `tests/test_main.py` asserts
    `fell_back` is `False` in the observed failure, which is the honest answer there.
    """

    def test_the_window_note_is_empty_until_somebody_says_how_many(self):
        bridge = _bridge()
        self.assertEqual(bridge.age_window_note, "",
                         "with no window published there is nothing to say, and an "
                         "invented 'last second' would be a claim about nothing")
        bridge.set_metrics(ai_fps=15.0, frame_age_p50_ms=315.5)
        self.assertEqual(bridge.age_window_note, "",
                         "a rate alone does not tell anyone how many frames the window is")

    def test_the_window_note_states_the_count_and_the_span_it_implies(self):
        bridge = _bridge()
        # **Publishing the window must notify on its own.** It would be easy to rely on
        # `_emit_for` firing `windowNotesChanged` whenever a metric changes, but a window
        # can change with no metric change at all - the first publish, or the deque bound
        # being altered by a rebuild - and then a QML binding bound to `age_window_note`
        # would never re-read. Asserted directly, with **no** metric set, so it cannot be
        # satisfied by the metric path.
        ticks = []
        bridge.windowNotesChanged.connect(lambda: ticks.append(1))
        bridge.set_age_window(256)
        self.assertEqual(len(ticks), 1,
                         "publishing the window must notify; a QML binding to "
                         "`age_window_note` has no other way to learn the window changed")
        bridge.set_metrics(ai_fps=15.0, frame_age_p50_ms=315.5,
                           frame_age_p95_ms=902.4)
        note = bridge.age_window_note
        self.assertIn("256", note, note)
        self.assertIn("analysed frames", note, note)
        # 256 / 15.0 = 17.07 s, and the note must show a rounded second rather than a
        # false precision it cannot have.
        self.assertIn("about 17 s", note, note)
        self.assertIn("15.0 fps", note, note)

    def test_a_zero_rate_does_not_divide(self):
        # The standing rule: never print `unavailable` as `0`, and never divide by a
        # window nobody measured.
        for fps in (0.0, None):
            bridge = _bridge()
            bridge.set_age_window(256)
            bridge.set_metrics(ai_fps=fps, frame_age_p50_ms=1.0)
            note = bridge.age_window_note
            self.assertIn("span depends on", note,
                          f"fps={fps!r}: the span is genuinely unknown and saying so "
                          f"beats implying 15 fps: {note!r}")
            self.assertNotIn("inf", note.lower())
            self.assertNotIn("nan", note.lower())

    def test_a_fallback_would_be_shown_as_a_fallback_on_the_badge(self):
        bridge = _bridge()
        bridge.set_route(requested="NPU", confirmed=("CPU",), state="active")
        self.assertTrue(bridge.device_fell_back,
                        "a confirmed device other than the requested one IS a fallback")
        self.assertEqual(bridge.device_display, "CPU (fell back from NPU)",
                         bridge.device_display)
        # And the requested device is still recorded, so the operator can see what was
        # asked for as well as what is running.
        self.assertEqual(bridge.device_requested, "NPU")

    def test_a_route_that_confirmed_what_it_asked_for_is_not_a_fallback(self):
        bridge = _bridge()
        # The GPU reports `'GPU.0'`; a substring test would call that a fallback.
        bridge.set_route(requested="GPU", confirmed=("GPU.0",), state="active")
        self.assertFalse(bridge.device_fell_back,
                         "the dotted spelling is the same device, not a different one")
        self.assertEqual(bridge.device_display, "GPU.0", bridge.device_display)

    def test_an_unproven_route_is_never_a_fallback(self):
        bridge = _bridge()
        bridge.set_route(requested="GPU", confirmed=(), state="pending")
        self.assertFalse(bridge.device_fell_back,
                         "nothing ran anywhere, so nothing fell back from anywhere")
        self.assertEqual(bridge.device_display,
                         "pending - GPU not yet confirmed", bridge.device_display)


class RequiredCopy(unittest.TestCase):
    """The brief's tests, kept because each one says exactly what it checks."""

    def test_the_recorded_footage_label_exists_in_the_ui_sources(self):
        # Spec §4: displayed continuously. Its absence was a real gap in the
        # previous build, so it is pinned in source, not just reviewed by eye.
        qml = _qml_sources()
        self.assertIn("Recorded footage", qml)
        self.assertIn("live inference", qml)

    def test_an_unavailable_metric_renders_as_unavailable_not_zero(self):
        b = _bridge()
        self.assertIsNone(b.system_cpu_percent)
        b.set_metrics(system_cpu_percent=None)
        self.assertEqual(b.display_text("system_cpu_percent"), "unavailable")

    def test_a_zero_metric_renders_as_a_number(self):
        b = _bridge()
        b.set_metrics(system_cpu_percent=0.0)
        self.assertNotEqual(b.display_text("system_cpu_percent"), "unavailable")

    def test_the_displayed_device_is_the_confirmed_one(self):
        b = _bridge()
        b.set_route(requested="NPU", confirmed=("CPU",), state="active")
        self.assertEqual(b.device_confirmed, "CPU")

    def test_no_magic_metric_numbers_live_in_qml(self):
        for p in sorted(UI_DIR.glob("*.qml")):
            text = p.read_text(encoding="utf-8")
            self.assertNotIn("0.25", text, f"{p.name} hardcodes a threshold")
            self.assertNotIn("640", text, f"{p.name} hardcodes the input size")


class NullNotZero(unittest.TestCase):
    """`unavailable` is never `0`, in either direction, on every optional metric."""

    def test_every_optional_metric_starts_absent(self):
        from app.ui.bridge import OPTIONAL_METRICS

        b = _bridge()
        for name in OPTIONAL_METRICS:
            # Every optional metric is also a QML property, so a binding can name it and
            # get an honest `None` rather than a binding error.
            self.assertIsNone(getattr(b, name), name)
            self.assertEqual(b.display_text(name), "unavailable", name)

    def test_a_measured_zero_renders_as_a_number_for_every_optional_metric(self):
        from app.ui.bridge import OPTIONAL_METRICS

        b = _bridge()
        for name in OPTIONAL_METRICS:
            b.set_metrics(**{name: 0.0})
            self.assertNotEqual(b.display_text(name), "unavailable", name)
            self.assertNotEqual(b.display_text(name), "0", name)

    def test_an_unavailable_metric_sample_nulls_the_property(self):
        from app.contracts.metric_sample import Validity

        b = _bridge()
        b.set_metrics(system_cpu_percent=44.0)
        b.set_samples([_sample("system_cpu_percent", None, validity=Validity.UNAVAILABLE,
                               window=0.0, now=0.0)])
        self.assertIsNone(b.system_cpu_percent)
        self.assertEqual(b.display_text("system_cpu_percent"), "unavailable")

    def test_a_stale_sample_keeps_its_value_but_says_so(self):
        # A stale reading is a reading. Degrading it to `None` would lose the last
        # measurement; the design says values older than 2 s are *stale*, and unavailable
        # is a different thing.
        from app.contracts.metric_sample import Validity

        b = _bridge()
        b.set_samples([_sample("app_cpu_percent", 12.0, validity=Validity.STALE,
                               window=0.25, now=0.0)])
        self.assertEqual(b.app_cpu_percent, 12.0)
        self.assertNotEqual(b.display_text("app_cpu_percent"), "unavailable")

    def test_an_unknown_metric_name_is_refused_rather_than_rendered(self):
        b = _bridge()
        with self.assertRaises(KeyError):
            b.display_text("system_cpu_utilisation")


class WindowFormatting(unittest.TestCase):
    """Ruling 23: the formatter must branch on `validity` before rendering a window."""

    def test_a_zero_window_on_an_unavailable_sample_renders_unavailable(self):
        from app.contracts.metric_sample import Validity

        b = _bridge()
        b.set_samples([_sample("system_cpu_percent", None, validity=Validity.UNAVAILABLE,
                               window=0.0, now=0.0)])
        self.assertEqual(b.window_text("system_cpu_percent"), "unavailable")

    def test_a_measured_window_is_rendered_with_its_unit(self):
        from app.contracts.metric_sample import Validity

        b = _bridge()
        b.set_samples([_sample("app_cpu_percent", 9.0, validity=Validity.VALID,
                               window=0.25, now=0.0)])
        self.assertIn("250", b.window_text("app_cpu_percent"))

    def test_a_long_window_is_rendered_in_seconds(self):
        from app.contracts.metric_sample import Validity

        b = _bridge()
        b.set_samples([_sample("app_cpu_percent", 9.0, validity=Validity.VALID,
                               window=2.5, now=0.0)])
        self.assertIn("s", b.window_text("app_cpu_percent"))

    def test_a_metric_with_no_recorded_sample_has_no_window(self):
        b = _bridge()
        b.set_metrics(system_cpu_percent=5.0)
        self.assertEqual(b.window_text("system_cpu_percent"), "unavailable")

    def test_the_bridge_never_divides(self):
        # A source-level check, because the obligation is about the *absence* of an
        # operation and no runtime assertion can prove one did not happen on some path.
        body = _module_source("ui/bridge.py").split("class UiBridge", 1)[1]
        for token in ("window_seconds /", "/ window", "window_seconds *"):
            self.assertNotIn(token, body,
                             f"UiBridge contains {token!r}; the window is formatted from "
                             f"the record, never divided by")


class OverlayContract(unittest.TestCase):
    """A coasted track produces no box. The obligation `is_current` was written for."""

    def test_a_coasted_track_produces_no_box(self):
        b = _bridge()
        b.set_overlay([_track(1, misses=1)], _frame())
        self.assertEqual(b.overlay_boxes, [])

    def test_a_current_track_produces_a_box(self):
        b = _bridge()
        b.set_overlay([_track(1, misses=0)], _frame())
        boxes = b.overlay_boxes
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["track_id"], 1)

    def test_only_the_current_ones_survive_a_mixed_set(self):
        # A real tracker returns both. This is the case the brief never mentions and the
        # one a booth actually hits: one person walking out of shot while another enters.
        from app.engine.tracker import Tracker

        class Detection:
            def __init__(self, box):
                self.class_id = 0
                self.score = 0.9
                self.box = box

        tracker = Tracker(iou_threshold=0.3, max_misses=2, min_hits=1)
        tracker.update([Detection((100, 100, 400, 800)), Detection((900, 100, 1200, 800))],
                       now=0.0)
        tracker.update([Detection((900, 100, 1200, 800))], now=0.5)
        all_tracks = tracker.tracks()
        self.assertTrue(any(not t.is_current for t in all_tracks),
                        "the fixture must actually contain a coasted track, or this "
                        "test proves nothing")

        b = _bridge()
        b.set_overlay(all_tracks, _frame())
        self.assertEqual([box["track_id"] for box in b.overlay_boxes],
                         [t.track_id for t in tracker.current_tracks()])

    def test_a_result_for_a_different_frame_is_not_accepted(self):
        b = _bridge()
        b.set_frame(_frame(fid=7))

        class Result:
            def __init__(self, frame):
                self.frame = frame

        self.assertFalse(b.accept_result(Result(_frame(fid=8))))
        self.assertTrue(b.accept_result(Result(_frame(fid=7))))

    def test_a_result_from_a_different_loop_of_the_clip_is_not_accepted(self):
        # Frame 7 of loop 0 and frame 7 of loop 1 are the same pixels with the same
        # frame_id. Binding to the id would paint loop 0's boxes onto loop 1's picture
        # the first time the clip rewinds.
        b = _bridge()
        b.set_frame(_frame(fid=7, loop=0))

        class Result:
            def __init__(self, frame):
                self.frame = frame

        self.assertFalse(b.accept_result(Result(_frame(fid=7, loop=1))))


class RouteHonesty(unittest.TestCase):
    """Review focus #5: the displayed device is the confirmed one."""

    def test_a_fresh_bridge_claims_no_device(self):
        b = _bridge()
        self.assertIsNone(b.device_confirmed)
        self.assertIn("pending", b.device_display)

    def test_a_requested_device_is_not_reported_as_confirmed(self):
        b = _bridge()
        b.set_route(requested="NPU", confirmed=None, state="pending")
        self.assertIsNone(b.device_confirmed)
        self.assertNotEqual(b.device_confirmed, "NPU")

    def test_a_fallback_is_visible_as_a_fallback(self):
        b = _bridge()
        b.set_route(requested="NPU", confirmed=("CPU",), state="active")
        self.assertTrue(b.device_fell_back)
        self.assertIn("fell back from NPU", b.device_display)

    def test_a_failed_route_does_not_claim_a_device(self):
        b = _bridge()
        b.set_route(requested="NPU", confirmed=None, state="failed")
        self.assertIsNone(b.device_confirmed)
        self.assertIn("no device is running", b.device_display)

    def test_the_gpu_dotted_spelling_is_not_substring_matched(self):
        b = _bridge()
        b.set_route(requested="GPU", confirmed=("GPU.0",), state="active")
        self.assertEqual(b.device_confirmed, "GPU.0")
        self.assertFalse(b.device_fell_back)


class UtilisationHonesty(unittest.TestCase):
    """No utilisation number may exist anywhere on the GPU or NPU cards."""

    def test_both_accelerator_cards_report_their_utilisation_unavailable(self):
        b = _bridge()
        self.assertFalse(b.gpu_available)
        self.assertFalse(b.npu_available)
        self.assertEqual(b.utilisation_unavailable_text, "Utilization unavailable")

    def test_the_absent_reason_is_a_measured_one_not_a_placeholder(self):
        b = _bridge()
        self.assertTrue(b.gpu_reason.strip())
        self.assertIn("counters", b.gpu_reason)
        self.assertIn("not attributable", b.gpu_reason)

    def test_the_accelerator_utilisation_metrics_are_absent_not_zero(self):
        b = _bridge()
        self.assertIsNone(b.gpu_utilisation_percent)
        self.assertIsNone(b.npu_utilisation_percent)
        # The property is `None`; the *rendering* is the word, never a digit.
        self.assertEqual(b.display_text("gpu_utilisation_percent"), "unavailable")
        self.assertEqual(b.display_text("npu_utilisation_percent"), "unavailable")

    def test_the_short_reason_is_a_summary_of_the_measured_one(self):
        # The card shows a short reason; the panel shows the measured text. They must be
        # the same fact at two lengths, not two claims - so the words the short version
        # uses must appear in the long one.
        #
        # `counters` is checked in both because it is in `VALUE_READS_REFUSED_REASON`.
        # `attributable` is checked in the **short** version only, because the measured
        # reason is about the read faulting and does not mention attribution; the
        # attribution finding is a *separate* measurement (M0: our pid appeared in 0 of
        # 469 samples of a machine-wide peak) and lives in `M0-qualification.md`. The
        # first version of this test asserted it in both and was wrong about the long
        # text, which is recorded here rather than quietly widened.
        b = _bridge()
        self.assertLess(len(b.gpu_reason), len(b.utilisation_measured_reason))
        self.assertIn("counters", b.gpu_reason)
        self.assertIn("counters", b.utilisation_measured_reason)
        self.assertIn("attributable", b.gpu_reason)

    def test_the_attribution_finding_is_sourced_to_a_committed_measurement(self):
        # The short reason asserts that a machine-wide peak is not attributable. That
        # claim has to exist in a committed artefact, or the card is asserting something
        # the repository does not say.
        from app.inference.pinned import REPO_ROOT

        report = (REPO_ROOT / "docs" / "qualification" / "M0-qualification.md")
        text = report.read_text(encoding="utf-8")
        self.assertIn("not attributable", text,
                      f"{report} no longer states the attribution finding, so "
                      f"UiBridge.gpu_reason claims something the repository does not")

    def test_no_qml_mentions_a_utilisation_percentage(self):
        # A source-level check: no QML file may name a utilisation metric with a value,
        # and **every one of the three device lines** must take its availability from the
        # bridge's measurement, never a literal.
        #
        # I7: the CPU card used to pass `utilisationAvailable: true`, hardcoded, which
        # would render an unavailable `psutil` reading in the live colour and undimmed -
        # the exact defect the two accelerator cards are careful about, applied to the one
        # card that is supposed to be measurable. The CPU is not special: all three now
        # answer the same question of the bridge.
        qml = _qml_sources()
        self.assertNotIn("gpu_utilisation_percent: ", qml.replace('"', ""))
        self.assertNotIn("utilisationAvailable: true", qml,
                         "no card may hardcode its own availability")
        # **M4 retarget.** The device-to-metric mapping moved again: the card is three rows
        # of verdict at the legibility ladder, and the utilisation figure lives in
        # `PresenterDrawer.qml` (the surface a presenter reads at arm's length). The guard's
        # teeth are unchanged: **each device maps to its own metric name**, and
        # **availability is read from the bridge, never asserted**.
        drawer = (UI_DIR / "PresenterDrawer.qml").read_text(encoding="utf-8")
        code = "\n".join(line for line in drawer.splitlines()
                         if not line.strip().startswith("//"))
        for device, metric in (("CPU", "system_cpu_percent"),
                               ("GPU", "gpu_utilisation_percent"),
                               ("NPU", "npu_utilisation_percent")):
            self.assertIn(f'deviceId === "{device}"', code,
                          f"the {device} line must name the {device}'s own metric, not a "
                          f"shared one")
            self.assertIn(f'"{metric}"', code,
                          f"and the {device} metric key must be the bridge's own")
        self.assertIn("bridge.metric_text[metricName]", code,
                      "the utilisation figure must come from the bridge's notified map")
        self.assertIn("bridge.unavailable_text", code,
                      "and an absent figure must read as the bridge's word, not a 0")
        # A card may not dim itself on utilisation - see ComputeCard.qml's header: dimming
        # the card that says the NPU stream is eleven seconds stale, because NPU utilisation
        # cannot be measured, would dim the one warning that matters.
        card = (UI_DIR / "ComputeCard.qml").read_text(encoding="utf-8")
        self.assertNotIn("utilisationAvailable", card,
                         "the card no longer renders a utilisation figure at all; if it does "
                         "again it must answer availability from the bridge")

    def test_the_cpu_card_answers_the_same_question_the_accelerators_do(self):
        # The bridge side of I7: `cpu_available` is derived from the value, so an
        # UNAVAILABLE CPU reading cannot claim availability.
        from app.contracts.metric_sample import Validity

        b = _bridge()
        self.assertFalse(b.cpu_available,
                         "a fresh bridge must not claim a CPU reading it does not have")
        b.set_samples([_sample("system_cpu_percent", 12.0, validity=Validity.VALID)])
        self.assertTrue(b.cpu_available)
        b.set_samples([_sample("system_cpu_percent", None,
                               validity=Validity.UNAVAILABLE)])
        self.assertFalse(b.cpu_available,
                         "an UNAVAILABLE CPU reading must not read as available")


class ExplanationContent(unittest.TestCase):
    """The design's section 9 requires six explanations to be reachable."""

    REQUIRED = ("yolo", "tracking", "zone", "cpu", "gpu", "metrics")

    def test_every_required_explanation_key_has_text(self):
        b = _bridge()
        for key in self.REQUIRED:
            self.assertTrue(b.explanation(key).strip(), key)

    def test_the_gpu_copy_states_the_measurement_conditions(self):
        # The settled fact: the NPU measures slower than the GPU on this workload, and
        # any copy stating a device ordering has to carry the conditions with it.
        text = _bridge().explanation("gpu")
        for figure in ("20.99", "3.30", "5.92"):
            self.assertIn(figure, text, "a published timing needs its measured range")
        for condition in ("single frame", "640x640", "batch 1", "30", "Arc 140V"):
            self.assertIn(condition, text,
                          f"the device comparison is missing the condition {condition!r}")
        self.assertIn("not a throughput claim", text)

    def test_the_scope_reduction_is_stated_in_the_same_words_as_the_configuration(self):
        from app.engine.retail_config import retail_sanity_manifest
        from app.ui.bridge import SCOPE_NOTE_LABEL

        b = _bridge()
        manifest = retail_sanity_manifest()
        # The **pack's own** wording, not a paraphrase. An earlier revision of the bridge
        # carried a hand-written summary and this test was written to match the summary;
        # reading the pack instead makes the summary unnecessary.
        self.assertIn("No product-transfer event is detectable", manifest.scope_note)
        self.assertIn("Shelf-to-basket is NOT delivered", manifest.scope_note)
        self.assertEqual(b.scope_note, manifest.scope_note)
        self.assertEqual(SCOPE_NOTE_LABEL, manifest.scope_note)

    def test_the_scope_reduction_on_screen_does_not_claim_a_dwell_measurement(self):
        # The pack's guard measures two sustained presences and explicitly says they
        # carry no duration. The screen must not imply otherwise.
        from app.engine.retail_config import retail_sanity_manifest

        note = retail_sanity_manifest().scope_note
        for forbidden in ("dwell", "seconds of presence"):
            self.assertNotIn(forbidden, note.lower())

    def test_an_unknown_explanation_key_returns_empty_not_an_exception(self):
        b = _bridge()
        self.assertEqual(b.explanation("no-such-key"), "")


class ThemeAndSources(unittest.TestCase):
    """AGENTS.md rule 10, and the file-level promises this UI makes."""

    def test_theme_holds_no_metric_values(self):
        from app import theme

        forbidden = ("conf", "iou", "threshold", "fps", "latency", "p50", "p95",
                     "window", "percent", "pad")
        for name in dir(theme):
            if not name.isupper():
                continue
            lowered = name.lower()
            for word in forbidden:
                self.assertNotIn(
                    word, lowered,
                    f"app/theme.py holds {name!r}, whose name carries the metric word "
                    f"{word!r}. A metric that lives in a layout module becomes a "
                    f"constant, and a constant that looks like a measurement is the "
                    f"failure this project exists to prevent.")

    def test_no_qml_file_hardcodes_a_colour_or_a_pixel_size(self):
        import re

        colour = re.compile(r'"#[0-9a-fA-F]{3,8}"')
        pixels = re.compile(r":\s*\d+\s*;")
        for path in sorted(UI_DIR.glob("*.qml")):
            text = path.read_text(encoding="utf-8")
            self.assertEqual(colour.findall(text), [],
                             f"{path.name} hardcodes a colour; it belongs in app/theme.py")
            self.assertEqual(pixels.findall(text), [],
                             f"{path.name} hardcodes a pixel size; it belongs in "
                             f"app/theme.py")

    def test_every_qml_component_is_used_by_another_file(self):
        # A component file nothing instantiates is a file that can rot silently.
        for path in sorted(UI_DIR.glob("*.qml")):
            others = "".join(p.read_text(encoding="utf-8")
                             for p in UI_DIR.glob("*.qml") if p.name != path.name)
            self.assertIn(path.stem, others,
                          f"{path.name} is not used by any other QML file")

    def test_the_scope_reduction_on_screen_does_not_claim_a_dwell_measurement(self):
        # The pack's guard measures two sustained presences and explicitly says they
        # carry no duration. The screen must not imply otherwise.
        from app.engine.retail_config import retail_sanity_manifest

        note = retail_sanity_manifest().scope_note
        for forbidden in ("dwell", "seconds of presence"):
            self.assertNotIn(forbidden, note.lower())

    def test_the_bridge_never_reads_the_collector_accessors(self):
        # The accessors share one frozen `_Tick` with `sample()`; a card that polls one
        # on its own timer can flip the booth's own sample to UNAVAILABLE.
        source = _module_source("ui/bridge.py")
        for forbidden in (".system_cpu_percent()", ".app_cpu_percent()", "collector."):
            self.assertNotIn(forbidden, source,
                             f"app/ui/bridge.py references {forbidden!r}")

    def test_the_bridge_has_no_simulated_value_source(self):
        # Checked on the *code*, not the whole class text: the class docstring and the
        # `set_metrics` docstring both mention `math.sin` and `random` while describing
        # exactly why they are absent, so a text search over the class body finds the
        # prohibition and mistakes it for a violation. The first version of this test
        # did that and failed.
        import ast
        import io
        import tokenize

        source = _module_source("ui/bridge.py")
        tree = ast.parse(source)
        offenders: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            body = node.body
            # Drop a leading docstring; everything after it is code.
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body = body[1:]
            for statement in body:
                for sub in ast.walk(statement):
                    name = None
                    if isinstance(sub, ast.Attribute):
                        name = sub.attr
                    elif isinstance(sub, ast.Name):
                        name = sub.id
                    if name in ("random", "sin", "cos", "secrets", "uuid4", "choices"):
                        offenders.append(f"{node.name}: {name}")
                    elif isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                            and sub.value.startswith("simulated"):
                        offenders.append(f"{node.name}: literal {sub.value!r}")
        self.assertEqual(offenders, [],
                         f"UiBridge references a simulated-value source: {offenders}. "
                         f"Release mode refuses to start with a simulated provider "
                         f"registered, so a fabricated value here would be a way around "
                         f"that gate.")

    def test_the_label_constant_agrees_with_the_qml_source(self):
        from app.ui.bridge import RECORDED_FOOTAGE_LABEL

        # The QML fallback writes the separator as the escape `\u2022`; the Python
        # constant carries the character. Both are the same string once the escape is
        # resolved, which is what is asserted - a naive substring search for the
        # constant in the QML source fails on the escape and would tempt someone into
        # changing one of the two.
        qml = _qml_sources()
        self.assertIn(RECORDED_FOOTAGE_LABEL.encode("unicode_escape").decode("ascii"),
                      qml,
                      "the QML fallback literal must spell the same characters as "
                      "RECORDED_FOOTAGE_LABEL once escapes are resolved")
        self.assertIn("Recorded footage \\u2022 live inference", qml)
        for fragment in RECORDED_FOOTAGE_COPY:
            self.assertIn(fragment, RECORDED_FOOTAGE_LABEL)

    def test_the_window_is_split_into_the_three_declared_bands(self):
        # 12 / 66 / 22, from the design. Asserted here rather than trusted, because the
        # sum is what decides whether the control bar falls off the bottom - which it did,
        # once, before the bands were taken as fractions of the *available* height.
        from app import theme

        self.assertAlmostEqual(sum(theme.GRID.values()), 1.0, places=6)
        self.assertAlmostEqual(theme.GRID["main"], 0.66, places=6)
        self.assertAlmostEqual(theme.VIDEO_WIDTH_FRACTION, 0.68, places=6)


def _mutation_script(ui_dir: Path, bridge_path: Path) -> str:
    """The same render, against a *mutated copy* of the UI.

    Two differences from `_render_script`, and both matter:

    - `app.ui.bridge` is imported from `bridge_path`, which is a copy with the
      `is_current` filter removed, so the mutation is real and isolated.
    - the overlay is not supplied by a subclass; the mutated `set_overlay` does the work.

    Written as a template for the same reason as the render script: a second copy of the
    UI wiring in a file would be a thing to keep in step.
    """
    return '''
"""Render the UI with the is_current filter removed. Mutation evidence only."""
import os
import shutil
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, @REPO@)

import importlib.util

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtGui import QFontDatabase, QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

out = sys.argv[1]
work = out + "_ui"

# Install the mutated bridge under the real module name, so the QML sees the same
# context property type name it expects.
spec = importlib.util.spec_from_file_location("app.ui.bridge", @BRIDGE@)
module = importlib.util.module_from_spec(spec)
sys.modules["app.ui.bridge"] = module
spec.loader.exec_module(module)

from app.engine.tracker import Track

app = QGuiApplication(sys.argv)
font_id = QFontDatabase.addApplicationFont(@FONT@)
if font_id < 0:
    sys.exit(3)

bridge = module.UiBridge(app)
theme = module.ThemeTokens(app)


class FakeFrame:
    identity = ("selftest", "A", 0, 0)
    width = 1920
    height = 1080
    # **M2: which camera this frame belongs to.** `UiBridge.set_frame_image` files the
    # pixels under this key, and the hero's tile reads that key - so a stand-in without it
    # is a frame belonging to no camera, and the render comes back with the hero empty and
    # no warning anywhere.
    stream_id = "A"


image = np.zeros((1080, 1920, 3), np.uint8)
image[:, :, 1] = 90
image[240:820, 620:1010, 0] = 200
image[240:820, 620:1010, 2] = 200
FakeFrame.image = image


class FakeZone:
    name = "counter_band"
    roi = (0.0, 0.20, 1.00, 0.60)


def track(track_id, box, misses):
    return Track(track_id=track_id, class_id=0, box=tuple(float(v) for v in box),
                 first_seen=0.0, last_seen=0.0, hits=2, misses=misses, confirmed=True)


frame = FakeFrame()
bridge.set_frame_image(frame)
bridge.set_zones([FakeZone()], frame)
bridge.set_overlay([track(1, (620, 240, 1010, 820), 0),
                    track(2, (1250, 260, 1600, 900), 1)], frame)
bridge.set_route(requested="NPU", confirmed=("CPU",), state="active")

shutil.rmtree(work, ignore_errors=True)
shutil.copytree(@UI@, work)

engine = QQmlApplicationEngine()
provider = module.FrameImageProvider(bridge)
engine.addImageProvider("frames", provider)
ctx = engine.rootContext()
ctx.setContextProperty("Theme", theme)
ctx.setContextProperty("bridge", bridge)
engine.load(QUrl.fromLocalFile(work + "/Main.qml"))
roots = engine.rootObjects()
if not roots:
    print("no root objects", file=sys.stderr)
    sys.exit(4)
loop = QEventLoop()
QTimer.singleShot(500, loop.quit)
loop.exec()
grab = roots[0].grabWindow()
if not grab.save(out):
    sys.exit(7)
print("overlay boxes with the filter removed: %d" % len(bridge.overlay_boxes))
shutil.rmtree(work, ignore_errors=True)
sys.exit(0)
'''.replace("@REPO@", repr(str(REPO_ROOT))) \
   .replace("@BRIDGE@", repr(str(bridge_path))) \
   .replace("@FONT@", repr(str(RENDER_FONT))) \
   .replace("@UI@", repr(str(ui_dir)))


class OffscreenRender(unittest.TestCase):
    """A real render at 1920x1080, checked for content rather than for having run."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="roadshow-render-")
        cls.png = _render_png(cls._tmp)
        from PySide6.QtGui import QImage

        cls.image = QImage(cls.png)
        if cls.image.isNull():
            raise AssertionError(f"the render produced no image at {cls.png}")

    def _pixels(self):
        import numpy as np

        image = self.image
        buffer = np.frombuffer(bytes(image.constBits()), dtype=np.uint8).reshape(
            image.height(), image.bytesPerLine())
        return buffer[:, : image.width() * 4].reshape(
            image.height(), image.width(), 4)

    def test_the_render_is_exactly_1920_by_1080(self):
        from app.theme import CANVAS_HEIGHT, CANVAS_WIDTH

        self.assertEqual(self.image.width(), CANVAS_WIDTH)
        self.assertEqual(self.image.height(), CANVAS_HEIGHT)

    def test_the_background_is_dark(self):
        import numpy as np

        luminance = self._pixels()[:, :, :3].astype(float).mean(axis=2)
        self.assertLess(float(luminance.mean()), 90.0,
                        "the booth screen is a dark surface; a light one means a fill "
                        "token is bound to a text colour")

    def test_the_mandatory_label_has_glyphs_on_the_screen(self):
        # The font trap: with 0 font families registered under offscreen every glyph is
        # blank and this assertion would pass vacuously. `addApplicationFont` in the
        # render script is what makes it meaningful, and this rectangle is where the
        # label is bound.
        import numpy as np

        region = self._pixels()[40:110, 20:440, :3]
        bright = int((region.astype(float).mean(axis=2) > 200).sum())
        self.assertGreater(bright, 200,
                           "no light pixels in the label's rectangle: either the label "
                           "is not rendered or no font was registered, and the second "
                           "case would make this a vacuous pass")

    def test_the_render_contains_the_video_frame(self):
        import numpy as np

        region = self._pixels()[300:600, 400:900, :3].astype(int)
        # The frame this render was given has a green cast with a cyan block in the
        # middle. Asserting on it proves the *pixels* reached the screen, which a pixel
        # histogram alone would not - and the block's exact colour is what proves the
        # BGR-to-RGB swap in `FrameImageProvider` is right: a swapped render turns the
        # cyan into a yellow-green block, which the green test below would still pass
        # and this one would not.
        green_dominant = int((region[:, :, 1] > region[:, :, 0] + 30).sum())
        cyan = int(((region[:, :, 2] > 150) & (region[:, :, 0] > 150)).sum())
        self.assertGreater(green_dominant, 10000,
                           "the frame's green cast is not on the screen")
        self.assertGreater(cyan, 5000,
                           "the frame's cyan block is not on the screen: either the "
                           "image never arrived or the channel order is wrong")

    def _accent_mask(self):
        """Accent-coloured pixels, as a boolean array over the render.

        `QImage.Format_RGB32` is stored **BGRA** on a little-endian host, so channel 0 is
        blue. Getting that backwards is why the first version of this mask counted zero
        pixels (measured: `accent pixels: 0` against a render with 13987 of them), and a
        mask that always reads zero turns every "the box is drawn" assertion into a
        vacuous pass.
        """
        import numpy as np

        pixels = self._pixels()[:, :, :3].astype(int)
        return ((pixels[:, :, 0] > 200) & (pixels[:, :, 1] > 150)
                & (pixels[:, :, 1] < 210) & (pixels[:, :, 2] < 120))

    def test_exactly_one_track_box_is_drawn_and_the_current_track_is_it(self):
        # The render script feeds one current track and one coasted track. **One** box may
        # appear, and it must be the current one.
        #
        # The geometry, measured from this render: the hero occupies x 24..1296, and the
        # accent-coloured strokes on a scanline through it are at x 24-25 (the hero's own
        # border), 435-437 and 691-693 (the two vertical edges of track 1's box), and
        # 1295-1296 (the hero's right border). Track 1 is the *current* track.
        import numpy as np

        accent = self._accent_mask()
        row = np.nonzero(accent[400])[0]
        runs = []
        if len(row):
            start = previous = row[0]
            for x in row[1:]:
                if x != previous + 1:
                    runs.append((int(start), int(previous)))
                    start = x
                previous = x
            runs.append((int(start), int(previous)))
        interior = [r for r in runs if r[0] > 30 and r[1] < 1290]
        self.assertEqual(len(interior), 2,
                         f"expected exactly one box (two vertical strokes) inside the "
                         f"hero, found {interior}")

    def test_the_coasted_track_box_is_not_drawn(self):
        # The end-to-end version of `OverlayContract::test_a_coasted_track_produces_no_box`.
        #
        # **This method used to be defined twice in this class**, with different bodies,
        # and Python silently kept the last one. The review of M1 found it: a shadowed
        # test is not a test anybody ran, and the shadowed copy sampled a slightly
        # different region. There is now one definition, and
        # `TestSuiteHygiene::test_no_test_method_is_defined_twice_in_one_class` keeps it
        # that way.
        #
        # Track 2 is coasted and sits at source x 1250..1600 of 1920, y 260..900 of 1080.
        # Projected into the hero (x 24..1296, y about 160..700) that is roughly
        # x 852..1084, y 280..577. The rectangle sampled here is that region **between the
        # two zone lines** - the zone rectangle is drawn in the same accent colour and
        # spans the full width, so a rectangle that crossed it would find the zone line
        # instead of the box and the test would pass for the wrong reason.
        accent = self._accent_mask()
        region = accent[300:450, 820:1120]
        self.assertEqual(int(region.sum()), 0,
                         f"{int(region.sum())} accent pixels in the coasted track's "
                         f"region: its box is being drawn over an object that is no "
                         f"longer there")

    def test_removing_the_filter_would_draw_the_coasted_box(self):
        # The mutation check, run in a child process against a **copy** of the UI with the
        # filter removed. Without it, `test_the_coasted_track_box_is_not_drawn` cannot be
        # distinguished from a test that is simply sampling the wrong pixels - and this
        # project has a standing finding that a test which passes with the fix removed is
        # not a test.
        import shutil
        import subprocess

        import numpy as np

        work = tempfile.mkdtemp(prefix="roadshow-mutation-")
        ui = Path(work) / "ui"
        shutil.copytree(UI_DIR, ui)
        bridge_py = Path(work) / "bridge.py"
        source = (REPO_ROOT / "app" / "ui" / "bridge.py").read_text(encoding="utf-8")
        # The mutation: drop the `is_current` filter from `set_overlay`.
        #
        # The counter is named `coasted`, not `skipped`: a missing frame is a *different*
        # reason a box cannot be placed, and the review of M1 found this method claiming
        # "holding coasted tracks" when the real cause was that no frame had been
        # decoded. Two counters, two status words. This assertion is the guard that caught
        # the rename - it is the reason the string is spelled out here rather than
        # regex-matched.
        mutated = source.replace(
            "            if not getattr(track, \"is_current\", False):\n"
            "                coasted += 1\n"
            "                continue\n",
            "",
        )
        self.assertNotEqual(mutated, source,
                            "the mutation did not apply; the filter's text has moved and "
                            "this test is now measuring nothing")
        bridge_py.write_text(mutated, encoding="utf-8")

        script = Path(work) / "mutate.py"
        script.write_text(_mutation_script(ui, bridge_py), encoding="utf-8")
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYTHONPATH"] = str(REPO_ROOT)
        completed = subprocess.run(
            [sys.executable, str(script), str(Path(work) / "mutated.png")],
            capture_output=True, text=True, timeout=300, cwd=str(work), env=env,
        )
        self.assertEqual(completed.returncode, 0,
                         f"the mutated render failed: {completed.stderr[-3000:]}")
        from PySide6.QtGui import QImage

        mutated_image = QImage(str(Path(work) / "mutated.png"))
        self.assertFalse(mutated_image.isNull())
        buf = np.frombuffer(bytes(mutated_image.constBits()),
                            dtype=np.uint8).reshape(
            mutated_image.height(), mutated_image.bytesPerLine())
        arr = buf[:, : mutated_image.width() * 4].reshape(
            mutated_image.height(), mutated_image.width(), 4)[:, :, :3].astype(int)
        accent = ((arr[:, :, 0] > 200) & (arr[:, :, 1] > 150) & (arr[:, :, 1] < 210)
                  & (arr[:, :, 2] < 120))
        drawn = int(accent[300:450, 820:1120].sum())
        self.assertGreater(drawn, 0,
                           "removing the is_current filter drew no box, so the "
                           "un-mutated assertion is not measuring the filter")
        shutil.rmtree(work, ignore_errors=True)

    def test_the_render_contains_the_metric_column(self):
        import numpy as np

        column = self._pixels()[160:840, 1320:1900, :3].astype(float).mean(axis=2)
        bright = int((column > 180).sum())
        # Measured at **4225** bright pixels for three cards with two figures, three
        # device names and three placement badges. The bound is below that, because a
        # bound above it would be asserting a font metric rather than the presence of the
        # column, and it would break the first time a figure's width changed.
        self.assertGreater(bright, 2000,
                           "the metrics column is empty; the three compute cards are "
                           "not being drawn")

    def test_the_render_is_not_a_single_flat_colour(self):
        import numpy as np

        pixels = self._pixels()[::8, ::8, :3]
        distinct = len(np.unique(pixels.reshape(-1, 3), axis=0))
        self.assertGreater(distinct, 40,
                           f"only {distinct} distinct colours in the render: the screen "
                           f"is effectively blank")


class SlotBindingAudit(unittest.TestCase):
    """The standing guard against frozen QML bindings.

    ## The defect class

    **A QML binding that calls a method declares no property dependency, so Qt evaluates
    it once - at `engine.load` - and never again.** Three times in this milestone:

    1. `device_display` was `constant=True` with no notify signal: the badge's Python value
       changed and the screen never did.
    2. every metric card bound to `bridge.display_text(...)`: on a live run whose bridge
       held `system_cpu_percent: 58.0`, the screen read `unavailable`.
    3. the card's `windowNote` bound to `bridge.window_text(...)` - eleven lines below the
       fix for (2) - so the "measured over N ms" line never appeared at all, with its
       `visible:` guard false for the whole session.

    Fixing an instance does not close the class, and M2/M3 will hit it again. **This is the
    general form**: any binding to a Slot whose result can change is frozen, and the only
    safe ones are the ones named in
    `app.ui.bridge.SLOT_CALLS_SAFE_IN_A_QML_BINDING`.

    ## What it also catches

    A call to a name the object **does not have**. `Main.qml` called `bridgeUi.explain(...)`
    and `Bridge.qml` had no `explain` function - it had been dropped in an earlier edit.
    Because the call sits behind `pinnedExplanationKey.length > 0` in a ternary, QML's
    short-circuit meant it was never evaluated: the window loaded, every card drew, and the
    "?" buttons were **dead at the booth**. A dangling call is invisible until somebody
    clicks, which is the worst time to find out.

    ## How it reads the QML

    Line-oriented, not a parser. It scans for `<identifier>(` in a binding expression and
    reports anything that looks like a call, then requires each to be classified. The
    tolerance is deliberate: a QML parse tree would be a second, worse way to write the
    rules, and the property under test is "did somebody add an unclassified call".
    """

    #: `bridge.<name>(` and `bridgeUi.<name>(` - the two QML-visible entry points.
    CALL = re.compile(r"\b(bridge|bridgeUi)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")

    def _calls(self) -> list:
        """Every `bridge.x(...)` / `bridgeUi.x(...)` in the QML, with file and line."""
        found = []
        for path in sorted(UI_DIR.glob("*.qml")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("//"):
                    continue  # a comment describes code; it does not run
                for match in self.CALL.finditer(line):
                    found.append((path.name, number, match.group(1),
                                  match.group(2), stripped))
        return found

    def test_the_audit_finds_the_bindings_it_is_meant_to_find(self):
        # If the scan silently matched nothing it would be a guard that guards nothing,
        # which is worse than no guard. Assert it found the known bindings.
        calls = self._calls()
        names = {(owner, name) for _, _, owner, name, _ in calls}
        self.assertIn(("bridge", "explanation"), names,
                      f"the scan found no bridge.explanation call at all, so it is not "
                      f"reading the QML: {calls}")
        self.assertTrue(calls, "the scan found no bridge calls in any QML file")

    def test_every_slot_call_in_a_binding_is_classified_constant_safe(self):
        from app.ui.bridge import SLOT_CALLS_SAFE_IN_A_QML_BINDING

        unsafe = [(f, n, f"{owner}.{name}", text)
                  for f, n, owner, name, text in self._calls()
                  if f"{owner}.{name}" not in SLOT_CALLS_SAFE_IN_A_QML_BINDING]
        self.assertEqual(unsafe, [],
                         f"a QML binding calls something that is not in "
                         f"SLOT_CALLS_SAFE_IN_A_QML_BINDING: {unsafe}. A binding over a "
                         f"method call declares no dependency, so Qt evaluates it once at "
                         f"engine.load and never again - the metric cards, the device badge "
                         f"and the window note were all frozen that way. Bind to a notified "
                         f"property instead, or add the name with a stated justification.")

    def test_the_classified_names_really_exist(self):
        # A name in the allow-list that no longer exists would make the guard vacuous for
        # that name: `bridgeUi.explain` was classified the moment it was restored, and if
        # it were deleted again the guard would keep passing while the panel went dead.
        import re as _re

        from app.ui.bridge import SLOT_CALLS_SAFE_IN_A_QML_BINDING, UiBridge

        qml_functions = set(_re.findall(
            r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            (UI_DIR / "Bridge.qml").read_text(encoding="utf-8")))
        self.assertTrue(SLOT_CALLS_SAFE_IN_A_QML_BINDING,
                        "the allow-list is empty, so the guard classifies nothing")
        for qualified in sorted(SLOT_CALLS_SAFE_IN_A_QML_BINDING):
            owner, _, name = qualified.partition(".")
            if owner == "bridge":
                self.assertTrue(hasattr(UiBridge, name),
                                f"{qualified!r} is classified constant-safe but UiBridge "
                                f"has no such member")
            else:
                self.assertIn(name, qml_functions,
                              f"{qualified!r} is classified constant-safe but Bridge.qml "
                              f"defines no such function")

    def test_no_qml_binding_calls_a_function_that_does_not_exist(self):
        # `bridge.explanation` and every `bridgeUi.*` helper must all be real.
        # `Main.qml` called `bridgeUi.explain(...)` and `Bridge.qml` had no `explain`
        # function - it had been dropped in an earlier edit. Because the call sits behind
        # `pinnedExplanationKey.length > 0` in a ternary, QML's short-circuit meant it was
        # never evaluated: the window loaded, every card drew, and the "?" buttons were dead
        # at the booth. A dangling call is invisible until somebody clicks.
        from app.ui.bridge import UiBridge

        bridge_qml = (UI_DIR / "Bridge.qml").read_text(encoding="utf-8")
        qml_functions = set(re.findall(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                                       bridge_qml))
        missing = []
        for file_name, line, owner, name, text in self._calls():
            if owner == "bridge":
                if not hasattr(UiBridge, name):
                    missing.append(f"{file_name}:{line} calls bridge.{name}(...) but "
                                   f"UiBridge has no such member")
            elif name not in qml_functions:
                missing.append(f"{file_name}:{line} calls bridgeUi.{name}(...) but "
                               f"Bridge.qml defines no function {name}()")
        self.assertEqual(missing, [],
                         f"a QML binding calls something that does not exist: {missing}. "
                         f"It loads a perfectly good window and fails only on click.")

    def test_the_window_note_is_read_from_a_notified_map_not_a_slot_call(self):
        # The specific instance the review named. Named explicitly so a regression to
        # `window_text(...)` in a binding fails here with a message that says what to do,
        # rather than only in the general scan above.
        #
        # **M2 Task 5: the mechanism moved, the ban did not.** The card is per stream now,
        # and the frame-age window is a *count of analysed frames for this camera*, which
        # is not in `window_notes` at all - that map is the telemetry sampler's windows, one
        # per metric, and none of them describes a stream's retained ages. So the card reads
        # it from the bridge's notified per-stream record instead. The guard therefore now
        # bans the Slot call **anywhere** in the card, which is strictly stronger than
        # banning it in one property, and requires the window to arrive through a notified
        # property.
        #
        # **M4: the window moved into the presenter drawer**, and the ban moved with it -
        # the guard reads *both* files now, so neither surface can reintroduce the call.
        drawer = (UI_DIR / "PresenterDrawer.qml").read_text(encoding="utf-8")
        code = "\n".join(line for line in drawer.splitlines()
                         if not line.strip().startswith("//"))
        self.assertNotRegex(
            code, r"\bwindow_text\s*\(",
            "the drawer calls the window_text Slot, so the window line is evaluated once at "
            "engine.load and never appears again")
        self.assertNotRegex(
            code, r"\bdisplay_text\s*\(",
            "display_text is a Slot too, and a Slot call in a binding declares no property "
            "dependency")
        self.assertIn("record.window_text", code,
                      "the window must come from the bridge's notified per-stream record, "
                      "which is what re-reads when the telemetry tick publishes")

    def test_every_device_card_binds_its_unavailable_wording_to_the_bridge(self):
        # R1. The accelerators say "Utilization unavailable" because there is no counter
        # for them; the CPU line must say its own thing, because `psutil.cpu_percent` works
        # here and "no sample yet" is the truth. One shared literal claimed a failure to
        # read a counter that is not failing.
        #
        # **M4: the line is in the presenter drawer.** The always-visible screen shows the
        # device *class* on each camera's badge and does not print a utilisation figure at
        # all; the wording is a presenter's claim about a counter, and it moved to the
        # surface a presenter reads.
        drawer = (UI_DIR / "PresenterDrawer.qml").read_text(encoding="utf-8")
        code = "\n".join(line for line in drawer.splitlines()
                         if not line.strip().startswith("//"))
        self.assertIn("bridge.unavailable_notice[deviceId]", code,
                      "the unavailable label must be per-device, keyed by deviceId")
        self.assertNotIn("text: bridge.utilisation_unavailable_text", code,
                         "one shared literal on every camera is what R1 is about")

    def test_the_event_counts_are_gated_on_frames_analysed_not_on_the_route(self):
        # R3. A confirmed placement arrives from the worker's route *before* any frame has
        # been analysed, so gating on it printed three hard zeros for an event machine
        # that had not run.
        #
        # **M4: the row moved into the presenter drawer**, with its gating; and the live
        # screen must not carry an ungated copy.
        drawer = (UI_DIR / "PresenterDrawer.qml").read_text(encoding="utf-8")
        code = "\n".join(line for line in drawer.splitlines()
                         if not line.strip().startswith("//"))
        self.assertIn("bridge.frames_analysed > 0", code,
                      "the event counts must be gated on frames analysed")
        self.assertNotIn("routeConfirmed", code,
                         "gating on the route leaves a window where a measured 0 is "
                         "printed for a machine that has analysed nothing")
        main_qml = (UI_DIR / "Main.qml").read_text(encoding="utf-8")
        main_code = "\n".join(line for line in main_qml.splitlines()
                              if not line.strip().startswith("//"))
        self.assertNotIn("eventCountRow", main_code,
                         "the event row is a presenter's reading now; two copies would be "
                         "two gates to keep in step")


class EventRowGating(unittest.TestCase):
    """R3 as a **render**, not a source check: the transition, deterministically.

    The live-path probe cannot catch this on its own. By the time it reads the screen the
    pipeline has analysed hundreds of frames, so gating on the route and gating on frames
    analysed produce the same output and a mutation between them passes. The window R3 is
    about is the startup one, so the test drives it directly: a real render of the real
    `Main.qml`, with the bridge **route confirmed and nothing analysed** - which is a state
    the worker genuinely produces, because a placement is established from the route and
    the route arrives before any result.

    Then one more sample set of frames analysed, and the row must switch to numbers.
    """

    @classmethod
    def setUpClass(cls):
        cls.rows = cls._render()

    def test_a_confirmed_route_with_nothing_analysed_does_not_print_zero(self):
        # The exact R3 state: `device_confirmed` non-empty, `frames_analysed == 0`.
        self.assertTrue(self.rows["route_confirmed"], self.rows)
        self.assertEqual(self.rows["frames_analysed_at_first_read"], 0, self.rows)
        self.assertIn("unavailable", self.rows["row_at_first_read"],
                      f"the footer printed a measured 0 before anything had been "
                      f"analysed: {self.rows['row_at_first_read']!r}")
        for kind in ("zone_occupancy", "candidate_transfer", "candidate_return"):
            self.assertIn(f"{kind} unavailable", self.rows["row_at_first_read"],
                          self.rows)

    def test_the_row_switches_to_numbers_once_a_frame_has_been_analysed(self):
        self.assertGreater(self.rows["frames_analysed_at_second_read"], 0, self.rows)
        self.assertNotIn("unavailable", self.rows["row_at_second_read"],
                         f"frames have been analysed, so this is a measurement now: "
                         f"{self.rows['row_at_second_read']!r}")
        for kind in ("zone_occupancy", "candidate_transfer", "candidate_return"):
            self.assertIn(kind, self.rows["row_at_second_read"], self.rows)

    @staticmethod
    def _render() -> dict:
        """Render `Main.qml` twice against one bridge, reading the event row each time."""
        code = _EVENT_ROW_SCRIPT.replace("@REPO@", repr(str(REPO_ROOT)))
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYTHONPATH"] = str(REPO_ROOT)
        with tempfile.TemporaryDirectory(prefix="roadshow-eventrow-") as tmp:
            script = Path(tmp) / "event_row.py"
            script.write_text(code, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(script)], capture_output=True, text=True,
                timeout=420, cwd=str(REPO_ROOT), env=env)
        if completed.returncode != 0:
            raise AssertionError("the event-row render failed:\n"
                                 + completed.stdout + completed.stderr)
        lines = [ln for ln in completed.stdout.splitlines() if ln.startswith("ROWS ")]
        if not lines:
            raise AssertionError("the render printed no rows:\n"
                                 + completed.stdout + completed.stderr)
        return json.loads(lines[-1][len("ROWS "):])


#: The child for `EventRowGating`. Drives the bridge directly - no pipeline - because the
#: property under test is a QML binding's reaction to two bridge states, and a pipeline
#: would make both states unreachable at a chosen moment.
_EVENT_ROW_SCRIPT = r'''
import json
import os
import sys

sys.path.insert(0, @REPO@)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer, QUrl, QObject
from PySide6.QtGui import QFontDatabase, QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from app.ui.bridge import ThemeTokens, UiBridge
from app.ui.provider import FrameImageProvider

# **The application first, the font second.** Measured: calling
# `QFontDatabase.addApplicationFont` before a `QGuiApplication` exists terminates the
# process with `0xC0000005` and prints nothing at all - which cost a bisection here. The
# existing render script has the order right and always did.
application = QGuiApplication(sys.argv)
QFontDatabase.addApplicationFont(r"C:\Windows\Fonts\arial.ttf")
bridge = UiBridge()
engine = QQmlApplicationEngine()
engine.addImageProvider("frames", FrameImageProvider(bridge))
context = engine.rootContext()
context.setContextProperty("bridge", bridge)
context.setContextProperty("Theme", ThemeTokens())
engine.load(QUrl.fromLocalFile(
    os.path.join(@REPO@, "app", "ui", "Main.qml")))
roots = engine.rootObjects()
if not roots:
    print("no root objects", file=sys.stderr)
    sys.exit(3)
root = roots[0]

# **M4: the event row lives in the presenter drawer.** It is a presenter's reading of how
# many events the machine has counted, not a number a passer-by reads from the aisle at 3 m,
# so the drawer is opened here and the row is read from it - as rendered text, which is the
# only thing that proves the binding re-evaluated.
root.setProperty("drawerOpen", True)


def row():
    item = root.findChild(object, "eventCountRow")
    return "" if item is None else str(item.property("text"))


def step_two():
    report["row_at_first_read"] = row()
    report["frames_analysed_at_first_read"] = bridge.frames_analysed
    report["route_confirmed"] = bool(bridge.device_confirmed)
    # Now the state the pipeline reaches a moment later.
    bridge.set_frames_analysed(7)
    report["row_at_second_read"] = row()
    report["frames_analysed_at_second_read"] = bridge.frames_analysed
    report["rows"] = "printed"
    print("ROWS " + json.dumps(report), flush=True)
    sys.stdout.flush()
    loop.quit()


report = {"rows": "", "row_at_first_read": "", "row_at_second_read": "",
          "frames_analysed_at_first_read": -1, "frames_analysed_at_second_read": -1,
          "route_confirmed": False}

# **Route confirmed, nothing analysed.** This is a state the worker really produces.
bridge.set_route(requested="NPU", confirmed=("CPU",), state="active")

# A `QEventLoop`, **not** `application.exec()`. Measured here: driving a `QQuickWindow`
# through `QGuiApplication.exec()` under the offscreen platform terminates the process with
# `0xC0000005` on this machine, with nothing on stdout or stderr. The existing render tests
# use a `QEventLoop` for exactly this reason and are unaffected.
loop = QEventLoop()
QTimer.singleShot(600, step_two)
QTimer.singleShot(30000, loop.quit)   # backstop; a wedged render must not hang a suite
loop.exec()
if not report["rows"]:
    print("NO ROWS", file=sys.stderr)
    sys.exit(4)
sys.exit(0)
'''


# ======================================================================================
# M3 Task 4 (attract mode) and Task 5 (the presenter drawer)
# ======================================================================================

class AttractModeState(unittest.TestCase):
    """The bridge's attract-mode state, asserted in Python. No window, no hardware.

    The screen test (`AttractModeOnTheScreen`) proves what reaches the display; this proves
    the properties attract mode rests on: that every headline is dated and sourced, that
    the numbers match the committed measurements they cite, and that attract mode owns
    exactly one field and touches nothing else.
    """

    def _seeded(self):
        from app.ui.bridge import UiBridge

        bridge = UiBridge()
        bridge.set_streams(["A", "B", "C"])
        for sid in ("A", "B", "C"):
            bridge.apply_route({"stream": sid, "requested": "CPU", "confirmed": ["CPU"],
                                "state": "active", "fell_back": False, "errors": []})
            bridge.set_stream_metrics(sid, age_p50_ms=11319.0, age_p95_ms=12286.0,
                                      backlog=187, analysis_fps=13.5, window_samples=256)
        bridge.set_metrics(system_cpu_percent=58.0, ai_fps=14.9)
        return bridge

    def test_every_headline_is_dated_and_sourced_to_a_committed_file(self):
        from app.inference.pinned import REPO_ROOT as PIN_ROOT
        from app.ui.bridge import ATTRACT_HEADLINES

        def resolves(source: str) -> list:
            """The committed paths a source names. A range may cite several files."""
            path = source.split(" (", 1)[0]
            matches = sorted(PIN_ROOT.glob(path)) if "*" in path else [PIN_ROOT / path]
            return [match for match in matches if match.exists()]

        self.assertTrue(ATTRACT_HEADLINES, "attract mode with no headline shows a title only")
        for headline in ATTRACT_HEADLINES:
            self.assertRegex(headline["date"], r"^\d{4}-\d{2}-\d{2}$",
                             f"a headline must carry the date it was recorded: {headline}")
            source = headline["source"]
            self.assertTrue(source, f"a headline must name its source: {headline}")
            self.assertTrue(resolves(source),
                            f"the source {source!r} does not exist in the repository, so the "
                            f"figure on screen cites nothing: {headline}")

    def test_the_headline_numbers_match_the_committed_measurements_they_cite(self):
        """**A dated figure whose number has drifted from its source is still a lie.**

        The headline ranges are re-derived here from the committed JSON files, so a claim
        that keeps its date while its numbers move fails rather than shipping.
        """
        import json

        from app.inference.pinned import REPO_ROOT as PIN_ROOT
        from app.ui.bridge import ATTRACT_HEADLINES

        def _range(values, digits):
            return f"{min(values):.{digits}f}-{max(values):.{digits}f}"

        latency_dir = PIN_ROOT / "docs" / "qualification" / "m2-headless"
        runs = [json.loads((latency_dir / f"single-frame-latency-run{n}.json").read_text(
            encoding="utf-8")) for n in range(1, 5)]
        by_row = [{row["row"]: row for row in run["rows"]} for run in runs]
        for device in ("CPU", "GPU", "NPU"):
            want = _range([round(rows[device]["latency_ms_p50"], 2) for rows in by_row], 2)
            self.assertIn(want, ATTRACT_HEADLINES[0]["text"],
                          f"the {device} single-frame range in the headline must be {want}, "
                          f"recomputed from the four committed runs: {ATTRACT_HEADLINES[0]}")

        holds = [json.loads((latency_dir / name).read_text(encoding="utf-8"))
                 for name in ("allcpu-steady-120s.json", "allcpu-steady-120s-hold2.json",
                              "allcpu-steady-120s-hold3.json")]
        p50 = _range([round(hold["frame_age_p50_ms"] / 1000.0, 1) for hold in holds], 1)
        p95 = _range([round(hold["frame_age_p95_ms"] / 1000.0, 1) for hold in holds], 1)
        backlogs = [entry["backlog"] for hold in holds for entry in hold["per_stream"]]
        backlog = f"{min(backlogs)}-{max(backlogs)}"
        for want in (p50, p95, backlog):
            self.assertIn(want, ATTRACT_HEADLINES[1]["text"],
                          f"the three-hold headline must carry {want}, recomputed from the "
                          f"three committed holds: {ATTRACT_HEADLINES[1]}")
        # **The words, not only the numbers.** "0 drops" and "all three active" are claims
        # about the same three holds, and a headline that keeps its numbers while one of
        # these stops being true is the same defect as a drifted range.
        self.assertEqual(sum(entry["drops"] for hold in holds for entry in hold["per_stream"]), 0,
                         "the headline says 0 drops; the committed holds record some")
        self.assertIn("0 drops", ATTRACT_HEADLINES[1]["text"])
        self.assertEqual([entry["state"] for hold in holds for entry in hold["per_stream"]],
                         ["active"] * 9,
                         "the headline says all three active in every hold")
        self.assertIn("all three active", ATTRACT_HEADLINES[1]["text"])

    def test_the_single_frame_headline_states_its_conditions_not_only_its_numbers(self):
        """The conditions are part of the claim: 30 inferences, 640x640, batch 1.

        A latency range without them is a number a visitor cannot compare with anything -
        and it is a number a later run with a different input size would silently invalidate.
        """
        import json

        from app.inference.pinned import REPO_ROOT as PIN_ROOT
        from app.ui.bridge import ATTRACT_HEADLINES

        run = json.loads((PIN_ROOT / "docs" / "qualification" / "m2-headless"
                          / "single-frame-latency-run1.json").read_text(encoding="utf-8"))
        text = ATTRACT_HEADLINES[0]["text"]
        self.assertIn(f"{run['num_requests_per_row']} inferences", text,
                      f"the run measured {run['num_requests_per_row']} per row: {text}")
        shape = run["input_shape"]
        self.assertEqual(shape[0], 1, f"the run was batch {shape[0]}, not 1")
        self.assertIn(f"{shape[2]}x{shape[3]}", text,
                      f"the input was {shape[2]}x{shape[3]}: {text}")
        self.assertIn("batch 1", text)
        # And the fourth row is not folded into the three devices as if it were one: the
        # fallback's own numbers are a separate run and are not on the slide.
        self.assertEqual(sorted(row["row"] for row in run["rows"]),
                         ["CPU", "CPU-fallback-baseline", "GPU", "NPU"],
                         "the committed run's rows are not the four this test assumes")

    def test_the_delivered_event_headline_is_what_the_m1_record_states(self):
        """**"2 events per clip" is a measurement, and the M1 report is where it lives.**

        Parsed from the record's own event block rather than trusted: two `zone_occupancy`
        events, one per person track. A headline that keeps its date while the count moves
        would cite a file that no longer says what the slide says.
        """
        from app.inference.pinned import REPO_ROOT as PIN_ROOT
        from app.ui.bridge import ATTRACT_HEADLINES

        record = (PIN_ROOT / "docs" / "qualification" / "M1-qualification.md").read_text(
            encoding="utf-8")
        event_lines = [line for line in record.splitlines()
                       if line.strip().startswith("kind=zone_occupancy")]
        self.assertEqual(len(event_lines), 2,
                         f"the M1 record now lists {len(event_lines)} zone_occupancy events: "
                         f"{event_lines}")
        self.assertIn("events 2", record, "the M1 record no longer states the event count")
        text = ATTRACT_HEADLINES[2]["text"]
        self.assertIn("2 events per clip", text, text)
        self.assertIn("people-in-zone presence", text, text)
        self.assertIn("NOT a people count", text, text)
        self.assertIn("NOT a dwell measurement", text, text)
        # And the count on the slide is the count in the record, not a coincidence of prose.
        self.assertIn(f"events {len(event_lines)}", record)

    def test_attract_mode_shows_no_live_reading(self):
        # The seeded live value is 58.0%. It must not appear in any recorded headline, and
        # the headlines must be the module constant - not something derived from live state.
        from app.ui.bridge import ATTRACT_HEADLINES

        bridge = self._seeded()
        self.assertEqual(bridge.display_text("system_cpu_percent"), "58.0%", bridge.metric_text)
        bridge.set_attract(True)
        self.assertEqual([dict(h) for h in bridge.attract_headlines],
                         [dict(h) for h in ATTRACT_HEADLINES])
        for headline in bridge.attract_headlines:
            self.assertNotIn("58.0%", headline["text"], headline)

    def test_entering_and_leaving_attract_touches_nothing_else(self):
        # **Byte for byte.** The whole instance state is compared, not a hand-picked subset:
        # attract mode owns exactly `_attract_active`, so entering and leaving it must leave
        # every other field identical.
        bridge = self._seeded()

        def state():
            return {key: value for key, value in bridge.__dict__.items()
                    if key != "_attract_active"}

        before = state()
        bridge.set_attract(True)
        self.assertTrue(bridge.attract_active)
        self.assertEqual(state(), before,
                         "entering attract mode changed something other than its own flag")
        bridge.set_attract(False)
        self.assertFalse(bridge.attract_active)
        self.assertEqual(state(), before,
                         "leaving attract mode did not restore the state it interrupted")

    def test_the_idle_policy_is_ninety_seconds_and_any_input_resets_it(self):
        from app.ui.bridge import ATTRACT_IDLE_SECONDS, UiBridge

        self.assertEqual(ATTRACT_IDLE_SECONDS, 90.0, "spec section 9's idle interval")
        bridge = UiBridge()
        bridge._last_activity = 1000.0
        self.assertFalse(bridge.attract_due(now=1000.0 + 89.9))
        self.assertTrue(bridge.attract_due(now=1000.0 + 90.0))
        # Any input leaves attract mode and restarts the countdown.
        bridge.set_attract(True)
        bridge.note_activity(now=2000.0)
        self.assertFalse(bridge.attract_active,
                         "presenter input must dismiss the attract overlay")
        self.assertFalse(bridge.attract_due(now=2000.0 + 89.0))
        self.assertTrue(bridge.attract_due(now=2000.0 + 90.0))

    def test_attract_mode_is_refused_in_a_release_build_with_a_simulated_provider(self):
        # Attract mode is the one surface a passer-by reads with no presenter next to it, so
        # the release gate is re-asserted where attract would start.
        from app.main import attract_mode_permitted

        with self.assertRaises(SystemExit) as caught:
            attract_mode_permitted(release=True, providers=("simulated_cpu_metrics",))
        self.assertIn("simulated_cpu_metrics", str(caught.exception),
                      "the refusal must name the provider it refused")
        self.assertTrue(attract_mode_permitted(release=False,
                                               providers=("simulated_cpu_metrics",)),
                        "a developer's own machine may simulate; the default is the release "
                        "gate")
        self.assertTrue(attract_mode_permitted(release=True, providers=("psutil",)))


#: The live items whose positions must be identical with the drawer open and closed.
#: **Named explicitly, and compared one by one**, because "nothing threw" is not the claim
#: the brief asks for - it asks that the hero, the badges, the tiles and the staleness line
#: do not move, item by item.
_DRAWER_STABLE_ITEMS = (
    # the hero
    "heroTile",
    # the header: the mandatory label, the title and the badge
    "recordedFootageText", "appTitle", "deviceBadgeText",
    # the two thumbnails, and all three pictures
    "thumb_B", "thumb_C",
    "tileImage_A", "tileImage_B", "tileImage_C",
    "tileCaptionText_A", "tileCaptionText_B", "tileCaptionText_C",
    # **the three badges** - one per camera, each stating that camera's confirmed device
    "streamBadgeBox_A", "streamBadgeBox_B", "streamBadgeBox_C",
    "streamBadgeText_A", "streamBadgeText_B", "streamBadgeText_C",
    # the per-camera name, state and age
    "cameraName_A", "cameraName_B", "cameraName_C",
    "cameraState_A", "cameraState_B", "cameraState_C",
    "streamAgeText_A", "streamAgeText_B", "streamAgeText_C",
    # the verdict lines, which are the reason the screen exists
    "verdictRow_A", "verdictRow_B", "verdictRow_C",
    "streamStaleText_A", "streamStaleText_B", "streamStaleText_C",
    "streamReasonText_C",
    # the three cards and the screen itself
    "card_A", "card_B", "card_C",
    "liveScreen",
)

#: The child for `AttractModeOnTheScreen` and `PresenterDrawerOnTheScreen`. One process,
#: because a `QGuiApplication` is a singleton; it drives the **real** `Main.qml` and reads
#: every string and every extent back through `tools.capture_ui`'s own reader, so there is
#: one definition of "what is on screen" rather than a second one written for this test.
_DRAWER_ATTRACT_SCRIPT = r'''
import json
import os
import sys

sys.path.insert(0, @REPO@)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer, QUrl, QObject, QPoint, QPointF, Qt
from PySide6.QtGui import QFontDatabase, QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickWindow
from PySide6.QtTest import QTest

import tools.capture_ui as harness
import app.theme as theme_module
from app.ui.bridge import ThemeTokens, UiBridge
from app.ui.provider import FrameImageProvider

application = QGuiApplication(sys.argv[:1])
QFontDatabase.addApplicationFont(r"C:\Windows\Fonts\arial.ttf")
bridge = UiBridge(application)
engine = QQmlApplicationEngine()
# The QML engine's own complaints, kept so a file that does not load is reported with the
# reason rather than as a missing window.
engine_warnings = []


def _note_warnings(warnings):
    for warning in warnings:
        engine_warnings.append(str(warning.toString()))


engine.warnings.connect(_note_warnings)
engine.addImageProvider("frames", FrameImageProvider(bridge))
context = engine.rootContext()
context.setContextProperty("bridge", bridge)
context.setContextProperty("Theme", ThemeTokens(application))
engine.load(QUrl.fromLocalFile(os.path.join(@REPO@, "app", "ui", "Main.qml")))
root = next((r for r in engine.rootObjects() if isinstance(r, QQuickWindow)), None)
if root is None:
    # **A `Main.qml` that fails to load is reported, not left to look like a harness
    # failure.** Measured while mutation-testing this file: a QML compile error produced
    # `StopIteration` here, no report, and the parent said "the child produced no report" -
    # which names the last symptom and not the cause.
    print("REPORT " + json.dumps({
        "errors": ["Main.qml produced no window; the QML engine said: "
                   + " | ".join(engine_warnings)],
        "done": True, "steps": [], "affordances": {},
        "drawer": {"explanation_texts": {}}, "attract": {}}))
    sys.stdout.flush()
    sys.exit(5)

# **The window is sized explicitly, and this line is the whole difference between a
# measurement and a joke.** Measured: without it `Main.qml` came up **1x1**, every item's
# extent was `(0, 0, 0, 0)`, every click landed on nothing and every `font.pixelSize` read
# `-1` - so the first version of this child "clicked" eight buttons that could not be
# clicked and read a census of `{-1: 31}`. `tools/capture_ui.py` sets the size the same
# way, from the canvas tokens rather than from literals.
root.setWidth(float(theme_module.CANVAS_WIDTH))
root.setHeight(float(theme_module.CANVAS_HEIGHT))

# A **live** state, so the attract render can prove it shows none of it.
bridge.set_streams(["A", "B", "C"])
for _sid in ("A", "B", "C"):
    bridge.apply_route({"stream": _sid, "requested": "CPU", "confirmed": ["CPU"],
                        "state": "active", "fell_back": False, "errors": []})
    bridge.set_stream_metrics(_sid, age_p50_ms=11319.0, age_p95_ms=12286.0,
                              backlog=187, analysis_fps=13.5, window_samples=256)
bridge.set_metrics(system_cpu_percent=58.0, ai_fps=14.9)

report = {"affordances": {}, "drawer": {"explanation_texts": {}}, "attract": {},
          "errors": [], "steps": []}
report["window"] = [float(root.width()), float(root.height())]


def walk():
    names = {}
    for name, item in harness.walk_items(root.contentItem()):
        if item is not None:
            names[name] = item
    return names


def render():
    """Force a render before reading.

    **The order is load-bearing and it is the harness's own rule**: `grabWindow()` forces a
    render, and only then is `font.pixelSize` populated on a `Text` (measured here: without
    a grab the census reads `-1` for everything) and only then is `truncated` decided.
    """
    root.grabWindow()
    application.processEvents()


def read():
    render()
    return harness.read_scene(
        root.contentItem(),
        window_rect=(0.0, 0.0, float(root.width()), float(root.height())))


def click_named(name):
    # Render first: a click is dispatched by hit-testing against the scene's current
    # geometry, and reading geometry that was never laid out is how the first version
    # clicked a 0x0 button in a 1x1 window and read `''` back.
    render()
    names = walk()
    if name not in names:
        raise AssertionError("no item named %r; the items are %s" % (name, sorted(names)))
    item = names[name]
    pos = item.mapToScene(QPointF(item.width() / 2.0, item.height() / 2.0))
    QTest.mouseClick(root, Qt.LeftButton, Qt.NoModifier,
                     QPoint(int(round(pos.x())), int(round(pos.y()))))


#: **The keys, from the bridge - the one definition the surfaces read.** The drawer's buttons
#: are driven from this list, so the child cannot silently click nothing.
KEYS = list(bridge.explanation_keys)
state = {"i": 0, "drawer_i": 0, "card_done": False}


def extents(scene):
    return {k: list(v) for k, v in scene.extents.items()}


def census(scene):
    """`font px -> how many items carry it **and are visible with text on screen**`.

    The filter is what makes this the census the M4 report quotes. **M4 added `visible`**:
    an item whose text is non-empty but hidden draws nothing, and counting it would put the
    reply to "what is on the screen" in the hands of an item that is not. (M3's census
    filtered on non-empty text only, which is why a hidden 18 px hover hint would have been
    counted as type on the booth screen.)
    """
    counts = {}
    for name, size in scene.font_pixels.items():
        if scene.strings.get(name) and scene.visible.get(name):
            counts[size] = counts.get(size, 0) + 1
    return counts


def has(name):
    """Whether `name` is in the scene graph at all.

    **A `Loader`'s `status` cannot be read from PySide6 here** - measured: both
    `property("status")` and `QMetaProperty.read` raise `RuntimeError: Can't find converter
    for 'QQuickLoader::Status'`, so a check written against it would have been a check that
    never ran. Presence of the loaded content is the claim that matters anyway: a Loader
    whose component is present is a drawer that is really there, and its absence is the
    state that reads as a blank panel.
    """
    return name in walk()


def step(name):
    report["steps"].append(name)


def affordance_step():
    # **M4: one touch entry point, not seven 18 px keys.** The seven `?` chips are gone from
    # the footer (their labels were 18 px, i.e. below the acuity floor at 3 m, and they are
    # all inside the drawer), so what is driven here is the header's `what the boxes mean`
    # button and then a card's own click.
    step("affordance:open_button")
    click_named("explainOpenMouse")
    report["explain_open"] = {
        "drawer_open": bool(root.property("drawerOpen")),
        "drawer_present": has("presenterDrawer"),
        "footer_button_present": has("explainButtonMouse_yolo"),
    }
    root.setProperty("drawerOpen", False)
    QTimer.singleShot(250, guard(card_open_step))


def card_open_step():
    step("affordance:card")
    click_named("cardExplainMouse_A")
    report["card_open"] = {
        "drawer_open": bool(root.property("drawerOpen")),
        "drawer_present": has("presenterDrawer"),
    }
    root.setProperty("drawerOpen", False)
    QTimer.singleShot(250, guard(drawer_step))


def drawer_step():
    step("drawer:closed_baseline")
    base = read()
    report["drawer"]["base_extents"] = extents(base)
    report["drawer"]["base_visible"] = dict(base.visible)
    report["drawer"]["base_census"] = census(base)
    report["drawer"]["present_before"] = has("presenterDrawer")
    root.setProperty("drawerOpen", True)
    QTimer.singleShot(300, guard(drawer_open_step))


def drawer_open_step():
    step("drawer:open")
    opened = read()
    report["drawer"]["strings"] = dict(opened.strings)
    report["drawer"]["open_extents"] = extents(opened)
    report["drawer"]["open_census"] = census(opened)
    report["drawer"]["overflow"] = list(opened.overflow)
    report["drawer"]["clip_sources"] = dict(opened.clip_sources)
    report["drawer"]["present_when_open"] = has("presenterDrawer")
    report["drawer"]["expected_scope_note"] = bridge.scope_note
    report["drawer"]["expected_explanations"] = {
        key: bridge.explanation(key) for key in KEYS}
    # **The drawer's buttons must not be the footer's buttons under another name.** M4
    # removed the footer's pinned panel outright (one explanation surface, at a legible
    # size), so what is asserted is that there is no second surface for a click to land in:
    # the window has no `pinnedExplanationKey` at all.
    report["drawer"]["pinned_property_exists"] = (
        root.property("pinnedExplanationKey") is not None)
    QTimer.singleShot(0, guard(drawer_explanation_step))


def drawer_explanation_step():
    step("drawer:explain:%d" % state["drawer_i"])
    if state["drawer_i"] < len(KEYS):
        key = KEYS[state["drawer_i"]]
        click_named("drawerExplainMouse_" + key)
        scene = read()
        report["drawer"]["explanation_texts"][key] = {
            "shown": str(scene.strings.get("drawerExplanationText")),
            "drawer_open": bool(root.property("drawerOpen")),
        }
        state["drawer_i"] += 1
        QTimer.singleShot(150, guard(drawer_explanation_step))
        return
    root.setProperty("drawerOpen", False)
    QTimer.singleShot(250, guard(drawer_closed_step))


def drawer_closed_step():
    step("drawer:closed_again")
    closed = read()
    report["drawer"]["closed_extents"] = extents(closed)
    report["drawer"]["closed_visible"] = dict(closed.visible)
    report["drawer"]["closed_census"] = census(closed)
    report["drawer"]["present_after_close"] = has("presenterDrawer")
    QTimer.singleShot(150, guard(keyboard_step))


def keyboard_step():
    step("keyboard:open")
    shortcut = root.findChild(QObject, "drawerOpenShortcut")
    report["drawer"]["open_shortcut_found"] = shortcut is not None
    if shortcut is not None:
        shortcut.activated.emit()
    QTimer.singleShot(250, guard(keyboard_open_step))


def keyboard_open_step():
    step("keyboard:opened")
    report["drawer"]["open_after_f1"] = bool(root.property("drawerOpen"))
    report["drawer"]["items_after_f1"] = sorted(
        name for name in walk() if name.startswith("drawerExplainMouse_"))
    escape = root.findChild(QObject, "escapeShortcut")
    report["drawer"]["escape_found"] = escape is not None
    if escape is not None:
        escape.activated.emit()
    QTimer.singleShot(250, guard(keyboard_closed_step))


def keyboard_closed_step():
    step("keyboard:closed")
    report["drawer"]["closed_after_escape"] = not bool(root.property("drawerOpen"))
    report["drawer"]["items_after_escape"] = sorted(
        name for name in walk() if name.startswith("drawerExplainMouse_"))
    QTimer.singleShot(150, guard(attract_step))


def attract_step():
    step("attract:baseline")
    # **The baseline, with the drawer closed and attract mode not yet entered.** The
    # round-trip assertion compares against this, not against the drawer-open read.
    baseline = read()
    report["attract"]["before_strings"] = dict(baseline.strings)
    report["attract"]["before_visible"] = dict(baseline.visible)
    report["attract"]["before_extents"] = extents(baseline)
    report["attract"]["before_census"] = census(baseline)
    report["attract"]["overlay_present_before"] = has("attractOverlay")
    bridge.set_attract(True)
    QTimer.singleShot(300, guard(attract_read_step))


def attract_read_step():
    step("attract:up")
    scene = read()
    report["attract"]["strings"] = dict(scene.strings)
    report["attract"]["visible"] = dict(scene.visible)
    report["attract"]["extents"] = extents(scene)
    report["attract"]["overflow"] = list(scene.overflow)
    report["attract"]["census"] = census(scene)
    report["attract"]["overlay_present"] = has("attractOverlay")
    report["attract"]["live_screen_visible"] = bool(scene.visible.get("liveScreen"))
    report["attract"]["overlay_visible"] = bool(scene.visible.get("attractOverlay"))
    report["attract"]["live_metric_text"] = bridge.display_text("system_cpu_percent")
    bridge.set_attract(False)
    QTimer.singleShot(300, guard(after_exit_step))


def after_exit_step():
    step("attract:down")
    scene = read()
    report["attract"]["after_exit_strings"] = dict(scene.strings)
    report["attract"]["after_exit_visible"] = dict(scene.visible)
    report["attract"]["after_exit_extents"] = extents(scene)
    report["attract"]["after_exit_census"] = census(scene)
    report["attract"]["live_screen_visible_after"] = bool(scene.visible.get("liveScreen"))
    report["attract"]["overlay_present_after"] = has("attractOverlay")
    report["done"] = True
    print("REPORT " + json.dumps(report))
    sys.stdout.flush()
    loop.quit()


def guard(fn):
    """**Every** step runs through this, not only the first.

    The first version wrapped the initial callback and nothing else, so an exception in
    `drawer_open_step` was swallowed by Qt's event loop, no further step was scheduled, the
    loop ran to its timeout and the child printed `NO REPORT` - which the parent reported
    as "the child produced no report", naming the harness rather than the defect. A step
    that raises now reports its own type and message, and the parent fails with that.
    """
    def wrapped():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a child failure is reported, not hidden
            report["errors"].append("%s: %s" % (type(exc).__name__, exc))
            report["failed_at"] = report["steps"][-1] if report["steps"] else "start"
            report["done"] = True
            print("REPORT " + json.dumps(report))
            sys.stdout.flush()
            loop.quit()
    return wrapped


loop = QEventLoop()
QTimer.singleShot(200, guard(affordance_step))
QTimer.singleShot(60000, loop.quit)
loop.exec()
if not report.get("done"):
    print("NO REPORT: the child's loop ended without a step completing; the last step "
          "started was %r" % (report["steps"][-1] if report["steps"] else "none"),
          file=sys.stderr)
    sys.exit(4)
sys.exit(0)
'''


def _expected_steps() -> list[str]:
    """The child's full step sequence, from the key list rather than typed out.

    **This is the harness's self-check.** The parent asserts every one of these ran, in
    order, so a report that stopped early cannot be read as a pass: the failure mode this
    guards is a child that produced *some* JSON and was accepted because the keys the tests
    happened to read were present.
    """
    steps = ["affordance:open_button", "affordance:card",
             "drawer:closed_baseline", "drawer:open"]
    steps += [f"drawer:explain:{i}" for i in range(len(KEY_ORDER) + 1)]
    steps += ["drawer:closed_again", "keyboard:open", "keyboard:opened",
              "keyboard:closed", "attract:baseline", "attract:up", "attract:down"]
    return steps


#: One child, three test classes: a `QGuiApplication` is a singleton and the render is the
#: expensive part. The first version ran it once per class - three windows, three launches,
#: ~40 s each when a step failed - for one measurement.
_REPORT_CACHE: dict | None = None


def _drawer_attract_report() -> dict:
    """Run the child once, **check it finished and that its report is not empty**, and
    return the report. The result is cached for the other classes."""
    global _REPORT_CACHE
    if _REPORT_CACHE is not None:
        return _REPORT_CACHE
    code = _DRAWER_ATTRACT_SCRIPT.replace("@REPO@", repr(str(REPO_ROOT)))
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = str(REPO_ROOT)
    with tempfile.TemporaryDirectory(prefix="roadshow-drawer-") as tmp:
        script = Path(tmp) / "drawer_attract.py"
        script.write_text(code, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True,
            timeout=180, cwd=str(REPO_ROOT), env=env)
    lines = [ln for ln in completed.stdout.splitlines() if ln.startswith("REPORT ")]
    if not lines:
        raise AssertionError(
            "the drawer/attract child produced no report - it did not finish. The child "
            "prints NO REPORT with the last step it started; both are below.\nSTDOUT:\n"
            + completed.stdout[-3000:] + "\nSTDERR:\n" + completed.stderr[-3000:])
    report = json.loads(lines[-1][len("REPORT "):])
    if report.get("errors"):
        raise AssertionError(
            f"the child failed at {report.get('failed_at')!r}: {report['errors']}")
    if not report.get("done"):
        raise AssertionError("the child printed a report but never marked it done")
    # **An empty report must never read as a pass**, so the shape is asserted here rather
    # than left to each test: a section that arrived as `{}` would make every `.get(...)`
    # in the classes below return `None`, and `assertIn("x", "")` on an absent string is
    # the vacuous shape this repository keeps finding.
    missing = [f"{section}.{key}" for section, key in (
        ("drawer", "strings"), ("drawer", "base_extents"), ("drawer", "open_extents"),
        ("drawer", "closed_extents"), ("drawer", "explanation_texts"),
        ("drawer", "base_census"), ("attract", "strings"),
        ("attract", "before_strings"), ("attract", "after_exit_strings"),
        ("attract", "before_visible"), ("attract", "after_exit_visible"),
        ("attract", "extents")) if not report.get(section, {}).get(key)]
    if missing:
        raise AssertionError(f"the child's report is empty where it must not be: {missing}")
    if report.get("steps") != _expected_steps():
        raise AssertionError(
            "the child did not run every step: expected "
            f"{_expected_steps()}, got {report.get('steps')}")
    if len(report["drawer"]["explanation_texts"]) != len(KEY_ORDER):
        raise AssertionError(
            f"the child clicked {len(report['drawer']['explanation_texts'])} drawer buttons, "
            f"not {len(KEY_ORDER)}: {sorted(report['drawer']['explanation_texts'])}")
    for section in ("explain_open", "card_open"):
        if report.get(section, {}).get("drawer_open") is not True:
            raise AssertionError(
                f"{section} did not open the explanation surface: {report.get(section)!r}")
    _REPORT_CACHE = report
    return report


class ExplanationEntryPoints(unittest.TestCase):
    """**Every explanation affordance is proved to work, not merely present.**

    M1 shipped a `?` panel that called `bridgeUi.explain(...)` when `Bridge.qml` had no
    `explain` function. The call sat behind `pinnedExplanationKey.length > 0` in a ternary,
    QML short-circuited it, and the six explanation buttons did nothing at all - on the
    screen whose purpose is explaining what the boxes mean. A button that is present and
    dead is invisible until somebody clicks it, which at a booth is the worst moment.

    **M4 moved the affordances and kept the proof.** The seven 18 px `?` chips cannot be
    read at 3 m and duplicated the drawer's own buttons, so the always-visible entry points
    are now two, both 44 px and both touch-reachable: the header's `what the boxes mean`
    button and a click on any camera card. Each is clicked here, through the real
    `Main.qml` and the real scene graph, and each must put the explanation surface in the
    scene graph. The per-key proof - that every button shows *its own* text - is
    `PresenterDrawerOnTheScreen::test_every_drawer_button_shows_its_own_explanation`.
    """

    @classmethod
    def setUpClass(cls):
        cls.report = _drawer_attract_report()

    def test_the_header_button_opens_the_explanation_surface(self):
        opened = self.report["explain_open"]
        self.assertTrue(opened["drawer_open"],
                        f"the header's `what the boxes mean` button did not open the drawer: "
                        f"{opened}")
        self.assertTrue(opened["drawer_present"],
                        "the drawer's Loader is open but its content is not in the scene - "
                        "the M1 dead-affordance shape, one layer down")
        self.assertFalse(opened["footer_button_present"],
                         "the footer's 18 px `?` row is still there, so there are two "
                         "explanation surfaces to keep in step")

    def test_a_cards_click_opens_the_explanation_surface(self):
        opened = self.report["card_open"]
        self.assertTrue(opened["drawer_open"],
                        f"clicking a camera card did not open the explanation surface: "
                        f"{opened}")
        self.assertTrue(opened["drawer_present"], opened)

    def test_each_required_explanation_is_reachable_from_the_screen(self):
        # The design's section 9 names six required explanations; the screen offers seven
        # (it adds `utilisation`). Every one of them must be a button that works.
        offered = self.report["drawer"]["explanation_texts"]
        for key in ("yolo", "tracking", "zone", "cpu", "gpu", "metrics"):
            self.assertIn(key, offered, key)
            self.assertTrue(offered[key]["shown"].strip(), f"{key} showed nothing")


class ExplanationKeysHaveOneDefinition(unittest.TestCase):
    """**The key list is a property of the bridge, not a literal in a QML file.**

    The drawer's `Repeater` was bound to `bridgeUi.explanationKeys` - a property of the
    *`Bridge.qml` object*, which a file loaded by a `Loader` cannot see. The model was
    `undefined`, the `Repeater` built **zero delegates**, and the drawer shipped with no
    explanation buttons at all while every one of its text bindings worked. The footer kept
    its own literal list, so the two surfaces could have offered different keys and nothing
    would have said so.

    One definition now: `app.ui.bridge.EXPLANATIONS`, exposed as `explanation_keys`, read by
    both surfaces. This class pins it, including the absence of a second copy in QML.
    """

    def test_the_bridge_offers_every_key_it_has_an_explanation_for(self):
        from app.ui.bridge import EXPLANATIONS, UiBridge

        bridge = UiBridge()
        self.assertEqual(list(bridge.explanation_keys), list(EXPLANATIONS),
                         "the key list must be the explanations' own keys, in their order")
        self.assertEqual(tuple(bridge.explanation_keys), KEY_ORDER,
                         f"the surfaces offer {bridge.explanation_keys}; the design's six "
                         f"plus `utilisation` are {list(KEY_ORDER)}")
        for key in bridge.explanation_keys:
            self.assertTrue(bridge.explanation(key).strip(),
                            f"{key!r} is offered as a button and has no explanation text")

    def test_no_qml_file_carries_a_second_copy_of_the_key_list(self):
        # A literal key list in QML is a second definition that can drift from
        # `EXPLANATIONS`, and drift is invisible until a presenter clicks a button whose
        # text comes from somewhere else.
        offenders = []
        for path in sorted(UI_DIR.glob("*.qml")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("//"):
                    continue
                if '"yolo"' in stripped and '"tracking"' in stripped:
                    offenders.append(f"{path.name}:{number}: {stripped}")
        self.assertEqual(offenders, [],
                         f"the explanation keys are listed in QML as well as in "
                         f"app/ui/bridge.py, so the two can disagree: {offenders}")

    def test_both_surfaces_read_the_bridge_list(self):
        from pathlib import Path as _Path

        drawer = (UI_DIR / "PresenterDrawer.qml").read_text(encoding="utf-8")
        bridge_qml = (UI_DIR / "Bridge.qml").read_text(encoding="utf-8")
        code = "\n".join(line for line in drawer.splitlines()
                         if not line.strip().startswith("//"))
        self.assertIn("model: bridge.explanation_keys", code,
                      "the drawer must build its buttons from the bridge's own key list")
        # And `Bridge.qml`, which the footer reads through, must derive the same list
        # rather than restate it.
        self.assertIn("bridge.explanation_keys", bridge_qml,
                      "Bridge.qml must derive `explanationKeys` from the bridge")
        self.assertNotIn('"yolo"', bridge_qml,
                         "Bridge.qml still carries a literal key list")

    def test_the_drawers_item_names_are_not_the_footers(self):
        # **Surface-specific names, deliberately.** A shared name is how M1's dead panel
        # hid: the item existed, under a name that read like the one that worked. The drawer
        # and the footer are different surfaces and must be distinguishable in the scene.
        drawer = (UI_DIR / "PresenterDrawer.qml").read_text(encoding="utf-8")
        self.assertIn('"drawerExplainMouse_"', drawer,
                      "the drawer's buttons must be named for the drawer")
        self.assertNotIn('"explainButtonMouse_"', drawer,
                         "the drawer reuses the footer's item name, so a test (or a "
                         "presenter's expectation) cannot tell the two surfaces apart")


#: The keys `Bridge.qml`'s `explanationKeys` lists, in order. Pinned here so a key that
#: silently leaves the screen fails this test rather than shrinking the check.
KEY_ORDER = ("yolo", "tracking", "zone", "cpu", "gpu", "utilisation", "metrics")


class PresenterDrawerOnTheScreen(unittest.TestCase):
    """**The drawer moves nothing, and it says the two things it has to say.**

    Every assertion about position is a comparison of the **live item's own measured
    extent** before and after the drawer opens, item by item. "Opening it did not throw" is
    a different and much weaker claim, and it is the one that would let a drawer reflow the
    hero while the screen still rendered.

    The items compared are named in `_DRAWER_STABLE_ITEMS` and include the hero, the three
    pictures, the three per-camera badges, the staleness line, the three cards and the event
    counts - i.e. every part of the picture the brief names, not a sample of it.
    """

    @classmethod
    def setUpClass(cls):
        cls.report = _drawer_attract_report()

    def test_opening_the_drawer_moves_nothing(self):
        drawer = self.report["drawer"]
        base = drawer["base_extents"]
        opened = drawer["open_extents"]
        moved = []
        for name in _DRAWER_STABLE_ITEMS:
            self.assertIn(name, base, f"{name} must be on screen for this to mean anything")
            self.assertIn(name, opened,
                          f"{name} disappeared when the drawer opened - the drawer must not "
                          f"replace any part of the live layout")
            if base[name] != opened[name]:
                moved.append((name, base[name], opened[name]))
        self.assertEqual(moved, [],
                         f"opening the presenter drawer moved {len(moved)} live item(s). A "
                         f"presenter opening the drawer mid-demo must not be able to break "
                         f"the picture: {moved}")

    def test_closing_the_drawer_puts_every_item_back_where_it_was(self):
        # **The other half of "moves nothing".** Opening it and closing it must leave the
        # live layout identical to the layout that existed before it was ever opened; a
        # drawer that moves something only when it closes is still a drawer that moved it.
        drawer = self.report["drawer"]
        base = drawer["base_extents"]
        closed = drawer["closed_extents"]
        moved = [(name, base[name], closed.get(name)) for name in _DRAWER_STABLE_ITEMS
                 if closed.get(name) != base[name]]
        self.assertEqual(moved, [],
                         f"{len(moved)} live item(s) did not return to their position after "
                         f"the drawer closed: {moved}")
        self.assertNotIn("presenterDrawer", closed,
                         "the drawer is still in the scene graph after closing, so 'closed' "
                         "is not a state the next open can be compared against")

    def test_the_drawer_really_opened_and_its_content_is_on_screen(self):
        # **Before asserting the content, assert the drawer is there.** An item that is not
        # on screen reads `""`, and every "is it not empty" assertion on `""` passes - the
        # exact vacuity this project keeps finding. Presence in the **scene graph** is the
        # check, not the `Loader`'s status: status cannot be read from PySide6 (measured -
        # `Can't find converter for 'QQuickLoader::Status'`), so a status-based guard would
        # have been a guard that never ran.
        drawer = self.report["drawer"]
        self.assertFalse(drawer["present_before"],
                         "the drawer was in the scene graph before it was opened")
        self.assertTrue(drawer["present_when_open"],
                         "opening the drawer put nothing in the scene graph - PresenterDrawer.qml "
                         "failed to load, and an empty drawer is the M1 dead-affordance shape")
        strings = self.report["drawer"]["strings"]
        for name in ("drawerTitle", "drawerScopeNote",
                     "drawerScopeReduction", "drawerExplanationText",
                     "drawerCameraHeading_A", "drawerCameraHeading_B",
                     "drawerCameraHeading_C",
                     # **M4: the six rows that left the always-visible screen**, one item
                     # per camera each, so "moved into the drawer" is checkable per camera
                     # and per figure.
                     "streamBacklogText_A", "streamBacklogText_B", "streamBacklogText_C",
                     "streamRateText_A", "streamRateText_B", "streamRateText_C",
                     "streamWindowText_A", "streamWindowText_B", "streamWindowText_C",
                     "streamActivityText_A", "streamActivityText_B", "streamActivityText_C",
                     "cardText_system_cpu_percent",
                     # and the once-only ones
                     "eventCountRow", "drawerPipelineText", "drawerStatusText",
                     "stalenessText"):
            self.assertIn(name, strings, f"{name} is not on screen: {sorted(strings)}")
            self.assertTrue(strings[name].strip(), f"{name} is on screen but empty")
        # The drawer's own rectangle has no `text`, so it is in `extents` and not in
        # `strings`; asserting it by name in `extents` is the presence check that applies.
        self.assertIn("presenterDrawer", drawer["open_extents"])
        # **Nothing it draws is clipped away.** The drawer clips, so a button or a line that
        # spills past its edge is reported as overflow and is only half on screen - measured,
        # the seven buttons fill 487 px of the 566 px inner width and the report is empty.
        # **M4: the drawer is fuller than it was** (it holds the six per-camera rows that left
        # the live screen), so this is the assertion that its 1032 px inner height is enough.
        self.assertEqual(drawer["overflow"], [],
                         f"the drawer's own content overflows its clipped edge, so part of it "
                         f"is not on screen: {drawer['overflow']}")
        # And its own affordances are there, one per key. **M4: there is no second surface to
        # be confused with** - the footer's `?` row is gone, and the header button opens this
        # drawer (asserted in `ExplanationEntryPoints`), so the drawer's buttons are the only
        # ones and they are all named.
        for key in KEY_ORDER:
            self.assertIn(f"drawerExplainMouse_{key}", drawer["open_extents"],
                          f"the drawer offers no working button for {key!r}: "
                          f"{sorted(n for n in drawer['open_extents'] if 'Explain' in n)}")
            self.assertNotIn(f"explainButtonMouse_{key}", drawer["open_extents"],
                             f"the footer still offers a second {key!r} button - one "
                             f"explanation surface is the M4 decision")
        self.assertFalse(drawer["pinned_property_exists"],
                         "the window still has a `pinnedExplanationKey`, so there is a "
                         "second explanation surface for a click to land in")

    def test_the_drawer_does_not_cover_the_picture(self):
        """**"Moves nothing" is not the same as "obscures nothing".**

        An overlay that reflows nothing can still be drawn over the hero. The drawer's left
        edge must be at or right of the hero column's right edge, recomputed from the
        measured extents rather than from the theme token's intent.
        """
        drawer = self.report["drawer"]
        hero_x, _, hero_w, _ = drawer["open_extents"]["heroTile"]
        thumb_x, _, thumb_w, _ = drawer["open_extents"]["thumb_B"]
        # The hero and the thumbnails share the video column; its right edge is the larger
        # of the two. `heroTile` and `thumb_B` are side by side, so take the far one.
        column_right = max(hero_x + hero_w, thumb_x + thumb_w)
        drawer_x, _, drawer_w, _ = drawer["open_extents"]["presenterDrawer"]
        self.assertGreaterEqual(drawer_x, column_right,
                                f"the drawer starts at x={drawer_x} and the video column ends "
                                f"at x={column_right}: a long explanation would cover the "
                                f"picture")
        self.assertGreater(drawer_w, 0.0, "the drawer has no width")

    def test_opening_the_drawer_shrinks_no_font(self):
        # **The live screen's census, before and after the drawer exists.** Only the three
        # ladder sizes may be present, and the holders of the two non-detail sizes are the
        # same items the canvas gate names: three camera names and three age headlines. The
        # authoritative count against the committed capture is
        # `test_capture_ui_canvas.the_font_census_is_the_one_on_record`; this one proves the
        # drawer moved none of it.
        from app import theme

        drawer = self.report["drawer"]
        base = {int(size): count for size, count in drawer["base_census"].items()}
        self.assertEqual(sorted(base), [theme.TYPE_BOOTH_DETAIL, theme.TYPE_BOOTH_LABEL,
                                        theme.TYPE_BOOTH_NUMERAL],
                         f"a font size outside the ladder appeared on the live screen: {base}")
        self.assertEqual(base[theme.TYPE_BOOTH_LABEL], 3,
                         f"the only {theme.TYPE_BOOTH_LABEL} px items are the camera names: "
                         f"{base}")
        self.assertEqual(base[theme.TYPE_BOOTH_NUMERAL], 3,
                         f"the only {theme.TYPE_BOOTH_NUMERAL} px items are the age "
                         f"headlines: {base}")
        self.assertEqual(drawer["closed_census"], drawer["base_census"],
                         "the census after closing the drawer differs from before opening it")

    def test_the_scope_note_is_the_packs_own_words_verbatim(self):
        # **Compared with the pack itself, not with `bridge.scope_note`.** The bridge's
        # property and the screen's string could agree while both were a paraphrase; only a
        # comparison against `app/engine/retail_config.py` says the screen quotes the pack.
        from app.engine.retail_config import retail_sanity_manifest

        self.assertEqual(self.report["drawer"]["strings"]["drawerScopeNote"],
                         retail_sanity_manifest().scope_note,
                         "the drawer must quote app/engine/retail_config.py's scope_note, "
                         "not a paraphrase of it")

    def test_the_drawer_states_what_the_delivered_event_is(self):
        # The pack's note says what the footage is **not**; the reduction sentence next to
        # it has to say what the delivered event **is**, and the two have to be on the same
        # surface, or a presenter reads only the half that is on screen.
        reduction = self.report["drawer"]["strings"]["drawerScopeReduction"]
        self.assertIn("people-in-zone presence", reduction, reduction)
        self.assertIn("2 events per clip", reduction, reduction)

    def test_the_drawer_states_what_the_delivered_event_is_not(self):
        reduction = self.report["drawer"]["strings"]["drawerScopeReduction"]
        self.assertIn("people-in-zone presence", reduction, reduction)
        self.assertIn("NOT a people count", reduction, reduction)
        self.assertIn("NOT a dwell measurement", reduction, reduction)
        self.assertIn("2 events per clip", reduction, reduction)

    def test_the_drawer_states_that_gpu_and_npu_utilisation_is_unavailable_and_why(self):
        # **M4: asserted through the button that states it.** The standalone paragraph was
        # removed from the drawer to make room for the six per-camera rows that left the
        # live screen, so the claim is now checked the way a presenter gets it: click
        # `utilisation` and read what the drawer shows. That is stricter than reading a
        # paragraph that was always on screen, because it drives the affordance too.
        shown = self.report["drawer"]["explanation_texts"]["utilisation"]["shown"]
        self.assertIn("unavailable", shown.lower(), shown)
        self.assertIn("GPU", shown, shown)
        self.assertIn("NPU", shown, shown)
        self.assertIn("not attributable", shown,
                      f"the reason a machine-wide peak is not used must be on the drawer: {shown}")

    def test_every_drawer_button_shows_its_own_explanation(self):
        """**Every one, not merely one.** The brief says the six affordances work; the
        drawer offers seven, and a button that is present and dead is what M1 shipped.
        """
        drawer = self.report["drawer"]
        expected = drawer["expected_explanations"]
        for key in KEY_ORDER:
            result = drawer["explanation_texts"].get(key)
            self.assertIsNotNone(result, f"the drawer's {key!r} button was never clicked")
            self.assertTrue(result["shown"].strip(),
                            f"clicking the drawer's {key!r} button produced no explanation - "
                            f"the M1 dead-affordance shape")
            self.assertEqual(result["shown"], expected[key],
                             f"the drawer's {key!r} button showed another key's explanation")

    def test_the_drawers_buttons_land_in_the_only_explanation_surface(self):
        # **M4 removed the second surface rather than distinguishing two.** The footer's
        # pinned panel is gone (`pinnedExplanationKey` does not exist), so there is nowhere
        # for a click to land except this drawer - and the click must leave the drawer open
        # and showing the text, which is the claim M1's dead affordance failed.
        drawer = self.report["drawer"]
        self.assertFalse(drawer["pinned_property_exists"],
                         "the window still has a `pinnedExplanationKey`: a second explanation "
                         "surface is a second place for a click to land in")
        for key, result in drawer["explanation_texts"].items():
            self.assertTrue(result["drawer_open"],
                            f"the drawer's {key!r} button closed the drawer it lives in")

    def test_the_drawer_is_keyboard_reachable_and_escape_closes_it(self):
        drawer = self.report["drawer"]
        self.assertTrue(drawer["open_shortcut_found"], "no keyboard shortcut opens the drawer")
        self.assertTrue(drawer["open_after_f1"],
                        "activating the drawer shortcut did not open the drawer")
        self.assertEqual(len(drawer["items_after_f1"]), len(KEY_ORDER),
                         f"after F1 the drawer's buttons are {drawer['items_after_f1']} - the "
                         f"drawer is open but empty")
        self.assertTrue(drawer["escape_found"], "no Escape shortcut is on the window")
        self.assertTrue(drawer["closed_after_escape"],
                        "Escape must close the presenter drawer")
        self.assertEqual(drawer["items_after_escape"], [],
                         f"Escape closed the window but left the drawer's items in the scene: "
                         f"{drawer['items_after_escape']}")


class AttractModeOnTheScreen(unittest.TestCase):
    """**Attract mode shows no number it cannot date.**

    The render seeds a full live state (`system_cpu_percent` 58.0, three cameras with
    11.3 s frame ages) and then enters attract mode. The assertions are: the live screen is
    not visible, the overlay is and covers the window, every headline is the recorded one,
    every headline carries the date it was recorded and the committed file it came from, no
    live figure appears, and leaving it restores every string and every position.
    """

    @classmethod
    def setUpClass(cls):
        cls.report = _drawer_attract_report()

    def test_attract_hides_the_live_screen_and_shows_the_overlay(self):
        attract = self.report["attract"]
        self.assertFalse(attract["overlay_present_before"],
                         "the attract overlay was in the scene graph before attract mode")
        self.assertTrue(attract["overlay_present"],
                        "attract mode put no overlay in the scene graph - it is a blank screen, "
                        "and a blank screen also shows no live number")
        self.assertFalse(attract["live_screen_visible"],
                         "the live screen must not be visible in attract mode: it is the only "
                         "way to be sure no live number is on screen")
        self.assertTrue(attract["overlay_visible"],
                        "the attract overlay is not visible, so attract mode shows nothing")
        # **It covers the window.** A partially painted overlay would leave a live card
        # showing at the edge while every assertion above still passed.
        width, height = self.report["window"]
        x, y, w, h = attract["extents"]["attractOverlay"]
        self.assertEqual((x, y, w, h), (0.0, 0.0, width, height),
                         f"the attract overlay is {x},{y},{w}x{h} in a {width}x{height} window")

    def test_the_title_and_the_recorded_label_are_on_screen(self):
        strings = self.report["attract"]["strings"]
        self.assertEqual(strings.get("attractTitle"), "Intel Edge AI Roadshow", strings)
        self.assertEqual(strings.get("attractRecordedLabel"),
                         "Recorded demonstration \u2014 metrics are not live", strings)

    def test_every_headline_is_a_recorded_measurement_with_its_date_and_source(self):
        from app.ui.bridge import ATTRACT_HEADLINES

        strings = self.report["attract"]["strings"]
        self.assertTrue(ATTRACT_HEADLINES)
        for index, headline in enumerate(ATTRACT_HEADLINES):
            text_name = f"attractHeadline_{index}"
            recorded_name = f"attractHeadlineRecorded_{index}"
            self.assertIn(text_name, strings, f"{text_name} is not on screen")
            self.assertIn(recorded_name, strings, f"{recorded_name} is not on screen")
            self.assertEqual(strings[text_name], headline["text"])
            # **The date and the source, on the same slide as the number.** A figure whose
            # date is not shown is a number nobody can date, which is the failure this mode
            # exists to prevent.
            recorded = strings[recorded_name]
            self.assertIn(headline["date"], recorded,
                          f"the date {headline['date']} must be shown with the figure: {recorded}")
            self.assertIn(headline["source"], recorded,
                          f"the committed source must be shown with the figure: {recorded}")

    def test_the_idle_screen_is_on_the_same_ladder_as_the_live_one(self):
        """**A booth with nobody at it is exactly when a passer-by reads the screen.**

        M3 Task 4's overlay used the 40 / 26 / 18 px ramp, so the idle screen - the one a
        visitor sees when no presenter is there - was drawn at **9.4'** of cap height at 3 m,
        below the acuity floor. M4 rebuilt it on the booth ladder, and this asserts the
        census contains no size below it. (The census is visible, non-empty named items; the
        overlay is the only thing drawn in attract mode, so it is the whole of that read.)
        """
        from app import theme

        census = {int(size): count for size, count in self.report["attract"]["census"].items()}
        self.assertTrue(census, "attract mode drew no text at all")
        below = {size: count for size, count in census.items()
                 if size < theme.TYPE_BOOTH_DETAIL}
        self.assertEqual(below, {},
                         f"the idle screen draws type below the booth ladder ({below}); a "
                         f"passer-by reading an unattended booth reads that type")
        self.assertIn(theme.TYPE_BOOTH_NUMERAL, census,
                      f"the attract title must be the headline size: {census}")

    def test_the_attract_overlay_overflows_nothing_of_its_own(self):
        """**The overlay's own text fits the window.**

        The read also names overflow inside the **hidden** live screen - four captions in
        the thumbnails, whose geometry is measured while their parent is not visible. Those
        are not drawn claims: `liveScreen.visible` is asserted `False` above, so nothing in
        that subtree paints, and the round-trip test proves the geometry is intact once the
        screen is visible again. What must be true is that nothing **on the overlay** spills
        past the window, because an attract slide with clipped text is a claim cut in half.
        """
        attract = self.report["attract"]
        own = [name for name in attract["overflow"] if name.startswith("attract")]
        self.assertEqual(own, [],
                         f"the attract overlay's own text overflows the window: {own}")
        foreign = [name for name in attract["overflow"] if not name.startswith("attract")]
        self.assertNotIn("attractOverlay", foreign)
        # Recorded, not asserted away: the live subtree is where they are, and it is hidden.
        self.assertFalse(attract["live_screen_visible"])

    def test_no_live_figure_appears_in_attract_mode(self):
        # The seeded live value is 58.0% (and 14.90 fps). Neither may appear anywhere in
        # the attract overlay: attract mode shows recorded measurements only.
        attract = self.report["attract"]
        self.assertEqual(attract["live_metric_text"], "58.0%", attract)
        overlay_text = " ".join(
            value for name, value in attract["strings"].items()
            if name.startswith("attract"))
        self.assertNotIn("58.0%", overlay_text,
                         f"a live figure reached the attract overlay: {overlay_text}")
        self.assertNotIn("14.90 fps", overlay_text, overlay_text)

    def test_leaving_attract_restores_the_live_screen_exactly(self):
        # **Item by item, not "nothing threw".** The live screen's visibility, every string
        # and **every measured position** must be exactly what they were before attract mode
        # was entered - a live screen that comes back in a different place has not been
        # restored, and only positions can say so.
        attract = self.report["attract"]
        self.assertTrue(attract["live_screen_visible_after"],
                        "the live screen must be visible again after attract mode exits")
        self.assertFalse(attract["overlay_present_after"],
                         "the attract overlay is still in the scene graph after attract mode "
                         "exited")
        before = attract["before_strings"]
        after = attract["after_exit_strings"]
        changed = [(name, value, after.get(name))
                   for name, value in before.items() if after.get(name) != value]
        self.assertEqual(changed, [],
                         f"{len(changed)} string(s) changed across an attract-mode round trip: "
                         f"attract mode must restore the state it interrupted, byte for byte: "
                         f"{changed[:8]}")
        moved = [(name, rect, attract["after_exit_extents"].get(name))
                 for name, rect in attract["before_extents"].items()
                 if attract["after_exit_extents"].get(name) != rect]
        self.assertEqual(moved, [],
                         f"{len(moved)} item(s) moved across an attract-mode round trip: "
                         f"{moved[:8]}")
        self.assertEqual(attract["after_exit_census"], attract["before_census"],
                         "the font census changed across an attract-mode round trip")


class TestSuiteHygiene(unittest.TestCase):
    """Properties of the test suite itself, which nothing else can assert."""

    def test_no_test_method_is_defined_twice_in_one_class(self):
        # `tests/test_ui_contract.py` had `test_the_coasted_track_box_is_not_drawn`
        # defined **twice** in `OffscreenRender`, with different bodies, and Python kept
        # the last one silently. The review of M1 found it. A shadowed test is not a test
        # anybody ran, and the copy that was shadowed sampled a different region - so the
        # suite reported a pass for a test whose text no longer described what it checked.
        #
        # Read from the **source**, not from the imported class: `unittest.TestCase`
        # collects a class's attributes, and the shadowed definition is already gone from
        # those by the time anything could look.
        import ast

        offenders = []
        for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
            # `utf-8-sig`, because several test files in this repository carry a byte-order
            # mark and `ast.parse` rejects one outright:
            # `SyntaxError: invalid non-printable character U+FEFF`. A scanner that
            # cannot open half the suite is a scanner measuring nothing.
            text = path.read_text(encoding="utf-8-sig")
            try:
                tree = ast.parse(text)
            except SyntaxError as exc:
                self.fail(f"{path.name} does not parse: {exc}")
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                seen: dict[str, int] = {}
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if child.name.startswith("test"):
                            seen[child.name] = seen.get(child.name, 0) + 1
                for name, count in seen.items():
                    if count > 1:
                        offenders.append(
                            f"{path.name}:{node.name}.{name} defined {count} times")
        self.assertEqual(offenders, [],
                         f"a test method defined more than once is silently shadowed: "
                         f"{offenders}")


if __name__ == "__main__":  # pragma: no cover - convenience for a human at a booth
    unittest.main()