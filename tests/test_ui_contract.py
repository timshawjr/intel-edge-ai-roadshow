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


def _render(tmpdir: str, seed: str = _SINGLE_STREAM_SEED) -> tuple[str, dict]:
    """`(png_path, {objectName: [text, visible]})` for one offscreen render.

    The strings come out of the child's **scene graph**, read from the named items the QML
    declares - not from the bridge's Python properties, and not from the PNG. A bridge can
    hold a correct value while the screen is frozen at its first evaluation, and that is the
    defect class this repository has now found three times (the `constant=True` device
    badge, the `display_text()` card bindings, the `window_text()` line).

    `seed` chooses what state exists before the scene loads; see `_SINGLE_STREAM_SEED` and
    `_ALLCPU_SEED`.
    """
    script = tmpdir.replace("\\", "/") + "/render.py"
    output = tmpdir.replace("\\", "/") + "/main.png"
    Path(script).write_text(_render_script(seed), encoding="utf-8")
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
    if not strings:
        raise AssertionError(
            "the render published no scene strings, so every string assertion would be "
            "reading nothing - which is exactly how the M1 card-text assertion proved "
            "two of three keys.\nstdout: " + completed.stdout[-3000:])
    return output, strings


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


def _render_script(seed: str = _SINGLE_STREAM_SEED) -> str:
    """The child-process program, built with the repo's real paths substituted in.

    Written as one piece with `%`-style placeholders rather than a separate `.py` file in
    `app/ui/`, because a `.py` under `app/ui/` would be imported by nothing and would be a
    second copy of the UI wiring that could drift from `app.main`'s.

    `seed` is the block that publishes route and metric state **before** the scene loads.
    It is a parameter rather than a second template so that the one-stream M1 render and
    the three-stream M2 render exercise the *same* QML, the same font registration, the same
    grab and the same string dump - a second template would be a second thing that can
    drift, which is how M0 and M1 each produced a vacuous assertion.
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

    Only the four attributes `UiBridge` reads are provided: identity, extents and image.
    The identity is the real four-tuple shape, because that is what `accept_result`
    compares.
    """

    identity = ("selftest", "A", 0, 0)
    width = 1920
    height = 1080


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
    print("SCENESTRINGS " + json.dumps(NAMES, sort_keys=True))


shutil.rmtree(work, ignore_errors=True)
shutil.copytree(@UI@, work)

engine = QQmlApplicationEngine()
engine.addImageProvider("frames", FrameImageProvider(bridge))
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
        cls.png, cls.strings = _render(tmp, _ALLCPU_SEED)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def text(self, name: str) -> str:
        self.assertIn(name, self.strings,
                      f"{name} is not on screen; the items that are: "
                      f"{sorted(self.strings)}")
        return self.strings[name][0]

    def visible(self, name: str) -> bool:
        self.assertIn(name, self.strings, f"{name} is not on screen at all")
        return self.strings[name][1]

    def test_every_item_this_class_reads_is_on_screen(self):
        # **First, and on purpose.** A missing item yields `""`, and every "is this not
        # empty / not zero" assertion on `""` passes. Naming the item first is the only
        # thing that stops that.
        for sid in ("A", "B", "C"):
            for prefix in ("streamAgeText_", "streamStaleText_", "streamWindowText_",
                           "streamBadgeText_", "streamBacklogText_", "streamRateText_",
                           "streamReasonText_", "streamActivityText_"):
                self.assertIn(prefix + sid, self.strings,
                              f"camera {sid}'s {prefix} item is missing, so any "
                              f"assertion about its contents would be asserting about "
                              f"nothing: {sorted(self.strings)}")
            self.assertIn("card_" + sid, self.strings,
                          f"there is no card item for camera {sid}")
        self.assertEqual(len([n for n in self.strings if n.startswith("card_")
                              and n != "cardText_system_cpu_percent"]), 3,
                         f"exactly three cameras, three cards: {sorted(self.strings)}")

    def test_the_eleven_second_old_picture_is_legible_as_stale_in_words(self):
        # The ruling. Each camera states its own age, **in words as well as a number**, and
        # the age in the sentence is the age in the numeral.
        for sid, p50, p95, backlog in (("A", "10.8 s", "11.6 s", "170 frames"),
                                       ("B", "11.1 s", "11.9 s", "179 frames"),
                                       ("C", "11.3 s", "12.3 s", "187 frames")):
            self.assertEqual(self.text(f"streamAgeText_{sid}"), p50,
                             f"camera {sid}'s measured p50 frame age is not on its card")
            stale = self.text(f"streamStaleText_{sid}")
            self.assertTrue(self.visible(f"streamStaleText_{sid}"),
                            f"camera {sid}'s staleness line is present but not visible, "
                            f"which is the frozen-binding shape")
            self.assertIn("STALE", stale,
                          f"camera {sid}: eleven seconds old must be a WORD, not only a "
                          f"number. It reads {stale!r}")
            self.assertIn(p50, stale,
                          f"camera {sid}: the sentence must carry the age, so the number "
                          f"and the word cannot disagree: {stale!r}")
            self.assertIn("2 s", stale,
                          f"camera {sid}: the sentence must carry the limit it exceeds: "
                          f"{stale!r}")
            # The window, the backlog and the rate. All three measured, all three on the
            # card that camera is on.
            self.assertEqual(self.text(f"streamWindowText_{sid}"),
                             "window: 256 analysed frames",
                             f"camera {sid}: the percentiles are of 256 frames and the "
                             f"card must say so")
            self.assertEqual(self.text(f"streamBacklogText_{sid}"),
                             f"backlog {backlog}",
                             f"camera {sid}'s backlog is not on its own card")
            self.assertIn("analysis", self.text(f"streamRateText_{sid}"))
            self.assertIn(p95, self.text(f"streamActivityText_{sid}"),
                          f"camera {sid}'s p95 is not on its card")

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
        self.assertIn("no device is being claimed for it", reason, reason)
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
        # accelerators' "Utilization unavailable".
        self.assertEqual(self.text("cardText_system_cpu_percent"), "no CPU sample yet")
        self.assertNotIn("cardText_gpu_utilisation_percent", self.strings,
                         "no camera is on the GPU, so no card may claim a GPU metric")
        self.assertNotIn("cardText_", self.strings,
                         f"a half-named cardText_ item exists, which is the exact shape "
                         f"the M1 test was rewritten to stop reaching for: "
                         f"{sorted(self.strings)}")

    def test_a_camera_with_nothing_measured_reads_unavailable_and_never_zero(self):
        # A **second render**, seeded with three declared streams and nothing else: no
        # route, no metrics, no frames. This is the screen's first frame.
        tmp = tempfile.mkdtemp(prefix="roadshow-cold-")
        try:
            _png, strings = _render(tmp, 'bridge.set_streams(["A", "B", "C"])\n')
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
            for prefix in ("streamWindowText_", "streamBacklogText_", "streamRateText_"):
                text, _vis = strings[f"{prefix}{sid}"]
                self.assertIn("unavailable", text,
                              f"{prefix}{sid} on a cold screen reads {text!r}; "
                              f"'0 frames of backlog' looks like a healthy empty queue")
            self.assertEqual(strings[f"streamBadgeText_{sid}"][0],
                             "no device is running",
                             "nothing has confirmed anything, so no camera claims a "
                             "device")
            self.assertEqual(strings[f"streamReasonText_{sid}"][1], False,
                             "a pending camera has nothing to explain")

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
        # **Obligation: the CPU card must not wear the accelerator's wording.** M1 checked
        # this by asserting `unavailable_notice["GPU"] == "Utilization unavailable"` on a
        # screen that had one camera on the CPU - i.e. on cards that were not there. Now
        # there really is one camera per device, so each wording is read off the card that
        # carries it.
        tmp = tempfile.mkdtemp(prefix="roadshow-devices-")
        try:
            _png, strings = _render(tmp, _THREE_DEVICE_SEED)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        expected = {"system_cpu_percent": "no CPU sample yet",
                    "gpu_utilisation_percent": "Utilization unavailable",
                    "npu_utilisation_percent": "Utilization unavailable"}
        for metric, wording in expected.items():
            name = "cardText_" + metric
            self.assertIn(name, strings,
                          f"there must be a card figure named for the {metric}: "
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
        # **M2 Task 5 retarget.** The device-to-metric mapping moved from three hand-written
        # cards in `Main.qml` into `ComputeCard.qml`, because the card is now per *stream*
        # and a stream's device is only known at run time - three fixed cards would each
        # have to claim a device the stream might not be on. The guard's teeth are the two
        # things it always cared about: **each device maps to its own metric name**, and
        # **availability is read from the bridge, never asserted**.
        card = (UI_DIR / "ComputeCard.qml").read_text(encoding="utf-8")
        code = "\n".join(line for line in card.splitlines()
                         if not line.strip().startswith("//"))
        for device, metric in (("CPU", "system_cpu_percent"),
                               ("GPU", "gpu_utilisation_percent"),
                               ("NPU", "npu_utilisation_percent")):
            self.assertIn(f'deviceId === "{device}" ? "{metric}"', code,
                          f"the {device} line must name the {device}'s own metric, not a "
                          f"shared one")
        self.assertIn("bridge.metric_text[metricName]", code,
                      "the utilisation figure must come from the bridge's notified map")
        self.assertIn("bridge.unavailable_text", code,
                      "and an absent figure must read as the bridge's word, not a 0")
        # A card may not dim itself on utilisation any more - see ComputeCard.qml's header:
        # dimming the card that says the NPU stream is eleven seconds stale, because NPU
        # utilisation cannot be measured, would dim the one warning that matters.
        self.assertNotIn("opacity: card.utilisationAvailable", code)

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
engine.addImageProvider("frames", module.FrameImageProvider(bridge))
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
        card = (UI_DIR / "ComputeCard.qml").read_text(encoding="utf-8")
        code = "\n".join(line for line in card.splitlines()
                         if not line.strip().startswith("//"))
        self.assertNotRegex(
            code, r"\bwindow_text\s*\(",
            "the card calls the window_text Slot, so the window line is evaluated once at "
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
        # for them; the CPU card must say its own thing, because `psutil.cpu_percent` works
        # here and "no sample yet" is the truth. One shared literal claimed a failure to
        # read a counter that is not failing.
        card = (UI_DIR / "ComputeCard.qml").read_text(encoding="utf-8")
        code = "\n".join(line for line in card.splitlines()
                         if not line.strip().startswith("//"))
        self.assertIn("bridge.unavailable_notice[deviceId]", code,
                      "the unavailable label must be per-card, keyed by deviceId")
        self.assertNotIn("text: bridge.utilisation_unavailable_text", code,
                         "one shared literal on every card is what R1 is about")

    def test_the_event_counts_are_gated_on_frames_analysed_not_on_the_route(self):
        # R3. A confirmed placement arrives from the worker's route *before* any frame has
        # been analysed, so gating on it printed three hard zeros for an event machine
        # that had not run.
        main_qml = (UI_DIR / "Main.qml").read_text(encoding="utf-8")
        code = "\n".join(line for line in main_qml.splitlines()
                         if not line.strip().startswith("//"))
        self.assertIn("bridge.frames_analysed > 0", code,
                      "the event counts must be gated on frames analysed")
        self.assertNotIn("routeConfirmed", code,
                         "gating on the route leaves a window where a measured 0 is "
                         "printed for a machine that has analysed nothing")


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