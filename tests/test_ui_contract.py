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

import os
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


def _render_png(tmpdir: str) -> str:
    """Render `app/ui/Main.qml` offscreen at 1920x1080 in a child process.

    Returns the PNG path. Raises on a non-zero exit so a QML syntax error fails loudly
    instead of leaving a stale or absent image for the assertions to read.
    """
    script = tmpdir.replace("\\", "/") + "/render.py"
    output = tmpdir.replace("\\", "/") + "/main.png"
    Path(script).write_text(_render_script(), encoding="utf-8")
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
    return output


def _render_script() -> str:
    """The child-process program, built with the repo's real paths substituted in.

    Written as one piece with `%`-style placeholders rather than a separate `.py` file in
    `app/ui/`, because a `.py` under `app/ui/` would be imported by nothing and would be a
    second copy of the UI wiring that could drift from `app.main`'s.
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
            .replace("@UI@", repr(str(UI_DIR))))


_RENDER_TEMPLATE = '''
"""Render app/ui/Main.qml offscreen and save a PNG. Run in its own process."""
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
bridge.set_route(requested="NPU", confirmed=("CPU",), state="active")
bridge.set_metrics(system_cpu_percent=18.4, app_cpu_percent=12.0, ai_fps=14.9,
                   frame_age_p50_ms=63.0, frame_age_p95_ms=88.0)
bridge.set_status("running the retail_checkout clip")

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
        # and none may set `utilisationAvailable: true` for an accelerator.
        qml = _qml_sources()
        self.assertNotIn("gpu_utilisation_percent: ", qml.replace('"', ""))
        for name in ("GPU", "NPU"):
            marker = f'deviceId: "{name}"'
            self.assertIn(marker, qml)
            tail = qml.split(marker, 1)[1][:400]
            self.assertIn("utilisationAvailable: bridgeUi.", tail,
                          f"the {name} card must take its availability from the bridge's "
                          f"measurement, never a literal")


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

    def test_the_coasted_track_box_is_not_drawn(self):
        # The end-to-end version of `OverlayContract::test_a_coasted_track_produces_no_box`.
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
        mutated = source.replace(
            "            if not getattr(track, \"is_current\", False):\n"
            "                skipped += 1\n"
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


if __name__ == "__main__":  # pragma: no cover - convenience for a human at a booth
    unittest.main()