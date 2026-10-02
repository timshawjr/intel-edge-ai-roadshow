"""`app.main`: the self-test, the release-mode gate, and the headless run summary.

Three things are verified here, and the third is the one that produces the milestone's
evidence.

## `--selftest`

Runs the whole path - preprocess, infer, decode, track, event, render-metadata - with
**no network and no display**, prints `PASS`, and exits 0. The test runs it as a
subprocess rather than in-process, because the only honest way to assert "exits 0" is to
run the thing and read its exit code, and because `--selftest` is the command
AGENTS.md's verification block names.

**`selftest --seconds 0` is a real run, not a stub.** The first version short-circuited
when nothing was provisioned, and the subprocess test could not tell the difference
between "checked everything" and "checked nothing and said PASS". The requirement is
narrower than it looks: `--selftest` must work on a machine with **no model and no clip**,
so a missing artefact is a *skip with a stated reason* and the exit code is still 0, but a
missing artefact is never silently counted as a pass. The summary says which stages ran
and which were skipped, and `test_the_selftest_reports_what_it_actually_ran` asserts that
distinction survives.

## Release mode refuses a simulated provider

`assert_no_simulation_in_release(release, providers)` raises `SystemExit` when a release
build has a simulated provider registered. The brief's version is kept and two tests are
added around it, because a function that exits on a list containing the word "simulated"
is trivially satisfiable and trivially bypassable:

- a **real** provider name that merely resembles one (`"psutil"`, `"pdh_gpu_engine"`) must
  pass - a rule that matched on any short name would shut down the real application; and
- the refusal must **name** the provider it refused, because "release mode refused to
  start" with no name sends somebody looking at the wrong subsystem.

The simulated-provider names are matched on an explicit set plus the `simulated`/`mock`/
`fake`/`stub` prefixes. **Prefix matching, not substring**: `"psutil"` contains no prefix
and is unaffected, and a provider called `"simulated_cpu_metrics"` is caught while a
provider called `"cpu_simulated"` - a name no real provider here uses - is not silently
assumed either way.

## The headless run

`--headless --seconds N --json PATH` runs the full pipeline against the provisioned clip
on the requested device and writes a summary: frames submitted, frames analysed, drops,
**per-device placement from the confirmed reading**, AI FPS, frame age p50/p95, system and
app CPU, and the business events with their `dedup_key`s.

Three properties of that summary are asserted here and each exists because its absence
would be a lie:

1. **A number is `null` when it was not measured.** Not `0`, not omitted. The summary is
   built from `MetricSample` records and an `UNAVAILABLE` sample contributes `None`.
2. **Placement is the confirmed reading**, not the requested device, and the summary says
   which one it carries: `"requested"` and `"confirmed"` are separate keys.
3. **Events carry their `dedup_key`s**, because a count without the keys cannot be checked
   against a replay.

The tests that need a provisioned clip and a real device **skip with a stated reason**
rather than passing silently - this repository's convention, and a skip is not a pass.
`test_the_headless_summary_shape_is_pinned` runs with a **synthetic** source and a
synthetic supervisor, so the summary's shape is verified everywhere and the real run is
only evidence about the numbers.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parent.parent

#: `app.main` must never import from `tools/`. Asserted at the source level in
#: `NoToolsImport`, because the boundary is a rule rather than a behaviour.
APP_DIR = REPO_ROOT / "app"


def _run(*args, timeout: int = 600) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "app.main", *args],
        capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT), env=env,
    )


def _provisioned() -> str | None:
    """Why this machine cannot run the real pipeline, or `None` if it can."""
    try:
        import openvino as ov
    except ImportError:
        return "openvino is not importable"
    from app.inference.pinned import REPO_ROOT as PIN_ROOT, verified_ir_path

    if "CPU" not in ov.Core().available_devices:
        return "no CPU plugin on this machine"
    if not (PIN_ROOT / "assets" / "media" / "retail-checkout.mp4").exists():
        return "the pinned clip is not provisioned"
    try:
        verified_ir_path()
    except Exception as exc:  # noqa: BLE001 - a skip reason, never a pass
        return f"the pinned IR is unusable: {exc}"
    return None


class SelfTestAndReleaseMode(unittest.TestCase):
    """The brief's two tests, kept verbatim."""

    def test_selftest_exits_zero(self):
        r = _run("--selftest")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PASS", r.stdout)

    def test_release_mode_refuses_a_simulated_provider(self):
        from app.main import assert_no_simulation_in_release
        with self.assertRaises(SystemExit):
            assert_no_simulation_in_release(release=True, providers=("simulated_cpu",))
        assert_no_simulation_in_release(release=False, providers=("simulated_cpu",))
        assert_no_simulation_in_release(release=True, providers=("psutil",))

    def test_the_selftest_reports_what_it_actually_ran(self):
        # A self-test that says PASS without saying what it ran is indistinguishable from
        # one that ran nothing. The summary carries a per-stage verdict and this asserts
        # the shape, whatever this particular machine could provision.
        from app.main import run_selftest

        summary = run_selftest()
        self.assertIn("stages", summary)
        self.assertTrue(summary["stages"], "the self-test reported no stages at all")
        for name, stage in summary["stages"].items():
            # Every stage must be one of the three recorded verdicts, and a `skipped` or
            # `failed` one must say why. The first version of this test also asserted the
            # stage's *name* appeared inside its own record, which is not a property
            # anything has to have; the name is the key, and the key is checked by the
            # iteration itself.
            self.assertIn(stage["verdict"], ("ran", "skipped", "failed"),
                          f"{name}: {stage}")
            if stage["verdict"] == "skipped":
                self.assertTrue(stage["reason"].strip(),
                                f"{name} was skipped without a stated reason")

    def test_the_selftest_constructs_no_gui_application(self):
        # "Needs no display" means **no `QGuiApplication` is ever constructed**, not that
        # Qt is never imported. The first version of this test asserted that
        # `PySide6.QtGui` was absent from `sys.modules`, and it failed for a correct
        # reason: `app.ui.bridge` imports `QImage` and `QColor` because the frame image
        # provider builds a `QImage`, so the render-metadata stage cannot avoid QtGui.
        # Asserting the absence of an import would have pushed towards not testing the
        # renderer at all.
        #
        # So the check is on the object that actually needs a display: after the
        # self-test, `QGuiApplication.instance()` is `None`. Run in a subprocess because
        # this suite's other modules construct one.
        code = (
            "import sys, json;"
            "from app.main import run_selftest;"
            "s = run_selftest();"
            "from PySide6.QtGui import QGuiApplication;"
            "print(json.dumps({"
            "'display': s.get('display'),"
            "'ok': s['ok'],"
            "'app': QGuiApplication.instance() is not None"
            "}))"
        )
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYTHONPATH"] = str(REPO_ROOT)
        completed = subprocess.run([sys.executable, "-c", code],
                                   capture_output=True, text=True, timeout=300,
                                   cwd=str(REPO_ROOT), env=env)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertTrue(report["ok"], completed.stdout)
        self.assertFalse(report["app"],
                         "the self-test constructed a QGuiApplication, so it needs a "
                         "display and cannot run headless")

    def test_the_selftest_never_loads_main_qml(self):
        # The render-metadata stage checks formatting and overlay construction, not a
        # window. Asserted by asking for no `QQuickWindow`: if the stage ever grew a
        # window this is what would notice.
        code = (
            "import sys, json;"
            "from app.main import run_selftest;"
            "s = run_selftest();"
            "print(json.dumps({'quick': 'PySide6.QtQuick' in sys.modules,"
            "'ok': s['ok']}))"
        )
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYTHONPATH"] = str(REPO_ROOT)
        completed = subprocess.run([sys.executable, "-c", code],
                                   capture_output=True, text=True, timeout=300,
                                   cwd=str(REPO_ROOT), env=env)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertTrue(report["ok"], completed.stdout)
        self.assertFalse(report["quick"],
                         "the self-test imported QtQuick, so it can load a window")

    def test_the_selftest_summary_records_that_no_display_was_opened(self):
        from app.main import run_selftest

        summary = run_selftest()
        self.assertIsNone(summary.get("display"),
                          "the self-test opened a display; it must run with none")

    def test_the_selftest_makes_no_network_call(self):
        # `socket.socket` is replaced for the duration. A network call inside the
        # self-test raises rather than being counted, which is the point: AGENTS.md rule 7
        # says nothing is fetched at runtime, and this is the check that says so.
        import socket

        from app.main import run_selftest

        original = socket.socket

        class Refusing(socket.socket):
            def __init__(self, *args, **kwargs):
                raise AssertionError(
                    "the self-test opened a socket. Nothing is fetched at runtime; "
                    "models, media, labels and fonts are provisioned once, up front.")

        socket.socket = Refusing
        try:
            summary = run_selftest()
        finally:
            socket.socket = original
        self.assertTrue(summary["stages"])


class ReleaseGate(unittest.TestCase):
    """The gate, tested around the brief's example rather than only at it."""

    def test_a_real_provider_is_not_mistaken_for_a_simulated_one(self):
        from app.main import assert_no_simulation_in_release

        for provider in ("psutil", "pdh_gpu_engine", "openvino", "cv2", "PdhQuery"):
            assert_no_simulation_in_release(release=True, providers=(provider,))

    def test_no_providers_at_all_is_not_a_refusal(self):
        from app.main import assert_no_simulation_in_release

        assert_no_simulation_in_release(release=True, providers=())

    def test_the_refusal_names_the_provider(self):
        from app.main import assert_no_simulation_in_release

        with self.assertRaises(SystemExit) as caught:
            assert_no_simulation_in_release(release=True,
                                            providers=("psutil", "simulated_npu"))
        self.assertIn("simulated_npu", str(caught.exception))

    def test_the_registered_providers_of_this_build_are_real(self):
        # Whatever the application actually registers must be safe in release mode. If
        # this fails, `app.main` is registering a provider it would then refuse.
        from app.main import REGISTERED_PROVIDERS, assert_no_simulation_in_release

        assert_no_simulation_in_release(release=True, providers=REGISTERED_PROVIDERS)


class HeadlessSummary(unittest.TestCase):
    """The summary's shape, with synthetic inputs, so it is pinned everywhere."""

    def _summary(self, **kwargs):
        from app.main import build_run_summary

        defaults = dict(
            device_requested="CPU",
            state="active",
            confirmed=("CPU",),
            frames_submitted=10,
            frames_analysed=9,
            drops=1,
            elapsed_s=2.0,
            frame_ages_ms=(10.0, 20.0, 30.0),
            events=(),
            samples=(),
            extra_errors=(),
        )
        defaults.update(kwargs)
        return build_run_summary(**defaults)

    def test_the_summary_reports_the_numbers_it_was_given(self):
        summary = self._summary()
        self.assertEqual(summary["frames_submitted"], 10)
        self.assertEqual(summary["frames_analysed"], 9)
        self.assertEqual(summary["drops"], 1)

    def test_the_summary_carries_requested_and_confirmed_separately(self):
        summary = self._summary(device_requested="NPU", confirmed=("CPU",))
        self.assertEqual(summary["device"]["requested"], "NPU")
        self.assertEqual(summary["device"]["confirmed"], ["CPU"])
        self.assertTrue(summary["device"]["fell_back"])

    def test_a_failed_route_confirms_nothing(self):
        summary = self._summary(device_requested="NPU", confirmed=(),
                                state="failed")
        self.assertEqual(summary["device"]["confirmed"], [])
        self.assertIsNone(summary["device"]["fell_back"],
                          "an unproven route has not fallen back; it has not been shown")

    def test_an_absent_measurement_is_null_not_zero(self):
        from app.contracts.metric_sample import MetricSample
        from app.contracts.device_status import Validity

        sample = MetricSample(
            name="system_cpu_percent", value=None, unit="percent", provider="psutil",
            device_scope=None, process_scope=("ui",), window_seconds=0.0,
            sampled_at=1.0, validity=Validity.UNAVAILABLE,
            unavailable_reason="no window elapsed",
        )
        summary = self._summary(samples=(sample,))
        self.assertIsNone(summary["metrics"]["system_cpu_percent"]["value"])
        # `Validity` is a lower-case `str` enum - `valid` / `stale` / `unavailable` - which
        # is the same enum `DeviceStatus` uses. The first version of this test asserted
        # `"UNAVAILABLE"` and failed on a case difference, which is worth recording
        # because it is exactly the "two spellings of one value" problem the enum was
        # introduced to prevent.
        self.assertEqual(summary["metrics"]["system_cpu_percent"]["validity"],
                         Validity.UNAVAILABLE.value)
        self.assertTrue(summary["metrics"]["system_cpu_percent"]["reason"])

    def test_a_measured_zero_is_a_number(self):
        from app.contracts.metric_sample import MetricSample, Validity

        sample = MetricSample(
            name="app_cpu_percent", value=0.0, unit="percent", provider="psutil",
            device_scope=None, process_scope=("ui",), window_seconds=0.25,
            sampled_at=1.0, validity=Validity.VALID,
        )
        summary = self._summary(samples=(sample,))
        self.assertEqual(summary["metrics"]["app_cpu_percent"]["value"], 0.0)

    def test_frame_age_percentiles_are_ordered_and_stated(self):
        summary = self._summary(frame_ages_ms=(10.0, 20.0, 30.0, 40.0, 50.0))
        ages = summary["frame_age_ms"]
        self.assertLessEqual(ages["p50"], ages["p95"])
        self.assertIn("method", ages)

    def test_no_frame_ages_means_null_not_zero(self):
        summary = self._summary(frame_ages_ms=())
        self.assertIsNone(summary["frame_age_ms"]["p50"])
        self.assertIsNone(summary["frame_age_ms"]["p95"])

    def test_ai_fps_is_null_when_nothing_was_analysed(self):
        summary = self._summary(frames_analysed=0, elapsed_s=1.0, frame_ages_ms=())
        self.assertIsNone(summary["ai_fps"])

    def test_events_carry_their_dedup_keys(self):
        from app.engine.events import BusinessEvent

        event = BusinessEvent(kind="zone_occupancy", class_name="person", track_id=7,
                              source_loop_id=0, at_frame_id=120, at_time=4.0,
                              dedup_key="0:7:person:presence:counter_band")
        summary = self._summary(events=(event,))
        self.assertEqual(summary["events"][0]["dedup_key"],
                         "0:7:person:presence:counter_band")
        self.assertEqual(summary["event_counts"], {"zone_occupancy": 1})

    def test_the_summary_is_json_serialisable_with_nulls_intact(self):
        from app.contracts.device_status import Validity
        from app.contracts.metric_sample import MetricSample

        absent = MetricSample(
            name="system_cpu_percent", value=None, unit="percent", provider="psutil",
            device_scope=None, process_scope=("ui",), window_seconds=0.0,
            sampled_at=1.0, validity=Validity.UNAVAILABLE,
            unavailable_reason="no window elapsed",
        )
        summary = self._summary(samples=(absent,), frame_ages_ms=(), frames_analysed=0,
                                elapsed_s=1.0)
        text = json.dumps(summary)
        self.assertIn("null", text,
                      "an unmeasured figure must serialise as null, not as 0")
        self.assertIn('"ai_fps": null', text)
        self.assertIn('"p50": null', text)
        self.assertEqual(json.loads(text)["frames_submitted"], 10)


class CommandLine(unittest.TestCase):
    """The CLI surface the brief specifies, checked without running the pipeline."""

    def test_every_flag_the_brief_names_is_accepted(self):
        from app.main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--scenario", "retail", "--device", "GPU", "--source", "loop",
            "--seconds", "3", "--json", "out.json",
        ])
        self.assertEqual(args.scenario, "retail")
        self.assertEqual(args.device, "GPU")
        self.assertEqual(args.source, "loop")
        self.assertEqual(args.seconds, 3.0)
        self.assertEqual(args.json, "out.json")

    def test_an_unknown_device_is_refused_by_the_parser(self):
        from app.main import build_parser

        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--device", "TPU"])

    def test_the_headless_flag_is_accepted(self):
        from app.main import build_parser

        args = build_parser().parse_args(["--headless", "--seconds", "1"])
        self.assertTrue(args.headless)
        self.assertFalse(args.selftest)


class InputAcknowledgement(unittest.TestCase):
    """Plan failure mode 6: the UI thread must stay responsive under CPU saturation.

    ## What is actually measured

    A `QTimer` on the Qt main thread fires every 20 ms and records **lateness**: the gap
    between when the tick was due and when its slot ran. That is the same quantity a
    person at the booth experiences as "the screen did not respond" - if inference ran on
    the UI thread, every tick would be delayed by an inference.

    ## What "saturating" means here

    **CPU-only**, as the plan requires: `os.cpu_count()` busy-loop processes (8 on this
    machine) **plus** the real `Supervisor` running real CPU inference over the real clip.
    The saturators are `multiprocessing` processes, not threads, so they occupy cores
    rather than contending for the GIL - a thread-based saturator would mostly be testing
    the GIL. The gate is p95 lateness < 200 ms, and the **worst** lateness is reported
    alongside it rather than dropped, because a p95 that hides one 900 ms stall is not the
    number an operator needs.

    Runs in a subprocess: the saturators take every core for several seconds, and the rest
    of the suite must not be competing with them.
    """

    @classmethod
    def setUpClass(cls):
        cls.report = cls._measure()

    def test_the_ui_thread_stays_under_the_gate_while_the_cpu_is_saturated(self):
        self.assertGreaterEqual(self.report["ticks"], 40,
                                f"too few ticks to judge: {self.report}")
        self.assertLess(self.report["p95_ms"], 200.0, self.report)
        # Proof the load was real, so a fast UI thread cannot be explained by nothing
        # having run: the supervisor must have analysed frames, and the saturators must
        # have been up for the whole window.
        self.assertGreater(self.report["frames_analysed"], 0, self.report)
        self.assertGreaterEqual(self.report["saturators"], 4, self.report)

    def test_the_report_states_its_own_conditions(self):
        # A latency number without its load is not evidence. These are the fields that
        # make the number readable by somebody who was not at the machine.
        for key in ("p50_ms", "p95_ms", "max_ms", "ticks", "saturators",
                    "frames_analysed", "confirmed", "requested", "tick_ms"):
            self.assertIn(key, self.report)
        self.assertEqual(self.report["requested"], "CPU")
        self.assertEqual(self.report["confirmed"], ["CPU"])

    @staticmethod
    def _measure() -> dict:
        code = r'''
import json, os, sys, time
from pathlib import Path

import multiprocessing as mp

sys.path.insert(0, %r)

from PySide6.QtCore import QCoreApplication, QTimer

from app.engine.retail_config import RETAIL_CHECKOUT_CLIP, retail_sanity_manifest
from app.engine.source import ClipSource
from app.inference.pinned import MODEL_ID, pinned_input_shape
from app.supervisor.runner import Supervisor

TICK_MS = 20
SECONDS = 4.0


def burn(_):
    """One saturator core. CPU-only arithmetic; no allocation, no syscalls."""
    value = 0
    deadline = time.perf_counter() + SECONDS * 2.0
    while time.perf_counter() < deadline:
        for i in range(20000):
            value = (value * 31 + i) %% 1000003
    return value


lateness = []
app = QCoreApplication(sys.argv)

_, _, height, width = pinned_input_shape()
manifest = retail_sanity_manifest()
supervisor = Supervisor(device_id="CPU", stream_id="A", session_id="ack",
                        scenario_id=manifest.id,
                        conf_threshold=manifest.conf_threshold,
                        iou_threshold=manifest.iou_threshold)
clip = ClipSource(Path(RETAIL_CHECKOUT_CLIP), stream_id="A", session_id="ack",
                  scenario_id=manifest.id, target_fps=30.0)

stop = {"now": False}


def feed():
    while not stop["now"]:
        frame = clip.read()
        if frame is None:
            if not clip.restart():
                return
            continue
        supervisor.submit(frame)


feeder = __import__("threading").Thread(target=feed, daemon=True)

pending = {"due": 0.0}


def tick():
    now = time.perf_counter()
    lateness.append((now - pending["due"]) * 1000.0)
    pending["due"] = now + TICK_MS / 1000.0


timer = QTimer()
timer.timeout.connect(tick)
timer.start(TICK_MS)

saturators = mp.cpu_count()
procs = [mp.Process(target=burn, args=(i,)) for i in range(saturators)]
supervisor.start()
pending["due"] = time.perf_counter() + TICK_MS / 1000.0
timer.start(TICK_MS)
for proc in procs:
    proc.start()
feeder.start()

QTimer.singleShot(int(SECONDS * 1000), app.quit)
app.exec()

stop["now"] = True
timer.stop()
for proc in procs:
    proc.terminate()
    proc.join(timeout=2.0)
clip.close()
supervisor.stop()
plan = supervisor.route_plan()

ordered = sorted(lateness)


def pct(fraction):
    if not ordered:
        return None
    rank = max(1, int(-(-fraction * len(ordered) // 1)))
    return round(ordered[min(rank, len(ordered)) - 1], 3)


print(json.dumps({
    "p50_ms": pct(0.50),
    "p95_ms": pct(0.95),
    "max_ms": round(ordered[-1], 3) if ordered else None,
    "ticks": len(ordered),
    "tick_ms": TICK_MS,
    "saturators": saturators,
    "frames_analysed": supervisor.frames_analysed,
    "requested": "CPU",
    "confirmed": list(plan.confirmed_devices("A")),
}))
''' % str(REPO_ROOT)
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYTHONPATH"] = str(REPO_ROOT)
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                   text=True, timeout=600, cwd=str(REPO_ROOT), env=env)
        if completed.returncode != 0:
            raise AssertionError("the acknowledgement probe failed:\n"
                                 + completed.stdout + completed.stderr)
        return json.loads(completed.stdout.strip().splitlines()[-1])


class NoToolsImport(unittest.TestCase):
    """AGENTS.md's boundary: nothing under `app/` imports from `tools/`.

    Checked by reading every module's **source**, not by importing: an import check only
    sees what was executed on the path taken, and `app.main --selftest` does not reach
    every module.
    """

    def test_no_app_module_imports_from_tools(self):
        import re

        pattern = re.compile(r"^\s*(?:from|import)\s+tools(?:\.|\s|$)", re.MULTILINE)
        offenders = []
        for source in sorted(APP_DIR.rglob("*.py")):
            text = source.read_text(encoding="utf-8")
            for match in pattern.finditer(text):
                line = text[match.start():text.find("\n", match.start())]
                offenders.append(f"{source.relative_to(REPO_ROOT)}: {line.strip()}")
        self.assertEqual(offenders, [],
                         f"app/ modules import from tools/: {offenders}. The app layer "
                         f"ships; a tool is not in the package.")

    def test_the_application_does_not_import_a_simulation_module(self):
        # `assert_no_simulation_in_release` is the gate; a module whose name says
        # `simulated` and which is imported by `app/` would be a way around it.
        import ast

        offenders = []
        for source in sorted(APP_DIR.rglob("*.py")):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    leaf = name.rsplit(".", 1)[-1].lower()
                    if "simulat" in leaf or leaf in ("mock", "fake", "stub"):
                        offenders.append(f"{source.relative_to(REPO_ROOT)}: {name}")
        self.assertEqual(offenders, [],
                         f"app/ imports a simulated-value module: {offenders}")


class RealHeadlessRun(unittest.TestCase):
    """A real run against the real clip, when this machine can do one."""

    def test_a_cpu_headless_run_reports_a_confirmed_placement(self):
        reason = _provisioned()
        if reason:
            self.skipTest(
                f"cannot run the real pipeline: {reason}. This is skipped with that "
                f"reason rather than passing silently: no placement assertion about "
                f"this machine has been made.")
        with tempfile.TemporaryDirectory(prefix="roadshow-headless-") as tmp:
            out = Path(tmp) / "run.json"
            r = _run("--headless", "--device", "CPU", "--seconds", "4",
                     "--json", str(out))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            summary = json.loads(out.read_text(encoding="utf-8"))
        self.assertGreaterEqual(summary["frames_submitted"], 1)
        self.assertIn("confirmed", summary["device"])
        # A CPU run on a CPU machine should confirm CPU; if it does not, this run has
        # said something about the machine and the report needs to know.
        self.assertEqual(summary["device"]["confirmed"], ["CPU"], summary["device"])
        self.assertIsNotNone(summary["ai_fps"])


if __name__ == "__main__":  # pragma: no cover - convenience for a human at a booth
    unittest.main()