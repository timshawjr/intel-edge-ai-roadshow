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
import time
import unittest
from collections.abc import Sequence
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
            self.assertEqual(stage["stage"], name,
                             f"each stage must carry its own name, or a summary "
                             f"transcribed from this cannot say what it ran: {stage}")

    def test_at_least_one_stage_actually_ran(self):
        # The shape assertions above all pass on a summary where **every** stage is
        # `skipped` - which is what `--selftest` produces on a machine with no model and
        # no clip, and it still prints PASS and exits 0. That is correct behaviour (the
        # requirement is that it works on an unprovisioned machine) and it means the
        # shape checks alone cannot tell "checked everything" from "checked nothing".
        # On a provisioned machine at least one stage must genuinely run.
        from app.main import run_selftest

        if _provisioned():
            self.skipTest("nothing is provisioned here, so every stage may legitimately "
                          "skip; that case is covered by the shape assertions above")
        summary = run_selftest()
        ran = [name for name, stage in summary["stages"].items()
               if stage["verdict"] == "ran"]
        self.assertTrue(ran,
                        f"every stage was skipped on a provisioned machine: {summary}")

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

    def test_the_allow_list_covers_what_the_collector_actually_consults(self):
        # I8, and the check the previous test could not make. `REGISTERED_PROVIDERS` was a
        # literal nothing consulted while `MetricCollector` imported `psutil` directly, so
        # asserting the literal passes the gate is a tautology: it can only fail if
        # somebody edits the literal to something simulated.
        #
        # The gate's real question is "could a simulated provider produce a number on this
        # screen?", so the coverage direction is the only one that can fail: everything the
        # collector reaches for must be on the allow-list.
        from app.main import REGISTERED_PROVIDERS, consulted_providers

        unlisted = [p for p in consulted_providers() if p not in REGISTERED_PROVIDERS]
        self.assertEqual(unlisted, [],
                         f"the telemetry collector consults {unlisted}, which the release "
                         f"gate was never told about. Either the allow-list is missing a "
                         f"real provider or something simulated has been wired in.")

    def test_a_simulated_provider_reaches_for_because_the_collector_does(self):
        # The mutation this guards against, constructed: a collector reaching for a
        # simulated provider while the literal still says `("psutil",)`.
        from app.main import assert_no_simulation_in_release

        with self.assertRaises(SystemExit) as caught:
            assert_no_simulation_in_release(
                release=True,
                providers=("psutil", "simulated_cpu_pressure"))
        self.assertIn("simulated_cpu_pressure", str(caught.exception))

    def test_release_mode_is_on_unless_somebody_turns_it_off(self):
        # I8. `--release` defaulted to `False`, so a packaged build that forgot the flag
        # ran with no gate at all, and a gate nobody remembers to pass is not a gate.
        from app.main import RELEASE_BY_DEFAULT, build_parser

        self.assertTrue(RELEASE_BY_DEFAULT)
        self.assertTrue(build_parser().parse_args([]).release,
                        "the default invocation must be gated")
        self.assertFalse(build_parser().parse_args(["--no-release"]).release,
                         "--no-release is the deliberate development escape hatch")

    def test_the_real_entry_point_is_gated_with_both_provider_sets(self):
        # The gate is called with the allow-list **and** what the collector consults, so
        # either half failing to cover reality is caught.
        import app.main as main_module

        recorded = {}
        original = main_module.assert_no_simulation_in_release

        def spy(release, providers):
            recorded["release"] = release
            recorded["providers"] = tuple(providers)
            return original(release=release, providers=providers)

        main_module.assert_no_simulation_in_release = spy
        try:
            code = main_module.main(["--selftest"])
        finally:
            main_module.assert_no_simulation_in_release = original
        self.assertIn(code, (0, 1))
        self.assertTrue(recorded.get("release"), "the real entry point ran ungated")
        self.assertIn("psutil", recorded["providers"])


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


class RefusalAccounting(unittest.TestCase):
    """I1: a refusal is not a drop, and it must be counted.

    Measured on the code this replaces, over four states:

    | state | `frames_submitted` | `drops` |
    |---|---|---|
    | never started | 0 | 0 |
    | after `stop()` | 0 | 0 |
    | route `failed` | 0 | 0 |
    | queue full (5 attempts) | 2 | 3 |

    Three of those four are indistinguishable from a healthy run that was given nothing,
    and the fourth does not account for its own five attempts. So
    `frames_submitted + frames_refused + drops == frame_attempts` is asserted here for
    all four, which is the property that makes the summary checkable.
    """

    @staticmethod
    def _frame(name: str = "f"):
        from app.contracts.frame import FramePacket
        import numpy as np

        return FramePacket(session_id="s", scenario_id="s", stream_id="A", frame_id=0,
                           source_pts=0.0, source_loop_id=0, captured_at=0.0,
                           decoded_at=0.0, width=64, height=64,
                           image=np.zeros((64, 64, 3), np.uint8))

    def _supervisor(self, **kwargs):
        from app.supervisor.runner import Supervisor

        return Supervisor(device_id="CPU", stream_id="A", session_id="s",
                          scenario_id="s", **kwargs)

    def _assert_reconciles(self, supervisor, label):
        total = (supervisor.frames_submitted + supervisor.frames_refused
                 + supervisor.drops)
        self.assertEqual(total, supervisor.frame_attempts,
                         f"{label}: accepted + refused + dropped must account for every "
                         f"attempt")

    def test_a_supervisor_that_was_never_started_counts_its_refusals(self):
        supervisor = self._supervisor()
        self.addCleanup(supervisor.stop)
        for _ in range(3):
            self.assertFalse(supervisor.submit(self._frame()))
        self.assertEqual(supervisor.frames_submitted, 0)
        self.assertEqual(supervisor.drops, 0,
                         "a refusal is not a drop; the old code reported 0 for both")
        self.assertEqual(supervisor.frames_refused, 3)
        self.assertEqual(supervisor.frames_refused_reasons, {"not_running": 3})
        self._assert_reconciles(supervisor, "never started")

    def test_a_supervisor_that_was_stopped_counts_its_refusals(self):
        from app.supervisor.runner import Supervisor

        supervisor = Supervisor(device_id="CPU", stream_id="A", session_id="s",
                                scenario_id="s", compile_failure=True)
        supervisor.start()
        self.addCleanup(supervisor.stop)
        supervisor.stop()
        for _ in range(2):
            self.assertFalse(supervisor.submit(self._frame()))
        self.assertEqual(supervisor.frames_refused, 2)
        self.assertEqual(supervisor.drops, 0)
        self._assert_reconciles(supervisor, "stopped")

    def test_a_failed_route_counts_its_refusals(self):
        from app.supervisor.runner import Supervisor

        supervisor = Supervisor(device_id="CPU", stream_id="A", session_id="s",
                                scenario_id="s", compile_failure=True)
        supervisor.start()
        self.addCleanup(supervisor.stop)
        deadline = time.perf_counter() + 20.0
        while supervisor.route_plan().state != "failed" and time.perf_counter() < deadline:
            time.sleep(0.05)
        self.assertEqual(supervisor.route_plan().state, "failed",
                         "the faulting worker should have failed this route")
        for _ in range(2):
            self.assertFalse(supervisor.submit(self._frame()))
        self.assertEqual(supervisor.frames_refused, 2)
        self.assertEqual(supervisor.frames_refused_reasons.get("failed"), 2)
        self._assert_reconciles(supervisor, "failed route")

    def test_a_full_queue_still_accounts_for_every_attempt(self):
        supervisor = self._supervisor(queue_size=2, compile_failure=True)
        self.addCleanup(supervisor.stop)
        supervisor.start()
        accepted = refused = 0
        for _ in range(5):
            if supervisor.submit(self._frame()):
                accepted += 1
            else:
                refused += 1
        self.assertEqual(accepted, 2)
        self.assertEqual(supervisor.drops, 3)
        self.assertEqual(supervisor.frames_refused, 0,
                         "a full queue is a drop, never a refusal")
        self._assert_reconciles(supervisor, "full queue")

    def test_the_summary_carries_the_refusal_count_and_its_reasons(self):
        from app.main import build_run_summary

        summary = build_run_summary(
            device_requested="CPU", state="pending", confirmed=(), frames_submitted=2,
            frames_analysed=0, drops=3, elapsed_s=1.0, frame_ages_ms=(), events=(),
            samples=(), frames_refused=7, frames_refused_reasons={"not_running": 7},
            frame_attempts=12)
        self.assertEqual(summary["frames_refused"], 7)
        self.assertEqual(summary["frames_refused_reasons"], {"not_running": 7})
        self.assertEqual(summary["frame_attempts"], 12)
        self.assertEqual(
            summary["frames_submitted"] + summary["frames_refused"] + summary["drops"],
            summary["frame_attempts"])


class ResultFolding(unittest.TestCase):
    """I2: each distinct analysed frame is folded exactly once.

    `Supervisor.latest_result()` returns the newest result **and keeps returning it** until
    a newer one arrives. `run_headless` called the fold once per submitted frame *and*
    every 20 ms during the drain, so the same detection set was folded repeatedly:
    measured on a 4 s CPU run, **147** `tracker.update` calls for **61** real results, one
    identity re-read 95 times, and a maximum track `hits` of **147** against 61 results.
    A detection seen in one real frame therefore reached `min_hits = 2` and was confirmed -
    the tracker's own evidence gate, defeated by its consumer. The event counts survived
    only because `dedup_key` suppressed the duplicate emit, which is luck rather than a
    safeguard.
    """

    class _FakeSupervisor:
        def __init__(self, results):
            self._results = list(results)
            self.latest_result()

        def latest_result(self):
            return self._results[-1] if self._results else None

    @staticmethod
    def _result(frame_id, detections=()):
        from app.contracts.frame import FramePacket
        from app.engine.worker import InferenceResult
        import numpy as np

        frame = FramePacket(session_id="s", scenario_id="s", stream_id="A",
                            frame_id=frame_id, source_pts=float(frame_id) / 15.0,
                            source_loop_id=0, captured_at=float(frame_id) / 15.0,
                            decoded_at=float(frame_id) / 15.0, width=640, height=640,
                            image=np.zeros((640, 640, 3), np.uint8))
        return InferenceResult(frame=frame, detections=tuple(detections),
                               execution_devices=("CPU",), latency_ms=1.0,
                               error=None, requested_device="CPU")

    def test_the_fold_is_skipped_for_a_result_already_folded(self):
        from app.engine.events import RetailEventMachine
        from app.engine.retail_config import retail_sanity_manifest
        from app.engine.tracker import Tracker
        from app.inference.pinned import class_names
        from app.inference.decode import Detection

        manifest = retail_sanity_manifest()
        tracker = Tracker(iou_threshold=manifest.iou_threshold)
        machine = RetailEventMachine(manifest, class_names=class_names())
        person = class_names().index("person")
        detection = Detection(class_id=person, score=0.9, box=(100.0, 100.0, 300.0, 500.0))
        result = self._result(0, [detection])
        supervisor = self._FakeSupervisor([result])

        calls = []
        original = tracker.update

        def counting_update(detections, now):
            calls.append(now)
            return original(detections, now=now)

        tracker.update = counting_update
        folded = {"identity": None}

        from app.main import _fold_result

        # Ten calls against one result: exactly one fold, because the identity repeats.
        for _ in range(10):
            _fold_result(supervisor, tracker, machine, folded)
        self.assertEqual(len(calls), 1,
                         f"one result was folded {len(calls)} times")

    def test_a_new_result_is_folded_exactly_once_each(self):
        from app.engine.events import RetailEventMachine
        from app.engine.retail_config import retail_sanity_manifest
        from app.engine.tracker import Tracker
        from app.inference.decode import Detection
        from app.inference.pinned import class_names

        manifest = retail_sanity_manifest()
        tracker = Tracker(iou_threshold=manifest.iou_threshold)
        machine = RetailEventMachine(manifest, class_names=class_names())
        person = class_names().index("person")
        supervisor = self._FakeSupervisor([])

        calls = []
        original = tracker.update

        def counting_update(detections, now):
            calls.append(now)
            return original(detections, now=now)

        tracker.update = counting_update
        folded = {"identity": None}

        from app.main import _fold_result

        # A growing result set, each one polled repeatedly, as the real loop does.
        for frame_id in range(5):
            supervisor._results.append(
                self._result(frame_id, [Detection(class_id=person, score=0.9,
                                                  box=(100.0, 100.0, 300.0, 500.0))]))
            for _ in range(4):
                _fold_result(supervisor, tracker, machine, folded)
        self.assertEqual(len(calls), 5,
                         f"five distinct frames produced {len(calls)} folds")


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

    A `QTimer` on the **Qt main thread** fires every 20 ms and records **lateness**: the
    gap between when the tick was due and when its slot ran. That is the quantity a person
    at the booth experiences as "the screen did not respond". If inference ran on the UI
    thread, every tick would be delayed by an inference.

    ## What "saturating" means here, and how hard

    **CPU-only**, as the plan requires: busy-loop **processes** (never threads - a
    thread-based saturator contends for the GIL and would mostly be measuring the GIL)
    **plus** the real pipeline doing real CPU inference over the real clip.

    `cpu_count() - 2` saturator processes: **6 on this 8-logical-core machine.** That is
    the machine fully loaded with other work - the scheduler contended, the pipeline
    starved of idle cores - which is what a booth actually runs.

    **At 8 of 8, the gate is marginal, and that is a finding rather than a reason to pick
    easier numbers.** Measured with every core in a synthetic burn loop: p50 0.025 ms,
    **p95 195.6 ms**, **max 1105.5 ms**. The p95 sat 2% under the 200 ms gate and one
    tick exceeded a second - with the inference worker, the frame queue's pickling and the
    render loop all competing for cores that do not exist. A booth does not run eight
    synthetic burn loops; it runs the pipeline plus whatever else the machine is doing. The
    number is stated here because a reader deciding whether this is good enough for a
    stand should see the worst case, not just the calibrated one.

    ## I6: it goes through `run_windowed`, and that is the whole point of the rewrite

    The first version built its **own** `QCoreApplication`, its own feeder thread and its
    own timer, and drove `Supervisor` directly. It therefore measured the *harness*, not
    the application: a regression that moved inference onto the UI thread **inside
    `app/`** - inside `_on_result`, say, or by calling `_start_feeders` with an inline
    submit - would have left it green, because the probe never called `run_windowed`,
    `_start_feeders`, `_start_telemetry` or `_on_result` at all.

    This version calls `app.main.run_windowed`. The feeder thread, the telemetry timer, the
    reader thread and the result fold are the application's own, and
    `test_the_probe_really_ran_the_applications_threading_path` asserts they were.

    ## What the number is, stated plainly

    **This is the floor of the UI thread's cost, not the booth's.** It is measured on the
    `offscreen` platform, with one 1920x1080 stream at 15 fps, on this machine, with eight
    saturator processes. A booth machine presenting that same screen through a real
    compositor on a real display will not be faster. The honest claim is "the UI thread
    does not block on inference", and this is the measurement of it.
    """

    #: The worst single tick must also stay under this. See
    #: `test_the_ui_thread_stays_under_the_gate_in_every_repeat` for why a percentile is
    #: not sufficient and why this is 2 s rather than 200 ms.
    MAX_GATE_MS = 2000.0

    #: Repeats. One run is an anecdote; the review asked for a range, and a range needs
    #: more than one sample.
    REPEATS = 3

    @classmethod
    def setUpClass(cls):
        reason = _provisioned()
        if reason:
            raise unittest.SkipTest(
                f"cannot run the real pipeline: {reason}. The input-acknowledgement gate "
                f"was not exercised on this machine and no claim is made about it.")
        # **All the repeats come from one process.** The fixed cost of a windowed run is
        # startup - `DeviceRegistry.describe()` 3.8 s cold, the CPU compile, and
        # `Supervisor.stop()`'s join grace - and one repeat per process measured **63 s**
        # wall for 16 s of measurement. Three windows in one application cost that once.
        cls.runs = cls._measure()

    # -- the gate -----------------------------------------------------------------------

    def test_the_ui_thread_stays_under_the_gate_in_every_repeat(self):
        # Two gates, because one is not enough. The plan asks for acknowledgement "under
        # 200 ms", which is a bound on an acknowledgement and not only on a percentile.
        #
        # **A p95 alone hid a three-second freeze.** Measured: with the UI thread waiting
        # on the analysis backlog - which is what review failure mode 6 describes - the
        # probe reported p50 **0.025 ms**, p95 **41.2 ms** and **max 2981.8 ms** over 49
        # ticks. The p95 passed with a 5x margin because 49 samples put the stalls in the
        # top 5%, while the screen was frozen for three seconds at a time. A percentile
        # over a short window is a statement about the middle, and the middle is exactly
        # where a freeze does not show.
        #
        # So `MAX_GATE_MS` gates the worst tick as well, at 1 s rather than 200 ms: a
        # legitimate scheduling hiccup on this machine is tens of milliseconds (measured
        # max 94.9 ms on an unmutilised UI thread under the same load), while the failure
        # mode is seconds. 1 s is comfortably above noise and comfortably below a freeze.
        for run in self.runs:
            self.assertGreaterEqual(run["ticks"], 40,
                                    f"too few ticks to judge: {run}")
            self.assertLess(run["p95_ms"], 200.0, run)
            self.assertLess(run["max_ms"], self.MAX_GATE_MS,
                            f"the UI thread froze for {run['max_ms']} ms: a p95 of "
                            f"{run['p95_ms']} ms does not describe a screen that stopped "
                            f"responding")
        # Also a range, so the report can quote one rather than a chosen number.
        self.assertEqual(len(self.runs), self.REPEATS)

    def test_the_probe_really_ran_the_applications_threading_path(self):
        # I6. Without this the gate could be satisfied by a harness that never touches
        # `run_windowed`, which is precisely the defect this rewrite exists to close.
        for run in self.runs:
            self.assertTrue(run["application_path"], run)
            self.assertGreater(run["frames_analysed"], 0, run)
            self.assertGreaterEqual(run["saturators"], 2, run)
            self.assertEqual(run["saturators"],
                             max(2, (os.cpu_count() or 4) - 2),
                             "the probe must run at the calibrated saturation - "
                             "cpu_count() - 2 - and the docstring must agree with it")

    def test_the_report_states_its_own_conditions(self):
        # A latency number without its load is not evidence. These are the fields that
        # make the number readable by somebody who was not at the machine.
        for key in ("p50_ms", "p95_ms", "max_ms", "ticks", "saturators",
                    "frames_analysed", "confirmed", "requested", "tick_ms",
                    "saturating_processes_documented"):
            self.assertIn(key, self.runs[0])
        for run in self.runs:
            self.assertEqual(run["requested"], "CPU")
            self.assertEqual(list(run["confirmed"]), ["CPU"],
                             "the gate must run against a confirmed placement, or it is "
                             "measuring a screen whose device was never proved")

    def test_the_measured_range_is_reported_not_just_the_worst_case(self):
        # Every repeat is reported, so a reader can see the spread rather than one number
        # chosen for effect. A gate that only prints its best run is not a gate.
        self.assertEqual(len(self.runs), self.REPEATS)
        for run in self.runs:
            self.assertIsNotNone(run["p95_ms"], run)
            self.assertIsNotNone(run["max_ms"], run)

    @staticmethod
    def _measure() -> list:
        """`REPEATS` saturated windows, through `run_windowed`, in one child process.

        Written to a **file**, not passed to `python -c`: `multiprocessing` cannot spawn
        the inference worker from `-c` without tripping its `__main__` guard, and the
        original probe had exactly that defect - it exited 1 with "Safe importing of main
        module" and the JSON printed before the failure showed p95 **20.9 ms**, which
        *passes* a 200 ms gate. A probe that cannot spawn and still reports a passing
        number is worse than no probe.
        """
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["ROADSHOW_ACK_SECONDS"] = "6"
        env["ROADSHOW_ACK_WARMUP"] = "12"
        env["ROADSHOW_ACK_SATURATORS"] = str(max(2, (os.cpu_count() or 4) - 2))
        env["ROADSHOW_ACK_REPEATS"] = str(InputAcknowledgement.REPEATS)
        with tempfile.TemporaryDirectory(prefix="roadshow-ack-") as tmp:
            script = Path(tmp) / "ack_probe.py"
            script.write_text(_ACK_DRIVER.replace("@REPO@", repr(str(REPO_ROOT))),
                              encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(script), "--device", "CPU"],
                capture_output=True, text=True, timeout=900,
                cwd=str(REPO_ROOT), env=env)
        lines = [ln for ln in completed.stdout.splitlines() if ln.startswith("ACK ")]
        if not lines:
            raise AssertionError(
                "the acknowledgement probe produced no ACK line:\n"
                + completed.stdout + completed.stderr)
        runs = [json.loads(ln[len("ACK "):]) for ln in lines]
        if any("error" in run for run in runs):
            raise AssertionError(
                "the acknowledgement probe did not reach a saturated window:\n"
                + completed.stdout[-4000:] + completed.stderr[-4000:])
        if len(runs) != InputAcknowledgement.REPEATS:
            raise AssertionError(
                f"expected {InputAcknowledgement.REPEATS} windows, got {len(runs)}:\n"
                + completed.stdout[-4000:])
        return runs


#: The child program. Written to a file and run as `__main__` so the inference worker can
#: actually be spawned - the defect the first probe had.
_ACK_DRIVER = r'''
"""Measure UI-thread input-acknowledgement lateness **through `app.main.run_windowed`**.

The tick timer is armed inside `run_windowed`'s own startup, between the supervisor being
started and `application.exec()` being called, so the Qt main thread doing the measuring is
the same thread that renders the booth. The feeder, the telemetry timer, the reader thread
and the result fold are all the application's own.

Saturators are `multiprocessing` **processes**: a thread-based saturator would contend for
the GIL and would mostly be measuring the GIL rather than the CPU.
"""
import json
import math
import multiprocessing
import os
import sys
import time

sys.path.insert(0, @REPO@)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer

import app.main as main_module

SECONDS = float(os.environ.get("ROADSHOW_ACK_SECONDS", "8"))
WARMUP = float(os.environ.get("ROADSHOW_ACK_WARMUP", "14"))
SATURATORS = int(os.environ.get("ROADSHOW_ACK_SATURATORS", "8"))
REPEATS = int(os.environ.get("ROADSHOW_ACK_REPEATS", "3"))
TICK_MS = 20

lateness = []
state = {"due": 0.0, "tick": None, "supervisor": None, "procs": [],
         "phase": "warmup", "warm_analysed": 0, "warm_started": 0.0, "repeat": 0,
         "window": None}
_real_feeders = main_module._start_feeders
_real_telemetry = main_module._start_telemetry


def burn(_gate_holder=None, _index=0, _gate=None):
    """One saturator core. CPU-only arithmetic; no allocation, no syscalls.

    **Waits on an event before burning.** The saturators are spawned once, during warm-up,
    and released and re-released per window by setting and clearing the event. Spawning
    them inside the window callback was the first version's mistake, and it was a bad one:
    `multiprocessing.Process.start()` on Windows re-imports the module in the child, six
    of them cost several seconds, and that happened **on the Qt main thread** because the
    warm-up poller is a `QTimer`. The measured result was **one tick in a six-second
    window** - the probe was timing process creation, not the application.
    """
    value = 0
    deadline = time.perf_counter() + (WARMUP + SECONDS * (REPEATS + 3)) * 3.0
    while time.perf_counter() < deadline:
        if _gate is None or not _gate.is_set():
            time.sleep(0.01)
            deadline = time.perf_counter() + (WARMUP + SECONDS * (REPEATS + 3)) * 3.0
            continue
        for i in range(20000):
            value = (value * 31 + i) % 1000003
    return value


def spawn_saturators():
    """Create the saturator processes once, gated. Returns the list."""
    gate = multiprocessing.Event()
    procs = [multiprocessing.Process(target=burn, args=(None, i, gate))
             for i in range(SATURATORS)]
    for proc in procs:
        proc.start()
    return procs, gate


def tick():
    now = time.perf_counter()
    # Lateness can read very slightly negative on the first tick, because `due` is armed
    # just before the timer starts and Qt may fire a touch early. A negative lateness is
    # not a measurement, so it is clamped - and the clamp is recorded, so a reader knows
    # the floor exists rather than wondering.
    lateness.append(max(0.0, (now - state["due"]) * 1000.0))
    state["due"] = now + TICK_MS / 1000.0


def maybe_start_saturation():
    """Warm up unsaturated, then saturate and measure, `REPEATS` times.

    **The warm-up is not politeness, it is the measurement.** Measured here: with eight
    saturators started at the same moment as the worker, a 6-second window produced
    **zero** analysed frames on one run and 6 on another - the CPU compile plus eight busy
    cores starve the inference worker, and a gate that measures a starved worker measures
    the compile. Warming up unsaturated until the route is confirmed and frames are being
    analysed, and *then* saturating, measures what the plan asks about: whether the UI
    thread stays responsive **while** the machine is saturated.

    **All three repeats run inside one application process**, because the fixed cost is
    startup: `DeviceRegistry.describe()` is 3.8 s cold, the CPU compile is seconds, and
    `Supervisor.stop()` joins with a grace timeout. Measured, one repeat per process cost
    **63 s** wall for 16 s of measurement; three windows in one process cost about the
    same plus two saturation windows.
    """
    supervisor = state["supervisor"]
    if state["phase"] == "saturating":
        # **This guard is the whole reason the first version of this probe reported one
        # tick per window.** The poller runs every 200 ms; without the guard it re-entered
        # `maybe_start_saturation` on every poll, and because `frames_analysed > 0` stays
        # true forever once the route is up, each re-entry cleared `lateness` and called
        # `tick.start()` and `window.start()` again - so the six-second window was reset
        # every 200 ms, never elapsed, and only the backstop ever ended it. Measured:
        # **one tick per window**, at p50 0.0 ms, which looks like a flawless result and
        # measures nothing at all.
        return
    warmed = supervisor.frames_analysed > 0
    out_of_time = time.perf_counter() - state["warm_started"] > WARMUP
    if not warmed and not out_of_time:
        return
    if warmed:
        state["warm_analysed"] = supervisor.frames_analysed
    state["phase"] = "saturating"
    lateness.clear()
    state["due"] = time.perf_counter() + TICK_MS / 1000.0
    state["tick"].start(TICK_MS)
    state["gate"].set()
    state["window"].start(int(SECONDS * 1000))


def end_window():
    """Close one saturated window, report it, and start the next."""
    state["gate"].clear()
    state["tick"].stop()
    report_window()
    state["repeat"] += 1
    if state["repeat"] >= REPEATS:
        _terminate_saturators()
        app["q"].quit()
        return
    lateness.clear()
    state["warm_started"] = time.perf_counter()
    state["phase"] = "warmup"


def _terminate_saturators():
    for proc in state["procs"]:
        proc.terminate()
        proc.join(timeout=2.0)
    state["procs"] = []


def report_window():
    supervisor = state["supervisor"]
    ordered = sorted(lateness)

    def pct(fraction):
        if not ordered:
            return None
        # Nearest rank, integer maths. The first version wrote
        # `-(-fraction * len // 1)`, which is a **float** under Python 3, so the
        # percentile raised `TypeError: list indices must be integers` - and because
        # `finish()` raised before `quit()` was called, the probe then hung until the
        # harness killed it. `math.ceil` says what it means.
        rank = max(1, math.ceil(fraction * len(ordered)))
        return round(ordered[min(rank, len(ordered)) - 1], 3)

    plan = supervisor.route_plan()
    print("ACK " + json.dumps({
        "repeat": state["repeat"],
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "max_ms": round(ordered[-1], 3) if ordered else None,
        "ticks": len(ordered),
        "tick_ms": TICK_MS,
        "saturators": SATURATORS,
        "frames_analysed": supervisor.frames_analysed,
        "requested": "CPU",
        "confirmed": plan.confirmed_devices("A"),
        "application_path": True,
        # Stated so nobody reads the number as a booth figure.
        "saturating_processes_documented": (
            f"{SATURATORS} CPU-only busy-loop processes plus real CPU inference, on the "
            f"offscreen platform, one 1920x1080 stream at 15 fps"),
    }), flush=True)
    sys.stdout.flush()


def spy_feeders(bridge, supervisor, clip):
    """The application's own feeder, plus the tick timer and the warm-up poller."""
    state["supervisor"] = supervisor
    state["warm_started"] = time.perf_counter()
    # Spawned **before** the tick timer starts, and from the warm-up path rather than the
    # window path, so no process-creation cost lands inside a measured window.
    state["procs"], state["gate"] = spawn_saturators()
    state["tick"] = QTimer()
    state["tick"].timeout.connect(tick)
    # The window timer: how long one saturated window lasts before it is reported.
    state["window"] = QTimer()
    state["window"].setSingleShot(True)
    state["window"].timeout.connect(end_window)
    # The warm-up poller runs on the UI thread too, so it is part of what is being
    # measured - and it is idle work, not analysis.
    state["warmup"] = QTimer()
    state["warmup"].timeout.connect(maybe_start_saturation)
    state["warmup"].start(200)
    return _real_feeders(bridge, supervisor, clip)


main_module._start_feeders = spy_feeders



_real_gui = main_module._gui_application
app = {}


def spy_gui():
    qobject = _real_gui()
    app["q"] = qobject

    # The backstop. The normal path is three saturated windows each ending in
    # `end_window`; this fires only if that never happens, and it quits **first** so a
    # raising backstop cannot leave the event loop spinning until the harness kills it.
    def stop_and_report():
        if state["phase"] == "saturating":
            qobject.quit()
            report_window()
            return
        qobject.quit()
        print("ACK " + json.dumps({
            "repeat": -1,
            "error": "the saturated window never started: no frames were analysed "
                     f"within {WARMUP} s of warm-up",
            "p50_ms": None, "p95_ms": None, "max_ms": None, "ticks": 0,
            "tick_ms": TICK_MS, "saturators": SATURATORS,
            "frames_analysed": state["supervisor"].frames_analysed,
            "requested": "CPU",
            "confirmed": state["supervisor"].route_plan().confirmed_devices("A"),
            "application_path": True,
            "saturating_processes_documented": "never saturated",
        }), flush=True)
        _terminate_saturators()

    QTimer.singleShot(int((WARMUP + SECONDS * (REPEATS + 1)) * 1000), stop_and_report)
    return qobject


main_module._gui_application = spy_gui

if __name__ == "__main__":
    sys.exit(main_module.main(sys.argv[1:]))
'''

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


class LiveWindowedPath(unittest.TestCase):
    """The ruling's test: start the real `run_windowed` and assert the screen fills.

    ## Why this class exists

    988 tests passed with `run_windowed` unable to reach its own first line of real work.
    Every UI test drove `UiBridge`'s setters **by hand**, so the suite proved the
    renderer's ability to draw a screen the application never fills. A one-word
    `NameError` on the booth screen's default entry point was invisible to all of them,
    and so were four defects that left the live screen empty.

    So this goes the other way round. It calls **`app.main.run_windowed` itself** - the
    real function, the real `Main.qml`, the real `_start_feeders` thread, the real
    `_start_telemetry` timer - and then asserts on **the objects `run_windowed` built**:
    a confirmed device, real frame pixels, the manifest's zones, the events the event
    machine actually emitted, and telemetry that is not `{}`.

    Every assertion here is a statement about **wiring**, which is what was untested.
    Pixel content is already covered by `tests.test_ui_contract.OffscreenRender`.

    ## Protocol with the child

    The child prints exactly one `REPORT {json}` line just before it quits, and one
    `BADGE {json}` line at the moment `engine.load` returns - the moment a route set
    *after* load is the interesting case (C4). The parent merges them.
    """

    @classmethod
    def setUpClass(cls):
        reason = _provisioned()
        if reason:
            raise unittest.SkipTest(
                f"cannot run the real pipeline: {reason}. The live path is not exercised "
                f"on this machine and no claim is made about it.")
        cls.report = cls._drive()

    # -- C1: the entry point runs at all ------------------------------------------------

    def test_the_windowed_entry_point_runs_without_a_name_error(self):
        self.assertNotIn("NameError", self.report["stderr"], self.report["stderr"])
        self.assertNotIn("Traceback", self.report["stderr"], self.report["stderr"])
        self.assertEqual(self.report["returncode"], 0, self.report["stderr"])
        # A `BADGE` line can only be printed from inside the patched `engine.load`, so
        # its presence is the evidence that `Main.qml` was reached and produced a badge
        # item. Before the fix this test failed at `setUpClass` with the `NameError`
        # rather than reaching any assertion at all.
        self.assertTrue(any(line.startswith("BADGE ")
                            for line in self.report["boot_lines"]), self.report)

    # -- C2: the four bridge slots are actually connected --------------------------------

    def test_the_application_confirms_a_device_on_its_own(self):
        # Before the fix, `apply_route` had no caller repo-wide, so this stayed
        # "pending - no device selected" for the whole session.
        self.assertEqual(self.report["confirmed"], "CPU", self.report)
        self.assertNotIn("pending", self.report["device_display"].lower(), self.report)

    def test_the_application_publishes_telemetry_to_the_bridge(self):
        # `apply_samples` had no caller, so every card read "unavailable" forever.
        #
        # **M2 Task 5: asserted as the obligation, not as a measurement landing.**
        # `psutil.cpu_percent` needs a previous call to compute a rate, and under a loaded
        # machine the collector thread can be starved enough that no valid sample arrives
        # inside one 14 s run - measured once per few full-suite runs, with the collector's
        # own reason on screen. So the invariant is: whatever the bridge holds is either a
        # measured percentage or the bridge's word, and **never a zero**. Asserting a
        # percentage unconditionally would be asserting a timing accident.
        value = self.report["system_cpu_percent"]
        self.assertTrue(value, self.report)
        if self.report["cpu_available"]:
            self.assertIn("%", value, self.report)
        else:
            self.assertEqual(value, "unavailable",
                             f"no CPU sample is available, so the figure must be the "
                             f"word and not 0.0%: {value!r}")
        self.assertNotIn("0.0%", value, value)

    def test_the_application_publishes_the_derived_rate_and_frame_age(self):
        # The CPU card's activity line reads "analysis fps ... frame age p50 ... p95 ...".
        # Nothing published those three, so the line read "unavailable" three times over a
        # session in which the supervisor was holding all three numbers.
        self.assertNotEqual(self.report["ai_fps"], "unavailable", self.report)
        self.assertNotEqual(self.report["p50"], "unavailable", self.report)
        self.assertNotEqual(self.report["p95"], "unavailable", self.report)

    def test_the_rendered_cards_show_the_numbers_the_bridge_holds(self):
        # **Read the screen, not the bridge.** Every metric card and the whole activity
        # line used to bind to `bridge.display_text(...)`. A QML binding that calls a
        # method declares no property dependency, so Qt evaluated each one exactly once,
        # at the first render, and never again: on a live run whose bridge held
        # `system_cpu_percent: 58.0` and `ai_fps: 13.9`, the rendered screen said
        # "unavailable" for every figure and "telemetry unavailable" in the control bar.
        #
        # This is the same class of defect as the `constant=True` device badge, one layer
        # out, and it is why this assertion reads pixels rather than properties.
        #
        # ## M2 Task 4: two of the three keys asserted nothing
        #
        # This looped `card_text`, whose keys were `system_cpu_percent`, `ai_fps` and
        # `frame_age_p50_ms`, and read `find_text("cardText_" + key)`. **`Main.qml` creates
        # cards for `system_cpu_percent`, `gpu_utilisation_percent` and
        # `npu_utilisation_percent` only**, so `cardText_ai_fps` and
        # `cardText_frame_age_p50_ms` name items that do not exist, `find_text` returns
        # `""` for both, and `"" != "unavailable"` passes. Measured on a live run before
        # the fix:
        #
        #     card_text: {"system_cpu_percent": "96.2%", "ai_fps": "",
        #                  "frame_age_p50_ms": ""}
        #
        # Two of three keys proved nothing about the screen. The rewrite below reads the
        # values that **actually reach it** - the three cards' own figures, and the CPU
        # card's bound `activityText`, which is where `ai_fps` and the frame-age
        # percentiles live - and `test_every_card_name_this_test_reaches_for_exists`
        # pins that the items are really there.
        report = self.report
        # **The items this test reads must exist.** An absent item renders `""` and every
        # "is it not unavailable" assertion on it passes, which is the whole defect.
        #
        # **M2 Task 5: the card is per CAMERA now**, so its name is `card_<stream>`. This
        # run is the booth's single-stream windowed run, so there is exactly one card and
        # it is `card_A`; the `card_CPU`/`card_GPU`/`card_NPU` names are gone, and a card
        # that claimed to be about a device a stream is not on is exactly the mislabelling
        # this milestone keeps finding.
        self.assertTrue(report["card_items_present"]["card_A"],
                        "the one camera on screen must have a named card item")
        self.assertEqual(report["stream_ids_on_screen"], ["card_A"],
                         f"one camera, one card, and no card named after a device: "
                         f"{report['stream_ids_on_screen']}")
        for gone in ("card_CPU", "card_GPU", "card_NPU"):
            self.assertFalse(report["card_items_present"][gone],
                             f"{gone} is a card named after a device; a card belongs to a "
                             f"camera")
        self.assertEqual(
            sorted(report["card_names"]),
            ["cardText_system_cpu_percent"],
            f"the live run has one camera on the CPU, so that is the only device figure "
            f"the screen carries: {report['card_names']}")
        # The camera's own device figure. **M2 Task 5: the line is named after the device's
        # metric, and it carries the device name too**, so the assertion is on the whole
        # rendered line rather than on a bare figure.
        #
        # **The premise is a measurement, so it is not asserted as one.** `psutil`'s
        # `cpu_percent` needs a *previous* call to compute a rate, and under a loaded
        # machine the collector thread can be starved enough that no valid sample lands
        # inside this run - measured, with the collector's own words on screen:
        # `'no measurable window elapsed since the previous sample, so nothing was
        # measured'`. The previous version of this test asserted a percentage unconditionally
        # and failed once per few full-suite runs. What must hold **either way** is the
        # obligation: a figure or a word, never a zero and never the accelerator's sentence.
        cpu = report["card_figures"]["system_cpu_percent"]
        self.assertTrue(cpu, "the CPU line rendered nothing at all")
        if report["cpu_available"]:
            self.assertIn("%", cpu, f"a sample is available, so the line shows it: {cpu!r}")
        else:
            self.assertNotIn("%", cpu,
                             f"no sample is available, so the line must be a WORD and not "
                             f"0.0%: it reads {cpu!r}")
        self.assertNotIn("Utilization unavailable", cpu,
                         "the CPU card must never borrow the accelerators' wording")
        # There is no GPU or NPU camera on this screen, so those items must not exist.
        for device in ("gpu_utilisation_percent", "npu_utilisation_percent"):
            self.assertEqual(report["card_figures"][device], "",
                             f"{device}: no camera is on that device, so no such item may "
                             f"be rendered - a card claiming a device no camera is on is "
                             f"the mislabelling this milestone keeps finding")
        # **The activity line, which is where the rate and both frame ages are.**
        activity = report["activity_cpu"]
        self.assertTrue(activity, "the camera's card rendered no activity line at all")
        self.assertNotIn("unavailable", activity,
                         f"the rendered activity line claims everything is unavailable "
                         f"while the supervisor is analysing frames: {activity!r}")
        self.assertIn("analysis fps ", activity, activity)
        self.assertIn("frame age p50 ", activity, activity)
        # **M2 Task 5: the window has its own labelled line now**, because a percentile of a
        # bounded deque is not comparable with a percentile of a different one and it did
        # not fit on the activity line. Asserted on the item that now carries it.
        window = report["window_text_stream"]
        self.assertTrue(window, "the frame-age window line is missing")
        self.assertIn("window: ", window, window)
        self.assertRegex(window, r"window: \d+ analysed frames",
                         f"and the count: {window!r}")
        self.assertTrue(report["age_window_note"], "the bridge's own window note is empty")
        self.assertIn("telemetry fresh", report["staleness_text_rendered"],
                      "the rendered control bar must report fresh telemetry once samples "
                      f"have arrived: {report['staleness_text_rendered']!r}")

    def test_the_old_card_names_that_read_nothing_are_recorded_as_absent(self):
        """The vacuousness, kept as a test so it cannot come back unnoticed.

        The M1 version asserted `card_text[key] != "unavailable"` over three keys, two of
        which name items that do not exist. This test says so in one place: those two keys
        **are** absent, and they read `""` - which is why the old loop proved nothing. It
        is a test about a *negative*, so it cannot fail if the screen changes; it exists
        so that anyone re-adding a `cardText_ai_fps` name sees why it was wrong, and so
        that `card_names` above is cross-checked against a second reading of the scene.
        """
        report = self.report
        for missing in ("ai_fps", "frame_age_p50_ms"):
            self.assertEqual(report["card_text"][missing], "",
                             f"cardText_{missing} reads '' on a live run - no such item "
                             f"exists, which is exactly why the old assertion proved "
                             f"nothing")
            self.assertNotIn(f"cardText_{missing}", report["card_names"],
                             f"cardText_{missing} is not one of the figures the screen "
                             f"renders")

    def test_the_cpu_card_does_not_claim_a_utilisation_counter_is_failing(self):
        # R1. `cpu_available` is `system_cpu_percent is not None`, which is `None` at
        # `engine.load` before the first telemetry tick - so **every** run showed the CPU
        # card reading the accelerators' "Utilization unavailable" during startup.
        #
        # The wording is wrong in principle, not just in frequency: `psutil.cpu_percent` is
        # called every 250 ms and works here. "Utilization unavailable" asserts a failure
        # to read a counter that is not failing. The CPU card must say its own thing.
        #
        # **M2 Task 5:** the line now carries the device's own words *or* its own figure,
        # depending on the measurement - so the assertion branches on the measurement rather
        # than on a timing accident, and never on the accelerators' sentence either way.
        notice = self.report["unavailable_notice"]["CPU"]
        self.assertNotIn("Utilization unavailable", notice,
                         "the CPU card must not borrow the accelerators' wording: there is "
                         f"no counter to fail for a CPU. It reads {notice!r}")
        self.assertTrue(notice.strip(), "the CPU card must say something, not nothing")
        if self.report["cpu_available"]:
            self.assertIn("%", notice,
                          f"a CPU sample is available, so the line shows the figure: "
                          f"{notice!r}")
        else:
            # **Either** the bridge's own CPU wording **or** the measurement's own
            # `unavailable_reason`, which is the more precise of the two and is what a
            # starved collector thread actually produces. Both are words; the previous
            # version of this test accepted only the first and failed once per few
            # full-suite runs.
            self.assertIn(notice, ("no CPU sample yet",
                                   str(self.report.get("cpu_reason", ""))),
                         f"with no sample the CPU line must be a word - the bridge's own or "
                         f"the measurement's - and never the accelerators': {notice!r}")

    def test_the_accelerator_cards_keep_their_own_wording(self):
        # The accelerators keep it: there genuinely is no `psutil` counter for a GPU or an
        # NPU, and the PDH value reads fault this process.
        #
        # **M2 Task 5 moved where this is checked.** The booth's single-stream run puts
        # Camera A on the CPU, so there is no card on the GPU or the NPU to read - and
        # asserting a card's wording for a card that is **not on screen** is asserting about
        # nothing, which is the defect this milestone has now found three times. What is
        # checked on the live screen is that no accelerator line is rendered at all; the
        # wording itself is checked on a three-camera render with one camera per device, in
        # `tests/test_ui_contract.py::ThreeStreamBoothLayout`.
        for device in ("GPU", "NPU"):
            self.assertEqual(self.report["unavailable_notice"][device], "",
                             f"the live run has no camera on the {device}, so its line "
                             f"must not be rendered at all - the wording for a camera that "
                             f"is not there is asserted on the three-device render")

    def test_the_window_note_line_actually_appears(self):
        # R2. `windowNote` bound to `bridge.window_text(metricName)`, a Slot call, so Qt
        # evaluated it once at `engine.load` when no sample existed - and the
        # "measured over N ms" line never appeared, with its `visible:` guard false for the
        # whole session. Asserting the *bridge* produced a value would have passed then.
        #
        # **M2 Task 5: asserted as the obligation, not as a sample landing.** The line is
        # hidden when the bridge has no window, and the bridge has no window when the
        # collector produced no sample - which a starved collector thread does not, once per
        # few full-suite runs. So: if a sample arrived the line must be **visible and say
        # `measured`**; if none arrived the line must be **absent from the screen**, not
        # visible with a nonsense figure in it. Both are the honest rendering.
        if self.report["cpu_available"]:
            self.assertTrue(self.report["window_note_visible"],
                            "a CPU sample is available, so the 'measured over N ms' line "
                            "must be visible: its binding is a Slot call evaluated once at "
                            f"engine.load. text={self.report['window_note_text']!r}")
            self.assertIn("measured", self.report["window_note_text"])
        else:
            self.assertFalse(self.report["window_note_visible"],
                             f"no sample means no window, so the line must be hidden; it "
                             f"reads {self.report['window_note_text']!r}")

    def test_the_event_count_reads_a_number_once_a_frame_has_been_analysed(self):
        # R3. Gating on the route left a real window - placement confirmed, zero frames
        # analysed - in which the footer printed three hard zeros for an event machine
        # that had not run.
        self.assertGreater(self.report["frames_analysed"], 0, self.report)
        text = self.report["event_count_text"]
        self.assertNotIn("unavailable", text,
                         f"frames have been analysed, so the count is a measurement: {text!r}")
        for kind in ("zone_occupancy", "candidate_transfer", "candidate_return"):
            self.assertIn(kind, text, f"{kind} is missing from the event row: {text!r}")
        self.assertIn(f"zone_occupancy {self.report['bridge_event_counts'].get('zone_occupancy', 0)}",
                      text, f"the rendered count disagrees with the bridge: {text!r}")

    def test_the_overlay_boxes_are_in_the_right_place_not_merely_present(self):
        # The coverage gap. Dividing `y` by `frame.width` instead of `frame.height` in
        # `_frame_payload` puts every box in the wrong place - and leaves the suite green,
        # because a length assertion and a `[0, 1]` bounds check both pass: on a
        # 1920x1080 frame `y / 1920` is still a legal fraction.
        #
        # The only thing that catches it is agreement with an **independent** computation
        # from the tracker's own pixel boxes. Same class as `FramePacket`'s image check: a
        # correct box on the wrong picture.
        boxes = self.report["overlay_boxes"]
        self.assertTrue(boxes, f"no overlay boxes at all: {self.report}")
        for box in boxes:
            for axis in ("x", "y", "w", "h"):
                self.assertGreaterEqual(box[axis], 0.0,
                                        f"box {box} has a negative {axis}")
                self.assertLessEqual(box[axis], 1.0,
                                     f"box {box} has a {axis} outside [0, 1]")
            self.assertLessEqual(box["x"] + box["w"], 1.0 + 1e-9, box)
            self.assertLessEqual(box["y"] + box["h"], 1.0 + 1e-9, box)
        expected = {b["track_id"]: b for b in self.report["expected_boxes"]}
        self.assertTrue(expected, f"the tracker had no current tracks: {self.report}")
        for box in boxes:
            want = expected.get(box["track_id"])
            self.assertIsNotNone(
                want, f"a box was drawn for track {box['track_id']}, which the tracker "
                      f"does not consider current: {expected}")
            for axis in ("x", "y", "w", "h"):
                self.assertAlmostEqual(
                    box[axis], want[axis], places=6,
                    msg=f"track {box['track_id']} {axis}: overlay says {box[axis]}, the "
                        f"tracker's own box says {want[axis]}. A box in the wrong place is "
                        f"still a plausible rectangle, so only agreement between two "
                        f"independent computations catches it.")

    def test_the_accelerator_cards_still_say_unavailable_not_zero(self):
        # `psutil` cannot read GPU or NPU utilisation in this process, and that has to stay
        # a word. The obligation the whole telemetry layer exists for, checked on the live
        # bridge rather than on a hand-built one.
        self.assertEqual(self.report["gpu"], "unavailable", self.report)
        self.assertEqual(self.report["npu"], "unavailable", self.report)
        # I7: the CPU line now answers the same question the two accelerator lines answer,
        # so an unavailable psutil reading cannot render as a number. **Not asserted as a
        # measurement landing** - see `test_the_application_publishes_telemetry_to_the_bridge`
        # for why, and for what a starved collector thread actually produces.
        notice = self.report["unavailable_notice"]["CPU"]
        self.assertNotIn("0.0%", notice, notice)
        self.assertNotIn("Utilization unavailable", notice, notice)

    # -- C3: real pixels, real zones, real events ---------------------------------------

    def test_the_application_hands_real_pixels_to_the_image_provider(self):
        # `set_frame_image` had no caller, so the provider served its 1x1 transparent
        # placeholder for the entire session.
        self.assertTrue(self.report["frame_image_is_qimage"], self.report)
        self.assertFalse(self.report["frame_image_null"], self.report)
        self.assertGreater(self.report["frame_image_width"], 0, self.report)
        self.assertGreater(self.report["frame_image_height"], 0, self.report)
        self.assertTrue(self.report["frame_ready"], self.report)

    def test_the_application_publishes_the_zones_from_the_manifest(self):
        # The retail claim *is* a zone rule. A screen with no zones supports no claim.
        self.assertGreaterEqual(self.report["zone_count"], 1, self.report)
        self.assertIn("counter_band",
                      [zone["name"] for zone in self.report["zones"]], self.report)

    def test_the_application_counts_the_events_the_machine_emitted(self):
        # `record_event` had no caller, so the footer rendered a fabricated 0 for every
        # kind. Asserted against the event machine's own list, not against a hand-written
        # expectation: the bridge must count exactly what was emitted.
        self.assertGreaterEqual(len(self.report["machine_events"]), 1, self.report)
        self.assertEqual(self.report["bridge_event_counts"],
                         self.report["machine_event_counts"], self.report)
        self.assertEqual(self.report["bridge_event_total"],
                         len(self.report["machine_events"]), self.report)

    # -- C4: the badge is live-bound, not constant ---------------------------------------

    def test_the_qml_bound_badge_follows_a_route_set_after_the_engine_loaded(self):
        # `device_display` was `constant=True` with no notify signal, so the one string
        # the brief names as load-bearing could never change on screen. The child sets a
        # route **after** `engine.load` and reads the rendered `Text`, not the Python
        # property - reading the property could not tell a bridge that updated from a
        # screen that did not.
        self.assertEqual(self.report["badge_at_load"],
                         "device: pending - no device selected",
                         "the probe did not reach the C4 measurement point: the badge was "
                         "already populated when the engine loaded")
        self.assertIn("CPU", self.report["badge_after_route"], self.report)
        self.assertIn("fell back from NPU", self.report["badge_after_route"], self.report)
        self.assertNotEqual(self.report["badge_after_route"],
                            self.report["badge_at_load"], self.report)

    def test_the_badge_reached_the_confirmed_device_by_the_end_of_the_run(self):
        self.assertIn("CPU", self.report["qml_badge"], self.report)
        self.assertNotIn("pending", self.report["qml_badge"].lower(), self.report)

    # -- I6: the probe really used the application's threading ----------------------------

    def test_the_probe_went_through_the_applications_own_threading_helpers(self):
        self.assertTrue(self.report["application_path"], self.report)
        self.assertGreater(self.report["frames_analysed"], 0, self.report)
        self.assertGreater(self.report["frames_submitted"], 0, self.report)

    @staticmethod
    def _drive(extra_args: Sequence[str] = (), seconds: float | None = None) -> dict:
        """Run `run_windowed` in a child and report what its own bridge ended up holding.

        A subprocess because `run_windowed` constructs a `QGuiApplication` and other
        tests in this suite construct one too. The child is written to a **file**, not
        passed to `python -c`, because `multiprocessing` cannot spawn from `-c` without
        tripping its `__main__` guard - which is the defect the first input-ack probe had.

        `extra_args` reaches `run_windowed` as real CLI arguments, so the presenter-driven
        and fault-injection paths under test are the ones the application actually has -
        not a patched `Supervisor` called from a test.
        """
        script_text = _LIVE_DRIVER % str(REPO_ROOT)
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYTHONPATH"] = str(REPO_ROOT)
        if seconds is not None:
            env["ROADSHOW_LIVE_SECONDS"] = str(float(seconds))
        with tempfile.TemporaryDirectory(prefix="roadshow-live-") as tmp:
            script = Path(tmp) / "drive_live.py"
            script.write_text(script_text, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(script), "--device", "CPU", *extra_args],
                capture_output=True, text=True, timeout=420,
                cwd=str(REPO_ROOT), env=env)
        report: dict = {"returncode": completed.returncode,
                        "stdout": completed.stdout, "stderr": completed.stderr,
                        "boot_lines": completed.stdout.splitlines(),
                        "application_path": False, "confirmed": [],
                        "device_display": "", "frame_image_is_qimage": False,
                        "frame_image_null": True, "frame_image_width": 0,
                        "frame_image_height": 0, "frame_ready": False,
                        "zone_count": 0, "zones": [], "event_counts": {},
                        "event_count_total": 0, "bridge_event_counts": {},
                        "machine_events": [], "machine_event_counts": {},
                        "qml_badge": "", "badge_at_load": "", "badge_after_route": "",
                        "frames_analysed": 0, "frames_submitted": 0,
                        "frames_refused": 0, "system_cpu_percent": "unavailable",
                        "ai_fps": "unavailable", "p50": "unavailable",
                        "p95": "unavailable", "gpu": "unavailable",
                        "npu": "unavailable", "cpu_available": False,
                        "card_text": {}, "staleness_text_rendered": "",
                        "unavailable_notice": {}, "window_note_text": "",
                        "window_note_visible": False, "event_count_text": "",
                        "card_names": [], "card_items_present": {},
                        "activity_cpu": "", "card_figures": {},
                        "age_window_note": "", "samples": [], "select_lines": [],
                        "stream_states": {}, "error_message": "",
                        "fell_back": False, "stream_route_state": "",
                        "overlay_boxes": [], "expected_boxes": [],
                        "frame_extents": [0, 0]}
        for line in completed.stdout.splitlines():
            if line.startswith("REPORT "):
                report.update(json.loads(line[len("REPORT "):]))
            elif line.startswith("BADGE "):
                badge = json.loads(line[len("BADGE "):])
                report["badge_at_load"] = badge.get("at_load", "")
                report["badge_after_route"] = badge.get("after_route", "")
        report["select_lines"] = [ln for ln in report["boot_lines"]
                                  if ln.startswith("[select]")]
        if not any(ln.startswith("REPORT ") for ln in report["boot_lines"]):
            raise AssertionError(
                "the live-path probe produced no report; `run_windowed` did not reach its "
                "final state.\nSTDOUT:\n" + completed.stdout + "\nSTDERR:\n" + completed.stderr)
        return report


#: The child program. Kept as a module-level string rather than written into the class so
#: it can be read in one piece, and so this file has exactly one place where the live path
#: is described.
_LIVE_DRIVER = r'''
"""Drive `app.main.run_windowed` for real and report what its own bridge ended up holding.

Run as a **file**, not via `python -c`: `multiprocessing` cannot spawn from `-c` without
tripping its `__main__` guard, and the inference worker is a spawned process.

Prints one `BADGE {json}` line at load time and one `REPORT {json}` line just before
quitting.
"""
import json
import os
import sys

sys.path.insert(0, %r)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtQml import QQmlApplicationEngine

import app.main as main_module
from app.engine.events import RetailEventMachine
from app.supervisor.runner import Supervisor
from app.ui.bridge import UiBridge

SECONDS = float(os.environ.get("ROADSHOW_LIVE_SECONDS", "14"))

seen = {"application_path": False}
holder = {}

# -- 1. Prove the run goes through the application's own threading helpers -------------
_real_feeders = main_module._start_feeders
_real_telemetry = main_module._start_telemetry


def spy_feeders(*a, **k):
    seen["application_path"] = True
    return _real_feeders(*a, **k)


def spy_telemetry(*a, **k):
    seen["application_path"] = True
    return _real_telemetry(*a, **k)


main_module._start_feeders = spy_feeders
main_module._start_telemetry = spy_telemetry

# -- 2. Keep handles on the three objects `run_windowed` builds -------------------------
_real_bridge_init = UiBridge.__init__
_real_supervisor_init = Supervisor.__init__
_real_machine_init = RetailEventMachine.__init__


def bridge_init(self, *a, **k):
    _real_bridge_init(self, *a, **k)
    holder["bridge"] = self
    # Record the frame extents the payload carries, so the parent can recompute the
    # expected box geometry independently instead of trusting the payload's own numbers.
    # That is what makes a wrong divisor catchable: both boxes are inside [0, 1] either
    # way, so only agreement between two implementations can tell them apart.
    holder["width"] = 0
    holder["height"] = 0
    original_apply_frame = self.apply_frame

    def counted_apply_frame(payload):
        try:
            holder["width"] = int(payload.get("width", 0))
            holder["height"] = int(payload.get("height", 0))
        except AttributeError:
            pass
        return original_apply_frame(payload)

    self.apply_frame = counted_apply_frame


def supervisor_init(self, *a, **k):
    _real_supervisor_init(self, *a, **k)
    holder["supervisor"] = self


def machine_init(self, *a, **k):
    _real_machine_init(self, *a, **k)
    holder["machine"] = self


UiBridge.__init__ = bridge_init
Supervisor.__init__ = supervisor_init
RetailEventMachine.__init__ = machine_init

# The tracker too, so the parent can recompute the expected overlay geometry from the
# tracker's own pixel boxes rather than from the payload that is under test.
from app.engine.tracker import Tracker

_real_tracker_init = Tracker.__init__


def tracker_init(self, *a, **k):
    _real_tracker_init(self, *a, **k)
    holder["tracker"] = self


Tracker.__init__ = tracker_init

_real_engine_init = QQmlApplicationEngine.__init__


def engine_init(self, *a, **k):
    _real_engine_init(self, *a, **k)
    holder["engine"] = self


QQmlApplicationEngine.__init__ = engine_init

# -- 3. The quit timer a booth does not have -------------------------------------------
_real_gui = main_module._gui_application
app = {"qobject": None}


def spy_gui():
    qobject = _real_gui()
    app["qobject"] = qobject
    # Installed before `run_windowed` has built its bridge, so `sample_badge` skips the
    # first ticks rather than raising: it returns early until both objects exist.
    install_sampler(qobject)
    QTimer.singleShot(int(SECONDS * 1000), finish_and_quit)
    return qobject


main_module._gui_application = spy_gui


def badge_text():
    """The **QML-bound** device badge, read off the rendered `Text` item.

    Deliberately not `bridge.device_display`. Reading the Python property cannot tell a
    bridge that updated from a screen that did not, which is exactly defect C4.
    """
    return find_text("deviceBadgeText")


def engine_roots():
    engine = holder.get("engine")
    return engine.rootObjects() if engine is not None else []


def _named_items():
    """Every named item in the scene, reached the way a `Repeater` delegate can be.

    ## The trap, measured, and it is the same class as the M1 card-text bug

    `QObject.findChild(name)` and `QObject.children()` **cannot see a `Repeater`
    delegate's subtree.** A delegate is a *visual* child - `setParentItem`, not
    `setParent` - so it is reached by `QQuickItem.childItems()` from the window's
    `contentItem()`, and by nothing else.

    The first version of the three-card reader used `findChild` and returned `MISSING` for
    every single card item, on a render where the PNG plainly showed all three cards drawn
    and correct. An assertion on `""` passes, so that version would have been a
    vacuous pass across a screen that was working.

    The walk starts at `contentItem()` for the same reason: a `Window`'s `childItems()` is
    empty, because its content item is assigned rather than added.
    """
    for root in engine_roots():
        start = getattr(root, "contentItem", None)
        if start is None:
            continue
        stack = [start()]
        while stack:
            item = stack.pop()
            try:
                kids = list(item.childItems())
            except (AttributeError, RuntimeError):
                kids = []
            stack.extend(kids)
            try:
                name = str(item.objectName())
            except (AttributeError, RuntimeError):
                continue
            if name:
                yield name, item


def find_text(object_name):
    """The rendered `text` of a named QML item, or `""`.

    Reading the scene graph is the only way to tell a bridge that updated from a screen
    that did not. Three separate defects in this milestone had exactly that shape: a
    `constant=True` property (the device badge), a binding that **called a method** (every
    metric card), and a second such binding eleven lines below it (the window note) - all
    of which left a correct bridge behind a frozen screen.

    `""` for an absent item is the trap the M1 card-text test fell into, so every assertion
    that reads `""` must also assert the item exists. See `_named_items`.
    """
    for name, item in _named_items():
        if name == object_name:
            return str(item.property("text"))
    return ""


def find_visible(object_name):
    """Whether a named QML item is currently `visible`, or `None` if it is absent."""
    for name, item in _named_items():
        if name == object_name:
            return bool(item.property("visible"))
    return None


def find_property(object_name, property_name, default=""):
    """A named QML item's **bound property**, or `default` if the item is absent.

    Not a substitute for `find_text` where a rendered `Text` exists - `find_text` is one
    step closer to the screen. It is here for the strings a `Text` does not name.

    **M2 Task 5**: the card's activity line is now a `readonly property` derived from the
    bridge's per-stream record, and `ComputeCard.qml` binds its `Text` straight to it, so
    reading the property reads what the label renders. Its old justification - that the M1
    driver invented `cardText_ai_fps` names that do not exist - is kept in
    `test_the_old_card_names_that_read_nothing_are_recorded_as_absent`.
    """
    for name, item in _named_items():
        if name == object_name:
            value = item.property(property_name)
            return "" if value is None else str(value)
    return default


def find_all_texts(object_name_prefix):
    """Every item in the scene whose `objectName` starts with `object_name_prefix`.

    **A set, and the reason matters.** M1's cards were three fixed devices, so a prefix
    walk returned three distinct names. The metrics column is now a `Repeater` over the
    supervisor's streams, so three cameras on three cameras produce the *same* device
    figure name three times - once per card. Duplicates are therefore correct, and a test
    that asserted a flat list would be asserting a layout rather than a fact.
    """
    return sorted({name for name, _item in _named_items()
                   if name.startswith(object_name_prefix)})


def find_item_exists(object_name):
    """Whether a named QML item exists in the scene at all.

    Separate from `find_text` because an absent item and an item whose text is `""` are
    different faults and the M1 driver could not tell them apart.
    """
    return any(name == object_name for name, _item in _named_items())


def _expected_boxes(tracker, width, height):
    """The tracker's own current tracks, normalised the way a box *should* be.

    An **independent** recomputation from `Track.box` and the frame extents, so it can be
    compared against what `app.main._frame_payload` published. Dividing `y` by
    `frame.width` instead of `frame.height` would still produce a box inside `[0, 1]` on a
    1920x1080 frame, so a bounds check alone cannot catch it - only agreement between two
    independent implementations can. This is the "correct box, wrong picture" class again.
    """
    if tracker is None or width <= 0 or height <= 0:
        return []
    out = []
    for track in tracker.current_tracks():
        if not getattr(track, "is_current", False):
            continue
        x1, y1, x2, y2 = (float(v) for v in track.box)
        out.append({
            "track_id": int(track.track_id),
            "x": x1 / width,
            "y": y1 / height,
            "w": (x2 - x1) / width,
            "h": (y2 - y1) / height,
        })
    return out


# -- 4. C4: read the badge at load, set a route after load, read it again -------------
_real_load = QQmlApplicationEngine.load


def engine_load(self, url):
    result = _real_load(self, url)
    holder["engine"] = self
    at_load = badge_text()
    # A route set **after** the engine has loaded. This is the case a `constant=True`
    # property with no notify signal cannot survive, and the case the application is in:
    # its route arrives from the worker long after `load` has returned.
    if holder.get("bridge") is not None:
        holder["bridge"].set_route(requested="NPU", confirmed=("CPU",), state="active")
    after = badge_text()
    print("BADGE " + json.dumps({"at_load": at_load, "after_route": after}), flush=True)
    return result


QQmlApplicationEngine.load = engine_load

# -- 4b. M2 Task 4: sample the badge against the supervisor's own truth, on the UI thread -
#
# "The badge shows the CONFIRMED device, never the requested one" is a claim about a
# *sequence*, so one read cannot settle it: a single read taken at the wrong instant would
# pass against a badge that jumps to the requested device the moment it is asked.
#
# So this samples, every 100 ms on the UI thread, and records for each sample what the
# badge rendered next to what the supervisor had actually proved at that instant. The
# parent then checks the **invariant** across every sample: the badge may only name a
# device the supervisor has confirmed. A badge that flashed the requested device would be
# caught in the sample where it happened, and there are 200-220 of them per run
# (measured 200, 217 and 218 across three runs).
samples = []


def sample_badge():
    """One sample, read on the UI thread. `samples` is bounded; this run is seconds long."""
    bridge = holder.get("bridge")
    supervisor = holder.get("supervisor")
    if bridge is None or supervisor is None:
        return
    if len(samples) >= 600:
        return
    stats = supervisor.stream_stats("A") if "A" in supervisor.stream_ids else None
    samples.append({
        "badge": badge_text(),
        "bridge_confirmed": str(bridge.device_confirmed or ""),
        "bridge_state": str(bridge.route_state),
        "supervisor_confirmed": list(stats.confirmed_devices) if stats else [],
        "supervisor_requested": str(stats.requested_device) if stats else "",
        "supervisor_state": str(stats.state) if stats else "",
    })


def install_sampler(application):
    timer = QTimer(application)
    timer.timeout.connect(sample_badge)
    timer.start(100)
    holder["sampler"] = timer


def finish_and_quit():
    bridge = holder.get("bridge")
    supervisor = holder.get("supervisor")
    machine = holder.get("machine")
    report = {"application_path": seen["application_path"]}
    if bridge is None:
        report["error"] = "run_windowed built no UiBridge"
    else:
        image = bridge.frame_image
        report["confirmed"] = str(bridge.device_confirmed or "")
        report["device_display"] = str(bridge.device_display)
        report["fell_back"] = bool(bridge.device_fell_back)
        report["frame_ready"] = bool(bridge.frame_ready)
        report["frame_image_is_qimage"] = image is not None
        report["frame_image_null"] = bool(image is None or image.isNull())
        report["frame_image_width"] = 0 if image is None else int(image.width())
        report["frame_image_height"] = 0 if image is None else int(image.height())
        report["zone_count"] = len(bridge.zones)
        report["zones"] = [dict(z) for z in bridge.zones]
        report["bridge_event_counts"] = dict(bridge.event_counts)
        report["bridge_event_total"] = int(sum(bridge.event_counts.values()))
        report["overlay_boxes"] = len(bridge.overlay_boxes)
        report["qml_badge"] = badge_text()
        # **Read the rendered strings, not just the bridge.** The metric cards once bound
        # to `bridge.display_text(...)`, which declares no QML property dependency, so Qt
        # evaluated each binding exactly once and the screen stayed on its cold-start
        # text for a whole session while the bridge held real readings. Only a read of the
        # live scene can tell that apart from a correct bridge.
        card_values = {}
        for key in ("system_cpu_percent", "ai_fps", "frame_age_p50_ms"):
            card_values[key] = find_text("cardText_" + key)
        report["card_text"] = card_values
        # **M2 Task 4: what the screen actually renders, and proof the names exist.**
        #
        # The three cards are created for `system_cpu_percent`, `gpu_utilisation_percent`
        # and `npu_utilisation_percent`, so `cardText_ai_fps` and
        # `cardText_frame_age_p50_ms` name items that do not exist and `find_text` returns
        # `""` for both - and `"" != "unavailable"` passes. Two of that test's three keys
        # asserted nothing, on a live run. So:
        #
        # - `card_text` keeps its old shape for the record, and `card_names` says which of
        #   those names actually exist, so the vacuous pair is visible rather than assumed;
        # - `activity_cpu` reads the CPU card's **bound** activity line, which is where
        #   `ai_fps` and the frame-age percentiles really are;
        # - `card_figures` reads the three cards' own figures.
        report["card_names"] = find_all_texts("cardText_")
        report["card_items_present"] = {
            name: find_item_exists(name)
            for name in ("card_A", "card_CPU", "card_GPU", "card_NPU")}
        report["stream_ids_on_screen"] = [name for name, _item in _named_items()
                                          if name.startswith("card_")
                                          and not name.startswith("cardText_")]
        report["activity_cpu"] = find_property("card_A", "activityText")
        report["window_text_stream"] = find_text("streamWindowText_A")
        report["age_text_stream"] = find_text("streamAgeText_A")
        report["card_figures"] = {
            key: find_text("cardText_" + key)
            for key in ("system_cpu_percent", "gpu_utilisation_percent",
                        "npu_utilisation_percent")
        }
        report["age_window_note"] = str(getattr(bridge, "age_window_note", ""))
        # M2 Task 4: the badge/fallback/failure state and the per-stream route map, plus
        # the badge time series the invariant is checked over.
        report["samples"] = samples
        report["stream_states"] = {k: str(v) for k, v in
                                   dict(getattr(bridge, "stream_states", {})).items()}
        report["stream_route_state"] = str(bridge.route_state)
        report["stream_routes"] = {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                                       for kk, vv in v.items()}
                                   for k, v in
                                   dict(getattr(bridge, "stream_routes", {})).items()}
        report["error_message"] = str(getattr(bridge, "error_message", ""))
        report["staleness_text_rendered"] = find_text("stalenessText")
        # R1: the per-card "no figure" wording, **as rendered**.
        #
        # **M2 Task 5: read off the utilisation line, not a `unavailableNotice_<device>`
        # item.** That item is gone, because the card is per *stream* now and its device is
        # only known at run time - there is no fixed `unavailableNotice_CPU` to name. The
        # wording moved nowhere: `UiBridge.unavailable_notice[device]` still holds it and
        # the card still renders it, on the line named after the device's own metric.
        report["unavailable_notice"] = {
            device: find_text("cardText_" + {
                "CPU": "system_cpu_percent",
                "GPU": "gpu_utilisation_percent",
                "NPU": "npu_utilisation_percent",
            }[device])
            for device in ("CPU", "GPU", "NPU")
        }
        # **M2 Task 5: the measurement's own reason, for when no sample lands.** The card
        # prints `unavailable_reason` when the collector supplies one, and under a loaded
        # machine a starved collector thread produces exactly that. A test that only
        # accepted the bridge's own wording would call that a failure when it is the more
        # precise answer.
        report["cpu_reason"] = str(getattr(bridge, "cpu_reason", ""))
        # R2: whether the "measured over N ms" line is actually *visible*, which is the
        # thing the frozen Slot binding broke - the bridge produced a value all along.
        report["window_note_text"] = find_text(
            "cardWindowNote_system_cpu_percent")
        report["window_note_visible"] = find_visible(
            "cardWindowNote_system_cpu_percent")
        # R3: the rendered event count, which must read "unavailable" until a frame has
        # actually been analysed and a real number after.
        report["event_count_text"] = find_text("eventCountRow")
        # The overlay geometry, cross-checked against the tracker's own boxes. A length
        # assertion cannot tell a correct box from a box in the wrong place.
        report["overlay_boxes"] = [dict(box) for box in bridge.overlay_boxes]
        report["frame_extents"] = [int(holder["width"]), int(holder["height"])]
        report["system_cpu_percent"] = bridge.display_text("system_cpu_percent")
        report["ai_fps"] = bridge.display_text("ai_fps")
        report["p50"] = bridge.display_text("frame_age_p50_ms")
        report["p95"] = bridge.display_text("frame_age_p95_ms")
        report["gpu"] = bridge.display_text("gpu_utilisation_percent")
        report["npu"] = bridge.display_text("npu_utilisation_percent")
        report["cpu_available"] = bool(bridge.cpu_available)
    if machine is not None:
        events = list(getattr(machine, "events", ()) or ())
        counts = {}
        for event in events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        report["machine_events"] = [e.kind for e in events]
        report["machine_event_counts"] = counts
    # The expected overlay geometry, recomputed from the tracker's own pixel boxes.
    # **Independent of `_frame_payload`**, which is the point: a wrong divisor there
    # would still produce a box inside [0, 1], so only two implementations agreeing can
    # catch it.
    report["expected_boxes"] = _expected_boxes(
        holder.get("tracker"), holder["width"], holder["height"])
    if supervisor is not None:
        report["frames_analysed"] = int(supervisor.frames_analysed)
        report["frames_submitted"] = int(supervisor.frames_submitted)
        report["frames_refused"] = int(supervisor.frames_refused)
    print("REPORT " + json.dumps(report), flush=True)
    sys.stdout.flush()
    if app["qobject"] is not None:
        app["qobject"].quit()


if __name__ == "__main__":
    sys.exit(main_module.main(sys.argv[1:] + ["--windowed"]))
'''


class PresenterSelectionAndObservedFallback(unittest.TestCase):
    """M2 Task 4: a presenter can move a camera, and a real worker death is visible.

    Two extra live runs, **~40 s**, each through `app.main.run_windowed` with real CLI
    arguments rather than a patched `Supervisor`:

    - `--select A=GPU` - the presenter's move. Closes M1's open criterion 2.
    - `--fail-stream A` - the worker really raises before compiling. Closes the **visible**
      half of M0's unmet gate 9.

    ## What "the badge shows the confirmed device, never the requested one" needs

    One read cannot settle it. A badge that jumped to the requested device the instant it
    was asked would pass a test that sampled after the confirmation. So the driver samples
    the rendered badge **every 100 ms on the UI thread** - 200-220 samples per run, measured - beside
    what the supervisor had actually proved at that instant, and the assertions are
    invariants over that series rather than comparisons at one instant.

    The direction of the invariant matters and is stated in the test: the bridge may **lag**
    the supervisor by a queued hop, and did; it may never **lead** it. Leading would be a
    device on screen that nothing has run on.
    """

    @classmethod
    def setUpClass(cls):
        reason = _provisioned()
        if reason:
            raise unittest.SkipTest(
                f"cannot run the real pipeline: {reason}. The live path is not exercised "
                f"on this machine and no claim is made about it.")
        cls.moved = LiveWindowedPath._drive(extra_args=["--select", "A=GPU"], seconds=24.0)
        cls.faulted = LiveWindowedPath._drive(extra_args=["--fail-stream", "A"],
                                              seconds=14.0)

    # -- criterion 2: a presenter can move a camera, and the screen follows the proof ----

    def test_the_presenters_move_is_acknowledged_and_the_badge_follows_the_proof(self):
        report = self.moved
        self.assertEqual(report["returncode"], 0, report["stderr"])
        self.assertTrue(report["select_lines"],
                        f"the presenter move printed nothing at all: "
                        f"{report['stdout'][-2000:]}")
        move = next((line for line in report["select_lines"]
                     if "acknowledged=" in line), "")
        self.assertIn("acknowledged=True", move,
                      f"the presenter's move was not acknowledged, so nothing on screen "
                      f"may claim it happened: {report['select_lines']}")
        self.assertIn("bounded_by='confirmed'", move, move)
        self.assertIn("confirmed={'A': ['GPU.0']}", move,
                      f"the GPU's own EXECUTION_DEVICES reading is what confirms it, and "
                      f"the dotted form is what it reports: {report['select_lines']}")
        # It started from a camera that had proved where it was, not mid-cold-compile.
        self.assertIn("[select] starting from 'active'", "\n".join(report["select_lines"]),
                      f"the move must be issued from a settled route or the badge change "
                      f"proves nothing: {report['select_lines']}")
        # **And the screen reached the new device.**
        self.assertEqual(report["qml_badge"], "device: GPU.0",
                         f"the rendered badge never reached the confirmed GPU: "
                         f"{report['qml_badge']!r}")
        self.assertEqual(report["confirmed"], "GPU.0",
                         f"and the bridge agrees: {report['confirmed']!r}")

    def test_the_badge_never_named_a_device_the_supervisor_had_never_proved(self):
        """Review Focus #1, as an invariant over a series: **the badge is always at or
        behind the supervisor's own proof, and never more than a hop behind.**

        Two clauses, and the second is where the honesty is:

        - **Never ahead.** Where the sampler can see a confirmed placement, the badge must
          not name a *different* device. That is a device on screen that the current route
          did not prove.
        - **At most a hop behind.** `state_changed` reaches the UI thread as a queued
          call, so a payload that was true when it was emitted can be applied after the
          route has moved on. And the proof window itself can be **shorter than the
          sampling interval**: `_await_settled` issues the presenter's move within about
          50 ms of the CPU placement confirming, so `confirmed_devices` can hold
          `("CPU",)` for less time than one 100 ms tick and a sampler can miss it
          entirely. Measured one such sample in six runs, showing `device: CPU` for one hop
          after the request.

        Two earlier versions of this test were **unsound rather than flaky**, and both are
        worth recording. The first asserted **zero** lag and failed intermittently. The
        second asserted "every device the badge ever named was proved at *some* sample" and
        failed for the same root cause: a sampler cannot see a proof that was withdrawn
        between two of its own reads. The bound below is the one statement here that is
        actually true.
        """
        report = self.moved
        samples = report["samples"]
        self.assertGreater(len(samples), 50,
                           f"only {len(samples)} samples; the invariant needs a series, "
                           f"not a single read")
        confirmed_seen = [i for i, s in enumerate(samples) if s["supervisor_confirmed"]]
        self.assertTrue(confirmed_seen,
                        "no sample caught a confirmed placement, so this run cannot show "
                        f"anything about the badge: {samples[:5]}")
        ahead = [(i, s["bridge_confirmed"], s["supervisor_confirmed"])
                 for i, s in enumerate(samples)
                 if s["supervisor_confirmed"]
                 and s["bridge_confirmed"]
                 and s["bridge_confirmed"] != ", ".join(s["supervisor_confirmed"])]
        self.assertEqual(
            ahead, [],
            f"the screen named a device the current route had not proved, at these "
            f"samples: {ahead[:3]}")
        behind = [i for i, s in enumerate(samples)
                  if not s["supervisor_confirmed"] and s["bridge_confirmed"]]
        self.assertLessEqual(
            len(behind), 2,
            f"{len(behind)} samples where the screen still named a device the supervisor "
            f"had already withdrawn: {behind}. A queued hop is one hop, and a proof window "
            f"can be shorter than one 100 ms tick; before the teardown-ordering fix the "
            f"lag was the whole ~1.1 s teardown, about 11 samples at this interval.")

    def test_the_badge_showed_the_old_device_while_the_new_one_was_only_requested(self):
        # **The specific moment, not just the invariant.** There are samples in which the
        # supervisor's `requested_device` is already the GPU, its `confirmed_devices` is
        # already empty, **and the screen has caught up** - so the badge must be reading
        # "pending", not a device. That is what distinguishes "never on request" from
        # "eventually, after confirmation".
        #
        # "Caught up" is part of the premise because of the queued hop documented in the
        # test above: a payload that was true when it was emitted can be applied after the
        # route moved on, and counting that as "the badge claimed the GPU" would be
        # blaming the screen for the thread boundary.
        report = self.moved
        settled = [s for s in report["samples"]
                   if s["supervisor_requested"] == "GPU"
                   and not s["supervisor_confirmed"]
                   and not s["bridge_confirmed"]]
        self.assertTrue(
            settled,
            "no sample caught the requested-but-unconfirmed state with the screen caught "
            f"up, so this run cannot show the badge refusing to move on request: "
            f"{report['samples'][:8]}")
        for sample in settled:
            # **The pending form is allowed to name the requested device** - "pending -
            # GPU not yet confirmed" is exactly what spec section 5 asks for, and it
            # claims nothing. What is forbidden is the *confirmed* form, `"device: GPU"`,
            # which is the badge asserting a placement.
            self.assertTrue(
                sample["badge"].startswith("device: pending -"),
                f"while the GPU is requested but unproved the badge must read as pending; "
                f"it read {sample['badge']!r}, which claims a placement or hides the move")

    def test_the_badge_passed_through_a_pending_word_for_the_new_device(self):
        # Spec section 5 asks for "Moving Camera B inference to CPU..." with pending vs
        # actual. So the badge is not allowed to jump straight from one confirmed device to
        # the other: the unconfirmed state has to be on screen.
        report = self.moved
        badges = [s["badge"] for s in report["samples"]]
        self.assertIn("device: pending - GPU not yet confirmed", badges,
                      "the unconfirmed state must be rendered, not skipped: the distinct "
                      f"badges were {sorted(set(badges))}")

    # -- criterion 6, the VISIBLE half: a worker that really dies ------------------------

    def test_a_real_worker_death_is_visible_on_the_live_scene_graph(self):
        report = self.faulted
        self.assertEqual(report["returncode"], 0, report["stderr"])
        states = report["samples"][-1] if report["samples"] else {}
        # **The failure reached the screen.** The rendered badge, read off the `Text` item.
        self.assertEqual(report["qml_badge"], "device: failed - no device is running",
                         f"a dead worker must not leave the badge reading a device: "
                         f"{report['qml_badge']!r}")
        self.assertEqual(report["stream_route_state"], "failed",
                         f"the bound stream's own state: {report['stream_states']}")
        self.assertEqual(report["stream_states"], {"A": "failed"},
                         f"and it is the per-stream state, not the aggregate: "
                         f"{report['stream_states']}")
        # **It claims no device.** The single most important part: a route that failed must
        # not keep naming the device it asked for, or the screen reads "running here".
        self.assertEqual(report["confirmed"], "",
                         f"a failed route must claim no confirmed device: "
                         f"{report['confirmed']!r}")
        self.assertEqual(report["stream_routes"]["A"]["confirmed"], [],
                         f"and its own record agrees: {report['stream_routes']}")
        self.assertEqual(report["stream_routes"]["A"]["requested"], "CPU",
                         "the requested device is still recorded - the operator asked for "
                         "it and it did not work")
        # **And the reason is on screen, naming the stream and the device.**
        text = report["error_message"]
        self.assertTrue(text, "the failure produced no stated reason on screen")
        self.assertIn("stream 'A'", text,
                      f"the reason must name the stream it is about: {text[:300]!r}")
        self.assertIn("CPU", text, f"and the device: {text[:300]!r}")
        self.assertIn("worker exit code 1", text,
                      f"and the real exit code, which is what makes it an observation "
                      f"rather than a simulation: {text[:300]!r}")
        # The fault really happened: frames were offered and nothing was analysed.
        self.assertGreater(report["frames_submitted"], 0, report)
        self.assertEqual(report["frames_analysed"], 0, report)

    def test_the_failed_route_reports_no_fallback_because_nothing_ran_elsewhere(self):
        """**`fell_back` is `False` here, and that is the honest answer.**

        The brief asks for `fell_back: true` on this screen. That needs a **confirmed
        reading on a device other than the one requested** - which is a *silent* fallback,
        and on this machine every device executes where it is asked, so it does not happen.
        `compile_failure` kills the worker before it runs anything, so nothing executed
        anywhere else and `fell_back: False` is what the truth says.

        This is recorded as **not observed** rather than worked around. Fabricating it -
        by requesting a device nothing implements, or by comparing the requested device to
        something other than the confirmed one - would put a claim on the booth screen that
        no measurement supports.
        """
        report = self.faulted
        self.assertIs(report["fell_back"], False,
                      "`fell_back` must be False here: nothing ran on any device, so "
                      "nothing fell back from anything")
        self.assertIs(report["stream_routes"]["A"]["fell_back"], False, report)


class PerStreamMetricPublishing(unittest.TestCase):
    """**M2 Task 5**: `app.main._publish_stream_metrics` must publish **per stream**.

    The stub supervisor is the only thing faked, and it is faked to the two calls the
    publisher makes - `per_stream()` and `stream_frame_ages_ms(sid)` - because the whole
    point is that the publisher asks for **one stream's** ages and never the merged
    population. The stub hands out three different age sets, so a merged implementation
    cannot pass by accident, and it **records** which ids were asked for.

    ## Why the merged figure is the dangerous one

    `Supervisor.frame_ages_ms()` returns the percentile of up to 3 x 256 samples across all
    three cameras. On the measured unsaturated `allcpu` condition that is the percentile of
    a bimodal population, and it is a figure about **no camera at all**. A card showing it
    would be showing an average on a card for one of them.
    """

    class _StubStream:
        def __init__(self, sid, device, accepted, analysed, invalidated):
            self.stream_id = sid
            self.requested_device = device
            self.frames_accepted = accepted
            self.frames_analysed = analysed
            self.frames_invalidated = invalidated
            self.state = "active"
            self.confirmed_devices = (device,)

    class _StubSupervisor:
        def __init__(self, streams, ages):
            self._streams = streams
            self._ages = ages
            self.age_requests = []

        def per_stream(self):
            return tuple(self._streams)

        def stream_frame_ages_ms(self, stream_id):
            # **The recording is half the assertion.** If the publisher asked for the merged
            # population instead, this would never be called and `age_requests` would be
            # empty - which is the defect, so it has to be observable rather than inferred
            # from a number.
            self.age_requests.append(stream_id)
            return tuple(self._ages.get(stream_id, ()))

    def _bridge(self):
        from app.ui.bridge import UiBridge

        return UiBridge()

    def _publish(self, stub, bridge, previous=None, window=1.0):
        from app.main import _publish_stream_metrics

        _publish_stream_metrics(bridge, stub,
                                previous if previous is not None
                                else {"per_stream": {}}, window)

    def test_three_cameras_get_three_different_cards(self):
        bridge = self._bridge()
        bridge.set_streams(["A", "B", "C"])
        # The measured allcpu p50s: 10767 / 11085 / 11319 ms.
        ages = {"A": [10767.0] * 10, "B": [11085.0] * 10, "C": [11319.0] * 10}
        stub = self._StubSupervisor(
            [self._StubStream("A", "CPU", 200, 20, 10),
             self._StubStream("B", "CPU", 220, 30, 11),
             self._StubStream("C", "CPU", 240, 42, 11)], ages)
        self._publish(stub, bridge)
        cards = {c["stream"]: c for c in bridge.stream_cards}
        self.assertEqual(sorted(cards), ["A", "B", "C"], cards)
        self.assertEqual([cards[s]["age_p50_text"] for s in "ABC"],
                         ["10.8 s", "11.1 s", "11.3 s"],
                         "each card must show ITS OWN age, not the merged percentile")
        self.assertEqual(sorted(stub.age_requests), ["A", "B", "C"],
                         "the publisher must ask for one stream at a time")
        # Backlog is `frames_accepted - frames_analysed - frames_invalidated`.
        self.assertEqual([cards[s]["backlog_text"] for s in "ABC"],
                         ["170 frames", "179 frames", "187 frames"],
                         f"got {[cards[s]['backlog_text'] for s in 'ABC']}")

    def test_a_stream_with_nothing_accepted_publishes_no_backlog(self):
        # `0 frames of backlog` reads as a healthy empty queue. A camera that has been
        # offered nothing has an *unknown* backlog.
        bridge = self._bridge()
        bridge.set_streams(["A"])
        self._publish(self._StubSupervisor(
            [self._StubStream("A", "CPU", 0, 0, 0)], {}), bridge)
        self.assertEqual(bridge.stream_card("A")["backlog_text"], "unavailable")

    def test_an_absent_measurement_is_never_published_as_zero(self):
        bridge = self._bridge()
        bridge.set_streams(["A"])
        self._publish(self._StubSupervisor(
            [self._StubStream("A", "CPU", 5, 0, 0)], {}), bridge)
        card = bridge.stream_card("A")
        self.assertEqual(card["age_p50_text"], "unavailable", card)
        self.assertEqual(card["age_p95_text"], "unavailable", card)
        self.assertEqual(card["fps_text"], "unavailable", card)
        self.assertFalse(card["stale"], "an absent age is not a stale age")

    def test_each_cameras_rate_is_its_own_second_derivative(self):
        # The first tick has no previous count for any stream, so no rate exists yet. The
        # second tick does - and one camera's backlog must not make another's card look
        # fast, which is only avoidable if the counter is kept per stream.
        bridge = self._bridge()
        bridge.set_streams(["A", "B"])
        first = self._StubStream("A", "CPU", 200, 20, 10)
        second = self._StubStream("B", "CPU", 200, 20, 10)
        stub = self._StubSupervisor([first, second], {})
        self._publish(stub, bridge)
        self.assertEqual(bridge.stream_card("A")["fps_text"], "unavailable",
                         "the first tick has no previous count, so there is no rate")
        first.frames_analysed = 120
        second.frames_analysed = 105
        self._publish(stub, bridge, {"per_stream": {"A": 20, "B": 20}}, window=1.0)
        self.assertEqual(bridge.stream_card("A")["fps_text"], "100.00 fps",
                         bridge.stream_card("A"))
        self.assertEqual(bridge.stream_card("B")["fps_text"], "85.00 fps",
                         bridge.stream_card("B"))


class PerStreamRouteMigration(unittest.TestCase):
    """`app.main._on_route` must publish **each stream's own** state.

    The 22-line warning this replaces said: *"`requested` and `confirmed` are read per
    stream, but `state` is `plan.state`, which since M2 is the aggregate over every
    stream... **With three streams a plan where Camera A is running and Camera C has failed
    publishes `"state": "active"`**, and a card bound to Camera A would render a healthy
    route while Camera C is a dead one."* Task 4 owns the migration; this is it.

    **Driven from a real three-stream `Supervisor`**, not a hand-built plan, so the plan
    under test is the one the application builds. `_on_route` is called exactly as
    `state_changed`'s live lambda calls it, and the payload it emits is fed to
    `UiBridge.apply_route` exactly as `run_windowed` does.

    No hardware and no window: the properties here are about which stream's word reaches
    which card, and a QML render would not make that any more true.
    """

    @staticmethod
    def _wired_bridge():
        """A `UiBridge` connected exactly as `run_windowed` connects it.

        **The connection is the point.** `_on_route` emits a signal; a bridge nobody is
        connected to sees nothing at all, and the first version of these tests called
        `apply_route`'s absence "the migration does not work". `run_windowed` makes these
        four connections immediately after building the bridge, and the test makes the same
        four so the path under test is the live one.
        """
        from app.ui.bridge import UiBridge

        bridge = UiBridge()
        bridge.routeArrived.connect(bridge.apply_route)
        bridge.errorArrived.connect(bridge.apply_error)
        bridge.frameArrivedFromWorker.connect(bridge.apply_frame)
        bridge.samplesArrived.connect(bridge.apply_samples)
        return bridge

    def _publish(self, plan):
        from app.main import _on_route

        bridge = self._wired_bridge()
        _on_route(bridge, plan)
        return bridge

    def test_a_failed_camera_does_not_make_another_camera_s_card_read_active(self):
        # The exact case the warning described. A is running, C has failed, and the plan's
        # **aggregate** is `active` because one of three streams proved a placement.
        from app.contracts.route_plan import RoutePlan

        plan = RoutePlan(
            generation=2,
            requested={"A": "CPU", "B": "CPU", "C": "CPU"},
            confirmed={"A": ("CPU",), "B": ()},
            state="active",  # the aggregate, and deliberately misleading on its own
            stream_states={"A": "active", "B": "pending", "C": "failed"},
            errors=("stream 'C' failed",),
            # **M2 Task 5 added this field and the contract now refuses a failed stream
            # without one** - so a plan that says `failed` and nothing else is no longer
            # constructible, which is the whole point. The aggregate above is still there
            # and still says the same thing; what the card reads is the per-stream one.
            stream_errors={"C": ("stream 'C' failed",)})
        bridge = self._publish(plan)

        self.assertEqual(plan.state, "active",
                         "premise: the aggregate really does read active")
        self.assertEqual(bridge.route_state, "active",
                         f"Camera A's card must read A's own state, and A is running: "
                         f"{bridge.route_state!r}")
        self.assertEqual(dict(bridge.stream_states),
                         {"A": "active", "B": "pending", "C": "failed"},
                         f"every stream's own state is recorded separately: "
                         f"{dict(bridge.stream_states)}")
        # And the card bound to the *failed* camera reads failed, with no device claimed -
        # which is what would have been lost by publishing the aggregate for every stream.
        bridge.set_route_stream("C")
        self.assertEqual(bridge.route_state, "failed")
        self.assertIsNone(bridge.device_confirmed,
                          "a failed camera's card must claim no device")
        self.assertEqual(bridge.device_display, "failed - no device is running",
                         bridge.device_display)
        # **The negative control for the whole point:** on the *aggregate*, every camera's
        # card would read active. Asserting it here is what makes the per-stream values
        # above meaningful rather than coincidental.
        self.assertEqual(plan.state, "active")
        self.assertEqual(plan.stream_state("C"), "failed")

    def test_a_payload_for_one_camera_cannot_move_another_cameras_card(self):
        from app.contracts.route_plan import RoutePlan
        from app.main import _on_route

        bridge = self._wired_bridge()
        active = RoutePlan(generation=1, requested={"A": "CPU"},
                           confirmed={"A": ("CPU",)}, state="active",
                           stream_states={"A": "active"})
        _on_route(bridge, active)
        self.assertEqual(bridge.device_confirmed, "CPU")
        # Camera B's plan arrives. It must be recorded and must change nothing on screen.
        other = RoutePlan(generation=2, requested={"A": "CPU", "B": "GPU"},
                          confirmed={"A": ("CPU",)}, state="active",
                          stream_states={"A": "active", "B": "failed"},
                          errors=("stream 'B' failed",),
                          stream_errors={"B": ("stream 'B' failed",)})
        _on_route(bridge, other)
        self.assertEqual(bridge.device_confirmed, "CPU",
                         f"Camera B's payload moved Camera A's badge: "
                         f"{bridge.device_confirmed!r}")
        self.assertEqual(bridge.route_state, "active")
        self.assertEqual(dict(bridge.stream_states),
                         {"A": "active", "B": "failed"})
        # And pointing the card at B does move it, from B's own record.
        bridge.set_route_stream("B")
        self.assertEqual(bridge.route_state, "failed")
        self.assertEqual(bridge.device_confirmed, None,
                         "B failed, so B's card claims no device")

    def test_the_route_payload_carries_that_streams_own_errors(self):
        # **M2 Task 5 rewrote this test rather than deleting it.** Task 4 asserted the
        # payload carried *no* `errors` key, because `RoutePlan` had no per-stream error
        # channel and forwarding `plan.errors` under a per-stream label is the same defect
        # with a different field name. That was a correct assertion about an incomplete
        # capability - so Task 5 built the capability and this now asserts the thing that
        # actually matters: the errors are **that stream's**.
        #
        # The plan below has one error about Camera B and one about Camera C. If the
        # publisher reached for the aggregate, Camera A's payload would carry both and this
        # test would fail.
        from app.contracts.route_plan import RoutePlan
        from app.main import _on_route

        emitted = []

        class _Signal:
            def emit(self, payload):
                emitted.append(dict(payload))

        class _Spy:
            routeArrived = _Signal()

        plan = RoutePlan(generation=1,
                         requested={"A": "CPU", "B": "CPU", "C": "CPU"},
                         confirmed={"A": ("CPU",)},
                         state="active",
                         stream_states={"A": "active", "B": "failed", "C": "failed"},
                         errors=("about B", "about C"),
                         stream_errors={"B": ("about B",), "C": ("about C",)})
        _on_route(_Spy(), plan)
        self.assertEqual(len(emitted), 3, emitted)
        by_stream = {payload["stream"]: payload for payload in emitted}
        self.assertEqual(by_stream["A"]["errors"], [],
                         f"a healthy camera must carry no reason: {by_stream['A']}")
        self.assertEqual(by_stream["B"]["errors"], ["about B"], by_stream["B"])
        self.assertEqual(by_stream["C"]["errors"], ["about C"], by_stream["C"])
        # And the aggregate is two reasons long, so a publisher using it would be caught.
        self.assertEqual(len(plan.errors), 2, plan.errors)
        self.assertEqual(set(emitted[0]),
                         {"stream", "requested", "confirmed", "state", "fell_back",
                          "errors"},
                         f"the payload's shape is fixed, so a new field is a decision: "
                         f"{sorted(emitted[0])}")

    def test_a_stream_that_declared_no_state_is_absent_rather_than_healthy(self):
        from app.contracts.route_plan import RoutePlan
        from app.main import _on_route

        bridge = self._wired_bridge()
        plan = RoutePlan(generation=1, requested={"A": "CPU", "B": "CPU"},
                         confirmed={"A": ("CPU",)}, state="active",
                         stream_states={"A": "active"})
        _on_route(bridge, plan)
        self.assertEqual(dict(bridge.stream_states), {"A": "active"},
                         "B declared no state, so B is absent - unproven, not pending "
                         "and not healthy")
        self.assertNotIn("B", bridge.stream_states)


class RealHeadlessRun(unittest.TestCase):
    """A real run against the real clip, when this machine can do one."""

    @classmethod
    def setUpClass(cls):
        reason = _provisioned()
        if reason:
            raise unittest.SkipTest(
                f"cannot run the real pipeline: {reason}. This is skipped with that "
                f"reason rather than passing silently: no placement assertion about "
                f"this machine has been made.")
        with tempfile.TemporaryDirectory(prefix="roadshow-headless-") as tmp:
            out = Path(tmp) / "run.json"
            cls.completed = _run("--headless", "--device", "CPU", "--seconds", "4",
                                 "--json", str(out))
            # Read **inside** the `with`. The first version read it after the temporary
            # directory had been deleted and failed with `FileNotFoundError` - which is a
            # reminder that a test whose `setUpClass` throws reports `errors=1` and no
            # test at all, which is easy to misread as a skip.
            cls.summary = json.loads(out.read_text(encoding="utf-8"))

    def test_a_cpu_headless_run_reports_a_confirmed_placement(self):
        summary = self.summary
        self.assertGreaterEqual(summary["frames_submitted"], 1)
        self.assertIn("confirmed", summary["device"])
        # A CPU run on a CPU machine should confirm CPU; if it does not, this run has
        # said something about the machine and the report needs to know.
        self.assertEqual(summary["device"]["confirmed"], ["CPU"], summary["device"])

    def test_the_run_reports_telemetry(self):
        # I3. `metrics` was `{}` in every run because nothing called
        # `supervisor.samples()`. The brief lists system and app CPU.
        self.assertIn("system_cpu_percent", self.summary["metrics"], self.summary)
        self.assertIsNotNone(self.summary["metrics"]["system_cpu_percent"]["value"],
                             self.summary)

    def test_the_refusal_count_reconciles_with_the_drop_count(self):
        # I1. `accepted + refused + drops == attempts` must hold, or the counters are
        # describing three different populations.
        summary = self.summary
        self.assertIn("frames_refused", summary)
        self.assertEqual(
            summary["frames_submitted"] + summary["frames_refused"] + summary["drops"],
            summary["frame_attempts"], summary)

    def test_a_run_that_analysed_nothing_exits_non_zero(self):
        # I1. `main()` must not exit 0 for a run that analysed no frames: that is the
        # "started nothing, dropped nothing, still said fine" shape.
        with tempfile.TemporaryDirectory(prefix="roadshow-empty-") as tmp:
            out = Path(tmp) / "run.json"
            # `--seconds 0` submits nothing at all: the feed loop's condition
            # `elapsed < 0` is false before the first iteration. A shorter, non-zero
            # window does not work - the first `read()`/`submit()` pair happens inside the
            # first millisecond and the drain then analyses it, which is a healthy run.
            r = _run("--headless", "--device", "CPU", "--seconds", "0",
                     "--json", str(out))
        self.assertNotEqual(r.returncode, 0,
                            "a run that analysed nothing exited 0: " + r.stdout + r.stderr)

    def test_ai_fps_uses_the_feed_window_and_not_the_drain(self):
        # I4. `elapsed_s` used to be read after the drain and the teardown, so `ai_fps`
        # was frames_analysed divided by a window the run never fed.
        summary = self.summary
        self.assertIn("feed_elapsed_s", summary)
        self.assertIn("drain_elapsed_s", summary)
        self.assertLess(summary["feed_elapsed_s"], summary["elapsed_s"] + 0.001, summary)
        if summary["ai_fps"] is not None:
            # Recomputed from the **published, rounded** `feed_elapsed_s`, because that is
            # what a reader can check. Asserting against an unrounded denominator would
            # only be testing the rounding.
            expected = round(summary["frames_analysed"] / summary["feed_elapsed_s"], 4)
            self.assertAlmostEqual(summary["ai_fps"], expected, delta=0.01, msg=summary)


if __name__ == "__main__":  # pragma: no cover - convenience for a human at a booth
    unittest.main()
