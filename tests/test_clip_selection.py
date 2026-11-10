"""`--clip` and `--pack`: one selection, used by headless and windowed alike.

The application used to name `RETAIL_CHECKOUT_CLIP` at three call sites, so eight of the
nine provisioned clips were unreachable. These tests pin the replacement:

- the default invocation still runs `retail-checkout` under `retail_sanity`;
- every catalogue id and every pack id is accepted by the real parser;
- one resolved selection is passed to `run_headless` and `run_windowed`;
- no `Path(RETAIL_CHECKOUT_CLIP)` remains on the run path;
- the headless summary names the pack and clip it actually ran;
- the UI's selection properties are notified and the QML binds to them, not to a literal.
"""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from app.engine.clips import SelectionError, default_selection, resolve_selection
from app.main import build_parser, build_run_summary, main as cli_main

REPO_ROOT = Path(__file__).resolve().parent.parent


class TheParserAcceptsEveryClipAndPack(unittest.TestCase):
    def test_the_default_invocation_names_no_clip_or_pack(self):
        args = build_parser().parse_args([])
        self.assertIsNone(args.clip)
        self.assertIsNone(args.pack)

    def test_every_catalogue_id_is_a_valid_clip_choice(self):
        from app.engine.clips import clip_ids

        ids = clip_ids()
        self.assertEqual(len(ids), 9)
        for clip_id in ids:
            args = build_parser().parse_args(["--clip", clip_id])
            self.assertEqual(args.clip, clip_id)

    def test_every_pack_id_is_a_valid_pack_choice(self):
        from app.engine.packs import pack_ids

        self.assertEqual(set(pack_ids()),
                         {"retail_sanity", "smart_city", "medical_ops",
                          "depot_logistics"})
        for pack_id in pack_ids():
            args = build_parser().parse_args(["--pack", pack_id])
            self.assertEqual(args.pack, pack_id)

    def test_an_unknown_clip_is_refused_by_the_parser(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--clip", "not-a-clip"])

    def test_an_unknown_pack_is_refused_by_the_parser(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--pack", "not_a_pack"])


class TheDefaultIsUnchanged(unittest.TestCase):
    def test_no_flags_still_resolves_to_the_m1_clip_and_pack(self):
        selection = default_selection()
        self.assertEqual(selection.clip.id, "retail-checkout")
        self.assertEqual(selection.pack_id, "retail_sanity")

    def test_headless_with_no_selection_receives_the_default(self):
        seen = {}

        def fake_headless(**kwargs):
            seen.update(kwargs)
            return {"frames_analysed": 1, "device": {"state": "active"},
                    "frames_submitted": 1, "frames_refused": 0,
                    "frames_refused_reasons": {}, "errors": []}

        stdout = io.StringIO()
        with mock.patch("app.main.run_headless", fake_headless), \
                contextlib.redirect_stdout(stdout):
            code = cli_main(["--headless", "--seconds", "0.1"])
        self.assertEqual(code, 0, stdout.getvalue())
        self.assertEqual(seen.get("clip"), "retail-checkout")
        self.assertEqual(seen.get("pack"), "retail_sanity")

    def test_windowed_with_no_selection_receives_the_default(self):
        seen = {}
        with mock.patch("app.main.run_windowed",
                        lambda **kwargs: seen.update(kwargs) or 0):
            code = cli_main(["--windowed"])
        self.assertEqual(code, 0)
        self.assertEqual(seen.get("clip"), "retail-checkout")
        self.assertEqual(seen.get("pack"), "retail_sanity")


class OneSelectionReachesBothRuns(unittest.TestCase):
    def test_a_clip_is_passed_to_headless_with_its_own_pack(self):
        seen = {}

        def fake_headless(**kwargs):
            seen.update(kwargs)
            return {"frames_analysed": 1, "device": {"state": "active"},
                    "frames_submitted": 1, "frames_refused": 0,
                    "frames_refused_reasons": {}, "errors": []}

        stdout = io.StringIO()
        with mock.patch("app.main.run_headless", fake_headless), \
                contextlib.redirect_stdout(stdout):
            code = cli_main(["--headless", "--seconds", "0.1",
                             "--clip", "car-detection"])
        self.assertEqual(code, 0, stdout.getvalue())
        self.assertEqual(seen.get("clip"), "car-detection")
        self.assertEqual(seen.get("pack"), "smart_city")

    def test_a_pack_is_passed_to_windowed_with_its_default_clip(self):
        seen = {}
        with mock.patch("app.main.run_windowed",
                        lambda **kwargs: seen.update(kwargs) or 0):
            code = cli_main(["--windowed", "--pack", "medical_ops"])
        self.assertEqual(code, 0)
        self.assertEqual(seen.get("clip"), "people-detection")
        self.assertEqual(seen.get("pack"), "medical_ops")

    def test_a_clip_and_its_pack_together_reach_windowed(self):
        seen = {}
        with mock.patch("app.main.run_windowed",
                        lambda **kwargs: seen.update(kwargs) or 0):
            code = cli_main(["--windowed", "--pack", "smart_city",
                             "--clip", "car-detection"])
        self.assertEqual(code, 0)
        self.assertEqual(seen.get("clip"), "car-detection")
        self.assertEqual(seen.get("pack"), "smart_city")

    def test_a_clip_and_a_different_pack_are_refused_before_any_run(self):
        started = []
        stderr = io.StringIO()
        with mock.patch("app.main.run_headless",
                        lambda **kwargs: started.append(kwargs)), \
                contextlib.redirect_stderr(stderr):
            code = cli_main(["--headless", "--clip", "retail-checkout",
                             "--pack", "smart_city"])
        self.assertEqual(started, [])
        self.assertEqual(code, 2)
        self.assertIn("belongs to pack", stderr.getvalue())

    def test_a_missing_selected_clip_is_refused_before_any_run(self):
        # A catalogued clip whose file is absent is a stated reason and exit 2, not a
        # `ClipSource` `FileNotFoundError` with a traceback.
        missing = REPO_ROOT / "assets" / "media" / "not-provisioned.mp4"
        self.assertFalse(missing.exists())
        with mock.patch("app.engine.clips.ClipRecord.exists", return_value=False), \
                mock.patch("app.main.run_headless",
                           lambda **kwargs: self.fail("a missing clip still ran")):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli_main(["--headless", "--clip", "classroom"])
        self.assertEqual(code, 2)
        self.assertIn("not provisioned", stderr.getvalue())

    def test_play_record_refuses_a_live_selection(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = cli_main(["--play-record", "run.json", "--clip", "classroom"])
        self.assertEqual(code, 2)
        self.assertIn("--clip", stderr.getvalue())


class NoHardcodedClipOnTheRunPath(unittest.TestCase):
    def test_the_run_path_does_not_build_a_clip_source_from_a_literal(self):
        source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines()
                         if not line.strip().startswith("#"))
        self.assertNotIn("Path(RETAIL_CHECKOUT_CLIP)", code)
        self.assertNotIn("from app.engine.retail_config import RETAIL_CHECKOUT_CLIP", code)
        # The two live `ClipSource` constructions must use the resolved selection.
        self.assertIn("ClipSource(selection.clip.absolute_path", code)


class TheSummaryNamesTheSelection(unittest.TestCase):
    def test_the_headless_summary_carries_pack_clip_and_licence(self):
        summary = build_run_summary(
            device_requested="CPU", state="active", confirmed=("CPU",),
            frames_submitted=1, frames_analysed=1, drops=0, elapsed_s=1.0,
            frame_ages_ms=(1.0,), events=(), samples=(),
            selection={"pack": "smart_city", "clip": "car-detection",
                       "license": "CC BY 4.0"},
        )
        self.assertEqual(summary["selection"]["pack"], "smart_city")
        self.assertEqual(summary["selection"]["clip"], "car-detection")
        self.assertEqual(summary["selection"]["license"], "CC BY 4.0")

    def test_a_selection_that_was_not_given_is_null(self):
        summary = build_run_summary(
            device_requested="CPU", state="pending", confirmed=(),
            frames_submitted=0, frames_analysed=0, drops=0, elapsed_s=0.0,
            frame_ages_ms=(), events=(), samples=(),
        )
        self.assertIsNone(summary["selection"])


class TheBridgeNamesTheRunningPackAndClip(unittest.TestCase):
    def test_a_fresh_bridge_names_the_default_pack_and_clip(self):
        from app.ui.bridge import UiBridge

        bridge = UiBridge()
        self.assertEqual(bridge.pack_id, "retail_sanity")
        self.assertEqual(bridge.clip_id, "retail-checkout")
        self.assertIn("retail_sanity", bridge.selection_label)
        self.assertIn("retail-checkout", bridge.selection_label)

    def test_set_selection_changes_every_string_together(self):
        from app.engine.packs import load_pack
        from app.ui.bridge import UiBridge

        bridge = UiBridge()
        pack = load_pack("medical_ops")
        bridge.set_selection(pack=pack, clip=pack.default_clip)
        self.assertEqual(bridge.pack_id, "medical_ops")
        self.assertEqual(bridge.clip_id, "people-detection")
        self.assertEqual(bridge.scope_note, pack.manifest.scope_note)
        self.assertEqual(bridge.delivered_event, pack.delivered_event)
        self.assertEqual(bridge.pack_explanation, pack.explanation)
        self.assertIn("medical_ops", bridge.selection_label)

    def test_a_clip_its_pack_does_not_contain_is_refused(self):
        from app.engine.packs import load_pack
        from app.ui.bridge import UiBridge

        bridge = UiBridge()
        pack = load_pack("depot_logistics")
        wrong = load_pack("smart_city").default_clip
        with self.assertRaises(ValueError):
            bridge.set_selection(pack=pack, clip=wrong)

    def test_the_qml_reads_the_notified_selection_properties(self):
        main_qml = (REPO_ROOT / "app" / "ui" / "Main.qml").read_text(encoding="utf-8")
        drawer = (REPO_ROOT / "app" / "ui" / "PresenterDrawer.qml").read_text(
            encoding="utf-8")
        self.assertIn("bridge.selection_label", drawer)
        self.assertIn("bridge.delivered_event", drawer)
        self.assertIn("bridge.scope_note", drawer)
        self.assertNotIn("bridge.selection_label", main_qml,
                         "the header has no room for a clip id at 44 px; the presenter "
                         "drawer is the surface that names the selection")
        self.assertNotIn("people-in-zone presence, 2 events per clip",
                         "\n".join(line for line in drawer.splitlines()
                                   if not line.strip().startswith("//")))


if __name__ == "__main__":
    unittest.main()
