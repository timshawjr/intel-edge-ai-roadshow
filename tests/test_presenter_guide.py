"""M6 Task 3: the presenter guide, checked against the CLI and the record.

A presenter guide is only worth the commands it prints. So this suite does not read the
guide as prose and nod at it:

1. **Every `python -m app.main` command in the guide is parsed by the real parser.**
   `build_parser()` is the thing `--help` prints, so a flag the parser does not know makes
   the test fail rather than the presenter at the booth.
2. **The guide covers every flag the brief names**, including the two that are easy to
   leave out because they are awkward: `--fail-stream` and `--no-release`.
3. **The guide repeats the code's own refusal.** `--streams` is refused in `--headless`
   with a stated reason; the test invokes the CLI and reads that refusal, and asserts the
   guide says the same thing.
4. **The three open gates are in the guide with their numbers** - content (9 of 12 clips,
   0 rights-cleared), legibility (passes at 2 m, fails at 3 m on the measured monitor) and
   the 2 s warm transition (p50 2.238 s, 37 of 50 over the target). A presenter who does
   not say these will be contradicted by the room.
5. **The NPU figure is cold versus settled**, and the guide does not claim a throughput
   ranking the trials do not support. The booth's 15 fps is set by the source.
6. **The retail event is people-in-zone presence, 2 per clip** - not a people count and
   not a dwell measurement.
"""

from __future__ import annotations

import contextlib
import io
import re
import unittest
from pathlib import Path

from app.main import build_parser, main as cli_main
from app.ui.bridge import RECORDED_DEMO_LABEL

REPO = Path(__file__).resolve().parents[1]
GUIDE = REPO / "docs" / "PRESENTER-GUIDE.md"

REQUIRED_FLAGS = (
    "--streams",
    "--select",
    "--fail-stream",
    "--windowed",
    "--headless",
    "--selftest",
    "--release",
    "--no-release",
    "--device",
    "--seconds",
    "--json",
    "--source",
    "--scenario",
)

#: A line in the guide that runs the application. The guide writes the venv interpreter
#: or plain `python`; both are the same command a presenter types.
_APP_COMMAND = re.compile(
    r"^(?:\s*(?:PS>\s*)?)(?:\.venv\\Scripts\\python\.exe|python)\s+-m\s+app\.main"
    r"(?P<args>[^\r\n`]*)",
    re.MULTILINE,
)


def _guide() -> str:
    if not GUIDE.is_file():
        raise AssertionError(
            f"{GUIDE.relative_to(REPO)} does not exist; the presenter guide is the "
            "deliverable this test exists to check"
        )
    return GUIDE.read_text(encoding="utf-8")


def _paragraphs(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def _paragraph_with(text: str, needle: str) -> str:
    for paragraph in _paragraphs(text):
        if needle in paragraph:
            return paragraph
    raise AssertionError(f"no paragraph of the guide contains {needle!r}")


class CommandsTheGuidePrintsExistInTheCli(unittest.TestCase):
    def test_every_app_main_command_parses_against_the_real_parser(self):
        text = _guide()
        commands = list(_APP_COMMAND.finditer(text))
        self.assertGreaterEqual(
            len(commands), 5,
            "the guide prints fewer than five application commands; it cannot be "
            "checked if it does not show the presenter what to type",
        )
        parser = build_parser()
        for match in commands:
            args = match.group("args").strip()
            with self.subTest(command=match.group(0).strip()):
                try:
                    parser.parse_args(args.split())
                except SystemExit as exc:  # argparse exits 2 on an unknown option
                    self.fail(
                        f"the guide prints `python -m app.main {args}`, which the real "
                        f"parser rejects (exit {exc.code}); --help is the contract"
                    )

    def test_the_guide_covers_every_flag_the_brief_names(self):
        text = _guide()
        missing = [flag for flag in REQUIRED_FLAGS if flag not in text]
        self.assertEqual(
            missing, [],
            f"the guide never mentions {missing}; a presenter cannot learn a flag that "
            "is not written down",
        )

    def test_the_working_compare_records_form_is_documented(self):
        text = _guide()
        self.assertIn(
            "python -m tools.compare_records", text,
            "the guide must print the invocation that runs: `python -m "
            "tools.compare_records`",
        )
        self.assertNotIn(
            "python tools/compare_records.py", text,
            "the tool's own docstring form omits that `tools/` is not on sys.path and "
            "fails with ModuleNotFoundError: No module named 'app'; the guide must not "
            "send a presenter down that path",
        )
        self.assertTrue(
            (REPO / "tools" / "compare_records.py").is_file(),
            "the module the guide names must exist",
        )

    def test_streams_in_headless_is_refused_and_the_guide_says_why(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = cli_main(["--headless", "--streams", "3"])
        message = stderr.getvalue()
        self.assertEqual(
            code, 2,
            f"the CLI accepted --headless --streams 3 (exit {code}); the guide's "
            "statement that it is refused would be false",
        )
        self.assertIn("not supported by --headless", message)
        self.assertIn("single-stream", message)

        text = _guide()
        self.assertIn(
            "not supported by --headless", text,
            "the guide must repeat the CLI's own refusal, not paraphrase it away",
        )
        self.assertIn(
            "single-stream", text,
            "the guide must give the reason: the headless summary is single-stream",
        )
        self.assertIn(
            "--windowed", _paragraph_with(text, "not supported by --headless"),
            "the refusal must point the presenter at --windowed for the multi-camera "
            "layout",
        )


class TheThreeOpenGatesAreInTheGuide(unittest.TestCase):
    def test_the_content_gate_is_nine_of_twelve_and_zero_rights_cleared(self):
        paragraph = _paragraph_with(_guide(), "9 of 12")
        self.assertIn("0 rights-cleared", paragraph)
        self.assertIn("shelf-to-basket", paragraph)
        self.assertIn("not provisioned", paragraph)

    def test_legibility_passes_at_two_metres_and_fails_at_three(self):
        text = _guide()
        paragraph = _paragraph_with(text, "11.2")
        self.assertIn("16.8", paragraph)
        self.assertIn("12.2", paragraph)
        self.assertIn("8.1", paragraph)
        self.assertIn("S3422DWG", paragraph)
        self.assertIn("3 m", paragraph)
        self.assertRegex(
            paragraph.lower(), r"fail",
            "the guide must say the 3 m case fails, not merely print an angle",
        )

    def test_the_two_second_transition_target_is_reported_as_missed(self):
        paragraph = _paragraph_with(_guide(), "2.238")
        self.assertIn("2.261", paragraph,
                      "the committed full-deadline file's p50 is 2.261 s; it must be visible")
        self.assertIn("2.628", paragraph,
                      "the committed 2 s-gate file's p50 is 2.628 s; the guide must not "
                      "quote only the flattering file")
        self.assertIn("37 of 50", paragraph)
        self.assertIn("2.0 s", paragraph)
        self.assertIn("missed", paragraph.lower())


class WhatTheGuideSaysAboutTheNumbers(unittest.TestCase):
    def test_the_npu_figure_is_cold_against_settled(self):
        paragraph = _paragraph_with(_guide(), "5.50")
        for number in ("5.57", "3.96", "4.01"):
            self.assertIn(number, paragraph,
                          f"the NPU paragraph must carry {number} ms")
        self.assertIn("cold", paragraph.lower())
        self.assertIn("settled", paragraph.lower())
        self.assertIn("5.2", paragraph)
        self.assertIn("5.9", paragraph,
                      "the guide must name the earlier 5.2-5.9 ms figure as cold")

    def test_no_throughput_ranking_the_trials_do_not_support(self):
        text = _guide()
        paragraph = _paragraph_with(text, "15 fps")
        self.assertIn("source", paragraph.lower())
        self.assertIn("ranks nothing", paragraph.lower())
        for number in ("307", "310", "246", "249"):
            self.assertIn(number, text,
                          f"the isolated-inference range must name {number} fps")
        for forbidden in (
            "npu is faster",
            "gpu is faster",
            "fastest device",
            "outperforms",
            "wins on throughput",
        ):
            self.assertNotIn(
                forbidden, text.lower(),
                f"the guide claims {forbidden!r}, which the paired trials do not support "
                "for the booth stream",
            )

    def test_the_retail_event_is_presence_not_a_count_or_a_dwell(self):
        text = _guide()
        self.assertIn("people-in-zone presence", text)
        self.assertIn("2 per clip", text)
        self.assertIn("not a people count", text)
        self.assertIn("not a dwell", text)


class TheBoothInstructionsMatchTheCode(unittest.TestCase):
    def test_fail_stream_is_a_qualification_aid_not_a_booth_feature(self):
        paragraph = _paragraph_with(_guide(), "--fail-stream")
        self.assertIn("QUALIFICATION AID", paragraph)
        self.assertIn("not a booth feature", paragraph)
        self.assertIn(
            "QUALIFICATION AID, not a booth feature",
            build_parser().format_help(),
            "the guide's wording must match what --help actually says",
        )

    def test_recorded_mode_carries_the_notice_and_disables_live_controls(self):
        text = _guide()
        self.assertIn(RECORDED_DEMO_LABEL, text,
                      "the guide must quote the notice exactly as the screen shows it")
        paragraph = _paragraph_with(text, RECORDED_DEMO_LABEL)
        self.assertIn("disabled", paragraph.lower())
        self.assertIn("--play-record", text)

    def test_the_guide_says_the_package_is_unsigned(self):
        text = _guide()
        self.assertIn("unsigned", text.lower())
        self.assertIn("docs/install/SIGNING.md", text,
                      "the guide must point at the signing verdict rather than imply "
                      "the package is signed")


if __name__ == "__main__":
    unittest.main()
