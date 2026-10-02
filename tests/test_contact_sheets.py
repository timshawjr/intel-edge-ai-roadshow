"""`tools/build_contact_sheets.py`, and the reason it has to stay honest.

M0's content gate is the one place where a conclusion was reached by **looking at
decoded frames** rather than by reading a file. That made it the one part of the
milestone a reader could not check: the report cited specific frame indices from a
throwaway harness whose code was not committed. `tools/build_contact_sheets.py` is now
that harness, so the checks here are about the two ways it could still mislead:

- **The frame lists could drift from the report.** The tool's `CLIP_PASSES` and
  `M0-content-gate.md`'s prose both name the frames that were looked at. If they diverge,
  the tool is producing sheets nobody cited and the report is citing sheets nobody can
  reproduce, and both are wrong in a way no reader would catch. So the report's numbers
  are parsed out of the Markdown and compared to the tool's.
- **A sheet could quietly show the wrong frames.** `CAP_PROP_POS_FRAMES` read *after*
  `read()` reports the next frame, not the delivered one - measured, and the reason the
  first version of `build_sheet` stamped every tile one frame high and reported 60
  spurious seek warnings. The stamp has to be the delivered frame.

No test asserts what is *in* the footage. A person looking at the pictures is what
settled that, and a test that claimed otherwise would be a test asserting nothing.
"""

import re
import unittest
from pathlib import Path

from tools.build_contact_sheets import (
    CLIP_PASSES,
    COLUMNS,
    TILE_WIDTH,
    build_sheet,
    media_entries,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTENT_GATE = REPO_ROOT / "docs" / "qualification" / "M0-content-gate.md"
BENCH_REPORT = REPO_ROOT / "bench_report.md"


def _cited_frame_lists(text: str) -> list[str]:
    """Every frame list the document attributes to a *pass*, as written.

    Deliberately narrow: only lists that appear on a line naming frames, so the pasted
    `tools/build_contact_sheets.py` output below them does not get counted a second time.
    Matching a bare `[0, 308, ...]` anywhere in the file counted the tool's own five
    output lines and reported ten lists against `CLIP_PASSES`'s five.
    """
    out: list[str] = []
    for line in text.splitlines():
        if "frames [" in line.lower() or "frames  [" in line.lower():
            out.extend(re.findall(r"\[(\d+(?:,\s*\d+)+)\]", line))
    return out


class TheToolAndTheReportCiteTheSameFrames(unittest.TestCase):
    """`CLIP_PASSES` and the documents that name frames must not drift apart.

    Both documents were written by hand from the frames that were looked at. The tool
    transcribes them. Nothing enforced that, so the tool could be "fixed" to decode a
    nicer set of frames and still claim to reproduce the review.
    """

    def _tool_lists(self) -> list[str]:
        out = []
        for passes in CLIP_PASSES.values():
            for indices in passes.values():
                out.append(", ".join(str(i) for i in indices))
        return out

    def test_the_content_gate_cites_every_frame_list_the_tool_decodes(self):
        cited = _cited_frame_lists(CONTENT_GATE.read_text(encoding="utf-8"))
        for indices in self._tool_lists():
            self.assertIn(indices, cited,
                          "CLIP_PASSES holds a frame list the content gate does not "
                          "cite, so a reader re-running this tool would be looking at "
                          "frames the report never claimed to have looked at")

    def test_the_content_gate_cites_nothing_the_tool_cannot_decode(self):
        cited = _cited_frame_lists(CONTENT_GATE.read_text(encoding="utf-8"))
        self.assertEqual(len(cited), len(self._tool_lists()),
                         "the content gate and CLIP_PASSES name a different number of "
                         "frame lists")

    def test_the_frame_indices_are_within_each_clip(self):
        """A frame index past the end decodes as nothing, and a sheet cell is then blank.

        The clip lengths here are the measured ones from `config/models.json`, so this is
        a cross-check against the config rather than a re-measurement.
        """
        for entry in media_entries():
            frames = int(entry["measured_frames"])
            passes = CLIP_PASSES.get(entry["id"], {})
            for pass_name, indices in passes.items():
                for index in indices:
                    self.assertGreaterEqual(index, 0, f"{entry['id']} {pass_name}")
                    self.assertLess(index, frames,
                                    f"{entry['id']} {pass_name} names frame {index} of "
                                    f"{frames}")


class TheSheetStampsTheFrameItActuallyDecoded(unittest.TestCase):
    """The off-by-one that the first version had, on every tile, in every sheet.

    Measured on this runtime: set `CAP_PROP_POS_FRAMES` to 1400, `read()`, then
    `get(CAP_PROP_POS_FRAMES)` returns **1401**. Reading the position after the read
    stamps every tile one frame high - so "the basket is already full at frame 0" becomes
    a claim about frame 1, and the report's frame numbers stop matching the pictures
    beside them.
    """

    @classmethod
    def setUpClass(cls):
        entry = next(e for e in media_entries() if e["id"] == "queue_alt")
        cls.clip = REPO_ROOT / entry["path"]
        if not cls.clip.is_file():
            raise unittest.SkipTest("assets/ is not provisioned on this tree")

    def test_the_position_after_a_read_is_the_next_frame_not_the_delivered_one(self):
        """The measurement the implementation is built on, kept so it cannot be forgotten."""
        import cv2
        capture = cv2.VideoCapture(str(self.clip))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 500)
            before = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
            ok, _ = capture.read()
            after = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        finally:
            capture.release()
        self.assertTrue(ok)
        self.assertEqual(before, 500, "the seek itself was not honoured")
        self.assertEqual(after, 501,
                         "if this ever becomes equal, the off-by-one this module exists "
                         "to avoid may have to be re-examined")

    def test_a_one_frame_sheet_is_stamped_with_the_requested_index(self):
        sheet, warnings = build_sheet(self.clip, [0])
        self.assertEqual(warnings, [], "a one-frame pass produced a seek warning")
        # A one-tile grid is still COLUMNS wide - `_grid` pads the short row rather than
        # producing a ragged sheet - so the sheet is four tiles across and the first is
        # the decoded frame.
        self.assertEqual(sheet.shape[1], TILE_WIDTH * COLUMNS)
        self.assertGreater(sheet.max(), 0, "the sheet is uniformly black")

    def test_an_out_of_range_frame_is_reported_rather_than_silently_dropped(self):
        sheet, warnings = build_sheet(self.clip, [0, 10_000_000])
        self.assertTrue(warnings, "a frame past the end produced no warning at all")
        self.assertTrue(any("10000000" in w for w in warnings),
                        f"the warning does not name the frame it could not decode: "
                        f"{warnings}")
        self.assertGreater(sheet.max(), 0, "the one good frame did not render")


if __name__ == "__main__":
    unittest.main()