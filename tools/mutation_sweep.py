"""`tools.mutation_sweep` - the M3 mutation sweep, as a committed, rerunnable artefact.

## Why this exists

M3 Tasks 1 and 2 both reported mutation sweeps in prose: "16 mutants, 15 killed, 1 survivor",
"24 mutants, 22 killed, 2 declared controls". **Neither sweep was committed**, so a reader had
to take the numbers on trust - and the same reports record how a sweep *lies*: "eleven of
sixteen anchors matched nothing", and "a skipped mutant looks exactly like a passing one unless
you read the summary". The prose names the trap and then rests its strongest claim on prose.

This tool is the remedy. It:

- carries the mutants as data (`MUTANTS`), each with the exact file, the exact `anchor` text and
  the exact `replacement`;
- **refuses an anchor that does not apply**: `anchor_text.count(anchor) != 1` is recorded as
  `unapplied`, and an unapplied anchor on a mutant that is **not a declared control** makes the
  process **exit non-zero**. A mutant that silently matched nothing can no longer look like a
  passing one;
- normalises CRLF to LF in memory before matching, because an editor rewrote six files with CRLF
  during Task 2 and every multi-line anchor then matched nothing;
- runs each mutant in its own process with `-B` and `PYTHONDONTWRITEBYTECODE=1`, and purges the
  mutated module's `__pycache__`: Python invalidates a `.pyc` on *(mtime, size)*, and a
  byte-identical restore leaves both unchanged, so a stale mutant value can be imported by the
  next process (measured: `CARD_REASON_CHARS` read `41` from a file that said `40`);
- writes a **per-mutant JSON** so the result is evidence rather than a sentence, and
- restores every file byte-for-byte, in a `finally`, so an interrupted sweep cannot leave the
  tree mutated.

## Declared controls

A control is a mutant that is **supposed to survive** - it edits prose, not behaviour - so its
survival is the sweep demonstrating that it is measuring the anchor and not flattering itself.
Controls are marked `control=True`. An unapplied anchor on a control is not a failure of the
code under test (there is nothing in it to change), but it is still reported.

## What `status` means

| status | meaning |
|---|---|
| `killed` | the test(s) failed with the mutant applied - the intended result |
| `survived` | the test(s) passed with the mutant applied - a test gap, or a control |
| `unapplied` | the anchor did not match exactly once; **nothing was mutated** |
| `skipped` | every selected test skipped itself (e.g. no provisioned assets) |
| `timeout` | the mutant hung past the per-mutant budget |

Exit code is non-zero if any non-control mutant is `unapplied`, `survived`, `timeout`, or
`skipped` without `requires_provisioned`.

## Usage

    .venv\\Scripts\\python.exe tools\\mutation_sweep.py
    .venv\\Scripts\\python.exe tools\\mutation_sweep.py --only T1.M13 --timeout 600
    .venv\\Scripts\\python.exe tools\\mutation_sweep.py --check-anchors
    .venv\\Scripts\\python.exe tools\\mutation_sweep.py --self-test-skipped-anchor
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT = (REPO_ROOT / ".superpowers" / "sdd"
               / "2026-10-01-m3-booth-screen-1920x1080" / "mutation-sweep.json")

#: Test id prefixes, so an entry reads as `SCENE.<name>` rather than a 90-character string.
SCENE = "tests.test_capture_ui.SceneReaderRules"
PROBE = "tests.test_capture_ui.RealQmlProbeScene"
BOOTH = "tests.test_capture_ui.BoothCapture"
SELF = "tests.test_capture_ui.HarnessSelfChecks"
LIVE = "tests.test_capture_ui.LiveCaptureOnTheRealApplication"
CANVAS = "tests.test_capture_ui_canvas.CanvasLayoutAtFourScales"
CUT = "tests.test_capture_ui_canvas.TheCutIsMadeInPythonNotByQt"
SEED = "tests.test_capture_ui_canvas.TheSeedToRouteRule"
CLIP = "tests.test_capture_ui_canvas.ClippingIsTheDecision"


@dataclasses.dataclass(frozen=True)
class Mutant:
    ident: str
    fault: str
    path: str
    anchor: str
    replacement: str
    tests: tuple[str, ...]
    control: bool = False
    requires_provisioned: bool = False


# --------------------------------------------------------------------------------------
# Task 1 - the capture harness (report table M1-M16)
# --------------------------------------------------------------------------------------

_T1 = (
    Mutant(
        "T1.M1", "the walk stops descending after the root",
        "tools/capture_ui.py",
        "            kids = list(item.childItems())",
        "            kids = []  # mutant",
        (f"{SCENE}.test_walk_items_yields_delegates_that_qobject_cannot_reach",)),
    Mutant(
        "T1.M2", "the clip is always the window, never an ancestor",
        "tools/capture_ui.py",
        "            if bool(ancestor.clip()):",
        "            if False:  # mutant",
        (f"{PROBE}.test_a_child_that_spills_past_its_clipping_panel_is_reported_as_overflow",)),
    Mutant(
        "T1.M3", "unusable geometry reads as a zero-sized box that fits",
        "tools/capture_ui.py",
        '    for value in values:\n'
        '        if value != value or value in (float("inf"), float("-inf")):\n'
        '            return None\n',
        '    for value in values:\n'
        '        if value != value or value in (float("inf"), float("-inf")):\n'
        '            return (0.0, 0.0, 0.0, 0.0)  # mutant\n',
        (f"{SCENE}.test_a_nan_height_is_geometry_unavailable_and_never_fits",)),
    Mutant(
        "T1.M4", "a reported string is cut to 40 characters",
        "tools/capture_ui.py",
        "            strings[name] = str(raw)",
        "            strings[name] = str(raw)[:40]  # mutant",
        (f"{SCENE}.test_a_truncated_string_is_reported_whole_and_also_named",)),
    Mutant(
        "T1.M5", "absent is reported even when the walk could not descend",
        "tools/capture_ui.py",
        "                   and walk_complete)",
        "                   )  # mutant",
        (f"{SCENE}.test_absent_empty_and_unreachable_are_three_different_answers",)),
    Mutant(
        "T1.M6", "a walk failure does not stop the walk claiming completeness",
        "tools/capture_ui.py",
        "    walk_complete = not failures",
        "    walk_complete = True  # mutant",
        (f"{SCENE}.test_absent_empty_and_unreachable_are_three_different_answers",)),
    Mutant(
        "T1.M7", "an item with no text property is reported as a string",
        "tools/capture_ui.py",
        "        if raw is not None:",
        "        if raw is not None or True:  # mutant",
        (f"{SCENE}.test_an_item_with_no_text_property_is_not_reported_as_an_empty_string",)),
    Mutant(
        "T1.M8", "the pixel size is computed from the window, not read off the image",
        "tools/capture_ui.py",
        '        "pixel_width": int(image.width()),\n'
        '        "pixel_height": int(image.height()),\n',
        '        "pixel_width": int(window.width()),  # mutant\n'
        '        "pixel_height": int(window.height()),\n',
        (f"{BOOTH}.test_the_png_is_the_requested_pixel_size_and_not_the_logical_size",)),
    Mutant(
        "T1.M9", "the read audit window is never closed",
        "tools/capture_ui.py",
        '        globals()["READ_WINDOW_CLOSED"] = True',
        "        pass  # mutant: the window is never marked closed",
        (f"{BOOTH}.test_the_strings_come_from_the_scene_and_not_from_the_qml_source",)),
    Mutant(
        "T1.M10", "a missing measurement falls back instead of raising",
        "tools/capture_ui.py",
        "    if not steady_path.is_file():",
        "    if False:  # mutant",
        (f"{SELF}.test_a_missing_measurement_raises_rather_than_defaulting",)),
    Mutant(
        "T1.M11", "any scale is accepted",
        "tools/capture_ui.py",
        "    if scale_percent not in SCALES:",
        "    if False:  # mutant",
        (f"{SELF}.test_a_fifth_scale_is_refused_rather_than_measured",)),
    Mutant(
        "T1.M12", "the overflow tolerance is large enough to hide every overflow",
        "tools/capture_ui.py",
        "TOLERANCE_PX = 0.5",
        "TOLERANCE_PX = 100000.0  # mutant",
        (f"{PROBE}.test_a_child_that_spills_past_its_clipping_panel_is_reported_as_overflow",)),
    Mutant(
        "T1.M13", "the live child loses its __main__ guard",
        "tools/capture_ui.py",
        'if __name__ == "__main__":\n    sys.addaudithook(_hook)',
        'if True:  # mutant: no __main__ guard\n    sys.addaudithook(_hook)',
        (f"{LIVE}.test_the_harness_reads_the_applications_own_scene_graph",),
        requires_provisioned=True),
    Mutant(
        "T1.M14", "the CLI exit code ignores an unaccounted item",
        "tools/capture_ui.py",
        "        if result.absent or result.unreachable or result.geometry_unavailable:",
        "        if False:  # mutant",
        (f"{SELF}.test_the_cli_fails_loudly_when_it_could_not_account_for_an_item",)),
    Mutant(
        "T1.M15", "a real elided Text is reported as not truncated",
        "tools/capture_ui.py",
        "        if flag is True:",
        "        if False:  # mutant",
        (f"{PROBE}.test_a_long_elided_text_is_named_and_reported_whole",)),
    Mutant(
        "T1.M16", "CONTROL: an unrelated comment is edited",
        "tools/capture_ui.py",
        "#: Slack when comparing an extent against a clip, in logical pixels.",
        "#: Slack used when comparing an extent against a clip, in logical pixels.",
        (f"{SCENE}.test_walk_items_yields_delegates_that_qobject_cannot_reach",),
        control=True),
)

# --------------------------------------------------------------------------------------
# Task 2 - the 1920x1080 layout (report table T2.1-T2.24)
# --------------------------------------------------------------------------------------

_T2 = (
    Mutant(
        "T2.1", "the card height ignores the Column's gaps again (the bug Task 2 fixed)",
        "app/ui/Main.qml",
        "- Theme.GAP * Math.max(0, bridge.stream_ids.length - 1))",
        "- 0 * Theme.GAP * Math.max(0, bridge.stream_ids.length - 1))  // mutant",
        (f"{CANVAS}.test_each_card_gets_the_band_less_the_gaps",)),
    Mutant(
        "T2.2", "the divisor uses the band height, not the camera count (a swapped divisor)",
        "app/ui/Main.qml",
        "/ Math.max(1, bridge.stream_ids.length))",
        "/ Math.max(1, parent.width))  // mutant",
        (f"{CANVAS}.test_each_card_gets_the_band_less_the_gaps",)),
    Mutant(
        "T2.3", "the card column is placed off the video column",
        "app/ui/Main.qml",
        "anchors.left: heroColumn.right",
        "anchors.left: parent.left  // mutant",
        (f"{CANVAS}.test_the_card_column_starts_where_the_video_column_ends",)),
    Mutant(
        "T2.4", "the video column's width fraction applied to the height (a swapped divisor)",
        "app/ui/Main.qml",
        "Math.round(parent.width * Theme.VIDEO_WIDTH_FRACTION)",
        "Math.round(parent.height * Theme.VIDEO_WIDTH_FRACTION)  // mutant",
        (f"{CANVAS}.test_the_card_column_starts_where_the_video_column_ends",)),
    Mutant(
        "T2.5", "the card stops clipping",
        "app/ui/ComputeCard.qml",
        "    clip: true\n    border.color: isFailed ? Theme.COLOR_BAD",
        "    clip: false  // mutant\n    border.color: isFailed ? Theme.COLOR_BAD",
        (f"{CLIP}.test_the_card_the_two_columns_and_the_tiles_all_clip",)),
    Mutant(
        "T2.6", "the right column stops clipping",
        "app/ui/ComputeCard.qml",
        "parent.width * Theme.CARD_COLUMN_SPLIT)\n"
        "        spacing: Theme.SPACING.xs\n"
        "        clip: true",
        "parent.width * Theme.CARD_COLUMN_SPLIT)\n"
        "        spacing: Theme.SPACING.xs\n"
        "        clip: false  // mutant",
        (f"{CLIP}.test_the_card_the_two_columns_and_the_tiles_all_clip",)),
    Mutant(
        "T2.7", "the tile stops clipping",
        "app/ui/VideoTile.qml",
        "    clip: true",
        "    clip: false  // mutant",
        (f"{CLIP}.test_the_card_the_two_columns_and_the_tiles_all_clip",)),
    Mutant(
        "T2.8", "the columns lose their names",
        "app/ui/ComputeCard.qml",
        'objectName: "cardLeftColumn_" + card.streamId',
        'objectName: ""  // mutant',
        (f"{CLIP}.test_the_card_the_two_columns_and_the_tiles_all_clip",)),
    Mutant(
        "T2.9", "the reason is not cut at all",
        "app/ui/bridge.py",
        "            if len(reason) > budget:",
        "            if False:  # mutant",
        (f"{CUT}.test_a_long_reason_is_cut_with_the_number_that_is_missing",)),
    Mutant(
        "T2.10", "the marker loses the count",
        "app/theme.py",
        'CARD_REASON_MARKER: Final[str] = "... +{hidden} more"',
        'CARD_REASON_MARKER: Final[str] = "... +more"  # mutant',
        (f"{CANVAS}.test_the_failed_cameras_reason_says_how_much_it_is_not_showing",)),
    Mutant(
        "T2.11", "the cut is one character too generous (40 -> 41)",
        "app/theme.py",
        "CARD_REASON_CHARS: Final[int] = 40",
        "CARD_REASON_CHARS: Final[int] = 41  # mutant",
        (f"{CUT}.test_a_reason_exactly_at_the_budget_is_not_marked_as_cut",)),
    Mutant(
        "T2.12", "the card shows the complete reason again",
        "app/ui/ComputeCard.qml",
        "            text: card.record.reason_display",
        "            text: card.record.reason  // mutant",
        (f"{CANVAS}.test_the_failed_cameras_reason_says_how_much_it_is_not_showing",)),
    Mutant(
        "T2.13", "the staleness sentence is shrunk (the exact false pass)",
        "app/ui/ComputeCard.qml",
        '            objectName: "streamStaleText_" + card.streamId\n'
        '            visible: card.record.staleness !== ""\n'
        "            text: card.record.staleness\n"
        "            color: Theme.COLOR_WARN\n"
        "            font.family: Theme.FONT_FAMILY\n"
        "            font.pixelSize: Theme.TYPE_DETAIL",
        '            objectName: "streamStaleText_" + card.streamId\n'
        '            visible: card.record.staleness !== ""\n'
        "            text: card.record.staleness\n"
        "            color: Theme.COLOR_WARN\n"
        "            font.family: Theme.FONT_FAMILY\n"
        "            font.pixelSize: Theme.TYPE_DETAIL - 4  // mutant",
        (f"{CANVAS}.test_no_font_size_on_the_screen_changed",)),
    Mutant(
        "T2.14", "the staleness line is hidden",
        "app/ui/ComputeCard.qml",
        '            visible: card.record.staleness !== ""',
        "            visible: false  // mutant",
        (f"{CANVAS}.test_the_staleness_line_is_present_and_visible_at_every_scale",)),
    Mutant(
        "T2.15", "the staleness sentence stops naming the age",
        "app/ui/bridge.py",
        '            staleness = (f"STALE - this picture is {_age(age_p50)} old; "',
        '            staleness = ("STALE - this picture is old; "  # mutant',
        (f"{CANVAS}.test_the_staleness_line_is_present_and_visible_at_every_scale",)),
    Mutant(
        "T2.16", "the column split goes back inline",
        "app/ui/ComputeCard.qml",
        "width: Math.max(Theme.TYPE_NUMERAL_SMALL * Theme.CARD_COLUMN_MIN_EM,",
        "width: Math.max(Theme.TYPE_NUMERAL_SMALL * 4,  // mutant",
        (f"{CLIP}.test_the_layout_tokens_are_in_theme_and_not_in_the_qml",)),
    Mutant(
        "T2.17", "the activity line's budget goes back inline",
        "app/ui/ComputeCard.qml",
        "            maximumLineCount: Theme.CARD_ACTIVITY_LINES",
        "            maximumLineCount: 2  // mutant",
        (f"{CLIP}.test_the_layout_tokens_are_in_theme_and_not_in_the_qml",)),
    Mutant(
        "T2.18", "CONTROL: a comment in the card is edited",
        "app/ui/ComputeCard.qml",
        "    // **The card clips. M3 Task 2's `clip: true` decision.**",
        "    // **The card clips. M3 Task 2 made that decision.**",
        (f"{CLIP}.test_the_card_the_two_columns_and_the_tiles_all_clip",),
        control=True),
    Mutant(
        "T2.19", "CONTROL: a comment in the canvas test file is reworded",
        "tests/test_capture_ui_canvas.py",
        '        """**The fix this task made.**',
        '        """**The fix this task introduced.**',
        (f"{CLIP}.test_the_card_the_two_columns_and_the_tiles_all_clip",),
        control=True),
    Mutant(
        "T2.20", "the failed camera is rendered active with no device (the bug Task 2 shipped)",
        "tools/capture_ui.py",
        '            "state": entry.get("seed_state", "failed" if failed else entry["state"]),',
        '            "state": entry.get("state", "failed"),  # mutant',
        (f"{SEED}.test_the_default_seed_fails_exactly_one_camera",)),
    Mutant(
        "T2.21", "the failed camera keeps its confirmed device",
        "tools/capture_ui.py",
        '        if failed and not entry.get("keep_confirmed"):',
        "        if False:  # mutant",
        (f"{SEED}.test_the_failed_camera_claims_no_device_and_carries_the_committed_reason",)),
    Mutant(
        "T2.22", "the failed camera's stated reason is dropped",
        "tools/capture_ui.py",
        "        if failed and not errors:",
        "        if False:  # mutant",
        (f"{SEED}.test_the_failed_camera_claims_no_device_and_carries_the_committed_reason",)),
    Mutant(
        "T2.23", "the child no longer pastes the shared seed-to-route rule",
        "tools/capture_ui.py",
        "''' + _ROUTES_FROM_SEED_SOURCE + r'''",
        "''' + r'''  # mutant: the child drops the shared rule",
        (f"{SEED}.test_the_child_and_the_module_share_one_definition",)),
    Mutant(
        "T2.24", "the reason on a camera that has a device is dropped (the six-row case degrades)",
        "tools/capture_ui.py",
        '        errors = list(entry.get("errors") or [])',
        "        errors = []  # mutant",
        (f"{SEED}.test_the_worst_case_adds_a_reason_to_a_camera_that_has_a_device",)),
)

# --------------------------------------------------------------------------------------
# The review fixes - C1 (glyph pixels), I1 (caption), I2 (expect_nonempty), I4 (fields)
# --------------------------------------------------------------------------------------

_FIX = (
    Mutant(
        "FIX.C1", "the glyph box is indexed with logical extents against a device image",
        "tools/capture_ui.py",
        "            x0, y0 = box[0] * scale_x, box[1] * scale_y\n"
        "            x1, y1 = (box[0] + box[2]) * scale_x, (box[1] + box[3]) * scale_y",
        "            x0, y0 = box[0], box[1]  # mutant\n"
        "            x1, y1 = box[0] + box[2], box[1] + box[3]",
        (f"{BOOTH}.test_every_expected_name_has_painted_pixels_at_every_scale",)),
    Mutant(
        "FIX.C1b", "the reported glyph_scale is hardcoded to 1.0",
        "tools/capture_ui.py",
        '        evidence["glyph_scale"] = [scale_x, scale_y]',
        '        evidence["glyph_scale"] = [1.0, 1.0]  # mutant',
        (f"{BOOTH}.test_the_pixel_count_in_the_json_matches_the_png_it_claims_to_describe",)),
    Mutant(
        "FIX.I1", "the thumbnail caption loses its objectName",
        "app/ui/VideoTile.qml",
        '            objectName: "tileCaptionText_" + tile.streamId',
        '            objectName: ""  // mutant',
        (f"{SELF}.test_the_caption_the_instrument_checks_is_named_in_the_qml",)),
    Mutant(
        "FIX.I1b", "the caption regresses to the M2 -1 (stream_confirmed_labels)",
        "app/ui/bridge.py",
        '", ".join(str(d) for d in (entry.get("confirmed") or ()))',
        '"-1"  # mutant: the M2 QVariantList stringification',
        (f"{BOOTH}.test_every_camera_caption_is_named_read_and_correct",)),
    Mutant(
        "FIX.I2", "readiness ignores expect_nonempty and settles on presence alone",
        "tools/capture_ui.py",
        '        if not state["missed"] and not state["empty"]:',
        '        if not state["missed"]:  # mutant',
        (f"{SELF}.test_a_required_nonempty_string_that_is_empty_stops_the_capture",)),
    Mutant(
        "FIX.I2b", "settle_polls is not carried out of the child",
        "tools/capture_ui.py",
        '        "settle_polls": state["polls"],',
        '        "settle_polls": 0,  # mutant',
        (f"{BOOTH}.test_the_record_carries_the_fourth_per_field_state_and_the_walk_failures",)),
    Mutant(
        "FIX.I4", "text_property is dropped from the record",
        "tools/capture_ui.py",
        '            "strings": dict(self.strings),\n'
        '            "text_property": dict(self.text_property),\n'
        '            "font_pixels": dict(self.font_pixels),',
        '            "strings": dict(self.strings),\n'
        '            "font_pixels": dict(self.font_pixels),',
        (f"{BOOTH}.test_the_committed_records_are_what_this_code_produces_today",)),
    Mutant(
        "FIX.I4b", "walk_failures is dropped from the record",
        "tools/capture_ui.py",
        '            "extents": {k: list(v) for k, v in self.extents.items()},\n'
        '            "walk_complete": self.walk_complete,\n'
        '            "walk_failures": list(self.walk_failures),',
        '            "extents": {k: list(v) for k, v in self.extents.items()},\n'
        '            "walk_complete": self.walk_complete,',
        (f"{BOOTH}.test_the_committed_records_are_what_this_code_produces_today",)),
    Mutant(
        "FIX.audit", "the reader opens a file inside the read window and the audit must say so",
        "tools/capture_ui.py",
        "        return harness.read_scene(window.contentItem(), window_rect=window_rect,",
        "        open(__file__).close()  # mutant: an open inside the read window\n"
        "        return harness.read_scene(window.contentItem(), window_rect=window_rect,",
        (f"{BOOTH}.test_the_strings_come_from_the_scene_and_not_from_the_qml_source",)),
)

MUTANTS: tuple[Mutant, ...] = tuple(_T1 + _T2 + _FIX)


# --------------------------------------------------------------------------------------
# Running one mutant
# --------------------------------------------------------------------------------------

def _purge_pycache(path: pathlib.Path) -> None:
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_mutant(mutant: Mutant, *, timeout: float) -> dict:
    """Apply, test, restore. Returns a record; never leaves the tree mutated."""
    path = REPO_ROOT / mutant.path
    original = path.read_bytes()
    original_text = original.decode("utf-8")
    normalized = original_text.replace("\r\n", "\n")
    matches = normalized.count(mutant.anchor)
    record = {
        "ident": mutant.ident,
        "fault": mutant.fault,
        "path": mutant.path,
        "control": mutant.control,
        "requires_provisioned": mutant.requires_provisioned,
        "tests": list(mutant.tests),
        "anchor_matches": matches,
        "file_sha256_before": _sha256(path),
        "status": None,
        "seconds": None,
        "detail": "",
    }
    if matches != 1:
        record["status"] = "unapplied"
        record["detail"] = (f"the anchor matched {matches} time(s), not exactly once. "
                            f"Nothing was mutated. A skipped mutant looks like a passing one.")
        return record

    mutated = normalized.replace(mutant.anchor, mutant.replacement, 1)
    if mutated == normalized:
        record["status"] = "unapplied"
        record["detail"] = "the replacement is identical to the anchor; nothing changed"
        return record

    started = time.perf_counter()
    try:
        path.write_text(mutated, encoding="utf-8", newline="\n")
        _purge_pycache(path)
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["QT_QPA_PLATFORM"] = "offscreen"
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "unittest", *mutant.tests],
            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=timeout)
        combined = completed.stdout + completed.stderr
        if completed.returncode != 0:
            record["status"] = "killed"
        elif "OK (skipped=" in combined:
            record["status"] = "skipped"
            record["detail"] = "every selected test skipped itself"
        elif "OK" in combined:
            record["status"] = "survived"
            record["detail"] = "the test(s) passed with the mutant applied"
        else:
            record["status"] = "survived"
            record["detail"] = "no recognised unittest summary"
        record["detail"] = (record["detail"] or "") + " | " + combined.strip().splitlines()[-1] \
            if combined.strip() else record["detail"]
    except subprocess.TimeoutExpired:
        record["status"] = "timeout"
        record["detail"] = f"the mutant exceeded {timeout:.0f}s"
    finally:
        path.write_bytes(original)
        _purge_pycache(path)
    record["seconds"] = round(time.perf_counter() - started, 3)
    return record


def check_anchors(mutants: tuple[Mutant, ...]) -> list[dict]:
    """Report, without mutating, how many times each anchor matches its file."""
    out = []
    for mutant in mutants:
        path = REPO_ROOT / mutant.path
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        out.append({"ident": mutant.ident, "path": mutant.path, "control": mutant.control,
                    "anchor_matches": text.count(mutant.anchor)})
    return out


def _exit_code(records: list[dict]) -> int:
    failures = []
    for rec in records:
        if rec["control"]:
            # A control that is *killed* means the anchor is doing something real; that is not
            # a failure of the code, it is a failure of the control. An unapplied control is
            # permitted by the tool's contract, and reported.
            if rec["status"] == "killed":
                failures.append(rec["ident"] + " (control was killed)")
            continue
        if rec["status"] in ("unapplied", "survived", "timeout"):
            failures.append(rec["ident"] + f" ({rec['status']})")
        elif rec["status"] == "skipped" and not rec["requires_provisioned"]:
            failures.append(rec["ident"] + " (unexpected skip)")
    return 1 if failures else 0


def _print_records(records: list[dict]) -> None:
    print(f"{'mutant':10} {'status':10} {'s':>7}  fault")
    for rec in records:
        print(f"{rec['ident']:10} {rec['status']:10} "
              f"{(rec['seconds'] if rec['seconds'] is not None else 0):7.2f}  {rec['fault']}")
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    print("totals:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the JSON")
    parser.add_argument("--only", action="append", default=[],
                        help="run only this mutant id (repeatable)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-mutant wall-clock budget in seconds")
    parser.add_argument("--check-anchors", action="store_true",
                        help="match every anchor against its file and exit; mutate nothing")
    parser.add_argument("--list", action="store_true", help="list the mutants and exit")
    parser.add_argument("--self-test-skipped-anchor", action="store_true",
                        help="prove the exit code is non-zero when a non-control anchor does "
                             "not apply")
    args = parser.parse_args(argv)

    if args.list:
        for mutant in MUTANTS:
            print(f"{mutant.ident:10} {'control' if mutant.control else 'mutant ':8} "
                  f"{mutant.path}: {mutant.fault}")
        return 0

    if args.self_test_skipped_anchor:
        bad = Mutant("SELFTEST.bad-anchor", "an anchor that cannot match anything",
                     "tools/capture_ui.py", "\nTHIS ANCHOR DOES NOT EXIST IN THE FILE\n",
                     "unreachable", ("tests.test_capture_ui.SceneReaderRules"
                                     ".test_walk_items_yields_delegates_that_qobject_cannot_reach",))
        record = run_mutant(bad, timeout=args.timeout)
        _print_records([record])
        code = _exit_code([record])
        print(f"self-test exit code (non-zero is the pass): {code}")
        return code

    if args.check_anchors:
        results = check_anchors(MUTANTS)
        bad = [r for r in results
               if r["anchor_matches"] != 1 and not r["control"]]
        controls = [r for r in results if r["anchor_matches"] != 1 and r["control"]]
        for r in results:
            flag = "" if r["anchor_matches"] == 1 else "  <-- NOT APPLIED"
            print(f"{r['ident']:10} {r['anchor_matches']:>3}  {r['path']}{flag}")
        print(f"{len(results) - len(bad) - len(controls)} of {len(results)} anchors apply exactly "
              f"once; {len(bad)} non-control anchor(s) do not; {len(controls)} control anchor(s) "
              f"do not.")
        return 1 if bad else 0

    selected = MUTANTS
    if args.only:
        wanted = set(args.only)
        selected = tuple(m for m in MUTANTS if m.ident in wanted)
        missing = wanted - {m.ident for m in selected}
        if missing:
            print(f"unknown mutant id(s): {sorted(missing)}", file=sys.stderr)
            return 2

    records = []
    for mutant in selected:
        record = run_mutant(mutant, timeout=args.timeout)
        records.append(record)
        print(f"{record['ident']:10} {record['status']:10} "
              f"{(record['seconds'] if record['seconds'] is not None else 0):7.2f}s  "
              f"{record['fault']}")
        sys.stdout.flush()

    code = _exit_code(records)
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        head = ""
    payload = {
        "tool": "tools/mutation_sweep.py",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "repo_head": head,
        "exit_code": code,
        "totals": {s: sum(1 for r in records if r["status"] == s)
                   for s in sorted({r["status"] for r in records})},
        "mutants": records,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_records(records)
    print(f"exit code {code}; wrote {out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
