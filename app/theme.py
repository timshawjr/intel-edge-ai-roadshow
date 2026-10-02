"""`app.theme` - layout tokens and nothing else.

**Rule 10 of AGENTS.md says layout tokens live here, and this module takes that
literally: it holds sizes, spacing, colours and type ramp values, and it holds no metric.**

A metric is a number that was measured. Putting one here would make it a constant, and a
constant that looks like a measurement is the exact failure this repository exists to
prevent - a card showing `0.25` because a layout file says so, rather than because
`MetricSample.value` said so. So:

- **No thresholds, input sizes, frame counts, latencies, rates or windows are here.**
  `conf_threshold` comes from `ScenarioManifest`, the input shape from
  `app.inference.pinned.pinned_input_shape()`, and every rate and percentage from
  `MetricSample` or a measured counter.
- `tests/test_ui_contract.py` reads this module's **source** and fails if any name
  matching the threshold vocabulary (`conf`, `iou`, `threshold`, `fps`, `latency`,
  `p50`, `p95`, `window`, `percent`, `pad`) appears. That is the enforcement; this
  paragraph is the intention.

**These are pixels, not measurements.** The canvas is **1920 x 1080 logical** per spec
section 3 and the design's section 9, split **68% video / 32% metrics** horizontally and
**12% header / 66% video+metrics / 22% pipeline+controls** vertically. Those are the
design's stated proportions, not something derived from this machine's monitor - and
the design records that a 1920x1080 window **cannot be shown on this machine's live
desktop** (usable height 960 px after the taskbar, measured), so the 1080p canvas is
reachable through the offscreen harness and the live window is letterboxed inside it.
`MAIN_WINDOW` is therefore the *logical design* canvas, not a claim about the display.

Type sizes follow spec section 3: 36-44 px headlines, 48-64 px main numerals, 24-28 px
main labels, and nothing below 18 px for detail. `TYPE_*` are the ramp; a card that
needs a size it does not have here is a card whose design has not been decided.

`FONT_FAMILY` is `"Arial"` because it is **discovered, not chosen**: measured on this
machine, `QFontDatabase.families()` returns **411 families under the `windows` platform
plugin and 0 under `offscreen`**, and this file is the application font the booth asks
for rather than one the renderer falls back to silently.
`tests/test_ui_contract.py` registers `C:/Windows/Fonts/arial.ttf` with
`QFontDatabase.addApplicationFont` for the offscreen render, because **an offscreen
render with no registered font draws no glyphs at all** - measured, and the render
assertion would then pass on a screen with no text on it.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "ACCENT", "BG", "BG_PANEL", "BG_PANEL_DEEP", "BG_VIDEO", "BORDER",
    "CANVAS_HEIGHT", "CANVAS_WIDTH", "COLOR_BAD", "COLOR_GOOD", "COLOR_NEUTRAL",
    "COLOR_WARN", "FONT_FAMILY", "GAP", "GRID", "MARGIN", "OVERLAY_BOX",
    "OVERLAY_LABEL", "OVERLAY_TRACK", "OVERLAY_ZONE", "RADIUS", "SPACING",
    "TEXT_ON_ACCENT", "TEXT_ON_PANEL",
    "TYPE_DETAIL", "TYPE_HEADLINE", "TYPE_LABEL", "TYPE_NUMERAL", "TYPE_NUMERAL_SMALL",
    "VIDEO_WIDTH_FRACTION",
]

# -- canvas ----------------------------------------------------------------------
#: The logical design canvas, spec section 3. Not a claim about this monitor: the design
#: records that the live desktop here is 2293 x 960 with a 2245 x 960 working area, so
#: 1920 x 1080 is reachable only through the offscreen harness and the live window is
#: letterboxed inside it (design section 9.1, measured).
CANVAS_WIDTH: Final[int] = 1920
CANVAS_HEIGHT: Final[int] = 1080

#: Video/metrics split, design section 9: "~68% width video / 32% metrics".
VIDEO_WIDTH_FRACTION: Final[float] = 0.68

#: Vertical bands, design section 9: "12% header+title, 66% video+metrics,
#: 22% pipeline+controls".
GRID: Final[dict[str, float]] = {
    "header": 0.12,
    "main": 0.66,
    "footer": 0.22,
}

# -- palette ---------------------------------------------------------------------
#: High contrast, dark surface: this is a booth screen read from a distance, and the
#: design requires "tabular numerals, stable widths, high contrast".
BG: Final[str] = "#0b0f14"
BG_PANEL: Final[str] = "#141b23"
BG_PANEL_DEEP: Final[str] = "#0e141a"
BG_VIDEO: Final[str] = "#05080b"
BORDER: Final[str] = "#26313d"
ACCENT: Final[str] = "#4fb6ff"

#: Text on a panel. Separate from `BG` because `BG` is the window background and using
#: it as a text colour on a panel is exactly the mistake the first offscreen render
#: caught: dark text on a dark card reads as nothing at all. Measured, not stylistic - the
#: render came back with an invisible "Recorded footage" label and an invisible CPU card
#: heading because both were `BG` on `BG_PANEL`.
TEXT_ON_PANEL: Final[str] = "#e8eef4"

#: Text on an accent or status fill (the confirmed-device badge, the mandatory label).
TEXT_ON_ACCENT: Final[str] = "#0b0f14"

#: Status colours, carried **beside a text label** and never instead of one. The design
#: requires "consistent colours plus text labels - never colour alone".
COLOR_GOOD: Final[str] = "#4ad07a"
COLOR_WARN: Final[str] = "#f0b429"
COLOR_BAD: Final[str] = "#ff6b6b"
COLOR_NEUTRAL: Final[str] = "#8b98a5"

# -- type ramp, spec section 3 ---------------------------------------------------
FONT_FAMILY: Final[str] = "Arial"
TYPE_HEADLINE: Final[int] = 40          # 36-44 px headlines
TYPE_NUMERAL: Final[int] = 56          # 48-64 px main numerals
TYPE_NUMERAL_SMALL: Final[int] = 34
TYPE_LABEL: Final[int] = 26            # 24-28 px main labels
TYPE_DETAIL: Final[int] = 18           # >= 18 px detail

# -- spacing ---------------------------------------------------------------------
MARGIN: Final[int] = 24
GAP: Final[int] = 16
SPACING: Final[dict[str, int]] = {"xs": 4, "sm": 8, "md": 16, "lg": 28, "xl": 48}

#: Corner radius for cards and panels.
RADIUS: Final[int] = 10

# -- overlay strokes --------------------------------------------------------------
#: Overlay stroke widths, in logical pixels. **Not** a detector threshold.
OVERLAY_BOX: Final[int] = 3
OVERLAY_ZONE: Final[int] = 2
OVERLAY_LABEL: Final[int] = 22
OVERLAY_TRACK: Final[int] = 18