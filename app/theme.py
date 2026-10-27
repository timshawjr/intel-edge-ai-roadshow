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

import math
from typing import Final

__all__ = [
    "ACCENT", "ACTUAL_MONITOR_CANVAS_SCALE", "ACTUAL_MONITOR_DESKTOP_HEIGHT_PX",
    "ACTUAL_MONITOR_DESKTOP_SCALE", "ACTUAL_MONITOR_DESKTOP_WIDTH_PX",
    "ACTUAL_MONITOR_HEIGHT_MM", "ACTUAL_MONITOR_MM_PER_LOGICAL_PX",
    "ACTUAL_MONITOR_NAME", "ACTUAL_MONITOR_NATIVE_HEIGHT_PX",
    "ACTUAL_MONITOR_NATIVE_WIDTH_PX", "ACTUAL_MONITOR_VIEWING_DISTANCES_M",
    "ACTUAL_MONITOR_WIDTH_MM", "BG", "BG_PANEL", "BG_PANEL_DEEP", "BG_VIDEO", "BORDER",
    "ARIAL_CAP_HEIGHT_EM", "ARIAL_X_HEIGHT_EM", "BOOTH_PANEL_DIAGONAL_IN",
    "BOOTH_PANEL_WIDTH_MM", "BOOTH_VIEWING_DISTANCE_M", "CARD_BADGE_INSET",
    "CANVAS_HEIGHT", "CANVAS_WIDTH", "CARD_ACTIVITY_LINES",
    "CARD_REASON_CHARS",
    "CARD_REASON_MARKER",
    "COLOR_BAD", "COLOR_GOOD", "COLOR_NEUTRAL",
    "COLOR_WARN", "DRAWER_WIDTH_FRACTION", "FONT_FAMILY", "GAP", "GRID", "MARGIN",
    "MM_PER_LOGICAL_PX", "OVERLAY_BOX",
    "OVERLAY_LABEL", "OVERLAY_ZONE", "RADIUS", "READING_ACUITY_ARCMIN",
    "READING_MARGINAL_ARCMIN", "READING_MIN_ARCMIN", "READING_MIN_HEIGHT_MM",
    "READING_MIN_HEIGHT_PX", "SPACING", "THUMB_ROW_FRACTION",
    "TEXT_ON_ACCENT", "TEXT_ON_PANEL",
    "TYPE_BOOTH_DETAIL", "TYPE_BOOTH_LABEL", "TYPE_BOOTH_NUMERAL",
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
#:
#: **M4 Task 1: 13 / 66 / 21.** The main band's 66% and the video column's 68% are the
#: design's proportions and are unchanged. The footer lost 1% because it no longer carries a
#: pipeline strip and a control bar - it carries one verdict line per camera, full width, at
#: the legibility ladder - and the header gained it because the mandatory label is now 44 px
#: tall and the error line, when there is one, sits under it.
GRID: Final[dict[str, float]] = {
    "header": 0.13,
    "main": 0.66,
    "footer": 0.21,
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

#: The presenter drawer's width, as a fraction of the canvas. **M3 Task 5.**
#:
#: The drawer is an **overlay**, not a member of the layout `Column`: it is anchored to the
#: window's right edge and spans the full height, so opening it cannot reflow the hero, the
#: tiles, the badges or the staleness line. 0.32 keeps its left edge at x = 1306 on the
#: 1920 px canvas, which is to the right of the hero column (x 24..1297) - so a long
#: explanation can never obscure the picture, which is the design's requirement.
DRAWER_WIDTH_FRACTION: Final[float] = 0.32

# -- spacing ---------------------------------------------------------------------
MARGIN: Final[int] = 24
GAP: Final[int] = 16
SPACING: Final[dict[str, int]] = {"xs": 4, "sm": 8, "md": 16, "lg": 28, "xl": 48}

#: Corner radius for cards and panels.
RADIUS: Final[int] = 10

# -- M4: the card's internal layout ------------------------------------------------------
#
# **M3's four card tokens are gone, and that is the shape of this task.** `CARD_COLUMN_SPLIT`
# (0.42), `CARD_COLUMN_MIN_EM` (4), `CARD_STALENESS_LINES` (2) and `CARD_REASON_LINES` (2)
# described a two-column card carrying a wrapped staleness sentence and a two-line reason at
# 18 px. At the legibility ladder the card is one column of three rows and the sentences are
# single lines in `Main.qml`'s full-width footer, so those tokens have nothing left to
# describe - and a token nothing binds to is a token that lies about the layout.
#
# What survives is the activity line's budget, because the drawer still renders it (at the
# drawer's own 18 px, where the drawer is read at arm's length by a presenter).

#: Lines the per-camera activity line may occupy **in the presenter drawer**.
CARD_ACTIVITY_LINES: Final[int] = 2

#: **Characters of a failure reason shown before it is cut**, and the marker that admits the
#: rest. M3 Task 2.
#:
#: The supervisor's stated reason for a dead worker is **2331 characters** - measured, from
#: `docs/qualification/m3-capture/100pct.json`. It cannot be shown, and shortening it is not
#: optional, because **a silently shortened error is a hidden error**. So the cut is decided
#: **in Python, at a stated budget**, and the screen says how much it is not showing. That puts
#: the decision in one place, makes it assertable, and leaves Qt nothing to elide silently.
#:
#: **40 is measured against the render, not chosen.** The right column is 42% of a 583 px card
#: = **244.9 logical px**, and at `TYPE_DETAIL` (18 px) Arial the render breaks the reason as
#: `route transition to generation 1` (**29 characters**) then `was not... +2291 more`
#: (**21**), so **29 characters a line** at this size. The marker `... +2291 more` is **14**
#: characters, and 40 + 14 = **54** is what the budget has to fit in two lines.
#:
#: **The guarantee is not this arithmetic, it is `truncated: []`.** Whether Qt had to elide
#: anything is measured, not estimated: `tests/test_capture_ui_canvas.py` asserts the harness
#: reports an empty `truncated` list at 100 / 125 / 150 / 200 %, so if the budget were one
#: character too generous the harness would say so and the test would fail. That is what makes
#: this a measurement rather than a guess.
CARD_REASON_CHARS: Final[int] = 40

#: The marker's wording. **In `app/theme.py` because it is the budget's other half**: how many
#: characters are shown and the sentence that admits the rest are one decision, and splitting
#: them across a Python file and a QML file is how they drift apart.
CARD_REASON_MARKER: Final[str] = "... +{hidden} more"

# -- overlay strokes --------------------------------------------------------------
#: Overlay stroke widths, in logical pixels. **Not** a detector threshold.
OVERLAY_BOX: Final[int] = 3
OVERLAY_ZONE: Final[int] = 2
OVERLAY_LABEL: Final[int] = 22

# ================================================================================
# M4: the booth target, and the legibility ladder derived from it
# ================================================================================
#
# **This block is viewing geometry, not a metric.** It holds no number that was measured
# from this machine's inference, telemetry or video: what it holds is the *distance a
# passer-by stands at*, the *panel the booth is required to provide*, and the type sizes
# that follow from those two by trigonometry. Everything here is either a **choice** (the
# panel and the distance) or **arithmetic on that choice** (the sizes).
#
# **The 55-inch panel is a choice, not a measurement, and it is stated as a requirement on
# the booth — not as the panel this screen was measured on.** Those are two different
# claims and M4 originally ran them together: it judged the delivered screen against the
# 55-inch requirement and reported the pass as the result. The monitor actually attached to
# this machine is measured in the section at the end of this file (DELL S3422DWG, 80 x 33 cm,
# 0.310 mm per logical pixel) and it is roughly **half** the scale of the 55-inch panel. On
# the attached monitor the screen passes 16 arcmin of cap height at 2 m (16.8 arcmin at
# 44 px) and fails at 3 m (11.2 arcmin); its x-height fails the 16 arcmin band at both
# distances. That is the achieved result. The 55-inch panel is what would make 3 m work.
#
# A 27-inch desk monitor is not a booth display, and at 3 m it is arithmetically impossible
# to satisfy at three cameras: 16 arcmin needs 45 px of cap height there, i.e. a 63 px
# Arial, and three cameras' worth of that does not fit 1920 x 1080 next to the pictures. So
# the requirement is stated instead of assumed:
#
#   **a 55-inch 16:9 1080p panel, viewed from 3 m.**
#
# Why 55 inches, and what it is not:
#
#   - it is the most common trade-show booth display size, and 3 m is a booth aisle;
#   - it is **not measured**, and this module does not claim it is. It is a hardware
#     requirement the operator has to meet for the screen to read at 3 m. The M4 legibility
#     review carries the sensitivity table: on a 40-inch panel the same screen's body text
#     is 12.1 arcmin (FAIL) and on the attached monitor it is 11.2 arcmin (FAIL) at 3 m;
#   - the alternative - shrinking the canvas so the same content is drawn larger - is the
#     route this task explicitly rejects. The canvas is unchanged at 1920 x 1080.
#
# The arithmetic, in the order it is applied:
#
#   width_mm = diagonal_in * 25.4 * 16 / hypot(16, 9)            = 1217.59 mm
#   mm/px    = width_mm / CANVAS_WIDTH                            = 0.63416 mm
#   cap_mm   = 2 * distance_m * 1000 * tan( (16 arcmin / 2) in radians )
#                                                                 = 13.963 mm
#   cap_px   = cap_mm / mm_per_px                                 = 22.02 px
#   font_px  = cap_px / (measured x-height of Arial in em)        = 42.47 px -> 44
#
# **The last line is the one that decides the number, and it uses x-height, not cap
# height.** A digit's height is the cap height, so a threshold expressed in cap height
# flatters every mixed-case sentence on the screen by ~28 %: Arial's lowercase is 0.5184 em
# against a 0.7164 em cap. M3's review measured that gap and disclosed it; a screen whose
# *sentences* are the thing a passer-by must read has to clear the threshold on the height
# the sentences are actually drawn at. So the schedule below is built on the x-height and
# therefore lands on 44 px, which also happens to be the figure M4's brief arrives at from
# the 16-arcmin-at-2-m direction.
#
# At 44 px on this panel: cap height 31.52 px = 19.99 mm = **22.9 arcmin**, x-height
# 22.81 px = 14.46 mm = **16.6 arcmin**. Both clear 16.
#
# The ratios and the sizes are pinned by `tests/test_capture_ui_canvas.py` against the
# rendered scene, and the ratios themselves are measured from this machine's Arial below
# rather than taken from a table.
BOOTH_PANEL_DIAGONAL_IN: Final[float] = 55.0
BOOTH_VIEWING_DISTANCE_M: Final[float] = 3.0

#: The panel's width in millimetres, from the diagonal and a 16:9 aspect ratio.
BOOTH_PANEL_WIDTH_MM: Final[float] = (
    BOOTH_PANEL_DIAGONAL_IN * 25.4 * 16.0 / math.hypot(16.0, 9.0)
)

#: Millimetres per logical pixel **on the 55-inch panel the booth is required to provide**.
#: This is the requirement's conversion, NOT the attached monitor's: for anything that
#: claims what the delivered screen achieves, use `ACTUAL_MONITOR_MM_PER_LOGICAL_PX` below.
MM_PER_LOGICAL_PX: Final[float] = BOOTH_PANEL_WIDTH_MM / CANVAS_WIDTH

#: The reading bands, from ISO 9241-3 (16 arcmin for comfortable sustained reading) and
#: Snellen 20/20 (5 arcmin is one arcminute per stroke). These are *reading standards*, not
#: detector thresholds: nothing in the pipeline reads them.
READING_MIN_ARCMIN: Final[float] = 16.0
READING_MARGINAL_ARCMIN: Final[float] = 10.0
READING_ACUITY_ARCMIN: Final[float] = 5.0

#: **Measured on this machine's Arial**, not copied from a font table: at 44 px,
#: `QFontMetricsF.capHeight()` is 31.52 px and `xHeight()` is 22.81 px, so the ratios are
#: 0.7164 and 0.5184. `tests/test_capture_ui_canvas.py` re-measures both from the font the
#: render actually used and fails if they drift.
ARIAL_CAP_HEIGHT_EM: Final[float] = 0.7164
ARIAL_X_HEIGHT_EM: Final[float] = 0.5184

#: The cap height a character must have to clear `READING_MIN_ARCMIN` at the stated
#: distance, in millimetres. `2 d tan(theta / 2)`, the same formula the review uses.
READING_MIN_HEIGHT_MM: Final[float] = (
    2.0 * BOOTH_VIEWING_DISTANCE_M * 1000.0
    * math.tan(math.radians(READING_MIN_ARCMIN / 60.0 / 2.0))
)

#: The same, in logical pixels on the stated panel.
READING_MIN_HEIGHT_PX: Final[float] = READING_MIN_HEIGHT_MM / MM_PER_LOGICAL_PX

#: **The derived minimum body size**: the smallest Arial pixel size whose *x-height* clears
#: the reading minimum, rounded up to a whole multiple of 4. On the stated panel this is
#: 44. A card that needs a smaller size for detail text is a card whose design has not been
#: decided - not a licence to shrink the type.
TYPE_BOOTH_DETAIL: Final[int] = (
    int(math.ceil(
        READING_MIN_HEIGHT_MM / MM_PER_LOGICAL_PX / ARIAL_X_HEIGHT_EM / 4.0)) * 4
)

#: Headings: a step above the body so a camera's name reads as a heading rather than as one
#: more line of prose. **Not** derived; it is a step, and the step is the design decision.
TYPE_BOOTH_LABEL: Final[int] = 48

#: The frame age. The largest live figure on the screen, as spec section 7 requires, and
#: the one number a passer-by is entitled to read from the aisle.
TYPE_BOOTH_NUMERAL: Final[int] = 56

#: The thumbnail row's share of the video column's height. **Was `0.16` inline in
#: `Main.qml`.** It has to hold a `TYPE_BOOTH_DETAIL` caption bar, so it is a token now: at
#: 16 % a 660 px band gives a 105 px thumbnail and the 44 px caption alone would be more
#: than half of it.
THUMB_ROW_FRACTION: Final[float] = 0.22

#: The card's internal row heights, so the three cards' geometry can be recomputed without
#: measuring the render. `CARD_AGE_ROW` is one `TYPE_BOOTH_NUMERAL` line plus the padding a
#: `Text` adds; the other two are their own type sizes plus the badge box's padding.
CARD_BADGE_INSET: Final[int] = 4

# ================================================================================
# The monitor actually attached to this machine — measured, and not the booth panel
# ================================================================================
#
# **This block is a measurement, and it is the geometry the delivered screen is judged
# against.** It exists because M4's first pass judged the same render against the 55-inch
# requirement above and presented that pass as the result; the booth does not have a
# 55-inch panel, and the screen's achieved angular size is set by the monitor that is
# attached. The claim the screen must lead with is this block's, not the requirement's.
#
# Source, read on this machine on 2026-10-27 with WMI:
#
#   Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorBasicDisplayParams
#     InstanceName             MaxHorizontalImageSize  MaxVerticalImageSize  Active
#     DISPLAY\DELD129\..._0    80                      33                    True
#   Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorID
#     InstanceName             Manufacturer  UserFriendlyName  SerialNumber
#     DISPLAY\DELD129\..._0    DEL           DELL S3422DWG     ...
#
# `MaxHorizontalImageSize` / `MaxVerticalImageSize` are in **whole centimetres** (WMI's
# documented unit), so 80 cm x 33 cm carry +/- 5 mm each. The native mode is 3440 x 1440;
# Windows runs it at 150 %, so the desktop the app sees is 2293 x 960 logical pixels.
#
# The 1920 x 1080 canvas cannot be shown one-to-one in a 960-logical-pixel-tall desktop, so
# it is **height-limited** (letterboxed to its height):
#
#   960 / 1080                    = 0.8889        canvas scale in the desktop
#   1920 x 0.8889                 = 1706.7        canvas width, logical px
#   2560 x 1440 physical px on a 3440 x 1440 panel (the 150 % scale cancels)
#
# and then:
#
#   mm/px = (1706.7 / 2293) x (800 / 1920) = 0.3101   (the desktop's own width)
#   mm/px = (2560 / 3440) x 800 / 1920     = 0.3101   (native physical pixels)
#   mm/px = 330 / 1440 x 2560 / 1920       = 0.3056   (from the rounded 33 cm height)
#
# **0.310 mm per logical pixel** is the figure used, and it is the upper end of that
# bracket. Using the smaller height-derived value would flatter every verdict below, so it
# is not used. M3's rounded 0.310 is the same number, which is the useful cross-check: M3's
# anchor *was* the attached monitor, and the 55-inch panel is roughly twice its scale
# (0.634 / 0.310 = 2.05).
#
# The angular consequences, on the shipped ladder (cap 0.7164 em, x-height 0.5184 em):
#
#   44 px  cap 9.77 mm  2 m 16.8' PASS   3 m 11.2' FAIL   x 12.2' / 8.1'
#   48 px  cap 10.66 mm 2 m 18.3' PASS   3 m 12.2' FAIL   x 13.3' / 8.8'
#   56 px  cap 12.43 mm 2 m 21.4' PASS   3 m 14.2' FAIL   x 15.5' / 10.3'
#
# So on the attached monitor the screen **passes 16 arcmin of cap height at 2 m and fails
# at 3 m**, and its mixed-case sentences (x-height) do not reach 16 arcmin at either
# distance. `tests/test_capture_ui_canvas.py` asserts both geometries: the attached monitor
# is the gating case and its numbers are pinned; the 55-inch panel is a separate,
# clearly-labelled requirement.
ACTUAL_MONITOR_NAME: Final[str] = "DELL S3422DWG"
ACTUAL_MONITOR_WIDTH_MM: Final[float] = 800.0
ACTUAL_MONITOR_HEIGHT_MM: Final[float] = 330.0
ACTUAL_MONITOR_NATIVE_WIDTH_PX: Final[int] = 3440
ACTUAL_MONITOR_NATIVE_HEIGHT_PX: Final[int] = 1440
ACTUAL_MONITOR_DESKTOP_WIDTH_PX: Final[int] = 2293
ACTUAL_MONITOR_DESKTOP_HEIGHT_PX: Final[int] = 960
ACTUAL_MONITOR_DESKTOP_SCALE: Final[float] = 1.5

#: The 1920x1080 canvas's scale when it is fitted to the measured desktop's height.
ACTUAL_MONITOR_CANVAS_SCALE: Final[float] = (
    ACTUAL_MONITOR_DESKTOP_HEIGHT_PX / CANVAS_HEIGHT
)

#: Millimetres per logical pixel on the attached monitor. **The measured conversion**, and
#: the one any claim about the delivered screen must use.
ACTUAL_MONITOR_MM_PER_LOGICAL_PX: Final[float] = (
    ACTUAL_MONITOR_WIDTH_MM
    * (CANVAS_WIDTH * ACTUAL_MONITOR_CANVAS_SCALE / ACTUAL_MONITOR_DESKTOP_WIDTH_PX)
    / CANVAS_WIDTH
)

#: The distances the specification names: "readable at 2-3 m on the actual monitor".
ACTUAL_MONITOR_VIEWING_DISTANCES_M: Final[tuple[float, ...]] = (2.0, 3.0)
