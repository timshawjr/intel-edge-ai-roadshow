# M4 — the legibility review: what the booth screen reads at three metres

**What this document is.** The measurement M4 Task 1 exists to make: for every element a
passer-by has to read on the always-visible booth screen, **the rendered cap height in
millimetres, the angular size at the stated viewing distance, and a verdict** — computed from
the **rendered scene graph and the saved PNG**, not from a token multiplied by a constant.
M3's review answered the same question and found the answer was no; this one says what was
removed to change it, what the screen costs, and what it still cannot do.

**Read §1 first.** The result is conditional on a **stated booth panel** — a 55-inch 1080p
display viewed from 3 m — and that panel is a **choice, recorded as a requirement on the
booth, not a measurement of it**. M3 could not establish a booth display either. On a 40-inch
panel the same screen passes on cap height and its lowercase is marginal; on M3's 27-inch
anchor nothing on the screen passes, and no layout at 1920×1080 could make it pass at three
cameras.

---

## 1. The stated target, and what kind of claim it is

`app/theme.py` now carries the target as named constants, and this is the whole of the
assumption:

| token | value | what kind of number it is |
|---|---|---|
| `BOOTH_PANEL_DIAGONAL_IN` | **55.0 in** | **a choice.** A trade-show booth display default. **Not measured.** |
| `BOOTH_VIEWING_DISTANCE_M` | **3.0 m** | **a choice.** The distance M3's brief and M4's brief name as the booth aisle. **Not measured.** |
| `BOOTH_PANEL_WIDTH_MM` | 1217.59 mm | arithmetic: `55 × 25.4 × 16 / hypot(16, 9)` |
| `MM_PER_LOGICAL_PX` | **0.63416** | arithmetic: panel width / 1920 logical px |
| `READING_MIN_ARCMIN` | 16′ | ISO 9241-3's minimum character height for comfortable sustained reading |
| `ARIAL_CAP_HEIGHT_EM` | 0.7164 | **measured on this machine's Arial**: `QFontMetricsF.capHeight()` is 31.52 px at 44 px |
| `ARIAL_X_HEIGHT_EM` | 0.5184 | **measured on this machine's Arial**: `xHeight()` is 22.81 px at 44 px |
| `READING_MIN_HEIGHT_MM` | 13.963 mm | arithmetic: `2 · 3000 · tan(16′/2)` |
| `TYPE_BOOTH_DETAIL` | **44 px** | arithmetic, from the **x-height** requirement below |

### Why 55 inches, and why x-height decides the size

**M3 assumed a 27-inch desk monitor** (0.310 mm/logical px) and said in the same paragraph that
it was "an assumption about the booth, not a measurement of it". It was right to say so, and
the assumption was the wrong shape for a booth: a 27-inch monitor is what sits on a desk. A
55-inch 1080p display is a common trade-show panel, and it is the **smallest** panel on which
this screen clears the threshold on *both* the height of its digits and the height of its
lowercase — which matters because most of what a passer-by must read is a sentence.

**A digit's height is the cap height. A sentence's is not.** Arial's lowercase is 0.5184 em
against a 0.7164 em cap — **28 % shorter** — so a threshold expressed in cap height flatters
every mixed-case sentence on the screen by about a quarter. M3 measured that gap and disclosed
it; M4 has to act on it, because the two lines the screen exists for ("STALE - this picture is
11.7 s old; the limit is 2 s", "no device is running") are sentences. So the ladder is derived
from the **x-height**:

```
font_px = 13.963 mm / 0.63416 mm/px / 0.5184 = 42.4  ->  44  (rounded up to a multiple of 4)
```

At 44 px on this panel: **cap 31.52 px = 19.99 mm = 22.9′**, **x-height 22.81 px = 14.47 mm =
16.6′**. Both clear 16′.

The same curve, for the record, is why 44 px is the number M4's brief arrives at from the
other direction. Its arithmetic quotes "9.2 mm of cap height ≈ 44 px" for 16′ at 3 m. **9.2 mm
is 16′ at 2 m**, not 3 m — 16′ at 3 m is 13.96 mm of cap height, i.e. a **63 px** Arial at
M3's 0.310 mm/px. That correction matters and is recorded rather than smoothed over: on M3's
27-inch assumption the screen would have needed 63 px text, not 44 px, and three cameras of
that do not fit 1920×1080 beside three pictures. The 44 px the brief names is right; the
justification it gives is a factor of 1.5 out.

### What changes if the panel is not 55 inches

**Nothing about the render; everything about the claim.** The screen is a fixed 1920×1080
logical layout, so the arcminute figures scale linearly with the panel's width:

| panel | width (mm) | mm/px | 44 px cap | 44 px x-height | both clear 16′? |
|---|---:|---:|---:|---:|---|
| 27 in (M3's assumption) | 598 | 0.3113 | 11.2′ | 8.1′ | **no** |
| 32 in | 708 | 0.3690 | 13.3′ | 9.6′ | **no** |
| 40 in | 886 | 0.4612 | 16.7′ | 12.1′ | **no** (cap passes, lowercase marginal) |
| **55 in (stated)** | **1218** | **0.6342** | **22.9′** | **16.6′** | **yes** |
| 65 in | 1439 | 0.7495 | 27.1′ | 19.6′ | yes |

So the honest statement of the result is: **this screen reads at three metres on a 55-inch
1080p panel or larger.** It does not read at three metres on the 27-inch monitor M3 assumed,
and it is not close — every element would be at 8.1′ of x-height, half the threshold. A booth
that installs a 27-inch or 40-inch panel has an M4 regression, not a working screen, and the
preflight does not measure panel size, so this is a **requirement in a document, not a gate**.
That is recorded as a finding in §6.

---

## 2. The method

Three independent things, and the measurement is only as good as the weakest of them:

1. **The rendered font size** comes from `font_pixels` in the committed capture — read out of
   the running scene graph, one entry per named `Text`. Not from `app/theme.py`: a token check
   proves the token was applied, not that the text was drawn, at that size, on that surface.
2. **Cap height and x-height** come from `QFontMetricsF` on the **same registered Arial at the
   size the scene reports**. The two ratios are asserted against `app/theme.py` on every run
   (`test_the_font_ratios_this_test_divides_by_are_the_fonts_real_ones`), so they cannot drift
   away from the font that is rasterised.
3. **The ink is measured in the saved PNG**, inside each item's own reported extent, and must
   span at least the cap height. That is the step that stops this being a property check: an
   item whose colour matches its surface, or whose box is clipped away, or that is drawn
   smaller than its declared size, has no ink band of the right height.

The angle is `2·atan(h / 2d)` at `BOOTH_VIEWING_DISTANCE_M`. The element set is **every
visible named `Text` on the live screen** — 24 of them — with the exclusion list asserted
empty, so nothing on the always-visible surface is exempt from the threshold. That is a direct
answer to M3 §3's finding that the title, the mandatory label and every caption were outside
its census: they are all named now, and all measured.

The tests are `tests/test_capture_ui_canvas.py::TheBoothScreenReadsAtThreeMetres` (11 tests,
green) and the non-vacuity guards are in that class's docstring: presence, visibility,
non-empty text and a non-degenerate extent come **before** any angle; a `Repeater` delegate is
asserted to be reached by the walk; a `NaN` extent fails; a zero-height ink band fails (the
1×1-window shape); and the threshold is shown to **discriminate** — the same formula applied to
M3's 18 px and 34 px sizes comes out below 16′, so the assertion is measuring the type and not
the panel.

---

## 3. The table — every visible element, at 3 m on a 55-inch panel

Read from `docs/qualification/m3-capture/100pct.json` (the committed render) and computed by
the same method as the test. `cap` is the cap-height angle, `x` the x-height angle; both are
asserted.

| element | font px | cap px | cap mm | cap | x px | x mm | x | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `recordedFootageText` — the mandatory label | 44 | 31.52 | 19.99 | 22.9′ | 22.81 | 14.47 | 16.6′ | **PASS** |
| `appTitle` — `Intel Edge AI Roadshow` | 44 | 31.52 | 19.99 | 22.9′ | 22.81 | 14.47 | 16.6′ | **PASS** |
| `deviceBadgeText` — `device: CPU` | 44 | 31.52 | 19.99 | 22.9′ | 22.81 | 14.47 | 16.6′ | **PASS** |
| `explainOpenButtonText` — `what the boxes mean` | 44 | 31.52 | 19.99 | 22.9′ | 22.81 | 14.47 | 16.6′ | **PASS** |
| `cameraName_A/B/C` | 48 | 34.39 | 21.81 | 25.0′ | 24.89 | 15.78 | 18.1′ | **PASS** |
| `cameraState_A/B/C` | 44 | 31.52 | 19.99 | 22.9′ | 22.81 | 14.47 | 16.6′ | **PASS** |
| **`streamAgeText_A/B/C` — the frame age** | **56** | **40.11** | **25.44** | **29.1′** | 29.05 | 18.42 | 21.1′ | **PASS** |
| `streamBadgeText_A/B/C` — the device badge | 44 | 31.52 | 19.99 | 22.9′ | 22.81 | 14.47 | 16.6′ | **PASS** |
| **`streamStaleText_A/B` — the staleness verdict** | **44** | 31.52 | 19.99 | **22.9′** | 22.81 | 14.47 | **16.6′** | **PASS** |
| `streamReasonText_C` — the failed camera's reason | 44 | 31.52 | 19.99 | 22.9′ | 22.81 | 14.47 | 16.6′ | **PASS** |
| `tileCaptionText_B/C` — each thumbnail's camera and device | 44 | 31.52 | 19.99 | 22.9′ | 22.81 | 14.47 | 16.6′ | **PASS** |
| `tileNoFrameText_A/B/C` — `no frame from Camera … yet` | 44 | 31.52 | 19.99 | 22.9′ | 22.81 | 14.47 | 16.6′ | **PASS** |

**The whole always-visible screen passes, on both measures, at 3 m and 55 inches.** The
smallest element on it is 16.6′ of x-height (44 px), and the frame age — the figure spec
section 7 says must be legible at a glance — is **29.1′ of cap height**, 1.8× the threshold.
M3 measured the same headline at **8.5′** and every 18 px line at **4.6′**, below the acuity
floor.

**The two elements the screen exists for, named:** `streamStaleText_A`/`_B` at **22.9′ cap /
16.6′ x** and `streamAgeText_A/B/C` at **29.1′ cap / 21.1′ x**. Both are asserted *by name* in
`test_the_staleness_line_and_the_per_camera_age_are_among_the_passing`, and both are among the
passing set rather than merely present.

*Measured ink bands, for the record:* the ink check asserts each band spans at least the cap
height and at most one line box. Two items failed that on the first render — the two thumbnail
"no frame" sentences — because the caption bar is drawn after them and covered their lower
half: **29 px of ink against a 31.5 px cap**. That is exactly the defect the ink check exists
to catch, and the fix was to centre those sentences in the area *above* the caption bar
(`VideoTile.qml`), not to relax the check.

---

## 4. What was removed from the always-visible screen, and where it went

The arithmetic is why this is a removal and not a font swap. Recomputed row by row at the
ladder, an M3 card's eleven rows are **764 px** for one camera (name and state 54, age 63,
staleness sentence 98 at two lines in the 567 px column, frame-age window 49, activity 98,
badge 52, backlog 49, rate 49, utilisation 49, window note 49, reason 98, plus ten 4 px gaps
and 16 px of margins), so **three cameras are 2 292 px in the 660 px band they share** - 3.5x
over. The card keeps what a passer-by reads from the aisle, and six figures per camera moved
to the presenter drawer:

| removed from the always-visible screen | where it is now |
|---|---|
| per-camera **backlog** (`streamBacklogText_*`) | presenter drawer, per camera |
| per-camera **analysis rate** (`streamRateText_*`) | presenter drawer, per camera |
| per-camera **frame-age window** (`streamWindowText_*`) | presenter drawer, per camera |
| per-camera **activity line** (`streamActivityText_*` — rate and p50/p95) | presenter drawer, per camera |
| per-camera **device utilisation figure** (`cardText_<metric>`) and its **sampling window** (`cardWindowNote_<metric>`) | presenter drawer, per camera |
| the **event counts** (`eventCountRow`) | presenter drawer |
| the **pipeline strip**, twelve 18 px labels | **removed**; `PipelineStrip.qml` deleted; the drawer states the topology in one sentence |
| the **`status:` and `telemetry fresh` lines** | presenter drawer |
| the **seven footer `?` keys** and the **pinned explanation panel** | **removed**; the drawer's own seven buttons are the explanation surface, opened by a 44 px header button, a click on any card, or F1 |
| the **header scope note** (elided mid-word at 18 px — an M3 finding) | presenter drawer, verbatim and in full (it already was there) |
| the **hover hint** on each card | **removed**, see §6 — it was clipped by the card and could never be seen |

**Nothing was deleted from the product's claims except the pipeline strip, and that is stated.**
Every figure that left the always-visible screen is on the drawer **with the same item name**,
so the M2/M3 tests that read it were retargeted to the drawer rather than dropped, and the
`test_ui_contract` assertions on those names still run. `PipelineStrip.qml` is gone because
twelve 18 px labels cannot be legible at 3 m and the stage list does not fit the drawer's
566 px inner width without eliding; the claim it made — decode, preprocess, inference, decode
detections, track, event, with **only inference on the chosen device** — is carried by
`drawerPipelineText`.

**What stayed, and got bigger:** the mandatory label, the title, the header device badge, three
live pictures each captioned with **its own** camera and device, one column per camera with its
name, state, **frame age** (the largest type on the screen) and confirmed-device badge, and
**one verdict line per camera** — the staleness sentence in words, or the camera's own stated
failure reason. The layout is still 1920×1080 with the design's 66 % main band and 68 % video
column; the header gained 1 % of the band (its type is 44 px tall now) and the footer lost 1 %
(three verdict lines instead of a pipeline strip and a control bar).

---

## 5. The four captures, re-rendered

`python -m tools.capture_ui --out docs/qualification/m3-capture` was re-run on 2026-10-27. All
four scales rendered, and:

| file | pixels | logical | dpr | `glyph_scale` | `overflow` | `truncated` | `absent` | `unreachable` |
|---|---|---|---|---|---|---|---|---|
| `100pct.png` | 1920×1080 | 1920×1080 | 1.00 | 1.0 | `[]` | `[]` | `[]` | `[]` |
| `125pct.png` | 2400×1350 | 1920×1080 | 1.25 | 1.25 | `[]` | `[]` | `[]` | `[]` |
| `150pct.png` | 2880×1620 | 1920×1080 | 1.50 | 1.5 | `[]` | `[]` | `[]` | `[]` |
| `200pct.png` | 3840×2160 | 1920×1080 | 2.00 | 2.0 | `[]` | `[]` | `[]` | `[]` |

The font census is **`{44: 18, 48: 3, 56: 3}`** — visible, non-empty named items only — at
every scale. The drawer is not in these captures (it is closed), so its fit is measured
separately: `PresenterDrawerOnTheScreen` opens it, reads every item and asserts `overflow ==
[]` for all seven explanation selections; the worst (the longest explanation, `gpu`) ends at
y = 972 in a drawer whose inner bottom is 1056.

**The captures still contain no video frame** (the seed drives strings and layout, not
pictures), and M3 §5.4's finding stands: these four files prove the layout, the strings, the
pixels and the census, not the video surface.

---

## 6. Findings

1. **The result is conditional on a 55-inch panel, and nothing gates that.** A booth that
   installs a 40-inch panel gets 16.7′ cap and **12.1′ x-height** — marginal lowercase — and a
   27-inch panel gets 8.1′, half the threshold. `tools/preflight.py` does not measure panel
   size and cannot; this is a **hardware requirement recorded in `app/theme.py` and here**, and
   the presenter guide must state it. This is the most important sentence in this document.
2. **The M3 arithmetic conflated 2 m and 3 m.** Its brief's "9.2 mm of cap height ≈ 44 px" is
   16′ at **2 m**; 16′ at 3 m is 13.96 mm. At M3's 0.310 mm/px that is a **63 px** Arial, not
   44 px. The number M4 shipped is 44 px because the **panel changed** to 55 inches, not
   because 44 px is 16′ at 3 m on a 27-inch screen. Said plainly because the two routes to
   "44 px" are not the same argument.
3. **`PipelineStrip.qml` was deleted, and its claim moved into prose.** Twelve 18 px labels
   could not be made legible at 3 m without a footer band the layout does not have, and the
   drawer cannot hold six stage names without eliding them. `drawerPipelineText` states the
   topology and the placement rule in words, and the `gpu` explanation states the measured
   per-device latency. A presenter who wants the diagram has the words.
4. **The seven footer `?` keys are gone; one button replaces them.** The M1 lesson was that a
   booth machine may be touch-only and a hover-only affordance is then dead, so the drawer has
   a named 44 px entry point in the header plus a click on any card, and F1 is not the only
   way in. The per-key proof moved to the drawer's own buttons, all seven of which are clicked
   by the test suite.
5. **The hover hint was dead and was removed.** It was anchored above the card (`parent.top`),
   and M3's `clip: true` on the card clipped it away — it could never be seen. Naming it in M4
   made `capture_ui` report it as **overflow**, which is how the dead affordance surfaced. It
   is removed rather than repaired: the drawer is the explanation surface and a hover-only hint
   is unreachable on a touch booth.
6. **A camera with a stated failure reason shows the reason instead of its staleness
   sentence.** One verdict line per camera is the budget, and the reason is the more specific
   statement; the age, its warning colour and the card's warning border still carry the
   staleness. In the seeded capture camera C is both stale and failed, and M3 showed both
   sentences. This is a change in what a passer-by can read and it is recorded here because
   trimming a line to make a layout fit without saying so is the failure this project keeps
   finding.
7. **The attract screen was rebuilt on the same ladder.** M3 Task 4's idle overlay used 40 /
   26 / 18 px, so a booth with nobody at it showed text at 9.4′ — below the acuity floor — on
   the screen a passer-by is most likely to see alone. Its title is now 56 px, its recorded
   label 48 px and its headlines 44 px, and the attract census is asserted to contain no size
   below the ladder.
8. **The drawer is 18 px, and that is deliberate.** It is the presenter's surface, read at
   arm's length, and its figures are the ones removed from the aisle surface. It is not part
   of the passenger legibility contract, and the M4 legibility test asserts the *live* screen.
   Writing this down because "the drawer is small" would otherwise be indistinguishable from
   "the drawer was forgotten".
9. **The threshold is a reading standard, not a measurement of this room.** ISO 9241-3's 16′
   is a sustained-reading minimum; it says nothing about the ambient light, the contrast ratio
   (not computed here), or how long a visitor actually looks. The palette is high-contrast by
   design (`#e8eef4` on `#141b23`), and no contrast ratio was measured in M3 or here.

---

## 7. What the tests assert, so a reader can check the claims

| claim in this document | test |
|---|---|
| every visible named text clears 16′ on cap **and** x-height | `TheBoothScreenReadsAtThreeMetres::test_every_visible_text_on_the_live_screen_clears_the_threshold` |
| the staleness line and the ages are among the passing | `…::test_the_staleness_line_and_the_per_camera_age_are_among_the_passing` |
| the element set is the whole screen, not a legend | `…::test_nothing_visible_is_exempt_from_the_threshold` |
| the angle is measured from the render, not the token | `…::test_every_measured_element_has_an_ink_band_of_at_least_its_cap_height`, `…::test_the_font_ratios_this_test_divides_by_are_the_fonts_real_ones` |
| the threshold discriminates (M3's sizes fail it) | `…::test_the_threshold_discriminates_between_the_m3_sizes_and_the_shipped_ones` |
| the age is the largest thing on the screen | `…::test_the_largest_thing_on_the_screen_is_a_frame_age` |
| the four captures overflow, truncate and omit nothing | `CanvasLayoutAtFourScales::test_every_scale_reached_the_state_it_was_asked_to_capture` |
| the drawer holds the removed figures and does not overflow | `PresenterDrawerOnTheScreen` |
| the layout tokens are in `app/theme.py`, not the QML | `ClippingIsTheDecision::test_the_layout_tokens_are_in_theme_and_not_in_the_qml` |

**Mutation checks.** Each new assertion was shown to fail by constructing its fault; the log is
in the task report (`task-1-report.md`), not here, because a mutation log is evidence about the
tests rather than about the screen.
