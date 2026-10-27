# M4 — the legibility review: what the booth screen reads on the monitor it is measured on

**What this document is.** The measurement M4 Task 1 exists to make: for every element a
passer-by has to read on the always-visible booth screen, **the rendered cap height in
millimetres, the angular size at the stated viewing distance, and a verdict** — computed from
the **rendered scene graph and the saved PNG**, not from a token multiplied by a constant.
M3's review answered the same question and found the answer was no; this one says what was
removed to change it, what the screen costs, and what it still cannot do.

**Read §1 first, because the verdict changed.** The screen is judged against the monitor
**actually attached to this machine** — a **DELL S3422DWG, 80 x 33 cm, 3440x1440 native, a
2293x960 desktop, 0.310 mm per logical pixel** — not against a 55-inch hypothetical. On that
monitor the screen **passes 16 arcmin of cap height at 2 m and fails at 3 m**; its mixed-case
sentences (x-height) do not clear 16 arcmin at **either** distance. A 55-inch 1080p panel
would pass at 3 m, and that is a **booth procurement requirement**, not an achieved
measurement. The first version of this document judged the render against that panel and led
with the pass; §6 records the correction.

---

## 1. Two geometries: the measured monitor, and the panel the booth must procure

There is exactly one geometry that is a **measurement of the delivered screen** and one that
is a **requirement on the booth**. They are a factor of two apart, and confusing them changes
the verdict, so they are kept apart everywhere below.

### 1.1 The monitor actually attached — measured, not assumed

Read from WMI on this machine on 2026-10-27:

```
PS> Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorBasicDisplayParams |
      Select InstanceName, MaxHorizontalImageSize, MaxVerticalImageSize, Active
InstanceName                            MaxHorizontalImageSize MaxVerticalImageSize Active
DISPLAY\DELD129\4&16626495&0&UID24631_0                     80                   33   True

PS> Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorID   (decoded)
DISPLAY\DELD129\4&16626495&0&UID24631_0 | DEL | DELL S3422DWG | 809587030
```

`MaxHorizontalImageSize` / `MaxVerticalImageSize` are WMI's **whole-centimetre** fields, so
80 x 33 cm carries **±5 mm** on each dimension; `WmiMonitorID`'s `UserFriendlyName` names the
panel. `Win32_VideoController` reports the native mode **3440x1440**, and Windows runs it at
**150 %** scaling, so the desktop the app sees is **2293x960 logical**.

| quantity | value | kind |
|---|---|---|
| panel | **DELL S3422DWG** | measured (`WmiMonitorID`) |
| physical size | **80 cm x 33 cm** | measured (`WmiMonitorBasicDisplayParams`), ±5 mm |
| native mode | 3440 x 1440 | measured |
| desktop | 2293 x 960 logical (150 %) | measured |
| canvas | 1920 x 1080 logical | spec section 3 |
| **mm per logical px** | **0.310** | arithmetic, below |

The 1920x1080 canvas cannot be shown one-to-one in a 960-logical-pixel-tall desktop, so it is
**height-limited**: 960/1080 = 0.8889, i.e. a **1706.7 x 960** logical canvas, which is
**2560 x 1440** physical pixels on the 3440-pixel-wide panel. Three independent forms of the
same conversion:

```
(1706.7 / 2293) x (800 / 1920) = 0.3101   the desktop's own width
(2560 / 3440) x 800 / 1920     = 0.3101   native physical pixels
(330 / 1440) x 2560 / 1920     = 0.3056   from the rounded 33 cm height
```

**0.310 mm per logical pixel** is the figure used — the upper end of the bracket. Using the
smaller height-derived value would flatter every verdict, so it is not used. This is the same
0.310 M3 recorded, and that agreement is the useful cross-check: **M3's anchor was the
attached monitor.** The panel's 16:9 canvas is about **595 mm** wide (593 mm from its native
pixel pitch) — the width of a 27-inch 16:9 monitor (598 mm) — because a 21:9 panel pillarboxes
it. In angular terms the attached monitor is a 27-inch screen, not a 34-inch one.

### 1.2 The angles on the attached monitor — the achieved result

Cap height is the digits' height; x-height is the height a mixed-case sentence is actually
drawn at. Both are required to clear 16′. `cap mm` and `x mm` use Arial's **measured**
ratios, 0.7164 em and 0.5184 em (`QFontMetricsF` on this machine's registered Arial, re-read
by the test on every run).

| element | px | cap mm | 2 m cap | 3 m cap | 2 m x | 3 m x | verdict at 2 m | verdict at 3 m |
|---|---:|---:|---:|---:|---:|---:|---|---|
| body — mandatory label, title, device badge, verdict sentences, captions | 44 | 9.77 | **16.8′** | **11.2′** | 12.2′ | 8.1′ | cap **PASS**, lowercase marginal | **FAIL** |
| camera names | 48 | 10.66 | **18.3′** | **12.2′** | 13.3′ | 8.8′ | cap **PASS**, lowercase marginal | **FAIL** |
| frame age | 56 | 12.43 | **21.4′** | **14.2′** | 15.5′ | 10.3′ | cap **PASS**, lowercase marginal | **FAIL** |

**The verdict on the actual monitor, in two parts.**

- **It passes at 2 m on cap height and fails at 3 m.** At 2 m every element clears 16′ of
  cap height (the smallest is 16.8′); at 3 m none does (the largest is 14.2′).
- **The mixed-case sentences do not clear 16′ at either distance.** At 2 m the 44 px
  x-height is **12.2′** — inside the 10–16′ marginal band, not the comfort band — and at 3 m
  it is 8.1′. The frame age's 15.5′ at 2 m is just short as well. So "passes at 2 m" is a
  **cap-height** statement and nothing broader; the report says so in the same sentence.
  `tests/test_capture_ui_canvas.py` pins each of these numbers and asserts that no element
  clears 16′ of x-height at 2 m, so the qualification cannot be dropped without a test
  failing.

### 1.3 The 55-inch panel is a procurement requirement, not the delivered result

`app/theme.py` carries the requirement as named constants, because a booth screen needs a
stated target even when the development machine is smaller:

| token | value | what kind of number it is |
|---|---|---|
| `BOOTH_PANEL_DIAGONAL_IN` | **55.0 in** | **a choice.** A trade-show booth display default. **Not measured.** |
| `BOOTH_VIEWING_DISTANCE_M` | **3.0 m** | **a choice.** The distance M3's brief and M4's brief name as the booth aisle. **Not measured.** |
| `BOOTH_PANEL_WIDTH_MM` | 1217.59 mm | arithmetic: `55 × 25.4 × 16 / hypot(16, 9)` |
| `MM_PER_LOGICAL_PX` | **0.63416** | arithmetic: requirement panel width / 1920 logical px |
| `READING_MIN_ARCMIN` | 16′ | ISO 9241-3's minimum character height for comfortable sustained reading |
| `ARIAL_CAP_HEIGHT_EM` | 0.7164 | **measured on this machine's Arial**: `QFontMetricsF.capHeight()` is 31.52 px at 44 px |
| `ARIAL_X_HEIGHT_EM` | 0.5184 | **measured on this machine's Arial**: `xHeight()` is 22.81 px at 44 px |
| `READING_MIN_HEIGHT_MM` | 13.963 mm | arithmetic: `2 · 3000 · tan(16′/2)` |
| `TYPE_BOOTH_DETAIL` | **44 px** | arithmetic, from the **x-height** requirement below |

**Why 55 inches, and why x-height decides the size.** A 55-inch 1080p panel is the smallest
common trade-show panel on which this screen clears the threshold on *both* the height of its
digits and the height of its lowercase — which matters because most of what a passer-by must
read is a sentence. A digit's height is the cap height; a sentence's is not. Arial's lowercase
is 0.5184 em against a 0.7164 em cap — **28 % shorter** — so a threshold expressed in cap
height flatters every mixed-case sentence by about a quarter. M3 measured that gap and
disclosed it; M4 acts on it, because the two lines the screen exists for ("STALE — this
picture is 11.7 s old; the limit is 2 s", "no device is running") are sentences. So the
ladder is derived from the **x-height** on the requirement panel:

```
font_px = 13.963 mm / 0.63416 mm/px / 0.5184 = 42.4  ->  44  (rounded up to a multiple of 4)
```

At 44 px on the **required** panel: **cap 31.52 px = 19.99 mm = 22.9′**, **x-height 22.81 px =
14.47 mm = 16.6′**. Both clear 16′ at 3 m. That is what the booth must provide for the screen
to read at 3 m. **It is not what this machine measured.**

**What the requirement costs, stated as a requirement.** A booth that installs a 55-inch
panel gets 22.9′ / 16.6′ at 3 m and the screen reads. A booth that installs the monitor this
machine has gets 11.2′ / 8.1′ and it does not read at 3 m. Because `tools/preflight.py` cannot
measure a panel, this is a **hardware requirement in a document, not a gate**; it is the
largest open risk in the deliverable and §6 records it as such.

### 1.4 Sensitivity: the same render on other panels, at 3 m

| panel | canvas width (mm) | mm/px | 44 px cap | 44 px x-height | both clear 16′? |
|---|---:|---:|---:|---:|---|
| 24 in 16:9 | 531 | 0.2766 | 10.0′ | 7.2′ | **no** |
| 27 in 16:9 | 598 | 0.3113 | 11.2′ | 8.1′ | **no** |
| **attached: S3422DWG, pillarboxed (measured)** | **595** | **0.310** | **11.2′** | **8.1′** | **no** |
| 32 in 16:9 | 708 | 0.3690 | 13.3′ | 9.6′ | **no** |
| 40 in 16:9 | 886 | 0.4612 | 16.7′ | 12.1′ | **no** (cap passes, lowercase marginal) |
| **55 in 16:9 (requirement)** | **1218** | **0.6342** | **22.9′** | **16.6′** | **yes** |
| 65 in 16:9 | 1439 | 0.7495 | 27.1′ | 19.6′ | yes |

The attached monitor and a 27-inch 16:9 panel are the same case to within a rounding error,
which is the point: the panel is 34 inches diagonally but its 16:9 canvas is about 595 mm wide.
The 55-inch row is a requirement, and no tool in this repository can check that a booth meets it.

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

The angle is `2·atan(h / 2d)`, evaluated at **both** geometries. The **gating** case is the
monitor attached to this machine (`ACTUAL_MONITOR_MM_PER_LOGICAL_PX`, 0.310 mm/logical px) at
**2 m and 3 m**. The **55-inch requirement** (`MM_PER_LOGICAL_PX`, 0.63416 mm/logical px) at
3 m is asserted separately and labelled: it is what the booth must procure, not what was
delivered. A test that defaulted to the requirement's panel would pass at 3 m while the
delivered screen fails there, so the default is the attached monitor and a test asserts that
default.

The element set is **every visible named `Text` on the live screen** — 24 of them — with the
exclusion list asserted empty, so nothing on the always-visible surface is exempt from the
threshold. That is a direct answer to M3 §3's finding that the title, the mandatory label and
every caption were outside its census: they are all named now, and all measured.

The tests are `tests/test_capture_ui_canvas.py::TheBoothScreenReadsAtThreeMetres` — **16
tests, green** (11 from the first version, 5 attached-monitor tests added by the correction) —
and the non-vacuity guards are in that class's docstring: presence, visibility, non-empty text
and a non-degenerate extent come **before** any angle; a `Repeater` delegate is asserted to be
reached by the walk; a `NaN` extent fails; a zero-height ink band fails (the 1×1-window shape);
and the threshold is shown to **discriminate** — the same formula applied to M3's 18 px and
34 px sizes comes out below 16′, so the assertion is measuring the type and not the panel. The
attached-monitor angles are pinned per font size, so a shrunk token or a changed geometry
fails the test rather than silently changing the record.

---

## 3. The element census, on the 55-inch panel the booth is required to provide

**This section is the requirement's geometry, not the delivered result.** Every visible
element is one of three font sizes, so the delivered verdict for all 24 items is §1.2's:
**44 px → 16.8′ cap at 2 m / 11.2′ at 3 m; 48 px → 18.3′ / 12.2′; 56 px → 21.4′ / 14.2′**,
with x-height below 16′ throughout. The table below names each item and gives its size on the
required panel, so a reader can map any row to the attached monitor's numbers.

Read from `docs/qualification/m3-capture/100pct.json` (the committed render) and computed by
the same method as the test. `cap` is the cap-height angle at **3 m on the 55-inch
requirement**, `x` the x-height angle; both are asserted there.

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

**On the required panel the whole always-visible screen passes, on both measures, at 3 m.**
The smallest element on it is 16.6′ of x-height (44 px), and the frame age — the figure spec
section 7 says must be legible at a glance — is **29.1′ of cap height**, 1.8× the threshold.
M3 measured the same headline at **8.5′** and every 18 px line at **4.6′**, below the acuity
floor.

**On the attached monitor it does not.** The same 44 px row is 16.8′ of cap at 2 m and 11.2′
at 3 m; the frame age is 21.4′ / 14.2′. That is the achieved result and it is the one this
document leads with. The 55-inch column exists so the booth requirement can be read against
the render, not so the pass can be borrowed from a panel the booth does not have.

**The two elements the screen exists for, named:** `streamStaleText_A`/`_B` and
`streamAgeText_A/B/C`. On the required panel they are **22.9′ cap / 16.6′ x** and **29.1′ cap
/ 21.1′ x**; on the attached monitor they are **16.8′ / 12.2′** and **21.4′ / 15.5′** at 2 m,
falling to **11.2′ / 8.1′** and **14.2′ / 10.3′** at 3 m. They are asserted *by name* against
both geometries — `test_the_staleness_line_and_the_per_camera_age_are_among_the_passing` (the
requirement) and the `attached_monitor` tests (the delivered screen) — and both are measured,
not merely present.

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

1. **The original verdict was judged against a panel the booth does not have.** The first
   version of this document measured the render against a hypothetical 55-inch 1080p panel
   (0.63416 mm/logical px) and reported the pass as the result. The monitor attached to this
   machine is a **DELL S3422DWG at 0.310 mm/logical px** — the 34-inch panel's 16:9 canvas is
   about 595 mm wide, the same as a 27-inch monitor — so the assumed scale was roughly **twice**
   the real one. On the actual monitor the screen **passes 16′ of cap height at 2 m (16.8′ at
   44 px) and fails at 3 m (11.2′)**, and its x-height does not clear 16′ at either distance
   (12.2′ / 8.1′). **This is the most important paragraph in this document**, and it replaces
   the first version's claim that the screen "reads at three metres".
2. **The 55-inch panel is now only a procurement requirement, and nothing gates it.** A booth
   that installs one clears 22.9′ / 16.6′ at 3 m and reads; a booth that installs anything
   under 55 inches does not (40 in → 16.7′ / 12.1′; the attached monitor → 11.2′ / 8.1′).
   `tools/preflight.py` does not measure panel size and cannot; the requirement is recorded in
   `app/theme.py`, here and in the presenter guide. A 55-inch panel is **not** the delivered
   result and must never be reported as one.
3. **The M3 arithmetic conflated 2 m and 3 m, and the first version of this review kept the
   error alive.** Its brief's "9.2 mm of cap height ≈ 44 px" is 16′ at **2 m**; 16′ at 3 m is
   13.96 mm, which on the attached monitor is a **63 px** Arial, not 44. The 44 px ladder is
   right for 2 m (it measures 16.8′), and it is right for 3 m only on the 55-inch requirement.
   The first version wrote that "the panel changed to 55 inches", which was the mistake: the
   panel did not change, the claim did. Said plainly because the two routes to "44 px" are not
   the same argument and only one of them is true here.
4. **`PipelineStrip.qml` was deleted, and its claim moved into prose.** Twelve 18 px labels
   could not be made legible at 3 m without a footer band the layout does not have, and the
   drawer cannot hold six stage names without eliding them. `drawerPipelineText` states the
   topology and the placement rule in words, and the `gpu` explanation states the measured
   per-device latency. A presenter who wants the diagram has the words.
5. **The seven footer `?` keys are gone; one button replaces them.** The M1 lesson was that a
   booth machine may be touch-only and a hover-only affordance is then dead, so the drawer has
   a named 44 px entry point in the header plus a click on any card, and F1 is not the only
   way in. The per-key proof moved to the drawer's own buttons, all seven of which are clicked
   by the test suite.
6. **The hover hint was dead and was removed.** It was anchored above the card (`parent.top`),
   and M3's `clip: true` on the card clipped it away — it could never be seen. Naming it in M4
   made `capture_ui` report it as **overflow**, which is how the dead affordance surfaced. It
   is removed rather than repaired: the drawer is the explanation surface and a hover-only hint
   is unreachable on a touch booth.
7. **A camera with a stated failure reason shows the reason instead of its staleness
   sentence.** One verdict line per camera is the budget, and the reason is the more specific
   statement; the age, its warning colour and the card's warning border still carry the
   staleness. In the seeded capture camera C is both stale and failed, and M3 showed both
   sentences. This is a change in what a passer-by can read and it is recorded here because
   trimming a line to make a layout fit without saying so is the failure this project keeps
   finding.
8. **The attract screen was rebuilt on the same ladder.** M3 Task 4's idle overlay used 40 /
   26 / 18 px, so a booth with nobody at it showed text at 9.4′ — below the acuity floor — on
   the screen a passer-by is most likely to see alone. Its title is now 56 px, its recorded
   label 48 px and its headlines 44 px, and the attract census is asserted to contain no size
   below the ladder.
9. **The drawer is 18 px, and that is deliberate.** It is the presenter's surface, read at
   arm's length, and its figures are the ones removed from the aisle surface. It is not part
   of the passenger legibility contract, and the M4 legibility test asserts the *live* screen.
   Writing this down because "the drawer is small" would otherwise be indistinguishable from
   "the drawer was forgotten".
10. **The threshold is a reading standard, not a measurement of this room.** ISO 9241-3's 16′
   is a sustained-reading minimum; it says nothing about the ambient light, the contrast ratio
   (not computed here), or how long a visitor actually looks. The palette is high-contrast by
   design (`#e8eef4` on `#141b23`), and no contrast ratio was measured in M3 or here.

---

## 7. What the tests assert, so a reader can check the claims

| claim in this document | test |
|---|---|
| the attached monitor's geometry is the measured one (80 x 33 cm, 2293x960, 0.310 mm/px) | `TheBoothScreenReadsAtThreeMetres::test_the_attached_monitor_geometry_is_the_measured_panel_not_the_booth_requirement` |
| **every element clears 16′ of cap height at 2 m on the attached monitor** | `…::test_every_element_clears_the_band_on_cap_height_at_two_metres_on_the_attached_monitor` |
| **every element fails at 3 m on the attached monitor**, at the pinned numbers | `…::test_the_attached_monitor_fails_at_three_metres_on_both_measures_and_is_pinned` |
| **no mixed-case sentence clears 16′ of x-height at 2 m** ("passes at 2 m" is cap-only) | `…::test_the_mixed_case_sentences_do_not_clear_the_band_at_two_metres_on_the_attached_monitor` |
| the 55-inch panel is a requirement that would pass at 3 m, not the delivered result | `…::test_the_fifty_five_inch_panel_is_a_requirement_not_the_delivered_result` |
| every visible named text clears 16′ on cap **and** x-height on the required panel | `…::test_every_visible_text_on_the_live_screen_clears_the_threshold` |
| the staleness line and the ages are among the passing on the required panel | `…::test_the_staleness_line_and_the_per_camera_age_are_among_the_passing` |
| the element set is the whole screen, not a legend | `…::test_nothing_visible_is_exempt_from_the_threshold` |
| the angle is measured from the render, not the token | `…::test_every_measured_element_has_an_ink_band_of_at_least_its_cap_height`, `…::test_the_font_ratios_this_test_divides_by_are_the_fonts_real_ones` |
| the threshold discriminates (M3's sizes fail it) | `…::test_the_threshold_discriminates_between_the_m3_sizes_and_the_shipped_ones` |
| the age is the largest thing on the screen | `…::test_the_largest_thing_on_the_screen_is_a_frame_age` |
| the four captures overflow, truncate and omit nothing | `CanvasLayoutAtFourScales::test_every_scale_reached_the_state_it_was_asked_to_capture` |
| the drawer holds the removed figures and does not overflow | `PresenterDrawerOnTheScreen` |
| the layout tokens are in `app/theme.py`, not the QML | `ClippingIsTheDecision::test_the_layout_tokens_are_in_theme_and_not_in_the_qml` |

**Mutation checks.** Each new assertion was shown to fail by constructing its fault
(a wrong mm/px, a wrong cap ratio, a wrong x-height ratio, a wrong distance set, a shrunk
token, and a test that defaults to the 55-inch conversion); the log is in the task report
(`task-1-report.md`), not here, because a mutation log is evidence about the tests rather
than about the screen.
