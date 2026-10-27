# M3 — the readability review at 2–3 m, and the 1920×1080 DPI gate

**What this document is.** The one review M3's plan asks for and M0–M2 could not produce: for every
element a passer-by has to read on the booth screen, **the rendered height in millimetres at
1920×1080, the angular size at 2 m and at 3 m, and a verdict against a stated legibility
threshold** — including the elements that **fail**.

**What it is not.** It is not a checklist, and it is not a pass. Read §4 first if you read nothing
else: **at 2 m and at 3 m, on the display this review assumes, every element on the screen is either
marginal or fails, and all of the 18 px detail text is below the acuity floor at 3 m.** The
measurement is sound; the screen is not readable from where a passer-by stands. That is M4's
finding to act on and M3's to record.

Three things this document deliberately does **not** do:

- it does **not** claim the four committed captures prove legibility at 125/150/200 %. They do not,
  and that is stated again in §6.1;
- it does **not** re-seed the captures to match the live screen. The seed understates the screen —
  measured, §6.4 — and the fix is to record the discrepancy, not to make the evidence look better;
- it does **not** close M3 exit criterion 6 (the input-acknowledgement blindness demonstration) or
  any carried gate. §9 says which criteria are met and which are not.

Every number below is either measured by a command in §11 or read from a committed file, and each
is labelled with which. Assumptions are labelled **assumed** and never presented as measurements.

---

## 1. The display assumption, stated before any millimetre appears

**The physical display size is not established anywhere in this repository.** M0 measured the
desktop's *logical* size (`2293 × 960`) and the panel's model (`DELL S3422DWG`); it did not measure
a diagonal, a pixel pitch, or the size of any booth display. The spec fixes a **1920 × 1080 logical
canvas**, not a physical screen. So this review is parameterised by millimetres per logical pixel,
and the parameter is **assumed**.

**Primary assumption: `0.310 mm` per logical pixel** — a 1920 × 1080 canvas **595 mm (23.4 in)
wide**.

Why that value, and not a rounder one:

| anchor | source | canvas width | mm/px |
|---|---|---:|---:|
| a 27-inch 16:9 monitor at native 1920×1080 | standard panel geometry: 685.8 mm diagonal → 597.7 mm width | 597.7 mm | **0.3113** |
| the **measured** DELL S3422DWG, 34-inch 21:9 3440×1440, with a 16:9 canvas **pillarboxed to its height** | M0's monitor identity (`DELD129`) + panel geometry: 333.5 mm height → 592.9 mm width at 16:9 | 592.9 mm | **0.3088** |

The two independent anchors agree to **0.8 %**, and their midpoint is 0.310 mm/px. That is the
number used in §4. **It is an assumption about the booth, not a measurement of it.**

### What changes if the assumption is wrong

Angular size scales linearly with panel width. The whole table at other plausible panel sizes:

| panel | mm/px | 18 px cap | 26 px cap | 34 px cap | 40 px cap |
|---|---:|---|---|---|---|
| 24-inch 16:9 (531 mm) | 0.2768 | 6.2′ / 4.1′ | 9.0′ / 6.0′ | 11.4′ / 7.6′ | 13.8′ / 9.2′ |
| **27-inch 16:9 (598 mm) — assumed** | **0.3113** | **7.0′ / 4.6′** | **10.2′ / 6.8′** | **12.8′ / 8.6′** | **15.5′ / 10.3′** |
| 32-inch 16:9 (708 mm) | 0.3690 | 8.2′ / 5.5′ | 12.0′ / 8.0′ | 15.2′ / 10.1′ | 18.4′ / 12.3′ |
| 34-inch 21:9 S3422DWG, pillarboxed (593 mm) | 0.3088 | 6.9′ / 4.6′ | 10.1′ / 6.7′ | 12.7′ / 8.5′ | 15.4′ / 10.3′ |
| 40-inch 16:9 (886 mm) | 0.4612 | 10.3′ / 6.9′ | 15.1′ / 10.0′ | 19.0′ / 12.7′ | 23.0′ / 15.3′ |
| 55-inch 16:9 (1218 mm) | 0.6342 | 14.2′ / 9.4′ | 20.7′ / 13.8′ | 26.2′ / 17.4′ | 31.6′ / 21.1′ |

*(arcmin at 2 m / arcmin at 3 m; cap height from §3.)*

**What survives a wrong assumption, and what does not:**

- **Survives:** the ordering. The 34 px age headline is always the largest live number after the
  40 px title; 18 px detail text is always the smallest; nothing is ever bigger than the title.
- **Survives:** the 18 px problem at 3 m on anything smaller than about a 55-inch panel — it is
  below the 5′ acuity floor on every panel up to 40 inches, and still under 10′ on a 55-inch.
- **Changes:** the verdict on the 34 px headline. It is `marginal` at both distances on a 27-inch
  screen, **passes at 2 m on a ≥ 32-inch**, and **passes at 3 m only on a ≥ 55-inch** panel.
- **Changes:** the title. `marginal` on a 27-inch, a clear pass on a ≥ 40-inch.

So on a booth wall with a 55-inch 1080p panel the *three age headlines* would be legible at 3 m and
everything else still would not. On the 27-inch anchor, **nothing on the screen clears 16′ at
either distance.**

### The font size that would be needed

Cap height is 0.716 em for Arial, so the font pixel size needed to hit a threshold is
`cap_mm / 0.716 / (mm_per_px)`:

| panel | 16′ at 2 m | 16′ at 3 m | 5′ (acuity floor) at 3 m |
|---|---:|---:|---:|
| 24-inch | 47 px | 70 px | 22 px |
| **27-inch (assumed)** | **42 px** | **63 px** | **20 px** |
| 32-inch | 35 px | 53 px | 17 px |
| 40-inch | 28 px | 42 px | 13 px |
| 55-inch | 21 px | 31 px | 10 px |

The shipped ramp is `TYPE_DETAIL 18`, `TYPE_LABEL 26`, `TYPE_NUMERAL_SMALL 34`, `TYPE_HEADLINE 40`.
**On the assumed display, the ramp's *largest* size (40 px) does not reach the 3 m comfort
threshold either.** M4's choice is a bigger panel, a bigger ramp for the booth, or an explicit
decision that a passer-by reads only the three amber numbers — and that decision belongs in the
design, not in a QML patch made to satisfy a review.

---

## 2. The thresholds, and why these ones

Three bands, each from a stated rule, none invented for this document:

| band | character height | where it comes from |
|---|---|---|
| **PASS** — comfortable sustained reading | **≥ 16 arcmin** | ISO 9241-3's ergonomic minimum character height for VDT work is 16′ (20–22′ preferred). This is the "a passer-by can read it without stopping" threshold. |
| **marginal** — readable if you stop and look | 10′ – 16′ | Half-way house, and the 20/40 band: a 5×5 character whose strokes subtend 2′ needs 10′ of height. |
| **FAIL** — not comfortably readable | 5′ – 10′ | Above the raw acuity limit but below any reading standard. |
| **FAIL (below acuity)** — not resolvable at all | **< 5′** | A Snellen 20/20 letter is 5 × 1 arcmin. Below 5′ of character height a person with *normal* vision cannot resolve the letterform; no amount of contrast or staring helps. |

Character height here is **cap height**, because the numbers a passer-by reads are digits, and a
digit's height is the cap height. For mixed-case sentences the lowercase **x-height is 0.72 of the
cap height**, so every sentence in §4 is worse than its row says by about 28 %: the 18 px
`STALE - this picture is …` line has a 9.3 px x-height, i.e. **5.0′ at 2 m and 3.3′ at 3 m** — at
3 m it is below the acuity floor even setting aside the cap height.

These thresholds are this review's, stated so a reader can disagree with them and recompute. Nothing
in `app/`, `config/` or the tests was changed to meet them.

---

## 3. The type census on the screen — and what the census does not cover

**The pinned census is `{18: 29, 26: 1, 34: 3}`**, read out of the running scene graph at 100 / 125 /
150 / 200 % and re-read here from the committed records. It counts every item **whose text is not
empty**, which is the definition the pinning test uses
(`tests/test_capture_ui_canvas.py::...test_the_font_census_is_the_one_on_record`):

```
census over non-empty: {18: 29, 26: 1, 34: 3}
blank named items: streamReasonText_A, streamReasonText_B, tileCaptionText_A   (all 18 px, all invisible)
```

So the raw `font_pixels` map has 36 entries (32 at 18 px, 1 at 26 px, 3 at 34 px) and three of the
18 px entries are **invisible and blank**, which is why the census is 29 and not 32. All four scales
report the identical census.

**A finding: the census covers *named* items, and the screen has legible text that is not named.**
`strings` is `objectName → text` for every **named** item the walk reaches — that is the harness's
own stated contract in `tools/capture_ui.py`. The following text is on the 100 % screen, is outside
the census, and was therefore outside the guarantee the capture README stated before this commit
("No other size appears on the screen"):

| text | font size | where in the QML | in the census? |
|---|---:|---|---|
| `Intel Edge AI Roadshow` (the title) | **40 px** (`TYPE_HEADLINE`) | `Main.qml`, header | **no** — no `objectName` |
| `Recorded footage · live inference` (the mandatory honesty label) | 26 px (`TYPE_LABEL`) | `Main.qml`, header | **no** |
| `Camera A` / `active` / `failed` (the card heading and state) | 26 px | `ComputeCard.qml` | **no** |
| `events:` (the row's prefix) | 26 px | `Main.qml`, footer | **no** |
| `REDUCED SCOPE. No product-transfer event is detectable on ret…` (the scope note, **elided**) | 18 px | `Main.qml`, header | **no** |
| the pipeline strip's 6 stage labels, 6 device lines and 5 arrows | 18 px | `PipelineStrip.qml` | **no** |
| `status: running` | 18 px | `Main.qml`, footer | **no** |
| the 7 drawer explanation button labels | 18 px | `Main.qml`, footer | **no** |

**`40 px` is a real size on the screen and the census cannot see it.** The census's own claim is
narrower than the sentence that quotes it, and this review measured the unnamed items directly out
of the PNG instead (§4, §11). The one-line correction to the capture README is made in this commit:
the census is a census of named items.

Two of these omissions matter beyond arithmetic:

- **The scope note is elided on the booth screen.** The header shows `REDUCED SCOPE. No
  product-transfer event is detectable on ret…` — cut mid-word by `elide: Text.ElideRight` at a
  width of `screen.width * 0.68 * 0.42`. The full wording exists one click away in the `zone`
  explanation, and the drawer states it verbatim, so the honest scope note **is** on the screen.
  But the always-visible header carries a truncated version of the sentence that says what the demo
  does not do, and no harness reports that truncation because the item has no name. That is the M3
  Task 1 failure shape ("a silently shortened error is a hidden error") surviving in a different
  string.
- **The 40 px title and the 26 px honesty label are the two largest things on the screen**, and
  neither is in the census the report quotes. If a future change shrank or elided them, the font
  census would not notice.

---

## 4. The readability table — the measurement

Method, in one paragraph, because the numbers are only worth what the method is: the **font pixel
size** comes from `font_pixels` in the running scene (`100pct.json`); the **cap height** is
`0.716 × font px` for Arial, **corroborated against the PNG** by measuring the ink band inside each
item's reported extent (threshold: brightest channel > 70), which is also how the unnamed items were
measured. **Millimetres** are `cap_px × 0.310`. **Arcminutes** are `2·atan(h / 2d)` at d = 2 m and
d = 3 m. The script is in §11; its output is what is pasted here.

| element (what the passer-by must read) | font px | cap px | mm | at 2 m | at 3 m | verdict at 2 m | verdict at 3 m |
|---|---:|---:|---:|---:|---:|---|---|
| Title `Intel Edge AI Roadshow` | 40 | 29 | 8.99 | 15.5′ | 10.3′ | marginal | marginal |
| `Recorded footage · live inference` — the honesty label | 26 | 19 | 5.89 | 10.1′ | 6.7′ | marginal | **FAIL** |
| `device: CPU` — the placement proof | 26 | 19 | 5.89 | 10.1′ | 6.7′ | marginal | **FAIL** |
| `Camera A` heading + `active`/`failed` state | 26 | 19 | 5.89 | 10.1′ | 6.7′ | marginal | **FAIL** |
| `events:` prefix | 26 | 19 | 5.89 | 10.1′ | 6.7′ | marginal | **FAIL** |
| **the frame-age headline `11.7 s` (×3)** | 34 | 24 | 7.44 | 12.8′ | 8.5′ | marginal | **FAIL** |
| `STALE - this picture is 11.7 s old; the limit is 2 s` (×3) | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |
| per-camera badge `CPU` / `no device is running` (×3) | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |
| `backlog 187 frames` (×3) | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |
| `analysis 13.44 fps` (×3) | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |
| `window: 256 analysed frames` (×3) | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |
| `analysis fps 13.44 fps · frame age p50 … p95 …` (×3) | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |
| events row `zone_occupancy … candidate_transfer … candidate_return` | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |
| pipeline strip `decode` / `preprocess` / `Infer` / `track` / `event` + `CPU` | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |
| `status: running` / `telemetry unavailable` | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |
| scope note `REDUCED SCOPE …` (elided) | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |
| drawer explanation buttons | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |

*Measured ink bands that corroborate the cap-height column: `11.7 s` 27 px (halo included) at
34 px; `CPU` 13 px at 18 px; `backlog 187 frames` 16 px ascender-to-descender at 18 px; `device: CPU`
20 px (dark-on-green) at 26 px; `Camera A` 20 px at 26 px; pipeline `decode` 13 px at 18 px.*

### The failures, named

**At 3 m, on the assumed 27-inch display, not one element on the screen clears 16′.**
Specifically:

1. **The three frame-age headlines — the numbers spec §7 makes the whole screen about — are
   12.8′ at 2 m and 8.5′ at 3 m.** They are the best case on the screen and they still fail the
   comfort threshold at both distances. This is the finding that matters: "the frame-age number
   must be legible at a glance" is a spec requirement, and at 3 m it is not.
2. **All 18 px text fails at 2 m and is below the acuity floor at 3 m.** That includes the sentence
   that says **why** the number is what it is (`STALE - this picture is 11.7 s old; the limit is
   2 s`), the per-camera device badges, the backlog, the analysis rate, the window size, the event
   row, the whole pipeline strip and the status line. A passer-by at 3 m sees three amber numbers
   and three coloured badges and cannot read a word around them.
3. **The scope note and the honesty label are in the failing set.** The two pieces of text that say
   "this is recorded footage, the scope is reduced" are 18 px (and elided) and 26 px (marginal).
   The screen's *honesty* is what a passer-by most needs to be able to read, and it is the text
   that fails.
4. **The title is marginal at both distances** — it is 40 px and the ramp has nothing larger.

**At 2 m** the four 26 px elements and the 34 px headline are `marginal` (10.1′–12.8′): readable if
a person stops and looks, not readable at a glance. Everything at 18 px fails.

**These are recorded as failures, not fixed by shrinking a font or re-seeding a capture.** The
correct place for the fix is M4: the booth's physical panel, the ramp for a 2–3 m viewing distance,
and a decision about which numbers a passer-by is *entitled* to read at that distance. A font
enlarged here would have needed a new layout at 1920×1080 — which is exactly the kind of change that
would have quietly broken the four-scale evidence this milestone exists to produce.

**One thing the table does not say, and this document will not imply:** it does not say the screen
is blank or unreadable at arm's length. At 0.5–1 m (a presenter standing at the screen) the 18 px
text is **14–28′** and perfectly legible. The failure is specifically **at the 2–3 m the brief names**.

---

## 5. What the four committed captures are — and are not

### 5.1 Rendered, and measured from the image

| file | pixels (measured from the PNG) | JSON `pixel_width × pixel_height` | logical | dpr (`QT_SCALE_FACTOR`) | `image.devicePixelRatio()` |
|---|---|---|---|---|---|
| `100pct.png` | **1920 × 1080** | 1920 × 1080 | 1920 × 1080 | 1.00 | 1.00 |
| `125pct.png` | **2400 × 1350** | 2400 × 1350 | 1920 × 1080 | 1.25 | 1.00 |
| `150pct.png` | **2880 × 1620** | 2880 × 1620 | 1920 × 1080 | 1.50 | 1.00 |
| `200pct.png` | **3840 × 2160** | 3840 × 2160 | 1920 × 1080 | 2.00 | 1.00 |

Verified in this session by decoding each PNG with OpenCV and comparing against its JSON — the file
size and the record agree exactly at all four scales. **`QImage.devicePixelRatio()` is 1.00 at every
scale**, including the 3840 × 2160 grab, which is why the pixel size is read from the image and never
from the scale factor or from `dpr`. A gate that read `dpr` would pass at 100 % and fail at 200 %.

**All four scales really were rendered.** That is the half of the DPI gate the render can settle.

### 5.2 Identical strings and extents at every scale — and what that does *not* mean

Measured: `strings` and `extents` are **identical** at 100/125/150/200 %, and `overflow`,
`truncated`, `absent`, `unreachable` and `geometry_unavailable` are **empty at all four**.

That is **structurally correct**: a scale factor is a rendering scale, not a layout change. Qt lays
out the same logical scene and rasterises it into more device pixels.

**It is not evidence that text is legible at 125/150/200 %.** This document does not read it as a
DPI pass, and the committed README, the M3 ledger (Ruling 2) and the capture JSON all already say
so. What the four files vary is **pixel density**; what legibility depends on is **the physical size
of the canvas and the observer's distance**, which no capture encodes. The legibility question is
answered in §4 and nowhere else in this milestone.

### 5.3 `glyph_pixels` spot-check, and the self-check that it is measuring device pixels

For each scale, the number of non-background pixels inside the same named item's reported extent:

| item | 100 % | 125 % | 150 % | 200 % | ratio 200/100 |
|---|---:|---:|---:|---:|---:|
| `card_A` | 121 847 | 190 269 | 274 750 | 487 388 | **4.0000** |
| `streamAgeText_A` | 11 932 | 18 816 | 26 847 | 47 728 | 4.0000 |
| `thumb_B` | 23 383 | 35 790 | 53 933 | 94 082 | 4.0238 |
| `streamReasonText_C` | 9 760 | 15 300 | 22 020 | 39 120 | 4.0082 |
| `deviceBadgeText` | 4 290 | 6 623 | 9 675 | 17 160 | 4.0000 |

**The counts scale with the *area* ratio, not the linear ratio** — 4.000× from 100 % to 200 %, not
2×. That is the arithmetic proof that the count is indexing the **device-pixel** image: the
first version indexed a 3840 × 2160 buffer with logical extents and reported `0` for plainly drawn
items at three of the four scales. **76 of the 77 glyph entries are positive; the one zero is**
`tileCaptionText_A`, which is **invisible and empty** on this screen (the hero has no caption), so
zero is the correct answer for it — a zero that is a fact, not a bug.

*A note on provenance: the M3 ledger (Ruling 3 — a gitignored file) quotes `card_A` 128 260 and
`streamReasonText_C` 14 640 for this screen. Those are from an earlier render: the committed records
were re-rendered after the layout work, and `card_A` in the committed JSON is 121 847. The table
above is read from the committed JSONs and is the one this review stands behind. Same 4.0000 ratio,
different absolute counts — which is itself a reminder to quote the file, not the prose.*

### 5.4 **No capture contains a video frame**

Look at `100pct.png`: the hero reads `no frame from Camera A yet`, and both thumbnails read
`no frame from Camera B/C yet`. **All four committed captures have empty video surfaces** — the
seed drives the strings and the layout, not the picture. `thumb_B`'s 23 383 "glyph" pixels are the
`no frame from Camera B yet` sentence and the tile's own chrome inside the reported extent, not
pixels from a camera.

So the four captures prove, to the standard they claim:

- the **layout** at 1920×1080 and at three scale factors (extents, clipping, no overflow);
- the **strings**, read out of the running scene graph in a closed audit window
  (`opened_during_read: []`, `read_window_closed: true` — the read did not parse QML source);
- the **pixel size** of each render, measured from the image;
- the **font census** of the named items.

They do **not** prove the video surface, the overlay boxes, or that a frame is painted at the right
scale. Those were measured live at M2 (`docs/qualification/M2-qualification.md`: three tiles, three
independent revisions, `image://frames/A?v=269` etc.). **This review says so plainly rather than
letting "committed screenshot" imply "the screen, photographed".** A capture with a seeded frame
would be better evidence and is not in this milestone's scope.

### 5.5 The seeded staleness **understates** the live screen — measured, and it holds

The captures are **seeded**, not a live run. The staleness figures in them (`11.7 s`, `10.7 s`,
`11.1 s`) come from `docs/qualification/m2-headless/allcpu-steady-120s.json`, which was produced by
`tools/route_stress.py::measure_steady_state`. **That function was read as part of this review.** It
builds a bare `Supervisor(device_ids=("CPU","CPU","CPU"))`, a `_Feeder`, and a **`QCoreApplication`**
— **no Qt window, no `QQmlApplicationEngine`, no `FrameImageProvider`, no render loop, and no
`_on_result`/`_store_image` on a Qt main thread.** So the seed's frame age is the age of a picture
measured on a *lighter process* than the one the booth runs.

The claim was "the seed understates what the booth shows". **It holds, and by more than the missing
UI path alone would suggest. Three measurements, same condition (three CPU streams, 15 fps offered
per stream, no churn), on this machine:**

| path | hold | frame age p50 (worst stream) | p95 | backlog | analysed |
|---|---|---:|---:|---:|---:|
| **the committed seed** (M2, 2026-10-03, bare `Supervisor`) | 120 s | **11 319 ms** | 12 286 ms | 173–187 | ~13.5 of 15 fps per stream |
| **bare `Supervisor`, re-measured today** (`tools.route_stress --steady-seconds 120`) | 120.2 s | **4 465 ms** | 4 848 ms | 58–73 | 1729–1744 of 1802 |
| **the live windowed app** (`tools.input_ack_under_load`, `app.main.run_windowed`, 130 s warm-up) | 163.5 s / 193.6 s | **40 111 / 48 572 ms** | 43 217 / 51 682 ms | 642–836 | 5 172 / 6 145 frames, all three streams |

**What this establishes:**

1. **The seed understates the screen by 3.4×–4.5×.** At the same condition, the committed captures
   say 10.7–11.7 s; the live windowed application says **40.1 s and 48.6 s** over two windows of one
   process. The screen a passer-by looks at is **thirty to fifty seconds stale**, not eleven.
2. **The cause is the presentation path, not this machine.** Today's bare `Supervisor` reaches
   **4.5 s** — *better* than M2's committed 11.3 s — while the windowed app on the same machine,
   same day, same clip, same three CPU workers reaches **40–49 s**. The difference between the two
   today is the Qt window, the provider, the render loop, and the UI-thread work per result. The
   seed was measured with none of them.
3. **The gap is larger than the extra hold explains.** The live windows ran 163–194 s rather than
   120 s, which at M2's accumulation rate would add ~5 s; the measured gap is ~29–37 s, and the
   backlog is 642–836 frames against a seed's 173–187.

**What it does not establish:** the *share* of the gap between "the UI path costs worker
throughput" and "this machine has drifted slower since 2026-10-03" is not isolated. Today's headless
run is **faster** than M2's committed one, which argues against drift being the cause, but two
measurements on one day do not settle a cause. **The direction is not in doubt; the attribution is
not complete.** It is also worth recording that the live windowed run **dropped 32 frames** where
the headless holds dropped none — the screen sheds frames as well as lagging.

**The response is not to re-seed.** The captures keep the figures their seed file actually contains,
this review says what those figures understate, and the honest next step is an M4 capture of a
**live** screen under the booth's real load — which needs the attract-mode idle timer handled, since
a 150 s live capture on this tree lands in attract mode after 90 s (observed: a live 150 s capture
for this review came back showing the attract overlay, not the live screen).

### 5.6 The 960 px display copy: does it change what the captures assert?

`app/engine/worker.py` now reduces each successful result to a **960 px wide** display copy **after**
`infer_once`, scaling the detection boxes with the frame. The detector still consumes the
full-resolution frame. Numbers, from the committed extents and the geometry of `VideoTile.qml`
(`PreserveAspectFit`):

| surface | extent | 16:9 picture inside it | before (1920 px copy) | after (960 px copy) |
|---|---|---:|---:|---:|
| hero | 1273 × 538.4 | **957 × 538** | 0.498× downscale | **0.997× — essentially 1:1** |
| thumbnail | 628.5 × 105.6 | 187.7 × 105.6 | 0.098× downscale | 0.196× downscale |

**What it changes: nothing the four captures assert.** The captures contain no frame pixels at all
(§5.4), the extents are unchanged (this review compared them), the font census is unchanged (the
pinning test passed in this session's suite), and the overlay boxes are drawn in the QML from
normalised coordinates, so their on-screen position cannot move when the frame's pixel count does.

**What it changes that nothing committed measures:** the hero now paints at ~1:1 instead of
half-scale — the picture should be **sharper**, not softer — while the thumbnails still downscale ~5×,
which is fine for 187 px. It also means the 1920 × 1080 "frame" a reader might imagine the committed
capture would contain is now a 960 × 540 display copy in the live app. **No committed artefact
shows either one**, which is the gap §5.4 already records.

---

## 6. The gates, run in this session

Run on the branch `main`, on a machine that was asserted clean before each timing run (the census in
`tools.input_ack_under_load.py`, or `Get-Process python` returning 0). Output pasted verbatim.

### 6.1 `tools\verify_sources.py`

```
STATUS  ID                    KIND      DETAIL
----------------------------------------------
PASS    yolo11n-fp16          xml       xml+OpenVINO-IR, 412503 bytes, sha256 4845b46cc9b6ac379a44484258b7ed98f836ed117d91f127c7728ea068250a95
PASS    yolo11n-fp16:weights  binary    binary, 5283218 bytes, sha256 f14632518ebb4e312a89c26faa4cd2f95d4264b6f5f22aa6e0d5707b4667863e
PASS    yolo11n-fp16:labels   text      text+utf-8, 80 lines, 701 bytes, sha256 82bf6a434f132c4d78534c0f2e0f991fc22c73cd911f6d5fc74a418c39c339ec
PASS    retail_checkout       mp4       mp4+ftyp, 64996307 bytes, sha256 e107db69155af4774f907abe8ddbf24493dfcf9d81d51f3e71ff25300e4f5906
PASS    store_aisle_alt       mp4       mp4+ftyp, 9214573 bytes, sha256 3526fee39cac70d6366471e4324a6d63b337c8f8ab99c015521bee0d60ed6e04
PASS    queue_alt             mp4       mp4+ftyp, 3291918 bytes, sha256 a5964aa259099a482a8b360ffc2c57b5a30f84d5919236a4dad01f8e929ac07c

Source verification: 6 PASS, 0 FAIL
```

exit 0.

### 6.2 `tools\preflight.py`

```
PASS  Assets: pinned content verification
PASS  Assets: sanity fixture
PASS  Assets: model IR pin
PASS  Runtime: OpenVINO and Python
PASS  Budget: free disk on the repo volume
PASS  Budget: free physical memory
PASS  Device available: CPU
PASS  Device available: GPU
PASS  Device available: NPU
PASS  Device discovery cost
PASS  Recorded qualification run
PASS  Recorded per-device compile, latency and placement
PASS  Sanity fixture on CPU
PASS  Recorded sanity fixture, per device
WARN  Telemetry: in-process PDH GPU/NPU utilisation
PASS  Telemetry: recorded LUID-to-device map
WARN  Telemetry: LUID-to-device attribution and hardware identity
PASS  Startup: this gate's import to the first inference
Summary: 16 PASS, 2 WARN, 0 FAIL - 18 total rows
Status: Limited
Exit code: 0. Ready and Limited both exit 0, because in both the demo runs; only Needs attention exits 1.
```

The two `WARN` rows are the two carried telemetry items (in-process PDH value read; derived
LUID attribution), unchanged since M0 and M2. Nothing was relaxed to get this output.

### 6.3 `python -m app.main --selftest`

```
  ok   preprocess           letterbox 640x640 pad=114 scaled=640x360
  ok   infer                29.19 ms on confirmed ['CPU']
  ok   decode-track-event   1 zone_occupancy event(s) from a synthetic presence
  ok   render-metadata      unavailable renders as unavailable, and no device is claimed before an inference confirms one
PASS
```

exit 0.

### 6.4 `python -m unittest discover -s tests`

```
Ran 1376 tests in 522.476s

OK
```

exit 0. **Wall clock 523.5 s**, measured around the process with
`System.Diagnostics.Stopwatch`; **no skips** — including the live input-acknowledgement window, which
reported its seven assertions green in this run.

**Four runs, reported in full because three of them are not the one above:**

| run | tests | result | wall | note |
|---|---:|---|---:|---|
| 1 | 1368 | **FAILED (failures=1)** | 520.3 s | the known RSS-ratio flake `test_route_stress...test_nothing_grows_that_the_retained_records_do_not_account_for` — pre-existing, unchanged, passes in isolation (M2 recorded it failing 3 times in 15 full runs) |
| 2 | 1368 | OK (skipped=1) | 518.8 s | the live window was **refused** by the clean-machine census and the suite skipped with the reason, which is the blocker fix working — see §7 |
| 3 | 1376 | OK | 524.9 s | after the census fix in §7, which removed the false competitor |
| 4 | **1376** | **OK** | **523.5 s** | **the gate run** — the tree this document is committed with |

Run 1 is not hidden and was not re-labelled: the flake is pre-existing, it is not the blocker this
task was given, and it was not fixed here. Run 4 repeats run 3 after a **docstring-only** edit to
`tools/input_ack_under_load.py`, so the gate figure belongs to the committed tree. The **8 extra
tests** between run 2 and run 3 are the seven live-window assertions (previously collapsed into a
single skip) plus the new census test.

### 6.5 `--streams 3` in headless mode — refused, with a reason

```
$ .venv\Scripts\python.exe -m app.main --headless --streams 3 --seconds 1
--streams 3 is not supported by --headless: the headless run builds one supervisor for one stream
and its summary is single-stream, so accepting this flag would run one camera while claiming 3.
Use --windowed for the multi-camera layout, or omit --streams.
EXIT=2
```

This used to be **silently ignored** — `--headless --streams 3` ran one stream and said nothing,
which is why headless reported ~301 frames whether asked for 1 or 3 cameras. **Refusing with a
stated reason is the correct behaviour**, exit code 2, and it is recorded here rather than fixed
into agreement.

---

## 7. The blocker carried into this task, and what was actually wrong

The suite arrived at **1362/1363 with one error**: `TheCoReportedWindowOnTheRealApplication`'s
`setUpClass` raised because Task 3's clean-machine guard saw a `python.exe` outside the run's process
tree appear during the window. Task 3's report said the test "takes a census and skips with the
reason". It did not, for the mid-window path: `measure()` raised a bare `RuntimeError`, so a busy
machine turned the suite red with a message that read like a code defect.

**Three changes, and the refusal was not weakened:**

1. **A typed refusal.** `MachineNotClean(RuntimeError)` is raised by `assert_machine_clean()` and by
   `measure()`'s mid-window census. Being a `RuntimeError` subclass, **every existing caller still
   refuses exactly as before**; `measure()` still returns **no row** on a dirty machine, so a figure
   measured on a dirty machine still cannot reach a record. The type only lets a caller tell "the
   machine was busy" (skip) from "the probe measured nothing" (error).
2. **The live test translates it into a skip, with the reason and with the explicit statement that
   no figure was recorded**, and names the experiment to run instead. Every other `RuntimeError`
   still errors — asserted by a test that fails if the skip is widened to all of them.
3. **The root cause of the false positive: the census took two process-table snapshots.** It asked
   `_our_process_tree()` (snapshot A) which PIDs were ours and `python_processes()` (snapshot B)
   which PIDs were Python. A child of ours that started between the two reads was in B's list but
   not in A's descendant set — so **our own measurement worker was reported as a competitor**.
   Measured: the census refused a clean run with exactly one extra PID, twice, on a machine that
   idles with zero `python.exe`, and the probe spawns three workers a moment after it starts. The
   census now takes **one** snapshot and answers both questions from it. **This is the change that
   makes the live window run in the suite** (run 3 above, no skips).

**Mutation checks: 6 mutants, 6 killed, 0 controls.** Each mutant is a plausible way to lose the
behaviour, and each is killed by a named test:

| # | mutant | killed by |
|---|---|---|
| A | the skip translation is removed (the refusal escapes as an error again) | `…test_a_window_refused_mid_run_skips_with_the_reason_and_names_no_figure` |
| B | the skip is widened to every `RuntimeError` | `…test_any_other_failure_is_still_an_error` |
| C | `MachineNotClean` stops being a `RuntimeError` | `…test_the_refusal_is_a_typed_runtime_error` |
| D | the mid-window census raises a bare `RuntimeError` again | `…test_a_competitor_seen_mid_window_returns_no_row` |
| E | the pre-flight census no longer refuses before spawning | `…test_an_already_dirty_machine_skips_before_anything_is_spawned` |
| F | the census goes back to two snapshots | `…test_the_census_reads_one_snapshot_so_our_own_new_child_is_not_a_competitor` |

One mutant initially appeared to survive and was not: the mutation harness was reading a **stale
`.pyc`**. With the bytecode deleted before each run, all six are killed deterministically. Recorded
because "a survivor" and "a mutant that never ran" have looked identical in this project before
(Ruling 7).

---

## 8. The 1920×1080 DPI gate — verdict

**The cell is `passed with documented failures`.** It is a **new value** in
`docs/qualification/qualification-matrix.csv`, added the same way M2 added `failed` when it needed a
value the vocabulary did not have. Justification, both halves:

1. **All four scale factors were really rendered.** The pixel dimensions are measured from each
   saved image, match their records, and `QImage.devicePixelRatio()` is 1.00 at every scale, so the
   size signal is the raster, not a variable read back. `overflow`, `truncated`, `absent`,
   `unreachable` and `geometry_unavailable` are empty at all four; the font census is identical at
   all four. That is the gate.
2. **The readability review found failures, and they are documented here** (§4). Per this task's
   instruction, that is the second admissible path to a non-`passed` cell rather than to a silent
   `passed`.

**Why not `not passed`:** the DPI gate is the render-and-scale gate, and it is met; the readability
finding is a different claim with a documented failure attached. **Why not a clean `passed`:** the
run would have to be rounded up, and M2 set the standard when it **downgraded** an M0 `passed` to
`not tested` because the evidence did not support it. The same standard applies in the other
direction.

**Why not `failed` on the readability half:** the failure is a *design* finding (type sized for
arm's length, not 2–3 m), not a regression. It belongs to M4, as the brief says: "a booth screen
with unreadable numbers is a finding for M4".

---

## 9. M3 exit criteria

| # | criterion (plan §) | verdict | evidence |
|---|---|---|---|
| 1 | Renders at 1920×1080 with **no overflow and nothing pushed outside the canvas** at 100/125/150/200 % | **PASSED** | four committed captures, pixel sizes measured from the images (§5.1); `overflow`/`absent`/`unreachable`/`geometry_unavailable` empty at all four; the four-scale layout tests green in the 1376-test run |
| 2 | Every number a passer-by must read is **legible at 2–3 m, or the failure is documented** | **MET IN ITS SECOND FORM ONLY — the numbers are not legible** | §4: nothing clears 16′ at 3 m on the assumed display; the 34 px age headline is 12.8′ at 2 m and 8.5′ at 3 m; all 18 px text is below the acuity floor at 3 m. Documented, not fixed |
| 3 | A **committed screenshot** exists at each scale factor, with its strings read out of the scene graph | **PASSED** | `docs/qualification/m3-capture/{100,125,150,200}pct.{png,json}`, committed by M3 Tasks 1-2 and
re-rendered at Task 5, unchanged by this commit; `opened_during_read: []`, `read_window_closed: true`, `walk_complete: true`, 33 non-empty strings at every scale |
| 4 | Attract mode shows **no number it cannot date**, and restores the interrupted state exactly | **PASSED** (delivered by Task 4, re-verified here) | three dated headlines with their sources (`recorded 2026-10-03 · docs/qualification/M2-qualification.md`, etc.) and **no live figure** — `tests/test_ui_contract.py::AttractModeOnTheScreen::test_no_live_figure_appears_in_attract_mode` and `…test_leaving_attract_restores_the_live_screen_exactly` green in this run; a live 150 s capture for this review landed in attract mode and showed those dated claims, not a live reading |
| 5 | The presenter drawer **moves nothing**, and states the retail scope note **verbatim** | **PASSED** (delivered by Task 5, re-verified here) | `…PresenterDrawerOnTheScreen::test_opening_the_drawer_moves_nothing`, `…test_closing_the_drawer_puts_every_item_back_where_it_was`, `…test_the_drawer_states_what_the_delivered_event_is` / `…_is_not`, `…test_the_drawer_states_that_gpu_and_npu_utilisation_is_unavailable_and_why` — all green in this run; 32 items identical before/during/after, drawer left edge **x = 1306** against a video column ending at **1297** |
| 6 | The input-ack gate's blindness to inference backlog is **demonstrated**, not asserted | **UNMET** | the tool, its tests and the clean-machine guard exist and are green, but **the four-condition matrix was never obtained**. One of the four conditions was measured for this review (below) and shows the disagreement; that is **not** the matrix and is not claimed as the demonstration |

### Criterion 6, in its own words

The tool was run for one condition because this review needed a live windowed frame age (§5.5). The
result, on a clean machine (`"clean": true, "competing_pids": []`), two windows of one process:

```
condition                      rep  streams  sat  load_s  tick p50  tick p95  tick max   age p50   age p95  backlog   gate    age
three-cpu-no-saturators          0        3    0     164       0.0      10.2      25.2   40111.1   43216.9      709   pass  stale
three-cpu-no-saturators          1        3    0     194       0.0      11.3      26.6   48572.3   51681.5      836   pass  stale

gate says pass                : 2
frame age says stale          : 2
gate passes while age is stale: 2  ['three-cpu-no-saturators']
```

**The gate passes while the picture is 40–49 seconds old, twice.** That is the shape of the problem
criterion 6 exists to demonstrate. It is **not** the demonstration: the criterion asks for the
four-condition matrix (app load, the existing gate's load, one camera per device, and the saturated
pair), and three of those conditions were not run. **The criterion stays UNMET until the matrix
runs**, and the reason it has not is recorded rather than glossed: two measurements that each need
an idle machine were run concurrently by the previous task, so every window was refused by this
guard. This session ran its two timing measurements **sequentially, on an asserted-clean machine**,
which is the discipline that had been missing.

---

## 10. Carried forward — not M3's to close, and not closed here

| item | status | owner |
|---|---|---|
| **M0 gate 9's SILENT half** (a fallback that happens without the UI saying so) | **open and cannot close on this machine** — all three devices execute where asked, and the compile-failure seam kills the worker before it runs anything | M6 |
| **The 2 s warm-transition target** | **MISSED** — p50 2.238–2.261 s, 37/50 over 2 s at the real 2.0 s default; the distribution is documented | M4/M6 |
| **The leak at the real 2 s timeout: 182.4 MiB, 175.3 MiB unexplained** | **open, not characterised** | M4 |
| **GPU/NPU utilisation attribution** | **open** — the inference PID never appears as a counter instance; the cards read "Utilization unavailable", never 0 % | M4 |
| **Code signing** | **unmet** | M6 |
| **Shelf-to-basket** | **unmet by design** — the delivered event is people-in-zone presence, not a count, not a dwell | design |
| **The readability failures in §4** | **new, documented, not fixed** — type sized for arm's length at 2–3 m; the fix is a booth panel and a booth ramp | **M4** |
| **The seeded staleness understatement (§5.5)** | **new, documented, not re-seeded** — a live capture under the booth's real load is needed | **M4** |
| **The scope note is elided on screen** and no harness reports it (§3) | **new finding** — the full wording is in the `zone` explanation and the drawer; the header cuts the sentence mid-word | design/M4 |
| **The census covers named items only** (§3) | **new finding, corrected in the capture README and here** — 40 px text is on the screen and outside the census | M3 (this commit) |
| **No committed capture contains a frame** (§5.4) | **new finding** — the screenshots prove layout and strings, not the video surface | M4 |
| **`--fail-stream`** | **a qualification aid, not a booth feature** — see the presenter-guide note below | M6 |
| **A 60 s+ unattended soak** | **not run** | M6 |

### Presenter-guide note: `--fail-stream`

`--fail-stream A|B|C` is a **fault-injection flag for qualification**, not a booth feature. It
exists so the visible-failure path (a worker that really dies, `device: failed - no device is
running`, the worker's own stated reason) can be exercised without breaking a real device. **Do not
use it while a customer is watching, and do not present it as a demo feature.** The same applies to
`--fault-stream` in `tools/route_stress.py`. The presenter guide must say this in those words; this
document is where that wording is fixed until the guide exists.

---

## 11. Method appendix — the measurement, reproducible

The millimetre and arcminute table is generated by this script, run from the repo root with the
provisioned venv. It reads the **committed** capture, so a reader can re-run it and get §4's table.

```python
"""Compute the M3 readability table: px -> mm -> arcmin at 2 m and 3 m, with verdicts."""
import math

MM_PER_PX = 0.310          # 1920 px over a 595 mm-wide canvas - the assumption in section 1


def arcmin(height_mm: float, distance_m: float) -> float:
    return math.degrees(2 * math.atan(height_mm / (2 * distance_m * 1000))) * 60


def verdict(angle: float) -> str:
    if angle >= 16:
        return "PASS"
    if angle >= 10:
        return "marginal"
    if angle >= 5:
        return "FAIL"
    return "FAIL (below acuity)"


ELEMENTS = [
    ("Title 'Intel Edge AI Roadshow'", 40, 29),
    ("'Recorded footage . live inference'", 26, 19),
    ("'device: CPU' badge", 26, 19),
    ("'Camera A' heading + state", 26, 19),
    ("'events:' prefix", 26, 19),
    ("Frame-age headline '11.7 s' (x3)", 34, 24),
    ("'STALE - this picture is 11.7 s old; the limit is 2 s' (x3)", 18, 13),
    ("Per-camera badge 'CPU' / 'no device is running' (x3)", 18, 13),
    ("'backlog 187 frames' (x3)", 18, 13),
    ("'analysis 13.44 fps' (x3)", 18, 13),
    ("'window: 256 analysed frames' (x3)", 18, 13),
    ("'analysis fps ... frame age p50 ... p95' (x3)", 18, 13),
    ("events row (zone_occupancy / candidate_transfer / candidate_return)", 18, 13),
    ("pipeline 'decode'/'CPU' strip (12 labels)", 18, 13),
    ("'status: running' / 'telemetry unavailable'", 18, 13),
    ("scope note 'REDUCED SCOPE ...' (elided)", 18, 13),
    ("drawer explanation buttons", 18, 13),
]

print("| element | font px | cap px | mm | 2 m | 3 m | verdict at 2 m | verdict at 3 m |")
print("|---|---:|---:|---:|---:|---:|---|---|")
for name, font, cap in ELEMENTS:
    mm = cap * MM_PER_PX
    a2, a3 = arcmin(mm, 2.0), arcmin(mm, 3.0)
    print(f"| {name} | {font} | {cap} | {mm:.2f} | {a2:.1f}' | {a3:.1f}' | "
          f"{verdict(a2)} | {verdict(a3)} |")
```

**How the cap heights were measured, and where the corroboration is.** The PNG ink measurement is a
twelve-line script: crop each item's reported `extent`, take pixels whose brightest channel is above
70, group the rows into contiguous bands, and read the band height. Against the committed
`100pct.png` it gives, per item: `11.7 s` → 27 px (the antialiasing halo adds ~3 rows to the 24 px
cap height); `CPU` → 13 px; `Camera A` → 20 px; `backlog 187 frames` → 16 px
(ascender-to-descender); `device: CPU` → 20 px, measured as **dark pixels inside the green badge**
because light-on-green inverts the ink test; `decode` → 13 px. Those measurements came out of the
same script as the census verification in §3 and are quoted here rather than left implied. The
cap-height column in §4 uses the nominal `0.716 em` (the PNG corroborates it and shows the halo),
which is the **smaller** of the two — the table does not flatter the screen.

**What was not measured, and is not claimed:** the viewing distance a real passer-by stands at; the
booth's panel; the room's contrast and ambient light (the palette is high-contrast by design —
`#e8eef4` on `#141b23` — but no contrast ratio was computed or measured here); the time a passer-by
actually looks. The review answers exactly one question — **how big is the text on the screen, in
millimetres and in angle, at 2 m and at 3 m** — and answers it with numbers a reader can check.

---

## 12. Bottom line

- **The 1920×1080 DPI gate: `passed with documented failures`.** Four scales really rendered, pixel
  sizes measured from the images; the readability review found failures and they are in §4.
- **The screen is not readable at 2–3 m.** On a 27-inch 1080p panel, nothing clears 16 arcmin at
  3 m; the three frame-age headlines — the numbers the spec says must be legible at a glance — are
  12.8′ at 2 m and 8.5′ at 3 m; every 18 px line is below the acuity floor at 3 m. On a 55-inch
  panel the headlines pass and the 18 px detail still fails at 3 m.
- **The seed understates the live screen, measured:** 10.7–11.7 s seeded against **40.1 s and
  48.6 s** live, same condition, same day, 130 s warm-up; and today's bare `Supervisor` (4.5 s) is
  faster than the seed, so the presentation path — not this machine — is where the difference comes
  from. The captures were **not** re-seeded.
- **M3 exit criteria 1, 3, 4, 5 pass; criterion 2 passes only in its "documented" form; criterion 6
  — the input-ack blindness demonstration — is UNMET**, because the four-condition matrix still does
  not exist. One condition was measured for this review and the gate passed while the picture was
  40–49 s old; that is a window, not the matrix, and it is not claimed as the demonstration.
