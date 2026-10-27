# M3 capture records — the booth screen at four DPI scales, offscreen

Produced by `python -m tools.capture_ui --out docs/qualification/m3-capture`, first on 2026-10-05
and **re-rendered on 2026-10-06** on the demo machine after the M3 Tasks 1-2 review. Each pair is a
PNG and the JSON of **every string read out of the running scene**. The 2026-10-06 render is the one
committed: `glyph_pixels` is now counted in **device** pixels (see below), the thumbnail caption is
**named** (`tileCaptionText_B` / `tileCaptionText_C`) so the M2 `Camera B · -1` defect it was built
to catch is no longer invisible to it, and the record carries `text_property`, `walk_failures`,
`settle_polls` and `glyph_scale`.

**Why offscreen at all:** the demo machine's desktop is 2293x960, so **a 1920x1080 canvas cannot be
displayed on it at all** — M0 measured the usable height at 960 px. The M3 DPI gate is therefore
untestable by looking at it, and this harness is what makes the gate existable.

## What is here

| file | pixels | logical | dpr |
|---|---|---|---|
| `100pct.png` / `100pct.json` | 1920x1080 | 1920x1080 | 1.00 |
| `125pct.png` / `125pct.json` | 2400x1350 | 1920x1080 | 1.25 |
| `150pct.png` / `150pct.json` | 2880x1620 | 1920x1080 | 1.50 |
| `200pct.png` / `200pct.json` | **3840x2160** | 1920x1080 | 2.00 |

**The pixel size is measured from the saved image, never computed from the scale.** At 200% the
logical canvas is still 1920x1080 and the file is 3840x2160; asserting the logical size would pass
at every scale and prove nothing. **`QImage.devicePixelRatio()` is `1.00` at every scale**, including
the 3840x2160 grab, so it is reported in each JSON and is *not* the size signal.

**The strings and the logical extents are identical across all four scales** — verified by
comparing the four maps. The scale is a rendering scale, not a layout change: Qt renders the same
logical layout at more pixels. So the honest reading of these four files is **narrow**: the logical
layout is unchanged, the pixel count grew 1x / 1.25x / 1.5x / 2x, and nothing overflows at any of
them. **It is not evidence that the layout is legible at 200%** — pixel density is the only thing
these files vary, and legibility at 2-3 m is measured by **M3 Task 6's readability review**, not
inferred from four PNGs.

## What each JSON carries

`strings` — every named item's `text`, read from the running scene graph. `truncated`, `overflow`,
`absent`, `unreachable`, `geometry_unavailable`, `extents`, `clip_sources`, `visible`,
**`font_pixels`**, **`text_property`**, **`walk_failures`**, **`settle_polls`**, **`glyph_scale`**,
`empty_nonempty`, `opened_during_read`, `read_window_closed`, `qobject_hits` / `qobject_missed`,
`qobject_child_count`, `glyph_pixels`, `seed_provenance`.

- **`text_property`** is the fourth per-field state: an item that is *reached* but has **no `text`
  property** (a `Rectangle`, an `Item`) is `false` here and is deliberately **absent** from `strings`,
  so "this label is blank" is never true about a box. `card_A` is the example.
- **`walk_failures`** is *why* `walk_complete` might be `false`; it is `[]` at every scale.
- **`settle_polls`** is how many readiness polls ran before the scene qualified. At every scale it is
  **1** — in seed mode the bindings are resolved before the first 25 ms poll. **1 is a construction
  check, not a rendering check**: a capture in which every binding yielded `""` would also settle on
  poll 1, which is why `expect_nonempty`/`empty_nonempty` exist and why the live capture asserts
  `settle_polls > 1`.
- **`empty_nonempty`** names required-nonempty items that were present but blank; `[]` at every scale.
- **`glyph_scale`** is the factor the pixel count applied to the logical extents before indexing the
  **device-pixel** image: `[1.0, 1.0]`, `[1.25, 1.25]`, `[1.5, 1.5]`, `[2.0, 2.0]`.

**`strings` came out of the scene, not out of the source.** The reader runs inside an audit window:
`opened_during_read` is the list of files opened while the strings were produced, and it is empty at
every scale. `read_window_closed` says the window was closed, because an audit that is never closed
proves nothing. Both fields are read from the child's report by `capture()` — the review found the
audit list had been read from `scene`, where it never existed, so the wiring terminated in a dead
lookup.

**`glyph_pixels` is counted in device pixels and is not a legibility number.** For a named item it is
the number of pixels **inside its reported extent** that differ from the image's modal colour. The
image is a **device-pixel** image (at 125 % a 1920x1080 logical canvas is a **2400x1350** file), so
the logical box is scaled by `image.width() / window.width()` before it indexes the buffer. The first
version did not, and reported `0` for plainly-drawn items at three of the four scales — see the M3
section of `bench_report.md`. What a positive count proves is that **something is drawn inside the
box**; it is not a glyph count, a contrast measure, or evidence of legibility at 2-3 m (Task 6).

## What these four captures say

Re-rendered by **M3 Task 2**, after the 1920x1080 layout work. Every statement below was read out
of the committed JSONs, not asserted.

- **`overflow`, `truncated`, `absent`, `unreachable` and `geometry_unavailable` are all empty at all
  four scales.** `overflow` is no longer a statement about the window alone: **`app/ui/` now sets
  `clip: true`** on `ComputeCard.qml`'s root and its two `Column`s and on `VideoTile.qml`'s root, so
  `clip_sources` names the actual box each item was measured against —
  `ancestor:card_A`/`card_B`/`card_C`, `ancestor:cardLeftColumn_*`, `ancestor:cardRightColumn_*`,
  `ancestor:heroTile`, `ancestor:thumb_B`/`thumb_C`, and `window` for the nine header and footer
  items. The columns are named for exactly this reason: an unnamed clipping ancestor reports as the
  bare string `ancestor:` and a reader cannot tell which box overflowed.
- **`truncated` is empty, which is a change and not an absence.** At M3 Task 1 this list named
  `streamReasonText_C` at all four scales: the failed camera's stated reason is **2 331 characters**
  and did not fit the three lines Qt was eliding it into, with **nothing on screen saying so**. The
  cut is now made **in Python**, at `Theme.CARD_REASON_CHARS`, and the screen reads
  `route transition to generation 1 was not... +2291 more` — 54 characters naming **2 291 of the
  2 331 that are not shown**. `record.reason` still publishes the complete message; the JSON's
  `strings` carries what the card showed, and the empty `truncated` list is the proof that **no
  second, silent cut was left behind**.
- **`font_pixels` was added by Task 2**, and it is what makes "no font size changed" checkable. At
  100%: **18 px on 29 items**, **26 px on 1** (`deviceBadgeText`, `TYPE_LABEL`), **34 px on 3** (the
  three `streamAgeText_*` headlines, `TYPE_NUMERAL_SMALL`). No other size appears among the
  **named** items - and that qualifier is a correction made by M3 Task 6: the screen also
  carries a **40 px** title, 26 px card headings and 18 px footer and scope-note text that
  have no `objectName`, and this census cannot see any of them. See
  [M3-readability-review.md](../M3-readability-review.md), section 3.
- **`glyph_pixels` vs `qobject_missed` is the delegate-read proof, in numbers.** `QObject.findChild`
  reaches **4** of the **21** expected names and misses **17** — every one of the misses a `Repeater`
  delegate — while the walk reaches all 21 and the PNG's own pixels show non-background pixels
  inside each one's reported extent **at every one of the four scales**. The 21st and 22nd names are
  the thumbnail captions (`tileCaptionText_B` / `tileCaptionText_C`), named by this review round
  because the `Text` that renders `Camera B · -1` had **no `objectName`** and was therefore outside
  the instrument built to catch exactly that defect.
- **The three healthy-camera metric rows are in a placeholder state, and the layout was not tuned for
  it.** These captures are seeded from a committed measurement, not from a live run, so
  `cardText_system_cpu_percent` reads `no CPU sample yet` and both `cardWindowNote_*` read
  `measured unavailable`. That is an honest fourth screen state — neither the "telemetry present"
  state nor the "failed camera" state the layout was sized against. It is **not a rendering fault**
  and it is **not evidence the live cards will look like this**; a real run fills those rows with a
  percentage and a window note. It is flagged here so nobody reads a seeded placeholder as a
  measured figure.

## The busiest state, and what it cost

**These four captures are the three-streams-stale state plus one failed camera.** That is the state
the brief rules is the real one, and it is what `allcpu-steady-120s.json` plus
`route-stress-fault-visible-fallback.json` describe together.

**It is not the hardest case, and the harness is told so rather than left to imply it.** The failed
camera claims no device, so its right column shows **four** rows with **67.3 px** of slack
(measured: the reason ends at 744.67, the column's inner bottom is 812.0). A camera that **fell
back successfully** has a confirmed device — so `cardText_` and `cardWindowNote_` appear — *and* a
reason from the transition: **six** rows in that column, the reason last, **43.3 px** of slack
(measured). `tests/test_capture_ui_canvas.py::test_a_camera_with_a_device_and_a_long_reason_still_fits_its_card`
seeds that six-row case explicitly and asserts the six rows are present **before** asserting the
layout fits, because a test that passes over four rows has not reached the hard case.

## Where the seeded numbers come from

Not typed in. `seed_provenance` in each JSON names the file per field:

- `age_p50_ms`, `age_p95_ms`, `backlog`, `analysis_fps`, `window_samples` from
  `docs/qualification/m2-headless/allcpu-steady-120s.json` — one 120 s hold of spec section 5 row 4,
  three streams on the CPU.
- `failed_reason` from `docs/qualification/m2-headless/route-stress-fault-visible-fallback.json` —
  the committed supervisor message, which embeds the worker's own stated compile-failure reason.
- `failed_state` is **seeded, not measured**, and says so in the JSON: it is a fault injected so the
  failed card's string is on screen to be measured.

If either measurement file is missing, the harness **raises** with the path. It does not fall back to
a plausible number.

## Reproducing

```
.venv\Scripts\python.exe -m tools.capture_ui --out docs\qualification\m3-capture
```

Needs no inference worker and no provisioned assets: the bridge is seeded from the committed
measurements and the **real** `Main.qml`, `UiBridge`, `FrameImageProvider` and `ThemeTokens` are used.
For a capture of the **real application** on real hardware:

```
.venv\Scripts\python.exe -m tools.capture_ui --device CPU --scale 100 --out <dir>
```

which drives `app.main.main(["--windowed", ...])` and needs the provisioned clip and IR. Without
them it raises `NotProvisioned` rather than passing quietly.