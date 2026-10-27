# M3 capture records — the booth screen at four DPI scales, offscreen

Produced by `python -m tools.capture_ui --out docs/qualification/m3-capture`, first on 2026-10-05
and **re-rendered on 2026-10-06** on the demo machine after the M3 Tasks 1-2 review, then
**re-rendered again on 2026-10-27 by M4 Task 1** after the legibility rebuild. Each pair is a
PNG and the JSON of **every string read out of the running scene**. The 2026-10-06 render
established `glyph_pixels` in **device** pixels, the named thumbnail captions and the
`text_property` / `walk_failures` / `settle_polls` / `glyph_scale` fields. **The 2026-10-27
render changed what is on the screen**: the always-visible surface was rebuilt at the
legibility ladder (44 / 48 / 56 px) and six per-camera figures, the event row, the pipeline
strip, the control bar and the footer `?` row moved into the presenter drawer. See
[M4-legibility-review.md](../M4-legibility-review.md) for why, and the section "What M4
changed in these captures" below for exactly what moved.

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

Re-rendered by **M3 Task 2**, after the 1920x1080 layout work, and **again by M4 Task 1** after
the legibility rebuild. Every statement below was read out of the committed JSONs, not asserted.

- **`overflow`, `truncated`, `absent`, `unreachable` and `geometry_unavailable` are all empty at all
  four scales.** `overflow` is not a statement about the window alone: **`app/ui/` sets
  `clip: true`** on `ComputeCard.qml`'s root and its column and on `VideoTile.qml`'s root, so
  `clip_sources` names the actual box each item was measured against —
  `ancestor:card_A`/`card_B`/`card_C`, `ancestor:cardColumn_*`, `ancestor:heroTile`,
  `ancestor:thumb_B`/`thumb_C`, and `window` for the header and footer items. The column is
  named for exactly this reason: an unnamed clipping ancestor reports as the bare string
  `ancestor:` and a reader cannot tell which box overflowed.
- **The drawer's own overflow is measured too, and it is empty.** `tools/capture_ui.py` renders
  the live screen with the drawer **closed**, so the drawer's fit is asserted by
  `tests/test_ui_contract.py::PresenterDrawerOnTheScreen`, which opens it, reads every item and
  asserts `overflow == []` for all seven explanation selections. Measured with the longest
  explanation selected (`gpu`) and three cameras: the last line ends at **y = 972** in a drawer
  whose inner bottom is **1056**.
- **`truncated` is empty, which is a change and not an absence.** At M3 Task 1 this list named
  `streamReasonText_C` at all four scales: the failed camera's stated reason is **2 331 characters**
  and did not fit the three lines Qt was eliding it into, with **nothing on screen saying so**. The
  cut is made **in Python**, at `Theme.CARD_REASON_CHARS`, and the screen reads
  `Camera C: route transition to generation 1 was not... +2291 more` — 54 characters of reason
  plus the camera's name, naming **2 291 of the 2 331 that are not shown**. `record.reason` still
  publishes the complete message; the JSON's `strings` carries what the screen showed, and the
  empty `truncated` list is the proof that **no second, silent cut was left behind**.
- **`font_pixels`** is what makes "no font size changed" checkable. **M4 changed every size
  on the live screen on purpose**, and the census is now
  **`{44: 18, 48: 3, 56: 3}`**: 44 px on eighteen items (the mandatory label, the title, the
  header badge, the explanation button, the three camera states, the three device badges, the
  two staleness sentences, the failed camera's reason, the two thumbnail captions, the three
  "no frame" sentences and the three age headlines' supporting text), 48 px on the three
  camera names, 56 px on the three frame ages. **The census now counts visible items only**:
  an item with text that is not drawn is not type on the screen, and M3's non-empty-only
  definition would have counted a hidden 18 px hover hint.
- **`glyph_pixels` vs `qobject_missed` is the delegate-read proof, in numbers.** `QObject.findChild`
  reaches **5** of the **30** expected names and misses **25** — every one of the misses a
  `Repeater` delegate — while the walk reaches all 30 and the PNG's own pixels show
  non-background pixels inside each one's reported extent **at every one of the four scales**.
  The 30 expected names are the live screen's own: `recordedFootageText`, `appTitle`,
  `deviceBadgeText`, the three cards, the three camera names and states, the three ages, the
  three device badges and their boxes, the two staleness sentences, the failed camera's
  reason, the three verdict rows, the hero, the two thumbnails, their captions and Camera A's
  "no frame" sentence. (`streamStaleText_C` exists and is correctly blank: camera C has a
  stated failure reason, and one line per camera is the budget — see the M4 review.)

## What M4 changed in these captures, and what it did to the element set

| M3 item (18 px) | M4 |
|---|---|
| title `Intel Edge AI Roadshow` | still on screen, **named** (`appTitle`) and at 44 px |
| mandatory `Recorded footage · live inference` | still on screen, **named** (`recordedFootageText`) and at 44 px |
| device badge, camera headings, `events:` prefix, status line | badge and camera names on screen at 44/48; `events:` and `status:` in the drawer |
| the three frame ages | still on screen at **56 px** — the largest type on the screen |
| the three staleness sentences | on screen at **44 px**, full width, one line per camera, prefixed with the camera's name |
| the failed camera's reason | on screen at 44 px on the same line as its staleness sentence would be, cut at `CARD_REASON_CHARS` with `... +N more` |
| per-camera device badge | on screen at 44 px (each camera's own confirmed device) |
| per-camera backlog, analysis rate, frame-age window, activity line, utilisation figure, telemetry window note | **moved to the presenter drawer** |
| the event row | **moved to the presenter drawer** |
| the pipeline strip (12 labels) | **removed**; the drawer states the topology in one sentence |
| the seven footer `?` keys, the pinned panel, `status:`/`telemetry` | **removed**; the drawer holds the seven explanations and is opened by a 44 px header button, by clicking a card, or by F1 |

The four PNGs therefore show a different screen from the M3 commit. **The pixel dimensions,
the scale factors and the empty `overflow` / `truncated` / `absent` / `unreachable` lists are
unchanged** — what changed is the type and the element set, which is the point of M4.

- **The device-telemetry rows are in a placeholder state, and the layout was not tuned for
  it.** These captures are seeded from a committed measurement, not from a live run, so the
  presenter drawer's `cardText_system_cpu_percent` reads `no CPU sample yet` and its
  `cardWindowNote_*` reads `measured unavailable`. That is an honest fourth screen state —
  neither the "telemetry present" state nor the "failed camera" state the layout was sized
  against. It is **not a rendering fault**, and it is **not evidence the live drawer will look
  like this**: a real run fills those rows with a percentage and a window note. It is flagged
  here so nobody reads a seeded placeholder as a measured figure. **These rows are in the
  drawer now**, so they are not in the live capture JSON at all; the drawer's render is
  measured by `tests/test_ui_contract.py::PresenterDrawerOnTheScreen`.

## The busiest state, and what it cost

**These four captures are the three-streams-stale state plus one failed camera.** That is the state
the brief rules is the real one, and it is what `allcpu-steady-120s.json` plus
`route-stress-fault-visible-fallback.json` describe together.

**It is not the hardest case, and the harness is told so rather than left to imply it.** The failed
camera claims no device, so its card shows **three** rows (name and state, age, device) and its
verdict line carries its stated reason. The case that actually stresses this layout is a camera
that **fell back successfully**: its device badge is the longest badge string there is
(`GPU.0 (fell back from CPU)`, **538 px** at 44 px in a 551 px box) *and* it carries a reason from
the transition, so it is both stale and explained and its one verdict line must choose the reason.
`tests/test_capture_ui_canvas.py::test_a_camera_with_a_device_and_a_long_reason_still_fits_the_screen`
seeds that shape explicitly and asserts the longest badge and the reason-outranks-staleness rule
**before** asserting the geometry, because a test that passes over the ordinary state has not
reached the hard case.

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