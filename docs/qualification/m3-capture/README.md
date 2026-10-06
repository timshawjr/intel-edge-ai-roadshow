# M3 capture records — the booth screen at four DPI scales, offscreen

Produced by `python -m tools.capture_ui --out docs/qualification/m3-capture`, on 2026-10-05 on the
demo machine. Each pair is a PNG and the JSON of **every string read out of the running scene**.

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

**The strings are identical across all four scales** — verified. The scale is a rendering scale, not
a layout change: Qt renders the same logical layout at more pixels. What *does* change with scale is
pixel density, and Task 2's gate is what measures whether that density is legible.

## What each JSON carries

`strings` — every named item's `text`, read from the running scene graph. `truncated`, `overflow`,
`absent`, `unreachable`, `geometry_unavailable`, `extents`, `clip_sources`, `visible`,
`opened_during_read`, `read_window_closed`, `qobject_hits` / `qobject_missed`,
`qobject_child_count`, `glyph_pixels`, `seed_provenance`.

**`strings` came out of the scene, not out of the source.** The reader runs inside an audit window:
`opened_during_read` is the list of files opened while the strings were produced, and it is empty at
every scale. `read_window_closed` says the window was closed, because an audit that is never closed
proves nothing.

## What these four captures already say

- **`overflow` is empty at all four scales**, and `clip_sources` is `window` for every item —
  **no QML file in `app/ui/` sets `clip: true`** (verified: `grep -c 'clip\s*:'` is 0), so the only
  clip available on this screen is the window's own rect. A card's text spilling past its own card
  is invisible to this measure, and the harness says so rather than pretending otherwise.
- **`truncated` names `streamReasonText_C` at all four scales.** The failed camera's stated reason is
  2 331 characters and does not fit the three lines `ComputeCard.qml` gives it
  (`maximumLineCount: 3`). The JSON reports **all 2 331 characters** *and* the fact that the visible
  rendering is the elided prefix — never the prefix alone.
- **`glyph_pixels` vs `qobject_missed` is the delegate-read proof, in numbers.** `QObject.findChild`
  reaches **4** of the 19 expected names and misses **15** — every one of the misses a `Repeater`
  delegate — while the walk reaches all 19 and the PNG's own pixels show non-background pixels
  inside each one's reported extent.

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