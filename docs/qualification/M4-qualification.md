# M4 — sustained trials and legibility: qualification report

**What this document is.** The milestone record for M4. It reports the five obligations the plan sets
(SD §M4 exit criteria) by **outcome**, not by intention, and every figure in it traces to a committed
record or to a command run in this session. It is **documentation only**: this task changed no code, no
`config/*.json` and no tool, and it ran nothing that writes a config file.

It **extends** `M0-qualification.md`, `M0-content-gate.md`, `M1-qualification.md`, `M2-qualification.md`,
`M3-readability-review.md`, and the two standalone M4 accounts `M4-legibility-review.md` and
`M4-utilisation-attribution.md`. Where a number here differs from an earlier document, this one says which
measurement supersedes which, and why. Where an earlier document was right, this one says so and does not
re-litigate it.

Machine: **NUC14PROAI**, Windows 11 Pro 10.0.26200, measured 2026-10-27. Branch `main`.

---

## The five obligations, and what each one returned

| # | obligation (plan §M4) | outcome |
|---|---|---|
| 1 | Every element a passer-by must read is measured on **the monitor actually attached**, clears 16′ of cap height at 2 m, and its failure at 3 m is documented | **PASSED AT 2 M, FAILED AT 3 M — the failure is documented.** The screen is readable at 2 m on cap height only; x-height fails at **both** distances. The 55-inch case is a **procurement requirement, never an achieved measurement.** §3 |
| 2 | The four-condition input-ack matrix exists, with tick lateness and frame age in the same rows | **MET — and the claim is confirmed in its worst form.** The gate passed **8 of 8** windows while **6 of 8** showed a picture up to 55 s old with up to 1 143 frames queued. §4 |
| 3 | The 182 MiB leak at the real 2 s timeout is bounded and explained, or its cause is stated | **MET — bounded and explained.** +153.7 MiB before, −10.4 MiB after, same 50 transitions and the same 25 timeouts; the cause is a retired queue's blocked feeder thread, measured, and pruning alone released nothing. §5 |
| 4 | GPU/NPU utilisation is attributed per process, or refused with evidence | **MET, BY ATTRIBUTION.** M0's unmet gate 2 closes: idle 0.0% → busy 97.19% GPU and 64.86% NPU on this application's own pid, with a distractor process excluded. §6 |
| 5 | Sustained paired trials exist per device, with conditions stated and no unsupported generalisation | **MET — and they rank less than expected.** 27 trials, 3 conditions, zero fallbacks, placement confirmed on all 27. **At the booth's 15 fps the throughput ranks nothing:** all three devices deliver 300/300. §7 |

**The one sentence a reader can carry away.** On this machine the booth's own 15 fps screen is not
device-limited and the screen still fails the 3 m legibility target on the monitor it was measured on;
the gate that is supposed to notice a failing screen is a **green light on a failing system**; the leak
that made the real 2 s deadline unusable is **gone and explained**; the accelerator figures are now
**this application's own**, at the stated cost that they are the busiest engine and not a device total;
and the only measurement that ranks the GPU against the NPU is **isolated sustained inference**, where
the GPU leads by about a quarter — not the delivered stream, where neither leads because the source
sets the rate.

---

## 1. The machine, the method and the cleanliness rule

### 1.1 The machine, as read by the runs

| | |
|---|---|
| host | `NUC14PROAI` (a hostname, not a SKU — M0's three identity records still disagree) |
| OS | Windows 11 Pro, `10.0.26200` (`Windows-11-10.0.26200-SP0`) |
| CPU | Intel(R) Core(TM) Ultra 9 288V, 8 logical / 8 physical cores |
| GPU | Intel(R) Arc(TM) 140V GPU (16GB), iGPU, driver **32.0.101.6737** |
| NPU | Intel(R) AI Boost, driver **32.0.100.5540** |
| memory | 31.5 GiB |
| runtime | OpenVINO `2026.4.0-22959-99c81491cc3-releases/2026/4`, Python 3.12.10 |
| model | `yolo11n-fp16`, input `[1, 3, 640, 640]`, **batch 1**, static |
| clip | `assets/media/retail-checkout.mp4`, 1920x1080, 29.97002997 fps, 3 387 frames, 113.01 s |
| display copy | reduced to 960 px wide **after** inference; the detector still sees full resolution |

These are the values the records carry (`machine` blocks of `m4-sustained-trials.json`,
`m4-utilisation-control.json`, and the preflight rows in §2). This is **one machine, one model, one
shape, one OpenVINO build and one pair of driver versions**; AGENTS.md rule 13 applies — a driver update
invalidates the NPU blob cache and every figure in §6 and §7 must be re-run before it is quoted again.

### 1.2 The rule this milestone actually followed

M3's handoff carries a standing failure class — **the measurement process polluting the measurement**,
where 66 orphaned workers poisoned two milestones of timings — and M4 did not repeat it:

- **Every timing, leak and latency measurement asserted the machine clean first**, and the census is in
  the record, not only in a report: `"clean": true, "competing_pids": []` in
  `m4-input-ack.json`, `m4-sustained-trials.json` and all four `m4-leak-*.json`.
- **No two measurements ran at once.** Task 2 records killing its own accidental second census because
  that census *was* a competing `python.exe` to the running poller; the delivered matrix ran alone.
- **The suite was run after the measurements**, not beside them (§2 is this session's run).
- **A guard that refuses is a result.** `tools/input_ack_under_load.py` refuses a dirty window rather
  than recording a figure; `ACKLD_ALLOW_DIRTY` was never set in any row quoted here.

### 1.3 Where the records are

Every record named below is **committed**; nothing in §3–§7 depends on `logs/` or on `.superpowers/sdd/`,
both of which are gitignored. Two exceptions are named rather than glossed: **one preflight row** reads
`logs/qualification/m0.json` (M0's recorded run — §2.2 says so, and it is a *recorded* value, not a
re-measurement), and the **mutation tallies** live in the task reports under `.superpowers/sdd/`. No gate
row and no number in §3–§7 depends on a mutation tally.

| record | what it is | timestamp | task |
|---|---|---|---|
| `docs/qualification/m4-input-ack.json` | the four-condition matrix, 8 windows | 2026-10-27 (file's own timestamp; the record carries no `measured_at` field) | 2 |
| `docs/qualification/m4-leak-before-2s.json` | 50 transitions at the product's 2.0 s deadline, unfixed code | 2026-10-27 (file's own timestamp) | 3 |
| `docs/qualification/m4-leak-before-30s.json` | the same at a deliberately widened 30 s deadline | 2026-10-27 (file's own timestamp) | 3 |
| `docs/qualification/m4-leak-after-2s.json` | the same 2.0 s run after the fix | 2026-10-27 (file's own timestamp) | 3 |
| `docs/qualification/m4-leak-after-30s.json` | the same 30 s run after the fix | 2026-10-27 (file's own timestamp) | 3 |
| `docs/qualification/m4-utilisation-control.json` | the idle→busy control for the per-process utilisation figure | `measured_at` 2026-10-27T15:29:24 | 5 |
| `docs/qualification/m4-sustained-trials.json` | 27 trials, 3 conditions, 3 devices, 3 repeats | `measured_at` 2026-10-27T16:42:12 | 4 |
| `docs/qualification/m3-capture/{100,125,150,200}pct.{png,json}` | the rendered booth screen the legibility numbers are measured from | M3, re-rendered in M4 fit | 1 |

Tools added in M4 and committed: `tools/input_ack_under_load.py` (extended), `tools/leak_probe.py`,
`tools/sustained_trials.py`, `tools/probe_utilisation.py`.

### 1.4 What this report will not do

- **It does not restate a number it did not see.** Where a figure is quoted from an earlier milestone it
  says so and names the document; where something was not measured it says **not run** or **not
  established**.
- **It does not round a two-part result into one value.** §3 is the case that matters.
- **It does not add measurements together.** Frame age, inference cost, CPU utilisation and throughput
  are four different quantities and are never averaged, summed, or substituted for one another.

---

## 2. The gate set, run in this session

Machine asserted clean immediately before the run (`Get-Process python` → **0**), on branch `main`, tree
committed. Timings are measured around each process with `System.Diagnostics.Stopwatch`.

### 2.1 `tools\verify_sources.py`

```
$ .venv\Scripts\python.exe tools\verify_sources.py
STATUS  ID                    KIND      DETAIL
----------------------------------------------
PASS    yolo11n-fp16          xml       xml+OpenVINO-IR, 412503 bytes, sha256 4845b46cc9b6ac379a44484258b7ed98f836ed117d91f127c7728ea068250a95
PASS    yolo11n-fp16:weights  binary    binary, 5283218 bytes, sha256 f14632518ebb4e312a89c26faa4cd2f95d4264b6f5f22aa6e0d5707b4667863e
PASS    yolo11n-fp16:labels   text      text+utf-8, 80 lines, 701 bytes, sha256 82bf6a434f132c4d78534c0f2e0f991fc22c73cd911f6d5fc74a418c39c339ec
PASS    retail_checkout       mp4       mp4+ftyp, 64996307 bytes, sha256 e107db69155af4774f907abe8ddbf24493dfcf9d81d51f3e71ff25300e4f5906
PASS    store_aisle_alt       mp4       mp4+ftyp, 9214573 bytes, sha256 3526fee39cac70d6366471e4324a6d63b337c8f8ab99c015521bee0d60ed6e04
PASS    queue_alt             mp4       mp4+ftyp, 3291918 bytes, sha256 a5964aa259099a482a8b360ffc2c57b5a30f84d5919236a4dad01f8e929ac07c

Source verification: 6 PASS, 0 FAIL
EXIT=0 WALL_S=0.14
```

Verified **by content** — the tool reads each artefact and checks its signature and sha256 against
`config/models.json`; it does not grade a status code.

### 2.2 `tools\preflight.py`

```
$ .venv\Scripts\python.exe tools\preflight.py
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
PASS  Telemetry: per-process GPU/NPU utilisation
PASS  Telemetry: recorded LUID-to-device map
WARN  Telemetry: LUID-to-device attribution and hardware identity
PASS  Startup: this gate's import to the first inference
Summary: 17 PASS, 1 WARN, 0 FAIL - 18 total rows
Status: Limited
Exit code: 0. Ready and Limited both exit 0, because in both the demo runs; only Needs attention exits 1.
EXIT=0 WALL_S=4.43
```

The row names above are verbatim; each row's long detail text is omitted here rather than paraphrased.
Two details from the one WARN, quoted from its own output, because they are what the row is about:

```
WARN  Telemetry: LUID-to-device attribution and hardware identity
      - config/telemetry_map.json records direct_luid_link.available=false, 'no CIM or Win32
        source joins a LUID to a PCI function'. That is superseded by measurement: DXGI's
        IDXGIAdapter::GetDesc returns Description, VendorId, DeviceId and AdapterLuid together.
        Measured this run: 'Intel(R) Arc(TM) 140V GPU (16GB)' PCI 8086:64A0 LUID 0x000113B4,
        'Microsoft Basic Render Driver' PCI 1414:008C LUID 0x0001174C. ...
      - the recorded LUIDs ['0x00010E9E', '0x000112A0'] do not include this boot's GPU LUID(s)
        ['0x000113B4']: Windows assigns adapter LUIDs per boot, so the file's mapping is stale.
        config/telemetry_map.json is verified data and is NOT edited here; the runtime no longer
        depends on it for attribution, which is why a stale file does not make the cards wrong.
```

The gate reads **`Limited`, exit 0**, and the single WARN is a **recorded-config row, not a device
row**: the runtime resolves the device split from DXGI and from the one adapter exposing
`engtype_Neural`, so a stale map makes the *file* wrong, not the cards. §6.7 keeps this in the limits
list rather than making it disappear by editing a config file this task may not touch.

Two rows are worth naming. **`Telemetry: per-process GPU/NPU utilisation`** is now a `PASS` row
(`349 live GPU Engine instances; this process's own GPU reading is a number — and absence is rendered as
unavailable, never 0`), where it was a `WARN` from M0 to M3; that is §6's change. And
**`Recorded per-device compile, latency and placement`** still quotes `logs/qualification/m0.json` and
says so — it is a **recorded** value, not a re-measurement, and this report does not treat it as one.

### 2.3 `python -m app.main --selftest`

```
$ .venv\Scripts\python.exe -m app.main --selftest
  ok   preprocess           letterbox 640x640 pad=114 scaled=640x360
  ok   infer                26.70 ms on confirmed ['CPU']
  ok   decode-track-event   1 zone_occupancy event(s) from a synthetic presence
  ok   render-metadata      unavailable renders as unavailable, and no device is claimed before an inference confirms one
PASS
EXIT=0 WALL_S=0.98
```

No network, exit 0. The `26.70 ms` is a **single warm CPU inference in this run**, not a latency claim;
§7's paired trials are the latency record.

### 2.4 `python -m unittest discover -s tests`

Machine asserted clean before the run (`Get-Process python` → **0**) and no `python.exe` remained after it.
Full suite, no test selection, no skips:

```
$ .venv\Scripts\python.exe -m unittest discover -s tests
.......................................................................................
[... the suite's own progress dots and the tests' own output, unedited ...]
----------------------------------------------------------------------
Ran 1459 tests in 599.133s

OK
EXIT=0 WALL_S=600.04
```

**1 459 tests, `OK`, exit 0, wall clock 600.04 s measured around the process** — the committed count at
`b8e314b`, which is the point: this task added no test, because it changed no code. No failure, no error
and **no skip**. The progress dots are truncated in the paste because they are 1 459 characters of the
same character; the summary, the verdict, the exit code and the wall time are verbatim.

### 2.5 What was deliberately not run in this session

- **`tools/probe_telemetry.py` — not run.** It rewrites `config/telemetry_map.json` on every run, and
  this task may not change a config file. The row it feeds is `PASS` in §2.2 on the committed map, and
  the staleness the WARN describes is recorded in §6.7 rather than papered over by regenerating the file.
- **No measurement was re-taken for this report.** Every figure in §3–§7 is from the committed record
  named in §1.3; re-running a 27-trial campaign or a four-condition matrix to decorate a document would
  be a new measurement with a new machine state, and it would not be the same record.

---

## 3. Legibility — PARTIAL, and the numbers say so

### 3.1 What the numbers are measured from

Not from a token multiplied by a constant. For every **visible, non-empty, named** `Text` on the live
booth screen (24 of them), the rendered font size is read from the scene graph, Arial's **measured**
ratios are applied (`QFontMetricsF` on this machine: cap `0.7164 em`, x-height `0.5184 em`, re-asserted
by the test on every run), and the **ink band is measured in the saved PNG** and required to span the cap
height. A `NaN` extent fails, a 1x1 window fails, and the exclusion list is asserted empty. The full
table, method and test-by-test map are in `M4-legibility-review.md` §2, §3 and §7.

### 3.2 The angular table, at both geometries

**On the monitor actually attached** — DELL S3422DWG, 80 x 33 cm, native 3440x1440, running a 2293x960
desktop at 150 %, **0.310 mm per logical pixel** (WMI; derivation in `M4-legibility-review.md` §1.1).
This is the **achieved** result:

| element | px | cap mm | 2 m cap | 3 m cap | 2 m x-height | 3 m x-height |
|---|---:|---:|---:|---:|---:|---:|
| body — mandatory label, title, device badge, verdict sentences, captions | 44 | 9.77 | **16.8′** pass | **11.2′ fail** | **12.2′ fail** | **8.1′ fail** |
| camera names | 48 | 10.66 | **18.3′** pass | **12.2′ fail** | **13.3′ fail** | **8.8′ fail** |
| **frame age** (the largest type on the screen) | **56** | 12.43 | **21.4′** pass | **14.2′ fail** | **15.5′ fail** | **10.3′ fail** |

**On the 55-inch 1080p panel the booth is required to procure** — 0.63416 mm/logical px, a **requirement
and not a measurement**:

| element | px | 2 m cap | 3 m cap | 2 m x-height | 3 m x-height |
|---|---:|---:|---:|---:|---:|
| body | 44 | 34.4′ | **22.9′** | 24.9′ | **16.6′** |
| camera names | 48 | 37.5′ | **25.0′** | 27.1′ | **18.1′** |
| frame age | 56 | 43.6′ | **29.1′** | 31.5′ | **21.1′** |

Both measures clear 16′ at 3 m **only** on the requirement panel. The sensitivity table — 24 in 10.0′ /
7.2′, 27 in 11.2′ / 8.1′, the attached monitor **11.2′ / 8.1′**, 32 in 13.3′ / 9.6′, 40 in 16.7′ / 12.1′,
55 in 22.9′ / 16.6′, 65 in 27.1′ / 19.6′ — is in the review, §1.4. The attached 34-inch panel's 16:9
canvas is **about 595 mm wide**, i.e. the width of a 27-inch monitor, because a 21:9 panel pillarboxes it.

### 3.3 The verdict, in two parts, because one value would soften it

**`passed at 2 m, failed at 3 m on the actual monitor (cap height; x-height fails at both distances)`.**

1. **At 2 m the screen passes on cap height** — every element clears 16′, the smallest at 16.8′.
   That is a **cap-height** statement. The mixed-case sentences a passer-by actually reads are at
   **12.2′** of x-height — inside the marginal band, below the comfort band — and the frame age's 15.5′
   is just short. **No element clears 16′ of x-height at 2 m**, and the test suite asserts that, so the
   qualification cannot be dropped without a failure.
2. **At 3 m the screen fails on both measures** — the largest cap height on the screen is 14.2′, and the
   body text is 11.2′. That is the achieved result and it is what the deliverable leads with.

The matrix row carries that value verbatim (`qualification-matrix.csv`, the M4 Task 1 row). It is a
value this matrix did not have before M4; it was added the way M2 added `failed` and M3 added `passed
with documented failures`, because rounding a 2 m pass and a 3 m fail into one value would be the
softening this project keeps rejecting.

### 3.4 The 55-inch case is a requirement, and a mutation proves it cannot pass alone

The first version of Task 1 judged the render against a hypothetical 55-inch panel and led with the
pass. That was corrected in `c45ff13`, before any of the work in §4–§7. The correction is not cosmetic:

- **`app/theme.py` now carries a measured block** — `ACTUAL_MONITOR_*` (DELL S3422DWG, 800x330 mm,
  3440x1440 native, 2293x960 desktop, 150 %, **0.3101 mm/logical px**, distances 2.0 and 3.0 m) — and
  `MM_PER_LOGICAL_PX` is documented as **the 55-inch requirement's conversion, not the attached
  monitor's**. Its value did not change; its meaning did.
- **The test class is 16 tests**, of which the attached monitor is the **default and gating** geometry:
  each of its angles is pinned per font size, the 3 m failure is pinned, and one test asserts that no
  element clears 16′ of x-height at 2 m. The 55-inch panel is a **separate, labelled conditional** whose
  test also asserts that it *is* a requirement.
- **Six mutations were constructed and all six were killed** — a wrong panel width (five tests), a wrong
  cap ratio, a wrong x-height ratio, a dropped distance, a shrunk `TYPE_BOOTH_DETAIL` token (six failures
  and one error), and **`angle()` defaulted to the 55-inch conversion, which kills five tests**. That
  last one is the structural answer to "a test that only asserts the requirement must not pass": the
  default geometry is asserted to be the attached monitor and to measure 16.8′ at 2 m, so a
  requirement-only suite fails.

### 3.5 What the redesign cost, and what it bought

The screen was redesigned **by removal**, not by shrinking (`8d4778e`, corrected by `c45ff13`). What left
the always-visible screen, and where it went:

| removed from the aisle surface | where it is now |
|---|---|
| per-camera **backlog**, **analysis rate**, **frame-age window**, **activity line**, **utilisation figure**, **telemetry window** | presenter drawer, per camera, **same item names** |
| the **event counts**, the **`status:`** line, the **`telemetry fresh`** line | presenter drawer |
| the **header scope note** (elided mid-word at 18 px — an M3 finding) | presenter drawer, verbatim and in full |
| the **seven footer `?` keys** and the pinned explanation panel | removed; a 44 px header button, a click on any card, or F1 opens the drawer that holds all seven explanations |
| **`PipelineStrip.qml`** — twelve 18 px labels | **deleted**; `drawerPipelineText` states the topology and the placement rule in one sentence |
| the per-card **hover hint** | removed — it was anchored outside the card and clipped by the card's own `clip: true`, so it could never be seen |

**Kept and enlarged:** the mandatory `Recorded footage · live inference` label, the title, the header
device badge, three live pictures each captioned with its own camera and confirmed device, one column per
camera with name (48 px), state, **frame age (56 px — the largest thing on the screen)** and device
badge, and **one verdict line per camera**: the staleness sentence in words, or that camera's own stated
failure reason.

Two consequences are recorded rather than buried. **A camera with a stated failure reason now shows the
reason instead of its staleness sentence**, because one verdict line per camera is the budget at 44 px
and the reason is the more specific statement; the age, its colour and the card's warning border still
carry the staleness. And **the drawer is 18 px**: it is the presenter's arm's-length surface and is not
part of the passenger legibility contract, so a future task that moves a figure back to the aisle surface
must give it the ladder.

### 3.6 What §3 does not claim

- **No contrast ratio was measured** in M3 or in M4. The palette is high-contrast by design
  (`#e8eef4` on `#141b23`); a number would be a different measurement, and it is absent.
- **The threshold is a reading standard, not a measurement of this room.** ISO 9241-3's 16′ is a
  sustained-reading minimum; it says nothing about ambient light or how long a visitor looks.
- **The four captures still contain no video frame.** They prove layout, strings, pixels and the font
  census, not the video surface — M3's finding stands.
- **`tools/preflight.py` cannot measure a panel.** The 55-inch requirement is a **hardware requirement in
  a document**, and a booth that installs 40 inches or smaller has an M4 regression with no gate to catch
  it. This is the largest open risk in the deliverable.

---

## 4. The input-acknowledgement matrix — MET, and the claim is confirmed in its worst form

M3's exit criterion 6 and M3's matrix row are **closed here**. The obligation was to obtain the
four-condition matrix with tick lateness and frame age **in the same rows**; the result is worse than M2's
claim, and the claim is therefore now measured rather than asserted.

### 4.1 The run

```powershell
.venv\Scripts\python.exe -m tools.input_ack_under_load --seconds 30 --repeats 2 `
    --out logs/qualification/m4-input-ack.json
```

Record: **`docs/qualification/m4-input-ack.json`** (committed, byte-identical to the run file). Exit 0,
wall **699.7 s**. Machine asserted clean before and after (`Get-Process python` → 0; the tool's own census
`"clean": true, "competing_pids": []`), `conditions_failed: []`, `ACKLD_ALLOW_DIRTY` never set.

**Warm-ups were not shortened.** The three-stream conditions held their full **130 s** budget
(`load_applied_s` 163.1 / 193.1 s), because frame age is a consequence of accumulated queue depth and a
six-second window cannot show the booth's real state; `one-cpu-saturated` used the existing gate's 12 s
(45.1 / 75.2 s). Every window is **live** through `app.main.run_windowed`, not seeded. The saturator count
is `cpu_count() - 2 = 6` on this 8-core machine.

### 4.2 The matrix

| condition | backlog per stream | drops | gate | frame age | age p50 / p95 (worst stream per window, as the tool reports it) |
|---|---|---|---:|---|---|
| **three cameras, own devices** (CPU / GPU.0 / NPU) | **1 / 1 / 1** | 0 | **pass** | **fresh** | 65.1 / 38.5 / 39.5 ms per stream (CPU / GPU / NPU); p95 68.2 / 40.3 / 41.3 ms |
| **three cameras, all CPU** (spec §5 row 4) | **587–752** | 29 | **pass** | **stale** | 36.4–44.0 s / 39.0–46.9 s |
| **three CPU cameras + 6 saturators** | **764–1 143** | 32 | **pass** | **stale** | 34.5–55.2 s / 51.2–72.1 s |
| **one CPU camera + 6 saturators** | **83 → 174** | 0 | **pass** | **stale** | 3.48–9.20 s / 5.30–11.35 s |

Tick lateness, from the same rows: **p95 5.6–12.6 ms against a 200 ms gate**, **worst tick 12.3–91.9 ms
against a 2 000 ms gate**. The record's own summary:

```
gate_pass = 8/8        frame_age_stale = 6/8        gate_passes_while_frame_age_stale = 6/8
disagreeing_conditions = one-cpu-saturated, three-cpu-no-saturators, three-cpu-saturated
```

(The tool's raw per-condition table and the per-stream detail, with each stream's confirmed device, are in
`m4-input-ack.json` and in `bench_report.md` §M4.7.)

### 4.3 The verdict, in those words

**The gate is not merely insensitive to inference backlog — it is a green light on a failing system.**

In **every one of the eight measured windows** it passed, with **at least 15× headroom** on its own
thresholds (worst tick 91.9 ms against 2 000 ms; tick p95 12.6 ms against 200 ms), while in **six of those
same windows** the frame age was stale by the application's own 2 s rule. In the two all-CPU layouts —
including the one the booth will actually be in if the presenter leaves all three cameras on the CPU
(spec §5 row 4) — the oldest camera on screen was **34.5–55.2 seconds old at p50**, with **639–1 143
frames queued**, and the gate called it healthy. A green tick beside a picture from most of a minute ago is
not a near miss and is not a threshold to tune; it is the wrong quantity, and that is now a measured
statement.

### 4.4 One qualification, measured rather than conceded

**The gate does see *load*; it cannot see *queue wait*.** Worst tick rose from **20.7 ms** with three
cameras on their own devices to **91.9 ms** under three CPU streams plus six saturators — a real 4× — and
still sat 22× inside the gate. So the honest form is not "the gate sees nothing", it is "the gate looks at
the UI thread's own punctuality while the picture is destroyed by queue latency elsewhere".

**And the same matrix shows the lever.** With one camera per device the application is not merely
acceptable, it is **good**: backlog **1 frame per stream**, **zero drops**, frame age **p50 38.5 / 39.4 /
65.1 ms** across GPU, NPU and CPU, both repeats. The failure is **CPU capacity on a shared device**, not the
screenshot — which is why the booth layout, not the widget, is the fix.

### 4.5 Limits of this record, stated

1. **`tick p50` reads `0.0` in every row and is not informative.** `time.monotonic()` has ~16 ms
   resolution on this machine, the tick period is 20 ms, and early ticks are clamped at zero. The clamp
   count is printed by the probe but **is not carried into the JSON or `AckResult`**, so it cannot be
   quoted for this matrix. Cite **p95 and max**, never p50. That is a defect in the record's field list,
   not in the measurement, and it is recorded here rather than fixed in a documentation task.
2. **The two repeats are not independent restarts.** The application runs continuously, so repeat 1 opens
   on a deeper queue: backlog 639 → 752 and 839 → 1 143, `one-cpu-saturated` 3.5 s → 9.2 s. That is stated
   as a range across repeats, as required, not as one chosen member.
3. **`one-cpu-saturated` is stale but mild** (3.5 s / 9.2 s). The tens-of-seconds figures belong to the
   three-camera CPU layout, and conflating them would overstate that condition.
4. **Drops are not backlog.** The three-camera CPU conditions dropped 29–32 frames per window; the other
   two conditions dropped none. A queue that sheds and a queue that grows are different failures, and both
   are reported.
5. **One machine, one day.** Eight windows on this 8-core Core Ultra 9 288V with Arc 140V and AI Boost. The
   saturator count is machine-specific and not comparable across hardware.

### 4.6 The tool had to be fixed to obtain the matrix

Worth recording because it is why the matrix did not exist before: `spy_feeders` is installed as
`app.main._start_feeders`, and `run_windowed` starts **one feeder per camera**, so with `--streams 3` the
wrapper was entered three times and armed everything three times — **18 saturators spawned**, the probe's
`state["procs"]` holding only the last six, and **twelve survivors waiting on gates nobody sets**. Because
`multiprocessing`'s exit finaliser joins non-daemon children, the probe could not exit: the first full
attempt **reported both `three-cpu-saturated` windows and was then killed at the parent's 1 800 s timeout**
(2 288.5 s wall), losing all four conditions to a temp-directory `PermissionError`. Fixed with an arm-once
guard and `daemon=True` on the saturators; both are covered by tests that fail when either is removed.
**No threshold and no census rule changed** — `GATE_P95_MS = 200.0` and `GATE_MAX_TICK_MS = 2000.0` are
untouched.

---

## 5. The 182 MiB leak at the real 2 s timeout — BOUNDED AND EXPLAINED

M2 left this open: **182.4 MiB total growth, 175.3 MiB unexplained**, at the 2.0 s deadline a presenter
actually experiences, against a clean 15.5 MiB at a widened 30 s. M2 called its cause a hypothesis and
refused to close it. M4 measured it, fixed it, and reproduced the fix and the failure.

### 5.1 The two deadlines, before and after

Instrument: **`tools/leak_probe.py`** (committed). **50 transitions**, M2's count, a **fresh process** per
run, the machine asserted clean in every record, feed **240x320 at 15 fps** — M2's `FEED_SIZE`, so the
numbers are comparable with M2's.

| | 2.0 s deadline, **before** | 2.0 s deadline, **after** | 30 s deadline, **before** | 30 s deadline, **after** |
|---|---:|---:|---:|---:|
| transitions / timed out | 50 / **25** | 50 / **25** | 50 / 0 | 50 / 0 |
| RSS start → end (MiB) | 94.6 → **248.4** | 94.3 → **83.9** | 95.7 → 103.8 | 95.7 → 96.1 |
| **total growth** | **+153.7 MiB** (+3 148.7 KiB/toggle) | **−10.4 MiB** (−212.1 KiB/toggle) | +8.2 MiB | +0.4 MiB |
| retired-queue ledger | 6 → **158** | 6 → 6, **0 after `stop()`** | 6 → 156 | 6 → 6, **0 after `stop()`** |
| unexplained remainder | **+154.1 MiB** | −12.0 MiB | −3.7 MiB | −11.9 MiB |
| transition elapsed p50 / p95 | 3.161 / 4.207 s | 3.114 / 4.262 s | 3.265 / 4.631 s | 3.303 / 4.654 s |
| `stop()` | 0.101 s | 0.166 s | 0.151 s | 0.162 s |
| wall | 149.7 s | 151.3 s | 166.9 s | 168.5 s |

Records: `m4-leak-before-2s.json`, `m4-leak-after-2s.json`, `m4-leak-before-30s.json`,
`m4-leak-after-30s.json`.

**The shape is M2's, the byte count is not, and that is stated rather than smoothed.** M2 measured
182.4 MiB / 175.3 MiB with 24 unacknowledged transitions and 204 ledger entries; this run measured
153.7 / 154.1 with 25 timeouts and 158 entries. Different run, different machine state, same phenomenon —
and the **inversion is the finding**: a **2.0 s** deadline, which times out 25 of 50 transitions, grew
RSS by ~3 MiB **per transition**, while a **30 s** deadline with **zero** timeouts was flat. The 2 s series
is a straight line (+25.2, +57.3, +89.4, +122.1, +153.7 MiB at transitions 10/20/30/40/50); the 30 s
series is flat within allocator noise (+0.9, +1.7, +5.0, +6.0, +8.2).

### 5.2 The cause, measured — and the hypothesis was right in mechanism and wrong on ownership

A timed-out transition retires a worker that is still compiling, so **nothing has read its input queue**.
The supervisor's feeder thread has already put frames into it; the queue's own feeder thread writes them
into a Windows named pipe whose reader (the child) is gone, parks in `Connection._send_bytes`, and
everything it has not sent stays in the queue's `_buffer`. Measured on the drain that the fix performs:
**20 of 52 retire calls discarded anything**, **22–31 frames per non-empty drain** — about 2 s of a 15 fps
feed, which is exactly the 2 s deadline — at **p50 15.2 ms, max 23.8 ms, 0.73 s total**.

**And the list is not the owner.** `tools/leak_probe.py` ends every run by emptying `_transitions` and
`_retired_queues`, collecting, and re-reading RSS. On the unfixed 2.0 s run that released **nothing**: the
remainder was **+154.1 MiB against a +153.7 MiB total**. Dropping the references cannot free frames the
**feeder thread** still holds. The queue count was nearly identical in both deadlines (158 vs 156), so
"more timeouts, more queues" is not the mechanism; the difference is **what is inside them**.

**Two consequences, both recorded:**

- **`cancel_join_thread()` was necessary and is not the fix.** It solves a *rendezvous* problem —
  `multiprocessing`'s atexit finaliser joining a feeder that will never return — and it does that
  correctly. It says nothing about the two seconds of picture the feeder is holding. That is why the
  original fix looked complete.
- **Pruning the ledger alone would have fixed the accounting and left the leak** — the brief's candidate
  fix 2, measured and rejected as a *sole* fix. "The object is still referenced" and "the object owns the
  memory" are different diagnoses, and only the probe could tell them apart.

**M2's hypothesis, scored honestly.** M2 wrote that "each recovery path retains a worker or a queue that a
clean run never builds". Half right: the **path** is the timeout path, and a retired queue is what leaks.
Wrong on the second half: the clean run builds the **same number** of queues. Ownership, not count.

### 5.3 The fix, and the two measured choices inside it

`app/supervisor/runner.py`'s `_retire()` now, in order: `cancel_join_thread()` (unchanged, still first);
**discard what the queue still holds**; `close()`; wait a bounded `RETIRE_FEEDER_GRACE_SECONDS` for the
feeder to see the sentinel; and **drop the ledger entry only if that thread is confirmed gone**. A queue
whose feeder survives stays in the ledger and is cancelled again by `stop()`'s sweep.

- **`get(timeout=…)`, not `get_nowait()`, and that was measured.** `get_nowait()` asks `PeekNamedPipe`,
  which reports **zero bytes for a message whose write is still in flight**; an early version left **4 of
  8** retired queues with a live blocked feeder. Polling the pipe handle *is* signalled for a partial
  message, and its read completes it: **6 of 6 trials** drained all eight frames and the feeder exited
  within 300 ms. The 2 ms timeout is why the drain costs what it costs.
- **Pruning is gated on evidence.** The ledger has two other readers — `stop()`'s re-entry guard and
  `stop()`'s final sweep — and both stay correct: with live queues registered the guard is false and a real
  `stop()` proceeds; after a stop that released everything it is true and a second `stop()` returns
  immediately. The sweep now sweeps the **unreleased** queues, which is what it was always for.

**The fix was not paid for with speed.** Transition elapsed **p50 3.161 → 3.114 s**, p95 4.207 → 4.262 s,
max 4.230 → 4.273 s across the same 50 transitions. The drain's 0.73 s total is inside that.

### 5.4 The atexit hang is still protected — and the proof hangs when it is removed

`multiprocessing`'s atexit finaliser joins its feeder threads; a retired queue whose feeder is parked for
ever is an interpreter that never exits. Three pieces of evidence, none of them a claim:

1. **`stop()` is prompt after real timeout traffic**: 0.166 s at the 2.0 s deadline and 0.162 s at 30 s,
   against a `STOP_GRACE_SECONDS` of 5.0 — no grace is being paid.
2. **Both probe children exit 0** after 50 timeout-heavy transitions, which is the finaliser running with
   the ledger in whatever state the run left it.
3. **A dedicated regression test forces the pre-fix shape and would hang without the protection**:
   `tests/test_runner.py::RetiredQueueRelease::test_the_interpreter_still_exits_when_a_retired_queue_cannot_be_drained`
   constructs a blocked feeder, makes the drain a no-op, retires the queue, calls `stop()` and requires
   `ledger 1`, `joincancelled True` and a normal exit inside 90 s. It passes in ~1.4 s. **With
   `cancel_join_thread()` removed it hangs and is killed at 120 s** — the hang the protection exists to
   prevent, reproduced on demand.

Two existing test expectations had to be corrected because they **encoded the leak as design**:
`retired_queues_end - start > 0` ("retired queues are kept too") now asserts they are released, with the
bound `2 × streams + 2`. **A test that asserts a leak is permanent is a test that protects the leak.**

### 5.5 What §5 does not claim

- **M2's exact 182.4 MiB is not reproduced to the byte.** 153.7 MiB here, same shape and order.
- **A 1080p run was not measured.** The feed is 240x320, M2's, for comparability. The ~27× byte ratio to
  1920x1080 is **arithmetic**, labelled as arithmetic.
- **RSS falling by 10.4 MiB is not memory "returned to the OS".** It is the allocator handing arenas back;
  the meaningful statement is that the ledger goes 6 → 0 and the growth is inside noise.
- **This does not bound a shift-long soak.** 50 transitions is M2's count; a 60 s+ unattended soak is M6's.
- **The 30 s deadline is not what a presenter experiences.** It exists only to separate the two conditions;
  the product's default is untouched and the probe prints which deadline it used.

### 5.6 One gate was changed in M4, and it is flagged here rather than absorbed

`tests/test_route_stress.py::test_nothing_grows_that_the_retained_records_do_not_account_for` — the
pre-existing RSS-ratio flake M2 recorded as open — failed in a full-suite run during Task 3 with
`-5709824 not less than or equal to -7155712 ... which is backwards: releasing memory cannot cost memory`.
**Both readings were negative**: the process was giving memory back and the "release" gave back 1.4 MB
less. The asset the assertion was written for no longer exists: after this fix the retention it empties is
**seven `_Transition` records**, and freeing seven small mappings cannot move RSS down.

Two changes, both measurements rather than thresholds:

- `tools/route_stress.py` now reads RSS **before** the `gc.get_objects()` census, not after it — one census
  over ~9 852 tracked objects raised RSS by **176 KiB and kept it** (five calls, plateau).
- The assertion gained an **8 MiB tolerance on the inversion direction only**, documented with the measured
  facts. **The strict half is untouched**: `leak_bytes < rss_start`. The regression the tolerance could hide
  would have to be smaller than 8 MiB when the pre-fix retention was **154.1 MiB** — **5% of the effect it
  protects**.

**This is the one gate M4 changed, it is a judgement call, and it is recorded for review rather than
buried.** What changed is *what the number is allowed to mean* when the thing it was written to detect has
been removed; no growth bound was loosened.

---

## 6. GPU and NPU utilisation — ATTRIBUTED per process, closing M0's unmet gate 2

M0's second gate recorded that the busiest `engtype_Neural` and `engtype_Compute` instances on this machine
belonged to **other processes**, that the inference process's own pid **never appeared as a counter
instance**, and that **"no utilisation figure may be shown until M4 can attribute it."** M4 attributes it.
The full route-by-route account is `M4-utilisation-attribution.md`; this section states the outcome, the
control and the limits.

### 6.1 The two-milestone blocker was a bug in this repository

`PdhGetFormattedCounterValue` takes **four** arguments — `(hCounter, PDW_FMT_DOUBLE, lpType, pValue)` —
and had been bound with **three**, which terminates the process with `0xC0000005`. The "platform cannot do
this" conclusion drawn at M0 was therefore false for two milestones. Correctly bound, `rc = 0` and
`\Memory\Available Bytes` returned **25 560 748 032.0**. Compounding it, `PdhGetFormattedCounterArrayW`'s
items are **24-byte `PDH_FMT_COUNTERVALUE_ITEM_W`** structures and were being read as 16-byte values. Both
bindings are now pinned by tests that run in child processes, so a regression is observable as a process
exit rather than as a silent zero.

**Ruling, because it is the transferable lesson: "the platform cannot do this" is a claim like any other,
and this one was false.**

### 6.2 The counter is per-PID by construction

`\GPU Engine(*)\Utilization Percentage` instance names carry `pid_<n>_luid_<…>_phys_<n>_eng_<n>_engtype_<…>`
— **335–361 instances** on this machine. There is no **attributable** machine-wide figure here and none is
produced: the provider filters each instance by **this application's own pid** and by the identified
adapter.

- **GPU identification**: DXGI `IDXGIAdapter::GetDesc` returns description, vendor, device and LUID
  together — `Intel(R) Arc(TM) 140V GPU (16GB)`, PCI `8086:64A0`, LUID `0x000113B4`.
- **NPU identification**: there is no Intel- or NPU-named counter set on this build. The NPU appears as
  `engtype_Neural` instances under its **own** adapter LUID `0x000117B6`, and is identified as the one
  adapter exposing a Neural engine. Zero or two candidates yields `None` with the reason — never a guess.
  **A GPU result does not imply an NPU one.**
- The value read costs **0.37–0.84 ms** for the whole array (n=30); a full sample is 1.77–3.38 ms.

### 6.3 The control — what makes this a measurement rather than a number

`tools/probe_utilisation.py`, committed output `docs/qualification/m4-utilisation-control.json`
(`measured_at` 2026-10-27T15:29:24). **The machine was asserted clean before any worker started**: 349
live instances, busiest **0.00%**. One worker was idle for 5 s, then ran OpenVINO inference for 15 s, and
a second provider sampled the probe's own pid throughout.

```
GPU:  idle n=5  median 0.0  max 0.0   |  busy n=26  median 97.19  min 97.06  max 100.0
      probe's own pid while busy: max 0.0 (n=28)
      busiest row while working: 100.00% pid_13728_..._eng_5_engtype_Compute
      next-busiest: 1.42% pid_13116_..._eng_0_engtype_3D
NPU:  idle n=5  median 0.0  max 0.0   |  busy n=26  median 64.86  min 63.76  max 66.08
      probe's own pid while busy: max 0.0 (n=28)
      busiest row while working: 66.97% pid_15624_..._eng_0_engtype_Neural
      next-busiest: 2.79% pid_13116_..._eng_0_engtype_3D
verdict GPU: busy=True idle=True distractor=True -> PASS
verdict NPU: busy=True idle=True distractor=True -> PASS
```

- **It moves when this process's load moves**: 0.0 → 97.19 (GPU) and 0.0 → 64.86 (NPU) on the same pid.
  A stale, cached or machine-wide figure would not.
- **It is not another process's**: the worker's engine reads 97–100% while the **probe's own pid reads
  0.0%**, and another process on the **same** adapter reads 1.42% (GPU phase) and 2.79% (NPU phase).
- **The control is load-bearing**: removing `PdhCollectQueryData` from `read_array()` makes the same probe
  report `busy median 0.0` for both devices and exit `VERDICT: FAILED` (1). That is exactly the bug the
  first probe run caught.

**This is the control the withdrawn 62%/68% figure never had.** At M0 the load's own pid appeared in
**0 of 469** samples; here the worker's own pid is the busiest instance in every busy sample, and the
probe's pid — a different process on the same machine — reads 0.0% in all of them.

### 6.4 What the cards do now — three states, none of them a fabricated zero

- **A sample exists** → a reading with its measurement window, e.g. `GPU 97.2%` over 250 ms. Availability
  is derived from the value itself (`gpu_available = metrics[...] is not None`), so a number and
  "available" cannot drift apart.
- **A measured `0.0%`** → **renders as a reading.** An idle engine that exists is a measurement, not an
  absence.
- **No engine instance for this application** → **`Utilization unavailable`** with a stated reason:
  *"this application has no GPU engine instance in this counter sample: Windows creates an instance only
  for a process that has opened an engine on that adapter. No instance is not a measured 0."*

The old "counters fault this process" wording is gone. A live run on this machine **renders `GPU 0.0%`**
on a CPU route, because the UI's own 3D engine is idle — an honest zero, and new on the booth screen, so
the presenter script must say what it means.

### 6.5 The 62% / 68% figure stays withdrawn

M0's machine-wide peak is **not** reinstated, and the provider has no path to produce one. It appears in
the product's own wording only as *withdrawn*, and `grep` finds it nowhere as a current figure. The
reason stands: **a figure that does not move when this process's load moves is not this process's
figure.**

### 6.6 Why every route is listed, including the ones that failed

`M4-utilisation-attribution.md` §4 names each route tried and its failure mode: in-process PDH after the
binding fix (**works, shipped**); a helper process (**not needed** — the in-process call works, and a
helper adds a spawn per sample and a new silent-absence mode); `Get-Counter` / CIM (**correct but
unusable at cadence** — 1 099–1 458 ms per `Get-Counter` call, 329–335 ms warm and 3 185 ms cold per CIM
query, and the only stdlib route is a PowerShell process per sample, so it is used as an independent
cross-check only); Intel/NPU/VPU counter sets (**do not exist on this build**); `engtype_Neural` under the
NPU's LUID (**works per process, shipped**); DXGI `GetDesc` (**works, shipped** — and it is the
LUID↔PCI link `config/telemetry_map.json` claimed did not exist).

### 6.7 Limits of §6, stated

1. **The published figure is the busiest engine, not a device total.** A process using several engines in
   parallel is understated relative to a sum. That is a stated meaning, not an error — but **a presenter
   comparing it with Task Manager may see different numbers**.
2. **The NPU's ~65% is the engine's busy fraction, not its ceiling**; the inference loop spends host time
   between inferences.
3. **A first provider construction costs ~0.75 s** (the rate-counter prime), once per spawn, on the
   collector-rebuild path. It is visible and it is not hidden.
4. **`config/telemetry_map.json` is stale and was NOT updated** — its LUIDs are per-boot values from
   2026-10-01 and this boot uses different ones. The runtime no longer depends on it; preflight reports the
   discrepancy as §2.2's single WARN. **The constraint held even though regenerating the file would have
   made the WARN disappear: a cleaner-looking gate is not a better one.** Refreshing it is an operator step
   (`tools/probe_telemetry.py`), not a runtime one.
5. **One adapter LUID remains UNATTRIBUTED** (`0x00011232`, 81 instances, only `3D` engines). It identifies
   neither device and was left unattributed rather than assigned to the nearest one.

---

## 7. Sustained paired trials — MET, and they rank less than expected

The obligation was to compare the devices under stated, sustained, paired conditions, and not to
generalise past them. The headline is a negative result: **at the booth's own rate the throughput ranks
nothing**, and the comparison that *does* separate the accelerators is the one that is not the booth's
stream.

### 7.1 The campaign

```powershell
.venv\Scripts\python.exe tools\sustained_trials.py --seconds 20 --warmup 3 --repeats 3 `
    --json docs\qualification\m4-sustained-trials.json
```

**27 trials** = 3 conditions × 3 devices × 3 repeats, **each in a fresh process**, **interleaved and
rotated** (rep0 CPU→GPU→NPU, rep1 GPU→NPU→CPU, rep2 NPU→CPU→GPU), so thermal or background drift lands on
every device roughly equally instead of on whichever ran first and last. **540.33 s** of measured window,
**44 025** frames analysed. Campaign wall **758.8 s**. Machine asserted clean before the campaign and
polled every 2 s during it (`"clean": true, "competing_pids": []`).

Warm-up was identical for every trial and the compile is **outside** every window: the first frame is
submitted alone, the queue is drained, 3 s of warm-up is fed, the clip is rewound to frame 0 through the
application's own loop-reset semantics, the queue is drained again, and only then does the measured window
open. **Every window starts with an empty pipeline**, so the frame age it reports is that device's queue,
not the compile's backlog. The NPU's compile is visible in the wall times: **30.1 s** for an NPU trial
against **28.1 s** for the CPU or GPU trial with the same window and warm-up, and 26.1 s for `sustained`.

### 7.2 Placement, on every run

`EXECUTION_DEVICES` is read from each request's own compiled model **after** inference and normalised
through `normalise_execution_devices` (the NPU plugin reports the bare string `'NPU'`, the CPU `['CPU']`,
the GPU `['GPU.0']`). Every reading in a window had to agree; two different readings in one window is a
refusal, not a choice. **7 909 readings** in total (2 700 booth, 5 191 source, 18 sustained).

**Placement confirmed on all 27 runs; zero fallbacks, zero failed frames, zero drops.** A row cannot
report a request it did not receive, and `TrialResult` refuses a row whose `confirmed` does not name
`requested` when `fell_back` is false.

### 7.3 The table

| condition | CPU | GPU | NPU |
|---|---|---|---|
| **booth feed, 15 fps** (real pipeline) | 300/300 frames, 14.99–15.00 fps | 300/300, 14.99–15.00 fps | 300/300, 15.00 fps |
| **source rate, 29.97 fps** (real pipeline) | **26.3–26.6 fps**, 527–532 frames, age p50 **1.19–1.31 s** | 600/600, 29.96 fps | 600/600, 29.96 fps |
| **isolated sustained `infer`**, batch 1 (no decode, no queue, no display copy) | 45.4 fps | **307.4–310.4 fps** | **246.4–249.1 fps** |

Latency, p50 / p95, ms:

| condition | CPU | GPU | NPU |
|---|---|---|---|
| booth | 22.52–22.59 / 29.46–32.14 | 4.39–4.58 / 4.57–5.24 | 6.60–6.63 / 6.82–6.85 |
| source | 23.43–23.80 / 26.09–26.60 | 4.41–4.51 / 4.68–5.22 | 6.44–6.52 / 6.65–6.71 |
| sustained | 21.57–21.61 / 23.65–23.70 | 3.09–3.11 / 3.89–3.93 | 3.96–4.01 / 4.29–4.33 |

Frame age, p50 / p95, ms, from `captured_at` — a **separate column on purpose**: at `source` the CPU's
inference p50 is 23.4–23.8 ms while its frame age p50 is **1.19–1.31 s** (p95 2.17–2.35 s). The difference
is queue, and adding or averaging the two would produce a number no measurement produced.

| condition | CPU | GPU | NPU |
|---|---|---|---|
| booth | 48.12–48.50 / 55.69–57.45 | 30.57–31.12 / 32.38–32.65 | 32.56–33.11 / 33.97–34.94 |
| source | **1 189.39–1 311.82 / 2 169.42–2 346.42** | 29.43–29.66 / 30.56–30.75 | 30.97–31.09 / 31.93–32.73 |
| sustained | n/a (no queue) | n/a | n/a |

`n/a` means **not applicable, not zero**.

### 7.4 What the comparison supports — and what it does not

**Supported, on this machine, this model and this workload:**

1. **At the booth's own feed rate the three devices are indistinguishable on throughput, and all three
   deliver it.** 300 of 300 frames, 14.99–15.00 fps, zero drops, every repeat, on all three. **The booth's
   stream is not device-limited here.** What differs is latency (p50 4.4–4.6 ms GPU, 6.6 ms NPU, 22.5–22.6
   ms CPU) and frame age (30.6–31.1 / 32.6–33.1 / 48.1–48.5 ms p50).
2. **Asked for every source frame, the two accelerators hold it and the CPU does not.** GPU 600/600 and NPU
   600/600 at 29.96 fps with a frame age p50 of ~30 ms; CPU 527–532 of ~600 at 26.3–26.6 fps, **1.2–1.3 s
   behind**. This is the only throughput ordering the delivered stream establishes, and it is **CPU vs the
   accelerators, not GPU vs NPU**.
3. **Isolated sustained inference is the only condition that ranks the two accelerators: GPU 307.4–310.4
   fps against NPU 246.4–249.1 fps, 1.24–1.25×, in each of three interleaved repeats**, with p50 3.09–3.11
   against 3.96–4.01 ms (1.27–1.29×). **The single-frame ordering did not invert at sustained load**, and
   the integrated-Arc-shares-memory hypothesis in the brief did not materialise on this workload. It is
   reported as **refuted**, not softened.
4. **The NPU figure quoted since M0 is cold.** In the same process and the same trial, the median of the
   **first 30** inferences was **5.50–5.57 ms** while the 20 s window measured **3.96–4.01 ms** — a factor
   of 1.37–1.40. The GPU's first 30 were 3.02–3.10 against 3.09–3.11 (1.00) and the CPU's 21.47–21.63
   against 21.57–21.61 (1.00), so **this is the NPU's settling, not the instrument's**. The published
   5.2–5.9 ms is an honest measurement of a 30-inference burst in a fresh process that **overstates the
   NPU's sustained cost by about 40%**. **Any document quoting 5.9 ms beside a sustained number must say
   which one it means; M0, M1 and M3 were not edited, and this paragraph is the correction.**

**Not supported:**

- **No GPU-vs-NPU throughput ranking of the delivered stream, in either direction.** Both accelerators
  deliver the clip's rate at both feed rates, so delivered-stream throughput is set by **the source**, not
  the device. A booth claim of the form "the NPU is faster than the GPU" (or the reverse) is not supported
  by these rows.
- **No general claim about "the NPU" or "the GPU".** One machine, one model (`yolo11n-fp16`), one shape
  (640x640, batch 1), one clip, one OpenVINO build, one pair of driver versions. Nothing transfers to
  another SKU, a different batch size, a dynamic shape, a heavier model or a different driver.
- **No claim that `sustained` is what the booth runs.** It has no decode, no queue and no display copy. It
  is the device's own cost and is reported as a separate condition for exactly that reason.
- **No causal explanation of the pipeline/isolated gap.** The NPU's in-pipeline p50 is 6.44–6.63 ms against
  3.96–4.01 isolated (≈1.6×); the GPU's rises much less (4.39–4.58 against 3.09–3.11, ≈1.4×). Whether
  concurrent decode, pickling, the display copy or the per-frame placement read is responsible is **not
  established** here, and it is recorded rather than explained.

### 7.5 The retail claim survived the measurement unchanged

**Two `zone_occupancy` presence events per clip**, on all **18 pipeline windows** across both feed rates and
all three devices — 36 events, `dedup_key`s of the form `1:<track_id>:person:presence:counter_band` — plus
one full pass through the application's own headless entry point analysing **1 694 of 1 694** frames and
reporting `event_counts {'zone_occupancy': 2}`. 1 694 analysis frames is the whole 3 387-frame clip at step
2, so a full pass could not have hidden a third event.

**Track ids are not stable across feed rates** (14 / 71–74 at 15 fps, 18 / 116–119 at 29.97): the tracker
assigns ids in confirmation order and the two rates see different frames. The **count, kind and zone are
identical**, and a future test must not pin a track id.

**This remains people-in-zone presence — not a people count and not a dwell measurement.** `BusinessEvent`
carries no duration, the shelf-to-basket requirement is unmet by design, and nothing in §7 changes that.

### 7.6 Memory, recorded and not asserted

`rss_start`/`rss_end` are the **host resident set of the trial's process tree**, not accelerator memory.
The deltas: `booth` **+33.0 to +41.8 MiB** on all three devices; `source` GPU **+39.0 to +39.4**, NPU
**+47.3 to +47.5**, CPU **+443.6 to +474.3**; `sustained` CPU +0.1, NPU +0.3 and the GPU **−8.0 MiB**.
**The CPU's `source` growth is not a leak claim** — it is the backlog in the same row (1.19–1.31 s at ~30 fps
is 35–40 unconsumed 1920x1080 frames at 6.2 MB each, plus their pickled copies). **The row with the growth
is the row with the lag.** No threshold is asserted anywhere in this tool or its tests; a leak claim needs
`tools/leak_probe.py`'s method, not this one.

### 7.7 Limits of §7

1. **`sustained` `lat_max` sits far above p95 on all three devices** (GPU 6.68–19.68 ms against p95
   3.89–3.93; NPU 5.62–14.48 against 4.29–4.33; CPU 37.39–38.59 against 23.65–23.70). These are single-call
   outliers consistent with a scheduler or driver stall; **the cause is not identified**. They are reported
   untrimmed and no percentile in this report is computed from a trimmed sample.
2. **The 6.5 ms pipeline NPU cost against 4.0 ms isolated is unexplained** (§7.4).
3. **One pair of drivers, one runtime build.** A driver update invalidates the blob cache (AGENTS.md rule
   13) and requires this campaign to be **re-run before any of its numbers is quoted again**.
4. **The `sustained` rows measure `infer` only.** They cannot speak to decode, display, or the UI thread.

---

## 8. M4 exit criteria — stated one by one

| # | criterion (plan §M4) | verdict | evidence |
|---|---|---|---|
| 1 | Every element a passer-by must read is measured on **the monitor actually attached** — clears 16′ of cap height at 2 m, and its failure at 3 m is documented | **MET IN ITS DOCUMENTED FORM ONLY — the failure is real** | §3.2–3.4: at 0.310 mm/logical px, 44 px is **16.8′** at 2 m (pass) and **11.2′** at 3 m (fail); 56 px is 21.4′ / 14.2′; **x-height fails at both distances** (12.2′ / 8.1′). The 55-inch panel (22.9′ / 16.6′) is a **procurement requirement**, and a mutation that defaults the test to it **kills five tests**. `tests/test_capture_ui_canvas.py::TheBoothScreenReadsAtThreeMetres`, 16 tests |
| 2 | The four-condition input-ack matrix exists, with tick lateness and frame age in the same rows | **MET** | §4: `m4-input-ack.json`, 8 windows, `gate_pass 8/8`, `frame_age_stale 6/8`, backlog 1 through 1 143 and age 38.5 ms through 55.2 s in the same rows as tick p95 5.6–12.6 ms |
| 3 | The 182 MiB leak at the real 2 s timeout is bounded and explained, or its cause is stated | **MET — bounded, explained and fixed** | §5: +153.7 MiB → **−10.4 MiB** over the same 50 transitions and 25 timeouts; ledger 6 → 158 → **6 → 0**; cause is the retired queue's blocked feeder thread, measured; pruning alone released **nothing**; the 30 s case stayed clean (+8.2 → +0.4 MiB) |
| 4 | Sustained paired trials exist per device, with conditions stated and no unsupported generalisation | **MET** | §7: 27 trials, fresh process each, interleaved and rotated, 540.33 s, 44 025 frames, **placement confirmed 27/27, zero fallbacks**; the wording states what the rows do and do not rank |
| 5 | GPU/NPU utilisation is attributed per process or refused with evidence | **MET, BY ATTRIBUTION** | §6: `m4-utilisation-control.json` — GPU 0.0 → 97.19%, NPU 0.0 → 64.86%, the probe's own pid 0.0% while the worker read 97–100%, a distractor on the same adapter at 1.42%; the machine-wide 62%/68% figure stays withdrawn |

**Criterion 1 is the one that needs its wording read carefully.** The plan's first clause is "clears 16′
of cap height at 2 m", which is met; the second is "its failure at 3 m is documented", which is met by
`M4-legibility-review.md` and §3 here. **What is not met is the sentence a reader may want to hear — that
the booth screen reads at three metres on this machine's monitor.** It does not, and no rounding of this
table will make it.

---

## 9. NOT closed — and not written as closed

| item | status | owner |
|---|---|---|
| **Legibility at 3 m on the attached monitor** | **FAILS** — 11.2′ of cap height and 8.1′ of x-height at 44 px. The 55-inch panel that would fix it is a **procurement requirement with no gate** (`preflight` cannot measure a panel) | procurement / M6 |
| **M2's 2 s warm-transition target** | **STILL MISSED, and untouched by M4.** p50 2.238 s and **37 of 50 toggles over 2 s at the real 2.0 s default** — M2's committed distribution stands; **M4 did not re-measure it** and does not claim to have moved it | M6 |
| **M0 gate 9's SILENT half** (a fallback that happens without the UI saying so) | **OPEN and cannot close on this machine.** All three devices execute where asked, so `confirmed != requested` never occurs naturally, and the compile-failure seam is a loud failure. No fault was invented to manufacture a silent one | M6 fault injection |
| **GPU/NPU figure is the busiest engine, not a device total** | **STATED MEANING, NOT A CLOSED GAP.** A presenter comparing it with Task Manager may see different numbers | presenter guide |
| **Retail shelf-to-basket** | **UNMET BY DESIGN.** The delivered event is **people-in-zone presence, 2 per clip** — not a people count, not a dwell measurement | design / M5 |
| **Code signing** | **UNMET.** M6 ships unsigned with the step documented | M6 |
| **A second annotated frame with a non-square object** | **NOT DONE.** The `w`/`h` transposition blind spot stands — a transposed box passes the current fixture | M1 follow-up / M5 |
| **A 60 s+ unattended soak** | **NOT RUN.** §5 bounds 50 transitions, not a shift | M6 |
| **Contrast ratio / ambient light** | **NOT MEASURED**, in M3 or M4 | presentation |
| **The drawer is 18 px** | **DELIBERATE** — the presenter's arm's-length surface, not the passenger contract | design |
| **`tick p50` clamp count and the two non-independent repeats** | **RECORDED LIMITS** of the matrix (§4.5), not defects to fix here | M6 / tool owner |
| **The 8 MiB RSS-ratio tolerance** | **A CHANGED GATE, FLAGGED FOR REVIEW** (§5.6) | reviewer |
| **`config/telemetry_map.json` is stale** (per-boot LUIDs) | **REPORTED, NOT EDITED.** The runtime no longer depends on it; refreshing it is an operator step | operator |
| **`sustained` `lat_max` outliers; retail track ids across feed rates; the 6.5 vs 4.0 ms NPU gap** | **UNEXPLAINED, RECORDED** (§7.7) | M6 |
| **One driver pair / one OpenVINO build** | **A DRIVER UPDATE INVALIDATES THE BLOB CACHE** and requires §6–§7 to be re-run before requoting | operator |

---

## 10. Known costs and limits of M4

- **The suite is the expensive gate** — this session's run is in §2.4, and the count grew
  **1 376 → 1 459** across M4: legibility **17** (11 new, 1 attract-mode ladder, 5 attached-monitor), the
  matrix tool **2**, the retired-queue release **3**, sustained trials **39**, utilisation **22** — with the
  removals and retargets stated in each task report. Most of the wall time is live two- and three-process
  Qt/inference work, not unit tests.
- **M4 changed two things a future reader must know about.** `PipelineStrip.qml` was **deleted** (a removal,
  not a layout edit), and one test in `tests/test_route_stress.py` was **re-stated** with an 8 MiB
  inversion-only tolerance (§5.6). Neither is a threshold being lowered to make a gate green; both are
  recorded as what they are.
- **One matrix status value exists because of M4**: `passed at 2 m, failed at 3 m on the actual monitor
  (cap height; x-height fails at both distances)`. A reader matching on exact status strings will see it;
  that is deliberate and stated.
- **The measurement discipline is now the default, and it cost time.** Four measurements each needed an
  idle machine; the campaign alone is 758.8 s of wall, the matrix 699.7 s, and two leak conditions ~150 s
  each. Task 3 needed **three full-suite runs**: the first failed on an assertion that encoded the leak as
  design plus a test's global feeder-thread check, and the second on the RSS flake that §5.6 now bounds.
- **`logs/` and `.superpowers/sdd/` remain gitignored.** The mutation tallies the task reports cite
  (legibility 6/6, leak fix 5/5, matrix 2/2, utilisation 8/8, sustained trials 38/38) are **recorded there
  and not committed**; this report does not restate them as committed evidence, and no gate above depends
  on one. What is committed is the tests they were aimed at, in the suite that §2.4 runs.
- **The 55-inch requirement is unenforceable by tooling.** It is the largest open risk in the deliverable
  and the one most likely to be forgotten between now and the booth.

---

## 11. What M4 establishes

- **The booth screen is measured where it will be shown, and the failure is documented rather than
  argued away**: 16.8′ of cap height at 2 m and 11.2′ at 3 m, with x-height short at both distances —
  and a 55-inch panel stated as the requirement that would change that.
- **The input-ack gate is a green light on a failing system**, measured in four conditions and eight
  windows, with the frame age and the backlog in the same rows as the tick lateness that passes.
- **The 182 MiB leak is gone, and the reason is known**: a retired queue's blocked feeder thread held the
  frames; draining releases them, pruning alone does not, and the atexit-hang protection is intact and
  proven by a test that hangs when it is removed.
- **Accelerator utilisation is this application's own figure**, per pid, with a control that moves when
  this process's load moves and does not move for a distractor on the same adapter — at the stated cost
  that it is the busiest engine and not a device total.
- **A device comparison now exists and it does not flatter either accelerator:** all three devices deliver
  the booth's stream; the CPU cannot hold the source's; and only isolated sustained inference ranks GPU
  against NPU (1.24–1.25× in the GPU's favour), with the single-frame ordering **not** inverting.
- **The recorded NPU single-frame figure is cold**, and this milestone says so in the same breath as the
  sustained number instead of leaving a reader to find the inconsistency.

---

## Appendix A — every number in this report, and where it came from

| claim | source |
|---|---|
| monitors and geometry: DELL S3422DWG 80x33 cm, 3440x1440, 2293x960, 0.310 mm/logical px | WMI `WmiMonitorBasicDisplayParams` / `WmiMonitorID` / `Win32_VideoController`, dated 2026-10-27; derivation `M4-legibility-review.md` §1.1; constants `app/theme.py::ACTUAL_MONITOR_*` |
| 44 / 48 / 56 px angles at 2 m and 3 m on both geometries | `docs/qualification/m3-capture/100pct.json` read through the test's own `angle()`, with Arial cap 0.7164 em and x-height 0.5184 em measured by `QFontMetricsF`; pinned in `TheBoothScreenReadsAtThreeMetres` |
| the redesign's removals and their destinations | `8d4778e` / `c45ff13`, `app/ui/PresenterDrawer.qml`, `M4-legibility-review.md` §4; the capture harness's `BOOTH_EXPECTED` list |
| 24 visible named elements; census `{44: 18, 48: 3, 56: 3}` | `docs/qualification/m3-capture/100pct.json` `strings`/`items_seen`, re-rendered in M4 |
| `gate_pass 8`, `frame_age_stale 6`, `gate_passes_while_frame_age_stale 6` | `docs/qualification/m4-input-ack.json` `summary` |
| backlog 1/1/1; 587–752; 764–1 143; 83→174; drops 0/29/32/0 | `m4-input-ack.json` `results[*].backlog_per_stream`, `.drops` |
| frame ages 38.5–65.1 ms; 36.4–44.0 s; 34.5–55.2 s; 3.48–9.20 s | `m4-input-ack.json` `results[*].frame_age_p50_ms` and `_by_stream` |
| tick p95 5.6–12.6 ms; worst tick 12.3–91.9 ms; thresholds 200 / 2 000 ms | `m4-input-ack.json` `results[*].tick_p95_ms`, `.tick_max_ms`, `gate_thresholds_quoted_not_tuned` |
| 130 s warm-up budget; `load_applied_s` 163–193 s; 12 s / 45–75 s | `m4-input-ack.json` `backlog_warmup_note`, `results[*].conditions_documented` |
| leak 94.6 → 248.4 MiB, +153.7, ledger 6 → 158, remainder +154.1 | `m4-leak-before-2s.json` (`rss_start`, `rss_end`, `leak_bytes`, `unexplained_bytes`, `retained_queues_end`) |
| leak 94.3 → 83.9 MiB, −10.4, ledger 6 → 6 → 0 | `m4-leak-after-2s.json` (`leak_bytes`, `retained_queues_*`, `retained_queues_after_stop`) |
| 30 s: +8.2 MiB before, +0.4 MiB after | `m4-leak-before-30s.json`, `m4-leak-after-30s.json` `leak_bytes` |
| p50 3.161 → 3.114 s, p95 4.207 → 4.262 s (2.0 s); 3.265 → 3.303 s, 4.631 → 4.654 s (30 s) | the two 2.0 s records' `elapsed_p50_s` / `elapsed_p95_s`, and the two 30 s records' same fields |
| 20 of 52 drains discarded anything; 22–31 items; p50 15.2 ms, max 23.8 ms, 0.73 s | `tools/leak_probe.py`'s drain census, quoted in the Task 3 report §2 (the probe prints it; the count is not carried in the JSON) |
| `get_nowait` left 4 of 8 queues; `get(timeout)` drained 6 of 6 | direct measurements recorded in the Task 3 report §4 |
| the atexit test hangs at 120 s when `cancel_join_thread()` is removed | `tests/test_runner.py::RetiredQueueRelease` mutation M5, Task 3 report §7 |
| RSS-ratio flake: −5 709 824 ≤ −7 155 712; 176 KiB census; 8 MiB = 5% of 154.1 MiB | Task 3 report §6.2; the tolerance is in `tests/test_route_stress.py` |
| GPU 0.0 → 97.19% (n=26, 97.06–100.0); NPU 0.0 → 64.86% (63.76–66.08); probe pid 0.0; distractor 1.42% / 2.79% | `docs/qualification/m4-utilisation-control.json` `phases[*]`, `verdict` |
| 335–361 instances; 349 live at clean; 0.37–0.84 ms array read; 1 099–1 458 ms `Get-Counter`; 329–335 / 3 185 ms CIM | `M4-utilisation-attribution.md` §1, §4; `tools/probe_utilisation.py` output |
| `25 560 748 032.0` from `\Memory\Available Bytes` after the binding fix; 24-byte array items | `app/telemetry/pdh.py` and its child-process tests, `M4-utilisation-attribution.md` §1 |
| LUIDs `0x000113B4`, `0x000117B6`, `0x0001174C`, `0x00011232`; PCI `8086:64A0` / `1414:008C` | DXGI `GetDesc` via `app/telemetry/dxgi.py`; `m4-utilisation-control.json` `machine.adapters`; preflight §2.2 |
| 27 trials; 540.33 s; 44 025 frames; 758.8 s; 7 909 placement readings | `m4-sustained-trials.json` `conditions.wall_s` and the sum/rows of `results` (computed); `M4-sustained-trials` report §3, §5 |
| booth 300/300 ×3; source 600/600 GPU/NPU, CPU 527–532 at 26.3–26.6 fps; sustained 307.4–310.4 / 246.4–249.1 / 45.4 fps | `m4-sustained-trials.json` `results[*]`, `ranges` |
| p50/p95 latencies and frame ages per condition | `m4-sustained-trials.json` `results[*].latency_*`, `frame_age_*`, `ranges[*].summary` |
| NPU first-30 5.50–5.57 ms vs 3.96–4.01 ms; GPU 3.02–3.10 vs 3.09–3.11; CPU 21.47–21.63 vs 21.57–21.61 | `m4-sustained-trials.json` `results[*].cold30_p50_ms` and `latency_p50_ms` |
| 30.1 s vs 28.1 s per-trial wall (NPU compile ≈2 s); 26.1 s `sustained` | the campaign's per-trial wall lines, Task 4 report §3 |
| 2 `zone_occupancy` events per clip; 36 events over 18 windows; 1 694/1 694 frames in a full pass | `m4-sustained-trials.json` `results[*].event_counts` and `events`; the headless pass output in the Task 4 report §8 |
| track ids 14 / 71–74 at 15 fps and 18 / 116–119 at 29.97 | `m4-sustained-trials.json` `results[*].events` |
| RSS deltas +33.0 to +41.8; +443.6 to +474.3; −8.0 MiB | `m4-sustained-trials.json` `results[*].rss_start`/`rss_end` |
| gate outputs, test count, wall times | §2.1–§2.4 of this document, run in this session |
| commits `8d4778e`, `c45ff13`, `2690564`, `09cbedd`, `3c55fe8`, `b8e314b` | `git log --oneline` on `main` |

Arithmetic that is arithmetic, and is labelled as such wherever it appears: the ~27× byte ratio from
M2's 240x320 feed to 1920x1080; "about 2 s of a 15 fps feed" from 22–31 frames; the 1.24–1.25× ratio from
the two throughput ranges; the 8 MiB tolerance as 5% of the 154.1 MiB pre-fix retention; and the 35–40
unconsumed frames in §7.6. **Nothing else in this report is derived rather than measured.**
