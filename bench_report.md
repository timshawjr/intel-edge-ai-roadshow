# bench report

Every table in this file was produced by a command run in this repository on the target
machine. Nothing is copied from a datasheet, a vendor page or a prior build. Where a figure
is **quoted from a gitignored recording** rather than re-measured, the row says so and names
the file; everything else was re-measured in the session that wrote it and the command is
given.

- `docs/qualification/M0-qualification.md` — the full M0 qualification report
- `docs/qualification/M0-content-gate.md` — rights, footage review, content gate
- `docs/qualification/qualification-matrix.csv` — scenario x model x device, with untested cells explicit

---

## M0 — hardware and model path qualification

Date: 2026-10-01
Milestone: M0 (design §11)
Verdict: **`Limited` — two warnings, and the gate is honestly incomplete on two of its four
clauses.** See the dated waiver immediately below. `tools/preflight.py` reads
**`Limited`**, exit 0. All three accelerators are real, each compiled the pinned IR and
completed 100 timed requests, and each passed the hand-annotated detection fixture 2/2 with
`EXECUTION_DEVICES` proving placement on the device that was asked for.

The two warnings, named: **in-process PDH GPU/NPU utilization is unavailable** (the value-read
entry points terminate the process), and **the LUID-to-device mapping is derived rather than
identified, with three hardware-identity records in disagreement and no SKU claimed**.

## Waiver on the M0 gate — dated, and naming the two clauses that are not met

**Recorded 2026-10-01, M0, by the M0 session. This waiver is a statement that the M0 gate
is honestly incomplete, not that it was met.**

Design §11's M0 gate has **four** clauses. **Two of them are unmet at M0:**

| # | clause | M0 status |
|---|---|---|
| 1 | the pinned model is verified by content and hash | **MET** — `verify_sources.py`, 6 PASS / 0 FAIL; the IR hash, input shape and OpenVINO build of the recorded run are each compared against this machine before its numbers are quoted |
| 2 | **observed device activity supports the assignment** | **NOT MET.** This is the utilisation attribution M0 could not establish. The counters read real percentages by the CIM/Get-Counter route, but the inference process's PID **never appears as a counter instance** and unrelated `pythonw.exe` processes hold the busiest `engtype_Compute` and `engtype_Neural` engines — measured: our PID 8428 held 0 of 469 samples while PIDs 15012 and 1844 held the busiest compute and neural engines. The earlier **62% / 68%** attribution is **withdrawn**. What *is* established is placement, and it rests only on `EXECUTION_DEVICES` re-read from each request's own compiled model after the first inference: `('CPU',)`, `('GPU.0',)`, `('NPU',)` on the device that was asked for, 100 timed requests each. |
| 3 | **CPU fallback works** | **NOT OBSERVED.** The `[fallback]` marker has never fired because no device has ever failed. The fourth row of the table — an explicit CPU-only re-measurement of the same model at the same shape — shows the fallback *target* is healthy, and that is **not** the same claim. No failure has been injected. Observed failure injection is M6. |
| 4 | the decoder produces correct detections on every device | **MET** — the hand-annotated fixture returns PASS 2/2 on CPU (re-run live by `preflight.py` in this run), GPU and NPU |

`AGENTS.md` requires every preflight row to be `PASS`. Two rows are `WARN`, so
**`tools/preflight.py` reads `Limited`, not `Ready`, and exit 0** — a deliberate decision
to ship a booth on a partially-open gate rather than to delete a row.

**Why `WARN` is the correct level for both, and `FAIL` would be wrong.**

- The PDH row is `WARN` because **a route failure is not a counter failure.** `pdh.dll`
  loads, the counter opens, and the wildcard expansion enumerates this machine's live
  `GPU Engine` instances; what fails is the in-process *value read*, with a measured reason
  (`0xC0000005` even with handles proven valid by `PdhGetCounterTimeBase`). A `FAIL` would
  tell a presenter the NPU is broken when it compiled the pinned IR and completed 100 timed
  requests in front of them.
- The LUID row is `WARN` because the mapping is **derived from the engine taxonomy rather
  than driver-exposed**, and preflight says so in those words — it never says "identified".
  Adapter `0x00011232` (81 instances, `3D` only) is left **UNATTRIBUTED** rather than
  swept onto the nearest device, which a prefix or range rule would have done.

**What this waiver does not do.** It does not upgrade the verdict to "gate passed". The
verdict below is `Limited` for that reason. The card the presenter sees still reads
**"Utilization unavailable"**, never `0%`, and **no utilisation figure may be shown until
M4 solves attribution** — the waiver is about the *gate's* level, not about the number
becoming available.

### Assets, and what a fresh clone does not have

`assets/models/` and `assets/media/` are **gitignored**. `tools/provision_venv.py` (the
only step needing network) copies the content-verified artifacts from the local cache into
them.

Until that has run, the tree is not merely unqualified — it is **not runnable as
qualified**: measured with `assets/` removed, `unittest discover` gives
`FAILED (errors=14, skipped=1)`, exit 1, and `preflight.py` reports
**`Needs attention`**, exit 1. **The `Ran 449 tests … OK` result pasted in §7 below is
therefore not reproducible on a fresh clone**, and the two-step reproduction
(provision, *then* gate) is named rather than collapsed into "one command".

### 1. Source verification — content, never HTTP status

`tools/verify_sources.py` — **6 PASS, 0 FAIL**

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

### 2. Preflight — the one-word gate

`tools/preflight.py` — **16 PASS, 2 WARN, 0 FAIL — 18 total rows. Status: `Limited`. Exit 0.**

The full row-by-row output is in `docs/qualification/M0-qualification.md`. The rows that
carry M0's headline numbers, verbatim from that run:

```
PASS  Recorded qualification run
      4 rows read from logs\qualification\m0.json, written by
      tools/qualify_m0.py, last modified 0.0 days ago. The IR hash and the
      OpenVINO build in that file were checked against config/models.json and
      against this process before any of its numbers were read.

PASS  Recorded per-device compile, latency and placement
      RECORDED VALUES, quoted from logs\qualification\m0.json and not re-
      measured by this run. Rendered by tools/qualify_m0.render_table so this
      table and the M0 report's table come from the same code. Row totals
      below are the recorded ones.

       device     full name                                 compile s  warmup s
      requests    p50 ms    p95 ms    min ms    max ms      rps
      EXECUTION_DEVICES    rss                        note
      ------------------------------------------------------------------------
      ------------------------------------------------------------------------
      -----------------------------------------
      CPU        Intel(R) Core(TM) Ultra 9 288V               0.21 s    0.03 s
      100  24.07 ms  27.35 ms  21.34 ms  29.19 ms    40.96  CPU
      332.9 MiB (349052928 B)
      GPU        Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)      0.67 s    0.01 s
      100   3.76 ms   5.94 ms   2.97 ms   9.05 ms   240.06  GPU.0
      409.4 MiB (429338624 B)
      NPU        Intel(R) AI Boost                            1.69 s    0.01 s
      100   5.27 ms   8.54 ms   4.09 ms  17.51 ms   168.83  NPU
      480.6 MiB (503975936 B)
      CPU        Intel(R) Core(TM) Ultra 9 288V               0.16 s    0.03 s
      100  24.51 ms  27.52 ms  21.35 ms  38.18 ms    40.37  CPU
      501.5 MiB (525873152 B)
```

The last row is the **CPU-only fallback baseline** — a genuine second measurement, not a
copy: p50 **24.51 ms** against the CPU row's **24.07 ms**.

```
PASS  Sanity fixture on CPU
      a real compile, a real inference and a real decode on CPU at conf=0.25
      iou=0.45 match_iou=0.5, compared against the hand-drawn annotation: 2 of
      2 matched. EXECUTION_DEVICES=['CPU'] placed_on_requested=True. VERDICT
      PASS: every annotated object matched, the detection count is within the
      annotation's bound, and the run was placed on the requested device.
      KNOWN BLIND SPOT, not checked here: a transposed w/h. See the module
      docstring - it is not separable on this fixture by any measurement.

PASS  Recorded sanity fixture, per device
      RECORDED VALUES from logs\qualification\m0_sanity_fixture.json, written
      by tools/build_sanity_fixture.py, not re-measured by this run: CPU PASS
      2/2; GPU PASS 2/2; NPU PASS 2/2

WARN  Telemetry: in-process PDH GPU/NPU utilisation
      Utilization unavailable (spec section 7's wording, not 0%).

WARN  Telemetry: LUID-to-device attribution and hardware identity
      - the LUID-to-device mapping is DERIVED from the engine taxonomy, not a
      driver-exposed identity: no CIM or Win32 source on this machine joins a
      LUID to a PCI function. Preflight says 'derived' and never 'identified'.
      - adapter LUID 0x00011232 is recorded UNATTRIBUTED (81 instances, engine
      types ['3D']): exposes only ['3D'], which identifies neither the NPU nor
      the GPU; left unattributed rather than assigned to the nearest device.
      - three hardware-identity records disagree and none is treated as the
      winner: Win32_ComputerSystem 'ASUSTeK COMPUTER INC. NUC14LNK LN';
      hostname 'NUC14PROAI'; PCI subsystem ['SUBSYS_88F21043'].

PASS  Device discovery cost
      DeviceRegistry.describe() took 3.76 s in this run for all 3 devices,
      measured around the first call in this process. The cost is one memoised
      Windows PnP enumeration (powershell.exe), not one per device; it is
      startup-only and must never reach a per-frame or per-sample path. Spec
      section 1 sets a 30 s warm-start target, so 3.76 s is inside it.

PASS  Startup: this gate's import to the first inference
      4.13 s, measured with time.perf_counter in this run from this module's
      import to the first inference completing.

device controls a presenter will be offered:
  CPU  enabled  Intel(R) Core(TM) Ultra 9 288V (driver no separate driver artefact)
  GPU  enabled  Intel(R) Arc(TM) 140V GPU (16GB) (iGPU) (driver 32.0.101.6737)
  NPU  enabled  Intel(R) AI Boost (driver 32.0.100.5540)

Summary: 16 PASS, 2 WARN, 0 FAIL - 18 total rows
Status: Limited
Exit code: 0. Ready and Limited both exit 0, because in both the demo runs; only Needs attention exits 1.
```

**What the qualification table does and does not say.** `compile s` and `warmup s` are timed
separately from inference and are **never added to it**. The percentiles are nearest-rank
over the timed loop only; the single warmup inference is outside it. `rps` is
`requests / timed span` only. **Every row completed 100 timed requests.**
`EXECUTION_DEVICES` is re-read from each request's **own** compiled model after the warmup
inference — it is never the device that was asked for, and the three plugins return three
different shapes (`('CPU',)`, `('GPU.0',)`, and the bare string `'NPU'`).
**`rss` is the HOST process resident set, not accelerator memory**: no property this
milestone reads exposes the NPU's own buffer, and the figure rises monotonically down the
table only because the rows accumulate in one process.

`gflops` is **deliberately absent** — not read from the model card, not estimated. TOPS,
power, temperature and energy are out of scope and are **never inferred from utilization**.

### 3. Device discovery

```
ov.Core().available_devices() -> ['CPU', 'GPU', 'NPU']
  CPU  FULL_DEVICE_NAME = 'Intel(R) Core(TM) Ultra 9 288V'
  GPU  FULL_DEVICE_NAME = 'Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)'
  NPU  FULL_DEVICE_NAME = 'Intel(R) AI Boost'
```

Each was requested by **explicit `compile_model(device)`** — no `AUTO`, no `HETERO`, no
`MULTI`.

### 4. Model pin

`yolo11n-fp16`, IR sha256 `4845b46cc9b6ac379a44484258b7ed98f836ed117d91f127c7728ea068250a95`,
input `[1, 3, 640, 640]` **static, batch 1**, output `[1, 84, 8400]` on an **unnamed**
port, layout `yolo_v8_transposed`, boxes `cx, cy, w, h`, and
**`has_embedded_nms: false` — derived, not assumed** (all 734 ops walked, no
`NonMaxSuppression` of any version; the trailing extent 8400 is the raw anchor count
`80²+40²+20²`, not a post-NMS count). **No deviation from the "no embedded NMS"
preference.**

### 5. Letterbox pad — the `114` decision, and a measured A/B

The pad band is **114**, not 0. For the fixture frame (1920x1080 → 640x640): scale
0.333333…, scaled 640x360, pad offset `(0, 140)`, band **280 rows of 640 = 43.75% of the
model canvas**, 140 top and 140 bottom.

**⚠️ CANNOT VERIFY** the conversion's own value: this is a pre-built IR downloaded from
HuggingFace and **its conversion script is not in this repository**. 114 is the documented
Ultralytics default **plus** the measurement below — not a read of the conversion. **No claim
that 114 improves detections;** the claim is that it **removes a train/serve distribution
shift**.

Measured, CPU, conf 0.25 / iou 0.45 / match_iou 0.5, using the repository's own
`preprocess_bgr` / `decode_detections` and the committed annotation:

| pad value | detections | matched | verdict | the annotated apple |
|---|---|---|---|---|
| **114** (in effect) | 4 | **2/2** | PASS | `apple` **0.273557** — 9.4% above the annotated `min_score` 0.25 |
| 0 | 3 | **1/2** | **FAIL** | **absent from the frame entirely** |
| 255 | 3 | 2/2 | PASS | `apple` 0.282365 |

Full decode at 114: `person` 0.913189, `person` 0.692380, `apple` 0.273557, `oven` 0.250770.
At 0: `person` 0.910445, `person` 0.622733, `oven` 0.255576 — the apple is gone.

Six frames, class set and coordinates at 114 vs 0: coordinates shift on **all six**; the
class set differs on frames 0, 60 and 300 and is identical on 1200, 1800 and 2400.

**A committed record this contradicted — now corrected.** `tests/fixtures/retail_sanity/retail_sanity.json`
and the `tools/build_sanity_fixture.py` docstring both said the pad mutants "still yield
2/2 here". That is **true for 255 and false for 0** — pad 0 fails the fixture 1/2. **The
correction was applied in commit `71cc58e`**, in both named files. An earlier revision of
this report said it was *"recorded, not applied — outside write scope"*, which described
the pre-`71cc58e` tree and no longer does. The A/B detection-set delta tool that would
settle the remaining `255` mutant is **M1 work**.

### 6. Telemetry counters — the honest result

**In-process: unavailable.** `pdh.dll` loads and the counter opens, but
`PdhGetFormattedCounterValue` and `PdhGetRawCounterValue` **terminate the process with
`0xC0000005`** even with handles proven valid by `PdhGetCounterTimeBase`; the array read
returns 465 "valid-status" entries of 4328 with `p50 = 10`, `max = 2.7e16` (garbage), and
`PdhExpandWildCardPathW` returns `0xC0000BBD` with 0 names, so values cannot be zipped to
instance names. **Route failure, not counter failure** → the card reads
**"Utilization unavailable", never 0%**, and the validated provider belongs to **M4**.

**The 62% / 68% figure: withdrawn.** Task 3 recorded **NPU `engtype_Neural` peaked 62%**
and **GPU `engtype_Compute` peaked 68%** under real load (Get-Counter cross-check 61.81 /
51.54), and **preflight's telemetry row used to cite** the task's report file under
`.superpowers/sdd/` as its evidence — a **gitignored** path that does not exist on a fresh
clone. **That citation is gone.** The row now states only that the counters read real
percentages by the CIM/Get-Counter route, that a machine-wide peak there is **not
attributable to this process**, and that **no utilisation figure may be shown until
attribution is solved**. It is *weaker* about the figure than it was, which is the point.

Re-measuring here, with an **idle baseline
first**, real simultaneous NPU+GPU load (3643 inferences each in 40.01 s, `EXECUTION_DEVICES`
`NPU` and `['GPU.0']`), and per-instance sampling:

| window | engtype_Neural peak | engtype_Compute peak | engtype_3D peak |
|---|---|---|---|
| **idle**, 24 queries, 469 instances/query, none of our load | 22 | **89** | 0 |
| **under load**, 50 queries, 483 instances/query | 31 | 67 | 0 |

`Get-Counter '\GPU Engine(*)\Utilization Percentage'` on the loaded machine: Neural 19.74
(11 instances), Compute 45.60 (29), 3D 0.39 (110).

**The load's own PID never appears as a counter instance** (`own_pid_instances_seen = 0`
across 50 queries). A per-instance peak sweep attributed the busiest engines to **other
processes** — two unrelated `pythonw.exe` (PIDs 1844, 15012) and a departed PID 7788 — with
Compute peaks of 47/27/25% and Neural peaks of 31/12/11%.

**So a machine-wide `engtype_*` peak on this machine is not evidence of this application's
accelerator utilization, and the 62% / 68% attribution is not supported.** The card's
reading stays "Utilization unavailable", and M4 must solve attribution before any
utilization figure may appear.

**Method note.** `Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine` exposes exactly
two properties on this build, `Name` and `UtilizationPercentage`. There is **no `Engtype`
property** — the engine type lives inside the `Name` string
(`..._luid_0x000112A0_phys_0_eng_0_engtype_Neural`). Filtering on `$.Engtype` matches nothing
and reads as "no activity" rather than "no such property".

**LUID attribution, derived not identified.** `0x00010E9E → GPU` and `0x000112A0 → NPU` are
resolved from the **engine taxonomy**; `Win32_LUID` has zero instances and
`MSFT_ComputeAccelerator` is not a registered CIM class on build 26200. Adapter
**`0x00011232` is UNATTRIBUTED** (81 instances, `3D` only) and is **numerically adjacent to
the NPU LUID `0x000112A0`** — a prefix, range or "not the GPU's, so it must be the NPU's"
rule would all have swept it onto the NPU card. It is left unattributed because that is what
the evidence supports.

### 7. The rest of the gate set

```
$ .venv\Scripts\python.exe -m unittest discover -s tests
----------------------------------------------------------------------
Ran 449 tests in 11.134s

OK
```

```
$ .venv\Scripts\python.exe -m app.main --selftest
```

**not run — `app.main` arrives in M1.** There is no application at M0: `app/` holds
contracts, `inference/` and `telemetry/` only, with no `main.py`, no QML and no workers.
There is no demo to run, and a tool run is not substituted for one.

### 8. Footage, measured

`cv2.VideoCapture` on the copied files, re-measured 2026-10-01:

| id | resolution | fps | frames | duration | in the spec's 45–90 s band? |
|---|---|---|---|---|---|
| `retail_checkout` | 1920x1080 | 29.97002997002997 | 3387 | **113.0129 s** | no — 23.0 s over |
| `store_aisle_alt` | 720x404 | 59.94005994005994 | 3921 | **65.41535 s** | **yes** |
| `queue_alt` | 768x432 | 10.0 | 1394 | **139.4 s** | no — 49.4 s over |

**1 of 12 required clips is in band; 9 clips do not exist.** The three remaining scenario
packs have no media at all. **Unmet M5 content gate.** No duration is rounded, padded or
trimmed to fit the band.

### 9. Content gate verdict, from looking at the footage

Contact sheets were built for all three clips — 12 evenly spaced frames each at native tile
resolution, every tile stamped with its frame index and timestamp — plus a denser second
pass over the regions where an event would be visible. **72 frames were looked at.**

- `retail_checkout` is a fixed high-angle camera on a **self-checkout / till**: counter,
  scanner plate, and a black basket **already full at frame 0**. A shopper repeatedly lifts
  items **out of the basket and onto the scanner plate** (frames 1500–1560, 1740–1800,
  2000, 2200, 2400, 2600). **No shelf appears anywhere in this clip.**
- `store_aisle_alt` is the only clip with **shelving in it** — a housewares aisle, shoppers
  reaching toward the shelves. **No basket, cart or trolley is visible in any frame looked
  at**, at 720x404 the shelf stock is a field of indistinguishable white blobs, and no item
  is seen leaving a shelf.
- `queue_alt` is an **office or lobby** with a round standing table, one person at a time.
  Not retail material.

**There is no shelf-to-basket clip in the provisioned material, and no shelf-to-basket
event in any clip.** The requested scenario is left **incomplete rather than claimed**, and
M0 commits to the documented reduced scope — **"items entering the checkout zone"** — as an
explicitly labelled intermediate milestone, with the shelf half of the scenario
acknowledged as not delivered on every screen, presenter line and export. Full reasoning in
`docs/qualification/M0-content-gate.md`.

### 10. Unmet gates and open blockers

| # | item | status |
|---|---|---|
| 1 | In-process PDH GPU/NPU utilization | **WARN, unresolved** — M4 owns the validated provider |
| 2 | Utilization **attribution** | **not established** — other processes are the top engine consumers; the inference PID never appears as a counter instance |
| 3 | Three 45–90 s clips per pack | **unmet M5 gate** — 1 of 12 in band, 9 clips absent |
| 4 | Shelf-to-basket footage | **unmet** — reduced scope committed, request left incomplete |
| 5 | Clip redistribution clearance | **not established** — no clip upgraded to "cleared"; two OMZ clips have no licence text in the record |
| 6 | Detector licence | **AGPL-3.0, unresolved** — M6 distribution question, operator/legal decision |
| 7 | Code signing | **unmet M6 gate** — M6 ships unsigned |
| 8 | 1920x1080 DPI gate | **not run — M3**; usable height here is 960 px, so a 1080p window cannot be shown |
| 9 | Fallback under observed failure | **not run** — the `[fallback]` marker has never fired; unit-tested only |
| 10 | AC status | **unavailable, permanently** — `Win32_Battery` returns 0 instances; Balanced power scheme recorded instead |
| 11 | Hardware SKU | **unreconciled** — three disagreeing records, none named the winner |
| 12 | The pad blind-spot record | **corrected in `71cc58e`** — it was wrong about `pad_value=0` (fails 1/2, not 2/2); `retail_sanity.json` and the `build_sanity_fixture.py` docstring now say so. The remaining `255` hole is M1 |
| 13 | `w`/`h` transposition | **open blind spot** — provably unseparable on this fixture; needs a different frame, not a tighter threshold |
| 14 | The `oven` false positive at 0.250770 | **unquantified** — deliberately unannotated, no precision/recall claim made |

**Reproduction.** All of §1–§3 and §7 come from `verify_sources.py`, `preflight.py` and
`unittest discover`.

**§9 is now committed and re-runnable.** The footage review was the one part of M0 whose
harness was not in this repository, and this file used to claim "the full scripts are
reproduced in the Task 9 report" — a gitignored file, i.e. not reproduced here at all. The
contact sheets are now built by **`tools/build_contact_sheets.py`**, which writes the five
sheets to `logs/contact_sheets/` and prints the frame indices it decoded. §9's frame lists
are its `CLIP_PASSES` table, and `tests/test_contact_sheets.py` fails if the two drift
apart.

**§5, §6 and §8 remain throwaway harnesses.** The pad A/B, the utilisation sweep and the
`w`/`h` aspect check were produced by ad-hoc scripts that used this repository's own
`app.inference.preprocess`, `app.inference.decode` and `tools.build_sanity_fixture`, wrote
nothing into the tree, and are **not reproducible from committed source** — the Task 9
report that held them is gitignored and was never a place a reader could look. Every figure
those harnesses produced is pasted in full above, so the tables stand as records rather than
as re-runnable claims. This is stated here rather than left for a reader to discover.

**Nothing above is reproducible from a clean checkout without provisioning first.** See
*Assets, and what a fresh clone does not have* below.
