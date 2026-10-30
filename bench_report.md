# bench report

Every table in this file was produced by a command run in this repository on the target
machine. Nothing is copied from a datasheet, a vendor page or a prior build. Where a figure
is **quoted from a gitignored recording** rather than re-measured, the row says so and names
the file; everything else was re-measured in the session that wrote it and the command is
given.

- `docs/qualification/M0-qualification.md` — the full M0 qualification report
- `docs/qualification/M0-content-gate.md` — rights, footage review, content gate
- `docs/qualification/M3-readability-review.md` — the 2–3 m readability review, the four committed captures, and the DPI gate verdict
- `docs/qualification/m3-capture/` — the four booth-screen captures (PNG + the JSON of every string read out of the scene)
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

---

## M1 — one truthful retail stream

Date: 2026-10-02
Milestone: M1 (design §11)
Full report: **`docs/qualification/M1-qualification.md`**. Everything below is a summary of
that document; the document is where the conditions live.

Verdict: **the application runs, on CPU, GPU and NPU, with placement proved by
`EXECUTION_DEVICES` and never by the request.** The gate set is green. **M1's own gate is not
fully closed** — four of its nine exit criteria are demonstrated only in part, and the report
says which and why.

The two things a reader might assume and must not:

1. **GPU and NPU utilisation are still unavailable.** The M0 finding is unchanged: the in-process
   value reads terminate the process with `0xC0000005`, and a machine-wide peak is not
   attributable to this process. **No utilisation figure appears anywhere in the product.** The
   cards read "Utilization unavailable". M4 owns the fix.
2. **The retail event is people-in-zone presence, not a product transfer.** Shelf-to-basket is
   **not delivered**, and the item-transfer reading of "checkout zone" was measured and found
   unsupported.

### M1.1 The gate set, run in the M1 session

```
$ .venv\Scripts\python.exe tools\verify_sources.py
Source verification: 6 PASS, 0 FAIL

$ .venv\Scripts\python.exe tools\preflight.py
Summary: 16 PASS, 2 WARN, 0 FAIL - 18 total rows
Status: Limited
Exit code: 0.

$ .venv\Scripts\python.exe -m app.main --selftest
  ok   preprocess           letterbox 640x640 pad=114 scaled=640x360
  ok   infer                25.38 ms on confirmed ['CPU']
  ok   decode-track-event   1 zone_occupancy event(s) from a synthetic presence
  ok   render-metadata      unavailable renders as unavailable, and no device is claimed before an inference confirms one
PASS

$ .venv\Scripts\python.exe -m unittest discover -s tests
Ran 1034 tests in 145.586s

OK
```

**16 PASS, 2 WARN, 0 FAIL is the same 18 rows as M0.** M1 added no preflight row and removed
none, because the honest state of the machine did not change. Both `WARN`s are the pre-existing
M0 telemetry rows — utilisation, and LUID attribution plus hardware identity — and **neither is
closed**. The suite's stderr carries expected noise: the supervisor's `compile_failure=True` seam
raises for real inside a child process, `argparse` prints a usage block when handed
`--device TPU`, and `verify_sources.py` prints `FAIL` rows inside a test that asserts a bad
artifact is refused. The result is `OK`.

### M1.2 The application runs — live scene graph, offscreen, CPU

`app.main.run_windowed` driven for real, 14 s, one 1920x1080 stream at 15 fps, offscreen, and
the objects it built read out of the live QML scene graph:

| | |
|---|---|
| exit code | 0 |
| `device_confirmed` / rendered badge | **`CPU`** / **`device: CPU`** |
| `fell_back` | `false` |
| frame image | **not null, 1920 x 1080**, real pixels from the pinned IR |
| zones from the manifest | **1** — `counter_band`, `(0.0, 0.20, 1.00, 0.60)` → 1920 x 432 px |
| overlay boxes | **3** (person, apple, person) |
| submitted / analysed / refused | 210 / **209** / **0** |
| `event_counts` | **`{"zone_occupancy": 1}`**, equal to the event machine's own |
| rendered event row | `zone_occupancy 1    candidate_transfer 0    candidate_return 0` |
| system CPU | **53.7%** (a second run of the same driver: 56.4%) |
| `ai_fps` | **14.50 fps** |
| frame age p50 / p95 | **58.7 ms** / **712.5 ms** |
| GPU / NPU cards | **`unavailable`**, with reasons |
| `cpu_available` | **`true`**, its notice reads `no CPU sample yet` |
| GPU / NPU notices | **`Utilization unavailable`** |
| window line / staleness line | `measured over 16 ms` (visible) / `telemetry fresh` |
| supervisor errors | **none** |

**The CPU card must not borrow the accelerators' wording and does not**: there is no counter to
fail for a CPU — `psutil.cpu_percent` runs every 250 ms and works here — so its absent state is
the weaker, different claim "no CPU sample yet".

### M1.3 Routing evidence — requested versus confirmed

Placement is `EXECUTION_DEVICES` read on the **compiled model after the first inference**,
compared through `is_device`, which handles the three inconsistent plugin shapes.

**Committed headless summaries** (`docs/qualification/m1-headless/`, 8 s feed, 2 s drain,
offscreen):

| file | requested | **confirmed** | state | fell_back | analysed | refused | drops | errors |
|---|---|---|---|---|---|---|---|---|
| `cpu.json` | CPU | **`CPU`** | active | `false` | 121 | 0 | 0 | `[]` |
| `gpu.json` | GPU | **`GPU.0`** | active | `false` | 121 | 0 | 0 | `[]` |
| `npu.json` | NPU | **`NPU`** | active | `false` | 121 | 0 | 0 | `[]` |

**Re-measured for the M1 report** — single frame, 640x640, batch 1, warm-up outside the timed
span, 30 timed inferences x 4 repeats:

| device | compile | confirmed | `is_device` | p50 per repeat (ms) |
|---|---|---|---|---|
| CPU | 0.19 s | `['CPU']` | `True` | 22.53, 22.64, 22.54, 22.37 |
| GPU | 0.73 s | `['GPU.0']` | `True` | 3.75, 3.68, 3.43, 3.54 |
| NPU | 1.62 s | `NPU` (bare string) | `True` | 5.78, 4.56, 4.48, 4.33 |

**The NPU is slower than the GPU on this machine.** Measured, not a defect, and it inverts the
shape a booth story usually has — the product's own explanation text says so.

**The conditions travel with the claim everywhere it appears**: single frame, 640x640, batch 1,
30 inferences, an **integrated** Arc 140V sharing system memory with the NPU. **This is not a
throughput claim**; paired throughput trials are M4. These figures are close to but not identical
with the Task 10 figures recorded in `app/engine/worker.py` (CPU 20.99–21.16 ms, GPU 3.30–3.47 ms,
NPU 5.92–5.99 ms) — both were measured by the same method on this machine, and the machine's
state differs between runs. The **ordering** held in both.

### M1.4 Frame age is not inference cost

| quantity | what it measures | CPU | GPU | NPU |
|---|---|---|---|---|
| **inference cost** p50, single frame, batch 1 | submit → complete inside the worker | 22.37–22.64 ms | 3.43–3.75 ms | 4.33–5.78 ms |
| **frame age** p50, in the 8 s feed | ingress → result, **including queue wait** behind a 15 fps feed | 58.06 ms | 45.87 ms | 49.58 ms |
| **frame age** p95, same feed | as above | 640.25 ms | 1001.20 ms | **2015.53 ms** |
| `ai_fps` | analysed / **feed window** | 15.11 | 15.11 | 15.10 |

The NPU's p95 frame age is **~2.0 s** while its inference cost is **~4–6 ms**. The gap is queue
wait behind the feed, not compute, and quoting either number alone would misdescribe the system.
All three devices sit at the 15 fps target because the feed is the limit — which is exactly why
these are two tables. `ai_fps` divides by `feed_elapsed_s` (**121 / 8.006 = 15.11**), not by
`elapsed_s` (10.139 = feed 8.006 + drain 2.013); `drain_elapsed_s` is its own field.

### M1.5 The business event actually emitted

**People-in-zone presence. Not a product transfer, not a people count, not a dwell measurement.**

Re-measured 2026-10-02 by running the shipped configuration through the real pipeline over the
real clip at stride 5 (678 of 3387 frames), CPU — 1960 detections, 0 abandoned:

```
events 2
  zone_occupancy person track=1  loop=0 frame=66  t=2.202  dedup=0:1:person:presence:counter_band
  zone_occupancy person track=41 loop=0 frame=551 t=18.385  dedup=0:41:person:presence:counter_band
```

**Two sustained presences, one per person-track.** `BusinessEvent` carries no duration field, so
a dwell figure could not be honest even if the screen wanted one. In the 8 s product-path runs
the count is 1 — the same event over a shorter window, on all three devices.

**Why it is not more.** No product-class zone transition fires on `retail_checkout` at the
pinned operating point — the full-clip run under M0's rectangles produced **0 events**. The only
reliably tracked product, the apple, is **stationary for the whole clip, max corner deviation
0.0006 over 483 frames**. And the person does not traverse: **17.441** frame-widths of cumulative
travel against **0.015** of net displacement. A traversal-shaped event would fire once per
113-second clip and read like a business event without being one. So the pack declares **no**
`origin_slot`/`destination_slot`, and `tests/test_retail_footage_guard.py` **fails the build if a
shipped configuration emits zero events**.

The screen shows `app/engine/retail_config.py`'s `scope_note` verbatim — "REDUCED SCOPE… Shelf-to-
basket is NOT delivered… COUNTER_BAND captures 65% of person detections, not all of them."

### M1.6 Input acknowledgement — a floor, not a bound

Through `app.main.run_windowed` itself, offscreen, 20 ms `QTimer` lateness on the Qt main
thread, **6 CPU-only saturator processes** (`max(2, cpu_count()-2)` = 6 of 8 logical cores) plus
real CPU inference, 3 repeats in one process:

| repeat | p50 | p95 | worst tick | ticks |
|---|---|---|---|---|
| 0 | 0.031 ms | 16.582 ms | 37.525 ms | 292 |
| 1 | 0.041 ms | 10.114 ms | 21.622 ms | 294 |
| 2 | 0.074 ms | 11.463 ms | 28.995 ms | 293 |

Passes a 200 ms p95 gate and a 2 s worst-tick gate, the worst tick 7x under the bound. **The
figure is weaker with each caveat:** it is measured offscreen on one stream with synthetic
saturators, so it is a **floor, not a bound**; it is **not comparable across machines** (2 of 2 on
a two-core box); and **at 8 of 8 saturated cores the same gate measured p95 195.6 ms and max
1105.5 ms** — a greater-than-one-second freeze that still reports green on a p95. **A booth
machine must be provisioned so inference never sees 8/8 saturation.**

### M1.7 Metrics: numerator, denominator, window, provider, scope

| metric | numerator | denominator | window | provider | scope |
|---|---|---|---|---|---|
| `system_cpu_percent` | CPU time across all 8 logical cores | 8 cores x window | recorded delta (0.062 s in `cpu.json`) | `psutil` | **machine-wide**, not this process |
| `app_cpu_percent` | sum over the tracked PIDs | 8 cores x window | recorded delta | `psutil` | `("ui","worker")` — the process tree |
| `ai_fps` | frames with a completed detection | **`feed_elapsed_s`** | the feed window | counted by the product | this run, this stream |
| `frame_age_p50/p95_ms` | ingress → result age | 121 samples, nearest rank, no interpolation | the run | counted by the product | this stream |
| `gpu_/npu_utilisation_percent` | — | — | — | **none that works in-process** | **`unavailable`**, with the reason |
| TOPS / power / temperature / energy | — | — | — | — | **out of scope, never inferred from utilisation** |
| `rss` figures | resident set | — | recorded delta | `psutil` | **host process only, not accelerator memory** |

`window_seconds` is a **measured delta**, not a designed interval, and is formatted from the
record — the bridge never divides. `app_cpu_percent` is a **genuine measured `0.0`** in `cpu.json`
and `gpu.json` and `0.8875` in `npu.json`: a real zero rendered as a number, which is the whole
distinction from `unavailable`.

### M1.8 The visible failure state — built and tested, never observed

| | |
|---|---|
| a confirmation that does not name the requested device renders as a **fallback** | demonstrated |
| a **failed** route claims no device (`no device is running`) | demonstrated |
| a dead worker restarts **exactly once**, then `failed` with a stated reason | demonstrated, against a fault the child process really raises |
| **an accelerator failing mid-run and the app falling back to CPU** | **NOT OBSERVED — M0 gate 9 stands** |

M1 built the visible failure state and the refusal accounting (`frames_refused` with reasons;
a run that analysed nothing exits non-zero rather than reporting a clean zero). **It did not
observe a failure**, because no device on this machine has ever failed. M1's CPU-only run is a
**measured CPU-only run**, which is a weaker claim than "fallback was observed to work".

### M1.9 M1 exit criteria — demonstrated and not demonstrated

| # | criterion | verdict |
|---|---|---|
| 1 | one clip end to end, real overlays, nothing precomputed | **demonstrated** (M1.2) |
| 2 | selected and actual devices match, by `EXECUTION_DEVICES` | **placement demonstrated; an interactive presenter-driven selection was NOT run — that is M2** |
| 3 | frame IDs prevent overlay drift, proven by a test | **demonstrated** (`fid 8` refused against a displayed `fid 7`; a result from another loop refused too) |
| 4 | event tests cover transfer / return / occlusion / duplicate / loop reset, and events are reported with `dedup_key`s | **PARTIAL.** The five tests exist and pass, and the `dedup_key`s are reported — but **no `candidate_transfer` is ever emitted on real footage**, because no product transition exists. **The transfer and return paths have never run on footage; they are covered against synthetic tracks only.** |
| 5 | the same clip works on each target validated in M0 | **demonstrated as placement and throughput** — 8 s offscreen feeds, **not** the 60 s run the plan's example names |
| 6 | a runtime failure produces a visible error/fallback state and the UI never claims the failed target | **state demonstrated; failure never observed** (M1.8) |
| 7 | "Recorded footage • live inference" displayed, pinned on the QML source | **demonstrated as text and as rendered glyphs; no committed screenshot exists** |
| 8 | unavailable renders as `unavailable`, a genuine `0` renders as a number | **demonstrated** |
| 9 | `verify_sources`, `preflight`, `--selftest` and the full suite green, output pasted | **demonstrated** — and "green" includes two named, still-open `WARN`s |

**M1's gate is not fully closed, and the four qualifications above are why.**

**No committed screenshot of the booth screen exists and this report points at none.** The
offscreen render harness writes a PNG into a temporary directory it deletes; `tools/capture_ui.py`
does not exist (M3). **The live booth runs at this machine's 2293x960 desktop, not a 1920x1080
canvas** — M0 measured usable height at 960 px, so a 1080p window cannot be displayed here at all,
and a passing offscreen 1920x1080 render is not a claim about this monitor.

### M1.10 Unmet gates and open blockers after M1

| # | item | status |
|---|---|---|
| 1 | In-process PDH GPU/NPU utilisation | **open, unchanged** — M4 |
| 2 | Utilisation **attribution** | **open, unchanged** — the inference PID never appears as a counter instance; unrelated processes hold the busiest engines |
| 3 | Three 45–90 s clips per pack | **unmet M5 gate** — unchanged |
| 4 | Shelf-to-basket footage | **not delivered** — and the item-transfer reading of "checkout zone" is now **measured and unsupported**. Delivered: people-in-zone presence |
| 5 | Clip redistribution clearance | **not established** — unchanged |
| 6 | Detector licence | **AGPL-3.0, unresolved** — M6 distribution question |
| 7 | Code signing | **unmet M6 gate** — M6 ships unsigned |
| 8 | 1920x1080 DPI gate | **not run — M3**; live booth is 2293x960 |
| 9 | Fallback under observed failure | **still not run.** M1 built and tested the visible state; nothing was ever observed to fail |
| 10 | AC status | **unavailable, permanently** — unchanged |
| 11 | Hardware SKU | **unreconciled** — unchanged |
| 12 | A 60-second per-device unattended run | **not run.** What exists is two 14 s windowed runs and three 8 s headless feeds. M6 soak |
| 13 | Second annotated frame with a non-square object | **open.** The `w`/`h` transposition blind spot stands — a swap preserves `cx`, `cy` and `w*h` exactly, and on the current annotation the aspect check is **inverted**. Needs a different frame, not a tighter threshold |
| 14 | Overlay-geometry test is vacuous on a square frame | **recorded, not fixed.** The live check compares the payload's boxes against an independent recomputation from `Track.box` and the frame extents; on a square frame the two divisors are equal. Latent, not active — the live frame is 1920x1080 |
| 15 | QML Slot-binding guard is line-oriented, not a QML parse | **deliberate, recorded.** Two-entry allow-list (`bridge.explanation`, `bridgeUi.explain`) that **must be maintained** |
| 16 | `unavailable_notice["CPU"]` can exceed its slot and elide | **recorded, not observed.** Single-line `Text` with `elide: Text.ElideRight`; not hit by today's string, but a later change to it must be checked |
| 17 | `oven` false positive at 0.250770 | **unquantified** — unchanged |
| 18 | Transfer / return event paths have never run on real footage | **open.** Test-covered against synthetic tracks; the footage cannot produce a transfer |

**Reproduction.** M1's four gate commands need `tools/provision_venv.py` to have run first —
`assets/` and `logs/` are gitignored. The headless summaries this section quotes are
**committed** at `docs/qualification/m1-headless/` and can be read without provisioning; only
re-measuring them needs it.

---

## M2 - three streams and device toggles

Full report, with every gate pasted: **[docs/qualification/M2-qualification.md](docs/qualification/M2-qualification.md)**.
Committed records: **`docs/qualification/m2-headless/`**, each with its command in that directory's
`README.md`. This section extends M0 and M1 and contradicts neither.

### M2.1 The gate set, run in the M2 session

| gate | result |
|---|---|
| `tools\verify_sources.py` | **6 PASS, 0 FAIL**, exit 0 - verified by content signature and sha256, never by HTTP status |
| `tools\preflight.py` | **16 PASS, 2 WARN, 0 FAIL** (18 rows), `Status: Limited`, exit 0 - **both WARNs are M0's two pre-existing telemetry rows** |
| `python -m app.main --selftest` | **PASS**, exit 0 |
| `python -m unittest discover -s tests` | **1210 tests** - run 1 `FAILED (failures=1)`, runs 2 and 3 `OK`; wall **359.1-359.9 s** |
| `tests.test_ui_contract.SlotBindingAudit` | **7 tests, OK** - and **M2 added no name to its two-entry allow-list** |
| `tests.test_route_stress.RouteStressInTheSuite` (in isolation) | **7 tests, OK**, 20.1 s |

`preflight` was **not run for the Tasks 1-2 review and no number was claimed for it then**; the table
above is this session's run, the first since. Suite wall time is **~360 s**, against **145.6 s** for
M1's committed 1034-test run and **323.8 s** measured by Task 5 in its own session (milestone working
log, not committed). The increase is the three-camera live-path test class, ~32 s of it.

### M2.2 Placement, re-measured on all three devices

Four full runs of `tools\qualify_m0.py --requests 30`, each in its own process, committed at
`single-frame-latency-run1..4.json`. `OVERALL: PASS - 4 of 4 rows succeeded on the device they
requested` on every one.

| device | `EXECUTION_DEVICES` | p50 ms, four runs | compile |
|---|---|---|---|
| CPU | `CPU` | 22.61 / 22.81 / 22.40 / 22.72 | 0.19 s |
| GPU | `GPU.0` | 3.45 / 3.54 / 3.42 / 3.40 | 0.62-0.63 s |
| NPU | `NPU` | 5.25 / 5.21 / 5.25 / 5.74 | 1.54-1.57 s |

**The NPU is slower than the GPU on this machine**, under the conditions named above. Not a throughput
claim; paired trials are M4. **A correction to the record:** the milestone working log quotes CPU
20.99-21.16 ms and NPU 5.92-5.99 ms. Those are *better* than anything measured in the M2 session, they
live only in a gitignored file, and they were taken earlier the same day on a measurably faster machine
state. **A per-device latency figure is a property of a session, not of a machine; quoting the best one
observed is not a measurement.**

Across 50 live transitions: `placements_confirmed` **50/50** and `placement_pids_matched` **50/50**.

### M2.3 The 2 s warm-transition target: **UNMET**

`total` is the whole toggle, `setup_s + transition_elapsed_s`. 50 toggles, 50/50 acknowledged, nothing
censored (`--timeout-s 30`, widened **on purpose** so the tail is measured rather than truncated).

| | min | p50 | p95 | max | over 2 s |
|---|---|---|---|---|---|
| before the teardown fix | 2.080 | 3.151 | 5.521 | 5.544 | **50/50** |
| **after** | **1.286** | **2.261** | **4.971** | **5.185** | **25/50** |

| | setup p50 | wait p50 | wait p95 | wait max |
|---|---|---|---|---|
| before | 1.642 s | 1.506 s | 3.369 s | 3.387 s |
| after | **0.730 s** | 1.510 s | 3.458 s | 3.501 s |

At the **real 2.0 s default** (`await_transition`'s own deadline, unchanged): acknowledged **23/50 →
26/50**; over 2 s **48/50 → 37/50**.

> **THE TARGET IS MISSED. The distribution is documented.** Both halves are the answer. The
> improvement is real, measured and committed, and it does not meet the 2 s target. **No timeout was
> touched to obtain any of these numbers.**
>
> The earlier published **1.04 / 1.67 / 1.31 s** figures were **half a toggle** - they excluded
> `begin_transition` - and they are wrong as a claim about this gate. They are in no committed file
> and are recorded here only so nobody reaches for them again.
>
> `stop()` teardown fell from 15.18 s to 0.87 s in the same fix: milestone working log, **not
> re-measured for this report**, quoted as a task record rather than as committed evidence.

### M2.4 Three streams on the CPU is an 11-15 second old picture that reads healthy

Unsaturated, **spec 5 row 4**, three 120 s holds, committed at `allcpu-steady-120s{,-hold2,-hold3}.json`.
15 fps offered per stream, `errors: []`, **drops 0**, all three streams `active`.

| hold | overall p50 | overall p95 | backlog per stream |
|---|---|---|---|
| 1 | 11 319 ms | 12 286 ms | 173 / 180 / 187 |
| 2 | 14 309 ms | 15 562 ms | 228 / 232 / 235 |
| 3 | 14 697 ms | 16 019 ms | 226 / 242 / 238 |

**The headline is the range: p50 10.7-14.7 s, p95 12.3-16.1 s, backlog 173-242 frames per stream**, at
~13.0-13.6 of 15 fps analysed. Two earlier holds in the gitignored log (10 767 and 11 085 ms) are lower
than all three committed holds; the picture gets older across repeated holds and **the cause is not
established**.

The screen says so, in words, per stream - read off the live QML scene graph from the committed
numbers:

| | Camera A | Camera B | Camera C |
|---|---|---|---|
| confirmed badge | `CPU` | `CPU` | `CPU` |
| age headline | `11.7 s` | `10.7 s` | `11.1 s` |
| staleness sentence | `STALE - this picture is 11.7 s old; the limit is 2 s` | `STALE - this picture is 10.7 s old; the limit is 2 s` | `STALE - this picture is 11.1 s old; the limit is 2 s` |
| window | `window: 256 analysed frames` | same | same |
| backlog / rate | `backlog 187 frames` / `analysis 13.44 fps` | `backlog 173 frames` / `analysis 13.56 fps` | `backlog 180 frames` / `analysis 13.50 fps` |

The 2 s threshold is spec section 7's and is printed on the card next to the age. **The card is never
dimmed for being stale** - dimming the card that warns would dim the warning. **Frame age and inference
cost are never merged**, and **unavailable is never `0`**: with no telemetry sample the CPU line reads
`no CPU sample yet`.

### M2.5 The input-ack gate cannot see any of that

Re-measured this session (`InputAcknowledgement`, 3 repeats, 6 CPU saturators + real CPU inference):
p50 0.015 / 0.024 / 0.000 ms, **p95 11.554 / 12.863 / 11.749 ms**, **worst tick 20.361 / 18.470 /
17.096 ms**, 299 ticks per repeat, `confirmed ['CPU']` every time. Against a 200 ms p95 gate and a 2 s
worst-tick gate, comfortably green - **and it is measuring the wrong quantity.**

> The gate times how long the **UI thread** waits between its own ticks. A frame sitting 173-242
> frames deep in a worker queue is invisible to it, because the UI thread is not waiting on the queue.
> **Ruling: the unsaturated `allcpu` frame-age row is the real latency figure for this application and
> leads this milestone. The input-ack gate is insensitive to inference backlog by construction, and a
> green gate on the wrong quantity is not a pass.** The two are never added, averaged or substituted.
> A combined experiment - the probe running against the three-camera `allcpu` load - was **not run**.

### M2.6 50 alternating warm route changes: no crash, no stale overlay, no mislabelled device

From `route-stress-50-after-teardown-fix.json`. The walk covers **all four rows** of spec 5's table in
its own order, one camera per toggle: `{A:CPU,B:GPU,C:NPU}` x12, `{A:CPU,B:CPU,C:NPU}` x13,
`{A:CPU,B:GPU,C:CPU}` x13, `{A:CPU,B:CPU,C:CPU}` x12.

`crashes 0` · `mislabelled_devices 0` · stale overlays **0** over `results_delivered 4179` ·
`discarded_results 942` · `error_results 0` · `errors_at_end []`.

**Leak, reported as total / design / remainder separately** - adding them is the mistake:

| | before the fix | after (30 s deadline) | after (real 2.0 s gate) |
|---|---|---|---|
| RSS start → end | 94.8 → 107.9 MiB | 95.7 → 111.1 MiB | 95.0 → **277.4 MiB** |
| **total growth** | 13.1 MiB (+268.6 KiB/toggle) | 15.5 MiB (+316.8 KiB/toggle) | **182.4 MiB** (+3.65 MiB/toggle) |
| retained transitions / queues | 1→51 / 6→156 | 1→51 / 6→156 | 1→51 / 6→204 |
| live objects, emptied + `gc` | 54 689 → **50 107** | 54 870 → **50 560** | 57 438 → **52 724** |
| **remainder after release + `gc`** | **3.53 MiB** | **5.76 MiB** | 175.3 MiB |

**Live objects go *down* across 50 transitions** in every run - 4 582 / 4 310 / 4 714. Nothing
accumulates per transition. The retained transition and queue records are counted as **design** growth
because they are deliberately retained.

**One arithmetic correction:** the working log calls the 3.5 MiB remainder "0.26% of RSS". It is
**3.3% of the 107.9 MiB end RSS**.

**The third column is a finding, and it is not characterised.** At the real 2.0 s gate - the run a
presenter experiences - RSS ends at 277.4 MiB with 175.3 MiB unexplained by retention, in the run with
24 unacknowledged transitions and a stated recovery per timeout. One hypothesis is that each recovery
path retains something a clean run never builds. **It is a hypothesis, it is not established, and it is
not measured.** A long-session leak trend is M6's soak.

**Stale painting: real but weak.** `discarded_results` is ~19 per toggle; before the teardown fix the
same field read **29 over 5 037 delivered results**, under 0.6 per toggle, because the queue was drained
*before* superseding so the discarded work was work always being thrown away. **The count went up
because the accounting improved, not because fifty more proofs appeared.** The identity-based refusal
tests are the real evidence.

### M2.7 Per-camera video surfaces - the gap M2 disclosed, closed

Task 5 disclosed *"per-camera video surfaces are not built"*. Closed, and measured on the **live path**,
not seeded: `app.main --windowed --streams 3`, three real workers on three devices, three decode
threads, 25 s.

```
returncode: 0
layout:      [startup] 3 camera(s): A -> CPU, B -> GPU, C -> NPU
  tile A: image://frames/A?v=269  1920x1080  no_frame_visible=False
  tile B: image://frames/B?v=339  1920x1080  no_frame_visible=False
  tile C: image://frames/C?v=340  1920x1080  no_frame_visible=False
hero tile stream: A
frames analysed/submitted: 1194 / 1281      ai_fps: 42.72
qml badge: device: CPU
```

Three tiles, three sources, three **independent** revisions. The provider has **no cross-stream
fallback**: a camera with no frame gets a 1x1 transparent image and its tile says `no frame from Camera
N yet`, naming itself - never another camera's pixels. The hero serves the bound stream and follows a
change of selection. **Each camera has its own `ClipSource`, decode thread, `Tracker` and
`RetailEventMachine`** - sharing a tracker interleaves track ids across cameras and lets one camera's
zone rule fire on another's picture. Mutation-checked: **20 mutants, 18 killed**, the two survivors
being deliberate no-op controls.

**Two defects this work produced, both found by rendering rather than by reading.** A `QVariantList`
concatenated with a string rendered the literal **`-1`** as a thumbnail caption - on a render whose
Python side held `confirmed: ('GPU.0',)` and whose picture above the caption was real; **nothing about a
screenshot of that caption says it is wrong.** And the provider's id parsing was wrong twice before it
was right (`args[-1]` is a `QSize`, not the id; the id is the whole `QUrl`, not its last segment), each
fault producing a correct-looking `source` and a **silently empty hero**.

### M2.8 The visible failure state - and the half that cannot close

M0 gate 9 has a **silent** half (`confirmed` differing from `requested`) and a **visible** one.

**Visible: delivered.** `--fail-stream A` on the live scene graph - 127 badge samples going from
`device: pending - CPU not yet confirmed` to **`device: failed - no device is running`**,
`device_confirmed` the **empty string**, `fell_back false` because nothing ran elsewhere, and the tile
saying `no frame from Camera A yet`, beside the worker's own stated reason. In the committed stress
record (`--fault-stream C` against the NPU): `confirmed_devices []`, `frames_analysed 0`, 253 accepted
and 896 refused all carrying the reason `not_running`, state never confirmed, **`restart_count` exactly
1**, `fell_back false`. **A failed route claiming no device is the correct rendering**, and the product
refuses to *construct* a failed route with no stated reason.

**Silent: OPEN, and it cannot be closed on this machine.** All three devices execute where asked, so
the substitution never occurs naturally, and `compile_failure` kills the worker *before it runs
anything* - a loud failure. **No fault was invented to produce a silent one**, because inventing one
and reporting that fallback was observed is the exact failure this project exists to prevent.

### M2.9 Presenter-driven selection - observed

`--select A=GPU` through `run_windowed`, this session:

```
[select] requested {'A': 'GPU'} -> acknowledged=True bounded_by='confirmed' generation=2 in 1.293 s;
          confirmed={'A': ['GPU.0']}; unconfirmed=[]
219 badge samples, three values in order:
  device: pending - CPU not yet confirmed
  device: pending - GPU not yet confirmed
  device: GPU.0
```

Sampled every 100 ms **on the UI thread**, beside what the supervisor had actually proved at that
instant. The badge may **lag** the supervisor by a queued hop, and did; it may never **lead** it. At the
end: `confirmed 'GPU.0'`, a real 1920x1080 frame, 4 overlay boxes, `event_counts {'zone_occupancy': 2}`.
**This closes M1's criterion 2**, which M1 recorded as demonstrated for placement but not as an
interactive act.

### M2.10 M2 exit criteria

| # | criterion | verdict |
|---|---|---|
| 1 | all four spec 5 toggle combinations execute | **demonstrated** (M2.6) - all four rows walked, 50/50 placements confirmed |
| 2 | completed jobs attributed to their **actual** devices, per stream | **demonstrated** (M2.2); the *presenter-driven* form measured separately (M2.9) |
| 3 | pending results cannot paint after a generation change | **demonstrated, WEAKLY** (M2.6) - ~19 discards per toggle, up from 0.6 because the accounting improved, not because there were more proofs |
| 4 | 50 alternating warm route changes: no crash / leak trend / stale overlay / mislabelled device | **demonstrated**, with the leak **qualified**: clean at a 30 s deadline (15.5 MiB total, 5.76 MiB remainder) and **not** clean at the real 2.0 s gate (182.4 MiB, 175.3 MiB unexplained) |
| 5 | warm transitions meet 2 s **or the miss is documented with its distribution** | **THE TARGET IS MISSED; THE DISTRIBUTION IS DOCUMENTED** (M2.3) - both halves |
| 6 | M1 criterion 2 (presenter-driven selection) and criterion 6 (observed fallback) | **both closed** - selection observed (M2.9); the **visible** half of the failure closed (M2.8) and the **silent** half scoped precisely as open |

### M2.11 Unmet gates and open blockers after M2

| # | item | status |
|---|---|---|
| 1 | In-process PDH GPU/NPU utilisation | **open, unchanged - M4.** Both `preflight` WARNs are these two telemetry rows |
| 2 | Utilisation **attribution** | **open, unchanged** - the inference PID never appears as a counter instance |
| 3 | Shelf-to-basket footage | **not delivered, and unmet by design.** The delivered event is people-in-zone presence, 2 per clip - **not a people count and not a dwell measurement** |
| 4 | Clip redistribution clearance | **not established** - unchanged |
| 5 | Detector licence | **AGPL-3.0, unresolved** - M6 |
| 6 | Code signing | **unmet M6 gate** - M6 ships unsigned |
| 7 | 1920x1080 DPI gate and `tools/capture_ui.py` | **closed by M3 (Tasks 1-2), see the M3 section below.** The live booth is 2293x960 and M0 measured usable height at 960 px, so the gate is measured **offscreen**; the four committed captures and their corrected device-space `glyph_pixels` are under `docs/qualification/m3-capture/` |
| 8 | AC status | **unavailable, permanently** - unchanged |
| 9 | Hardware SKU | **unreconciled** - unchanged |
| 10 | Second annotated frame with a non-square object | **open** - the `w`/`h` transposition blind spot stands and the current annotation's aspect check is **inverted** |
| 11 | Slot-binding guard's two-entry allow-list | **still line-oriented by design.** M2 added bindings and **added no name** |
| 12 | Count-bounded `frame_ages_ms` window | **now disclosed on the card**: `window: 256 analysed frames`, ~17 s at 15 fps and ~73 s at 28 frames/8 s |
| 13 | `tools/preflight.py` for the Tasks 1-2 review | **was not run then and no number was claimed**; first run since is this session's |
| 14 | **M0 gate 9's SILENT half** | **OPEN and cannot be closed on this machine** (M2.8) - M6 |
| 15 | **The 2 s warm-transition target** | **MISSED** (M2.3) - distribution documented |
| 16 | **Leak at the real 2.0 s gate** | **182.4 MiB total, 175.3 MiB unexplained, not characterised** (M2.6) - M6 |
| 17 | **Task 3's RSS-ratio flake** | `test_nothing_grows_that_the_retained_records_do_not_account_for` - failed once in this session's two full runs and once before it in nine full-suite runs; passes in isolation. It puts a ratio bound on a measurement it shares with five other tests. **Deliberately not fixed here** |
| 18 | A 60 s+ unattended soak | **not run.** What exists is a 25 s three-camera windowed run and the route-stress holds. M6 |
| 19 | `--fail-stream` | **a documented fault-injection flag on the shipped CLI** - a **qualification aid**, named as such in the presenter guide, so nobody ships it thinking it is a feature |
| 20 | Transfer / return event paths on real footage | **open** - test-covered against synthetic tracks; the footage cannot produce a transfer |
| 21 | `oven` false positive at 0.250770 | **unquantified** - unchanged |

### M2.12 Known costs and limits of M2

- **Suite ~360 s**, with a **pre-existing flake** (item 17 above) that failed once in this session's three
  full-suite runs and once before it in nine full-suite runs, and passes in isolation. **No test was
  lost**: against Task 5's tree the two touched test files gained 7 and 6 test methods and **zero were
  removed**.
- **A `Repeater` delegate is invisible to `QObject.findChild` and `QObject.children()`** (it is a
  *visual* child), and **a `Window`'s `childItems()` is empty** because the content item is *assigned*.
  Two readers built on those APIs reported `MISSING` for every item on a render whose PNG plainly
  showed all three cards. **That is a reader that would have been a vacuous pass** - the **fourth**
  instance of this shape in the project.
- **`Math.max` passes `NaN` through**: the metrics column came back completely empty with **no QML
  warning**, because at delegate creation the `Repeater` has not counted its own items.
- **Five live-path assertions were asserting a timing accident** (a starved collector thread yields no
  valid CPU sample in 14 s) and now assert the obligation either way.
- **A `Text` keeps its `text` while `visible: false`**, so reading a hidden "no frame" sentence
  unconditionally reports it on every tile.
- **`str(QSize(1920, 1080))` is `"PySide6.QtCore.QSize(1920, 1080)"`** on this PySide6, so an
  assertion written against a bare `1920x1080` compares a representation instead of a measurement.
- **A mutation sweep can report a survivor that was never aimed at the covering assertion.** It did
  once here. The remedy was a new assertion, not a softer claim.
- **This session's machine is measurably slower than earlier the same day** - CPU single-frame p50
  22.4-22.8 ms against 21.0-21.2 ms, and three-stream frame age 14.3-14.7 s against 10.7-11.3 s.
  **Not established why.** It is why this report quotes ranges and prefers the committed figure over
  the best one observed.

**Reproduction.** M2's gate commands need `tools/provision_venv.py` to have run first - `assets/` and
`logs/` are gitignored. Every record under `docs/qualification/m2-headless/` is **committed** and has
its command in that directory's `README.md`; nothing in this section rests on a file under `logs\`, or
on the gitignored milestone working directory, except where it is labelled as such.

---

## M3 — the booth screen at 1920×1080

Date: 2026-10-26 (Task 6, the closing task of the milestone)
Milestone: M3 (design §9, spec §8 / §10)

**Verdict: the layout gate passes at all four scale factors, and the screen is not readable at
2–3 m.** The four scales were really rendered and their pixel sizes measured from the images; the
readability review then computed the angular size of every element a passer-by must read and found
that **nothing on the screen clears 16 arcmin at 3 m** on the display the review assumes, and that
every 18 px line is below the 5 arcmin acuity floor at 3 m. The failures are recorded as failures in
[docs/qualification/M3-readability-review.md](docs/qualification/M3-readability-review.md) and are
M4's to fix. **One M3 exit criterion is unmet (6, the input-acknowledgement blindness
demonstration), and criterion 2 is met only in its "or the failure is documented" form —
because the numbers are not legible.**

`tools/preflight.py` is **16 PASS / 2 WARN / 0 FAIL (Limited)**, exit 0; `verify_sources` is 6 PASS /
0 FAIL; `--selftest` PASS; the suite is **1376 tests, OK, 522.5 s** (wall 523.5 s).

### M3.1 The gate set, run in the M3 Task 6 session

All on branch `main`, on a machine asserted clean before the timing runs (`Get-Process python` → 0;
the tool's own census reports `"clean": true`).

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

Source verification: 6 PASS, 0 FAIL                              exit 0
```

```
$ .venv\Scripts\python.exe tools\preflight.py
... 16 PASS rows elided here for width; the only two non-PASS rows are ...
WARN  Telemetry: in-process PDH GPU/NPU utilisation
WARN  Telemetry: LUID-to-device attribution and hardware identity
Summary: 16 PASS, 2 WARN, 0 FAIL - 18 total rows
Status: Limited
Exit code: 0.                                                   exit 0
```

The full 18-row preflight output is pasted verbatim in
[docs/qualification/M3-readability-review.md](docs/qualification/M3-readability-review.md), section 6.2.

The two `WARN`s are the carried telemetry rows (in-process PDH value read; derived LUID
attribution), unchanged since M0. Nothing was relaxed.

```
$ .venv\Scripts\python.exe -m app.main --selftest
  ok   preprocess           letterbox 640x640 pad=114 scaled=640x360
  ok   infer                29.19 ms on confirmed ['CPU']
  ok   decode-track-event   1 zone_occupancy event(s) from a synthetic presence
  ok   render-metadata      unavailable renders as unavailable, and no device is claimed before an inference confirms one
PASS                                                             exit 0
```

```
$ .venv\Scripts\python.exe -m unittest discover -s tests
Ran 1376 tests in 522.476s

OK                                                               exit 0, wall 523.5 s
```

**Four suite runs, all reported.** Run 1 (1368 tests, 520.3 s) **failed the known RSS-ratio flake**
`test_route_stress...test_nothing_grows_that_the_retained_records_do_not_account_for`, which M2
recorded failing 3 times in 15 full runs and passing 7/7 in isolation — pre-existing, unchanged, not
fixed here. Run 2 (1368, 518.8 s) was **OK with one skip**: the live input-acknowledgement window was
refused by the clean-machine census, which is the blocker fix below working. Run 3 (1376, 524.9 s)
followed that fix. **Run 4 (1376, 522.5 s) is the gate run above**, repeating run 3 after a
docstring-only edit so the figure belongs to the committed tree: **no skips**, including the seven
live-window assertions.

### M3.2 The four committed captures — and what they do not show

`docs/qualification/m3-capture/{100,125,150,200}pct.png` + `.json`, produced by
`python -m tools.capture_ui --out docs/qualification/m3-capture`. They were committed by M3
Tasks 1-2 and re-rendered at Task 5; **Task 6 changes none of the eight files**, and this
commit adds the review that reads them.

| scale | pixels, **measured from the PNG** | JSON record | logical | dpr | `image.devicePixelRatio()` |
|---|---|---|---|---|---|
| 100 % | **1920 × 1080** | 1920 × 1080 | 1920 × 1080 | 1.00 | 1.00 |
| 125 % | **2400 × 1350** | 2400 × 1350 | 1920 × 1080 | 1.25 | 1.00 |
| 150 % | **2880 × 1620** | 2880 × 1620 | 1920 × 1080 | 1.50 | 1.00 |
| 200 % | **3840 × 2160** | 3840 × 2160 | 1920 × 1080 | 2.00 | 1.00 |

Decoded with OpenCV and compared against each record, because a gate that reads `dpr` would pass at
100 % and fail at 200 %: `QImage.devicePixelRatio()` is **1.00 at all four**.

- `overflow`, `truncated`, `absent`, `unreachable`, `geometry_unavailable`: **empty at all four**.
- The **33 non-empty named strings and every logical extent are identical at all four**. That is
  structurally correct — a scale factor is a rendering scale, not a layout change — and it is
  **not evidence of legibility at 125/150/200 %**. The capture README says so in its own words and
  the review repeats it; this section does not upgrade it.
- **Font census `{18: 29, 26: 1, 34: 3}`**, identical at all four, over named items with non-empty
  text. The raw map has 36 entries because three are blank and invisible (`streamReasonText_A`,
  `streamReasonText_B`, `tileCaptionText_A`).
- **`glyph_pixels` scales with area, not length**: `card_A` 121 847 → 190 269 → 274 750 → 487 388,
  a **4.0000×** ratio from 100 % to 200 %. That is the arithmetic proof the count indexes the
  **device-pixel** buffer — the first version indexed a 3840×2160 buffer with logical extents and
  reported 0 for drawn items at three of four scales. 76 of 77 entries are positive; the one zero is
  `tileCaptionText_A`, which is invisible and empty, so zero is the correct answer.

**What they do not show — stated because a "committed screenshot" sounds like more than it is:**

1. **No capture contains a video frame.** All three tiles read `no frame from Camera N yet`. The
   captures prove layout, strings, clipping, pixel size and the font census; they do **not** prove
   the video surface or the overlay boxes. M2's live run is the picture evidence
   (`docs/qualification/M2-qualification.md`), and a live capture is M4 work.
2. **The draws are seeded from a measurement that understates the live screen** — §M3.4.
3. **The census covers named items only**, and the screen carries unnamed 40 px and 26 px text that
   it cannot see — §M3.7.

### M3.3 The readability review at 2–3 m — the failures

**The display is not established anywhere in the repository.** The review assumes **0.310 mm per
logical pixel** (a 595 mm-wide canvas) — the midpoint of two independent anchors that agree to
0.8 %: a 27-inch 16:9 panel (0.3113 mm/px) and the **measured** 34-inch DELL S3422DWG with a 16:9
canvas pillarboxed to its height (0.3088 mm/px). It states the whole table as a function of panel
width so a reader can substitute the real booth panel.

Thresholds: **16 arcmin** character height for comfortable reading (ISO 9241-3's ergonomic minimum),
**10–16′** marginal (the 20/40 band), **< 5′** below the acuity floor (a Snellen 20/20 letter is
5 × 1 arcmin). Character height is cap height; for mixed-case sentences the x-height is 0.72 of it,
so sentences are 28 % worse than their row says.

| element | font px | cap px | mm | 2 m | 3 m | verdict at 2 m | verdict at 3 m |
|---|---:|---:|---:|---:|---:|---|---|
| Title `Intel Edge AI Roadshow` | 40 | 29 | 8.99 | 15.5′ | 10.3′ | marginal | marginal |
| `Recorded footage · live inference` | 26 | 19 | 5.89 | 10.1′ | 6.7′ | marginal | **FAIL** |
| `device: CPU` (placement proof) | 26 | 19 | 5.89 | 10.1′ | 6.7′ | marginal | **FAIL** |
| `Camera A` + state | 26 | 19 | 5.89 | 10.1′ | 6.7′ | marginal | **FAIL** |
| `events:` prefix | 26 | 19 | 5.89 | 10.1′ | 6.7′ | marginal | **FAIL** |
| **frame-age headline `11.7 s` (×3)** | 34 | 24 | 7.44 | 12.8′ | 8.5′ | marginal | **FAIL** |
| `STALE - this picture is 11.7 s old; the limit is 2 s` (×3) | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |
| per-camera badge (×3) | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |
| `backlog 187 frames`, `analysis 13.44 fps`, `window: 256 analysed frames`, activity line (×3 each) | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |
| events row, pipeline strip, status line, scope note, drawer buttons | 18 | 13 | 4.03 | 6.9′ | 4.6′ | **FAIL** | **FAIL (below acuity)** |

**The four findings, in order of how much they matter:**

1. **The three frame-age headlines — the numbers spec §7 makes the whole screen about — are 12.8′ at
   2 m and 8.5′ at 3 m.** They are the largest live numbers on the screen and they fail the comfort
   threshold at both distances on the assumed panel.
2. **All 18 px text fails at 2 m and is below the acuity floor at 3 m.** That includes the sentence
   that says *why* the number is stale, the device badges, the backlog and rate, the event row, the
   pipeline and the status line.
3. **The two pieces of text that make the screen honest — the recorded-footage label and the reduced
   scope note — are in the failing set** (26′ marginal at 2 m / failing at 3 m; 18′ failing at both),
   and **the scope note is elided mid-word on screen** (`…on ret…`), because it has no `objectName`
   and no harness reports the cut.
4. **On a 55-inch 1080p panel** the age headlines pass at both distances (26.2′ / 17.4′) and the 18 px
   text still fails at 3 m (9.4′). **On the 27-inch anchor, nothing clears 16′ at either distance.**

**This is recorded, not fixed.** A font enlarged here would have needed a new 1920×1080 layout and
would have broken the four-scale evidence the milestone exists to produce. The fix is M4's: the
booth's physical panel, a ramp for a 2–3 m viewing distance, and an explicit decision about which
numbers a passer-by is *entitled* to read at that distance.

### M3.4 The seeded staleness **understates** the live screen — measured

The captures' staleness (`11.7 / 10.7 / 11.1 s`) is seeded from
`docs/qualification/m2-headless/allcpu-steady-120s.json`. That file came from
`tools/route_stress.py::measure_steady_state`, which builds a **bare `Supervisor` and a
`QCoreApplication`** — no Qt window, no `FrameImageProvider`, no render loop, no UI-thread
`_on_result`/`_store_image`. So the seed is the age of a picture measured on a lighter process than
the booth runs.

Three measurements of the same condition (three CPU streams, 15 fps offered each, no churn), on this
machine:

| path | hold | frame age p50 (worst stream) | p95 | backlog | analysed |
|---|---|---:|---:|---:|---:|
| **committed seed** (M2, 2026-10-03, bare `Supervisor`) | 120 s | **11 319 ms** | 12 286 ms | 173–187 | ~13.5 of 15 fps |
| bare `Supervisor`, **re-measured today** | 120.2 s | **4 465 ms** | 4 848 ms | 58–73 | 1729–1744 of 1802 |
| **live windowed app** (`app.main.run_windowed`, 130 s warm-up) | 163.5 / 193.6 s | **40 111 / 48 572 ms** | 43 217 / 51 682 ms | 642–836 | 5 172 / 6 145 frames |

**The seed understates the screen by 3.4×–4.5×, and the presentation path is where the difference
comes from**: today's *bare* `Supervisor` is **faster** than M2's committed figure (4.5 s against
11.3 s), while the windowed app on the same machine, same day, same clip reaches 40–49 s. The live
run also **dropped 32 frames** where the headless holds dropped none.

**What is not established:** the share of the gap between "the UI path costs worker throughput" and
"the machine has drifted since 2026-10-03". Today's headless run argues against drift being the
cause, but one day is not a cause. **The screenshots were not re-seeded**; the discrepancy is
recorded, and a live capture under the booth's real load is M4 work. (A live capture for this review
landed in **attract mode** after 90 s idle, which is why it was not used as the committed screenshot —
the idle timer has to be handled before a live capture can stand in for a seeded one.)

### M3.5 The 1920×1080 DPI gate — `passed with documented failures`

A **new value** in `docs/qualification/qualification-matrix.csv`, added the way M2 added `failed`
when the vocabulary did not have the value it needed.

- **The render half is met:** four scales really rendered, pixel sizes measured from the images,
  nothing overflowing, strings read out of the scene in a closed audit window.
- **The readability half failed**, and is documented. Per this task's instruction that is the second
  admissible path; a clean `passed` would be a round-up and `not passed` would misdescribe the render
  gate. M2 set the standard by **downgrading** an M0 `passed` to `not tested` when the evidence did
  not support it; this is the same standard applied in the other direction.

### M3.6 M3 exit criteria

| # | criterion | verdict | evidence |
|---|---|---|---|
| 1 | renders at 1920×1080, no overflow, nothing outside the canvas at 100/125/150/200 % | **PASSED** | four captures; pixel sizes measured from the PNGs; `overflow`/`absent`/`unreachable`/`geometry_unavailable` empty at all four |
| 2 | every number a passer-by must read is legible at 2–3 m **or the failure is documented** | **MET IN ITS SECOND FORM ONLY — the numbers are not legible** | M3.3: nothing clears 16′ at 3 m on the assumed panel; the 34 px age headline is 12.8′ at 2 m / 8.5′ at 3 m; all 18 px text is below the acuity floor at 3 m |
| 3 | a committed screenshot at each scale, strings read out of the scene graph | **PASSED** | `m3-capture/{100,125,150,200}pct.{png,json}`; `opened_during_read: []`, `read_window_closed: true`, `walk_complete: true`, 33 non-empty strings at every scale |
| 4 | attract mode shows no number it cannot date, and restores the interrupted state exactly | **PASSED** (Task 4, re-verified in the 1376-test run) | three dated headlines with sources, no live figure — `AttractModeOnTheScreen::test_no_live_figure_appears_in_attract_mode`, `…test_leaving_attract_restores_the_live_screen_exactly`; a live 150 s capture landed in attract mode and showed the dated claims, not a reading |
| 5 | the presenter drawer moves nothing and states the retail scope note verbatim | **PASSED** (Task 5, re-verified in the 1376-test run) | `PresenterDrawerOnTheScreen::test_opening_the_drawer_moves_nothing`, `…test_closing_the_drawer_puts_every_item_back_where_it_was`, `…test_the_drawer_states_what_the_delivered_event_is` / `…_is_not`; 32 items identical before/during/after; drawer left edge x = 1306 against a video column ending at 1297 |
| 6 | the input-ack gate's blindness to inference backlog is **demonstrated** | **UNMET** | the four-condition matrix was never obtained. One condition was measured for this review (M3.8) and the gate passed while the frame age was 40–49 s — that is a window, not the matrix, and is not claimed as the demonstration |

### M3.7 The blocker carried into Task 6, and the defect under it

The suite arrived at **1362/1363 with one error**: the live input-acknowledgement test's `setUpClass`
errored when the clean-machine guard saw a `python.exe` outside the run's process tree appear during
the window. On a busy machine the correct behaviour is to **skip with the reason**, not to error —
Task 3's report already claimed it did.

- **The refusal is now a type**, `MachineNotClean(RuntimeError)`, raised by `assert_machine_clean`
  and by `measure`'s mid-window census. Every existing caller keeps refusing exactly as before, and
  **`measure` still returns no row**, so a figure measured on a dirty machine still cannot be
  recorded. The type only lets a caller tell "busy machine" (skip) from "probe measured nothing"
  (error).
- **The live test translates that one exception into a skip** naming the reason, the competing PID
  and the fact that nothing was recorded, and names the experiment to run instead. Every other
  `RuntimeError` still errors — asserted by a test that fails if the skip is widened.
- **The root cause of the false positive: the census took two process-table snapshots.** It asked
  `_our_process_tree()` (snapshot A) which PIDs were ours and `python_processes()` (snapshot B)
  which PIDs were Python. **Our own worker, spawned between the two reads, was in B but not in A —
  and was reported as a competitor.** Measured twice, on a machine that idles with zero `python.exe`.
  The census now takes **one** snapshot. That is what makes the live window run in the suite
  (run 3: no skips).
- **Mutation-checked: 6 mutants, 6 killed, 0 controls** (skip removed; skip widened to all
  `RuntimeError`; type not a `RuntimeError`; bare `RuntimeError` in `measure`; pre-flight census
  removed; two snapshots restored). One mutant first appeared to survive because of a **stale
  `.pyc`**; with bytecode deleted, all six are killed deterministically.

### M3.8 The input-ack blindness: one condition measured, the criterion still unmet

Measured for this review's live-frame-age question, on an asserted-clean machine, two windows of one
process (`python -m tools.input_ack_under_load --condition three-cpu-no-saturators --seconds 30
--repeats 2 --warmup 130`):

```
condition                rep  streams sat  load_s  tick p50  tick p95  tick max   age p50   age p95  backlog   gate    age
three-cpu-no-saturators    0        3   0     164       0.0      10.2      25.2   40111.1   43216.9      709   pass  stale
three-cpu-no-saturators    1        3   0     194       0.0      11.3      26.6   48572.3   51681.5      836   pass  stale

gate says pass                : 2
frame age says stale          : 2
gate passes while age is stale: 2  ['three-cpu-no-saturators']
```

**The gate passed while the picture was 40–49 seconds old, twice.** That is the shape criterion 6
exists to demonstrate. It is **not** the demonstration: the criterion asks for the four-condition
matrix, and three conditions were not run. **Criterion 6 stays unmet**, and the reason is recorded —
the two measurements that each need an idle machine were run concurrently by the previous task, so
every window was refused. This session ran its timing measurements **sequentially** on an
asserted-clean machine.

### M3.9 The performance fix, and its ceiling

Commit `6e4253a`, measured with a temporary probe on the real windowed application under three
cameras (full detail in the M3 Task 3b report, which is **not committed** — `.superpowers/` is
gitignored — so these are **[quoted]** figures, per M0's convention):

| metric | before | after |
|---|---:|---:|
| UI-thread CPU per result (`_on_result`) | **19.7 ms** | **7.0 ms** |
| `_store_image` (BGR→RGB + `QImage` + `.copy()`) | **18.2 ms** | **6.0 ms** |
| return unpickle per result | 1.70 ms | 0.57 ms |
| UI apply throughput | 8.6 results/s | **32.3 results/s** |

**Root cause: 91 % of the UI thread's work was building a full-resolution `QImage` per result.** The
fix reduces each successful result to a **960 px wide display copy after `infer_once`**, scaling the
detection boxes with the frame; the detector still sees the full-resolution frame, which was the
forbidden change and was not made.

**The ceiling, stated rather than hidden: total UI-process CPU barely moved** (1.61–1.87 cores
before, 1.72 after), because the ~0.48–0.87 cores of unattributed Qt/OpenVINO threads are unaffected
and the submit-side pickle of the full-resolution frame is unchanged. **This fixes UI responsiveness,
not aggregate load.** Aggregate load is M4's problem.

Two defects found on the way and recorded: **`_on_result` runs on the UI main thread**, not the
reader thread its docstring claimed (per-thread instrumentation, 724/724 calls), so per-result tracker
and event work was landing on the thread that must stay responsive; and **`--headless --streams 3`
was silently ignored** — it ran one stream and said nothing, which is why headless reported ~301
frames whether asked for 1 or 3 cameras. That is now **refused with a stated reason**:

```
$ .venv\Scripts\python.exe -m app.main --headless --streams 3 --seconds 1
--streams 3 is not supported by --headless: the headless run builds one supervisor for one stream
and its summary is single-stream, so accepting this flag would run one camera while claiming 3.
Use --windowed for the multi-camera layout, or omit --streams.
                                                                 exit 2
```

**Correct behaviour, not a restriction to work around.**

### M3.10 Attract mode and the drawer

- **Attract mode** (Task 4, `fcef5cf`): after **90 s** with no presenter input, the live screen is
  replaced by an overlay carrying **three dated recorded claims with their sources** — the M2
  per-device latency range (`recorded 2026-10-03 · docs/qualification/M2-qualification.md`), the M2
  three-camera frame age (`… allcpu-steady-120s*.json (3 holds)`), and the M1 delivered event
  (`recorded 2026-10-02 · docs/qualification/M1-qualification.md`). **No live value appears in
  attract mode**; the live CPU figure was asserted absent from every `attract*` string at the same
  moment the bridge carried it. Leaving attract restores the interrupted state item by item, and the
  font census across a round trip is unchanged. The spec's slide **rotation** is deliberately not
  implemented.
- **The presenter drawer** (Task 5, `fb3cccd`): it is an overlay anchored to the right at
  **x = 1306**, so it cannot reflow the hero, the tiles, the badges or the staleness line — **32
  items are identical before, during and after**, and the drawer's left edge is to the right of the
  video column's 1297. It states the retail scope note **verbatim** (people-in-zone presence, not a
  people count, not a dwell measurement) and states that GPU/NPU utilisation is unavailable and why.
  `F1` opens, `Escape` closes.

**Presenter-guide note — `--fail-stream`.** `--fail-stream A|B|C` is a **qualification aid, not a
booth feature**: a fault-injection flag so the visible-failure path can be exercised without breaking
a real device. **Do not use it while a customer is watching, and do not present it as a demo
feature.** The same applies to `--fault-stream` in `tools/route_stress.py`.

### M3.11 Unmet gates and open blockers after M3

| # | item | status |
|---|---|---|
| 1 | **M3 exit criterion 6 — the input-ack blindness matrix** | **UNMET.** The tool exists and is tested; one condition measured (M3.8); the four-condition matrix was never obtained |
| 2 | **The screen is not readable at 2–3 m** | **NEW, documented in full** (M3.3) — the 34 px age headlines are 12.8′ / 8.5′; all 18 px text is below the acuity floor at 3 m on the assumed panel. **M4** |
| 3 | **The seeded captures understate the live screen** | **NEW, measured** — 10.7–11.7 s seeded against 40.1 / 48.6 s live; not re-seeded. A live capture under load is **M4** |
| 4 | **No committed capture contains a frame** | **NEW** — the screenshots prove layout and strings, not the video surface. **M4** |
| 5 | **The scope note is elided on screen** and no harness reports it | **NEW** — full wording in the `zone` explanation and the drawer; the header cuts it mid-word. design/M4 |
| 6 | **The font census covers named items only** | **NEW, corrected** — 40 px text (the title) is on screen and outside the census; the capture README is corrected |
| 7 | M0 gate 9's **SILENT** half | **open, cannot close on this machine** (all three devices execute where asked; compile failure kills the worker before it runs). M6 |
| 8 | The 2 s warm-transition target | **MISSED** — p50 2.238–2.261 s, 37/50 over 2 s at the real 2.0 s default. Documented. M4/M6 |
| 9 | The leak at the real 2 s timeout: **182.4 MiB total, 175.3 MiB unexplained** | **open, not characterised.** M4 |
| 10 | GPU/NPU utilisation attribution | **open** — the inference PID never appears as a counter instance; the cards read "Utilization unavailable", never 0 %. M4 |
| 11 | Shelf-to-basket | **unmet by design** — people-in-zone presence, not a count, not a dwell |
| 12 | Clip redistribution clearance; detector licence (AGPL-3.0); code signing | **unchanged / unresolved / unmet.** M6 |
| 13 | Hardware SKU `unreconciled`; AC status unavailable | **unchanged** |
| 14 | A 60 s+ unattended soak | **not run.** M6 |
| 15 | The RSS-ratio suite flake | **pre-existing, unchanged** — failed 1 of this session's 3 full runs, passes in isolation |

### M3.12 Known costs and limits of M3

- **Suite 522.5 s (wall 523.5 s)** on a clean machine, against M2's 359–360 s — and 518.8 s,
  520.3 s and 524.9 s across the other three runs of this session. Ruling 20 already
  recorded that the M3-era runtime is genuinely higher and that the harness tests do not account for
  it; **unexplained, and not attributed.**
- **The machine's headless throughput moved in the *opposite* direction from its windowed
  throughput** in this session: headless 4.5 s (faster than M2's 11.3 s) and windowed 40–49 s. Any
  future report that quotes one as the other is wrong; they are different paths and M3.4 is the
  measurement.
- **The harness once measured a 1×1 window.** M3 Task 4/5's report records it: the child never set
  the window size, so every button measured **0 × 0** and clicks landed on nothing — a harness that
  would have "verified" every affordance while clicking empty space. Fixed; the harness now sets
  the size the way `tools/capture_ui.py` does. Recorded because it is the same class as Ruling 1: **the measuring
  instrument was the thing measured**.
- **`--fail-stream` and `--fault-stream` remain in the shipped CLI** as qualification aids (M3.10).
- **The DPI evidence is four offscreen renders.** The live desktop here is 2293 × 960, so a
  1920 × 1080 window cannot be displayed at all; nothing in this section is a claim about this
  monitor.

**Reproduction.** `python -m tools.capture_ui --out docs/qualification/m3-capture` renders the four
captures (seed mode; no inference worker needed). The readability table is generated by the script in
§11 of [the review](docs/qualification/M3-readability-review.md). The gates are the four commands in
M3.1. Everything under `docs/qualification/m3-capture/` is committed; nothing in this section rests
on `logs/` or on the gitignored milestone directory except the four figures in M3.9, which are
labelled **[quoted]**.

---

## M4 — sustained trials and legibility

Date: 2026-10-27 (Task 1, legibility at 3 m; Task 2, the input-ack matrix — M4.7; Task 3, the
timeout-path leak; Task 4, the sustained paired trials; Task 5, utilisation attribution; Task 6, the
milestone close — each below, in the order the sections were written)
Milestone: M4

**Verdict, Task 1 — corrected. On the monitor actually attached to this machine the screen
passes 16 arcmin of cap height at 2 m and fails at 3 m.** The monitor is measured, not assumed:
**DELL S3422DWG, 80 x 33 cm, 3440x1440 native, 2293x960 desktop, 0.310 mm per logical pixel**
(WMI; M4.5). At that scale 44 px is **16.8′** of cap at 2 m and **11.2′** at 3 m, the frame age
(56 px) is **21.4′ / 14.2′**, and **no element clears 16′ of x-height at either distance**
(12.2′ / 8.1′ at 44 px) — so "passes at 2 m" is a **cap-height** statement, not a full pass.
A **55-inch 1080p panel would pass at 3 m** (22.9′ / 16.6′); that is a **booth procurement
requirement, not an achieved measurement**. The first version of this section judged the render
against that panel and led with the pass; it was the wrong monitor, and M4.5 records the
correction. **The redesign itself is unchanged and is not reverted**: it removed six figures per
camera, the event row, the pipeline strip, the control bar and the footer `?` row (all in the
presenter drawer, or, for `PipelineStrip.qml`, deleted and replaced by a sentence), and nothing
on the screen was shrunk. What changed is the geometry the result is judged against.

`tools/preflight.py` is **16 PASS / 2 WARN / 0 FAIL (Limited)**, exit 0; `verify_sources` is 6 PASS /
0 FAIL; `--selftest` PASS; the suite is **1393 tests, OK, 542.7 s** (wall 543.6 s) after the
correction (M4.6; the pre-correction run was 1388 tests, and the five extra are the
attached-monitor tests).

### M4.1 The angular table, at both geometries

Measured from `docs/qualification/m3-capture/100pct.json` — the rendered scene graph, the font
engine's own metrics at the size the scene reports, and the ink in the saved PNG. Full table and
method in [docs/qualification/M4-legibility-review.md](docs/qualification/M4-legibility-review.md).
Every visible element is one of these three sizes, so this table covers all 24 of them.

**On the attached monitor — the achieved result** (`mm/px = 0.310`):

| element | px | 2 m cap | 3 m cap | 2 m x-height | 3 m x-height |
|---|---:|---:|---:|---:|---:|
| body — label, title, badge, verdicts, captions | 44 | **16.8′** | **11.2′** | 12.2′ (marginal) | 8.1′ |
| camera names | 48 | **18.3′** | **12.2′** | 13.3′ (marginal) | 8.8′ |
| **frame age** | **56** | **21.4′** | **14.2′** | 15.5′ (just short) | 10.3′ |

**Passes at 2 m on cap height; fails at 3 m on both measures; x-height fails at 2 m too.**

**On the 55-inch panel the booth is required to provide — a requirement, not the result**
(`mm/px = 0.63416`, 3 m): 44 px **22.9′ / 16.6′**, 48 px **25.0′ / 18.1′**, 56 px
**29.1′ / 21.1′**. All clear 16′. A booth that installs that panel gets a screen that reads at
3 m; a booth that installs anything under 55 inches does not, and this machine's monitor is the
latter case. At 40 in the same render is 16.7′ / 12.1′ (lowercase marginal), at 32 in 13.3′ /
9.6′, and the attached monitor 11.2′ / 8.1′. **No tool in this repository can gate a panel
size.**

### M4.1a The correction: the measured monitor, and the 55-inch requirement

The first version of this section judged the render against the 55-inch panel and reported the
pass as the result. The specification says key content must be readable at 2–3 m "on the actual
monitor" (lines 17 and 326 of the operator's copy; the specification file is not committed in
this repository), and the actual monitor was measured on 2026-10-27:

```
PS> WmiMonitorBasicDisplayParams  ->  MaxHorizontalImageSize 80, MaxVerticalImageSize 33
PS> WmiMonitorID                  ->  DEL | DELL S3422DWG | 809587030
PS> Win32_VideoController         ->  3440 x 1440 native, 150 % scaling -> 2293 x 960 desktop
```

The 1920x1080 canvas is height-limited: 1706.7x960 logical = 2560x1440 physical, and
`(2560/3440) x 800/1920 = 0.3101 mm/logical px` (0.3056 from the rounded 33 cm height; 0.310 is
the upper bound and is used). This is **0.310 — exactly M3's anchor** — so M3 was measuring the
actual monitor and the 55-inch assumption was about twice its scale. The full derivation, WMI
source and sensitivity table are in the review, §1.

The verdict is therefore stated in two parts, and the second is a requirement:

1. **On the actual monitor: passes at 2 m on cap height, fails at 3 m; x-height does not clear
   16′ at either distance.** 44 px is 16.8′ at 2 m, 11.2′ at 3 m.
2. **A 55-inch panel would pass at 3 m** (22.9′ / 16.6′ at 44 px). That is a **booth
   procurement requirement**, never the delivered result.

### M4.2 What was removed, and where it went

| removed from the always-visible screen | where it is now |
|---|---|
| per-camera backlog, analysis rate, frame-age window, activity line, utilisation figure, telemetry window | presenter drawer, per camera, **same item names** |
| the event counts; `status:`; `telemetry fresh` | presenter drawer |
| the header scope note (elided mid-word at 18 px) | presenter drawer, verbatim and in full |
| the seven footer `?` keys and the pinned panel | removed; one 44 px header button, a card click, or F1 opens the drawer that holds all seven explanations |
| `PipelineStrip.qml` (twelve 18 px labels) | **deleted**; `drawerPipelineText` states the topology |
| the per-card hover hint | removed — it was clipped by the card's own `clip: true` and could never be seen |

Kept and enlarged: the mandatory label, the title, the device badge, three captioned pictures,
each camera's name/state/age/device badge, and **one verdict line per camera** (the staleness
sentence, or that camera's stated failure reason).

### M4.3 The gate set, run in the M4 Task 1 session, before the panel correction

The figures in this subsection are the **pre-correction** run and are kept as the session
record. M4.6 is the gate set on the corrected tree.

On branch `main`, machine asserted clean before the timing run (`Get-Process python` → 0).

```
$ .venv\Scripts\python.exe -m tools.verify_sources
PASS    yolo11n-fp16          xml       xml+OpenVINO-IR, 412503 bytes, sha256 4845b46cc9b6ac379a44484258b7ed98f836ed117d91f127c7728ea068250a95
PASS    yolo11n-fp16:weights  binary    binary, 5283218 bytes, sha256 f14632518ebb4e312a89c26faa4cd2f95d4264b6f5f22aa6e0d5707b4667863e
PASS    yolo11n-fp16:labels   text      text+utf-8, 80 lines, 701 bytes, sha256 82bf6a434f132c4d78534c0f2e0f991fc22c73cd911f6d5fc74a418c39c339ec
PASS    retail_checkout       mp4       mp4+ftyp, 64996307 bytes, sha256 e107db69155af4774f907abe8ddbf24493dfcf9d81d51f3e71ff25300e4f5906
PASS    store_aisle_alt       mp4       mp4+ftyp, 9214573 bytes, sha256 3526fee39cac70d6366471e4324a6d63b337c8f8ab99c015521bee0d60ed6e04
PASS    queue_alt             mp4       mp4+ftyp, 3291918 bytes, sha256 a5964aa259099a482a8b360ffc2c57b5a30f84d5919236a4dad01f8e929ac07c

Source verification: 6 PASS, 0 FAIL                              exit 0
```

```
$ .venv\Scripts\python.exe -m tools.preflight
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
Exit code: 0.                                                   exit 0
```

The two `WARN`s are the carried telemetry rows, unchanged since M0; nothing was relaxed.

```
$ .venv\Scripts\python.exe -m app.main --selftest
  ok   preprocess           letterbox 640x640 pad=114 scaled=640x360
  ok   infer                25.91 ms on confirmed ['CPU']
  ok   decode-track-event   1 zone_occupancy event(s) from a synthetic presence
  ok   render-metadata      unavailable renders as unavailable, and no device is claimed before an inference confirms one
PASS                                                             exit 0
```

```
$ .venv\Scripts\python.exe -m unittest discover -s tests
Ran 1388 tests in 543.753s
OK                                                               exit 0   wall 544.7 s
```

Re-run on the exact committed tree (comment-only edits since): **1388 tests, 1387 green,
542.7 s (wall 543.6 s)**. The one failure is the pre-existing RSS-ratio flake
(`test_route_stress…test_nothing_grows_that_the_retained_records_do_not_account_for`), which
passes in isolation on this tree and which M2 recorded failing 3 times in 15 full runs. Both
runs are in the task report §9; neither is rounded up.

**`tools/probe_telemetry.py` was run and reported `PROBE OK`** (the NPU counter is found and the
LUID map is consistent). **It writes `config/telemetry_map.json` on every run with this boot's
LUIDs**, which this task is not allowed to change, so the file was restored to its committed
state afterwards — the map's LUIDs are per-boot values and the committed copy is the one the
preflight grades against. Recorded because "the verification tool rewrote a config file" is
something a later reader needs to know before trusting `git status`.

### M4.4 What Task 1 does not claim

- **The 55-inch panel is a procurement requirement, not a gate** (M4.1a). The delivered screen
  was measured on the attached monitor and **fails at 3 m**; the requirement is the largest open
  risk, and no tool in this repository can check it.
- The four captures still contain **no video frame** — they prove layout, strings, pixels and
  census, not the video path (M3 §5.4 stands).
- The drawer is 18 px: it is the presenter's arm's-length surface and is not part of the
  passenger legibility contract.
- M4 Tasks 2–6 (the timeout leak, paired trials, utilisation, the soak) are **not** started.

### M4.5 The panel correction, in full

**What was wrong.** The first version of M4.1 adopted 55 inches and reported "reads at three
metres" as the result. The specification says "on the actual monitor" (lines 17 and 326), and the
actual monitor is the DELL S3422DWG measured above. **The redesign is unchanged**; the geometry
the result is judged against changed.

**What the tests now do** (`tests/test_capture_ui_canvas.py::TheBoothScreenReadsAtThreeMetres`,
**16 tests** — 11 original, 5 added): the attached monitor (0.310) is the **default and gating**
geometry; each of its angles is **pinned per font size** so a shrunk token or a changed geometry
fails rather than silently changing the record; the test asserts that **no element clears 16′ of
x-height at 2 m** (the qualification on "passes at 2 m"); and the 55-inch requirement is a
separate, labelled conditional whose test also asserts it is a requirement. The existing
Slot guard, `Repeater`/`NaN`/unsized-window guards and ink assertions are untouched.

**Mutation-checked — six faults, six kills** (class run after each, file restored):

| mutant | killed by |
|---|---|
| panel width 800 → 1600 mm (wrong mm/px) | 5 tests: geometry, 2 m cap, 3 m pin, x-height, requirement |
| cap ratio 0.7164 → 0.6500 | `test_the_font_ratios_this_test_divides_by_are_the_fonts_real_ones` |
| x-height ratio 0.5184 → 0.5500 | `test_the_font_ratios_this_test_divides_by_are_the_fonts_real_ones` |
| distance set (2.0, 3.0) → (3.0,) | geometry test |
| `TYPE_BOOTH_DETAIL` 44 → 40 px (shrunk token) | 6 failures + 1 error (census, 2 m cap, 3 m pin, x-height, ink/ink-adjacent, discrimination) |
| `angle()` default switched to the 55-inch conversion | 5 tests, including the geometry and pinned 3 m tests |

The last mutant is the one that matters for the correction: **a legibility test that defaults to
the 55-inch panel cannot pass this suite.**

**The suite, re-run after the correction** — see §M4.6 for the real summary and wall time.

### M4.6 Gates re-run after the correction

On branch `main`, machine asserted clean before the run (`Get-Process python` → 0). Output
verbatim, timings measured around the process.

```
$ .venv\Scripts\python.exe -m tools.verify_sources
Source verification: 6 PASS, 0 FAIL                              exit 0

$ .venv\Scripts\python.exe -m tools.preflight
Summary: 16 PASS, 2 WARN, 0 FAIL - 18 total rows
Status: Limited                                                 exit 0

$ .venv\Scripts\python.exe -m app.main --selftest
PASS                                                             exit 0

$ .venv\Scripts\python.exe -m unittest discover -s tests
Ran 1393 tests in 542.650s
OK                                                               exit 0   wall 543.58 s
```

**1393 tests, no skips, no failures, wall clock 543.58 s.** The five extra tests over the
pre-correction 1388 are the attached-monitor assertions. `tools/probe_telemetry.py` was not
re-run by this correction, so `config/telemetry_map.json` was not rewritten.

### M4.7 The input-acknowledgement matrix, measured on an idle machine (Task 2)

The four conditions M3 recorded as an unmet exit criterion, obtained at last. Record:
`docs/qualification/m4-input-ack.json` (committed, byte-identical to
`logs/qualification/m4-input-ack.json`). Command, the tool's own defaults:

```powershell
.venv\Scripts\python.exe -m tools.input_ack_under_load --seconds 30 --repeats 2 `
    --out logs/qualification/m4-input-ack.json
```

Census `Get-Process python` **0 before, 0 after**; the tool's own census in the record is
`"clean": true, "competing_pids": []`. Exit 0, wall **699.7 s**. Warm-ups were **not**
shortened: 130 s budget for the three-stream conditions (`load_applied_s` 163 → 193 s), 12 s for
`one-cpu-saturated`. Every row is live through `app.main.run_windowed`, `application_path: true`,
and `conditions_failed: []`.

```
condition                        rep  ticks  tick p50  tick p95  tick max   age p50   age p95  backlog  drops  frames  gate/age
three-cpu-no-saturators            0   1487       0.0      12.1      24.6   36429.2   39044.7      639     29    5358  pass/stale
three-cpu-no-saturators            1   1481       0.0      12.6      28.3   44030.7   46862.2      752     29    6364  pass/stale
three-own-devices-no-saturators    0   1499       0.0       7.2      27.4      65.1      68.2        1      0    7203  pass/fresh
three-own-devices-no-saturators    1   1498       0.0       5.6      20.7      65.1      68.7        1      0    8556  pass/fresh
one-cpu-saturated                  0   1500       0.0       6.6      12.3    3476.2    5295.7       83      0     548  pass/stale
one-cpu-saturated                  1   1500       0.0       5.6      13.9    9196.0   11348.5      174      0     908  pass/stale
three-cpu-saturated                0   1473       0.0       9.4      89.1   34523.0   51205.2      839     32    4763  pass/stale
three-cpu-saturated                1   1465       0.0      11.6      91.9   55150.2   72135.2     1143     32    5192  pass/stale
```

**The verdict.** The gate is **not merely insensitive to inference backlog but a green light on a
system that is failing**: 8 of 8 windows pass (p95 5.6–12.6 ms against 200 ms, worst tick 91.9 ms
against 2 000 ms) while 6 of 8 are stale on the application's own 2 s rule, and in spec §5 row 4
the picture on screen is **34.5–55.2 s** old at p50 with **639–1143** frames queued. M2's claim
survives. One qualification: the gate does see *load* (`tick max` 20.7 → 91.9 ms), it cannot see
*queue wait*; and `three-own-devices` is **fresh** (p50 65.1 / 38.5 / 39.4 ms, backlog 1, zero
drops), so the failure is CPU capacity, not the screenshot.

**The tool had to be fixed to obtain it.** `spy_feeders` is installed as `main_module.
_start_feeders`, and `run_windowed` starts one feeder per camera, so with `--streams 3` it was
entered three times and armed everything three times: 18 saturators spawned, `state["procs"]`
holding only the last six, twelve survivors waiting on gates nobody sets — and
`multiprocessing`'s exit finaliser joins non-daemon children, so the probe could not exit. The
first full attempt **reported both `three-cpu-saturated` windows and was then killed at the
parent's 1800 s timeout** (2288.5 s wall), losing all four conditions to a temp-dir
`PermissionError`. Fixed with an arm-once guard (`state["armed"]`) and `daemon=True` on the
saturators; both are mutation-checked (removing either makes a test fail). Thresholds and the
clean-machine census are untouched; `ACKLD_ALLOW_DIRTY` was never set.

**Gates, re-run after the change:**

```
$ .venv\Scripts\python.exe -m tools.verify_sources
Source verification: 6 PASS, 0 FAIL                              exit 0

$ .venv\Scripts\python.exe -m tools.preflight
Summary: 16 PASS, 2 WARN, 0 FAIL - 18 total rows
Status: Limited                                                  exit 0

$ .venv\Scripts\python.exe -m app.main --selftest
PASS                                                             exit 0

$ .venv\Scripts\python.exe -m unittest discover -s tests
Ran 1395 tests in 524.300s
OK                                                               exit 0   wall 525.2 s
```

**1395 tests, no skips, no failures, wall 525.2 s.** The two over 1393 are the arm-once guard and
the daemon assertion. The full report is
`.superpowers/sdd/2026-10-01-m4-sustained-trials-and-legibility/task-2-report.md`; it records the
process census, everything killed, the mutation tally and the numeric-claims audit.

## M4 Task 5 - GPU/NPU utilisation, attributed per process (2026-10-27)

M0's second gate is closed with a measurement, not a refusal. The in-process `0xC0000005`
was a three-argument binding of the four-argument `PdhGetFormattedCounterValue`; with the
documented signature the call returns real values, and `\GPU Engine(*)\Utilization
Percentage` is per process by construction - every instance name carries `pid_<n>`, the
adapter LUID and the engine type. A provider filters each instance by its own `pid_` and by
the identified adapter (DXGI: `Intel(R) Arc(TM) 140V GPU (16GB)`, PCI `8086:64A0`, LUID
`0x000113B4`; NPU: the one adapter exposing `engtype_Neural`, LUID `0x000117B6`).

**The control** (`tools/probe_utilisation.py`; output
`docs/qualification/m4-utilisation-control.json`): one worker, idle then busy, sampled by
its own pid. GPU `0.0%` idle -> median **97.19%** (n=26, min 97.06, max 100.0) busy; NPU
`0.0%` idle -> median **64.86%** (n=26, min 63.76, max 66.08); the probe's own pid read
**0.0%** while that worker was at 97-100%, and another process on the **same** adapter read
1.42%. The machine was asserted clean before the worker started: 349 instances, busiest
0.00%. The 62%/68% machine-wide figure stays withdrawn: the runtime never returns a peak.

The GPU and NPU cards now render a figure when the application has an engine instance, and
"Utilization unavailable" with the sample's reason when it does not. A measured `0.0%` stays
a reading. `config/telemetry_map.json` was not modified; its recorded LUIDs are stale on this
boot (reported by preflight), and the runtime resolves the split from DXGI and the Neural
engine instead.

```
$ .venv\Scripts\python.exe tools\verify_sources.py
Source verification: 6 PASS, 0 FAIL                              exit 0

$ .venv\Scripts\python.exe tools\preflight.py
Summary: 17 PASS, 1 WARN, 0 FAIL - 18 total rows
Status: Limited                                                  exit 0
(the WARN is the recorded-map / hardware-identity row: stale LUIDs and three disagreeing
identity records, both reported)

$ .venv\Scripts\python.exe -m app.main --selftest
PASS                                                             exit 0

$ .venv\Scripts\python.exe -m unittest discover -s tests
Ran 1420 tests in 587.903s
OK                                                               exit 0   wall 590.3 s

$ .venv\Scripts\python.exe tools\probe_utilisation.py
VERDICT: attributed - the figure tracks this process            exit 0
```

**1420 tests, no skips, no failures, wall 590.3 s** (1398 before this task; 22 added).
Eight mutations were constructed and all eight caught. The full report is
`.superpowers/sdd/2026-10-01-m4-sustained-trials-and-legibility/task-5-report.md`; the
standalone account is `docs/qualification/M4-utilisation-attribution.md`.

## M4 Task 3 - the 182 MiB leak at the real 2 s timeout (2026-10-27)

M2 left this open and refused to close it on a hypothesis. M4 reproduced it at both deadlines, found the
owner, fixed it, and re-ran the same walk. Instrument `tools/leak_probe.py`: 50 transitions (M2's count),
a **fresh process** per run, the machine **asserted clean in every record**, feed **240x320 at 15 fps**
(M2's `FEED_SIZE`, so the numbers are comparable with M2's). Records
`docs/qualification/m4-leak-{before,after}-{2s,30s}.json`.

| | 2.0 s, before | 2.0 s, after | 30 s, before | 30 s, after |
|---|---:|---:|---:|---:|
| transitions / timed out | 50 / **25** | 50 / **25** | 50 / 0 | 50 / 0 |
| RSS start -> end (MiB) | 94.6 -> **248.4** | 94.3 -> **83.9** | 95.7 -> 103.8 | 95.7 -> 96.1 |
| **total growth** | **+153.7 MiB** | **-10.4 MiB** | +8.2 MiB | +0.4 MiB |
| retired-queue ledger | 6 -> **158** | 6 -> 6, **0 after stop()** | 6 -> 156 | 6 -> 6, **0 after stop()** |
| unexplained remainder | **+154.1 MiB** | -12.0 MiB | -3.7 MiB | -11.9 MiB |
| transition elapsed p50 / p95 | 3.161 / 4.207 s | 3.114 / 4.262 s | 3.265 / 4.631 s | 3.303 / 4.654 s |
| stop() | 0.101 s | 0.166 s | 0.151 s | 0.162 s |

**The shape is M2's; the byte count is not, and that is stated rather than smoothed.** M2 measured 182.4
MiB / 175.3 MiB with 24 unacknowledged transitions and 204 ledger entries; M4 measured 153.7 / 154.1 with
25 timeouts and 158. Different run, different machine state, same phenomenon. **The inversion is the
finding**: a 2.0 s deadline that times out 25 of 50 transitions grew RSS at ~3 MiB per transition (running
total +25.2 / +57.3 / +89.4 / +122.1 / +153.7 MiB at transitions 10/20/30/40/50), while a 30 s deadline
with **zero** timeouts was flat within allocator noise (+0.9 / +1.7 / +5.0 / +6.0 / +8.2 MiB). The queue
**count** was nearly identical in both (158 vs 156), so "more timeouts, more queues" is not the mechanism:
the difference is **what is inside them**.

**The cause, measured.** A timed-out transition retires a worker that is still compiling, so **nothing has
read its input queue**. The queue's own feeder thread writes into a Windows named pipe whose reader is
gone, parks in `Connection._send_bytes`, and what it has not sent stays in the queue's `_buffer`:
**20 of 52 retire calls discarded anything, 22-31 frames per non-empty drain** - about 2 s of a 15 fps
feed, which is the deadline - at p50 15.2 ms, 0.73 s in total. **And the ledger is the access path, not
the owner**: the probe emptied `_transitions` and `_retired_queues` and collected, and on the unfixed 2.0 s
run that released **nothing** (+154.1 MiB remainder against a +153.7 MiB total). **Pruning alone would have
fixed the accounting and left the leak.** `cancel_join_thread()` solved the *atexit rendezvous* - a
different problem - and released nothing, which is why the original fix looked complete. M2's hypothesis
was right about the **path** and wrong about the **count and the owner**: a clean run builds the same
number of queues.

**The fix** (`app/supervisor/runner.py`): `cancel_join_thread()` first and unchanged, then **drain what the
queue still holds**, then `close()`, then wait a bounded grace for the feeder to exit, and **drop the
ledger entry only when that thread is confirmed gone**. A queue whose feeder survives stays registered for
`stop()`'s sweep. Two measured choices: the drain uses `get(timeout=...)`, not `get_nowait()`, because
`PeekNamedPipe` reports **zero bytes for a partially written message** (`get_nowait` left **4 of 8** queues
blocked, vs **6 of 6** drained with polling); and pruning is gated on evidence, because the ledger is also
`stop()`'s re-entry guard and its final sweep. The fix was **not paid for with speed**: transition elapsed
p50 3.161 -> 3.114 s, p95 4.207 -> 4.262 s.

**The atexit hang is still prevented, and the proof hangs when the protection is removed.**
`tests/test_runner.py::RetiredQueueRelease::test_the_interpreter_still_exits_when_a_retired_queue_cannot_be_drained`
forces the pre-fix shape, requires `ledger 1`, `joincancelled True` and a normal exit inside 90 s, and
passes in ~1.4 s - **with `cancel_join_thread()` removed it hangs and is killed at 120 s**. `stop()` costs
0.166 s after 50 timeout-heavy transitions against a 5.0 s grace.

**Not claimed**: M2's exact 182.4 MiB is not reproduced to the byte; a 1080p feed was **not measured** (the
~27x ratio is arithmetic); RSS falling 10.4 MiB is the allocator handing arenas back, not memory returned
to the OS; and a shift-long soak is M6's. **One gate was changed and is flagged**: the pre-existing
`test_route_stress` RSS-ratio flake now reads RSS before its own `gc.get_objects()` census (which itself
retained 176 KiB) and carries an **8 MiB tolerance on the inversion direction only** - 5% of the 154.1 MiB
pre-fix retention - with the strict `leak_bytes < rss_start` half untouched. Full account:
`docs/qualification/M4-qualification.md` section 5.

## M4 Task 4 - sustained paired trials, and what they actually rank (2026-10-27)

**27 trials**, three conditions x three devices x three repeats, each in a **fresh process**, **interleaved
and rotated** (rep0 CPU->GPU->NPU, rep1 GPU->NPU->CPU, rep2 NPU->CPU->GPU), **540.33 s** of measured
window, **44 025** frames, campaign wall 758.8 s, machine asserted clean before and polled every 2 s
during. Record `docs/qualification/m4-sustained-trials.json`; harness `tools/sustained_trials.py`.

**Placement confirmed on all 27 runs** from `EXECUTION_DEVICES` read after inference (`CPU`, `GPU.0`,
`NPU`), 7 909 readings, **zero fallbacks, zero failed frames, zero drops**.

| condition | CPU | GPU | NPU |
|---|---|---|---|
| **booth feed, 15 fps** | 300/300, 14.99-15.00 fps | 300/300, 14.99-15.00 fps | 300/300, 15.00 fps |
| **source rate, 29.97 fps** | **26.3-26.6 fps**, 527-532 frames, age p50 **1.19-1.31 s** | 600/600, 29.96 fps | 600/600, 29.96 fps |
| **isolated sustained `infer`**, batch 1 | 45.4 fps | **307.4-310.4 fps** | **246.4-249.1 fps** |

**AT THE BOOTH'S RATE THROUGHPUT RANKS NOTHING: it is set by the source.** All three devices deliver the
15 fps stream, 300 of 300 frames, zero drops, every repeat; what differs is latency (p50 4.4-4.6 ms GPU,
6.6 ms NPU, 22.5-22.6 ms CPU) and frame age (30.6-31.1 / 32.6-33.1 / 48.1-48.5 ms). At the source's own
rate the only ordering is **CPU vs the accelerators** (the CPU falls 1.2-1.3 s behind at 26.3-26.6 fps
while both accelerators hold 29.96 fps), which is not a GPU-vs-NPU claim either.

**The only condition that ranks the two accelerators is isolated sustained inference: GPU 307.4-310.4
against NPU 246.4-249.1 fps (1.24-1.25x), p50 3.09-3.11 against 3.96-4.01 ms, in each of the three
interleaved repeats. The single-frame ordering did NOT invert at sustained load** - the
integrated-Arc-shares-memory hypothesis in the brief is **refuted on this workload**, and it is written
down as refuted.

**The NPU figure quoted since M0 is cold.** In the same process and the same trial the median of the
**first 30** inferences is **5.50-5.57 ms** against **3.96-4.01 ms** over the 20 s window (1.37-1.40x),
while the GPU (3.02-3.10 against 3.09-3.11) and the CPU (21.47-21.63 against 21.57-21.61) show no such
gap. The published 5.2-5.9 ms is an honest measurement of a 30-inference burst in a fresh process that
**overstates the NPU's sustained cost by about 40%**. **Any document quoting 5.9 ms beside a sustained
number must say which it means**; M0, M1 and M3 were not edited, and this is the correction.

**The retail stream is unchanged: 2 `zone_occupancy` presence events per clip**, in all 18 pipeline
windows across both feed rates and all three devices (36 events), plus a full pass through
`app.main --headless` analysing 1 694 of 1 694 frames and reporting `event_counts {'zone_occupancy': 2}`.
Track ids are **not stable across feed rates** (14/71-74 at 15 fps, 18/116-119 at 29.97); count, kind and
zone are identical, and a future test must not pin a track id. **This is people-in-zone presence: not a
people count, not a dwell measurement**, and shelf-to-basket remains unmet by design.

**Stated limits.** `sustained` `lat_max` sits far above p95 on all three devices **and the cause is not
identified** (reported untrimmed); the CPU `source` rows carry a 1.2-1.3 s frame age **and** +443.6 to
+474.3 MiB of RSS, which is the same backlog and **not a leak claim**; the ~6.5 ms pipeline NPU cost
against ~4.0 ms isolated is **unexplained**; one machine, one model, one batch-1 640x640 shape, one
OpenVINO build and one driver pair - **a driver update invalidates the blob cache and requires a re-run
before requoting**.

## M4 close - the five exit criteria, the gate set and what stays open (2026-10-27)

The milestone report is **`docs/qualification/M4-qualification.md`**; matrix rows were added for the
input-ack matrix, the timeout-path leak, the paired trials and the utilisation attribution, each marked
`passed` only because M4 re-measured it. The legibility row from Task 1 stands unchanged.

```
$ .venv\Scripts\python.exe tools\verify_sources.py
Source verification: 6 PASS, 0 FAIL                              exit 0   wall 0.14 s

$ .venv\Scripts\python.exe tools\preflight.py
Summary: 17 PASS, 1 WARN, 0 FAIL - 18 total rows
Status: Limited                                                  exit 0   wall 4.43 s
(the WARN is the recorded-map / hardware-identity row: stale per-boot LUIDs and three
disagreeing identity records, both reported; the per-process utilisation row is now PASS)

$ .venv\Scripts\python.exe -m app.main --selftest
PASS                                                             exit 0   wall 0.98 s

$ .venv\Scripts\python.exe -m unittest discover -s tests
----------------------------------------------------------------------
Ran 1459 tests in 599.133s

OK                                                               exit 0   wall 600.04 s
```

`tools/probe_telemetry.py` was **not run** by this task: it rewrites `config/telemetry_map.json` and this
task may not change a config file. The row it feeds is PASS on the committed map.

| # | M4 exit criterion | verdict |
|---|---|---|
| 1 | legibility measured on the **attached** monitor: 16' cap at 2 m, 3 m failure documented | **MET IN ITS DOCUMENTED FORM ONLY** - 16.8' at 2 m, **11.2' at 3 m**, x-height 12.2' / 8.1' (fails both). The 55-inch panel (22.9' / 16.6') is a **procurement requirement**, and a mutation defaulting the test to it kills five tests |
| 2 | the four-condition input-ack matrix, tick and frame age in the same rows | **MET** - 8 windows, gate pass 8/8, stale 6/8, backlog 1 to 1 143 and age 38.5 ms to 55.2 s in the same rows as tick p95 5.6-12.6 ms |
| 3 | the 182 MiB leak bounded and explained | **MET** - +153.7 -> **-10.4 MiB**, ledger 6->158 -> **6->0**, cause measured, 30 s case clean (+8.2 -> +0.4 MiB) |
| 4 | sustained paired trials, conditions stated, no unsupported generalisation | **MET** - 27 trials, placement 27/27, zero fallbacks; the rows rank CPU vs accelerators on the delivered stream and GPU vs NPU only in isolation |
| 5 | utilisation attributed per process, or refused with evidence | **MET, BY ATTRIBUTION** - GPU 0.0 -> 97.19%, NPU 0.0 -> 64.86%, probe's own pid 0.0% while the worker read 97-100%, distractor 1.42%; 62%/68% stays withdrawn |

**Criterion 1 is the one to read twice.** The plan's first clause ("clears 16' of cap height at 2 m") is
met and its second clause ("its failure at 3 m is documented") is met. **What is not met is the sentence a
reader may want: the booth screen does not read at three metres on this machine's monitor.**

**Unmet gates and open blockers after M4** (unchanged rows are unchanged; the new ones are marked):

| item | status | owner |
|---|---|---|
| Legibility at 3 m on the attached monitor | **fails**; the 55-inch panel is a requirement with **no gate** | procurement / M6 |
| M2's 2 s warm-transition target | **still missed** (p50 2.238 s, 37/50 at the real 2.0 s default); **not re-measured by M4** | M6 |
| M0 gate 9's **silent** half | **open, cannot close on this machine** - all three devices execute where asked | M6 |
| The GPU/NPU figure is the **busiest engine, not a device total** | stated meaning; a presenter comparing to Task Manager may see different numbers | presenter guide |
| Retail shelf-to-basket | **unmet by design** - presence, 2 per clip, not a count, not dwell | design / M5 |
| Code signing | **unmet** | M6 |
| A second annotated frame with a non-square object | **not done**; the w/h blind spot stands | M1 follow-up / M5 |
| A 60 s+ unattended soak | **not run**; the leak result bounds 50 transitions | M6 |
| `config/telemetry_map.json` is stale (per-boot LUIDs) | **reported, not edited**; the runtime no longer depends on it | operator |
| `sustained` lat_max outliers; retail track ids across feed rates; the 6.5 vs 4.0 ms NPU gap | **unexplained, recorded** | M6 |
| The 8 MiB RSS-ratio tolerance (Task 3) | **a changed gate, flagged for review** | reviewer |

**M4's own costs, named.** The suite grew 1376 -> 1459 tests across the milestone and took 600.0 s of wall
here; four measurements each needed an idle machine (matrix 699.7 s, trials campaign 758.8 s, two leak
conditions ~150 s each); `PipelineStrip.qml` was **deleted** and one stress-test assertion was restated
with a bounded tolerance; and the matrix vocabulary gained a split value (`passed at 2 m, failed at 3 m`).
None of that is a threshold lowered to make a gate green, and each is recorded where it happened.

---

## M5 - records, export and the recorded-demonstration mode (2026-10-30)

The milestone report is **`docs/qualification/M5-qualification.md`**; five matrix rows were added (four
`passed`, one `failed` for the content gate), each marked `passed` only because M5 re-measured it. This task
is **documentation only**: no code, `config/*.json` or tool was changed. Two claims were verified by a
temporary mutation and both were restored (`git diff --numstat` 0 entries afterwards).

```
$ .venv\Scripts\python.exe tools\verify_sources.py
Source verification: 6 PASS, 0 FAIL                              exit 0   wall 0.14 s

$ .venv\Scripts\python.exe tools\preflight.py
Summary: 17 PASS, 1 WARN, 0 FAIL - 18 total rows
Status: Limited                                                  exit 0   wall 4.36 s
(the WARN is the recorded-map / hardware-identity row: this boot's GPU LUID is 0x000113B4 and the
 committed file records 0x00010E9E / 0x000112A0; the runtime no longer depends on it. The file was not
 edited.)

$ .venv\Scripts\python.exe -m app.main --selftest
PASS                                                             exit 0   wall 0.96 s

$ .venv\Scripts\python.exe -m unittest discover -s tests
----------------------------------------------------------------------
Ran 1573 tests in 597.673s

OK                                                               exit 0   wall 598.6 s
```

1 573 = M4's 1 459 + **114 new tests** (record 40, export 19, comparison 20, recorded mode and CLI 23,
content gate 12). Machine asserted clean before the suite and before the one measurement taken here
(`Get-Process python` and `pythonw` both 0).

| # | M5 exit criterion | verdict |
|---|---|---|
| 1 | a fingerprint covering everything that changes a number, and a mismatched pair refused with the fields named | **MET** - 11 fields, SHA-256 over all of them (`4a1030e4...a447217` reproduced); a model-id-only hash gives **`FAILED (failures=10)`**, each of the other ten fields failing by name; a real pair differing on `conf_threshold` and `stream_count` is refused with exactly those names |
| 2 | the export contains manifest, transitions, metrics and logs and excludes imagery by default, proven by content | **MET** - 6 members and **0 image-content hits inside every member's bytes** on a record holding three frames; opt-in gives 9 members and `png`/`jpeg`/`webp`; the README states the mode |
| 3 | comparisons report ranges and medians, label older records `previous run`, invalidate a baseline on a setting change | **MET** - medians with `[min, max]` on both sides and `within noise`; `previous run` on the older record in either argument order; refusal gives `eligible: no`, `invalidated`, `per_metric {}`, exit 1 |
| 4 | a playback shows the recorded notice, disables live controls, cannot be mistaken for live | **MET** - notice visible and still visible after 400 ms; live label and live screen not visible; control and `MouseArea` `enabled: false`, opacity 0.45, a real click emits `[]`; the live mirror restores the label and a working control |
| 5 | the content shortfall measured from the files and reported | **MET AS A REPORT - THE GATE IS NOT MET**: 3 of 12 clips, 1 of 4 packs, 1 of 3 inside 45-90 s, **0 rights-cleared**, `tools/content_gate.py` exit 1 |

**Records.** `WorkloadFingerprint` covers `model_id`, `model_sha256`, `input_shape`, `conf_threshold`,
`iou_threshold`, `stream_count`, `target_fps`, `device_layout`, `decode_path`, `openvino_version`,
`numpy_version`. The artifact **hash**, not the id, is the field that stops two exports of one network
sharing a baseline. A record cannot exist without a fingerprint; a `recorded` record cannot be
`comparison_eligible`; a `SETTLED` sample inside `warmup_s` raises, which is what stops the NPU's published
5.2-5.9 ms cold figure being compared as its 3.96-4.01 ms sustained figure.

**Export.** Default members are `manifest.json`, `routing_transitions.json`, `metrics_summary.json`,
`metrics_raw.jsonl`, `errors.log` and `README.txt`. The scan reads every member's whole content for
`\xFF\xD8\xFF`, `\x89PNG` and `RIFF....WEBP`; it finds **none** by default and all three formats when
`include_imagery=True`. `metrics_summary.json` carries settled median **and** min/max with cold values listed
separately; unavailable is `null` with a reason, never `0`; two exports are byte-identical.

**Comparison.** Two records were taken for this report (fresh NPU compile, three 10 s windows each, 3 s
warm-up outside the timed span, `EXECUTION_DEVICES` re-read after inference -> `('NPU',)`) and are
**committed** at `docs/qualification/m5-records/`. Matched pair: `latency_p50_ms` 3.9972 `[3.9923, 4.00325]`
against 3.99325 `[3.9892, 3.99365]`, delta -0.00395003, within noise; `throughput_fps` 243.767
`[242.781, 243.804]` against 243.896 `[243.506, 244.104]`; `npu_utilization_percent` `n/a` with a reason on
both sides; cold 5.5885 / 5.5649 ms **excluded**, 3 settled samples per side. Refused pair: `conf_threshold`
0.25 -> 0.30 and `stream_count` 1 -> 2 give `eligible: no`, `baseline status: invalidated`, the two fields
named, `per_metric: {}`, exit 1.

**Recorded mode.** Read from the rendered `Main.qml`: notice `Recorded demonstration - metrics are not live`
at 44 rendered px, visible and visible again after 400 ms; `live_label_visible` and `live_screen_visible`
both **False**; the control and its `MouseArea` `enabled: False` at opacity 0.45 and a real click emits
nothing; every figure carries `recorded 2026-10-01T10:00:00 (+12 s into the recording)`; leaving restores
`Recorded footage - live inference` and a control that emits `[['A','NPU']]`. A playback over live work is
refused with *"refusing to enter recorded mode while live work is running: 1 frames have been analysed on
this screen"* and the live screen stays live.

**Content gate - NOT MET, reported from the files.** `python -m tools.content_gate`, exit 1, measured with
`cv2.VideoCapture`:

| requirement | required | provisioned | gap |
|---|---:|---:|---|
| packs | 4 | **1** | **3 missing** |
| clips | 12 | **3** | **9 missing** |
| clips inside 45-90 s | 12 of 12 | **1 of 3** | `retail-checkout` +23.0129 s, `one-by-one` +49.4 s |
| rights-cleared | 12 | **0** | no licence text embedded, no sidecar beside any clip |

`retail-checkout.mp4` 1920x1080 / 29.970 fps / 3 387 frames / **113.0129 s** (used by the app);
`store-aisle-detection.mp4` 720x404 / 59.940 / 3 921 / **65.41535 s** (in band, not used);
`one-by-one-person-detection.mp4` 768x432 / 10.000 / 1 394 / **139.4 s** (not used). **No footage was
acquired, generated, trimmed or substituted**; missing rights-cleared footage is a procurement dependency.

**What stays open after M5** (unchanged rows are unchanged):

| item | status | owner |
|---|---|---|
| The content gate | **NOT MET** - 3 of 12 clips, 0 rights-cleared; shelf-to-basket undelivered | procurement / operator |
| Legibility at 3 m on the attached monitor | **fails** - 44 px is 11.2' cap / 8.1' x-height at 3 m; the 55-inch panel is a requirement with **no gate** | procurement / M6 |
| M2's 2 s warm-transition target | **still missed** (p50 2.238-2.261 s, 37/50); **not re-measured by M5** | M6 |
| M0 gate 9's **silent** half | **open, cannot close on this machine** | M6 |
| `config/telemetry_map.json` is stale (per-boot LUIDs) | **reported, not edited** | operator |
| The Tasks 4-5 QML-precedence claim | **not reproduced** - the unparenthesised binding hid the live screen in this session's two runs; the delivered behaviour is verified, the stated reason for the parentheses is not | reviewer |
| `python tools/compare_records.py` as its docstring documents | **fails with `ModuleNotFoundError`**; `python -m tools.compare_records` works and was used. A doc defect in a committed tool, not fixed here | tool owner |
| A JSON-loaded record cannot re-export imagery | **by design**, stated as a limit | design |
| `test_main` 30.0% substring flake; route-stress GC-timing flake | **known, not observed in this run** | M0-M4 owner / M6 |

**M5's costs, named.** The suite is 1 573 tests and **598.6 s** of wall; the comparison measurement cost
**70.08 s** and produced three committed record payloads; the two mutation checks each needed a restore and a
`git diff` check rather than a clean tree. The matrix gained one status value, `failed`, for a gate that is
measured and unmet. Nothing in this milestone edited a config file, and no number here is a target or an
estimate.