# M0 — Hardware and Model Path Qualification

Target machine: the CPU identifies itself as `Intel(R) Core(TM) Ultra 9 288V` and reports
`Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)` and `Intel(R) AI Boost`; Windows 11 Pro build
26200.

> **No SKU is claimed anywhere in this document.** Three hardware-identity records on this
> machine disagree — `Win32_ComputerSystem` says `ASUSTeK COMPUTER INC. NUC14LNK LN`, the
> hostname says `NUC14PROAI`, the PCI subsystem id says `SUBSYS_88F21043` — and none is
> treated as the winner. See *Environment* below. The family name "Core Ultra Series 2 /
> Lunar Lake" appears only as the CPU's own model string, never as a claim about a product
> line the hardware does not report.

> Nothing in this document is a target, a vendor claim, or a number copied from a
> datasheet. Every figure below was produced by a command run in this repository and its
> raw output is pasted verbatim. A quantity that could not be measured is written
> `unknown` and stays that way. Figures **quoted from a gitignored file** rather than
> re-measured are labelled **[quoted]** at the point of use — see *Provenance of the
> numbers*, below the Assets section.

## Assets

Provisioning: `python tools/provision_venv.py` (the only step needing network). Artifacts are
copied from the already-downloaded, content-verified local cache and re-verified in this
tree; nothing is re-downloaded. Every SHA-256 below was computed with `hashlib.sha256` over
the file copied into this tree.

### Pinned environment

`pip freeze` from the provisioned `.venv`, 2026-10-01 (pip is not in the freeze; the pip
that ships with the venv is not recorded because no pip upgrade is performed — see
`tools/provision_venv.py`):

```
numpy==2.5.3
opencv-python==5.0.0.93
openvino==2026.4.0
openvino-telemetry==2025.2.0
psutil==7.2.2
PySide6==6.11.2
PySide6_Addons==6.11.2
PySide6_Essentials==6.11.2
PyYAML==6.0.3
shiboken6==6.11.2
```

`openvino-telemetry` is a transitive dependency of `openvino==2026.4.0`, not a direct pin.
The six direct pins in `requirements.txt` are enforced by `tests/test_pins.py`.

### Source verification

Command: `.venv\Scripts\python.exe tools\verify_sources.py` — content signatures plus the
recorded SHA-256. HTTP status is never accepted as proof; a 200 can return an error page.

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

Six rows: the IR, the weights, the 80-line label file, and the three clips. The
`text` row is a later addition — a recorded hash with no verification row is a hash
nobody checks. The SHA-256 is the gate; the shape checks above it are necessary but
cannot tell one valid IR from another.

### Recorded artifacts

| id | path | sha256 | measured |
|---|---|---|---|
| yolo11n-fp16 (IR) | `assets/models/yolo11n-fp16/yolo11n.xml` | `4845b46c…68a95` | input `[1,3,640,640]`, static |
| yolo11n-fp16 (weights) | `assets/models/yolo11n-fp16/yolo11n.bin` | `f1463251…8663e` | FP16, 5283218 bytes |
| yolo11n-fp16 (labels) | `assets/models/yolo11n-fp16/labels.txt` | `82bf6a43…339ec` | 80 classes |
| retail_checkout | `assets/media/retail-checkout.mp4` | `e107db69…f5906` | 1920x1080, 29.97002997002997 fps, 3387 frames, 113.0129 s |
| store_aisle_alt | `assets/media/store-aisle-detection.mp4` | `3526fee3…6ed6e04` | 720x404, 59.94005994005994 fps, 3921 frames, 65.41535 s |
| queue_alt | `assets/media/one-by-one-person-detection.mp4` | `a5964aa2…9ac07c` | 768x432, 10.0 fps, 1394 frames, 139.4 s |

Media figures are `cv2.VideoCapture` measurements on the copied files. `duration_s` is
`frames / fps`; the values are not rounded to a target.

### Licences and distribution status

| id | licence | distribution_status |
|---|---|---|
| yolo11n-fp16 | AGPL-3.0 (model card, read 2026-10-01) | recorded as-is; AGPL-3.0 is copyleft and is an open distribution question for the booth package |
| retail_checkout | Pexels License - free for commercial use, no attribution required | licence text carried over from the source manifest as recorded; **NOT re-read at M0** and not claimed as booth redistribution clearance |
| store_aisle_alt | Open Model Zoo sample media | recorded as-is; not claimed as booth redistribution clearance |
| queue_alt | Open Model Zoo sample media | recorded as-is; not claimed as booth redistribution clearance |

No clip is upgraded to "cleared". All three licence strings were carried over from the
source manifest rather than re-read from the provider at M0, so all three distribution
statuses are recorded as-is.

### Unmet content gate

Answered in full, from the footage, in **[M0-content-gate.md](M0-content-gate.md)**. The
durations above are re-measured there; the headline result is that **no shelf-to-basket clip
exists in the provisioned material**, a reduced-scope "items entering the checkout zone"
fallback is committed to, and 1 of 12 required clips falls inside the 45–90 s band.

---

# What M0 measured

Everything below was produced by a command run in this repository. Figures that are
**quoted from a gitignored file** rather than re-measured in the session that wrote this
section are labelled **[quoted]** at the point of use; everything else was re-measured here
and the command is named.

**Provenance of the numbers in this document — read this first.**

| source | committed? | what it backs |
|---|---|---|
| `logs/qualification/m0.json`, `logs/qualification/m0_sanity_fixture.json` | **no — `logs/` is gitignored** | the four-row compile/latency/placement table and the per-device fixture verdicts. On a fresh clone these files are **absent** and preflight's corresponding rows degrade to `WARN`. The numbers are reproduced here, in `bench_report.md` and in `docs/qualification/qualification-matrix.csv` precisely so that M0's headline evidence lives in a **tracked** file. |
| `.superpowers/sdd/**` | **no — gitignored** | Task 3's 62% / 68% CIM utilisation measurement, which preflight's telemetry row cites as its evidence. That citation does not survive a fresh clone, and the figure is in any case **not reproducible by the session that wrote this document** — see *Telemetry*, below. |
| re-measured in the Task 9 session, 2026-10-01 | yes, because it is pasted here | environment, media durations, device discovery, the letterbox pad A/B, the telemetry counter routes, and the footage review. |

## Environment

Every value below is a `Get-CimInstance` / `Get-PnpDevice` / `powercfg` /
`System.Windows.Forms` reading taken on this machine, 2026-10-01.

| fact | measured value | source |
|---|---|---|
| CPU | `Intel(R) Core(TM) Ultra 9 288V`, 8 cores / 8 logical, 3300 MHz max clock | `Win32_Processor` |
| RAM | 33805881344 bytes (31.5 GiB) | `Win32_ComputerSystem.TotalPhysicalMemory` |
| OS | `Microsoft Windows 11 Pro`, version `10.0.26200`, build `26200`, 64-bit | `Win32_OperatingSystem` |
| GPU | `Intel(R) Arc(TM) 140V GPU (16GB)`, `PCI\VEN_8086&DEV_64A0&SUBSYS_88F21043&REV_04`, driver **32.0.101.6737** | `Win32_PnPSignedDriver`, `Get-PnpDevice -Class Display` |
| NPU | `Intel(R) AI Boost`, `PCI\VEN_8086&DEV_643E&SUBSYS_88F21043&REV_04`, PnP status `OK`, driver **32.0.100.5540** | `Get-PnpDevice -Class ComputeAccelerator` |
| Python | 3.12.10 | `platform.python_version()` |
| OpenVINO | `2026.4.0-22959-99c81491cc3-releases/2026/4` | `ov.__version__` |
| Power scheme | **Balanced**, GUID `381b4222-f694-41f0-9685-ff5bb260df2e` | `powercfg /getactivescheme` |
| **AC status** | **UNAVAILABLE.** `Get-CimInstance Win32_Battery` returns **0 instances** on this machine | measured |
| Display | one panel, `\\.\DISPLAY1`, primary, logical bounds `2293 x 960`; working area `(48,0) 2245 x 960` | `System.Windows.Forms.Screen.AllScreens` |
| Display scale | `LOGPIXELSX=96`, `LOGPIXELSY=96` → 100%, 1.0x | `Graphics.DpiX` / `DpiY` |
| Monitor | `DELL S3422DWG`, instance `DISPLAY\DELD129\4&16626495&0&UID24631_0` | `WmiMonitorID` |

**The SKU is not established, and this document does not name one.** Three hardware-identity
records on this machine disagree:

- `Win32_ComputerSystem`: `Manufacturer = ASUSTeK COMPUTER INC.`, `Model = NUC14LNK`,
  `SystemFamily = LN`, `SystemSKUNumber = NUC14LNK` — composed, as
  `config/telemetry_map.json` records it, as `ASUSTeK COMPUTER INC. NUC14LNK LN`
- hostname: **`NUC14PROAI`**
- PCI subsystem id on both demo devices: **`SUBSYS_88F21043`**

`NUC14PROAI` and `SUBSYS_88F21043` are consistent with a NUC14 Pro AI; `NUC14LNK` is what
`Win32_ComputerSystem` reports. **None of the three is treated as the winner here**, and
`docs/qualification/qualification-matrix.csv` carries `machine_sku = unreconciled` on every
row for exactly that reason. Picking one of three disagreeing records to print on a booth
screen is how a Panther Lake or Arrow Lake number gets published as a Lunar Lake number.
The `machine_sku` cell names all three; the resolution is an operator action on a physical
label or an invoice, not something a `Win32_` class can settle.

### The 1920x1080 canvas cannot be shown on this display

Measured usable height is **960 px**; the spec §3 design canvas is 1920x1080. A 1920x1080
window **cannot be displayed** on this desktop, so the live booth layout must stay inside
2245x960 and the DPI gate is only reachable through the offscreen render harness
(`tools/capture_ui.py`, **M3 work — not run at M0**). Both facts are recorded here so that
nobody later reads a passing 1080p screenshot as a claim about this monitor.

## Device discovery

Command: `.venv\Scripts\python.exe -c "import openvino as ov; c=ov.Core(); print(c.available_devices); [print(d, c.get_property(d,'FULL_DEVICE_NAME')) for d in c.available_devices]"`,
run from the repo root on 2026-10-01:

```
openvino 2026.4.0-22959-99c81491cc3-releases/2026/4 | python 3.12.10
ov.Core().available_devices() -> ['CPU', 'GPU', 'NPU']
  CPU  FULL_DEVICE_NAME = 'Intel(R) Core(TM) Ultra 9 288V'
  GPU  FULL_DEVICE_NAME = 'Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)'
  NPU  FULL_DEVICE_NAME = 'Intel(R) AI Boost'
```

(`SUPPORTED_DEVICES` was also asked for and is **not** a property the CPU plugin exposes:
`Cannot get unsupported property: SUPPORTED_DEVICES`. It is not used anywhere in this
project and is recorded here only so the failure is not mistaken later for a missing device.)

All three devices are real and were each asked for by explicit `compile_model(device)`.
No `AUTO`, no `HETERO`, no `MULTI` — explicit targets only, per design §7.

## Model pin and recorded I/O

Artifact `yolo11n-fp16`, IR `assets/models/yolo11n-fp16/yolo11n.xml`,
SHA-256 `4845b46cc9b6ac379a44484258b7ed98f836ed117d91f127c7728ea068250a95`, matching
`config/models.json` and re-checked by `tools/verify_sources.py` and by preflight's
`Assets: model IR pin` row on every run.

Recorded from the IR itself into `config/model_io/yolo11n-fp16.json`:

| fact | recorded value | how it was established |
|---|---|---|
| input | port `0`, names `['x']`, `[1, 3, 640, 640]`, **all four axes static**, `float32` | `describe_inputs` over the IR |
| batch | 1, static — the NPU rejects dynamic shapes (AGENTS.md rule 3) | as above |
| output | port `0`, shape `[1, 84, 8400]`, **unnamed** — `get_names()` is the empty set and `get_any_name()` raises `Check '!get_names().empty()' failed` | `describe_outputs` over the IR |
| output layout | `yolo_v8_transposed` | classified from the observed tensor, then **cross-checked by independent code from the same observation; the two agree** |
| box encoding | `cx, cy, w, h` | from the layout classification; a mutant sweep confirms the fixture catches reading it as `xyxy` |
| `has_embedded_nms` | **`false`** | **derived, not assumed.** All 734 ops (24 distinct types) were walked and contain no `NonMaxSuppression` of any version; the output's trailing extent is 8400, the raw anchor count `80²+40²+20²` for a 640 input, not a frame-dependent post-NMS count. Two independent lines of evidence. |
| feature-count check | axis 1 length 84 = 4 + 80 classes — agrees | independent cross-check |

**Deviation from the "no embedded NMS" preference: none.** The export matches the design's
preference exactly, so suppression is the host's job and `app.inference/decode.nms` does it.

Two recorded facts that will bite whoever writes the inference worker, both measured rather
than assumed (`config/model_io/yolo11n-fp16.json`, `fetch_api`):

- `InferRequest.infer` accepts an **int port index**, a port **name**, or a
  `ConstOutput` from the **compiled** model, and **rejects the IR model's port object** with
  `TypeError: Incompatible key type for input`.
- `InferRequest.get_tensor` accepts the port object or a `ConstOutput` and **rejects an int
  index**, while `get_output_tensor(0)` accepts the int.
- The output port has **no name**, so the output side has no name-based call at all. The
  decoder is handed the tensor under `port_key(0)` — the string `"0"`, this project's own
  key, assigned because it is the only identifier that exists for an unnamed port.

### The letterbox pad decision

The pad band is filled with **`114`**, not `0`. **Ruling 24, 2026-10-01**; `114` is
`app.inference.preprocess.DEFAULT_PAD_VALUE`.

For the fixture frame, 1920x1080 into 640x640: scale 0.333333…, scaled 640x360, pad offset
`(0, 140)`, so the band is **280 rows of 640 = 43.75% of the model canvas**, split 140 top
and 140 bottom.

> ⚠️ **CANNOT VERIFY.** 114 is a pre-built IR downloaded from HuggingFace. **The conversion
> script that produced it is not in this repository**, so 114 is *the documented Ultralytics
> default, plus a measured sensitivity* — **not** a read of the actual conversion. No claim
> is made that 114 improves detections. What is claimed is narrower: it **removes a
> train/serve distribution shift**, and the measurement below shows it is not cosmetic.

**Measured in the Task 9 session** (throwaway harness using the repository's own
`preprocess_bgr`, `decode_detections` and the committed annotation; CPU, conf 0.25, iou
0.45, match_iou 0.5):

| pad value | detections | matched | verdict | the apple |
|---|---|---|---|---|
| **114** (in effect) | 4 | **2/2** | PASS | `apple` **0.273557** — 9.4% above the annotated `min_score` 0.25 |
| 0 | 3 | **1/2** | **FAIL** | **no apple detection at all** above conf 0.25 |
| 255 | 3 | 2/2 | PASS | `apple` 0.282365 |

Full decode at pad 114: `person` 0.913189, `person` 0.692380, `apple` 0.273557,
`oven` 0.250770. At pad 0: `person` 0.910445, `person` 0.622733, `oven` 0.255576 — **the
apple is gone from the frame**.

Six frames of the same clip, class set and coordinates at pad 114 vs pad 0:

```
frame 0     boxes 114=4 boxes 0=3 class sets DIFFER    coords shifted 2/3
frame 60    boxes 114=3 boxes 0=3 class sets DIFFER    coords shifted 2/3
frame 300   boxes 114=2 boxes 0=3 class sets DIFFER    coords shifted 2/2
frame 1200  boxes 114=3 boxes 0=2 class sets IDENTICAL coords shifted 2/2
frame 1800  boxes 114=1 boxes 0=1 class sets IDENTICAL coords shifted 1/1
frame 2400  boxes 114=2 boxes 0=2 class sets IDENTICAL coords shifted 2/2
```

Coordinates shift on **all six** frames tested; the class set differs on three of six.

**A committed claim this measurement contradicts.** Both
`tests/fixtures/retail_sanity/retail_sanity.json`'s `known_blind_spots` and the
`tools/build_sanity_fixture.py` module docstring assert that *"Filling the band with `0` or
with `255` instead of `114` still yields 2/2 here"* and that the fixture therefore does not
discriminate the pad value. **That is true for 255 and false for 0**: with `pad_value=0` the
fixture returns 1/2 and the verdict is `FAIL`. Both files are outside this task's write
scope (`tests/` and `tools/`), so the correction is **recorded here and not applied**. It
changes no M0 verdict — with the pad value actually in effect the fixture passes on all
three devices — but the blind-spot entry is wrong about half its subject and should be
fixed before anyone relies on it.

The **A/B detection-set delta tool** that would settle the pad value properly across many
frames and many scenes is **M1 work. Not built at M0.**

## Per-device compile, latency and placement

`tools/qualify_m0.py`, `--requests 100`, one explicit compile per device, one warmup
inference, then a timed loop of **100 requests**. `EXECUTION_DEVICES` is re-read from each
request's **own** compiled model after the warmup inference — it is never the device that
was asked for.

**[quoted]** — the table below is rendered by `tools/qualify_m0.render_table` from
`logs/qualification/m0.json`, which is **gitignored**. Preflight renders the same table from
the same file through the same function, so this table and preflight's embedded table are
the same table. It was **not re-measured** for this section, and the preflight run pasted
further down quotes it rather than re-timing it.

```
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

**Every row completed 100 timed requests.** The percentiles are nearest-rank over the timed
loop only; the single warmup inference is outside the timed span and is reported separately
as `warmup s`; `rps` is `requests / timed span` only. `compile` and `warmup` are timed
separately from inference and are **never added to it**.

**`rss` is the HOST process resident set, measured with `psutil` — it is NOT accelerator
memory.** No property this milestone reads exposes the NPU's own buffer. The 480.6 MiB NPU
row is the Python process holding the IR, the tensors and the runtime, and it says nothing
about how much memory the NPU is using. The figure rises monotonically down the table
(332.9 → 409.4 → 480.6 → 501.5 MiB) because the rows accumulate in one process; it is not
a per-device memory cost.

`gflops` is **deliberately absent** from this table. It was not read from the model card
and is not estimated anywhere. TOPS, power, temperature and energy are out of scope and are
**not inferred from utilization** (AGENTS.md rule 2).

### `EXECUTION_DEVICES` per device

| requested | `EXECUTION_DEVICES` read back | on the requested device? |
|---|---|---|
| CPU | `('CPU',)` | yes |
| GPU | `('GPU.0',)` | yes — the `.0` suffix is real, and `app.inference.devices.is_device` is the one accessor that handles it |
| NPU | `('NPU',)` | yes — the NPU plugin returns the **bare string**, not a list |
| CPU (fallback baseline) | `('CPU',)` | yes |

This is the only placement proof the UI may ever display. A requested device preference is
never presented as proof of execution.

## Detection sanity fixture

The annotation was drawn **by eye on frame 0 of `assets/media/retail-checkout.mp4`,
before any model was run**, in source pixels, in `tests/fixtures/retail_sanity/retail_sanity.json`:

| object | class | box `xyxy` (source px) | `min_score` |
|---|---|---|---|
| shopper at the self-checkout | `person` (0) | `[645, 0, 1268, 658]` | 0.25 |
| red-and-yellow apple in the basket, bottom-left | `apple` (47) | `[0, 878, 150, 1062]` | 0.25 |

`max_detections` is 6, chosen by looking at the picture (two named objects, the basket, the
counter and its scanner, a trolley's lower half, a second shopper's feet) and **not** by
reading the detector. Thresholds `conf 0.25 / iou 0.45 / match_iou 0.5` were fixed before
the model was run, and `tests/test_decode.py` pins that equality so they cannot drift.

An independent review verified **every box edge against pixels**: all four edges of both
boxes land on real features within a few pixels, and the annotation is **tighter than the
model's on three of four sides** of the apple (`y1 878` against the model's 915, `x2 150`
against 176). It was not adjusted to make the model agree.

| device | result | evidence |
|---|---|---|
| CPU | **PASS 2/2** | re-run **live** by `tools/preflight.py` in the run pasted below: `2 of 2 matched`, `EXECUTION_DEVICES=['CPU'] placed_on_requested=True`, `VERDICT PASS` |
| GPU | **PASS 2/2** | **[quoted]** `logs/qualification/m0_sanity_fixture.json`, quoted by preflight |
| NPU | **PASS 2/2** | **[quoted]** same file |

### The one proven blind spot: a transposed `w`/`h`

Stated plainly, because a fixture with an undocumented hole is worse than one with a
documented one.

A pure `w`/`h` swap **preserves `cx`, `cy` and `w*h` exactly** — measured: 422304.4 px²
either way for the person, 26352.4 px² either way for the apple. So **no centre-based or
area-based statistic can separate it.** The only candidate is aspect ratio against an
independent estimate of the object's true aspect, and on this annotation that check is
*inverted*:

| object | annotated | correct decode | swap mutant |
|---|---|---|---|
| person | 0.9468 | 0.9754 (Δ **0.0303**) | 1.0252 (Δ **0.0827**) |
| apple | 0.8152 | 1.1728 (Δ **0.4387**) | 0.8527 (Δ **0.0459**) |

The apple is **inverted**: the *correct* decoder is 0.4387 from the annotation and the
*mutant* only 0.0459. Any tolerance tight enough to catch the swap **fails a healthy run**;
any tolerance loose enough to pass a healthy run **passes the swap**. The cause is that the
apple's annotation is deliberately tighter than the model's, and a tight box is a poor
estimate of aspect. An aspect check was implemented, measured, and **rejected** for this
reason — a guard that cries wolf teaches people to loosen it.

**The fix is a different frame with a non-square object annotated at the same tightness the
model draws it, not a tighter threshold.** So: this fixture is a standing check over
geometry and placement. **It is not a demonstration of catching every decoder bug**, and it
does not catch a transposed `w`/`h`.

The `oven` false positive at **0.250770** on the checkout counter is visible in every
passing run and is **deliberately left unannotated** — see the annotation's own reasoning.
It is **unquantified**: M0 did not measure its frequency, its persistence or its behaviour
on other frames, and no claim is made about it beyond "it appears on this frame at this
score".

## CPU-only fallback

A **genuine second measurement, not a copy of the CPU row**: its own explicit
`compile_model("CPU")`, its own 100 timed requests, and a p50 of **24.51 ms against the CPU
row's 24.07 ms**. A fallback that reuses the primary row's numbers would have proved nothing
about the fallback path.

### What the fallback row does not prove

**The `[fallback]` marker has never fired on this machine.** `tools/qualify_m0.py` marks a
row `[fallback]` when `EXECUTION_DEVICES` does not name the device that was asked for — and
no device on this machine has ever failed, so that branch has **never executed in a real
run**. It is covered by unit tests only.

So the honest statement is: *"a CPU-only route compiles and runs the pinned model at the
pinned shape on this machine, and the fallback-detection code path is unit-tested"* — **not**
*"fallback was observed to work"*. Observed failure and recovery is **M6** failure-injection
work.

## Startup and memory

| measurement | value | how |
|---|---|---|
| `DeviceRegistry.describe()` cost, all 3 devices | **3.76 s** | timed around the **first** call in preflight's process, 2026-10-01 |
| preflight: module import → first inference completing | **4.13 s** | `time.perf_counter` in that run |
| qualification run: module import → first inference | 2.837230 s (process uptime at the same moment 9.308614 s) | **[quoted]** `m0.json` |
| host RSS per row | 332.9 / 409.4 / 480.6 / 501.5 MiB | **[quoted]** `psutil`, host process, **not** accelerator memory |

The 3.76 s is **one memoised Windows PnP enumeration** (`powershell.exe`), **not one per
device**. It is startup-only and must never reach a per-frame or per-sample path. Spec §1
sets a 30 s warm-start target, so 3.76 s is inside it.

## Telemetry: what is available, what is not, and why

### In-process PDH value read: **unavailable**. The counters are not.

`pdh.dll` loads and the required entry points bind, and the counter opens — but
`PdhGetFormattedCounterValue` and `PdhGetRawCounterValue` **terminate the process with
`0xC0000005`**, even with handles proven valid by `PdhGetCounterTimeBase`. The array read
returns 465 "valid-status" entries of 4328 with values `p50 = 10`, `max = 2.7e16` — garbage
— and `PdhExpandWildCardPathW` returns `0xC0000BBD` with 0 names, so values cannot be zipped
to instance names. Value reads are therefore refused by default
(`app.telemetry.pdh.VALUE_READS_REFUSED_REASON`).

**This is a route failure, not a counter failure.** The counters exist and do read real
percentages by a route M0 does not use. Spec §7 therefore requires the NPU card to read
**"Utilization unavailable"** — **never 0%** — and spec §10 gives the validated provider to
**M4**. Preflight rates this `WARN`, which is the correct level: a `FAIL` would tell a
presenter the NPU is broken when it compiled and completed 100 requests in front of them.

### The 62% / 68% figure: quoted, and **not reproducible**

Preflight's telemetry row cites `.superpowers/sdd/.../task-3-fix-report.md` as the evidence
for **NPU `engtype_Neural` peaking 62%** and **GPU `engtype_Compute` peaking 68%** under
real load, cross-checked against `Get-Counter` at 61.81 / 51.54.

**Two problems, both recorded rather than papered over:**

1. **`.superpowers/` is gitignored.** The cited file does not exist on a fresh clone, so
   preflight's evidence pointer is dead for anyone who has not run M0 in this working tree.
   This is the durability gap this report exists to close: the method, the commands and the
   numbers are reproduced in this document so the claim does not depend on an untracked file.
2. **The session that wrote this report could not reproduce it, and found that a
   machine-wide peak on this machine is not attributable to this application's inference
   at all.** Measured here, 2026-10-01.

**Method.** A real inference load was put on the NPU and the GPU simultaneously — the
pinned IR, a `[1,3,640,640]` blob, a continuous `infer` loop on both devices, with
`EXECUTION_DEVICES` printed at the end to prove where it ran. The GPU Engine counters were
sampled through two independent routes, **per instance**, and an **idle baseline was
measured first with no load of ours running**, because a peak with no baseline is not
evidence of anything.

```
idle baseline, 24 queries of Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine,
469 instances per query, no OpenVINO load of ours:
  engtype_Neural   peak 22   mean 3.16
  engtype_Compute  peak 89   mean 2.34
  engtype_3D       peak  0   mean 0.00

under load, 50 queries, 483 instances per query, load PID 15548, which ran
NPU=3643 and GPU=3643 inferences in 40.01 s (91.1/s each) and reported
  NPU EXECUTION_DEVICES=NPU
  GPU EXECUTION_DEVICES=['GPU.0']
  engtype_Neural   peak 31   mean 3.80
  engtype_Compute  peak 67   mean 2.23
  engtype_3D       peak  0   mean 0.00
```

`Get-Counter '\GPU Engine(*)\Utilization Percentage'` on the loaded machine: `Neural` 19.74
over 11 instances, `Compute` 45.60 over 29, `3D` 0.39 over 110.

**The finding that matters: the load's own PID never appears as a counter instance.**
`own_pid_instances_seen = 0` across 50 queries, and a per-instance peak sweep during a
further loaded run attributed the busiest engines to **other processes**:

```
 47  pid_7788  <gone>                 luid_0x00010E9E ..._engtype_Compute
 31  pid_7788  <gone>                 luid_0x000112A0 ..._engtype_Neural
 27  pid_1844  pythonw                luid_0x00010E9E ..._engtype_Compute
 25  pid_15012 pythonw                luid_0x00010E9E ..._engtype_Compute
 12  pid_15012 pythonw                luid_0x000112A0 ..._engtype_Neural
 11  pid_1844  pythonw                luid_0x000112A0 ..._engtype_Neural
was the load PID itself ever an instance?  0
```

Two unrelated `pythonw.exe` processes on this machine are the top consumers of both the GPU
Compute engine and the NPU Neural engine, and **the idle baseline already reaches 22%
Neural and 89% Compute with none of our load running at all.**

**Therefore: a machine-wide `engtype_Compute` or `engtype_Neural` peak on this machine is
not evidence of this application's accelerator utilisation, and the 62% / 68% attribution
to our load is not supported by anything measured here.** The honest position is the one
preflight already takes for the card itself: **"Utilization unavailable"**. M4 owns the
validated provider, and M4 must solve attribution — per-process or per-adapter — before any
utilization figure may appear on a screen. A number that cannot be attributed to the
workload is worse than no number.

**A methodological note for whoever re-runs this.** `Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine`
exposes exactly **two** properties on this build, `Name` and `UtilizationPercentage`. There
is **no `Engtype` property**: the engine type exists only inside the `Name` string
(`..._luid_0x000112A0_phys_0_eng_0_engtype_Neural`). A `Where-Object { $_.Engtype -eq 'Neural' }`
filter matches **nothing** and reads as "no engine activity" rather than as "that property
does not exist" — which is exactly the failure that produced an empty result during this
session's first attempt.

### The LUID→device mapping is **derived**, not identified

No CIM or Win32 source on this machine joins a LUID to a PCI function. `Win32_LUID` reports
**zero instances**; `MSFT_ComputeAccelerator` is **not a registered CIM class** on build
26200; `Win32_VideoController` carries a `PNPDeviceID` but no LUID;
`Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory` carries only `Name` and
memory counters; `EnumDisplayDevicesW` returns
`PCI\VEN_8086&DEV_64A0&SUBSYS_88F21043&REV_04` for all three display adapter entries and no
LUID.

The mapping is therefore resolved from the **engine taxonomy** and is labelled **"derived"**
throughout. It is never called "identified". Recorded in `config/telemetry_map.json` with
`direct_luid_link.available = false` and the per-LUID reasoning:

| LUID | attributed to | method | evidence |
|---|---|---|---|
| `0x00010E9E` | **GPU** | engine taxonomy | exposes `Compute, Copy, GSC, VideoDecode, VideoProcessing` |
| `0x000112A0` | **NPU** | engine taxonomy | exposes `Neural`, and no other device on this machine has a neural engine |
| `0x00011232` | **UNATTRIBUTED** | none | exposes **only** `3D` — 81 instances — which identifies neither the NPU nor the GPU |

`0x00011232` is the interesting one: it is **numerically adjacent to the NPU LUID
`0x000112A0`**. A prefix rule, a numeric-range rule, or a "it is not the GPU's, so it must
be the NPU's" rule would all have swept it onto the NPU card. It is left unattributed
because that is what the evidence supports. Engines are never summed into a single total.

The map is **machine-specific**: it was probed on 2026-10-01 against build
`10.0.26200 26200`. A driver or Windows update can invalidate the LUIDs, and
`tools/probe_telemetry.py` must be re-run under load after one — the same discipline as
resetting the NPU blob cache after a driver update. `instances_observed` in that file is a
per-run, load-dependent observation from one probe (497 at probe time; 469 and 483 in the
two windows measured here) and is **not** a hardware property.

### The other telemetry providers

| metric | provider | M0 status |
|---|---|---|
| System CPU % | `psutil.cpu_percent`, all 8 logical cores | available |
| App CPU % | process-tree sum | available in principle; the process tree does not exist until M1 |
| GPU compute % | PDH `GPU Engine`, `engtype_Compute` | **unavailable in-process** — see above |
| NPU % | PDH `GPU Engine`, `engtype_Neural` | **unavailable in-process**, and **not attributable** even by the CIM route |
| Inference completions/s | counted by us | 100 per device per row — measured |
| AI FPS | unique frames with a completed detection | needs the stream; **not run** |
| Frame age | ingress-to-display of the matching result | needs the UI; **not run** |
| Inference latency | submit to completion | measured per device above |
| TOPS / power / temperature / energy | — | **out of scope, never inferred** |

## Preflight

`tools/preflight.py` is the one-word gate. Run 2026-10-01 from the repo root, full output
pasted as produced (line-wrapped by the terminal at 200 columns; the tool's own width is
78, and the embedded qualification table is re-wrapped here by the terminal only — the
figures are the tool's):

```
==============================================================================
PREFLIGHT - Intel Edge AI Roadshow, this machine, this run
==============================================================================
repository     : C:\dev\intel-edge-ai-roadshow
openvino       : 2026.4.0-22959-99c81491cc3-releases/2026/4
python         : 3.12.10
host           : NUC14ProAI  (this is a hostname, not a SKU)
windows        : Windows-11

PASS  Assets: pinned content verification
      6 artifacts verified by content signature and sha256 against
      config/models.json: yolo11n-fp16 (xml), yolo11n-fp16:weights (binary),
      yolo11n-fp16:labels (text), retail_checkout (mp4), store_aisle_alt
      (mp4), queue_alt (mp4)

PASS  Assets: sanity fixture
      retail_sanity.json and retail_000.jpg present; frame 0 of
      assets/media/retail-checkout.mp4, 2 hand-drawn expectations,
      max_detections bound 6

PASS  Assets: model IR pin
      yolo11n-fp16 IR assets/models/yolo11n-fp16/yolo11n.xml sha256
      4845b46cc9b6ac379a44484258b7ed98f836ed117d91f127c7728ea068250a95 matches
      config/models.json; pinned input shape [1, 3, 640, 640], static and
      batch 1

PASS  Runtime: OpenVINO and Python
      openvino 2026.4.0-22959-99c81491cc3-releases/2026/4, Python 3.12.10,
      psutil 7.2.2 - all read from the loaded modules in this run. Version
      pins are enforced by tests/test_pins.py, not duplicated here.

PASS  Budget: free disk on the repo volume
      220.4 GiB free of 381.5 GiB, measured by shutil.disk_usage in this run;
      policy floor 2 GiB

PASS  Budget: free physical memory
      16.2 GiB available of 31.5 GiB across 8 logical cores, measured by
      psutil in this run; policy floor 4 GiB

PASS  Device available: CPU
      Intel(R) Core(TM) Ultra 9 288V; discovered in
      ov.Core().available_devices(), runtime
      2026.4.0-22959-99c81491cc3-releases/2026/4, driver none (no separate
      driver artefact)

PASS  Device available: GPU
      Intel(R) Arc(TM) 140V GPU (16GB) (iGPU); discovered in
      ov.Core().available_devices(), runtime
      2026.4.0-22959-99c81491cc3-releases/2026/4, driver 32.0.101.6737

PASS  Device available: NPU
      Intel(R) AI Boost; discovered in ov.Core().available_devices(), runtime
      2026.4.0-22959-99c81491cc3-releases/2026/4, driver 32.0.100.5540

PASS  Device discovery cost
      DeviceRegistry.describe() took 3.76 s in this run for all 3 devices,
      measured around the first call in this process. The cost is one memoised
      Windows PnP enumeration (powershell.exe), not one per device; it is
      startup-only and must never reach a per-frame or per-sample path. Spec
      section 1 sets a 30 s warm-start target, so 3.76 s is inside it.

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

      legend: placement is the EXECUTION_DEVICES reading, re-read from the
      request's own compiled model after the first inference - never the
      device that was asked for. It is compared through is_device, so CPU, GPU
      and NPU all match despite the inconsistent shapes the plugins return
      ('CPU', 'GPU.0', and the bare string 'NPU'). A row whose reading does
      not name its requested device is flagged in the note column: it measured
      a real latency, on the wrong hardware, and that is worth more than a
      clean-looking row. 'rss' is the HOST process resident set, not
      accelerator memory; the NPU's own buffer is not visible through any
      property this milestone reads. compile and warmup are timed separately
      from inference and are never added to it. An errored row prints
      'unavailable' in every numeric column, whatever that row carries, and
      prints its error in full.

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
      Utilization unavailable (spec section 7's wording, not 0%). The provider
      is available and validated but not yet wired into the product's runtime
      path: pdh.dll loads and the counter opens (pdh.dll loaded and the
      required entry points are bound), and the native value-read entry points
      are refused by default because they terminate this process with
      0xC0000005 even with handles proven valid by PdhGetCounterTimeBase.
      Recorded evidence that the counters do exist and are measurable by a
      route M0 does not use: NPU engtype_Neural and GPU engtype_Compute read
      real percentages through
      Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine and Get-Counter
      under real OpenVINO load (Task 3, see
      .superpowers/sdd/2026-10-01-m0-hardware-and-model-
      qualification/task-3-fix-report.md). A route failure is not a counter
      failure, so this is a WARN and not a FAIL: the NPU compiled and
      completed its requests. Spec section 10 gives the validated provider to
      M4.

PASS  Telemetry: recorded LUID-to-device map
      2 rules loaded from config/telemetry_map.json, which load refuses on a
      collision: 0x00010E9E->GPU/3D, 0x000112A0->NPU/Neural. NPU rules: 1
      (engtype Neural).

WARN  Telemetry: LUID-to-device attribution and hardware identity
      - the LUID-to-device mapping is DERIVED from the engine taxonomy, not a
      driver-exposed identity: no CIM or Win32 source on this machine joins a
      LUID to a PCI function. Preflight says 'derived' and never 'identified'.
      - adapter LUID 0x00011232 is recorded UNATTRIBUTED (81 instances, engine
      types ['3D']): exposes only ['3D'], which identifies neither the NPU nor
      the GPU; left unattributed rather than assigned to the nearest device.
      - three hardware-identity records disagree and none is treated as the
      winner: Win32_ComputerSystem 'ASUSTeK COMPUTER INC. NUC14LNK LN';
      hostname 'NUC14PROAI'; PCI subsystem ['SUBSYS_88F21043']. Preflight
      claims no SKU. Reconciling them is the M0 report's job, not a
      measurement this gate can make.
      - config/telemetry_map.json's instances_observed=497 and its
      max_observed_utilization_percent are per-run, load-dependent
      observations from one probe, not hardware properties. They are
      deliberately not quoted here as a property of this machine.
      - the map was probed on 2026-10-01 against Windows build Microsoft
      Windows 11 Pro 10.0.26200 26200. It is machine-specific: a driver or
      Windows update can invalidate the LUIDs, and tools/probe_telemetry.py
      must be re-run under load. That is the same discipline as the blob-cache
      reset after a driver update.

PASS  Startup: this gate's import to the first inference
      4.13 s, measured with time.perf_counter in this run from this module's
      import to the first inference completing. Includes the pinned PowerShell
      driver enumeration inside DeviceRegistry.describe() and the IR read; the
      fixture rows above hold their own per-row figures.

device controls a presenter will be offered:
  CPU  enabled  Intel(R) Core(TM) Ultra 9 288V (driver no separate driver artefact)
  GPU  enabled  Intel(R) Arc(TM) 140V GPU (16GB) (iGPU) (driver 32.0.101.6737)
  NPU  enabled  Intel(R) AI Boost (driver 32.0.100.5540)

Summary: 16 PASS, 2 WARN, 0 FAIL - 18 total rows
Status: Limited
Exit code: 0. Ready and Limited both exit 0, because in both the demo runs; only Needs attention exits 1.
```

`Status: Limited` is the honest verdict for this machine, and the two `WARN` rows are named
above: in-process PDH utilization is unavailable, and the LUID attribution and hardware
identity are weaker than a presenter might assume. Neither is a failure. The
`WARN`-versus-`FAIL` distinction is computed from the measurement by `run_checks`, never
supplied by a caller.

## The rest of the gate set

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

$ .venv\Scripts\python.exe -m unittest discover -s tests
----------------------------------------------------------------------
Ran 449 tests in 11.134s

OK
```

```
$ .venv\Scripts\python.exe -m app.main --selftest
```

**not run — `app.main` arrives in M1.**

There is no application to run at M0. `app/` contains contracts, `inference/` and
`telemetry/` only: no `main.py`, no `theme.py`, no QML, no workers, no supervisor. The
"then run the actual demo for the duration the milestone requires" line of the working rules
has **no M0 equivalent**, and this document does not substitute a tool run for a demo.

---

# Unmet gates and open blockers

Every row here is a thing M0 did **not** establish. None of them is a surprise, and none is
worked around.

## Unmet gates

| # | gate | status | why it is open |
|---|---|---|---|
| 1 | **NPU utilization counter in-process** | **WARN, unresolved** | `PdhGetFormattedCounterValue` / `PdhGetRawCounterValue` terminate the process with `0xC0000005`. The card reads "Utilization unavailable". The validated provider belongs to **M4**. |
| 2 | **Utilization attribution** | **not established, new** | Even by the CIM route, the busiest `engtype_Neural` and `engtype_Compute` instances on this machine belong to **other processes**, and the inference process's own PID never appears as a counter instance. The idle baseline already reaches 22% / 89% with none of our load running. **No utilization figure may be shown until M4 can attribute it.** |
| 3 | **Three 45–90 s clips per pack** | **unmet M5 content gate** | 1 of 12 required clips is in band (65.41535 s). `retail_checkout` is 113.0129 s, `queue_alt` 139.4 s. Three packs have **no clips at all**. See [M0-content-gate.md](M0-content-gate.md). |
| 4 | **Shelf-to-basket footage** | **unmet; reduced scope committed** | No such clip exists in the provisioned material; answered by looking at 72 decoded frames. "Items entering the checkout zone" is committed as an explicitly labelled intermediate milestone and the requested scenario is left **incomplete, not claimed**. |
| 5 | **Clip redistribution clearance** | **not established** | No clip is upgraded to "cleared". The Pexels terms were not re-read at M0; the two Open Model Zoo clips have **no licence text in the record at all**. |
| 6 | **Detector licence** | **AGPL-3.0, unresolved** | Read from the model card; the source manifest recorded no licence. Copyleft is a distribution question for the M6 booth package. An operator/legal decision, recorded not resolved. |
| 7 | **Code signing** | **unmet M6 gate** | No certificate is available. **M6 ships unsigned**, with the signing step documented. |
| 8 | **1920x1080 DPI gate** | **not run — M3** | Usable height on this display is 960 px, so a 1920x1080 window cannot be shown. The gate needs the offscreen render harness (`tools/capture_ui.py`), which does not exist yet. |
| 9 | **Fallback observed under failure** | **not run** | The `[fallback]` marker has never fired; no device has ever failed. Unit-tested only. M6 failure injection. |
| 10 | **AC status** | **unavailable, permanently on this machine** | `Win32_Battery` returns 0 instances. The spec's "record AC status" row is satisfied by stating it is unavailable and recording the Balanced power scheme instead. Thermal readings are likewise not inferred. |
| 11 | **Hardware SKU** | **unreconciled** | Three disagreeing identity records; none named the winner. See *Environment*. |

## Open blockers and corrections

| # | item | why it matters |
|---|---|---|
| A | **The pad blind-spot record is wrong about `pad_value=0`.** Both `tests/fixtures/retail_sanity/retail_sanity.json` and the `tools/build_sanity_fixture.py` docstring say both pad mutants still return 2/2. Measured here: **255 does, 0 does not** — pad 0 returns 1/2 and the fixture **FAILS**, with the apple absent from the frame. | The blind-spot list is what a future reader trusts when deciding what the fixture does and does not protect. It is currently wrong about half its subject. **Outside this task's write scope (`tests/`, `tools/`), so recorded and not applied.** |
| B | **Preflight's telemetry row cites a gitignored file** for the 62% / 68% claim. | On a fresh clone the pointer is dead and the claim is unsourced. This report reproduces the method, the commands and the numbers, and records that the attribution is not supported. |
| C | **`assets/scenarios/retail_sanity/manifest.json` says the fixture "does NOT exist yet… created in Task 6".** It exists. | A stale forward reference in a rights/asset manifest. Recorded; not corrected here. |
| D | **The `oven` false positive at 0.250770 is unquantified.** | It appears on the fixture frame and is deliberately unannotated. Its frequency, persistence and behaviour on other frames are **not measured**, and no precision/recall claim is made for it. |
| E | **The `w`/`h` blind spot is open.** | A transposed `w`/`h` passes this fixture. Needs a different frame, not a tighter threshold. |
| F | **No INT8 artifact, and none planned for V1.** | Listed in the matrix as `not tested` so its absence does not read as an oversight. |
| G | **The three other scenario packs do not exist.** | No ids, no manifests, no ROIs, no event state machines, no media. The matrix records their cells `not tested` and does not invent pack names. |

## What M0 does establish

- This exact machine runs one pinned, hash-verified OpenVINO IR on **all three** of its
  real accelerators, 100 timed requests each, with `EXECUTION_DEVICES` proving placement on
  the requested device every time.
- The decoder produces **correct detections on all three devices** against an annotation
  somebody drew by eye before the model was run.
- The gate reads **`Limited`**, exit 0, with both warnings named — and neither warning is
  a broken accelerator.
- The model, the media, the decoder, the letterbox inverse, the fixture and the preflight
  gate are all in place, verified, and reproducible from a clean checkout with one command.

