# M1 — One Truthful Retail Stream: qualification report

Date: 2026-10-02
Machine: the CPU identifies itself as `Intel(R) Core(TM) Ultra 9 288V`; the accelerators are
`Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)` (driver 32.0.101.6737) and `Intel(R) AI Boost`
(driver 32.0.100.5540); Windows 11 Pro build 26200. The three hardware-identity records still
disagree and **no SKU is claimed** — see
[M0-qualification.md](M0-qualification.md#environment), which M1 does not contradict or reopen.

> **Read this first.** M1 built a running application and measured it. It did **not** close
> the M0 gate, and it did not close all of its own. Two things a reader might assume from the
> word "M1" are false and are stated in the same sentence throughout this document:
>
> 1. **The GPU and NPU utilisation counters are still unavailable**, and a machine-wide peak
>    is still not attributable to this process. No utilisation figure appears anywhere in
>    the product. That is the M0 finding, unchanged.
> 2. **The retail event is people-in-zone presence, not a product transfer.** Shelf-to-basket
>    is not delivered, and the item-transfer reading of "checkout zone" was measured and found
>    unsupported.

## What M1 added

| layer | what exists now | where |
|---|---|---|
| frame identity | `FramePacket` — session/scenario/stream/frame/loop ids, monotonic capture and decode times, `age_at` measured from capture | `app/contracts/frame.py` |
| zones | normalised ROI geometry, `contains`, `centroid_of`, `to_pixels`, order and range validation | `app/engine/zones.py` |
| tracking | IoU tracker, class-stable ids, `max_misses` coasting, `reset()` that clears the id counter so a loop cannot inherit one | `app/engine/tracker.py` |
| events | the retail state machine: presence dwell + dedup key, and the transfer/return/occlusion/loop-reset path | `app/engine/events.py` |
| source & queue | `ClipSource` with loop ids and pacing to the analysis rate; `BoundedFrameQueue` that counts every refusal | `app/engine/source.py`, `app/engine/queue.py` |
| inference | spawned worker, static 640x640 batch 1, `EXECUTION_DEVICES` re-read **after** the inference, failure returned as an error | `app/engine/worker.py` |
| telemetry | background collector → `MetricSample`s; `null` is not `0`; `unavailable` is a word | `app/telemetry/collector.py`, `app/contracts/metric_sample.py` |
| routing | `RoutePlan` with a generation, requested and confirmed kept apart; one automatic worker restart | `app/supervisor/runner.py`, `app/contracts/route_plan.py` |
| UI | the QML booth screen, the "Recorded footage • live inference" label, three compute cards, `unavailable` rendering | `app/ui/`, `app/theme.py` |
| entry point | `python -m app.main` with `--selftest`, `--headless`, `--windowed`, and a release-mode refusal for a simulated provider | `app/main.py` |

Nothing here is a metric value in UI code: layout tokens live in `app/theme.py`, and
`tests/test_ui_contract.py` fails if a QML file hardcodes a colour, a pixel size, a threshold
or the input size.

---

## The gate set, run in this session

Four commands, run from the repo root on 2026-10-02. Output is pasted as produced.

### 1. `tools\verify_sources.py` — 6 PASS, 0 FAIL

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

### 2. `tools\preflight.py` — 16 PASS, 2 WARN, 0 FAIL, exit 0, `Status: Limited`

The rows that carry M1's own headline numbers, verbatim:

```
PASS  Sanity fixture on CPU
      a real compile, a real inference and a real decode on CPU at conf=0.25
      iou=0.45 match_iou=0.5, compared against the hand-drawn annotation: 2 of
      2 matched. EXECUTION_DEVICES=['CPU'] placed_on_requested=True. VERDICT
      PASS: every annotated object matched, the detection count is within the
      annotation's bound, and the run was placed on the requested device.
      KNOWN BLIND SPOT, not checked here: a transposed w/h. See the module
      docstring - it is not separable on this fixture by any measurement.

WARN  Telemetry: in-process PDH GPU/NPU utilisation
      Utilization unavailable (spec section 7's wording, not 0%). The provider
      is available and validated but not yet wired into the product's runtime
      path: pdh.dll loads and the counter opens (pdh.dll loaded and the
      required entry points are bound), and the native value-read entry points
      are refused by default because they terminate this process with
      0xC0000005 even with handles proven valid by PdhGetCounterTimeBase. The
      counters themselves DO exist and read real percentages on this machine
      by the CIM/Get-Counter route
      (Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine and Get-
      Counter), but a machine-wide peak there is NOT attributable to this
      process: this process's PID never appears as a counter instance, and
      unrelated processes hold the busiest engines. No utilisation figure may be
      shown until attribution is solved. Measured evidence and the full method
      are in the committed report
      docs/qualification/M0-qualification.md (Telemetry), section 'Unmet
      gates' item 2. A route failure is not a counter failure, so this is a
      WARN and not a FAIL: the NPU compiled and completed its requests. Spec
      section 10 gives the validated provider to M4.

PASS  Device discovery cost
      DeviceRegistry.describe() took 3.31 s in this run for all 3 devices,
      measured around the first call in this process. The cost is one memoised
      Windows PnP enumeration (powershell.exe), not one per device; it is
      startup-only and must never reach a per-frame or per-sample path. Spec
      section 1 sets a 30 s warm-start target, so 3.31 s is inside it.

PASS  Startup: this gate's import to the first inference
      3.67 s, measured with time.perf_counter in this run from this module's
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

**The two `WARN`s are the pre-existing M0 telemetry rows, unchanged.** Neither is new at M1 and
neither is closed. The preflight row count is the same 18 as at M0: M1 added no row and removed
none, because the honest state of the machine did not change.

### 3. `python -m app.main --selftest` — PASS, exit 0

```
  ok   preprocess           letterbox 640x640 pad=114 scaled=640x360
  ok   infer                25.38 ms on confirmed ['CPU']
  ok   decode-track-event   1 zone_occupancy event(s) from a synthetic presence
  ok   render-metadata      unavailable renders as unavailable, and no device is claimed before an inference confirms one
PASS
```

No network, no display, no GUI application object. Note the third line: the self-test's event
comes from a **synthetic presence**, not from the footage. It is a wiring check and is not
evidence of a retail event; the footage evidence is in *The business events actually emitted*.

### 4. `python -m unittest discover -s tests` — 1034 tests, OK, exit 0

```
----------------------------------------------------------------------
Ran 1034 tests in 145.586s

OK
```

The stderr this run produced is worth naming rather than hiding: the supervisor's
`compile_failure=True` test seam prints a `RuntimeError: COMPILE FAILURE INJECTED BY THE
SUPERVISOR'S TEST SEAM` traceback from the child process, `argparse` prints a usage block when
the CLI is handed `--device TPU`, and `tools/verify_sources.py` prints `FAIL` rows inside
`tests/test_verify_sources.py` that assert a bad artifact is refused. All of it is expected
output from tests that are deliberately making something fail. The suite result is `OK`.

---

## The application runs — the live scene graph

`app.main.run_windowed` was driven for real, offscreen, through its own `_start_feeders` thread,
its own `_start_telemetry` timer and the real `Main.qml`, and the objects it built were then
read **out of the live QML scene graph**. This is the measurement behind the following table;
it is the driver `tests/test_main.py::LiveWindowedPath` uses, run directly so the numbers could
be quoted rather than merely asserted.

| what was read | value |
|---|---|
| exit code | 0 |
| `bridge.device_confirmed` | **`CPU`** |
| `bridge.device_fell_back` | `false` |
| the rendered badge text (`deviceBadgeText`) | **`device: CPU`** |
| `frame_ready` / image null | `true` / **`false`** |
| frame image size | **1920 x 1080** |
| zones published from the manifest | **1**, `counter_band`, `roi (0.0, 0.20, 1.00, 0.60)` → `1920 x 432 px` |
| overlay boxes published | **3** (track 1 person, track 6 apple, track 54 person) |
| frames submitted / analysed / refused | 210 / **209** / **0** |
| `event_counts` | **`{"zone_occupancy": 1}`**, equal to the event machine's own counts |
| rendered event row | `zone_occupancy 1    candidate_transfer 0    candidate_return 0` |
| `system_cpu_percent` rendered | **`53.7%`** (a second run of the same driver: `56.4%`) |
| `ai_fps` | **`14.50 fps`** |
| frame age p50 / p95 | **`58.7 ms` / `712.5 ms`** |
| `gpu_utilisation_percent` / `npu_utilisation_percent` rendered | **`unavailable`** / **`unavailable`** |
| `cpu_available` | **`true`** |
| rendered per-card "no figure" notice | CPU **`no CPU sample yet`**, GPU and NPU **`Utilization unavailable`** |
| rendered window line | `measured over 16 ms`, **visible** |
| rendered staleness line | `telemetry fresh` |
| supervisor errors | **none** (empty) |

Conditions, so the numbers can be compared rather than admired: `QT_QPA_PLATFORM=offscreen`,
`--device CPU`, `ROADSHOW_LIVE_SECONDS` default **14 s**, one 1920x1080 stream at the 15 fps
analysis rate over the pinned `retail_checkout` clip, this machine.

**`gpu` and `npu` read `unavailable` and carry the reason, and that is the M0 finding, not a
regression.** `psutil` has no counter for a GPU or an NPU at all, `app.telemetry.pdh`'s native
value reads terminate this process with `0xC0000005`, and a machine-wide peak is not
attributable to this process. The cards therefore say "Utilization unavailable" and show
inference activity instead. **No utilisation percentage appears anywhere in the product**, and
`tests/test_ui_contract.py::UtilisationHonesty` fails if one is introduced, including in the
QML source text.

**The CPU card must not borrow that wording, and does not.** There is no counter to fail for a
CPU: `psutil.cpu_percent` is called every 250 ms and works here. Its absent state is the weaker
and different claim, "no CPU sample yet".

### The duration was not the one the plan's example names

The plan's Step 1 shows `--seconds 60` per device. **That was not run at M1.** What exists is:
two 14 s windowed runs and three 8 s headless feeds (committed under
[`m1-headless/`](m1-headless/README.md)). This report does not present an 8 s or 14 s figure as
a 60 s soak, and `bench_report.md` says the same. **A 60-second unattended run per device is
M6 soak work and is not claimed.**

---

## Routing evidence — `requested` versus `confirmed`

Placement is read from `EXECUTION_DEVICES` on the **compiled model, after the first
inference**, and never from the requested device. The property is inconsistent across plugins
(`['CPU']`, `['GPU.0']`, and the bare string `'NPU'`), so every comparison in the product goes
through `app.inference.devices.is_device`, which handles all three shapes.

### From the headless stream runs — committed, at `docs/qualification/m1-headless/`

These are the exact summaries `python -m app.main --headless` wrote on 2026-10-02, committed
so the table can be checked against the tree rather than against a scrollback buffer.
Conditions are in [`m1-headless/README.md`](m1-headless/README.md): 8 s feed window, 2 s drain,
`QT_QPA_PLATFORM=offscreen`, the integrated Arc 140V sharing memory with the NPU.

| file | requested | **confirmed (`EXECUTION_DEVICES`)** | state | `fell_back` | errors | frames analysed / submitted / refused / drops |
|---|---|---|---|---|---|---|
| `cpu.json` | CPU | **`CPU`** | active | `false` | `[]` | 121 / 121 / 0 / 0 |
| `gpu.json` | GPU | **`GPU.0`** | active | `false` | `[]` | 121 / 121 / 0 / 0 |
| `npu.json` | NPU | **`NPU`** | active | `false` | `[]` | 121 / 121 / 0 / 0 |

**Every device was placed where it was asked to go, and the UI's device is the confirmed one.**
`GPU.0` is quoted verbatim rather than normalised to `GPU`: the confirmed reading is the
evidence, and a friendlier name would be a second claim.

### Re-measured for this report — 30 timed inferences per repeat, 4 repeats

Single frame (frame 0 of `retail_checkout`), letterboxed to 640x640, **batch 1**, static shape,
one warm-up inference outside the timed span, `conf 0.25 / iou 0.45`, pad 114. Run in this
session on 2026-10-02:

| device | compile | `EXECUTION_DEVICES` after warm-up | `is_device(dev)` | p50 per repeat (ms) |
|---|---|---|---|---|
| CPU | 0.19 s | `['CPU']` | `True` | 22.53, 22.64, 22.54, 22.37 |
| GPU | 0.73 s | `['GPU.0']` | `True` | 3.75, 3.68, 3.43, 3.54 |
| NPU | 1.62 s | `NPU` (bare string) | `True` | 5.78, 4.56, 4.48, 4.33 |

**The NPU is slower than the GPU on this machine.** This is a measured result, not a defect and
not a placeholder. It inverts the shape a booth story usually has, and the product's own
explanation text says so (`app/ui/bridge.py`, the `latency` explanation, pinned by
`tests/test_ui_contract.py::test_the_gpu_copy_states_the_measurement_conditions`).

**Conditions travel with the claim, everywhere it appears.** Single frame. 640x640. Batch 1. 30
inferences per repeat. An **integrated** Arc 140V sharing system memory with the NPU. **This is
not a throughput claim.** A single frame at batch 1 says nothing about sustained frames per
second with the decoder and the transport in the loop, and a throughput comparison needs M4's
paired trials.

These figures are close to, and not identical with, the numbers recorded at Task 10 in
`app/engine/worker.py` and [`m1-headless/README.md`](m1-headless/README.md) (CPU 20.99–21.16 ms,
GPU 3.30–3.47 ms, NPU 5.92–5.99 ms). Both were measured on this machine by the same method;
they differ because the machine's state differs between runs. The **ordering** — NPU slower than
GPU — held in both.

---

## Frame age and inference cost are two different claims

They are never merged, and this section exists because merging them is the easy mistake.

| quantity | what it actually measures | CPU | GPU | NPU |
|---|---|---|---|---|
| **Inference cost**, p50, single frame, batch 1 | submit → complete inside the worker | **22.37–22.64 ms** | **3.43–3.75 ms** | **4.33–5.78 ms** |
| **Frame age**, p50, in the 8 s headless feed | ingress → result available, **including queue wait behind a 15 fps feed** | **58.06 ms** | **45.87 ms** | **49.58 ms** |
| **Frame age**, p95, same feed | as above | **640.25 ms** | **1001.20 ms** | **2015.53 ms** |
| `ai_fps` (frames analysed / feed window) | analysis throughput at the 15 fps target | 15.11 | 15.11 | 15.10 |

Sources: the cost row is the measurement in *Routing evidence* above; the frame-age and
`ai_fps` rows are quoted from `docs/qualification/m1-headless/{cpu,gpu,npu}.json`, which are
committed.

**The NPU's p95 frame age is about 2.0 s while its inference cost is about 4–6 ms.** The gap is
queue wait behind the feed, not compute, and quoting either number without the other would
misdescribe the system. All three devices sit at the 15 fps `ai_fps` target because the feed is
the limit, which is precisely why the per-frame cost table and the frame-age table are separate
tables.

`ai_fps` divides by **`feed_elapsed_s`**, the window in which frames were actually fed — not by
`elapsed_s`, which additionally contains the drain and the teardown. For `cpu.json`:
`frames_analysed 121 / feed_elapsed_s 8.006 = 15.11`, and
`elapsed_s 10.139 = feed 8.006 + drain 2.013`. `drain_elapsed_s` is its own field so the
difference is visible rather than absorbed.

---

## The business events actually emitted

**The delivered event is people-in-zone presence. It is not a product transfer, not a people
count, and not a dwell measurement.**

`COUNTER_BAND = (0.00, 0.20, 1.00, 0.60)`, `MIN_PRESENCE_SECONDS = 2.0`,
`tracked_classes = ("person",)` — the only tracked class is the person. Presence is tested on
the tracker's **centroid**, and the counter/scanner surface is excluded from event logic because
its label is unstable (dominant label changes on **228 of 1792** adjacent frame pairs, 12.7%;
the surface is labelled **17** different ways across the clip). The band captures the sustained
presence at the counter and is **not** "where people are": measured, person detections with the
centroid inside the band are **3512 of 5393 (65.1%)**.

### The event, over the whole clip

Re-measured for this report by running the shipped configuration through the real pipeline over
the real clip — `preprocess_bgr` → `infer` → `decode_detections` → `Tracker` →
`RetailEventMachine` — at **stride 5** (678 of 3387 frames, `grab`+`retrieve`), CPU:

```
sampled_frames 678  detections 1960  abandoned 0  skipped_unmapped_class 0
events 2
  kind=zone_occupancy class=person track=1  loop=0 frame=66  t=2.202  dedup=0:1:person:presence:counter_band  uncertain=False
  kind=zone_occupancy class=person track=41 loop=0 frame=551 t=18.385  dedup=0:41:person:presence:counter_band uncertain=False
```

**Two sustained presences, one per person-track.** That is the whole claim. Each event carries a
`dedup_key` of the form `{loop}:{track}:{class}:{kind}:{zone}`, is stamped with the frame it
came from and that frame's capture time, and is never `uncertain` — an event this machine emits
has an unbroken confirmed track behind it.

In the 8 s product-path headless runs the count is **1** event, `0:1:person:presence:counter_band`
at frame 66, identically on all three devices. That is the same event seen over a shorter
window, not a different measurement.

### Why it is not more than this

Three measurements, all committed, all re-quoted here rather than re-derived:

- **No product-class zone transition fires on `retail_checkout` at the pinned operating point.**
  The full-clip run under M0's zone rectangles produced **0 events**.
- **The only reliably tracked product is stationary for the whole clip.** The apple — the one
  hand-annotated object in the repository, `[0, 878, 150, 1062]` on frame 0 — sits in the basket
  throughout, **max corner deviation 0.0006 over 483 frames**.
- **The person does not traverse.** 17.441 frame-widths of cumulative travel against 0.015 of
  net displacement, pacing behind the counter.

So a traversal-shaped event would fire once per 113-second clip and read like a business event
without being one. The pack therefore declares **no** `origin_slot`/`destination_slot`: naming a
transfer route it does not use would be a claim about the footage that the footage does not
support.

**A guard makes this unrepeatable.** `tests/test_retail_footage_guard.py` runs the real pipeline
over the real clip and **fails if a shipped configuration emits zero events**, and a second test
fails if a manifest builder added anywhere in `app/engine/` is not pointed at by the guard. A
zero-event configuration is a failure, not a pass and not a skip. (A skip, when the IR or the clip
is absent, prints "A skip is not a pass" and says the configuration has **not** been verified on
this machine.)

### The wording the screen shows

`app/engine/retail_config.py`'s `scope_note`, verbatim, and the screen reads it from the pack
rather than from a hand-written summary:

> REDUCED SCOPE. No product-transfer event is detectable on retail_checkout: the clip contains
> no shelf, and the only sustained movement is a person at the counter. Shelf-to-basket is NOT
> delivered. The counter surface is labelled 17 different ways and its label changes on 12.7% of
> adjacent frame pairs; it is excluded from tracked_classes. COUNTER_BAND captures 65% of person
> detections, not all of them. See app/engine/retail_config.py.

`tests/test_ui_contract.py` pins that the on-screen copy is the pack's own words and that it
**does not claim a dwell measurement** — because `BusinessEvent` carries no duration field, so a
dwell number could not be honest even if the screen wanted one.

---

## The visible error / fallback state — built, and what is *not* observed

M1 **built** the visible failure state and the refusal accounting. It did **not** observe an
accelerator fail.

| what | status | evidence |
|---|---|---|
| a route whose confirmation does not name the requested device renders as a **fallback**, not as the target | demonstrated | `tests/test_ui_contract.py`: `set_route(requested="NPU", confirmed=("CPU",))` → `device_fell_back` `True`, `device_display` contains `fell back from NPU` |
| a **failed** route claims no device at all | demonstrated | `set_route(requested="NPU", confirmed=None, state="failed")` → `device_confirmed` is `None`, `device_display` contains `no device is running` |
| a dead worker is detected, **restarted exactly once**, and a second death sets `failed` with a stated reason | demonstrated | `tests/test_runner.py`; `restart_count` asserted as exactly `1`, and the route asserted `failed` — a supervisor that restarted in a loop would never reach that state |
| the failure path is built, not asserted into existence | demonstrated | `compile_failure=True` makes the **child process really raise** before it compiles; the supervisor sees a dead worker, not a mock |
| **an accelerator actually failing mid-run and the application falling back to CPU** | **NOT OBSERVED** | no device on this machine has ever failed. M0's unmet gate 9 stands. |

**M0's `[fallback]` marker has never fired on this machine**, and M1 did not change that. What
changed is that the state is now *visible and tested* rather than merely possible.

M1's CPU-only run is a **measured CPU-only run**, which is a different and weaker claim than
"fallback was observed to work". No accelerator was made to fail. Failure injection that
exercises the real route-change path is **M6** work and M2's route-change matrix; this report
does not claim it closed.

### Refusal accounting: every attempt is accounted for

`frames_submitted`, `frames_attempts`, `frames_analysed`, `frames_refused` and
`frames_refused_reasons` are separate fields, and `drops` is the queue's own counter. A frame
that could not be placed is **counted with its reason**, never silently dropped. A run that
analysed nothing exits non-zero rather than reporting a clean zero.

| run | submitted | attempted | analysed | refused | refusal reasons | queue drops |
|---|---|---|---|---|---|---|
| live windowed, CPU (14 s) | 210 | — | 209 | **0** | `{}` | — |
| `cpu.json` | 121 | 121 | 121 | **0** | `{}` | 0 |
| `gpu.json` | 121 | 121 | 121 | **0** | `{}` | 0 |
| `npu.json` | 121 | 121 | 121 | **0** | `{}` | 0 |

---

## Input acknowledgement — a floor, not a bound

Plan review failure mode 6: the UI thread must not block on inference. A `QTimer` on the **Qt
main thread** fires every 20 ms and records **lateness** — the gap between when the tick was due
and when its slot ran. That is what a person at the booth experiences as "the screen did not
respond".

Measured through `app.main.run_windowed` itself — the application's own feeder thread, telemetry
timer, reader thread and result fold — on the offscreen platform, one 1920x1080 stream at 15 fps,
with **6 CPU-only saturator processes** plus real CPU inference. All three repeats in one
process, because startup (`DeviceRegistry.describe()` ~3.3 s cold, the CPU compile, the stop
grace) is a fixed cost and one repeat per process measures it three times.

Re-measured for this report, 2026-10-02:

| repeat | p50 | p95 | worst tick | ticks | frames analysed | requested / confirmed |
|---|---|---|---|---|---|---|
| 0 | 0.031 ms | 16.582 ms | 37.525 ms | 292 | 52 | CPU / `['CPU']` |
| 1 | 0.041 ms | 10.114 ms | 21.622 ms | 294 | 102 | CPU / `['CPU']` |
| 2 | 0.074 ms | 11.463 ms | 28.995 ms | 293 | 151 | CPU / `['CPU']` |

Against the 200 ms gate: every p95 and every worst tick passes, with the worst tick 7× under the
2 s bound the gate also applies. The suite's own gate is two-sided — a p95 alone hid a
**2981.8 ms** freeze in an earlier version — so both are asserted.

**Three caveats, and the figure is weaker with each of them:**

1. **This is a floor, not a bound.** It is measured offscreen, on one stream, on this machine,
   with synthetic saturators. A booth presenting the same screen through a real compositor on a
   real display will not be faster. The honest claim is "the UI thread does not block on
   inference"; this is the measurement of that, not a guarantee.
2. **It is not comparable across machines.** The saturator count is
   `max(2, cpu_count() - 2)` — **6 of 8** logical cores here, and **2 of 2** on a two-core box.
   The same gate under a much heavier load is a different measurement.
3. **At 8 of 8 saturated cores the same gate measured p95 195.6 ms and max 1105.5 ms** — a
   **greater-than-one-second freeze that still reports green** on a p95. That is recorded in
   `tests/test_main.py::InputAcknowledgement`'s docstring. **A booth machine must be provisioned
   so inference never sees 8/8 saturation**, and this report does not claim the gate holds under
   that condition.

---

## The metrics shown, and each one's numerator, denominator, window, provider and scope

Every figure on a card routes through `UiBridge`, which formats from the **record** and never
divides. `null` renders as `unavailable`; a genuine `0` renders as a number. Values below are
quoted from the committed `docs/qualification/m1-headless/cpu.json`, so each one's own
`provider`, `validity` and `window_seconds` field is visible.

| metric | numerator | denominator | window | provider | scope | M1 status |
|---|---|---|---|---|---|---|
| `system_cpu_percent` | CPU time across all 8 logical cores | 8 logical cores × window | the recorded delta, `0.062 s` in `cpu.json` | `psutil` | **machine-wide** — the whole system, not this process | available; rendered `53.7%` in the live run |
| `app_cpu_percent` | sum of CPU time over the tracked PIDs | 8 logical cores × window | recorded delta | `psutil` | `("ui", "worker")` — the process tree | available; **a measured `0.0` in `cpu.json` and `gpu.json`, `0.8875` in `npu.json`** — a real zero, not a null |
| `ai_fps` | frames with a completed detection | **`feed_elapsed_s`**, not `elapsed_s` | the feed window | counted by the product | this run, this stream | available; 15.11 / 15.11 / 15.10 |
| `frame_age_p50_ms`, `frame_age_p95_ms` | ingress → result age | 121 samples, nearest rank, **no interpolation** | the run | counted by the product | this stream | available; see the table above |
| `gpu_utilisation_percent` | — | — | — | none that works in-process | adapter | **`unavailable`**, with the reason |
| `npu_utilisation_percent` | — | — | — | none that works in-process | adapter | **`unavailable`**, with the reason |
| TOPS / power / temperature / energy | — | — | — | — | — | **out of scope, never inferred from utilisation** |
| `app_rss_bytes`, `rss:ui`, `rss:worker` | resident set | — | recorded delta | `psutil` | **host process only** | available; **not accelerator memory** |

Two things a reader would otherwise get wrong:

- **`window_seconds` is a measured delta, not a designed interval.** `0.062 s` in `cpu.json`
  and `0.219 s` in `npu.json` are what elapsed between samples. A window is never divided by; it
  is formatted from the record.
- **A `0.0` in `cpu.json`'s `app_cpu_percent` is a genuine measurement**, rendered as a number.
  The distinction from `unavailable` is the whole point of the metrics contract, and a
  `MetricSample` cannot hold both readings at once.

---

## "Recorded footage • live inference"

`RECORDED_FOOTAGE_LABEL = "Recorded footage • live inference"` (`app/ui/bridge.py`), rendered
as a persistent label on the booth screen. Spec §4 requires it to be displayed continuously; its
absence was a real gap in a previous build, so it is **pinned on the QML source** —
`tests/test_ui_contract.py` fails if either half of the string disappears from `app/ui/*.qml`,
and a further test asserts the label constant and the QML source agree.

### The screenshot path: not established

**There is no committed screenshot of the booth screen, and this report does not point at one.**
The offscreen render harness (`tests/test_ui_contract.py::OffscreenRender`) writes a 1920x1080
PNG into a **temporary directory that is deleted** when the class finishes. It asserts content —
the label's glyphs are on the screen, the frame's green cast and cyan block reached the pixels,
exactly one track box is drawn and it is the current one, the metric column is present, the
render is not one flat colour — but it keeps nothing.

The committed standalone screenshot tooling the plan's Step 2 asks for
(`tools/capture_ui.py`) **does not exist**; it is M3 work. **Not run — M3.**

**The live booth runs at this machine's 2293x960 desktop, not on a 1920x1080 canvas.** M0
measured usable height at **960 px**, so a 1920x1080 window cannot be displayed here at all. A
passing 1920x1080 offscreen render is **not** a claim about this monitor, and the 1920x1080 DPI
gate remains **M3**.

---

## M1 exit criteria — demonstrated, and not demonstrated

Stated criterion by criterion, with what was actually run. **No criterion is claimed that was not
demonstrated.**

| # | criterion | verdict | evidence |
|---|---|---|---|
| 1 | one retail clip runs end to end with **real** detection overlays and no precomputed detections | **DEMONSTRATED** | the live run above: 209 frames analysed, 3 overlay boxes, real pixels at 1920x1080. Every frame is decoded from the pinned IR on the CPU in this process; nothing is precomputed. |
| 2 | an explicit CPU/GPU/NPU selector exists, and the **selected and actual devices match**, proven by `EXECUTION_DEVICES` re-read after the first inference | **DEMONSTRATED for placement.** **NOT DEMONSTRATED as an interactive act.** | placement is demonstrated three times over: `cpu.json` / `gpu.json` / `npu.json` with `fell_back: false` on every row, and the latency table re-measured here with `is_device` `True` on all three. The **selector** exists in code and on screen (`--device {CPU,GPU,NPU}`, and the card badge), but **a presenter clicking a control and watching the route change was not run at M1** — device selection by the UI is **M2**'s route-transition work. |
| 3 | **frame IDs prevent overlay drift** — a result whose `frame_id` is not the displayed frame is discarded, proven by a test | **DEMONSTRATED** | `tests/test_ui_contract.py::OverlayContract::test_a_result_for_a_different_frame_is_not_accepted` (`fid 8` refused against a displayed `fid 7`), and `test_a_result_from_a_different_loop_of_the_clip_is_not_accepted`. The bridge compares the full identity four-tuple, so a result from another loop of the same clip is refused too. |
| 4 | event tests cover **transfer, return, occlusion, duplicate prevention and loop reset**, and emitted events are reported with their `dedup_key`s | **PARTIALLY DEMONSTRATED — read this carefully.** | the **tests** exist and pass for all five, in `tests/test_events.py`: `test_a_sustained_origin_to_destination_move_is_a_candidate_transfer`, `test_a_return_movement_is_a_separate_event_kind`, `test_losing_track_identity_cancels_the_transfer`, `test_the_same_key_is_never_emitted_twice`, `test_a_new_loop_resets_counters_and_clears_events`. The **dedup_keys are reported** — two events over the clip, `0:1:person:presence:counter_band` and `0:41:person:presence:counter_band`. **But no `candidate_transfer` is ever emitted on the real footage**, because no product transition exists to emit one: those five paths are covered against **synthetic** tracks. **The transfer and return paths have never run on footage.** |
| 5 | the same clip works on **each** target validated in M0 | **DEMONSTRATED as placement and throughput** | all three devices analysed the same clip over the same feed, `drops 0`, `errors []`, one identical event each. **Not demonstrated** as a booth demo: the runs are 8 s offscreen feeds, not the 60 s run the plan's example names. |
| 6 | a runtime failure produces a **visible error/fallback state**, and the UI never claims the failed target | **DEMONSTRATED as a state; NOT DEMONSTRATED as an observed failure** | the state is built and tested — a mismatch renders as `fell back from NPU`, a failed route renders `no device is running` and claims no device, a dead worker restarts once and then fails with a stated reason. **But no accelerator has ever failed on this machine**, so no failure has been *observed* falling back. M0's unmet gate 9 stands. |
| 7 | "Recorded footage • live inference" is displayed, pinned by a test on the QML source | **DEMONSTRATED as text and as rendered glyphs** | pinned on the QML source in `tests/test_ui_contract.py`, and asserted as **glyphs in the rendered pixels** — with the test explicitly refusing to pass vacuously on a render with no font registered. **No committed screenshot exists**, and the 1920x1080 DPI gate is **M3**. |
| 8 | unavailable metrics render as `unavailable`, never `0`; a genuine `0` still renders as a number | **DEMONSTRATED** | the live scene graph: `gpu` and `npu` read `unavailable`, `cpu_available` `true`, `app_cpu_percent` a real `0.0`. Pinned on every optional metric individually, including that no QML file mentions a utilisation percentage. |
| 9 | `verify_sources.py`, `preflight.py`, `--selftest` and the full unit suite are green, with real output pasted | **DEMONSTRATED** | 6 PASS/0 FAIL; **16 PASS / 2 WARN / 0 FAIL, exit 0**; `PASS` exit 0; **1034 tests, OK, exit 0**. All four pasted above. **"Green" includes two named WARNs**, which are the pre-existing M0 telemetry rows and are not closed. |

**Net: criteria 1, 3, 5, 7, 8 and 9 demonstrated in full. Criterion 2 demonstrated for
placement, not for an interactive selection. Criterion 4 demonstrated as test coverage and as
reported `dedup_key`s, with the transfer and return paths never having run on footage.
Criterion 6 demonstrated as a built and tested state, with no failure ever observed.** M1's gate
is therefore **not fully closed**, and the four qualifications above are the reason.

---

## Carried forward — not closed by M1, and not reported as closed

| item | status at M1 | owner |
|---|---|---|
| **GPU/NPU utilisation attribution** (M0 unmet gate 2) | **open.** In-process value reads fault this process with `0xC0000005`; by the CIM route a machine-wide peak is not attributable to this process and the inference PID never appears as a counter instance. The card reads "Utilization unavailable" and no utilisation figure exists anywhere in the product. | **M4** |
| **A real observed CPU fallback** (M0 unmet gate 9) | **open.** M1 built the visible failure state and the refusal accounting, and M1's CPU-only run is a measured CPU-only run. **No accelerator has actually failed mid-run and been observed falling back.** | **M2** route matrix / **M6** failure injection |
| **Retail shelf-to-basket** | **not delivered.** No such footage exists in the provisioned material, and the item-transfer reading of "checkout zone" was measured and is unsupported: 0 events under M0's rectangles over the full clip, the apple stationary at max corner deviation 0.0006 over 483 frames. The delivered event is people-in-zone presence. | **M5** |
| **A second annotated frame with a non-square object** | **not done.** The `w`/`h` transposition blind spot stands: a pure `w`/`h` swap preserves `cx`, `cy` and `w*h` exactly, so no centre-based or area-based statistic can separate it, and on the current annotation the aspect check is **inverted**. The fix is a different frame, not a tighter threshold. | **M1 follow-up / M5** |
| **The overlay-geometry test is vacuous on a square frame** | **recorded, not fixed.** The live-path geometry check compares the payload's boxes against an independent recomputation from `Track.box` and the frame extents, which catches a `y` divided by `width` instead of `height`. It is vacuous on a square frame, where the two divisors are equal. No such frame is used today — the live frame is 1920x1080 — so the hole is latent, not active. | **M2** |
| **Code signing** | **unmet.** No certificate available; M6 ships unsigned with the step documented. | **M6** |
| **1920x1080 DPI gate** | **not run.** Usable height on this display is 960 px, so a 1080p window cannot be shown; the gate needs the offscreen harness, which is M3 work. The live booth runs at this machine's 2293x960 desktop. | **M3** |
| **Two further known limits, recorded not hidden** | (a) The **QML Slot-binding guard is line-oriented, not a QML parse** — deliberately, per `tests/test_ui_contract.py::SlotBindingAudit`. It scans for `<identifier>(` in a binding expression and requires each call to be classified, with a **two-entry allow-list** (`bridge.explanation`, `bridgeUi.explain`) that **must be maintained**. A new binding to an unclassified call fails the audit; adding a name is a deliberate act with a stated justification. (b) **`unavailable_notice["CPU"]` can exceed its slot and elide.** The notice `Text` is single-line with `elide: Text.ElideRight` and a computed width, so a CPU absent-state longer than "no CPU sample yet" would be truncated on screen rather than wrapped. Not observed today; recorded so a later change to that string is checked. | **M2 / M3** |
| **Clip redistribution clearance, detector licence (AGPL-3.0), AC status, hardware SKU, the `oven` false positive** | unchanged from M0 — see [M0-qualification.md](M0-qualification.md#unmet-gates-and-open-blockers). M1 reopened none of them and closed none of them. | as recorded at M0 |

---

## What M1 does establish

- **One retail clip runs end to end** through a real application: 209 frames analysed, real
  pixels, 3 real overlay boxes, the manifest's zone, and a real event — offscreen, on CPU, exit 0.
- **Placement is proven, not requested**, on all three accelerators, twice over: from the three
  committed headless summaries and from a fresh 4-repeat latency measurement. `['CPU']`,
  `['GPU.0']`, `NPU` — each naming the device that was asked for, through `is_device`.
- **The business claim is the one the footage supports**: two sustained person-presences over
  the clip, one per person-track, each with its `dedup_key`. Not a transfer, not a count, not a
  dwell. And a guard fails the build if a shipped configuration emits none.
- **The NPU is slower than the GPU on this machine**, with its conditions attached wherever the
  claim appears, and the product's own copy says so.
- **Unavailable is a word, never a `0`**, on every optional metric, on the live bridge and in
  the QML source.
- **The UI thread does not block on inference**: p95 10.1–16.6 ms and worst tick 21.6–37.5 ms
  under 6 CPU-only saturators on 8 logical cores — a floor, not a bound, and marginal at 8 of 8.
- **The gate set is green and pasted**: 6 PASS/0 FAIL, 16 PASS/2 WARN/0 FAIL exit 0, `--selftest`
  PASS exit 0, 1034 tests OK exit 0.
- **What it does not establish is written down**, criterion by criterion, above — including that
  M1's own gate is not fully closed and why.

### Reproduction

From a clean checkout, **two steps, in this order**, because `assets/` and `logs/` are gitignored:

1. `.venv\Scripts\python.exe tools\provision_venv.py` — the only step needing network.
   **Until this has run, none of the above is reproducible**: the IR and the three clips are
   absent and the suite fails.
2. The four gate commands in this report.

The headless summaries this report quotes are **committed** at
[`docs/qualification/m1-headless/`](m1-headless/README.md) and do not depend on step 1 to be
read — only to be re-measured.