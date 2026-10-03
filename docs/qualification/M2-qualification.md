# M2 — Three streams and device toggles: qualification report

**Date:** 2026-10-03. **Machine:** the demo machine, `NUC14PROAI` (a hostname, not a SKU).
**Branch:** `main`. **Scope:** milestone M2 — a multi-stream supervisor, presenter-driven device
toggles, per-stream placement and per-stream staleness on screen, and the closing of the gap
M2 itself disclosed (per-camera video surfaces).

This report **extends** [M1-qualification.md](M1-qualification.md) and
[M0-qualification.md](M0-qualification.md); it contradicts neither. Where a figure here differs
from an earlier milestone's figure, both are printed and the difference is explained rather than
quietly resolved in M2's favour.

---

## The two headlines, first

1. **The 2 s warm-transition target is MISSED.** Criterion 5 permits "meets the 2 s target **or the
   miss is documented with its measured distribution**". The distribution is documented below, in
   full, on both sides of the teardown fix, and the verdict is **UNMET**. No timeout was widened to
   achieve any of the numbers in this section.
2. **Three streams on the CPU is an 11–15 second old picture, and every counter on the screen reads
   healthy.** The screen now says so in words, per stream, with the age, the window, the backlog and
   the analysis rate. **The input-acknowledgement gate cannot see this**, by construction, and a
   green gate on the wrong quantity is not a pass.

---

## What M2 added

| area | what exists now |
|---|---|
| **Supervisor** | N streams, each with its own worker process, its own generation counter, its own placement, its own queue and its own accounting. `RoutePlan` is per stream. |
| **Route transitions** | generation-scoped warm changes with a bounded acknowledgement; a superseded generation's results are discarded rather than painted; every completed job is attributed to the device that **actually ran it**, via `EXECUTION_DEVICES` re-read after the warm-up inference. |
| **Per-stream placement** | `requested` and `confirmed` are separate fields everywhere, including the badge. A failed route claims **no** device. |
| **Per-stream errors** | `RoutePlan.stream_errors` — a real channel, not the aggregate list — plus a `failed`-with-no-reason refusal at construction. |
| **Per-stream metrics** | per-camera frame age p50/p95, analysis rate, backlog and window, published per stream, with `None` preserved and never coerced to `0.0`. |
| **Per-camera video** | each tile serves **its own** stream's frames through `FrameImageProvider`, or says in words that its camera has produced nothing. **Each camera has its own clip, decode thread, tracker and event machine.** |
| **Presenter-driven selection** | `--select A=GPU` moves one camera while the app is running; the route goes `pending` and then `confirmed`, and the badge follows the **confirmed** device. |
| **A fault-injection flag** | `--fail-stream X` really kills one camera's worker. **A qualification aid, not a feature** — see "Not closed". |

---

## The gate set, run in this session

### 1. `tools\verify_sources.py` — 6 PASS, 0 FAIL, exit 0

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

Unchanged from M1. Every artifact is verified **by content signature and sha256**, never by HTTP
status code.

### 2. `tools\preflight.py` — 16 PASS, 2 WARN, 0 FAIL, exit 0, `Status: Limited`

Reproduced by content, not by paraphrase. Each row's verdict line from this run:

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
```

The numbers this run measured, quoted from the rows above:

| | |
|---|---|
| openvino | `2026.4.0-22959-99c81491cc3-releases/2026/4` |
| python / psutil | `3.12.10` / `7.2.2` |
| disk free | `184.9 GiB` of `381.5 GiB` (policy floor 2 GiB) |
| memory available | `20.3 GiB` of `31.5 GiB` across 8 logical cores (policy floor 4 GiB) |
| drivers | GPU `32.0.101.6737`, NPU `32.0.100.5540`, CPU none (no separate driver artefact) |
| `DeviceRegistry.describe()` | `3.28 s` for all three devices — one memoised PnP enumeration, startup-only |
| module import → first inference | `3.73 s`, `time.perf_counter`, this run |
| sanity fixture on CPU | 2 of 2 annotated objects matched, `EXECUTION_DEVICES=['CPU']`, `placed_on_requested=True` |

**Both `WARN`s are M0's two pre-existing telemetry rows and neither is closed.** The first is the
in-process PDH utilisation read that terminates this process with `0xC0000005`; the second is that
the LUID→device map is *derived* from the engine taxonomy rather than exposed by a driver, and three
hardware-identity records disagree. Both are M4's. **"Green" here means green including two named,
still-open warnings**, exactly as in M0 and M1.

> `tools\preflight.py` **was not run for the Tasks 1–2 review and no number was claimed for it at
> the time.** The table above is this session's run, the first one taken since. It is not a
> re-measurement of anything M2 changed: preflight exercises device discovery, the pinned IR, the
> sanity fixture and telemetry, none of which M2 touched.

### 3. `python -m app.main --selftest` — PASS, exit 0

```
  ok   preprocess           letterbox 640x640 pad=114 scaled=640x360
  ok   infer                23.72 ms on confirmed ['CPU']
  ok   decode-track-event   1 zone_occupancy event(s) from a synthetic presence
  ok   render-metadata      unavailable renders as unavailable, and no device is claimed before an inference confirms one
PASS
```

### 4. `python -m unittest discover -s tests` — 1210 tests

**Three full runs were taken, because the first one failed.** The failing run was taken on the tree as
it stood at that moment; the tree has since gained one test (a thumbnail-caption notify test, below), so
the current tree is the 1210 the two later runs report.

```
run 1:  Ran 1210 tests in 359.861s
        FAILED (failures=1)
        FAIL: test_nothing_grows_that_the_retained_records_do_not_account_for
               (tests.test_route_stress.RouteStressInTheSuite...)
        wall 360.6 s

run 2:  Ran 1210 tests in 359.891s
        OK

run 3 (current tree):  Ran 1210 tests in 359.077s
        OK
        wall 359.8 s
```

**No test was lost.** Compared with Task 5's tree, the two touched test files gained 7 and 6 test
methods respectively and **zero were removed**, checked by comparing the `test_*` method names of every
class parsed from both versions.

In isolation, `tests.test_route_stress.RouteStressInTheSuite` is **7 tests, OK**:

```
Ran 7 tests in 20.108s
OK
```

This is a **pre-existing flake carried from M2 Task 3**, not something this session introduced: it
puts a **ratio** bound on a measurement (`rss` growth against retained-record growth) that it shares
with five other tests in the same class. It has now failed twice across nine full-suite runs. The
suite wall time is **≈360 s**, up from M1's 270.7 s and M2 Task 5's 323.8 s; the increase is the
three-camera live-path test class (`PerCameraVideoOnTheLivePath`, 5 tests, ~32 s) plus Task 5's
layout tests.

### 5. The Slot-binding guard

```
tests.test_ui_contract.SlotBindingAudit
Ran 7 tests in 0.380s
OK
```

Green, and it is the guard M1 recorded as **line-oriented with a two-entry allow-list that must be
maintained**. M2 added bindings (per-stream maps, the per-camera `streamId`) and **no name was added
to the allow-list**: every new binding reads a property off a notified map. That is why the guard is
still green without a new entry — a fact worth stating, because the alternative (adding names to make
a guard pass) would have been invisible.

---

## Placement — requested versus confirmed, on every device

Re-measured in this session, four full runs of `tools\qualify_m0.py --requests 30`, each in its own
process, committed at `docs/qualification/m2-headless/single-frame-latency-run{1..4}.json`. **30
timed inferences per row, single frame, letterboxed 640×640, batch 1, static, warm-up outside the
timed span.** `OVERALL: PASS - 4 of 4 rows succeeded on the device they requested` on every run.

| device | `EXECUTION_DEVICES` | p50 ms, run 1 / 2 / 3 / 4 | range | compile s |
|---|---|---|---|---|
| CPU | `CPU` | 22.61 / 22.81 / 22.40 / 22.72 | **22.40 – 22.81** | 0.19 |
| GPU | `GPU.0` | 3.45 / 3.54 / 3.42 / 3.40 | **3.40 – 3.54** | 0.62 – 0.63 |
| NPU | `NPU` (the bare string) | 5.25 / 5.21 / 5.25 / 5.74 | **5.21 – 5.74** | 1.54 – 1.57 |

Each run also carries a **fourth** row, `CPU-fallback-baseline`: the same pinned model re-measured on
the CPU *after* the three device rows (22.92 / 22.75 / 22.56 / 22.96 ms). It is a separate run, not a
copy of the CPU row — **a fallback that reuses the primary row's numbers has proved nothing about the
fallback path.** All four files record `model_ir_sha256_matches_config_models_json: true` and
`num_requests_per_row: 30`.

**The NPU is slower than the GPU on this machine.** GPU 3.40–3.54 ms against NPU 5.21–5.74 ms, under
the conditions named in the table header, on an integrated Arc 140V sharing system memory with the
NPU. **This is not a throughput claim.** Paired trials are M4's.

**A correction to the record, stated because it changes the headline.** The milestone working log
quotes CPU 20.99–21.16 ms and NPU 5.92–5.99 ms for this same measurement. **Those numbers are
better than anything measured in this session** (CPU 22.40–22.81 ms), they are recorded only in a
gitignored file, and they were taken earlier on the same day. This session's numbers are the
committed ones. The NPU-slower-than-GPU ordering holds in both; the absolute CPU figure does not, and
the machine is measurably slower now than it was then — see the frame-age holds below for the same
effect at a larger scale. **A per-device latency figure is a property of a session, not of a machine,
and quoting the best one observed is not a measurement.**

M1's own committed per-device rows (`docs/qualification/m1-headless/*.json`) remain valid and are not
re-measured into this report's claim.

### Placement across 50 live transitions

From `route-stress-50-after-teardown-fix.json`: **`placements_confirmed` 50/50 and
`placement_pids_matched` 50/50** — every transition's confirmation named the device it asked for, and
the process that confirmed it was the process that was live. Per stream: A `CPU`, B `GPU.0`, C `CPU`,
all `fell_back: false`, all `state: active` at the end of the run.

**`fell_back: true` has never been observed on this machine.** See "Not closed".

---

## HEADLINE 1 — the 2 s warm-transition target is UNMET

**Criterion 5's second clause is satisfied and its first clause is not.** Both halves are stated here
because reporting only the improved number would be a misrepresentation.

### The full toggle distribution

`total` is the whole toggle, `setup_s + transition_elapsed_s`. **50 toggles, all 50 acknowledged,
nothing censored** (`--timeout-s 30`, widened on purpose so the tail is measured rather than
truncated).

| | min | p50 | p95 | max | over 2 s |
|---|---|---|---|---|---|
| **before the teardown fix** (`route-stress-50-measured.json`) | 2.080 | 3.151 | 5.521 | 5.544 | **50/50** |
| **after** (`route-stress-50-after-teardown-fix.json`) | **1.286** | **2.261** | **4.971** | **5.185** | **25/50** |

Where the time goes, from the same two files:

| | setup p50 (`begin_transition`) | wait p50 (`transition_elapsed_s`) | wait p95 | wait max |
|---|---|---|---|---|
| before | 1.642 s (min 1.087, max 2.197) | 1.506 s (min 0.987) | 3.369 s | 3.387 s |
| after | **0.730 s** (min 0.145, max 1.730) | 1.510 s (min 1.119) | 3.458 s | 3.501 s |

**The wait half is now the floor** and the setup half is no longer the dominant term. That is the
whole content of the fix, and it is why the before/after rows are not interchangeable.

### At the real 2.0 s default

This is what the booth actually does — `await_transition`'s own 2.0 s deadline, unchanged.

| | acknowledged | over 2 s | censored at the gate |
|---|---|---|---|
| before (`route-stress-50-gate-2s.json`) | **23/50** | **48/50** | yes |
| after (`route-stress-50-gate-2s-after-teardown-fix.json`) | **26/50** | **37/50** | yes |

### The earlier published 1.04 / 1.67 / 1.31 s figures were **half a toggle**

Those figures were published earlier in this milestone and **excluded `begin_transition`**. They
measured `transition_elapsed_s` alone — the wait, not the toggle. They are wrong as a claim about the
spec's 2 s gate, they are not in any committed file, and they are recorded here so that nobody
reaches for them again. **`total` is the number the gate is judged on.**

### Verdict

> **UNMET.** 25 of 50 toggles exceeded 2 s with the deadline widened so the whole distribution could
> be measured, and 37 of 50 exceeded it at the real 2.0 s default. The improvement from 50/50 to
> 25/50 is real, measured, committed, and **does not meet the target**. The miss is documented with
> its distribution, which is what criterion 5 permits. `stop()` teardown fell from 15.18 s to 0.87 s
> in the same fix; that figure comes from the milestone working log and was **not re-measured in this
> session**, so it is quoted as a task record and not as committed evidence.

---

## HEADLINE 2 — three streams on the CPU is an 11–15 second old picture that reads healthy

**Unsaturated, spec §5 row 4, three 120 s holds, three streams on the CPU, no route change and no
churn.** Committed at `docs/qualification/m2-headless/allcpu-steady-120s{,-hold2,-hold3}.json`.
15 fps offered per stream, 45 fps total, `errors: []`, **zero drops, all three streams `active`, zero
`fell_back`** on every row.

| hold | overall p50 | overall p95 | per-stream p50 | backlog per stream |
|---|---|---|---|---|
| 1 | 11 319 ms | 12 286 ms | 10 672 / 11 104 / 11 695 | 173 / 180 / 187 |
| 2 | 14 309 ms | 15 562 ms | 13 974 / 14 128 / 14 216 | 228 / 232 / 235 |
| 3 | 14 697 ms | 16 019 ms | 13 922 / 15 027 / 14 640 | 226 / 242 / 238 |

**Read the range, not the best row.** The milestone working log quotes 10 767 ms and 11 085 ms for
two earlier holds; those two are **not committed** and are lower than all three holds committed here.
The picture gets *older* across repeated holds on this machine, by about 30%. The cause is **not
established** — thermal state, page-cache pressure and background load were not instrumented, and
guessing between them would be exactly the kind of unsourced claim this project exists to avoid. What
**is** established is that the honest claim is a range: **p50 10.7–14.7 s, p95 12.3–16.1 s, backlog
173–242 frames per stream**, all three streams `active` with zero drops throughout.

Every counter reads healthy while this is happening: `state: active`, `drops: 0`, `errors: []`,
`frames_analysed` 1 559–1 627 per stream against 1 800–1 802 accepted. The analysis rate is
**~13.0–13.6 of 15 fps**. Nothing on the screen was wrong before M2; the picture was simply old.

### What the screen says about it, read off the scene graph

The cards were rendered from the **committed** hold files — every number below is a committed number
put through `UiBridge` and read back off the live QML scene graph. Hold 1:

| | Camera A | Camera B | Camera C |
|---|---|---|---|
| confirmed badge | `CPU` | `CPU` | `CPU` |
| age headline | `11.7 s` | `10.7 s` | `11.1 s` |
| staleness sentence | `STALE - this picture is 11.7 s old; the limit is 2 s` | `STALE - this picture is 10.7 s old; the limit is 2 s` | `STALE - this picture is 11.1 s old; the limit is 2 s` |
| window | `window: 256 analysed frames` | `window: 256 analysed frames` | `window: 256 analysed frames` |
| backlog | `backlog 187 frames` | `backlog 173 frames` | `backlog 180 frames` |
| rate | `analysis 13.44 fps` | `analysis 13.56 fps` | `analysis 13.50 fps` |
| header badge | `device: CPU` | | |

Hold 3, the worst committed hold: `13.9 s`, `15.0 s`, `14.6 s`, backlog `226 / 242 / 238`, rate
`13.11 / 12.98 / 13.01 fps`.

Four things about this screen are load-bearing and are worth stating because each could have been the
other way:

- **The sentence is in words, not a colour.** Spec §7 says "values older than 2 s are `stale`"; the
  card says `STALE - this picture is 11.7 s old; the limit is 2 s`, naming the age and the limit.
- **The card is never dimmed for being stale.** Dimming the card that says the NPU stream is stale
  *because NPU utilisation cannot be measured* would dim the one warning that matters.
- **Frame age and inference cost are never merged.** The window line says `256 analysed frames` (this
  camera's deque bound); the device's own figure carries its own sampler window in a separate line.
  Neither is ever presented as the other.
- **`unavailable` is never `0`.** With no telemetry sample seeded, the CPU line reads
  `no CPU sample yet`, and the aggregate line reads `telemetry unavailable` — the words, not a zero.

### The input-acknowledgement gate cannot see this

M1's input-ack gate measures **UI-thread tick lateness**: a 20 ms `QTimer` on the Qt main thread,
against a 200 ms p95 gate and a 2 s worst-tick gate, with 6 CPU-only saturator processes plus real
CPU inference. Re-measured in this session (`tests.test_main.InputAcknowledgement`, 3 repeats):

| repeat | p50 | p95 | worst tick | ticks | confirmed |
|---|---|---|---|---|---|
| 0 | 0.015 ms | 11.554 ms | 20.361 ms | 299 | `['CPU']` |
| 1 | 0.024 ms | 12.863 ms | 18.470 ms | 299 | `['CPU']` |
| 2 | 0.000 ms | 11.749 ms | 17.096 ms | 299 | `['CPU']` |

Against M1's committed figures (p95 16.582 / 10.114 / 11.463 ms) the same gate, in the same order of
magnitude. The working log's M2 figures (p95 5.94–8.62 ms) were taken in a lighter machine state and
are **not committed**; the table above is this session's measurement, taken by running the committed
test.

> **Ruling: this gate is insensitive to inference backlog by construction, and a green result from it
> is not evidence that the app is keeping up.** It times how long the **UI thread** waits between its
> own ticks. A frame sitting 242 frames deep in a worker queue is invisible to it, because the UI
> thread is not waiting on the queue — it is being told about a result it did not need yet. The
> unsaturated `allcpu` frame-age row above is the real latency figure for this application, and it
> leads this section. **The two quantities must never be added, averaged or substituted for one
> another, and a booth operator should read the per-camera age, not a green tick gate.**
>
> A combined experiment — the input-ack probe running against the three-camera `allcpu` load, so both
> numbers come from one run — was **not run**. It is the obvious next measurement and it is not
> claimed here.

---

## Per-camera video surfaces — the gap M2 disclosed, closed

M2's Task 5 report disclosed: *"per-camera video surfaces are not built — the hero and M1's two
honestly-labelled thumbnails are unchanged, and three live pictures need per-stream frames through
`FrameImageProvider` and a feeder per stream."* **Closed.**

- **One frame store per camera.** `UiBridge` keys pixels, revisions and overlay boxes by stream id,
  read from `frame.stream_id` — the identity the frame was submitted under, so it cannot disagree with
  the worker that produced it.
- **One address per camera.** `image://frames/<stream_id>?v=<that camera's revision>`, from the
  provider's **notified** revision map. There is **no cross-stream fallback**: a camera with no frame
  gets a 1×1 transparent image, never another camera's pixels.
- **One tile, one camera, and it says so.** A camera that has delivered nothing shows
  `no frame from Camera N yet`, naming itself. It does not hold a stale frame and does not borrow the
  hero's.
- **The hero follows the selection.** `UiBridge.set_route_stream` re-derives the hero's pixels, its
  revision and its overlay boxes from the newly bound stream, so pointing the screen at Camera C does
  not leave Camera A's rectangles drawn over Camera C's shopper.
- **One clip, one decode thread, one tracker and one event machine per camera.** Sharing a tracker
  interleaves track ids across cameras and lets one camera's zone rule fire on another's picture: a
  plausible, correctly-placed rectangle over the wrong object.
- **`--streams N`** (default 1, which behaves exactly as M0/M1 did). With 3: one camera per device
  the registry reports, `--device` first, and the run prints its own layout —
  `[startup] 3 camera(s): A -> CPU, B -> GPU, C -> NPU`.

### Measured on the live path, not seeded

One real run, three real workers on three devices, three decode threads, offscreen, 25 s:

```
returncode: 0
layout:      [startup] 3 camera(s): A -> CPU, B -> GPU, C -> NPU
hero tile stream: A     hero source: image://frames/A?v=269
  tile A: source=image://frames/A?v=269  size=1920x1080  no_frame_visible=False  text=''
  tile B: source=image://frames/B?v=339  size=1920x1080  no_frame_visible=False  text=''
  tile C: source=image://frames/C?v=340  size=1920x1080  no_frame_visible=False  text=''
frames analysed/submitted: 1194 / 1281
cards on screen: card_A
qml badge: device: CPU
```

Three tiles, three sources, three independent revisions (269 / 339 / 340 — three workers analysing
at three different rates), three real 1920×1080 frames, exit 0. **`ai_fps` 42.72** across three streams.

### Two defects this work produced, both found by rendering rather than by reading

**A `QVariantList` concatenated with a string rendered the literal `-1`.** The thumbnail caption was
`"Camera " + modelData + " · " + bridge.stream_routes[modelData].confirmed`, and `confirmed` is a
Python tuple crossing into QML. The caption on screen read **`Camera B · -1`** on a render whose
Python side held `confirmed: ('GPU.0',)` and whose picture above the caption was real. **Nothing about
a screenshot of that caption says it is wrong.** Fixed by joining in Python
(`UiBridge.stream_confirmed_labels`, a notified map of strings) and reading a plain string in QML.
Three tests now pin it, including one that publishes the confirmation **after** the scene is built, so
a wrong `notify=` cannot pass.

**The provider's own id parsing was wrong twice before it was right, and each failure was silent.**
`requestImage` is reached as `requestImage(id, requestedSize)` — `args[-1]` is a `QSize(-1, -1)`, not
the id — and the id itself is the **whole URL** as a `QUrl`, not the trailing segment. Both faults
produced a correct-looking `source`, a `paintedWidth` reporting a scaled 1×1, and **a hero that was
simply empty with no warning anywhere**. Both are now measured and asserted: the render publishes the
provider's entire request list, and the tests read it.

### Mutation checking

Twenty mutants, **eighteen killed**. The two survivors are **deliberate no-op controls** that *must*
survive, so that the claim "the sweep found everything" is falsifiable:

| mutant | verdict | note |
|---|---|---|
| A.3 — a stream with no frame becomes "ready" via a renamed method | **survived by design** | renames nothing any test reads |
| A.10 — every camera is fed the *same* `ClipSource` | **survived by design** | the layout line is printed from `stream_ids`, not from the clips, so nothing a test reads changes |
| A.1 / A.2 — the provider falls back to the hero's frame / to the newest frame from any camera | killed | the two cross-stream mislabelling routes |
| A.4 — every tile asks for `hero` again | killed | the pre-M2 defect itself |
| A.5 / A.6 / A.7 — no panel / no camera name / the hero's readiness on every tile | killed | |
| A.8 — the hero hardcoded to Camera A | killed | **survived the first sweep**; fixed by naming the hero and reading its own `streamId` |
| A.9 — the thumbnails dropped entirely | killed | |
| A.11 — the frame payload loses its stream id | killed | **survived the first sweep**; the binding was right and the pixels were wrong |
| A.12 — the frame stored under the bound stream | killed | |
| A.13 – A.20 — the caption concatenates the list / borrows the hero's device / prints nothing / names the wrong camera / the label map joins nothing / takes the wrong stream's entry / is not notified | killed | the last sweep: **8 of 8 killed** |

Two first-sweep survivors were **real gaps, not controls**, and both produced new assertions before
the sweep was re-run:

- **A.8** (the hero hardcoded to Camera A) survived because every assertion was about a *tile*, and the
  tiles are the same either way. Fixed by naming the hero `heroTile` and reading **its own**
  `streamId`, in the offscreen render and on the live path.
- **A.11** (the payload's `stream` key pinned to `"A"`) survived because the harness had aimed it at an
  assertion that only checked each tile's **binding** — and the binding was right while the **pixels**
  were wrong. Fixed by a new live assertion that **no** camera shows its `no frame` panel on a run
  with three live workers, plus an independent-revisions check; the harness was retargeted rather than
  the claim being softened.

That is worth stating plainly: **a sweep reports a survivor that was never aimed at the covering
assertion**, and it did so here. The remedy is a new assertion, not a softer claim.

---

## Pending results cannot paint after a generation change — demonstrated, weakly

From `route-stress-50-after-teardown-fix.json`: **`discarded_results` 942** over **4 179 delivered
results**, `discarded_mismatched` 0, **stale overlays 0**, **mislabelled devices 0**, **crashes 0**,
`error_results` 0, `errors_at_end: []`.

**Read that 942 honestly: it is ~19 results per toggle, not 19 demonstrations of the invariant.**
Before M2's teardown fix the same field read **29** over 5 037 delivered results — under 0.6 per
toggle, because the queue was drained *before* superseding, so the discarded work was work that was
always being thrown away. **The count went up because the drain now counts work that was always being
discarded. That is a better accounting, not fifty more proofs of the same thing.**

The invariant itself is pinned by tests that do not depend on timing: a result whose four-tuple
identity is not the displayed frame is refused, and a result from another loop of the clip is refused
too (M1 criterion 3, unchanged and still green).

---

## Leak and retention

Reported as **total, design growth, and remainder separately**, because adding them is the mistake this
section exists to avoid.

| | before the teardown fix | after (50 toggles, 30 s deadline) | after (50 toggles, real 2.0 s gate) |
|---|---|---|---|
| `rss_start` | 94.8 MiB | 95.7 MiB | 95.0 MiB |
| `rss_end` | 107.9 MiB | 111.1 MiB | 277.4 MiB |
| **total growth** | **13.1 MiB** (+268.6 KiB/toggle) | **15.5 MiB** (+316.8 KiB/toggle) | **182.4 MiB** (+3.65 MiB/toggle) |
| retained transitions | 1 → 51 | 1 → 51 | 1 → 51 |
| retained retired queues | 6 → 156 | 6 → 156 | 6 → **204** |
| live objects, emptied + `gc` | 54 689 → **50 107** | 54 870 → **50 560** | 57 438 → **52 724** |
| **remainder after release + `gc`** | **3.53 MiB** | **5.76 MiB** | 175.3 MiB |
| retention explains | 9.6 MiB | 9.7 MiB | 7.1 MiB |

**Live objects go *down* across 50 transitions**, by 4 582 / 4 310 / 4 714 in the three runs above.
Nothing is accumulating per transition; the retained transition and queue records are counted as design
growth because they are deliberately retained for the life of the process.

**One arithmetic correction.** The milestone working log describes the 3.5 MiB remainder as
"0.26% of RSS". It is **3.3% of the 107.9 MiB end RSS** (3 702 784 / 113 115 136). The 0.26% figure
does not follow from the committed numbers; the corrected percentage is used here.

**The third column is the finding, and it is not characterised.** At the real 2.0 s gate — the run a
presenter actually experiences — RSS ends at **277.4 MiB**, 182.4 MiB above where it started, with 175.3
MiB unexplained by retention after emptying the retained records and collecting. That run is the one
with **24 unacknowledged transitions and a stated recovery per timeout**. One hypothesis is that each
recovery path retains a worker or a queue that a clean run never builds; **it is a hypothesis, it is
not established, and it is not measured.** A leak trend over a long session is M6's soak, and this
report does not close it.

---

## The visible failure state — and the half that cannot close

M0's gate 9 has two halves: a **silent** one (`confirmed` differing from `requested`) and a **visible**
one. **The visible half is delivered; the silent half cannot be closed on this machine, and M2 does
not close it.**

**Why the silent half cannot occur here.** All three devices execute where they are asked. A silent
fallback requires a device to fail in a way the runtime absorbs — and `compile_failure` kills the
worker *before it runs anything*, which is a loud failure, not a silent one. There is no fault on this
machine that produces the quiet substitution M0 asked to be observed, and **no fault was invented to
produce one.**

**What is delivered, measured.** Two independent runs, both from this session or from a committed
file, and the two are **not** the same measurement:

**(a) On the live scene graph** — `--fail-stream A` through `app.main.run_windowed`, the worker process
really raising before it compiles anything:

```
127 badge samples
device: pending - CPU not yet confirmed
device: failed - no device is running

qml_badge            'device: failed - no device is running'
device_display       'failed - no device is running'
confirmed            ''                     <- a failed route claims NO device
fell_back            false                  <- nothing ran elsewhere, so there is nothing to fall back from
frame_ready          false
tile A               'no frame from Camera A yet'
stated reason        'COMPILE FAILURE INJECTED BY THE SUPERVISOR\'S TEST SEAM
                      (compile_failure=True). The worker process raised before compiling
                      anything. This says nothing about the requested device\'s health; it exists
                      so the supervisor\'s detect-a-dead-worker, restart-once and fail-visibly path
                      can be exercised without a device that genuinely wa...'
```

That last clause is in the product's own wording and is the right one: a fault injected to exercise a
path must not be allowed to look like evidence about a device.

**(b) In the stress record** — `route-stress-fault-visible-fallback.json`, six transitions with
`--fault-stream C`, camera C's whole lifetime:

| | |
|---|---|
| `requested_device` | `NPU` |
| `confirmed_devices` | **`[]`** |
| `state` | `pending` (never confirmed) |
| `frames_analysed` | **0** |
| `frames_accepted` / `frames_refused` | 253 / 896, every refusal carrying the reason `not_running` |
| `fell_back` | `false` |
| `restart_count` | 1 — **exactly one** restart, then failed |

**A failed route claiming no device is the correct rendering**, and it is pinned by tests at three
levels: `RoutePlan` refuses to be constructed with a `failed` stream and no per-stream stated reason
(`ValueError`), `Supervisor.route_plan()` populates the channel, and the card renders *"failed, and no
reason was stated for this stream"* rather than asking the presenter to guess.

> **A number corrected while writing this.** An earlier draft of this report attributed
> "20 submitted / 0 analysed" and `{"requested":"CPU","confirmed":[],"state":"failed"}` to the fault
> stress file. The committed file says otherwise: it was run with `--fault-stream C` against the
> **NPU**, camera C was offered 2 576 frames and accepted 253 before the supervisor began refusing
> them, and its final state is `pending`, not `failed`. The `0 analysed` and the empty `confirmed`
> list are real; the "20 submitted" and the `state: failed` pairing are not in this record and have
> been removed rather than restated.

### Presenter-driven selection — observed

Re-measured in this session through `tests.test_main.PresenterSelectionAndObservedFallback` (6 tests,
OK, 46.1 s), driving `app.main.run_windowed` with a real `--select A=GPU`:

```
[startup] 1 camera(s): A -> CPU
BADGE {"at_load": "device: pending - no device selected", "after_route": "device: CPU (fell back from NPU)"}
[select] requested {'A': 'GPU'} -> acknowledged=True bounded_by='confirmed' generation=2 in 1.293 s;
          confirmed={'A': ['GPU.0']}; unconfirmed=[]
```

**219 badge samples**, taken every 100 ms **on the UI thread**, and three distinct badge values **in
order**:

```
device: pending - CPU not yet confirmed
device: pending - GPU not yet confirmed
device: GPU.0
```

The invariant the assertions are built on: the badge may **lag** the supervisor by a queued hop, and
did; it may never **lead** it. A badge that jumped to the requested device the instant it was asked
would pass any single-instant check and would be a device on screen that nothing has run on. At the
end of the run: `confirmed 'GPU.0'`, `fell_back false`, a real 1920×1080 frame, one published zone,
**4 overlay boxes**, `event_counts {'zone_occupancy': 2}`, `analysis fps 13.87 · frame age p50 40.4 ms
p95 45.3 ms`.

The same class runs `--fail-stream A`, where the worker really raises before compiling: **127 badge
samples**, `device: pending - CPU not yet confirmed` → **`device: failed - no device is running`**,
`device_confirmed` the empty string, `fell_back false` (nothing ran elsewhere, so nothing to fall back
from), `frame_ready false`, and the tile saying `no frame from Camera A yet`.

This closes **M1's criterion 2**, which M1 recorded as demonstrated for placement but not as an
interactive act.

---

## M2 exit criteria — stated one by one

| # | criterion | verdict | evidence |
|---|---|---|---|
| 1 | all four §5 toggle combinations execute | **DEMONSTRATED** | the committed `route-stress-50-after-teardown-fix.json` walked **all four** rows of the §5 table in the table's own order, one camera per toggle: `{A:CPU,B:GPU,C:NPU}` ×12, `{A:CPU,B:CPU,C:NPU}` ×13, `{A:CPU,B:GPU,C:CPU}` ×13, `{A:CPU,B:CPU,C:CPU}` ×12. Every one acknowledged, and every placement confirmed by `EXECUTION_DEVICES` (`placements_confirmed` 50/50). |
| 2 | completed jobs attributed to their **actual** devices, per stream | **DEMONSTRATED** | four fresh `qualify_m0` runs, `OVERALL: PASS - 4 of 4` on every one, `CPU` / `GPU.0` / `NPU`; plus `placements_confirmed` 50/50 and `placement_pids_matched` 50/50 across the stress runs. **Caveat, and it is a real one:** the *presenter-driven* form of this was measured separately (219 badge samples, above), not folded into the same run. |
| 3 | pending results cannot paint after a generation change | **DEMONSTRATED, WEAKLY** | `discarded_results` 942 over 4 179 delivered, stale overlays 0. **Weak because that is ~19 per toggle and the pre-fix count was 0.6 per toggle** — the number went up because the accounting improved, not because fifty more proofs appeared. The identity-based refusal tests are the real evidence. |
| 4 | 50 alternating warm route changes, no crash / leak trend / stale overlay / mislabelled device | **DEMONSTRATED** | 50/50 acknowledged, `crashes 0`, `mislabelled_devices 0`, stale overlays 0, live objects **down** 4 310. **The leak trend is qualified above:** clean at a 30 s deadline (15.5 MiB total, 5.76 MiB remainder) and **not** clean at the real 2.0 s gate (182.4 MiB, 175.3 MiB unexplained). |
| 5 | warm transitions meet 2 s **or the miss is documented with its distribution** | **THE TARGET IS MISSED. THE DISTRIBUTION IS DOCUMENTED.** | 25/50 over 2 s with the deadline widened so the tail is measurable; 37/50 over 2 s at the real 2.0 s default; full min/p50/p95/max on both sides of the fix, plus the setup/wait split. **Both halves are the answer: the first clause is not met and the second is.** |
| 6 | M1's criterion 2 (presenter-driven selection) and criterion 6 (observed fallback) | **BOTH CLOSED — with criterion 6's closure scoped precisely** | selection: observed, **219 badge samples**, three badge values in order, acknowledged in 1.293 s. failure: **the visible half is closed** (a failed route claims no device and states its reason); **the silent half of M0 gate 9 remains open and cannot be closed on this machine.** |

---

## NOT closed — and not written as closed

| item | status | owner |
|---|---|---|
| **M0 gate 9's SILENT half** | **OPEN. Cannot be closed on this machine.** All three devices execute where asked, so `confirmed != requested` never occurs naturally, and `compile_failure` is a loud failure. The **visible** half is delivered and measured above. | **M6** fault injection / **M4** |
| **`--fail-stream`** | **A documented fault-injection flag on the shipped CLI.** It is a **qualification aid**, named as such in the presenter guide, so nobody ships it believing it is a feature. | presenter guide |
| **GPU/NPU utilisation attribution** | **open, unchanged.** In-process PDH reads fault the process; a machine-wide peak is not attributable to it. The cards read `Utilization unavailable`, never `0%`. | **M4** |
| **Retail shelf-to-basket** | **not delivered, and unmet by design.** The delivered event is people-in-zone presence, 2 per clip — **not a people count and not a dwell measurement**, because `BusinessEvent` carries no duration. | **M5** |
| **Code signing** | **unmet.** No certificate. M6 ships unsigned with the step documented. | **M6** |
| **1920×1080 DPI gate and `tools/capture_ui.py`** | **not run.** `tools/capture_ui.py` does not exist; the live booth runs at this machine's 2293×960 desktop and M0 measured usable height at 960 px. **This report commits no screenshot of the booth.** | **M3** |
| **A second annotated frame with a non-square object** | **not done.** The `w`/`h` transposition blind spot stands; the current annotation's aspect check is inverted. | M1 follow-up / **M5** |
| **The Slot guard's two-entry allow-list** | **still line-oriented, not a QML parse**, deliberately, with a two-entry allow-list that must be maintained. **M2 added no name to it** — see the gate section. | M2/M3 |
| **The count-bounded `frame_ages_ms` window** | **now on the card**: `window: 256 analysed frames`, ≈17 s at 15 fps and ≈73 s at 28 frames/8 s. A camera that keeps up gets a window over seconds of history; a camera that does not gets a window over almost none of its backlog. **The number is shown so a reader can judge it.** | closed as a disclosure |
| **`tools/preflight.py` for the Tasks 1–2 review** | **was not run then and no number was claimed.** First run since is this session's, above. | closed |
| **Task 3's RSS-ratio flake** | **`test_nothing_grows_that_the_retained_records_do_not_account_for`** — failed once in this session's two full runs and once before it, in nine full-suite runs; passes in isolation. It puts a ratio bound on a measurement it shares with five other tests. **Deliberately not fixed here** — the reviewer decision is whether a shared measurement should carry a ratio assertion at all. | reviewer |
| **The gate-2s run's leak** | 182.4 MiB total growth, 175.3 MiB unexplained. **Not characterised.** See the leak section. | **M6** |
| **Clip redistribution clearance, detector licence (AGPL-3.0), AC status, hardware SKU, the `oven` false positive** | **unchanged from M0 and M1.** M2 reopened none and closed none. | as recorded at M0 |

---

## Known costs and limits of M2

- **Suite wall time ≈360 s**, against **145.6 s** for M1's committed 1034-test run
  (`bench_report.md` §M1.4) and **323.8 s** measured by Task 5 in its own session (milestone working
  log, not committed). The three-camera live-path class is ~32 s of the increase.
- **A `Repeater` delegate is invisible to `QObject.findChild` and `QObject.children()`** — it is a
  *visual* child — and a `Window`'s `childItems()` is empty because the content item is *assigned*.
  Two readers built on those APIs reported `MISSING` for every item on a render whose PNG plainly
  showed all three cards. **That is a reader that would have been a vacuous pass**, and it is the
  **fourth** instance of this shape in the project. All readers walk
  `window.contentItem().childItems()`.
- **`Math.max` passes `NaN` through.** The metrics column came back **completely empty with no QML
  warning** because at delegate creation the `Repeater` has not counted its own items, so
  `parent.height / count` was `NaN` and `Math.max` returned it. Fixed with
  `parent.height / Math.max(1, bridge.stream_ids.length)`.
- **Five live-path assertions were asserting a timing accident** — a starved collector thread yields no
  valid CPU sample in 14 s — and now assert the obligation either way: a figure or a word, never
  `0.0%`, never the accelerator sentence.
- **A `Text` keeps its `text` while `visible: false`.** Reading a hidden "no frame" sentence
  unconditionally reports it on every tile, and an assertion that "this camera says it has no frame"
  then passes for a camera with three hundred frames.
- **`str(QSize(1920, 1080))` is `"PySide6.QtCore.QSize(1920, 1080)"`** on this PySide6, so an assertion
  written against a bare `1920x1080` compares a representation instead of a measurement.
- **A mutation sweep can report a survivor that was never aimed at the covering assertion.** It did
  once here. The harness was corrected rather than the claim softened.
- **This session's machine is measurably slower than earlier in the same day** — CPU single-frame p50
  22.4–22.8 ms against 21.0–21.2 ms, and three-stream frame age 14.3–14.7 s against 10.7–11.3 s. **Not
  established** why. It is the reason this report quotes ranges and prefers the committed figure over
  the best one observed.

---

## What M2 establishes

- **Placement is proven per stream, twice over**: four fresh four-device runs at `OVERALL: PASS`, and
  50/50 transition placements whose **confirming pid matched the live pid**.
- **A presenter can move one camera while the app runs**, and the badge follows the **confirmed**
  device through `pending` to `device: GPU.0` — 219 badge samples.
- **A failed route claims no device and states why**, and the product refuses to *construct* a failed
  route with no stated reason.
- **Three cameras run three streams on three devices, each serving its own frames**, or saying in
  words that it has none. Exit 0, 1 194 frames analysed, three independent revisions.
- **The 2 s warm-transition target is missed**, and the miss is documented with its full distribution,
  the setup/wait split, and both sides of the fix.
- **Three streams on the CPU is an 11–15 second old picture that every counter calls healthy**, the
  screen says so in words per stream, and **the input-ack gate is blind to it by construction.**
- **Pending results cannot paint after a generation change** — demonstrated, and honestly labelled as
  weak evidence with a named reason.
- **The gate set is green and pasted**, WARNs included and named.
- **What is not closed is written down**, including that M0 gate 9's silent half cannot be closed on
  this machine and that the gate-2s run leaks.

### Reproduction

From a clean checkout, **two steps, in this order**, because `assets/` and `logs/` are gitignored:

1. `.venv\Scripts\python.exe tools\provision_venv.py` — the only step needing network. Until this has
   run, none of the above is reproducible: the IR and the three clips are absent and the suite fails.
2. The gate commands, then the measurements:

```
.venv\Scripts\python.exe tools\verify_sources.py
.venv\Scripts\python.exe tools\preflight.py
.venv\Scripts\python.exe -m app.main --selftest
.venv\Scripts\python.exe -m unittest discover -s tests

.venv\Scripts\python.exe -m tools.route_stress --transitions 50 --timeout-s 30 ^
    --out docs\qualification\m2-headless\route-stress-50-after-teardown-fix.json
.venv\Scripts\python.exe -m tools.route_stress --transitions 50 ^
    --out docs\qualification\m2-headless\route-stress-50-gate-2s-after-teardown-fix.json
.venv\Scripts\python.exe -m tools.route_stress --steady-seconds 120 ^
    --out docs\qualification\m2-headless\allcpu-steady-120s-hold2.json
.venv\Scripts\python.exe -m tools.route_stress --transitions 6 --timeout-s 30 --fault-stream C ^
    --out docs\qualification\m2-headless\route-stress-fault-visible-fallback.json
.venv\Scripts\python.exe tools\qualify_m0.py --requests 30 ^
    --json-out docs\qualification\m2-headless\single-frame-latency-run1.json
.venv\Scripts\python.exe -m app.main --windowed --streams 3
```

> `tools\route_stress.py` **must** be run as `python -m tools.route_stress` or under `unittest`. As a
> plain script every worker child fails to spawn, because `spawn` re-imports `__main__`.

**Every committed record under `docs/qualification/m2-headless/` has its command in its row in that
directory's `README.md`.** Nothing in this report rests on a file under `logs\`, which is
gitignored, or on the milestone working directory, which is gitignored too — where a figure from
either is quoted above, it is labelled as such.