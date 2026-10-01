# Intel Edge AI Roadshow — design

Date: 2026-10-01
Status: approved in conversation
Source of truth: `Intel-Edge-AI-Roadshow-Specification.md` v1.0 (October 1, 2026)
Target: `C:\dev\intel-edge-ai-roadshow` (clean tree; no code carried over from any prior build)

## 1. Intent

An offline Windows 11 booth application that runs real local video inference across Intel Core
Ultra CPU, integrated GPU, and NPU. A passerby understands the business use case and sees AI working
within 30 seconds. A presenter explores device routing, workload detail, and measured performance
over 3–5 minutes.

The central experience: a retail camera detecting and tracking objects, three camera inference
workloads assigned to CPU/GPU/NPU. GPU/NPU switches **move their inference jobs onto the CPU**; they
do not disable hardware. The UI must stay responsive in CPU-only mode.

Working title: **Intel Edge AI — See the workload. Move the compute.** Restrained blue/cyan
identity, text-only branding. No Intel brand assets were supplied.

## 2. Verified environment (measured 2026-10-01)

Every value below was read from this machine during design, not copied from a prior build.

| Fact | Measured value | Source |
|---|---|---|
| CPU | Intel(R) Core(TM) Ultra 9 288V, 8 cores / 8 logical, 3300 MHz | `Win32_Processor` |
| Machine | ASUSTeK COMPUTER INC. NUC14LNK | `Win32_ComputerSystem` |
| RAM | 33805881344 bytes | `Win32_ComputerSystem.TotalPhysicalMemory` |
| OS | Microsoft Windows 11 Pro, build 26200 (10.0.26200) | `Win32_OperatingSystem` |
| GPU | Intel(R) Arc(TM) 140V GPU (16GB), iGPU, driver 32.0.101.6737 | `Win32_VideoController`, `PNPDeviceID PCI\VEN_8086&DEV_64A0` |
| NPU | Intel(R) AI Boost, `PCI\VEN_8086&DEV_643E`, PnP status OK | `Get-PnpDevice -Class ComputeAccelerator` |
| Python | 3.12.10 | `sys.version` |
| OpenVINO | 2026.4.0-22959-99c81491cc3 | `openvino.__version__` |
| OpenVINO devices | `['CPU', 'GPU', 'NPU']` | `ov.Core().available_devices` |
| OpenVINO CPU name | `Intel(R) Core(TM) Ultra 9 288V` | `get_property('FULL_DEVICE_NAME')` |
| OpenVINO GPU name | `Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)` | same |
| OpenVINO NPU name | `Intel(R) AI Boost` | same |
| Display | single panel, logical desktop 2293 x 960, primary; working area (48,0) 2245 x 960 | `System.Windows.Forms.Screen.AllScreens` |
| Display scale | `LOGPIXELSX=96`, `LOGPIXELSY=96` -> 100%, 1.0x | `GetDeviceCaps(88/90)` |
| Monitor | DELL S3422DWG | `WmiMonitorID` |
| Power scheme | Balanced, GUID `381b4222-f694-41f0-9685-ff5bb260df2e` | `powercfg /getactivescheme` |
| AC status | **not readable** — no `Win32_Battery` instance is reported | `Get-CimInstance Win32_Battery` |

This is Core Ultra Series 2 / Lunar Lake. It must never be presented as Panther Lake (Series 3)
or Arrow Lake. The app shows **"Running on: Intel(R) Core(TM) Ultra 9 288V"** from the actual
discovered device, not from a hardcoded family table.

## 3. Confirmed architecture decisions

Three decisions were put to the operator and approved:

1. **Process topology — Approach C.** QML UI in the main process; one spawned child process per
   inference worker; `multiprocessing.shared_memory` bounded frame rings; `QLocalSocket` with a
   versioned JSON control protocol. Rejected spec-literal hand-rolled named pipes (Approach A:
   largest hand-rolled protocol surface, most likely source of frame-identity defects) and
   single-process threaded workers (Approach B: CPU-only contention risks the 200 ms input gate;
   workers not restartable in isolation).
2. **V1 ships no product classifier.** CLIP zero-shot classification is cut. Spec §4 V1 retail
   requires only "YOLO-class detector; tracker; shelf-exit and basket-entry rule"; a classifier is
   §5's optional second phase. This removes the `torch`/`transformers` dev dependency and the
   baked-embedding build step, and removes the failure mode where CLIP names an object by a
   vocabulary word that contradicts the detector's own label. Retail events are labelled by
   supported COCO category and never as SKUs.
3. **M6 ships unsigned.** No code-signing certificate is available. The portable package is
   delivered unsigned and signing is recorded as an unmet M6 gate with the exact step documented.

## 4. Repository layout

```text
intel-edge-ai-roadshow/
  app/
    main.py                CLI, startup gate, scenario compiler, diagnostics
    theme.py               layout tokens; no metric values or magic numbers in UI code
    overlay.py             box/label/track/zone drawing
    contracts/             versioned schemas (see §6)
    supervisor/            process manager, route controller, health watchdog
    workers/               spawn targets: cpu pool, gpu, npu
    inference/             OpenVINO adapters, preprocessing, model decoders
    scenarios/             four packs: ROI geometry, event state machines
    telemetry/             providers, aggregation, validity/freshness
    ui/                    QML screens, components, presenter content
  assets/
    scenarios/             per-pack manifests (rights, hash, ROI, annotations)
    models/                approved model artifacts, metadata, licences
    media/                 rights-cleared clips
  config/
    scenarios.json         four normalized-zone pack definitions and event thresholds
    models.json            pinned artifacts, hashes, licence, provenance
    platform_profiles.json hardware/spec facts, each with a source URL
    telemetry_map.json     discovered PDH instance -> device mapping
  tools/
    provision.py           one-time offline provisioning
    qualify_m0.py          M0 qualification runner
    verify_sources.py      content-verified source checks
    preflight.py           local PASS/WARN/FAIL gate
    capture_ui.py          offscreen 1920x1080 + DPI render harness
    benchmark_matrix.py    long-run scenario x density acceptance matrix
  tests/
  docs/
    superpowers/specs/     this document
    qualification/         M0..M6 evidence
    installation.md
    presenter-guide.md
```

## 5. Assets and rights

Spec §4 wants three 45–90 s clips per pack (12 total). Available rights-cleared material is
shorter. Decision approved: **design to the material that exists, measure and record what is
actually there, and report the shortfall as an open M5 content gate.** No footage availability is
invented.

Assets are copied from the already-downloaded, content-verified local cache and re-verified in this
tree (SHA-256 plus content type, not HTTP status). Licence strings carry over from the recorded
manifest and are written into `assets/scenarios/*.json` with hash, measured resolution/FPS/duration,
ROI coordinates, event annotations, and distribution status.

Known licences: Pexels License (commercial use, no attribution), Mixkit Free License (commercial
use, no attribution), Open Model Zoo sample media (redistribution status recorded as-is; the spec
warns that tutorial downloads are not automatically redistribution-cleared).

**Unmet M5 content gate:** clip count and duration per pack fall short of the spec. To be reported
in the M5 report, not papered over.

Reusing one clip as multiple independent replay streams is permitted for a capacity test only, and
must be labelled "replayed stream copies", never "three live cameras".

## 6. Minimum contracts (spec §8)

Versioned dataclasses plus JSON schema, in `app/contracts/`:

- `DeviceStatus` — ID, full name, type, available/validated/ready, driver/runtime detail, supported
  scenario/model IDs, unavailable reason.
- `ScenarioManifest` — ID/version, title, business explanation, source manifests, model hashes,
  class allowlist, normalized ROIs, event rules, default workload profile, quality fixture IDs.
- `RoutePlan` — generation, stream/stage -> requested device, confirmed actual mapping, transition
  state, errors.
- `MetricSample` — name/value/unit, device/process scope, provider, timestamps/window,
  validity/staleness, measurement kind.
- `BenchmarkRecord` — workload fingerprint, environment manifest, raw samples, trial summary, mode,
  exclusions, comparison eligibility.

Operator settings persist separately from immutable scenario/model manifests.

## 7. Routing and inference

Stream-level assignment per spec §5. Camera A -> CPU, B -> GPU, C -> NPU. Density 3 repeats the
three ownership slots; density 6 repeats them twice. Each stream does real independent processing
and shared decode behaviour stays constant across comparisons.

Toggle semantics are exactly the spec §5 table. Explicit `compile_model(device)` targets only —
**no `AUTO`, no `HETERO`, no `MULTI`** in the primary routing demo. `EXECUTION_DEVICES`, re-read
after compile, is the only placement proof the UI may display.

Switching protocol (spec §5): request a new generation -> UI shows "Moving Camera B inference to
CPU..." with pending vs actual -> old route stops accepting new submissions, outstanding requests
finish or time out -> new target begins submitting -> route acknowledged, comparison windows
cleared, warm-up/settling samples marked. Toggle during a transition coalesces. Warm transition
target 2 s, measured; longer stays visibly pending with a documented reason. On target failure the
UI states the failure and the real fallback route and recovers to the last healthy route.

Per-request fields (spec §6): `session_id`, `scenario_id`, `stream_id`, `frame_id`, `source_pts`,
`source_loop_id`, monotonic capture/decode timestamps, `route_generation`, requested and actual
device, model hash. Per result: start/end times, boxes, class IDs, scores, error status.

A result is bound to its own `frame_id` and is discarded if its `route_generation` is obsolete. A
late result never paints onto a newer frame. Queues hold at most 2 waiting frames, newest-frame
policy in realtime mode, explicit per-stream drop counters. The hero renders the analyzed frame
with its own matching boxes; no stale boxes over unrelated frames.

V1 uses CPU decode/preprocessing for a stable baseline. Static shapes only — the NPU rejects
dynamic shapes. 640x640, batch 1, detector export without embedded NMS, one documented CPU
postprocessing path.

### 7.1 Model selection and pinning (spec §6)

Baseline candidate: **YOLO11n FP16**, statically exported to OpenVINO IR, batch 1, 640x640. It is
selected and pinned in M0 only after each of the following is validated on this machine and the
result recorded:

1. **Licensing** of the exact weights artifact, recorded in `config/models.json`.
2. **Output decoding** — the actual tensor names, shapes, and layout of the exported IR. The spec
   explicitly warns that not all YOLO variants share one output layout, so the decoder is written
   against the observed tensors, not against an assumed `(1, 84, 8400)` convention. If the export
   embeds NMS, that is recorded and the CPU postprocessing path is documented accordingly.
3. **Compile on each of CPU, GPU, and NPU** at the pinned static shape.
4. **Detection sanity** on a small annotated fixture, per device.
5. **Colour order, letterbox, normalization, coordinate transform, class mapping, threshold, and NMS
   behaviour verified identical across devices** — a test, not an assumption.

Artifact cache keys contain model hash, device identity, runtime version, driver/compiler
information, and compile properties, and invalidate on mismatch. Compiled blobs are never
distributed as universally compatible.

Weight storage precision and actual execution precision are reported separately, and execution
precision appears only when known. 416x416 is not evaluated unless the 640x640 baseline fails to
meet the frame-age target; if used, it becomes a separately named workload preset.

INT8 is out of scope for V1. If later pursued it is a separately named artifact requiring
calibration data, a reproducible conversion, and a before/after quality report, and comparisons
across it are labelled as different optimized artifacts.

## 8. Telemetry honesty rules

| Metric | Provider | Rule |
|---|---|---|
| System CPU % | `psutil.cpu_percent` over all logical cores | whole-machine, named interval; not process-only |
| App CPU % | process-tree sum (UI + every worker PID) | normalized to 0-100% of all logical capacity; null if the PID set is unknown |
| GPU compute % | PDH `GPU Engine`, `engtype_Compute`, instance matched by our adapter LUID | scope and aggregation documented; engines are never summed into a bogus total |
| NPU % | PDH `GPU Engine`, `engtype_Neural` for the AI Boost LUID, if present | otherwise **null** -> "Utilization unavailable" |
| Inference completions/s | counted by us | per device and per stream; counts real work |
| AI FPS | unique frames with completed detection per second | per stream; aggregate explicitly labelled |
| Frame age | monotonic ingress-to-display of the matching result | p50 and p95 |
| Inference latency | submit to completion | distinct from full application frame age |
| Dropped analysis frames | our counters | categorized separately per stream |
| Business events | event engine | with the scenario definition shown |

Hard rules, all traceable to spec §7:

- When the NPU counter is unavailable, the NPU card shows **inference activity** — completions/s,
  latency, in-flight — and never a utilization number.
- An application-derived duty cycle (`inference_seconds / elapsed`) may appear only in diagnostics,
  explicitly named as an application-derived estimate. It is never shown under a utilization label.
- Adapter LUID is recorded so an external GPU can never be confused with the Intel integrated GPU.
- Power, temperature, TOPS, and energy figures are omitted. They are never inferred from
  utilization, and hardware-spec TOPS stays separate from measured application FPS.
- Sampling 4 Hz; headline values 2 Hz over a rolling 5 s window; 60 s of chart history; raw
  timestamped samples retained for export. Values older than 2 s are `stale`. Unavailable is
  `null` and renders as unavailable, **never zero**.

**No simulated metrics ship in release mode.** A development banner is mandatory for any simulated
value, and `app.main` refuses to start in release mode with a simulated provider registered.

Comparisons: identical model artifact, input size, confidence threshold, sampled frames, stream
count, analysis rate, decode path, and thread budget. Throughput ratio = accelerated AI FPS / CPU
AI FPS, guarded against zero. CPU change expressed in **percentage points** ("-12 pp system CPU"),
never as an ambiguous "12% faster". No improvement and regressions are shown accurately. Comparison
cards are never populated before measurement. Saved records carry trial ranges and the median, are
compared only on matching workload fingerprints, and older records are marked "previous run" with
machine and timestamp.

## 9. UI

QML / Qt Quick on a 1920x1080 logical canvas, scaled to fit while respecting Windows DPI.
~68% width video / 32% metrics; 12% header+title, 66% video+metrics, 22% pipeline+controls. Hero
dominates; two thumbnails secondary. Type sizes per spec §3 (36-44 px headlines, 48-64 px main
numerals, 24-28 px main labels, >=18 px detail). Tabular numerals, stable widths, high contrast.
CPU/GPU/NPU carry consistent colours **plus** text labels — never colour alone. First-level screen
is limited to three compute cards, three outcome metrics, and one comparison statement.

Layout tokens live in `app/theme.py`; no metric values or magic numbers in UI code.

Clicking a thumbnail changes the hero without changing inference routing.

**"Recorded footage • live inference" is displayed continuously.** This label is mandatory and
must not be omitted. Any optional recording of a prior run shows "Recorded demonstration — metrics
are not live" and disables live device controls.

Tooltips open after ~350 ms, persist while hovered, are keyboard-focus accessible; click pins an
explanation panel; Escape closes. Long explanations never obscure the hero. Required explanation
content per spec §3: YOLO detection, tracking, zone rules, CPU, GPU/NPU, and metric definitions
(numerator, denominator, window, source, freshness, scope).

Modes: attract (rotate every ~45 s after 90 s idle, each start all-devices, measured A/B autoplay
off by default), presenter (any input pauses rotation), compare (controlled replay and measurement),
diagnostics (hardware, assets, drivers, logs, routing, telemetry provenance). Shortcuts F11, 1-4, G,
N, C, A, Space, R, Escape — never firing inside text fields. Reset restores the scenario's approved
configuration and clears business events, not driver or OS settings.

### 9.1 Display conflict and how the DPI gate is measured

Measured on this machine: the logical desktop is 2293 x 960 at 100% scaling, with a working area of
2245 x 960 after the taskbar. The panel is a DELL S3422DWG, but the desktop mode is not a 1:1 native
match for that panel and is not recorded as one.

Two consequences, both measured rather than assumed:

- A 1920 x 1080 window **cannot be shown** on the live desktop, because usable height is 960 px.
  The spec §3 design canvas is therefore only reachable through the offscreen harness.
- The spec requires letterboxing rather than clipping when the window does not fit. The live layout
  must stay within 2245 x 960 without cutting controls.

Resolution: the DPI acceptance gate is **measured, not asserted**. `tools/capture_ui.py` renders the
real QML in an offscreen window at exactly 1920 x 1080 under `QT_SCALE_FACTOR` 1.0 / 1.25 / 1.5 / 2.0,
writes PNGs, and programmatically asserts no overlap or clipping. The live booth runs at the native
2293 x 960 desktop, letterboxed. Both facts go into the M0 qualification report as environment
measurements, so no one can later read a passing 1080p screenshot as a claim about this monitor.

## 10. Verification

Unit tests: preprocessing and coordinate recovery, YOLO output decoding, event state machines,
tracker reset, route-generation filtering, queue bounds, workload fingerprinting, percentage-point
math, and unavailable/stale metric handling.

Integration tests: worker kill and recovery, IPC frame identity, and scenario transitions.

Hardware gates that only this machine can satisfy: accelerator availability, model compilation,
actual execution, switching, telemetry, sustained performance, packaging qualification. A CPU run
is never evidence of NPU readiness. A qualification matrix (scenario x model artifact x CPU/GPU/NPU
x machine SKU x driver/runtime version) records each cell as not tested / passed / failed with
evidence.

## 11. Milestones and gates

Sequenced M0 -> M6 as the spec requires, each ending runnable with a change report, evidence, and
open blockers. A mock screen is never proof of hardware behaviour.

- **M0** — Windows diagnostic runner + qualification report for this exact machine. Exact SKU, RAM,
  Windows build, driver versions, OpenVINO/Python versions, power state, monitor config. Discover
  CPU/GPU/NPU, run the same detector on each, probe telemetry providers. Gate: each required target
  compiles and completes >=100 requests, detection output passes a small annotated sanity fixture,
  observed device activity and runtime metadata support the assignment, CPU fallback works. All
  results are real measurements. If the NPU fails, diagnose before any UI polish.
- **M1** — one truthful retail stream: overlays, explicit CPU/GPU/NPU selector, tracking, ROI event
  logic, basic metrics, recorded-footage label, model explanation.
- **M2** — three streams, routing table, bounded queues, route transitions, workload badges, 1/3/6
  presets. All four toggle combinations; 50 alternating warm route changes without crash, leak
  trend, stale overlay, or mislabelled device.
- **M3** — complete QML layout, hero, cards, pipeline, tooltips, presenter drawer, keyboard, fullscreen,
  attract mode. Screenshot review at 1080p and each DPI scale; <200 ms input acknowledgement in
  CPU-only mode.
- **M4** — metric provenance, per-device completion rates, system/app CPU, validated GPU/NPU counters
  where available, comparison runner, fingerprinting, JSON/CSV export.
- **M5** — four complete vertical packs, each with an annotated acceptance set (>=20 positive events,
  >=20 relevant negative opportunities) and >=90% precision and recall on that curated set, reported
  as demo-fixture quality rather than a real-world claim.
- **M6** — unsigned portable package, bundled assets, setup guide, presenter guide, diagnostic export,
  known-good environment manifest. Clean install, cold provision, warm launch, four-hour offline
  soak with scenarios and device modes rotating, and the failure-injection matrix: missing NPU,
  missing clip, corrupt model, worker termination, sleep/resume, monitor reconnection.

## 12. Known dependencies and blockers carried into the build

| # | Item | Status | Handling |
|---|---|---|---|
| 1 | Three 45-90 s clips per pack | **unmet M5 content gate** | measured durations recorded; shortfall reported |
| 2 | OMZ sample-media redistribution | unverified | recorded as-is; not claimed as booth-redistribution clearance |
| 3 | Staged shelf-to-basket footage with clear transfer and no prolonged occlusion | **unmet** | if event-quality acceptance fails, deliver "items entering checkout zone" as a labelled reduced-scope milestone and do not mark the requested scenario complete |
| 4 | Code signing | **unmet M6 gate** | package delivered unsigned; signing step documented |
| 5 | Monitor is 2293x960 at 100% scale, not 1920x1080 | measured environment fact | offscreen 1920x1080 render harness for the DPI gate; live layout letterboxed to 2245x960; both reported in M0 |
| 6 | NPU utilization counter presence | to be determined in M0 | null and "Utilization unavailable" if absent; never a substitute figure |
| 7 | Intel brand assets | not supplied | restrained blue/cyan, text-only branding |
| 8 | AC status is not readable on this machine | measured: no `Win32_Battery` | recorded as a fixed environment condition in the A/B protocol; the spec's "record AC status" row is satisfied by stating it is unavailable, and the Balanced power scheme is recorded instead. Thermal readings are likewise not inferred. |

## 13. Out of scope for this release

Live camera input (a later enhancement; dependable prerecorded footage is the default), facial
identification, patient diagnosis, weapon targeting, cloud dependency, live LLM, INT8 quantization
(optional after baseline and requiring calibration data plus a before/after quality report), USB
camera support, per-stage detector/classifier routing, measured power, six-stream showcase, and
validation on Arrow Lake or Panther Lake.
