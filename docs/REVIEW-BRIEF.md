# Review brief — Intel Edge AI Roadshow

**Audience:** an automated reviewer (or a new human engineer) who has not seen this project before.
**Purpose:** give you enough context to review the code critically **without re-deriving the hardware and
measurement facts that took six milestones to establish**, and without repeating the failure modes that have
already fooled this project's own tests.

Read this whole file before opening a source file. Roughly half of it is warning.

---

## 1. What this is

A Windows 11 booth demo that runs a real object detector across an Intel CPU, GPU and NPU, concurrently,
on recorded footage, and shows the results on a 1920×1080 QML screen. It is a **local demonstration**, not a
production checkout system: no facial identification, no cloud, no LLM, no network at runtime.

| | |
|---|---|
| Repo | `https://github.com/timshawjr/intel-edge-ai-roadshow` (**private**) |
| Branch | `main` |
| State at handover | 86 commits, working tree clean |
| Tests | **1687 passing**, `exit 0`, suite ~617 s |
| Packaging | offline bundle at `dist/M6-bundle` (**unsigned**, `dist/` is gitignored) |
| Spec | `Intel-Edge-AI-Roadshow-Specification.md` — **not in the repo**; held by the operator |

Milestones M0–M6 are complete. **"Complete" here means every milestone ended runnable with recorded
evidence — not that every gate passed.** See §3.

---

## 2. How to verify in five minutes

```powershell
cd C:\dev\intel-edge-ai-roadshow
.venv\Scripts\python.exe -m app.main --selftest        # expect: PASS, exit 0
.venv\Scripts\python.exe tools\verify_sources.py       # expect: 6 PASS, 0 FAIL
.venv\Scripts\python.exe tools\preflight.py            # expect: 17 PASS, 1 WARN, 0 FAIL — Limited
.venv\Scripts\python.exe -m unittest discover -s tests # expect: Ran 1687 tests, OK, exit 0  (~10 min)
```

To watch it:

```powershell
.venv\Scripts\python.exe -m app.main                    # one camera, retail pack
.venv\Scripts\python.exe -m app.main --streams 3        # three cameras: A CPU, B GPU, C NPU
.venv\Scripts\python.exe -m app.main --pack smart_city
.venv\Scripts\python.exe -m app.main --clip bottle-detection
```

**Verify gates with a bare command.** Piping through `Select-Object`/`Select-String` masks the real exit code
— that has hidden failures in this project before.

---

## 3. Gate status — the headline

**18 gates: 11 passed, 2 passed only in documented form, 3 missed/unmet, 2 never tested.**
Full table: `docs/qualification/M6-qualification.md`.

| gate | status |
|---|---|
| Three devices compile, ≥100 requests each | passed |
| Detection passes the annotated fixture | passed |
| CPU fallback when GPU/NPU unavailable | **partial — visible half observed, silent half never tested** |
| Retail event ≥90 % precision/recall on a curated set | **not tested — no curated set exists** |
| People-in-zone presence, 2 per clip | passed (18/18 windows) |
| Input acknowledgement <200 ms | passed **as a floor**, calibrated at `cpu_count − 2` |
| GPU/NPU utilisation attributed | **passed** — per-PID counter |
| 1920×1080 at four DPI scales | passed with documented failures |
| **Legibility at 2–3 m** | **2 m cap pass, 3 m fail; x-height fails at both** |
| **Warm transition ≤2 s** | **MISSED — p50 2.628 s, 37/50 over at the real 2.0 s default** |
| 50 alternating transitions | passed, leak qualified (leak fixed in M4) |
| **Content: 12 clips, 4 packs, rights-cleared** | **NOT MET — 9 clips, 4/2/2/1 per pack, 0 rights-cleared** |
| Export excludes imagery | passed |
| Recorded mode cannot look live | passed |
| Offline bundle + manifest | passed, **unsigned** |
| Environment read from machine | passed |
| Presenter guide checked against CLI | passed |
| **Code signing** | **UNMET — no certificate** |

**Do not upgrade any of these.** If your review finds a `passed` that the evidence does not support, that is
the highest-value finding you can make.

---

## 4. Hardware and environment (measured, do not re-derive)

| | |
|---|---|
| CPU | Intel Core Ultra 9 288V ("Lunar Lake"), 8 cores / 8 logical, 3300 MHz |
| GPU | Intel Arc 140V (iGPU), `PCI\VEN_8086&DEV_64A0`, driver `32.0.101.6737` |
| NPU | Intel AI Boost, `PCI\VEN_8086&DEV_643E`, driver `32.0.100.5540` |
| RAM | 33 805 881 344 bytes (33.8 GB) |
| OS | Windows 11 Pro, build `26200`, 25H2 |
| Monitor | **DELL S3422DWG**, 80 cm × 33 cm, native 3440×1440, **running 2293×960** |
| Power | Balanced; **AC status unreadable** — `Win32_Battery` returns 0 instances |

**There is no defensible SKU.** Three records disagree and none wins:
`Win32_ComputerSystem` model `NUC14LNK LN`; hostname `NUC14PROAI`; PCI `SUBSYS_88F21043`. **Never claim a SKU.**
The environment manifest preserves all three.

**The monitor matters.** The 1920×1080 canvas letterboxes into a 2293×960 desktop, giving
**0.310 mm per logical pixel** — this is the figure every legibility number depends on.

**18 KB of "the platform can't do that" was wrong once already.** See §6.8.

---

## 5. Architecture

**Process topology (deliberate, not incidental):**

- **One UI process** (Qt/QML) — owns the window, the bridge, telemetry and the render loop.
- **One worker process per stream**, spawned via `multiprocessing` (Windows `spawn`).
- **One `BoundedFrameQueue` per stream** carrying `FramePacket`s.
- **One `ClipSource` + decode thread per camera** — three cameras are three independent clips, not one clip
  replayed three times.
- **`QLocalSocket`-style control is not used**; control is `multiprocessing.Queue` + Qt signals.

**Data path:** `ClipSource` → `FramePacket` (identity-carrying) → worker → infer → `InferenceResult`
(carries the `FramePacket` it came from) → reader thread → `Tracker` → event machine → `UiBridge` (Qt signal)
→ QML.

**Key invariant:** a result **carries its frame identity**. A late result for a superseded generation is
**discarded and counted**, never painted. A result for a superseded **loop** is discarded likewise.

**Placement proof:** `EXECUTION_DEVICES` is read **after** the first inference, **from the request's own
compiled model**. Requesting NPU is not evidence of NPU. The UI shows the **confirmed** device.

**Retail event scope:** the delivered event is **people-in-zone presence, 2 per clip** — **not** a people
count, **not** a dwell measurement. `app/engine/retail_config.py`'s `scope_note` is the canonical wording.

---

## 6. Hard-won runtime facts — each of these cost time to discover

**Do not "fix" these. They are measured behaviour, not bugs.**

1. **`openvino==2026.4.0` has no `openvino.runtime` submodule.** Use `import openvino as ov`.
2. **The detector output port is unnamed.** The tensor is filed under the port-index **string** key `'0'`.
   `infer` accepts an int index **or the compiled model's port object**, and **rejects the IR model's**.
   `get_tensor(0)` raises. Use the compiled port object or `get_output_tensor(0)`.
3. **`EXECUTION_DEVICES` is inconsistent by device:** NPU is a bare string `'NPU'`, CPU is `['CPU']`,
   GPU is `['GPU.0']`. **Always** `normalise_execution_devices` / `is_device`; never read it raw. A test that
   compares against `"CPU"` as a `list` will silently compare characters — one test did.
4. **Box encoding is `cx, cy, w, h`** — not `xyxy`.
5. **The letterbox pad is 114** (Ultralytics convention), 43.75 % of the 640×640 canvas.
6. **Static shapes only** — the NPU rejects dynamic shapes. Batch 1, 640×640.
7. **`time.monotonic()` has ~16 ms resolution on this machine.** 20 000 consecutive calls returned **one**
   distinct value. Use `perf_counter` for elapsed time. Windows `WaitForSingleObject` also has a ~15.6 ms tick
   floor: `Condition.wait(1 ms)` measures 15.4 ms while `time.sleep(1 ms)` measures 1.8 ms.
8. **`0xC0000005` from in-process PDH was a bug in this repo, not Windows.** It was a **3-argument binding of
   the 4-argument `PdhGetFormattedCounterValue`**, plus 24-byte `PDH_FMT_COUNTERVALUE_ITEM_W` array items read
   as 16 bytes. Correctly bound, per-process GPU/NPU utilisation works via
   `\GPU Engine(*)\Utilization Percentage` filtered by PID. **The project believed for two milestones that
   this was impossible.**
9. **`DeviceRegistry.describe()` costs 3.3–3.8 s cold.** It is startup-only; it must never touch a per-frame or
   per-sample path.
10. **`_on_result` runs on the UI main thread**, not the reader thread its docstring long claimed.
11. **`--streams 3` is refused in headless** with a stated reason (exit 2). It used to be silently ignored.
12. **`FramePacket` transport is pickled 6.2 MB at 1080p.** The display copy is now **960 px wide**, reduced
    **after** inference with boxes scaled; the detector still sees full resolution. **Never downscale what the
    detector consumes** — that trades a speed-up for lost detections.

---

## 7. The honesty regime — this project's actual subject

The code is ordinary. The discipline around measurement is the point, and it is what you should review.

**Seven failure classes have occurred, repeatedly.** Count them in the tests you read:

1. **A test that passes vacuously** — 22+ instances. A test that asserts on something that is always true.
   Examples: `assertGreaterEqual(x, 0)`; a routing test that passes for *any* supervisor; a boundary test that
   reads its own bound from the token under test.
2. **A confident number with no measurement behind it** — 20+ instances, **including the controller's own**
   ("3.5 MiB = 0.26 % of RSS" was 3.3 %).
3. **An empty result from a lookup, and a passing assertion over the empty result** — 5+ instances.
   A `Repeater` delegate is **invisible** to `QObject.findChild` and `children()`; `Math.max` passes `NaN`
   through; a **1×1 unsized window** made every click land on nothing while the tests passed.
4. **A rendered string wrong while everything around it is right.** A thumbnail caption read `Camera B · -1`
   with a correct badge and a correct picture above it.
5. **A docstring describing a mechanism that no longer exists** — 3 in one round.
6. **The measurement process polluting the measurement.** **66 orphaned `python.exe` processes** from
   interrupted runs poisoned two milestones of timings. This is why every timing task must assert a clean
   machine first.
7. **The instrument being wrong at the scale it is not tested at.** `glyph_pixels` was correct at 100 % and
   reported **0 for drawn items** at 125/150/200 %.

**Mutation testing is the standing countermeasure.** Every test is expected to have a constructed fault that
makes it fail. Where the reports cite "N mutants, N killed", that is the claim to spot-check.

**Verify the negative, not just the positive.** A good test here proves the *absence* too: no stale overlay,
no `0` for unavailable, no device claimed before inference confirms one, no image bytes in an export.

---

## 8. Where the evidence lives

| what | where |
|---|---|
| Qualification reports M0–M6 | `docs/qualification/M{0..6}-qualification.md` |
| Machine-readable gate matrix | `docs/qualification/qualification-matrix.csv` |
| Per-device headless runs | `docs/qualification/m1-headless/`, `m2-headless/` |
| DPI captures at 4 scales | `docs/qualification/m3-capture/*.png` + `*.json` |
| Legibility review | `docs/qualification/M3-readability-review.md`, `M4-legibility-review.md` |
| Utilisation attribution | `docs/qualification/M4-utilisation-attribution.md` + control JSON |
| Content gate | `docs/qualification/M5-content-gate.md` |
| Benchmark records | `docs/qualification/m5-records/` |
| Sustained trials | `docs/qualification/m4-sustained-trials.json` |
| Leak probe, before/after | `docs/qualification/m4-leak-{before,after}-{2s,30s}.json` |
| Install + signing | `docs/install/SETUP.md`, `SIGNING.md`, `KNOWN-GOOD-environment.json` |
| Presenter guide | `docs/PRESENTER-GUIDE.md` |
| Content sourcing brief | `docs/install/CONTENT-REQUEST.md` |
| Plans (per milestone) | `docs/superpowers/plans/` |

**Session ledgers with per-decision rulings** live in `.superpowers/sdd/<plan>/progress.md`. **These are
gitignored** and therefore exist only on the machine that produced them — do not treat them as committed
evidence, and do not assume a reviewer can open them.

---

## 9. Open gates and known defects — review these first

Ordered by size, not by ease.

### 9.1 Content is the largest open risk
- **9 clips** provisioned against 12 required; **4 / 2 / 2 / 1** per pack against 3 each.
- **0 rights-cleared.** Six carry `CC BY 4.0` sidecars = `licence file present; clearance not established`.
  Three have **no sidecar at all**: `retail-checkout`, `store-aisle-detection`, `one-by-one-person-detection`.
- **5 of 9 clips are outside the 45–90 s band.** Three were used anyway, by operator decision, and are
  recorded as a **deviation**, not as in-band.
- **The retail shelf-to-basket scene does not exist** in the provisioned material. The delivered retail event
  is presence, which is a documented reduction.
- **`python -m tools.content_gate` exits 1.** That is correct behaviour while unmet.

### 9.2 Every pack measures presence only
The spec asks smart city for **directional line crossing** and **stopped-object dwell**, medical for
**waiting-zone dwell**, depot for **loading-zone entry and dwell**. The engine has **no line-crossing kind and
no dwell-event kind** — only `zone_occupancy` and `candidate_transfer`. Every pack currently reports presence
and says so in its manifest and UI.

**If any user-visible string claims "line crossing" or "dwell", that is a defect.** None should.

### 9.3 The 2 s warm-transition target is missed
p50 **2.238 s** (M2 session), **2.261 s** (30 s-deadline artefact), **2.628 s** (real 2.0 s gate artefact);
**37 of 50** over budget at the real default. The gap is queue wait, not inference. Three figures exist
because three runs exist — **do not quote one without saying which**.

### 9.4 Legibility at 3 m fails
44 px body text = **16.8′ at 2 m, 11.2′ at 3 m**; x-height **12.2′ / 8.1′**. Threshold is 16′.
The 55-inch panel that would fix it at 3 m is a **procurement requirement**, not a result. A mutation kills
the 55-inch default so a requirement-only test cannot pass.

### 9.5 The input-ack gate is blind to backlog — and that is proven
8/8 windows passed the gate while **6/8 had a stale picture**. Three cameras on their own devices: backlog
**1 frame**. Three on the CPU (spec §5 row 4): backlog **587–752**. Same window, same process.

**Do not "fix" the gate by tightening a threshold.** The finding is that it measures UI-thread tick lateness,
which is insensitive to inference backlog by construction.

### 9.6 Silent fallback was never observed
All three devices execute where asked on this machine, so a *silent* fallback does not occur naturally, and
`compile_failure` kills the worker before it runs anything. **M0 gate 9's silent half cannot close here.**
The visible half is delivered and measured.

### 9.7 The bundle is missing its own guides
`dist/M6-bundle/docs/` contains only `SETUP.md`. It does **not** contain `PRESENTER-GUIDE.md`, `SIGNING.md`,
or `KNOWN-GOOD-environment.json`, and `SETUP.md` references the signing document. All four exist in the repo.
**The package is not self-contained for a presenter.** Fix is a few paths in `source_paths()` plus a rebuild.

### 9.8 Smaller, specific defects
- `tools/compare_records.py`'s documented invocation `python tools/compare_records.py …` **fails**
  (`ModuleNotFoundError: app`). The working form is `python -m tools.compare_records`.
- `tools/probe_telemetry.py` **rewrites `config/telemetry_map.json`** as a side effect. Running it dirties the
  tree. That file is stale (per-boot LUIDs) and is preflight's single WARN.
- `build_bundle.py --verify` cannot run alone; `--destination`/`--wheelhouse` are required, contradicting its
  own help.
- The presenter guide prints *"M2 measured p50 2.238 s"*; **no committed M2 file contains 2.238** (committed:
  2.261 and 2.628).
- `test_route_stress`'s RSS-ratio assertion carries an **8 MiB tolerance on the inversion direction only**
  (5 % of the 154.1 MiB pre-fix retention). **This gate was changed and flagged for review.**

### 9.9 Known flakes (pre-existing, not fixed)
- `tests/test_main.py` asserts `"0.0%" not in …`; it fails when CPU reads exactly `30.0 %`.
- `test_route_stress…test_nothing_grows…` is a shared-measurement ratio assertion; observed failing ~2–3 times
  in 15 full runs, passing in isolation.
- If you see either, **do not "fix" it by loosening the assertion** without saying so.

---

## 10. Licensing and provenance

| item | state |
|---|---|
| `yolo11n-fp16` weights | **AGPL-3.0** — strong copyleft, obligation **not evaluated** |
| Repo `LICENSE` | **absent** |
| `README.md` | **absent** |
| `NOTICE` / `ATTRIBUTION` | **absent** |
| Six sample clips | `CC BY 4.0` (intel-iot-devkit/sample-videos) — **attribution required** |
| Three original clips | `unverified`; `retail-checkout` provenance recorded as Pexels 32658425 |
| Two manifest entries say | *"Open Model Zoo sample media"* — **inconsistent** with the CC BY 4.0 sidecars |
| Code signing | **unmet**; `signtool` commands documented in `docs/install/SIGNING.md` |

**This is why the repo is private.** Publishing it would be a licensing decision made by accident.
**Do not weaken a security control** (Defender exclusions, ASR, execution policy, AppLocker) to work around the
unsigned state — that is forbidden by the project's own rules.

---

## 11. Do not re-derive these

A reviewer that "rediscovers" any of the following will waste a day and possibly introduce a regression:

- that the output port is unnamed, or that `get_tensor(0)` raises;
- that `EXECUTION_DEVICES` has three different shapes;
- that boxes are `cx,cy,w,h` or that the pad is 114;
- that `time.monotonic()` is too coarse here;
- that `0xC0000005` means PDH is unusable (it does not);
- that the NPU is slower than the GPU on single-frame p50 (it is — GPU 3.09–3.11 ms vs NPU 3.96–4.01 ms
  settled, and 307–310 vs 246–249 fps isolated);
- that the NPU's 5.2–5.9 ms is a **cold** figure (settled is 3.96–4.01 ms);
- that throughput at the booth's 15 fps **ranks nothing** (all three devices deliver 300/300; the source sets
  the rate);
- that the retail event is presence, not transfer.

---

## 12. Where a review earns its keep

Ranked by expected value:

1. **`app/supervisor/runner.py`** — the hardest file. Generation scoping, retirement, `_retired_queues`,
   the timeout path, `_halt_threads`, coalescing. It has had the most defects and the most fixes.
2. **`app/telemetry/collector.py`** — the null-not-zero contract, the shared `_Tick`, the bounded reason
   strings. Read the docstrings against the code: three separate stale-string defects lived here.
3. **`app/ui/*.qml` + `app/ui/bridge.py`** — the binding traps. **Every `Slot` call in a binding is suspect**
   unless it reads only constants. There is a standing line-oriented guard; judge whether it can be defeated.
4. **`app/engine/events.py`** — the state machine. Check that the docstrings still describe the code; several
   did not, and one described a race that could not happen.
5. **`tools/capture_ui.py`** — the measuring instrument. If it is wrong, four milestones of legibility claims
   are wrong. Check the device-space scaling and the `settled` semantics.
6. **The test suite, for vacuity.** This is the single highest-yield review activity on this repo.

**A useful reviewer question, asked of any test: "what fault makes this fail?"**
If you cannot answer it, the test is not evidence.

---

## 13. Milestone history in one line each

- **M0** — hardware qualification: three devices, ≥100 requests each, placement proven, 22 commits.
- **M1** — one real stream end to end: clip → infer → track → events → QML, live windowed path made testable.
- **M2** — three streams, four routing combinations, 50 alternating transitions, per-stream accounting.
- **M3** — 1920×1080 at 100/125/150/200 %, offscreen capture harness, attract mode, presenter drawer,
  readability review (**which found the screen unreadable at 3 m**).
- **M4** — sustained paired trials, per-process utilisation attribution (**closed M0's gate 2**),
  the 2 s-timeout leak (**+153.7 → −10.4 MiB**), legibility redesign by removal, input-ack matrix.
- **M5** — benchmark records with an 11-field fingerprint, export that excludes imagery by default,
  comparisons that refuse a mismatch, recorded-demonstration mode, **content gate measured and unmet**.
- **M6** — offline bundle with per-file SHA-256 and a proven network-free install, environment manifest,
  presenter guide checked against the CLI, diagnostic export, **signing documented and unmet**.

---

## 14. One last thing

The single most important property of this codebase is not performance or architecture. It is that
**a number on the screen is either measured, or it says it cannot be measured.**

If you find a place where that is not true — a `0` where a value is unavailable, a device badge that is not
confirmed, a percentage with no sample behind it, a docstring describing a mechanism that no longer exists —
you have found the kind of defect this project has spent six milestones hunting. **Report it.**
