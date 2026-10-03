# M2 — Three Streams and Device Toggles — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three camera inference workloads running concurrently — Camera A on CPU, Camera B on GPU, Camera C on NPU — with GPU and NPU switches that *move their inference jobs onto the CPU*, real route transitions, and 50 alternating warm route changes that survive without a crash, a leak, a stale overlay, or a mislabelled device.

**Architecture:** Extends M1's `Supervisor` from one stream to three, each with its own bounded queue, worker process and `RoutePlan` entry, all under one `route_generation`. A toggle opens a new generation, the old routes stop accepting submissions, in-flight results finish or time out, and only then does the UI acknowledge the new route. The same offscreen testability rule that closed M1's live-path defects carries through: every claim is proven by driving `app.main`, not a harness.

**Tech Stack:** Python 3.12.10, `openvino==2026.4.0`, `PySide6==6.11.2`, `numpy==2.5.3`, `opencv-python==5.0.0.93`, `psutil==7.2.2`. Tests use stdlib `unittest`; `pytest` is deliberately not added.

**Spec:** `docs/superpowers/specs/2026-10-01-intel-edge-ai-roadshow-design.md` and `Intel-Edge-AI-Roadshow-Specification.md` §5 (routing table and switching protocol), §6 (per-request identity), §9 (reliability), §10 M2.

## Global Constraints

- Pinned exactly; stdlib `unittest`; no `pytest`; no new dependency.
- Explicit `compile_model(device)` targets only. **No `AUTO`, `HETERO`, `MULTI`.**
- `EXECUTION_DEVICES`, re-read **after** the first inference from the request's own compiled model, is the only proof of placement. It is inconsistent: NPU bare string `'NPU'`, CPU `['CPU']`, GPU `['GPU.0']`. Always `is_device` / `normalise_execution_devices`.
- **Unavailable is `null`, never `0`.** Every optional metric renders through `bridge.display_text` / the per-card `unavailable_notice`; the CPU card must never wear the accelerator's "Utilization unavailable" wording.
- **GPU/NPU utilisation is unavailable** in the runtime path and a machine-wide peak is **not attributable to this process**. Cards show inference activity. Never a number.
- Static shapes: batch 1, 640×640, pad **114**, boxes **`cx,cy,w,h`**, `has_embedded_nms=False`.
- The retail event is **people-in-zone presence**, 2 events per clip — **not** a people count, **not** a dwell measurement. `app/engine/retail_config.py`'s `scope_note` is the exact wording to reuse.
- Nothing in `app/` imports from `tools/`. Layout tokens in `app/theme.py`.
- `time.monotonic()` has ~16 ms resolution here; use `perf_counter` for elapsed time. A published timing figure needs repeats, stated as a range.
- **Frame age and inference cost are separate claims and are never merged.** The NPU's p95 frame age is ~2.5 s while its inference cost is ~5.9 ms; the gap is queue wait.

**M1 facts this milestone builds on, all measured:** placement confirmed on all three devices
(CPU `CPU`, GPU `GPU.0`, NPU `NPU`, `fell_back: false`, 121/121 analysed, 0 refused, 0 drops each, ai_fps
15.10–15.11); single-frame p50 **CPU 20.99–21.16 ms, GPU 3.30–3.47 ms, NPU 5.92–5.99 ms** — the NPU is
slower than the GPU, and that is a real result; input acknowledgement p50 0.026–0.041 ms, p95 12.1–15.3 ms,
worst tick 22.6–24.2 ms at `max(2, cpu_count() - 2)` saturators; `Supervisor` already carries
`frames_refused` with reasons and reconciles `accepted + refused + drops == attempts`; `LiveWindowedPath`
already drives `app.main.main([... , "--windowed"])` and reads the live scene graph; the QML Slot-binding
guard exists and must stay green.

## Review Focus

1. **A result from a superseded generation painting onto a newer frame.** Discard by `route_generation`;
   never draw. The overlay already filters coasted tracks — the same discipline now applies to generations.
2. **The UI claiming GPU or NPU while the job runs on the CPU.** This is the entire point of the product. The
   displayed device is the *confirmed* one, per stream, and a fallback must read as a fallback on screen.
3. **Three workers starving the UI thread.** The M1 gate is a floor, not a bound, and it is not comparable
   across machines. At `cpu_count() - 2` it measured p95 12–15 ms; at 8/8 it measured max 1105 ms. Three
   concurrent streams make this worse, and the gate must catch it rather than absorb it.
4. **A toggle mid-stream dropping frames silently.** Drops are counted per stream, refusals are counted with
   reasons, and the three must still reconcile.
5. **A route transition claimed complete before the old worker actually stopped.** The acknowledgement must
   follow the confirmed placement of the new route, not the request for it.

---

### Task 1: Multi-stream supervisor

**Files:**
- Modify: `app/supervisor/runner.py`, `app/contracts/route_plan.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: M1 `Supervisor`, `RoutePlan`, `BoundedFrameQueue`, `InferenceResult`
- Produces: `Supervisor(device_ids: Sequence[str] = ("CPU", "GPU", "NPU"))` with `start()`, `stop()`, `submit(frame)`, `results() -> Mapping[str, InferenceResult]`, `route_plan() -> RoutePlan`, `select_devices(mapping: Mapping[str, str]) -> RoutePlan`, `per_stream() -> tuple[StreamStats, ...]`, `restart_count`; `StreamStats` — frozen dataclass `stream_id`, `requested_device`, `confirmed_devices`, `frames_accepted`, `frames_refused`, `frames_refused_reasons: Mapping[str, int]`, `drops`, `state`, `generations_seen`

- [ ] **Step 1: Write the failing tests**

Cover: three streams each place on their requested device; `submit` to an unknown stream is refused **and
counted with a reason**; per-stream drops do not leak across streams; `select_devices` bumps every affected
stream's generation and clears only those streams' proof; `accepted + refused + drops == attempts` holds
**per stream**; a stream whose worker fails does not take down the other two.

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv\Scripts\python.exe -m unittest tests.test_runner -v`
Expected: failures — `Supervisor` takes one device today.

- [ ] **Step 3: Implement the multi-stream supervisor**

One worker process and one `BoundedFrameQueue` per stream. `RoutePlan.generation` becomes the **highest**
generation across streams, and `confirmed` maps `stream_id -> tuple[str, ...]` so a per-stream fallback is
visible rather than collapsed into one verdict. **Mutation-check every new test** — construct the fault and
confirm the test fails.

- [ ] **Step 4: Run them and watch them pass**

Expected: all green, with the 1034 existing tests still passing.

- [ ] **Step 5: Commit**

`git commit -m "M2: multi-stream supervisor with per-stream accounting"`

---

### Task 2: The route transition protocol

**Files:**
- Modify: `app/supervisor/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Produces: `TransitionOutcome` — frozen dataclass `generation`, `requested`, `confirmed`, `elapsed_s`,
  `bounded_by: str` (`"timeout"` | `"confirmed"`), `discarded_results: int`; and
  `Supervisor.begin_transition(mapping) -> int` (returns the new generation),
  `Supervisor.await_transition(generation, timeout_s: float = 2.0) -> TransitionOutcome`

- [ ] **Step 1: Write the failing tests**

Cover: the four spec §5 toggle combinations produce the correct per-stream mapping; a result from a
superseded generation is **discarded and counted**, never delivered; `await_transition` returns within
timeout with `bounded_by="timeout"` and states which stream did not confirm; the transition is acknowledged
only after the **new** route confirms, never on request; a failed target falls back and says so.

- [ ] **Step 2: Run, watch fail, implement, watch pass**

Expected: fail, then pass.

- [ ] **Step 3: Commit**

`git commit -m "M2: generation-scoped route transitions with bounded acknowledgement"`

---

### Task 3: 50 alternating warm route changes

**Files:**
- Create: `tools/route_stress.py`
- Test: `tests/test_route_stress.py`

**Interfaces:**
- Consumes: `Supervisor`, `select_devices`, `await_transition`
- Produces: `StressResult` — frozen dataclass `transitions`, `warm_elapsed_p50`, `warm_elapsed_p95`,
  `warm_elapsed_max`, `over_two_second`, `discarded_results`, `rss_start`, `rss_end`, `leak_bytes`,
  `stale_overlays`, `mislabelled_devices`, `crashes`; CLI writing `logs/qualification/m2-route-stress.json`

- [ ] **Step 1: Write the failing test**

A short 6-transition version in the suite; the full 50 in the tool. Assert no crash, leak trend bounded, zero
stale overlays, zero mislabelled devices, and **record the warm-transition distribution rather than asserting a
threshold** — the 2 s target is a spec gate to be *measured*, and the report states whether it was met.

- [ ] **Step 2: Run, watch fail, implement, watch pass**

- [ ] **Step 3: Run the real 50-transition stress and record it**

Paste the real output. **If any transition exceeds 2 s, report it as an unmet gate — do not adjust the
timeout to make it pass.**

- [ ] **Step 4: Commit**

`git commit -m "M2: 50 alternating warm route changes with leak and placement evidence"`

---

### Task 4: Close M1's three open criteria

**Files:**
- Modify: `app/main.py`, `app/ui/bridge.py`, `app/ui/Main.qml`, `tests/test_main.py`, `tests/test_ui_contract.py`
- Test: `tests/test_main.py`, `tests/test_ui_contract.py`

**Interfaces:**
- Produces: presenter-driven device selection through the live scene graph; an **observed** fallback;
  and a non-vacuous card-text assertion

- [ ] **Step 1: Fix the vacuous test first (Ruling 62)**

`test_the_rendered_cards_show_the_numbers_the_bridge_holds` iterates three names but `Main.qml` creates cards
only for `system_cpu_percent`, `gpu_utilisation_percent`, `npu_utilisation_percent`, so `find_text` returns
`""` for two of them and `"" != "unavailable"` passes. **Two of three keys assert nothing.** Fix it to read
the values that actually reach the screen — `bridge.metric_text` inside the CPU card's `activityText` — and
**mutation-check it**: swap the divisor or blank a value and confirm it fails.

- [ ] **Step 2: Presenter-driven selection (closes criterion 2)**

Add device selection through `run_windowed` and assert the badge changes **after** confirmation, per stream.
M1 proved placement; this proves a presenter can cause it.

- [ ] **Step 3: An observed fallback (closes criterion 6)**

Make a worker fail for real — not a stub — and assert on screen that the stream shows its **confirmed** device,
`fell_back: true`, and a stated reason. **This is M0's unmet gate 9 and it must be observed, not simulated.**
Record it as measured.

- [ ] **Step 4: Commit**

`git commit -m "M2: presenter-driven selection, an observed fallback, and a non-vacuous card assertion"`

---

### Task 5: Three-stream UI

**Files:**
- Modify: `app/ui/Main.qml`, `app/ui/Bridge.qml`, `app/ui/VideoTile.qml`, `app/ui/ComputeCard.qml`, `app/ui/bridge.py`
- Test: `tests/test_ui_contract.py`

- [ ] **Step 1: Write the failing tests**

Hero plus two thumbnails; per-stream device badges; per-stream counters; a compute card **per stream per
device** showing confirmed placement; the pending-versus-actual transition state; and **"Recorded footage •
live inference" still continuously visible**. Every optional number still renders as `unavailable`, never `0`.
GPU/NPU cards still carry the accelerator wording; the CPU card still carries the collector's reason.

- [ ] **Step 2: Run, watch fail, implement, watch pass**

Keep the Slot-binding guard green — **no QML binding to a Slot that reads mutable state.**

- [ ] **Step 3: Commit**

`git commit -m "M2: three-stream booth layout with per-stream placement"`

---

### Task 6: M2 qualification report

**Files:**
- Create: `docs/qualification/M2-qualification.md`, update `docs/qualification/qualification-matrix.csv`, append `bench_report.md`

- [ ] **Step 1: Run every gate and paste real output**

`verify_sources.py`, `preflight.py`, `--selftest`, `unittest discover`, the three per-device runs, and the
50-transition stress. Paste all of it.

- [ ] **Step 2: Write the report**

The routing table with measured transitions; the warm-transition distribution **and whether the 2 s gate was
met**; the observed fallback; the leak measurement; what changed in input acknowledgement with three streams
(and whether the gate is still a floor); and every unmet gate carried from M0 and M1.

- [ ] **Step 3: Commit**

`git commit -m "M2: qualification report, transition evidence and observed fallback"`

---

## M2 exit criteria

1. All four GPU/NPU toggle combinations execute correctly.
2. Completed jobs are attributed to their **actual** devices, per stream.
3. Pending results cannot paint after a route generation changes.
4. 50 alternating warm route changes with no crash, leak trend, stale overlay, or mislabelled device.
5. Warm transitions meet the 2 s target **or the miss is documented with its measured distribution**.
6. M1 criterion 2 (presenter-driven selection) and criterion 6 (observed fallback) closed with evidence.

## Next milestone

M3 — the glanceable booth interface at the full 1920×1080 canvas with the offscreen DPI harness at
100/125/150/200%, attract mode, presenter drawer, and the 2–3 m readability review.