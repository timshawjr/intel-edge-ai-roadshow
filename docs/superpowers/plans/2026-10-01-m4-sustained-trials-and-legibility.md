# M4 — Sustained Trials, Legibility and the Real-Timeout Leak — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Four measured obligations from earlier milestones — **make the screen legible at 3 m**, **explain or eliminate the 182 MiB leak at the product's real 2 s timeout**, **run sustained paired trials so a device comparison means something**, and **attribute GPU/NPU utilisation to this process or say plainly that it cannot be done** — plus the four-condition input-ack matrix M3 never obtained.

**Architecture:** No new machinery. M4 is measurement and, where the measurement demands it, a **legibility redesign** of the booth screen: fewer, larger elements rather than more small ones. Every number M4 publishes comes from a committed artefact with its conditions attached.

**Tech Stack:** Python 3.12.10, `openvino==2026.4.0`, `PySide6==6.11.2`, `numpy==2.5.3`, `opencv-python==5.0.0.93`, `psutil==7.2.2`. Tests use stdlib `unittest`; `pytest` is deliberately not added.

**Spec:** `docs/superpowers/specs/2026-10-01-intel-edge-ai-roadshow-design.md` and `Intel-Edge-AI-Roadshow-Specification.md` §7 (staleness), §10 (M4).

## Global Constraints

- Pinned exactly; stdlib `unittest`; no `pytest`; no new dependency.
- Explicit `compile_model(device)` targets only. **No `AUTO`, `HETERO`, `MULTI`.**
- `EXECUTION_DEVICES`, re-read **after** the first inference from the request's own compiled model, is the only proof of placement. It is inconsistent: NPU bare string `'NPU'`, CPU `['CPU']`, GPU `['GPU.0']`. Always `is_device` / `normalise_execution_devices`.
- **Unavailable is `null`, never `0`.** A number that cannot be measured is `None` with a stated reason. A machine-wide peak is **not attributable to this process**.
- **Frame age and inference cost are separate claims and are never merged.** M3 measured live frame age p50 **40–48 s** against an inference cost of **~22 ms** on the CPU. That gap is queue wait and presentation, not inference.
- **`time.monotonic()` has ~16 ms resolution here**; use `perf_counter`. Windows `WaitForSingleObject` has a ~15.6 ms tick floor.
- **A published timing figure needs repeats, stated as a range.** One run is an anecdote.
- **The machine must be asserted clean before any timing, leak or latency measurement.** Sixty-six orphaned workers once poisoned every figure taken for two milestones. The census in `tests/test_input_ack_under_load.py` exists for this; reuse it.
- Nothing in `app/` imports from `tools/`. Layout tokens live in `app/theme.py`.
- **No QML binding to a Slot that reads mutable state.** The guard is line-oriented by design with a two-entry allow-list. A `Repeater` delegate is invisible to `findChild`/`children()`; `Math.max` passes `NaN` through; **a 1×1 unsized window makes every click land on nothing**. All three have produced passing tests over empty results.

**M3 facts this milestone builds on:** **1376 tests green, exit 0**; `verify_sources` 6 PASS/0 FAIL; `preflight` 16 PASS/2 WARN/0 FAIL (Limited); `--selftest` PASS. DPI gate `passed with documented failures`. Attract mode shows three dated claims. The drawer moves nothing. The display copy is 960 px wide; the detector still sees full resolution. Performance fix `6e4253a`: UI-thread CPU per result 19.7 → 7.0 ms, apply throughput 8.6 → 32.3 results/s.

## Review Focus

1. **A legibility fix that shrinks what is on screen to pass a measurement.** The honest route is fewer, larger elements; the dishonest one is a smaller canvas for the same content. Task 1 must state what it *removed*.
2. **A utilisation number that is not this process's.** If attribution cannot be achieved, the deliverable is a documented refusal, not a plausible percentage.
3. **A throughput comparison between devices measured under different conditions.** Pair them: same clip, same duration, same warm-up, same machine state, and report per-device distributions.
4. **A leak "fixed" by making the run shorter.** The 182 MiB appears at the 2 s timeout and not at 30 s; that difference is the clue, not an inconvenience.
5. **A benchmark taken on a dirty machine.** Assert the census first, every time.

---

### Task 1: Legibility at 3 m

**Files:**
- Modify: `app/theme.py`, `app/ui/Main.qml`, `app/ui/ComputeCard.qml`, `app/ui/VideoTile.qml`, `app/ui/PresenterDrawer.qml`
- Test: `tests/test_ui_contract.py`, `tests/test_capture_ui_canvas.py`
- Create: `docs/qualification/M4-legibility-review.md`

**Interfaces:**
- Produces: a legibility token ladder in `app/theme.py` derived from a **stated viewing distance and panel size**, and a review that computes angular size per element.

- [ ] **Step 1: Fix the measurement inputs first**

The M3 review could not establish panel size and gave its table as a function of width. **Establish a stated target** (panel inches and viewing distance) as a named constant in `theme.py`, with the source of the assumption recorded. **If it cannot be established, pick a defensible booth default and say it is a choice, not a measurement.**

- [ ] **Step 2: Write the failing tests**

For the **stated target only** (not every panel size), assert every element a passer-by must read clears **16′ at 3 m**, computed from the rendered scene graph — never from a token multiplied by a constant. Assert the **staleness line and the per-camera age are among the passing elements**, because they are the reason the screen exists. Assert no `Repeater` delegate is missed and no `NaN` passes.

- [ ] **Step 3: Run and watch them fail at the M3 sizes**

At `TYPE_DETAIL` 18 px nothing clears 16′ at 3 m. Expect broad failure.

- [ ] **Step 4: Redesign by removal, not by compression**

To reach 16′ at 3 m an element needs roughly **9.2 mm cap height**, about **44 px** at the measured 0.310 mm/logical px. That is a ~2.4× increase on the 18 px text. **It will not fit at the current information density.** So decide what leaves the always-visible screen — backlog, rate, window, activity and the event row are candidates — and move them into the drawer, where a presenter can look closely. **State what you removed and why.** Keep frame age, staleness, device badge and the recorded-footage label.

- [ ] **Step 5: Run, watch pass, and re-render the four captures**

Confirm nothing overflows, nothing leaves the canvas, the Slot guard is green, and the drawer still moves nothing.

- [ ] **Step 6: Commit**

`git commit -m "M4: the booth screen reads at three metres, by removing rather than shrinking"`

---

### Task 2: The input-ack matrix, finally

**Files:**
- Modify: `tools/input_ack_under_load.py` (only if the tool needs it)
- Test: `tests/test_input_ack_under_load.py`

**Interfaces:**
- Consumes: the Task 3 tool from M3 (895 lines, tested, with a machine-clean guard)
- Produces: `logs/qualification/m4-input-ack.json` — the four conditions, each with tick p50/p95/max **and** frame age p50/p95, backlog and drops, in one row

- [ ] **Step 1: Assert the machine is clean, then run all four conditions**

`three-cpu-no-saturators`, `three-own-devices-no-saturators`, `one-cpu-saturated`, `three-cpu-saturated`. `--seconds 30 --repeats 2`. **Nothing else may run on the machine.** This failed twice for exactly that reason.

- [ ] **Step 2: Record and state the verdict in one sentence**

If the gate passes while frame age is tens of seconds in the same window, the verdict is that the gate is **not merely insensitive but a green light on a failing system**. If the clean matrix says otherwise, say so.

- [ ] **Step 3: Commit**

`git commit -m "M4: the four-condition input-ack matrix, measured on an idle machine"`

---

### Task 3: The 182 MiB leak at the real 2 s timeout

**Files:**
- Modify: `app/supervisor/runner.py`, `app/engine/queue.py` as the evidence dictates
- Test: `tests/test_runner.py`
- Create: `tools/leak_probe.py`

**Interfaces:**
- Produces: `LeakResult` — frozen dataclass `timeout_s`, `transitions`, `rss_start`, `rss_end`, `leak_bytes`, `unexplained_bytes`, `live_objects`, `samples`

- [ ] **Step 1: Reproduce both timings, and write down why they differ**

The finding: **clean at a 30 s deadline** (15.5 MiB total, 5.76 MiB remainder) and **not clean at the real 2.0 s default** (182.4 MiB, 175.3 MiB unexplained). Reproduce both. **The difference between the two conditions is the clue** — a shorter deadline means more transitions *time out*, so the leak is likely on the **timeout path**: a worker or queue retired without being joined, released or drained.

- [ ] **Step 2: Write the failing test**

Construct a transition that times out, repeat it, and assert RSS growth is bounded and that every retired worker and queue is accounted for. **A test that only runs the 30 s path does not test the leak.**

- [ ] **Step 3: Fix, and prove it with a repeat count that would have caught it**

At least the transition count that produced 182 MiB. Report the distribution, not a single figure.

- [ ] **Step 4: Commit**

`git commit -m "M4: every worker and queue retired on the timeout path is released"`

---

### Task 4: Sustained paired trials

**Files:**
- Create: `tools/sustained_trials.py`
- Test: `tests/test_sustained_trials.py`

**Interfaces:**
- Produces: `TrialResult` — frozen dataclass `device`, `requested`, `confirmed`, `fell_back`, `seconds`, `frames`, `throughput_fps`, `latency_p50_ms`, `latency_p95_ms`, `latency_max_ms`, `frame_age_p50_ms`, `frame_age_p95_ms`, `drops`, `rss_start`, `rss_end`

- [ ] **Step 1: Write the failing test**

Pair the devices: **same clip, same duration, same warm-up, same machine state, interleaved** so a drift affects all three equally. Assert placement is confirmed per run and that a fallback cannot be reported as a target.

- [ ] **Step 2: Run the real trials and record them**

Long enough to be sustained, repeated enough to be a distribution. **This is the only basis on which a booth may quote a device comparison.**

- [ ] **Step 3: State what the comparison does and does not support**

Single-frame p50 put GPU ahead of NPU (3.3–3.5 ms vs 5.2–5.7 ms). **Throughput may tell a different story** — an integrated Arc sharing memory with the NPU is exactly the configuration where the ordering can invert at sustained load. Whatever it shows, the report states the conditions and refuses the generalisation the data does not support.

- [ ] **Step 4: Commit**

`git commit -m "M4: sustained paired trials, so a device comparison means something"`

---

### Task 5: GPU/NPU utilisation attribution

**Files:**
- Modify: `app/telemetry/` (a provider, or a documented refusal)
- Test: `tests/test_collector.py`
- Create: `docs/qualification/M4-utilisation-attribution.md`

- [ ] **Step 1: Try to attribute, and be specific about what was tried**

M0 established: in-process PDH reads fault with `0xC0000005`; CIM/`Get-Counter` works but returns a machine-wide peak. **The question is whether any route gives a per-process accelerator figure that this process can claim.** Name the counter, the API, and the failure mode for each route tried.

- [ ] **Step 2: Ship either the provider or the refusal**

If a figure can be attributed, ship it with `EXECUTION_DEVICES`-level rigour. **If it cannot, the deliverable is a documented refusal** and the cards keep reading `unavailable` with a reason. **Do not ship a machine-wide peak as this process's utilisation.**

- [ ] **Step 3: Commit**

`git commit -m "M4: utilisation attributed per process, or refused with the evidence"`

---

### Task 6: M4 qualification report

**Files:**
- Create: `docs/qualification/M4-qualification.md`, update `qualification-matrix.csv`, append `bench_report.md`

- [ ] **Step 1: Run every gate and paste real output**

- [ ] **Step 2: Report the four obligations by outcome**

Legibility — pass or documented failure, with the angular table. Leak — bounded or not, with both timings. Trials — the paired distribution and what it supports. Utilisation — attributed or refused.

- [ ] **Step 3: Commit**

`git commit -m "M4: qualification report, sustained trials and the timeout-path leak"`

## M4 exit criteria

1. Every element a passer-by must read is measured on **the monitor actually attached to the booth machine** (DELL S3422DWG, 0.310 mm/logical px): it clears **16′ of cap height at 2 m**, and its **failure at 3 m is documented** (11.2′ at 44 px). A 55-inch 1080p panel is the **procurement requirement** that would clear 16′ at 3 m; it is never reported as the delivered result.
2. The four-condition input-ack matrix exists, with tick lateness and frame age in the same rows.
3. The 182 MiB leak at the real 2 s timeout is **bounded and explained**, or its cause is stated.
4. Sustained paired trials exist per device, with conditions stated and no unsupported generalisation.
5. GPU/NPU utilisation is **attributed per process or refused with evidence**.

## Next milestone

M5 — recording and export. M6 — packaging, presenter guide, and the unsigned-ships decision.