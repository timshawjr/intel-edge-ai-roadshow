# M5 — Benchmark Records, Export and the Recorded-Demonstration Mode

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A booth run can be **recorded, fingerprinted, exported and compared** — and a recording played back can never be mistaken for a live run. Plus the **M5 content gate**, reported honestly: the provisioned clip count and durations fall short of the specification.

**Architecture:** A `BenchmarkRecord` is the unit of evidence. It carries a **workload fingerprint** (model, thresholds, stream count, analysis rate, device layout, decode path, runtime versions) and an **environment manifest** (machine, drivers, Windows build, power state, monitor). Two records may be compared **only** when their fingerprints match; everything else is marked `previous run` with its machine and timestamp, and a changed setting **invalidates the baseline** rather than silently rebasing it. Export is a ZIP of manifest, routing transitions, metrics and logs — **camera imagery is not exported by default**.

**Tech Stack:** Python 3.12.10, `openvino==2026.4.0`, `PySide6==6.11.2`, `numpy==2.5.3`, `opencv-python==5.0.0.93`, `psutil==7.2.2`. Tests use stdlib `unittest`; `pytest` is deliberately not added. `zipfile`, `json`, `hashlib` are stdlib.

**Spec:** `docs/superpowers/specs/2026-10-01-intel-edge-ai-roadshow-design.md` and `Intel-Edge-AI-Roadshow-Specification.md` §138 (recorded demonstration), §207/§229 (metrics, comparisons), §279/§292 (contracts and export), §330 (M4 deliverable), §338 (content gate), §342 (M6).

## Global Constraints

- Pinned exactly; stdlib `unittest`; no `pytest`; **no new dependency** (`zipfile` is stdlib).
- **Unavailable is `null`, never `0`.** A metric that was not measured is `None` with a reason, in the record as on the screen.
- **Frame age and inference cost are separate claims.**
- **A record without a fingerprint cannot be compared.** A mismatched fingerprint must be refused with the differing fields named, not silently compared.
- **A recording is never presented as live.** `Recorded demonstration - metrics are not live`, and **live device controls are disabled** while it plays. This is the mirror of the mandatory live label and is equally non-negotiable.
- **Camera imagery is not exported by default.** The ZIP contains manifest, transitions, metrics and logs; an explicit opt-in is required for anything else, and it must say so.
- **Every number in a record traces to a sample.** No derived or estimated figures.
- **The machine must be clean before any recorded measurement** — no orphaned workers, no competing suite.
- Nothing in `app/` imports from `tools/`. Layout tokens in `app/theme.py`.
- **Do not use `git add -A`** — it has committed four copies of every QML file in this repository.
- `time.monotonic()` has ~16 ms resolution here; use `perf_counter`.

**M4 facts this milestone builds on:** **1459 tests green, exit 0**, suite 600 s; `verify_sources` 6 PASS/0 FAIL; `preflight` 17 PASS/1 WARN/0 FAIL (Limited); `--selftest` PASS. GPU/NPU utilisation **attributed per process** via `\GPU Engine(*)\Utilization Percentage` filtered to PID (GPU idle 0.0% → busy 97.19%; NPU 0.0% → 64.86%; probe's own pid 0.0% while the worker read 97–100%). The 2 s-timeout leak is fixed (+153.7 → −10.4 MiB). The input-ack gate is demonstrated to pass while the picture is stale (8/8 pass, 6/8 stale). Sustained trials: at the booth's 15 fps **throughput ranks nothing**; isolated inference GPU 307–310 fps vs NPU 246–249 (1.24–1.25×); the NPU's published 5.2–5.9 ms is a **cold** figure against 3.96–4.01 ms sustained. Legibility is `passed at 2 m, failed at 3 m` on the measured monitor.

## Review Focus

1. **A record that cannot be reproduced.** The fingerprint must cover everything that changes the number — model artifact hash, thresholds, stream count, analysis rate, device layout, decode path, runtime versions. A missing field is how a comparison becomes a lie.
2. **A comparison across mismatched conditions.** Refuse and name the differing fields.
3. **A recording that looks live.** The label and the disabled controls are the whole point.
4. **An export that quietly includes imagery.** Default must exclude it.
5. **A content gate rounded up.** The clip count and duration shortfall is a real unmet gate and belongs in the report.

---

### Task 1: The BenchmarkRecord contract

**Files:**
- Create: `app/contracts/benchmark_record.py`
- Test: `tests/test_benchmark_record.py`

**Interfaces:**
- Produces: `WorkloadFingerprint` — frozen dataclass with `model_id`, `model_sha256`, `input_shape`, `conf_threshold`, `iou_threshold`, `stream_count`, `target_fps`, `device_layout`, `decode_path`, `openvino_version`, `numpy_version`; `fingerprint_hash() -> str`; `differing_fields(other) -> tuple[str, ...]`
- Produces: `EnvironmentManifest` — `machine_id`, `cpu`, `gpu`, `npu`, `gpu_driver`, `npu_driver`, `windows_build`, `ram_bytes`, `power_plan`, `ac_status`, `monitor`, `python_version`
- Produces: `BenchmarkRecord` — `mode` (`"live"` | `"recorded"`), `started_at`, `duration_s`, `fingerprint`, `environment`, `samples`, `trial_summary`, `exclusions`, `comparison_eligible`, `unavailable_reasons`

- [ ] **Step 1: Write the failing tests**

Cover: a fingerprint with a changed threshold, a changed model hash, a changed stream count or a changed device layout **differs** and names the field; `fingerprint_hash()` is stable across equal fingerprints and changes when any covered field changes; **a record whose mode is `recorded` cannot claim `comparison_eligible` against a live baseline**; an unavailable metric is `None` **with a reason**, never `0`; a record with no fingerprint cannot be constructed.

- [ ] **Step 2: Run them and watch them fail**

- [ ] **Step 3: Implement the contract**

**Mutation-check every test** — construct the fault and confirm it fails. A fingerprint that hashes only the model id would pass a naive test; make sure the test would catch it.

- [ ] **Step 4: Run and watch them pass**

- [ ] **Step 5: Commit**

`git commit -m "M5: benchmark records carry a fingerprint that can refuse a comparison"`

---

### Task 2: The export ZIP

**Files:**
- Create: `app/export/bundle.py`
- Test: `tests/test_export_bundle.py`

**Interfaces:**
- Produces: `export_bundle(record, destination, include_imagery: bool = False) -> Path`; the ZIP contains `manifest.json`, `routing_transitions.json`, `metrics_summary.json`, `metrics_raw.jsonl`, `errors.log`, and a `README.txt` stating what is and is not included

- [ ] **Step 1: Write the failing tests**

Cover: the ZIP contains every required member; **no image bytes are present by default**, asserted by scanning member names **and** contents for image magic numbers (not just names); `include_imagery=True` adds them **and** the README says so; a `recorded` mode record exports with the recorded-demonstration notice; raw metrics are one JSON object per line and re-parse; the export is reproducible in its **contents** even if the ZIP timestamp bytes differ.

- [ ] **Step 2: Run, watch fail, implement, watch pass**

- [ ] **Step 3: Commit**

`git commit -m "M5: export a bundle that cannot leak imagery by default"`

---

### Task 3: The comparison runner

**Files:**
- Create: `tools/compare_records.py`
- Test: `tests/test_compare_records.py`

**Interfaces:**
- Produces: `compare(baseline, candidate) -> Comparison`; `Comparison` — `eligible`, `differing_fields`, `baseline_label` (`"baseline"` | `"previous run"`), `candidate_label`, `per_metric: Mapping[str, MetricComparison]` with `median`, `range`, `delta`, `within_noise`

- [ ] **Step 1: Write the failing tests**

Cover: **mismatched fingerprints are refused** with the differing fields named; a matched pair compares ranges and medians, **not just medians**; an older record is labelled `previous run` with its machine and timestamp; a changed setting **invalidates the baseline** rather than rebasing; and a metric missing from one side is `None`, never `0`.

- [ ] **Step 2: Run, watch fail, implement, watch pass**

- [ ] **Step 3: Commit**

`git commit -m "M5: comparisons refuse a mismatched fingerprint and name the fields"`

---

### Task 4: Recorded-demonstration mode

**Files:**
- Modify: `app/main.py`, `app/ui/bridge.py`, `app/ui/Main.qml`, `app/ui/PresenterDrawer.qml`, `app/theme.py`
- Test: `tests/test_main.py`, `tests/test_ui_contract.py`

- [ ] **Step 1: Write the failing tests**

Cover: playing a record shows **`Recorded demonstration - metrics are not live` continuously**; **live device controls are disabled** (asserted on the rendered state, not on a flag); the live label is **absent** while it plays and returns when live resumes; every metric shown carries the **timestamp of the recording** it came from; and the screen **cannot** be placed in recorded mode while a live stream is running. Also assert the mirror case: the live label is present and the recorded notice absent.

- [ ] **Step 2: Run, watch fail, implement, watch pass**

Keep the Slot-binding guard green. **Do not erode the legibility sizes** — the recorded notice is as important as the live label and must meet the same angular target.

- [ ] **Step 3: Commit**

`git commit -m "M5: a played-back recording says so and cannot use live controls"`

---

### Task 5: The M5 content gate

**Files:**
- Create: `docs/qualification/M5-content-gate.md`
- Test: `tests/test_content_gate.py`

**Interfaces:**
- Produces: `content_inventory(scenario) -> ContentInventory` — clips, durations, resolutions, licences, rights status; and a gate verdict per specification requirement

- [ ] **Step 1: Inventory what exists, from the files**

Every clip in `assets/media/` and `assets/scenarios/`: duration, resolution, fps, licence, rights status, and whether it is used. **Measured from the files, not read from a manifest.**

- [ ] **Step 2: Write the failing test**

The test asserts the inventory **matches the files on disk** — a clip added or removed without updating the inventory fails. Assert the shortfall is **recorded, not rounded up**: if the spec requires more clips or longer durations than are provisioned, the gate says so.

- [ ] **Step 3: Report the shortfall**

The design document already records it: **clip count and duration per pack fall short of the spec.** State the numbers, the requirement, and the gap.

- [ ] **Step 4: Commit**

`git commit -m "M5: the content gate reports the shortfall rather than rounding it up"`

---

### Task 6: M5 qualification report

**Files:**
- Create: `docs/qualification/M5-qualification.md`, update `qualification-matrix.csv`, append `bench_report.md`

- [ ] **Step 1: Run every gate and paste real output**

- [ ] **Step 2: Report by outcome**

Records: what a fingerprint covers and what it refuses. Export: what the bundle contains and that imagery is excluded by default. Comparison: a real comparison of two records, or a documented refusal. Recorded mode: the label and the disabled controls, with the live mirror case. Content gate: the shortfall, with numbers.

- [ ] **Step 3: Commit**

`git commit -m "M5: qualification report, export evidence and the content shortfall"`

## M5 exit criteria

1. A `BenchmarkRecord` carries a fingerprint covering everything that changes a number, and a mismatched pair is **refused with the differing fields named**.
2. The export ZIP contains manifest, transitions, metrics and logs, and **excludes camera imagery by default**, proven by content not just names.
3. Comparisons report **ranges and medians**, label older records `previous run`, and invalidate a baseline when a setting changes.
4. A played-back recording shows **`Recorded demonstration - metrics are not live`**, disables live device controls, and cannot be mistaken for live.
5. The **content shortfall is measured from the files and reported**, not rounded up.

## Next milestone

M6 — installer or self-contained signed-package candidate, bundled assets, setup guide, presenter guide, diagnostic export, and a known-good environment manifest. Signing is expected to remain an **unmet gate**, recorded with the exact step documented.