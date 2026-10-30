# M6 — Package, Guides and the Signing Verdict

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A booth operator can take a folder to the demo machine, install it **without network**, run it, recover it when something goes wrong, and hand the machine back — with a presenter guide that says what is true and what is not, a known-good environment manifest, a diagnostic export, and a **signing verdict that is honest about being unsigned**.

**Architecture:** No new dependency is permitted, so **there is no frozen executable**. The deliverable is a **reproducible offline bundle**: source tree + pinned wheels + models + assets + a provisioning script + guides, with a manifest of every file's SHA-256. That is a weaker artifact than a signed installer and **must be labelled as such** — the spec asks for an "installer or self-contained signed-package candidate", and this milestone delivers the **candidate**, unsigned, with the exact signing step documented.

**Tech Stack:** Python 3.12.10, `openvino==2026.4.0`, `PySide6==6.11.2`, `numpy==2.5.3`, `opencv-python==5.0.0.93`, `psutil==7.2.2`. Tests use stdlib `unittest`; `pytest` is deliberately not added. `hashlib`, `zipfile`, `subprocess` are stdlib.

**Spec:** `docs/superpowers/specs/2026-10-01-intel-edge-ai-roadshow-design.md` and `Intel-Edge-AI-Roadshow-Specification.md` §270 (docs), §279 (contracts), §292 (export), §296 (no security weakening), §300 (each milestone ends runnable), §342 (M6 deliverable).

## Global Constraints

- Pinned exactly; stdlib `unittest`; no `pytest`; **no new dependency**.
- **The application must run as a standard user.** Never require elevation at runtime, and **never weaken a security setting** — no Defender exclusions, no ASR changes, no execution-policy change, no AppLocker bypass. If a step needs elevation, it is an **install step**, and the guide says so and lets the human run it.
- **No network at runtime.** Models, labels, videos and wheels are provisioned once, up front.
- **Unavailable is `null`, never `0`.** Legibility at 3 m still fails; the content gate is still unmet; M2's 2 s target is still missed. **The guides must not imply otherwise.**
- **Do not fabricate a signature, a hash, or a certification.** An unsigned package says it is unsigned.
- Every file in the bundle is hashed, and the manifest is verified on install.
- Nothing in `app/` imports from `tools/`.
- **Do not use `git add -A`** — it has committed four copies of every QML file in this repository.

**M5 facts this milestone builds on:** **1573 tests green, exit 0**, suite 598 s; `verify_sources` 6 PASS/0 FAIL; `preflight` 17 PASS/1 WARN/0 FAIL (Limited); `--selftest` PASS. Benchmark records carry an 11-field fingerprint; export excludes imagery by default; comparisons refuse a mismatch; recorded mode shows `Recorded demonstration — metrics are not live` at 44 px with controls genuinely disabled. **Content gate NOT MET: 3 of 12 clips, 0 rights-cleared.** Legibility `passed at 2 m, failed at 3 m` on the measured monitor. M2's 2 s transition target missed (p50 2.238 s). M0 gate 9's silent half unclosable here.

## Review Focus

1. **A package that works on this machine only because of something already installed.** The offline bundle must be proven by installing into a **clean** location with the network unavailable — or the guide must state exactly what it assumes.
2. **A guide that overstates what the product does.** The three open gates (content, 3 m legibility, 2 s transitions) belong in the presenter guide, not only in the qualification report. A presenter who says "it reads at three metres" will be contradicted by the room.
3. **A signing claim.** There is no certificate. Say so, in the guide and in the manifest.
4. **A diagnostic export that leaks imagery.** Reuse M5's export contract; imagery stays out by default.
5. **A "known-good environment" that was not recorded from the machine that works.** It must be read from the machine, not written from memory.

---

### Task 1: The reproducible offline bundle

**Files:**
- Create: `tools/build_bundle.py`
- Test: `tests/test_build_bundle.py`
- Create: `docs/install/SETUP.md`

**Interfaces:**
- Produces: `build_bundle(destination, wheelhouse) -> BundleManifest`; `BundleManifest` — `files: Mapping[str, str]` (relative path → SHA-256), `python_version`, `platform`, `signed: bool`, `signature_note`

- [ ] **Step 1: Write the failing tests**

Cover: every bundled file appears in the manifest with a **matching SHA-256 read from the file**; a tampered file fails verification; the manifest records `signed: False` and a `signature_note`; **the bundle contains no wheel that is not pinned in `requirements.txt`**, and no pinned requirement is missing; the bundle contains the models and the one used clip; and **`build_bundle` refuses to omit a file it cannot hash**.

- [ ] **Step 2: Run, watch fail, implement, watch pass**

- [ ] **Step 3: Prove the offline install**

Install into a **fresh directory** with no network, and run `--selftest`. **Paste the real output.** If a wheel cannot be provisioned offline on this machine, **say which and what the operator must supply** — do not claim an install you did not perform.

- [ ] **Step 4: Commit**

`git commit -m "M6: an offline bundle with a manifest that says it is unsigned"`

---

### Task 2: The known-good environment manifest

**Files:**
- Create: `tools/environment_manifest.py`
- Test: `tests/test_environment_manifest.py`
- Create: `docs/install/KNOWN-GOOD-environment.json`

**Interfaces:**
- Produces: `collect_environment() -> EnvironmentManifest` — machine id, CPU, GPU, NPU, drivers, Windows build, RAM, power plan, AC status, monitor, Python, OpenVINO, and the **three disagreeing SKU records recorded as three, not collapsed**

- [ ] **Step 1: Write the failing tests**

Cover: every field is **read from the machine**, not from a constant; a field that cannot be read is `None` **with a reason**, never a guess; the **three conflicting SKU records are preserved as three** (this machine has no single defensible SKU — `Win32_ComputerSystem` = `NUC14LNK LN`, hostname `NUC14ProNPU`… verify, and never invent one); AC status is `None` with a reason because **no `Win32_Battery` exists**.

- [ ] **Step 2: Run, watch fail, implement, watch pass, record it**

- [ ] **Step 3: Commit**

`git commit -m "M6: the known-good environment read from the machine, including what it cannot read"`

---

### Task 3: The presenter guide

**Files:**
- Create: `docs/PRESENTER-GUIDE.md`
- Test: `tests/test_presenter_guide.py`

- [ ] **Step 1: Write the failing tests**

The guide is **checked against the code and the record**, not written freehand. Cover: every command it prints **exists in the CLI** (`--streams`, `--select`, `--fail-stream`, `--windowed`, `--headless`, `--selftest`, `--release`); `--fail-stream` is described as a **qualification aid, not a booth feature**; the **three open gates appear** (content 3/12 and 0 rights-cleared; 3 m legibility fails; 2 s transition target missed); the NPU figure is stated **cold vs settled**; and **the guide does not claim a throughput ranking** the trials do not support.

- [ ] **Step 2: Run, watch fail, write, watch pass**

- [ ] **Step 3: Commit**

`git commit -m "M6: a presenter guide checked against the code and the gates"`

---

### Task 4: The diagnostic export

**Files:**
- Create: `tools/diagnostic_export.py`
- Test: `tests/test_diagnostic_export.py`

**Interfaces:**
- Produces: `diagnostic_bundle(destination) -> Path` — a ZIP with the environment manifest, device discovery, `--selftest` output, the three gate results, the benchmark record, and **no camera imagery by default**

- [ ] **Step 1: Write the failing tests**

Cover: the ZIP contains the environment, discovery, selftest and gate results; **no image content** is present, proven by scanning bytes not names; a failing gate is recorded **as failing**, not omitted; and running the export **does not modify `config/telemetry_map.json`** (a tool in this repo rewrites it as a side effect — do not repeat that).

- [ ] **Step 2: Run, watch fail, implement, watch pass**

- [ ] **Step 3: Commit**

`git commit -m "M6: a diagnostic export that carries the gates and no imagery"`

---

### Task 5: The signing verdict

**Files:**
- Create: `docs/install/SIGNING.md`
- Test: `tests/test_signing_document.py`

- [ ] **Step 1: Write the failing test**

The document must state: the package is **unsigned**; **no certificate is available**; the exact commands a human with a certificate would run (`signtool sign /fd SHA256 /tr <timestamp-url> /td SHA256 /a <file>`, and `signtool verify /pa /all`); **what the absence costs** (SmartScreen warning, enterprise policy may block it); and that **weakening a security control to work around it is forbidden**.

- [ ] **Step 2: Write the document, run the test, watch it pass**

- [ ] **Step 3: Commit**

`git commit -m "M6: unsigned ships, and the signing step is written down exactly"`

---

### Task 6: M6 qualification report

**Files:**
- Create: `docs/qualification/M6-qualification.md`, update `qualification-matrix.csv`, append `bench_report.md`

- [ ] **Step 1: Run every gate and paste real output**

- [ ] **Step 2: Report by outcome**

Bundle: what it contains, that it installs offline, and that it is **unsigned**. Environment: what was read and what could not be. Presenter guide: checked against the code. Diagnostic export: gates and no imagery. Signing: **unmet**, with the step written down.

- [ ] **Step 3: State the final deliverable status across M0-M6**

Every gate, **passed or not**, in one table — including the ones that failed. **This is the document a customer reads.**

- [ ] **Step 4: Commit**

`git commit -m "M6: qualification report and the deliverable status across M0-M6"`

## M6 exit criteria

1. An offline bundle exists with a **per-file SHA-256 manifest**, installs without network, and runs `--selftest`.
2. The known-good environment is **read from the machine**, with unreadable fields `None` and three conflicting SKU records preserved.
3. The presenter guide is **checked against the CLI and the gates**, including the three open ones.
4. The diagnostic export carries the gates and **no imagery**, proven by content.
5. **Signing is reported as unmet**, with the exact step documented and the cost stated.
6. A single table states **every gate across M0-M6**, passed or not.

## Final deliverable

Windows package (unsigned candidate), source, verified assets and licences, qualification evidence, and the presenter guide — with the open gates named rather than closed by assertion.