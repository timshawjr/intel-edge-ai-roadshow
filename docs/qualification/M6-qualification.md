# M6 — package, guides and the signing verdict: qualification report

**What this document is.** The milestone record for M6, and the **closing record of the project**: the one
table in §7 states every gate from M0 to M6 — passed or not — with the milestone that measured it and where
the evidence lives. It reports the M6 obligations **by outcome** and every figure traces to a command run in
this session or to a committed file named at the point of use.

It **extends** `M0-qualification.md`, `M0-content-gate.md`, `M1-qualification.md`, `M2-qualification.md`,
`M3-readability-review.md`, `M4-qualification.md`, `M4-legibility-review.md`,
`M4-utilisation-attribution.md`, `M5-qualification.md` and `M5-content-gate.md` — and does not contradict
any of them. Where a claim could not be re-established, this document says so and names the earlier record
that stands instead. This task is **documentation only**: it changed no code, no `config/*.json`, no
`requirements.txt` and no tool.

Machine: **NUC14PROAI** (hostname, not a SKU — the three identity records still disagree; §3), Windows 11 Pro
`10.0.26200`, branch `main`, HEAD `909c02b` at the start of this task, 2026-10-30. Python 3.12.10, OpenVINO
`2026.4.0-22959-99c81491cc3-releases/2026/4`, GPU driver 32.0.101.6737, NPU driver 32.0.100.5540.

**The tree was clean when measured.** `Get-Process python` and `Get-Process pythonw` returned **0** before
every measurement in §1, and `git status --short` was empty. **One side effect is disclosed rather than
hidden:** running `tools/probe_telemetry.py` (a command AGENTS.md's verification list names) **rewrote
`config/telemetry_map.json`** — 21 lines changed — because that tool probes and saves the map. The file was
restored with `git checkout -- config/telemetry_map.json` immediately, `git status --short` was empty
afterwards, and its SHA-256 is **`235B13AF…`** again, which is the value it carries in §5 before and after the
diagnostic export. That tool's write is the exact side effect M6 Task 4 exists to avoid, and the diagnostic
export does avoid it (§5).

---

## The M6 obligations, and what each one returned

| # | obligation (plan §M6 exit criteria) | outcome |
|---|---|---|
| 1 | An offline bundle exists with a per-file SHA-256 manifest, installs without network, and runs `--selftest` | **MET.** 64 payload files, 459,628,699 bytes; **64/64 SHA-256 verified**; installed into a **fresh directory** with the network made unusable, **all 10 wheels** from the local wheelhouse, `--selftest` **PASS**, exit 0, 32.44 s. **Unsigned** — `signed: false` in the manifest. §2 |
| 2 | The known-good environment is read from the machine, unreadable fields `None`, three conflicting SKU records preserved | **MET.** 69 fields re-read and compared against the committed file: **68 identical, 1 differs and it is the timestamp**. Three SKU records kept as three; `ac_status: null` with the `Win32_Battery` reason; both the `11 Pro` caption and the `Windows 10 Pro` registry value retained. §3 |
| 3 | The presenter guide is checked against the CLI and the gates, including the three open ones | **MET.** **11 of 11** application commands in the guide parse against the real parser; the three open gates are in it with their numbers; 13 committed tests, `OK`. One number in it has no committed source — flagged in §4.2 and §10, not silently repeated. §4 |
| 4 | The diagnostic export carries the gates and no imagery, proven by content | **MET.** **7 members**; **`IMAGE-CONTENT HITS: NONE`** by a whole-content byte scan in this session; all three gates present and **marked failed**; `config/telemetry_map.json` hash **unchanged**. §5 |
| 5 | Signing is reported as unmet, with the exact step documented and the cost stated | **MET AS A REPORT — THE SIGNING GATE IS UNMET.** There is no certificate on this machine. `signtool sign /fd SHA256 /tr <timestamp-url> /td SHA256 /a <file>` and `signtool verify /pa /all <file>` are written down; SmartScreen and enterprise-policy consequences are stated; weakening a security control is forbidden. §6 |
| 6 | A single table states every gate across M0-M6, passed or not | **MET** — §7, eighteen rows, no gate omitted and none softened. |

**The one sentence a reader can carry away.** The operator has a **Windows package (an unsigned candidate)**
that installs and self-tests with no network, a **known-good environment read from the machine it will run
on**, a **presenter guide checked against the code**, a **diagnostic export that carries the failures and no
imagery**, and **one table that names every open gate instead of closing it by assertion**. What the operator
does **not** have is a signed package, rights-cleared footage, a screen that reads at three metres, or a
two-second camera switch.

---

## 1. Method, and the gate set run in this session

### 1.1 Cleanliness

```
python procs before verify_sources : 0
python procs before preflight      : 0
python procs before the suite      : 0
pythonw procs                      : 0
```

`Get-Process python` and `Get-Process pythonw` were both empty at each of those points, and no measurement in
this report ran while another one ran: the four gates of §1.2 were run **sequentially**, before the suite,
and the suite was the last thing started.

### 1.2 The gate set

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
EXIT=0 WALL_S=0.14
```

Verified **by content** — signature and SHA-256 against `config/models.json`, never a status code.

```
$ .venv\Scripts\python.exe tools\preflight.py
Summary: 17 PASS, 1 WARN, 0 FAIL - 18 total rows
Status: Limited
Exit code: 0. Ready and Limited both exit 0, because in both the demo runs; only Needs attention exits 1.
EXIT=0 WALL_S=4.51
```

The row names, verbatim and in this run's order:

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
PASS  Telemetry: per-process GPU/NPU utilisation
PASS  Telemetry: recorded LUID-to-device map
WARN  Telemetry: LUID-to-device attribution and hardware identity
PASS  Startup: this gate's import to the first inference
```

The **single WARN is unchanged from M4 and M5** and is a **recorded-config row, not a device row**: this
boot's GPU LUID is `0x000113B4` while the committed `config/telemetry_map.json` records `0x00010E9E` /
`0x000112A0`, because Windows assigns adapter LUIDs per boot. The runtime resolves the split from DXGI and
the one adapter exposing `engtype_Neural`; a stale file does not make the cards wrong. **The file was not
edited** (AGENTS.md rule 6) — the WARN is the honest state of a verified file that is per-boot.

Figures from this run: disk **189.4 GiB free of 381.5 GiB**; memory **22.3 GiB available of 31.5 GiB** across
8 logical cores; discovery **2.82 s**; import-to-first-inference **4.02 s**; **358** live GPU Engine
instances; the CPU sanity fixture re-run **live** here: `2 of 2 matched`, `EXECUTION_DEVICES=['CPU']
placed_on_requested=True`, `VERDICT PASS`.

```
$ .venv\Scripts\python.exe -m app.main --selftest
  ok   preprocess           letterbox 640x640 pad=114 scaled=640x360
  ok   infer                24.40 ms on confirmed ['CPU']
  ok   decode-track-event   1 zone_occupancy event(s) from a synthetic presence
  ok   render-metadata      unavailable renders as unavailable, and no device is claimed before an inference confirms one
PASS
EXIT=0 WALL_S=1.01
```

No network, exit 0. The `24.40 ms` is **one warm CPU inference in this run**, not a latency claim; M4's
paired trials are the latency record.

```
$ .venv\Scripts\python.exe -m unittest discover -s tests
----------------------------------------------------------------------
Ran 1635 tests in 609.277s

OK
EXIT=0 WALL_S=610.33
```

**1 635 tests, `OK`, exit 0, 609.277 s inside unittest and 610.33 s of wall clock** measured around the
process. No failure, no error and no skip — `OK` rather than `OK (skipped=N)`. The run's stderr carries the
tests' own expected noise (the injected `COMPILE FAILURE` tracebacks from the supervisor seam, argparse's
`--device TPU` refusal, and the temporary verification fixtures being refused and then accepted); the summary
line above is the result, and it is the only line that is a result.

M6 adds **62 tests in five modules** — `test_build_bundle.py` 12, `test_environment_manifest.py` 13,
`test_presenter_guide.py` 13, `test_diagnostic_export.py` 13, `test_signing_document.py` 11 — to M5's
1 573. The count in the block above is the measured total, and the difference from 1 573 is those 62.

### 1.3 Where the evidence lives

| record | committed? | what it backs |
|---|---|---|
| `docs/install/SETUP.md`, `docs/install/SIGNING.md`, `docs/install/KNOWN-GOOD-environment.json` | **yes** | §2, §3, §6 — the operator-facing position |
| `docs/PRESENTER-GUIDE.md`, `tests/test_presenter_guide.py` | **yes** | §4 |
| `tools/build_bundle.py`, `tools/environment_manifest.py`, `tools/diagnostic_export.py` and their five test modules | **yes** | every re-run in §2, §3, §5 |
| `dist/M6-bundle` (and `dist/wheelhouse`) | **no — `dist/` is gitignored** | the bundle itself. It is **not committed**: this report describes it by its manifest, its sizes and the re-run in §2.3, not as a file in git. |
| the fresh-directory install at `%TEMP%\opencode\m6-task6-offline`, the diagnostic ZIP at `%TEMP%\opencode\m6-task6-diagnostic.zip`, the suite log at `%TEMP%\opencode\m6-suite.log` | **no — scratch** | the pasted output in §1.2, §2.3 and §5. Each was produced by committed code and the output is pasted here; the directories are disposable and are not evidence a reader can check out. |
| `docs/qualification/qualification-matrix.csv` | **yes** | the 42-row machine-readable matrix; M6 added five rows (§7 and Appendix A) |

---

## 2. The bundle — 64 files, verified, installed offline, and unsigned

### 2.1 What it is, and what it is not

It is a **reproducible offline source bundle**: the application source, the pinned wheels, the verified model
files, the one clip the application names, the scenario manifest and JSON config, a stdlib-only
`provision_offline.py`, the setup guide, and a manifest recording the SHA-256 of every payload file. It is
**not** a frozen executable and **not** a signed installer — the project permits no new dependency, so there
is no packaging runtime to freeze with.

```
64 payload files, 459,628,699 bytes
65 files on disk including bundle-manifest.json, 459,636,178 bytes
```

| top-level path | files | what it is |
|---|---:|---|
| `app/` | 43 | the application source, `.pyc`/`__pycache__` excluded |
| `wheelhouse/` | 10 | the pinned closure — see §2.2 |
| `assets/` | 5 | `yolo11n.xml` 412,503 B, `yolo11n.bin` 5,283,218 B, `labels.txt` 701 B, `retail-checkout.mp4` 64,996,307 B, `assets/scenarios/retail_sanity/manifest.json` 4,535 B |
| `config/` | 3 | `models.json`, `telemetry_map.json`, `model_io/yolo11n-fp16.json` |
| `docs/` | 1 | `docs/install/SETUP.md` |
| root | 2 | `requirements.txt` and `provision_offline.py`; `bundle-manifest.json` sits beside them and is **not** in its own `files` map, which is why the directory holds 65 files while the manifest lists 64 |

The manifest carries `schema_version`, `bundle_format: reproducible-offline-source-bundle`,
`python_version 3.12.10`, `platform Windows-11-10.0.26200-SP0`, **`signed: false`**, and a `signature_note`
that begins `UNSIGNED. This bundle carries no Authenticode signature and no certificate was available to
create one.` — and that note explicitly refuses to get around it: *"do not weaken Defender, ASR, AppLocker or
the execution policy to get around that."*

**A gap found in this task, reported and not fixed.** The bundle carries `docs/install/SETUP.md` but **not**
`docs/install/SIGNING.md`, **not** `docs/PRESENTER-GUIDE.md` and **not**
`docs/install/KNOWN-GOOD-environment.json`. Its `signature_note` points at `SETUP.md`, and `SETUP.md` points
at *"the M6 signing document"* — which is not inside the bundle. An operator holding only the bundle can
install and run it, but cannot read the exact signing step or the presenter guide without the repository.
That is a packaging omission, not a false claim; it is on the open list in §10 and would be fixed by adding
three paths to `tools/build_bundle.py`'s `source_paths()` and rebuilding.

### 2.2 The wheelhouse is a closure, and the build refuses anything else

`requirements.txt` pins **six** distributions; the installed pinned metadata declares **four** more. Re-running
the committed builder's own resolver in this session:

```
direct pins: 6 ['numpy', 'opencv-python', 'openvino', 'psutil', 'pyside6', 'pyyaml']
closure: 10
declared deps only: ['openvino-telemetry', 'pyside6-addons', 'pyside6-essentials', 'shiboken6']
    openvino-telemetry 2025.2.0 declared by ['openvino']
    pyside6-addons 6.11.2 declared by ['pyside6']
    pyside6-essentials 6.11.2 declared by ['pyside6']
    shiboken6 6.11.2 declared by ['pyside6']
wheelhouse validated: 10 [...]
exact match: True
```

**Zero unrelated wheels.** And the refusal is real, not asserted — four wheelhouses were constructed in this
session and fed to the committed `validate_wheelhouse`:

```
baseline 10 -> ACCEPTED
extra wheel requests -> REFUSED: wheelhouse contains wheels that are not pinned in requirements.txt or declared by a pinned distribution: requests
missing direct pin pyyaml -> REFUSED: wheelhouse is missing a pinned requirement or a dependency it declares: pyyaml
missing declared dep shiboken6 -> REFUSED: wheelhouse is missing a pinned requirement or a dependency it declares: shiboken6
wrong numpy version -> REFUSED: numpy: the pinned environment has 2.5.3 but the wheelhouse has 2.4.0
```

The wheels actually bundled: `numpy-2.5.3-cp312-cp312`, `opencv_python-5.0.0.93-cp37-abi3`,
`openvino-2026.4.0-22959-cp312-cp312`, `openvino_telemetry-2025.2.0-py3-none-any`,
`psutil-7.2.2-cp37-abi3`, `pyside6-6.11.2-cp310-abi3`, `pyside6_addons-6.11.2-cp310-abi3`,
`pyside6_essentials-6.11.2-cp310-abi3`, `pyyaml-6.0.3-cp312-cp312`, `shiboken6-6.11.2-cp310-abi3`
(nine are `win_amd64`; `openvino_telemetry` is the pure-Python `py3-none-any` wheel).

### 2.3 The offline install, re-run in this session

Two independent verifications of the digest set, then the install:

```
$ cd dist\M6-bundle; python provision_offline.py --verify-only
MANIFEST PASS: 64 files match their SHA-256
SIGNED: False - UNSIGNED. ...
EXIT=0 WALL_S=0.38

$ python tools\build_bundle.py --destination dist\M6-bundle --wheelhouse dist\wheelhouse --verify dist\M6-bundle
checked 64 files; ok=True
EXIT=0
```

The second command also refuses an **unlisted** file, so the 64/64 is a match of the manifest *and* the
directory, not a manifest checked against itself.

The install was performed into a **fresh directory outside the repository**, on a copy of the bundle, with
the network made unusable — a dead proxy on every spelling, a dead `PIP_INDEX_URL`, and `PIP_CONFIG_FILE=NUL`:

```
HTTP_PROXY / HTTPS_PROXY / http_proxy / https_proxy = http://127.0.0.1:9
PIP_INDEX_URL = http://127.0.0.1:9/simple
NO_PROXY = (empty)          PIP_CONFIG_FILE = NUL

$ cd <fresh dir>; python provision_offline.py --selftest
MANIFEST PASS: 64 files match their SHA-256
SIGNED: False - UNSIGNED. ...
creating venv at <fresh dir>\.venv from ...\Python312\python.exe
installing offline: <fresh dir>\.venv\Scripts\python.exe -m pip install --no-index --only-binary=:all: --find-links <fresh dir>\wheelhouse -r <fresh dir>\requirements.txt
Looking in links: <fresh dir>\wheelhouse
Processing ... openvino-2026.4.0-22959-cp312-cp312-win_amd64.whl (from requirements.txt (line 1))
Processing ... pyside6-6.11.2-cp310-abi3-win_amd64.whl (from requirements.txt (line 2))
Processing ... numpy-2.5.3-cp312-cp312-win_amd64.whl (from requirements.txt (line 3))
Processing ... opencv_python-5.0.0.93-cp37-abi3-win_amd64.whl (from requirements.txt (line 4))
Processing ... psutil-7.2.2-cp37-abi3-win_amd64.whl (from requirements.txt (line 5))
Processing ... pyyaml-6.0.3-cp312-cp312-win_amd64.whl (from requirements.txt (line 6))
Processing ... openvino_telemetry-2025.2.0-py3-none-any.whl (from openvino==2026.4.0->requirements.txt (line 1))
Processing ... shiboken6-6.11.2-cp310-abi3-win_amd64.whl (from PySide6==6.11.2->requirements.txt (line 2))
Processing ... pyside6_essentials-6.11.2-cp310-abi3-win_amd64.whl (from PySide6==6.11.2->requirements.txt (line 2))
Processing ... pyside6_addons-6.11.2-cp310-abi3-win_amd64.whl (from PySide6==6.11.2->requirements.txt (line 2))
Installing collected packages: openvino-telemetry, shiboken6, PyYAML, psutil, numpy, PySide6_Essentials, openvino, opencv-python, PySide6_Addons, PySide6
Successfully installed PySide6-6.11.2 PySide6_Addons-6.11.2 PySide6_Essentials-6.11.2 PyYAML-6.0.3 numpy-2.5.3 opencv-python-5.0.0.93 openvino-2026.4.0 openvino-telemetry-2025.2.0 psutil-7.2.2 shiboken6-6.11.2
--- pip freeze ---
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
--- end pip freeze ---
--- python -m app.main --selftest ---
  ok   preprocess           letterbox 640x640 pad=114 scaled=640x360
  ok   infer                25.21 ms on confirmed ['CPU']
  ok   decode-track-event   1 zone_occupancy event(s) from a synthetic presence
  ok   render-metadata      unavailable renders as unavailable, and no device is claimed before an inference confirms one
PASS
EXIT=0 WALL_S=32.44
```

- **`Looking in links:` names only the bundle's `wheelhouse`.** There is no "Looking in indexes" line, which
  is what `--no-index` does: an index is never consulted even if one is configured.
- **All 10 wheels installed**, and `pip freeze` lists exactly those 10 — no extra package arrived from
  somewhere else. The venv was created by the bundle's own script from the machine's CPython 3.12.10, which
  is an **install step a human performs**, not a runtime requirement.
- **`--selftest` PASS**, exit 0, in **32.44 s** including the install. The test is the same one §1.2 runs;
  its `25.21 ms` is one CPU inference, not a latency claim.

**The install proves the bundle is self-sufficient for running the application on this interpreter.** It
does **not** prove it on a machine with a different CPU, a different driver, or no prior CPython — those are
stated assumptions in `SETUP.md`, not measurements. And it says nothing about whether an organisation's
policy will *allow* an unsigned artifact to run: that is §6.

### 2.4 Unsigned

`dist/M6-bundle` carries **no Authenticode signature**. The manifest records `signed: false` and the note
quoted in §2.1; `provision_offline.py` prints that note on every run, including the two above. No certificate,
private key or timestamp account exists on this machine, and nothing in the tree fabricates one. The exact
signing step, its cost, and the prohibition on weakening security controls are in §6.

---

## 3. The known-good environment — read, not remembered

`tools/environment_manifest.py` reads the machine through PowerShell CIM (`Win32_*` + `WmiMonitorID`),
`powercfg`, `sys`/`platform`, `importlib.metadata` and `openvino.Core().available_devices`. The committed
record is `docs/install/KNOWN-GOOD-environment.json`.

**It was re-read in this session and compared field by field** — not trusted because it was committed:

```
fields compared: 69 | differing: 1
  generated_at_utc: committed='2026-10-30T22:36:58+00:00' now='2026-10-30T23:40:35+00:00'
sku_records values: ['NUC14LNK LN', 'NUC14PROAI', '88F21043']
ac_status: None | battery_instances: 0
windows caption: Microsoft Windows 11 Pro | registry: Windows 10 Pro
```

**68 of 69 fields are identical.** The only difference is the timestamp of collection, so the environment
recorded at Task 2 is still the environment this report measured on.

### 3.1 What was read

| field | value |
|---|---|
| CPU | Intel(R) Core(TM) Ultra 9 288V, GenuineIntel, 8 cores / 8 logical processors, 3300 MHz |
| GPU | Intel(R) Arc(TM) 140V GPU (16GB), PCI `8086:64A0`, driver 32.0.101.6737 (2025-04-15), `oem79.inf` |
| NPU | Intel(R) AI Boost, PCI `8086:643E`, status OK, driver 32.0.100.5540 (2026-08-19), `oem5.inf` |
| Windows | caption `Microsoft Windows 11 Pro`, 10.0.26200, build 26200, UBR 9711, display version 25H2, 64-bit |
| RAM | 33,805,881,344 bytes |
| power plan | Balanced `381b4222-f694-41f0-9685-ff5bb260df2e` |
| monitor | Dell S3422DWG(HDMI2 2.0), DEL/D129, serial 809587030, 2023 week 41, 3440×1440 at 59 Hz |
| runtime | Python 3.12.10; OpenVINO 2026.4.0; devices `['CPU', 'GPU', 'NPU']` |
| machine id | `1DBB66E9-47DF-0DEA-BEF3-BCFCE7DD5FE0` |

### 3.2 What could not be read — `null` with a reason, never a guess

- **AC status is `null`.** `Win32_Battery` returns **0 instances**: there is no battery, so AC/battery state
  cannot be read. `battery_instances: 0` and `unavailable.ac_status` carry the reason. Thermal and energy
  readings are not inferred from it (AGENTS.md rule 2).
- **No single SKU is defensible.** Three SKU-bearing records disagree and are preserved **as three**:
  `Win32_ComputerSystem.Model + SystemFamily` → `NUC14LNK LN`; the hostname → `NUC14PROAI`; the PCI
  `SUBSYS` in the GPU/NPU PNP id → `88F21043` (subsystem vendor 1043 = ASUSTeK). `sku_note` says the hostname
  is not a model and the PCI subsystem ID is not a marketing name. Nothing is promoted to "the" SKU.
- **The Windows name disagrees with itself and neither value is corrected.** `Win32_OperatingSystem.Caption`
  says `Microsoft Windows 11 Pro`; the registry `ProductName` says `Windows 10 Pro`. Both are recorded, with
  `product_name_note`; the build number 26200 and display version 25H2 are recorded beside them. A reader
  who needs the marketing name gets the disagreement, not a choice made for them.
- **`AdapterRAM` is not VRAM.** It reports 2,147,479,552 bytes — a 32-bit property that can be capped — and
  the record says so rather than presenting it as a measured memory size.

This is the machine the qualification record was measured on, and the only one. Nothing in this document
generalises from it to another box.

---

## 4. The presenter guide — checked against the code

### 4.1 Every command parses

The guide's application commands were extracted with the same regular expression the committed test uses and
parsed by `app.main.build_parser()` — the parser `--help` prints:

```
ACCEPTED --selftest
ACCEPTED --windowed --release
ACCEPTED --windowed --streams 3 --device NPU
ACCEPTED --windowed --select A=GPU
ACCEPTED --headless --device GPU --seconds 20 --json logs\run.json
ACCEPTED --headless --scenario retail --source once --device NPU --seconds 30 --json logs\once.json
ACCEPTED --headless --streams 3
ACCEPTED --windowed --streams 3 --fail-stream C
ACCEPTED --no-release --windowed
ACCEPTED --play-record docs\qualification\m5-records\candidate.json
ACCEPTED --selftest
commands=11 accepted=11 refused=0
```

**One nuance, stated rather than smoothed:** `--headless --streams 3` is accepted by argparse and then
**refused by `app.main` before any run**, exit 2 — *"not supported by --headless: the headless run builds one
supervisor for one stream and its summary is single-stream, so accepting this flag would run one camera while
claiming 3."* That refusal is exactly what the guide prints, and the committed test invokes the CLI and
asserts the guide repeats it rather than paraphrasing it away.

The committed suite module re-checks all of this and the content:

```
$ python -m unittest tests.test_presenter_guide -v
Ran 13 tests in 0.021s
OK
```

It asserts that every command parses, that the guide mentions `--streams`, `--select`, `--fail-stream`,
`--windowed`, `--headless`, `--selftest`, `--release`, `--no-release`, `--device`, `--seconds`, `--json`,
`--source`, `--scenario`; that `--fail-stream` is described as a **QUALIFICATION AID, not a booth feature** —
the same words `--help` uses; that recorded mode quotes `Recorded demonstration — metrics are not live` and
says the controls are disabled; and that the guide says the package is unsigned and points at
`docs/install/SIGNING.md`.

### 4.2 The three open gates are in the guide, with their numbers

| gate | what the guide says |
|---|---|
| content | **failed** — 3 of 12 clips, 0 rights-cleared, one inside the 45–90 s band, shelf-to-basket **not provisioned**; "do not describe the demo as a shelf-to-basket demonstration" |
| legibility | **fails at 3 m** — 16.8′ cap at 2 m against 11.2′ at 3 m, x-height 12.2′/8.1′, on the measured S3422DWG at 0.310 mm per logical pixel; the 55-inch panel is labelled a **procurement requirement** |
| transition | **missed** — p50 2.261 s, the committed 2 s-gate file's 2.628 s, **37 of 50** over the 2.0 s default, 25 of 50 over at the widened 30 s deadline |

It also states the NPU figure as **5.50–5.57 ms cold against 3.96–4.01 ms settled** and names the earlier
5.2–5.9 ms figure as the cold one; says the booth's 15 fps **"ranks nothing"**, with isolated sustained
inference (GPU 307–310 fps, NPU 246–249 fps) explicitly not a claim about the booth stream; and says the
retail event is **people-in-zone presence, 2 per clip**, not a count and not a dwell. Its "may say / must not
say" list forbids the four sentences that would be contradicted by the room.

**A number in the guide that has no committed source — flagged, not repeated as evidence.** The guide says
*"M2 measured p50 2.238 s"*. **No committed M2 record contains 2.238.** The committed values are:

| committed file | p50 | deadline | over 2 s |
|---|---:|---:|---:|
| `route-stress-50-after-teardown-fix.json` | **2.261 s** | 30 s (widened so the tail is measured) | 25/50 |
| `route-stress-50-gate-2s-after-teardown-fix.json` | **2.628 s** | the real **2.0 s** default | **37/50** |

The same sentence prints 2.261 and 2.628, so a reader of the guide sees the committed numbers; the 2.238 half
is uncommitted. `M3-readability-review.md`, `M4-qualification.md`, `M5-qualification.md` and this milestone's
brief all carry "2.238–2.261 s" as a range. This report quotes **2.261 s and 2.628 s** and treats the 2.238 as
an uncommitted working-log figure that should not have propagated. It is on the open list in §10; the guide was
**not edited** by this documentation task.

---

## 5. The diagnostic export — the failures travel, the imagery does not

Run against the committed record `docs/qualification/m5-records/candidate.json`:

```
$ python -m tools.diagnostic_export --destination <temp>\m6-task6-diagnostic.zip
wrote <temp>\m6-task6-diagnostic.zip
members (7): manifest.json, environment.json, device_discovery.json, selftest.txt, gates.json, benchmark_record.json, README.txt
  gate content      failed       3 of 12 clips provisioned, 0 rights-cleared; 1 inside the duration band; 1 pack(s)
  gate legibility   failed       fails: cap height at 3.0 m (11.2' < 16.0'); x-height at 2.0 m (12.2' < 16.0'); x-height at 3.0 m (8.1' < 16.0')
  gate transition   failed       37 of 50 warm transitions over 2.0 s at p50 2.628 s
all gates passed: False
signed: False  imagery_included: False
EXIT=0 WALL_S=6.42
```

**Seven members, and the three gates that are failing are the reason to open it.** A diagnostic bundle that
omitted a failing gate would make a booth failure look like an environment problem; `all_passed` is computed
from the statuses, and `GateResult` refuses any status outside `passed`/`failed`/`unavailable` — "a gate that
was not measured is `unavailable`, not passed".

### 5.1 No imagery, proven by content

Every member's **whole bytes** were scanned in this session — not a prefix, and not the member name — for PNG
`89504E47`, JPEG `FFD8FF`, GIF87a/89a, TIFF `II*\x00`/`MM\x00*` and `RIFF….WEBP`:

```
manifest.json                bytes=   1450  image_magic=None
environment.json             bytes=   4165  image_magic=None
device_discovery.json        bytes=    309  image_magic=None
selftest.txt                 bytes=    451  image_magic=None
gates.json                   bytes=   1675  image_magic=None
benchmark_record.json        bytes=   4518  image_magic=None
README.txt                   bytes=    860  image_magic=None
IMAGE-CONTENT HITS: NONE
```

`imagery_included: false` in the manifest, `signed: false`, and `README.txt` says so in words. The export
**refuses to write** a member whose bytes contain a magic number, so the zero is enforced by the writer and
then re-measured here; a scan that finds nothing because it cannot see anything would prove nothing, and the
committed tests show it finding constructed PNG/JPEG/WebP bytes.

### 5.2 The export does not rewrite the machine's verified data

```
telemetry_map BEFORE: 235B13AF0DDEEB96CDD2C2031CA28C70550045006BC1A9FB1B3180D9423B32BC
telemetry_map AFTER : 235B13AF0DDEEB96CDD2C2031CA28C70550045006BC1A9FB1B3180D9423B32BC
unchanged: True
```

This is the specific defect the task named: `tools/probe_telemetry.py` rewrites `config/telemetry_map.json`
as a side effect — it did so in this session when this report's author ran it, and the file was restored
(§ preamble). The diagnostic export reads the map and does not write it.

### 5.3 What the export's transition number is, and is not

`gates.json` reads `docs/qualification/m2-headless/route-stress-50-gate-2s-after-teardown-fix.json` and
reports its `p50 2.628 s`, `37 of 50 over 2.0 s`. That is a **quoted committed measurement from M2**, not a
new M6 measurement, and the gate's `source` field names the file. M6 did **not** re-run the 50-transition
stress: doing so would be a new performance measurement, and the record already contains the distribution.

---

## 6. Signing — UNMET, and the step written down exactly

**Verdict: the package is not signed.** There is no Authenticode signature on `dist/M6-bundle`, no
code-signing certificate, no private key and no timestamp-service account on this machine. The signing step
was **written down and not performed**, and the artifact records the position itself: `signed: false` plus a
`signature_note` that begins `UNSIGNED.`

The exact commands a human holding a certificate would run, from `docs/install/SIGNING.md`:

```powershell
signtool sign /fd SHA256 /tr <timestamp-url> /td SHA256 /a <file>
signtool verify /pa /all <file>
```

- `/fd SHA256` — the file digest algorithm.
- `/tr <timestamp-url>` and `/td SHA256` — an **RFC 3161** timestamp and its digest; the timestamp is what
  keeps the signature valid after the certificate expires.
- `/a` — choose the best available certificate automatically; with more than one installed, name it with
  `/sha1 <thumbprint>` or `/n <subject>` instead.
- `/pa /all` on verify — the default Authenticode policy, and **every** signature rather than the first.

A plain ZIP is not an Authenticode-supported type, so the `<file>` is the artifact actually shipped — an
installer or launcher if one is built — not the bundle directory. Installing a certificate and running
`signtool` may need **elevation**: that is an **install step** a human performs once, before the booth. The
application itself runs as a standard user and nothing at runtime requires elevation.

### 6.1 What the absence costs

- **SmartScreen shows "unknown publisher"** and may warn before the package runs. Signing reduces this; it
  does not guarantee the warning disappears immediately, because publisher reputation is earned over time.
- **An enterprise policy may block it outright.** WDAC, AppLocker or a SmartScreen policy can refuse an
  unsigned artifact with no "run anyway" path. That is an organisational policy decision, not a defect in
  the bundle — and it is the reason a customer's procurement rule may reject this package.
- **Some customers will not accept it.** A requirement for a signed artifact is a real gate this package
  does not meet.

### 6.2 What is forbidden, and was not done

No Defender exclusion was added; no ASR rule was changed; the execution policy was not changed; AppLocker
and WDAC were not bypassed; SmartScreen was not disabled and no "run anyway" instruction depends on a
weakened control. Weakening a machine's defences to work around a missing signature lowers them for every
artifact on it, and an unsigned package is not a reason to do that. No signature, certificate, thumbprint,
timestamp or certification is fabricated anywhere in this repository, and the manifest's own note says the
same thing to anyone who checks the artifact without reading this document.

---

## 7. Every gate, M0–M6, in one table

**This is the table a customer reads.** Every row is a gate the project set itself, with the milestone that
measured it and where the evidence is. No gate is omitted because it failed, and none is softened into
`passed` because a neighbouring gate passed. Status vocabulary is the matrix's, plus `unmet` for a
deliverable that is absent rather than measured (signing), and two split values where a single word would
soften a real finding: the legibility row keeps its measured split, and the fallback row is
`partially observed` because one half was seen and the half the gate names was not.

| # | gate | status | milestone that measured it | evidence |
|---:|---|---|---|---|
| 1 | Three devices compile and complete **≥100 requests each** | **passed** | M0 | `M0-qualification.md` §Per-device compile: CPU 100/100 p50 24.07 ms, GPU 100/100 3.76 ms, NPU 100/100 5.27 ms, `EXECUTION_DEVICES` proving placement on every row. The table is **[quoted]** from `logs/qualification/m0.json`, which is **gitignored**; the committed evidence is the M0 report and `tools/preflight.py`, which re-rendered the same table through the same code in §1.2. **Not re-measured** by M1–M6. |
| 2 | Detection output passes the **annotated sanity fixture** | **passed** | M0, CPU re-run live at M6 | CPU/GPU/NPU each **PASS 2/2** (`M0-qualification.md` §Detection sanity fixture). The CPU case was **re-run live by preflight in this session**: `2 of 2 matched`, `EXECUTION_DEVICES=['CPU'] placed_on_requested=True`, `VERDICT PASS`. The fixture's blind spot is documented: a transposed `w`/`h` passes it. |
| 3 | **CPU fallback works when GPU/NPU is unavailable** | **partially observed; the gate is not closed** | M0 (a CPU-only route), M2 (the visible half and one recovery) | **A silent device fallback has never been observed.** What *was* observed, each time stated: (a) a **CPU-only route** with its own compile and its own 100 requests, p50 24.51 ms (`M0-qualification.md` §CPU-only fallback); (b) the **visible failure half** — a dead worker claims **no** device, `confirmed ''`, `fell_back false`, and prints its reason (`route-stress-fault-visible-fallback.json`, re-rendered live at M2); and (c) **a recovery to CPU executed by the product**: at the real 2.0 s deadline one route was not acknowledged within 2.010 s and the committed record ends with *"Recovering {'C': 'CPU'} - the last healthy route - and stating the failure rather than claiming the requested device"* (`route-stress-50-gate-2s-after-teardown-fix.json`, `errors_at_end`). That is a **route timeout recovered visibly**, not a device that was unavailable and had its fallback absorbed — M0 gate 9's **silent** half (`confirmed` differing from `requested` with nothing said) remains **not tested**, and no fault was invented to manufacture one. |
| 4 | Retail event at **≥90 % precision/recall on a curated set** | **not tested** | — | **No curated set exists and no precision/recall figure was ever measured.** The sanity fixture is **two hand-drawn objects on one frame**, a decode/geometry check, not a scored set; `M0-content-gate.md` §5.2 says the acceptance set needs ≥20 positives and ≥20 negative opportunities per pack and that **whether the footage is even long enough is not established**. The `oven` false positive at 0.250770 is visible and **unquantified**. The curated set is part of the content gate, which is itself unmet (row 12). |
| 5 | **People-in-zone presence, 2 per clip** | **passed** | M1, re-measured M2 and M4 | M4: **exactly 2 `zone_occupancy` events in each of the 18 pipeline windows** (36 events, dedup keys `<track>:person:presence:counter_band`), plus a full pass through `app.main --headless` analysing **1 694 of 1 694** frames and reporting `{'zone_occupancy': 2}`. `M4-qualification.md` §7.5. It is presence — **not a count and not a dwell**. |
| 6 | **Input acknowledgement <200 ms** | **passed as a floor** | M1, re-measured M4 | M1: p95 10.1–16.6 ms, worst tick 21.6–37.5 ms, through `run_windowed` with **6 CPU-only saturators** = `max(2, cpu_count()-2)` on 8 logical cores (`M1-qualification.md` §Input acknowledgement). M4's four-condition matrix reproduced it: gate **8/8 windows**, tick p95 5.6–12.6 ms, worst 12.3–91.9 ms. **It is a floor, not a bound**: at **8 of 8** saturated cores the same gate measured p95 195.6 ms and a **1 105.5 ms** maximum that still reported green. |
| 7 | **GPU/NPU utilisation attributed** | **passed** | M4 | Per-PID `\GPU Engine(*)\Utilization Percentage`, filtered by this process's own `pid_<n>`: GPU 0.0 → **97.19 %** median busy (n=26), NPU 0.0 → **64.86 %**, the probe's own PID 0.0 % while the worker read 97–100 %, a distractor on the same adapter 1.42 %. The withdrawn machine-wide 62 %/68 % figure is **not reinstated**; the published number is the **busiest engine**, not a device total. `M4-utilisation-attribution.md`. |
| 8 | **1920×1080 at four DPI scales** (100/125/150/200 %) | **passed with documented failures** | M3 | Four committed captures `m3-capture/{100,125,150,200}pct.{png,json}`, pixel dimensions measured from the images, `QImage.devicePixelRatio()` 1.00 at every scale; `overflow`, `truncated`, `absent`, `unreachable`, `geometry_unavailable` **empty at all four**; font census identical. The **readability review found failures** and documented them (`M3-readability-review.md` §4, §8) — hence the split value rather than a clean `passed`. |
| 9 | **Legibility at 2–3 m on the actual monitor** | **2 m cap pass; 3 m fail; x-height fails both** | M4 | 0.310 mm per logical px on the DELL S3422DWG (measured, 3440×1440 at 150 %), Arial cap 0.7164 em, x-height 0.5184 em: 44 px is **16.8′** cap at 2 m and **11.2′** at 3 m; x-height **12.2′ / 8.1′**; 56 px is 21.4′ / 14.2′. **Nothing clears 16′ of x-height at 2 m**, and **every element fails at 3 m on both measures.** The 55-inch panel that would read at 3 m (22.9′/16.6′) is a **procurement requirement with no gate** — preflight cannot measure a panel. `M4-legibility-review.md` §3.2. |
| 10 | **Warm route transition ≤2 s** | **MISSED** | M2; **not re-measured** by M3–M6 | `route-stress-50-after-teardown-fix.json`: p50 **2.261 s**, 25/50 over at the widened 30 s deadline. At the product's real **2.0 s** default, `route-stress-50-gate-2s-after-teardown-fix.json`: p50 **2.628 s**, **37 of 50** over, 26/50 acknowledged. Both committed; M2's criterion permitted "meet it **or** document the miss", and the miss is documented with its full distribution. The 2.238 s figure in M3–M5 and the guide is **not in any committed file** (§4.2). |
| 11 | **50 alternating warm transitions — no crash, no leak, no stale overlay** | **passed, leak qualified** | M2, leak re-measured and fixed M4 | M2: at the widened 30 s deadline, **50/50 acknowledged**, `crashes` **0**, `errors_at_end` **[]**, `stale_overlays` **0**, `placements_confirmed` 50/50, `placement_pids_matched` 50/50. At the real 2.0 s default: **26/50 acknowledged**, `crashes` **0**, `stale_overlays` **0**, and **one** entry in `errors_at_end` — the route timeout recovered to CPU quoted in row 3. **The leak was real and is fixed**: at the real 2 s deadline, 25 of 50 timeouts leaked **+153.7 MiB** before; after the drain fix, **−10.4 MiB** over the same 50 transitions and 25 timeouts, ledger 6 → 0. The 30 s case was clean either way (+8.2 → +0.4 MiB). Not a shift-long soak — **not tested**. `M4-qualification.md` §5. |
| 12 | **Content: 12 clips across 4 packs, 45–90 s, rights-cleared** | **NOT MET** | M5, re-run at M6 | `python -m tools.content_gate` in this session, exit **1**: **3 of 12 clips, 1 of 4 packs, 1 of 3 inside the 45–90 s band, 0 rights-cleared**. Measured from the files with `cv2.VideoCapture`: `retail-checkout` 113.0129 s (23.0129 s over, the one the app uses), `one-by-one` 139.4 s (49.4 s over), `store-aisle` 65.41535 s (in band, unused). No licence text is embedded and no licence file sits beside any clip; the M0 licence strings are **provenance, not clearance**. No footage was acquired, trimmed, padded or substituted. `M5-content-gate.md`, `M6` §1.2. |
| 13 | **Export excludes imagery by default** | **passed** | M5, re-proved at M6 | M5: 6 members, **0 image-content hits** in every member's bytes on a record holding three frames; opt-in gives 9 members and `png`/`jpeg`/`webp`, so the zero is a measurement. **M6 re-proved it in a different artifact**: the diagnostic ZIP's 7 members scan to `NONE` (§5.1), and the writer refuses a member that carries a magic number. |
| 14 | **A recorded mode cannot be mistaken for live** | **passed** | M5 | Rendered `Main.qml` scene graph: notice `Recorded demonstration — metrics are not live` visible and **still visible after 400 ms**; live label and live screen **not visible**; the device control and its `MouseArea` `enabled: false` at opacity 0.45, a real click emits `[]`; every figure dated from the recording; leaving restores a working live control; a playback over live work is refused with its reason. `M5-qualification.md` §5. |
| 15 | **An offline bundle with a per-file SHA-256 manifest, installing without network and running `--selftest`** | **passed** | M6 | §2: **64/64 SHA-256**, `checked 64 files; ok=True`; installed into a **fresh directory** with a dead proxy, dead `PIP_INDEX_URL` and `PIP_CONFIG_FILE=NUL`; **all 10 wheels** from the local wheelhouse, no index; `--selftest` **PASS**, exit 0, **32.44 s**. The bundle is **not committed** (`dist/` is gitignored). |
| 16 | **A known-good environment read from the machine**, unreadable fields `None`, three SKU records preserved | **passed** | M6 | §3: 69 fields re-read, **68 identical, 1 differs (the timestamp)**; three SKU records kept as three; `ac_status: null` because `Win32_Battery` returns 0 instances; both `11 Pro` and `Windows 10 Pro` retained. |
| 17 | **A presenter guide checked against the CLI and the three open gates** | **passed** | M6 | §4: **11 of 11** commands parse against the real parser; the three open gates are in it with their numbers; 13 committed tests `OK`. It contains one uncommitted number (**2.238 s**), flagged in §4.2 and §10. |
| 18 | **Code signing** | **UNMET** | M6 | §6: no certificate, no signature; manifest `signed: false`; the exact `signtool` commands, the cost (SmartScreen "unknown publisher", enterprise policy may block) and the prohibition on weakening Defender/ASR/AppLocker/execution policy are written down. **Nothing was signed and nothing is described as signed or certified.** |

### 7.1 The same table, as the matrix

`docs/qualification/qualification-matrix.csv` now has **42 rows**. M6 added five: the offline bundle, the
known-good environment, the presenter guide, the diagnostic export (all `passed`) and signing (`unmet`). The
content gate remains the one `failed` row from M5, and the 2 s transition remains the one `failed` row from
M2. Rows 1–14 above map to existing matrix rows; where a matrix row is `not tested` (the silent fallback, the
M3 input-ack matrix that M4 later closed, the three absent packs, INT8), that is the matrix being read as it
is, not an omission.

---

## 8. By outcome — the four buckets, stated plainly

### 8.1 Passed, because the claiming milestone measured it

- **Three devices, ≥100 requests each, placement proven** (M0).
- **The annotated sanity fixture, 2/2 on all three devices** (M0; CPU re-run live at M6).
- **People-in-zone presence, 2 per clip, 18/18 windows** (M1/M2/M4).
- **Input acknowledgement under 200 ms — as a floor** at `cpu_count−2` saturators (M1, M4). It is not a
  bound: 8/8 saturation produced a 1 105.5 ms maximum that still reported green.
- **GPU/NPU utilisation attributed per process** (M4).
- **50 alternating warm transitions with no crash and no stale overlay**, and the leak found, explained and
  fixed (M2, M4).
- **Imagery excluded from exports by default**, proven by bytes (M5, re-proved in the M6 diagnostic export).
- **Recorded mode cannot be mistaken for live** (M5).
- **The offline bundle: 64/64 verified, installed with no network, `--selftest` PASS** (M6).
- **The environment read from the machine, including what it cannot read** (M6).
- **The presenter guide checked against the parser and the gates** (M6).

### 8.2 Met in documented form only — the failure is real and written down

- **1920×1080 at four DPI scales: `passed with documented failures`** (M3). The render gate is met; the
  readability review found failures and they are in the record.
- **The 2 s warm-transition criterion's second clause**: the miss is documented with its full distribution
  (M2). **The target itself is not met** and is listed in §8.3.
- **The content gate's reporting obligation**: the shortfall is measured from the files and reported, exit 1
  (M5). **The gate itself is not met** and is listed in §8.3.

### 8.3 Missed or unmet

- **The 2 s warm route transition** — p50 2.261 s at the widened deadline and 2.628 s at the real 2.0 s
  default, 37 of 50 over at the default. **MISSED.**
- **The content gate** — 3 of 12 clips, 1 of 4 packs, 1 of 3 in band, **0 rights-cleared**. **NOT MET.**
  Shelf-to-basket remains undelivered.
- **Legibility at 3 m on the attached monitor** — 11.2′ cap and 8.1′ x-height; x-height fails at 2 m too.
- **Code signing** — no certificate exists. **UNMET.**

### 8.4 Not tested

- **CPU fallback when a device is unavailable** — **the gate itself is not closed.** A CPU-only route was
  measured (M0), the visible failure half was demonstrated (M2), and a **route timeout was recovered to CPU by
  the product and said so** (M2, at the real 2.0 s deadline). What has never been observed is a **device**
  becoming unavailable and its fallback being absorbed **silently** — M0 gate 9's second half. That cannot
  occur naturally on this machine and was not manufactured.
- **Retail precision/recall on a curated set** — no curated set exists; no figure was ever measured.
- **A shift-long soak** — M4's leak probe covers 50 transitions, not a day.
- **INT8** — no artifact exists and none is planned for V1 (M0 recorded the absence as `not tested`).

---

## 9. The final deliverable status

**What the operator has:**

1. **A Windows package — an unsigned candidate.** `dist/M6-bundle`: 64 payload files, 459,628,699 bytes, a
   SHA-256 manifest verified 64/64, an offline installer that was proven in a fresh directory with the
   network unusable, and `--selftest` PASS. **Unsigned**: no certificate exists, and the manifest says so.
2. **The source.** The same tree that built the bundle, with the five M6 test modules and the tools that
   produced every number here.
3. **Verified assets and their licences — as far as they go.** Six artifacts verified by content signature
   and SHA-256 against `config/models.json` (in this session: **6 PASS, 0 FAIL**). The **model weights are
   AGPL-3.0**, recorded and unresolved (a distribution decision, not an engineering one). **No clip is
   rights-cleared**: 0 of 3 have a licence file or embedded licence text, and the M0 licence strings are
   provenance, not permission.
4. **Qualification evidence.** `M0`–`M6` reports, the content gates, the legibility and utilisation reviews,
   the committed captures, trial records, benchmark records, the 42-row matrix, and the gate set pasted in
   §1.2 with its wall times.
5. **The presenter guide.** `docs/PRESENTER-GUIDE.md`, checked against the parser and the gates, saying what
   may and may not be said — **including the three open gates**, which is the point of it.

**What the operator does not have, named rather than closed by assertion:**

- a **signed** package (no certificate);
- **rights-cleared footage** (0 of 12 clips, 3 of 12 provisioned, 1 of 4 packs);
- a **screen that reads at 3 m** on the attached monitor (and no x-height that clears 2 m);
- a **two-second camera switch** (p50 2.628 s at the real default);
- **shelf-to-basket** (the event is people-in-zone presence, 2 per clip);
- **observed device fallback** — a device that fails and has its work absorbed without the screen saying so:
  never seen. A CPU-only route, a stated failure and one visible recovery to CPU were seen (§7 rows 3, 11);
- a **curated precision/recall set**;
- **AC status** and a **single SKU** (both unreadable on this machine, recorded as such);
- a **bundle that carries its own signing document and presenter guide** (§2.1 gap).

### 9.1 The three largest open risks

1. **Content — 3 of 12 clips, 0 rights-cleared.** This is the largest, because it is the one that cannot be
   engineered: nine clips do not exist, one of the three is 49.4 s over the band, none has a redistribution
   grant, and shelf-to-basket was never filmed. A booth built on this footage demonstrates one scenario in
   one place, and may not be redistributable at all. Owner: procurement / operator.
2. **Legibility at 3 m.** On the actual DELL S3422DWG the screen is 11.2′ of cap height at 3 m — below the
   16′ threshold — and **no element clears 16′ of x-height even at 2 m**. A visitor standing three metres
   away cannot read the numbers, and no software change fixes a panel. The 55-inch requirement would read at
   3 m (22.9′/16.6′) and is a **procurement item with no gate**: preflight cannot measure a panel.
   Owner: procurement / operator.
3. **The missed 2 s warm transition.** p50 **2.628 s** and **37 of 50** over at the real default, against a
   2 s target. A presenter switching a camera's device will be seen to wait — and the run's own distribution
   shows the tail is several seconds. It is a measured miss with a documented distribution, not a rounding
   question. Owner: engineering, with M2's distribution as the baseline.

Below those three: **signing** (a procurement rule may reject the package outright), **no observed
fallback** (a device failure at a booth has unit-tested handling and a demonstrated loud path, not a proven
silent one), and the **stale per-boot LUID map** (a WARN, not a device fault).

---

## 10. Not closed — and not written as closed

| item | status | owner |
|---|---|---|
| **The content gate** | **NOT MET** — 3 of 12 clips, 1 of 4 packs, 1 of 3 in band, **0 rights-cleared**; shelf-to-basket undelivered. Re-run in this session, exit 1 | procurement / operator |
| **Legibility at 3 m on the attached monitor** | **FAILS** — 11.2′ cap, 8.1′ x-height; x-height also fails at 2 m. The 55-inch panel is a requirement with **no gate** | procurement |
| **The 2 s warm-transition target** | **MISSED** — p50 2.261 s at the widened deadline, **2.628 s** and 37/50 over at the real default. **Not re-measured by M3–M6** | engineering |
| **Code signing** | **UNMET** — no certificate on this machine; the step and its cost are written down, not performed | operator / procurement |
| **The bundle omits `SIGNING.md`, `PRESENTER-GUIDE.md` and `KNOWN-GOOD-environment.json`** | **FOUND IN THIS TASK, NOT FIXED** — `source_paths()` names only `docs/install/SETUP.md`; the manifest's signature note points at SETUP.md, which points at a signing document that is not in the bundle. Adding three paths and rebuilding is the fix | tool owner |
| **`docs/PRESENTER-GUIDE.md` prints "M2 measured p50 2.238 s"** | **NO COMMITTED SOURCE FOR 2.238** — the committed p50s are 2.261 s and 2.628 s. The same sentence prints both committed values, so the guide is not wrong about the gate; the 2.238 half should be removed or traced. **Not edited here** (this task is documentation of the record) | guide owner |
| **M0 gate 9's SILENT half** | **OPEN, cannot close on this machine** — all three devices execute where asked; the compile-failure seam is a loud failure; no fault was invented | M6 fault injection (unassigned) |
| **Retail precision/recall on a curated set** | **NOT TESTED** — no curated set exists; footage sufficiency for ≥20 positives per pack is not established | content / M5 acceptance set |
| **The `oven` false positive at 0.250770** | **UNQUANTIFIED** — deliberately unannotated; no precision/recall claim | detector owner |
| **AGPL-3.0 for the weights** | **UNRESOLVED** — recorded since M0; a distribution decision that the M6 bundle does not settle | operator / legal |
| **`config/telemetry_map.json` is stale** | **REPORTED, NOT EDITED** — the WARN row; per-boot LUIDs. **This session restored it after `probe_telemetry.py` rewrote it**; hash `235B13AF…` | operator |
| **`tools/probe_telemetry.py` rewrites a verified config file as a side effect** | **OBSERVED IN THIS SESSION** — it changed 21 lines of `config/telemetry_map.json`; restored. The diagnostic export does not (§5.2) | tool owner |
| **`python tools/build_bundle.py --verify <dir>` cannot be used alone** | **WART** — `--destination` and `--wheelhouse` are `required=True`, so `--verify` needs two arguments it does not use, contradicting its help text. Worked around by passing both | tool owner |
| **`python tools/compare_records.py …` as its docstring documents** | **DOES NOT RUN** (`ModuleNotFoundError: No module named 'app'`); `python -m tools.compare_records` works and is what the guide prints | tool owner |
| **The Tasks 4–5 QML-precedence claim** | **NOT REPRODUCED** — the unparenthesised binding hid the live screen twice; the delivered behaviour is verified, the stated reason for the parentheses is not | reviewer |
| **A JSON-loaded record cannot re-export imagery** | **BY DESIGN** — imagery is in-memory only; stated as a limit | design |
| **A shift-long soak** | **NOT TESTED** — M4's 50-transition probe is not a day | M6 (unassigned) |
| **`test_main` 30.0 % substring flake; route-stress GC-timing flake** | **KNOWN; not observed in this run's suite** | M0–M4 owner |

---

## 11. What M6 establishes

- **The package can be installed with no network, from a folder, and proves itself.** 64 files, 64/64
  SHA-256, ten wheels that are exactly the pinned closure, a fresh-directory install behind a dead proxy, and
  `PASS` from the self-test — after the same bytes were refused four different ways when the wheelhouse was
  wrong.
- **The environment is a reading, not a memory.** 69 fields re-read, 68 identical to the committed record,
  and the unreadable ones — AC status, a single SKU — are `null` with reasons rather than guesses.
- **The guide cannot drift from the CLI without a test failing.** 11 of 11 commands parse; the three open
  gates are in the text with the numbers that would otherwise be contradicted at the booth.
- **A diagnosis carries the failures and not the customers.** Seven members, all three gates marked failed,
  and not one image magic number in any member's bytes.
- **The signing position is stated, not implied.** Unsigned, no certificate, the exact commands, the cost,
  and a refusal to weaken anything to work around it.
- **And the closing table does not round anything up.** Three gates are missed or unmet, one is met only in
  documented form, two were never tested, and the largest of them — the footage — is a procurement
  dependency that no further engineering on this machine can close.

---

## Appendix A — every number in this report, and where it came from

| claim | source |
|---|---|
| 6 PASS / 0 FAIL, wall 0.14 s | `tools/verify_sources.py`, run in §1.2 |
| 17 PASS / 1 WARN / 0 FAIL, `Limited`, exit 0, wall 4.51 s; 189.4/381.5 GiB, 22.3/31.5 GiB, 2.82 s, 4.02 s, 358 instances | `tools/preflight.py`, run in §1.2 |
| `PASS`, exit 0, wall 1.01 s, `24.40 ms` | `app.main --selftest`, run in §1.2 |
| suite total and wall time; 62 new tests in 5 modules | the suite block in §1.2 (measured there); the per-module counts are `def test_` counts in the committed files |
| 64 payload files, 459,628,699 bytes; 65 files / 459,636,178 bytes on disk; the by-directory counts and file sizes | `bundle-manifest.json` and `os.path.getsize` over its entries, this session; `Get-ChildItem` for the assets/config/docs listing |
| `MANIFEST PASS: 64 files…`, exit 0, 0.38 s | `provision_offline.py --verify-only` from `dist/M6-bundle`, §2.3 |
| `checked 64 files; ok=True` | `tools/build_bundle.py … --verify dist\M6-bundle`, §2.3 |
| 6 direct pins, 10-wheel closure, 4 declared dependencies, `exact match: True` | `parse_requirements` / `dependency_closure` / `validate_wheelhouse` from the committed builder, §2.2 |
| the four REFUSED wheelhouses | `validate_wheelhouse` over four constructed wheelhouses, §2.2 |
| the install transcript, `Successfully installed` ×10, `PASS`, exit 0, 32.44 s | `provision_offline.py --selftest` in a fresh directory with a dead proxy / `PIP_CONFIG_FILE=NUL`, §2.3 |
| 69 fields compared, 1 differing (`generated_at_utc`), three SKU values, `ac_status None` | `tools.environment_manifest.collect_environment()` against `docs/install/KNOWN-GOOD-environment.json`, §3 |
| CPU/GPU/NPU/RAM/monitor/power-plan values | the same collected manifest, §3.1 |
| 11 commands, 11 accepted | the guide-parser check in §4.1, using the committed test's regex |
| 13 tests `OK`, 0.021 s | `python -m unittest tests.test_presenter_guide -v`, §4.1 |
| 2.261 s / 25-of-50 and 2.628 s / 37-of-50 | `docs/qualification/m2-headless/route-stress-50-after-teardown-fix.json` and `…-gate-2s-after-teardown-fix.json`, read in §4.2 |
| 7 members, their byte counts, `IMAGE-CONTENT HITS: NONE`, gate details, `all gates passed: False`, exit 0, 6.42 s | `python -m tools.diagnostic_export`, §5, and an independent whole-byte scan of the ZIP's members |
| `235B13AF…` before and after | `Get-FileHash config\telemetry_map.json -Algorithm SHA256`, §5.2 |
| 16.8′/11.2′/12.2′/8.1′; 0.310 mm per logical px; 22.9′/16.6′ on the 55-inch panel | `M4-legibility-review.md` §3.2 and `M4-qualification.md` §3.2 — M4's measurement, **quoted**, not re-derived |
| 18/18 windows, 36 events, 1 694/1 694 frames | `M4-qualification.md` §7.5; `m4-sustained-trials.json` |
| input-ack p95 10.1–16.6 ms / worst 21.6–37.5 ms (M1); 8/8 windows, p95 5.6–12.6 ms, worst 12.3–91.9 ms, 8/8 saturation p95 195.6 ms / max 1 105.5 ms | `M1-qualification.md` §Input acknowledgement; `M4-qualification.md` §4; `m4-input-ack.json`; the 8/8 row is in `tests/test_main.py::InputAcknowledgement`'s docstring |
| 97.19 % GPU / 64.86 % NPU, probe 0.0 %, distractor 1.42 % | `m4-utilisation-control.json`, `M4-utilisation-attribution.md` |
| four captures, pixel sizes, empty overflow/unreachable lists | `docs/qualification/m3-capture/*`, `M3-readability-review.md` §5, §8 |
| leak +153.7 → −10.4 MiB, ledger 6 → 0, 30 s case +8.2 → +0.4 MiB | `M4-qualification.md` §5; `m4-leak-*.json` |
| 3 of 12 clips, 1 of 4 packs, 1 of 3 in band, 0 rights-cleared; 113.0129 / 65.41535 / 139.4 s; +23.0129 / +49.4 s over; exit 1 | `python -m tools.content_gate`, §1.2 (run in this session); `M5-content-gate.md` |
| placement `('NPU',)`, 100 requests per device, p50 24.07 / 3.76 / 5.27 ms, CPU-only 24.51 ms | `M0-qualification.md` §Per-device compile and §CPU-only fallback — **[quoted]** there from gitignored `logs/qualification/m0.json`, re-rendered by preflight in §1.2 |
| 2/2 sanity fixture on three devices; live CPU `2 of 2 matched` | `M0-qualification.md` §Detection sanity fixture; preflight in §1.2 |
| signature/`signtool` commands, the cost, the prohibitions | `docs/install/SIGNING.md`, quoted in §6 |

Arithmetic that is arithmetic, and labelled as such where it appears: **459,636,178 − 459,628,699 = 7,479
bytes** for the manifest; **1 573 + 62 = 1 635** expected tests; **10 = 6 pins + 4 declared dependencies**;
and the **9 missing clips / 3 missing packs** counts from the requirement and the measured inventory.
**Nothing else in this report is derived rather than measured.**
