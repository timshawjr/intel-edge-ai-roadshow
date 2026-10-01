# M0 — Hardware and Model Path Qualification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove on this exact machine that OpenVINO compiles and executes one pinned detector on CPU, GPU, and NPU, and record what telemetry is actually available — as a qualification report built only from real measurements.

**Architecture:** A small set of pure, unit-tested modules (device registry, preprocessing, decoder) beneath a qualification runner that drives them. Discovery artifacts (device description, model I/O description, PDH instance map) are written to `config/` as recorded, re-verifiable data rather than being hardcoded in source.

**Tech Stack:** Python 3.12.10, `openvino==2026.4.0`, `openvino.runtime` (dev only), `numpy==2.5.3`, `opencv-python==5.0.0.93`, `psutil==7.2.2`, `ctypes` for PDH. Tests use `unittest` from the standard library — `pytest` is deliberately not added, keeping the pinned dependency set minimal.

**Spec:** `docs/superpowers/specs/2026-10-01-intel-edge-ai-roadshow-design.md` (approved design) and the source-of-truth `Intel-Edge-AI-Roadshow-Specification.md` §6, §7, §10 M0. Executors read both.

## Global Constraints

These bind every task. Values are copied from the approved design and the specification.

- Target machine is Intel Core Ultra 9 288V (Lunar Lake, Core Ultra Series 2), ASUS NUC14LNK, Windows 11 Pro build 26200, 33805881344 bytes RAM.
- Never present a Lunar Lake measurement as Panther Lake (Series 3) or Arrow Lake. Show the discovered `FULL_DEVICE_NAME`, never a hardcoded family string.
- Pinned exactly, no ranges, no "latest": `openvino==2026.4.0`, `PySide6==6.11.2`, `numpy==2.5.3`, `opencv-python==5.0.0.93`, `psutil==7.2.2`, `PyYAML==6.0.3`.
- **Static shapes only.** The NPU rejects dynamic shapes. Baseline is batch 1, 640x640.
- Explicit `compile_model` targets only. `AUTO`, `HETERO` and `MULTI` are not used in the primary routing demo.
- `EXECUTION_DEVICES`, re-read **after the first inference**, is the only proof of placement. A requested device preference is never evidence of execution.
- Unavailable is `null`, never `0`. Anything older than 2 s is stale.
- An application-derived duty cycle may appear only in diagnostics, explicitly named as an estimate — never under a utilization label.
- No network at runtime. No simulated metrics. No fabricated result may be reported as measured.
- Application runs as a standard user. No elevation at runtime; elevation is an install step only.
- Layout tokens belong in `app/theme.py` (M0 creates no UI).
- Commit after every task with the message form `M0: <what works now>` or a conventional `feat:`/`fix:` prefix.

## Review Focus

Failure modes the specification implies but no naive implementation would test. Each is pinned by a test in the owning task.

1. **PDH counter names fail on a non-English Windows, or the NPU `Neural` engine instance is absent.** A person at the booth must never see "NPU 0%" when that means *no counter exists* — they would conclude the NPU is broken. Expect: probe fails closed, `npu_percent` is `null`, and any NPU figure is labelled as unavailable rather than zero.
2. **GPU and NPU report overlapping physical device ids, so a naive mapping puts the NPU's engine on the GPU's card (or vice versa).** Expect: the LUID is the only tie-breaker, a collision is a hard error, and a swap is impossible to commit silently.
3. **The export's output layout is not the assumed `(1, 84, 8400)` convention.** A wrong transpose yields plausible-looking boxes over the wrong objects. Expect: the decoder is driven by observed tensor names/shapes, and a test asserts the classifier's verdict against the recorded model I/O description.
4. **Letterbox coordinate recovery omits the inverse pad/scale, which is invisible on a centred object and wrong on every other one.** Expect: round-trip tests across asymmetric sizes, and a test that a detection at the far corner recovers its source coordinates.
5. **A compiled blob is reused after a driver update.** NPU blob-cache compatibility is not guaranteed across driver versions. Expect: the cache key includes driver and runtime identity and invalidates on mismatch.

---

## File Structure

| File | Responsibility |
|---|---|
| `requirements.txt` | exact pins, no ranges |
| `tools/provision_venv.py` | one-time venv + dependency install (the only step needing network) |
| `tools/verify_sources.py` | content-verified source checks: XML, MP4 `ftyp`, binary payloads, SHA-256 inventory |
| `app/__init__.py` | package marker |
| `app/contracts/__init__.py` | re-exports the contract dataclasses |
| `app/contracts/device_status.py` | `DeviceStatus`, `Validity`, `to_json`/`from_json` |
| `app/contracts/telemetry_availability.py` | `TelemetryAvailability` — what a provider can and cannot supply |
| `app/inference/__init__.py` | package marker |
| `app/inference/devices.py` | `DeviceRegistry`, `TargetSpec`, `CompiledTarget` over `ov.Core` |
| `app/inference/preprocess.py` | `Letterbox`, `letterbox_params`, `preprocess_bgr`, `to_source` — pure |
| `app/inference/decode.py` | `ModelOutputSpec`, `describe_outputs`, `Detection`, `decode_detections` — pure |
| `app/telemetry/__init__.py` | package marker |
| `app/telemetry/pdh.py` | `PdhQuery`, `PdhCounterPath`, `PdhUnavailableError` — ctypes PDH reader |
| `app/telemetry/instance_map.py` | `EngineInstanceRule`, `InstanceMap`, `load_instance_map` — LUID-keyed matching |
| `tools/dump_model_io.py` | writes `config/model_io/<model_id>.json` from the pinned IR |
| `tools/probe_telemetry.py` | writes `config/telemetry_map.json`; fails closed without an NPU counter |
| `tools/qualify_m0.py` | the M0 runner: compile, warm, >=100 requests, EXECUTION_DEVICES, latency, memory |
| `tools/preflight.py` | local PASS/WARN/FAIL gate over assets, devices, compile, telemetry, sanity |
| `tests/fixtures/retail_sanity/*.jpg` + `retail_sanity.json` | hand-annotated detection fixture |
| `config/models.json` | pinned artifacts, hashes, licence, provenance |
| `config/model_io/yolo11n-fp16.json` | recorded IR tensor names/shapes/layout |
| `config/telemetry_map.json` | recorded LUID -> device engine rules |
| `docs/qualification/M0-qualification.md` | the report, with pasted raw output |
| `docs/qualification/M0-content-gate.md` | rights status and remaining footage requirements |
| `docs/qualification/qualification-matrix.csv` | scenario x model x device x machine x versions, per cell |
| `bench_report.md` | appended measured evidence |

---

### Task 1: Pinned environment and provisioned assets

Establishes the only toolchain the rest of M0 runs on, and the recorded asset inventory everything else cites.

**Files:**
- Create: `requirements.txt`, `tools/provision_venv.py`, `tools/verify_sources.py`
- Create: `config/models.json`
- Create: `assets/scenarios/retail_sanity/manifest.json`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `requirements.txt`; `tools/verify_sources.py` CLI exiting 0 on success; `config/models.json` loaded by Tasks 3, 5, 6 via `json.loads(Path("config/models.json").read_text(encoding="utf-8"))`, shaped as `{"models": [{"id", "ir_path", "sha256", "license", "provenance", "input_shape": [1,3,640,640]}], "media": [...]}`

- [ ] **Step 1: Write `requirements.txt` with exact pins**

```
openvino==2026.4.0
PySide6==6.11.2
numpy==2.5.3
opencv-python==5.0.0.93
psutil==7.2.2
PyYAML==6.0.3
```

- [ ] **Step 2: Write the failing test `tests/test_pins.py`**

```python
import re, unittest
from pathlib import Path

PINNED = {"openvino", "PySide6", "numpy", "opencv-python", "psutil", "PyYAML"}

class PinsAreExact(unittest.TestCase):
    def test_every_requirement_is_exactly_pinned(self):
        text = Path("requirements.txt").read_text(encoding="utf-8")
        seen = set()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            self.assertRegex(line, r"^[A-Za-z0-9_.-]+==[^=<>!]+$",
                             f"not an exact pin: {line!r}")
            seen.add(line.split("==", 1)[0])
        self.assertEqual(seen, PINNED)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_pins -v`
Expected: `ERROR` — `FileNotFoundError` for `requirements.txt`.

- [ ] **Step 4: Add `requirements.txt` (Step 1 content) and `tools/provision_venv.py`**

`provision_venv.py` creates `.venv` and runs `pip install -r requirements.txt`, printing the resolved `pip freeze` afterwards. It is the single documented step that needs network, and it must state so in its docstring.

- [ ] **Step 5: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_pins -v`
Expected: `ok` — proves every requirement is exactly pinned and the set matches.

- [ ] **Step 6: Provision the venv (network required, one time)**

Run: `.venv\Scripts\python.exe tools\provision_venv.py`
Expected: `pip install` completes for all six pins and prints a `pip freeze`. If network is unavailable, stop and report it — do not substitute different versions.

- [ ] **Step 7: Write the failing test `tests/test_verify_sources.py`**

```python
import unittest
from tools.verify_sources import looks_like_xml, looks_like_mp4, looks_like_binary, Source, Result

class ContentVerification(unittest.TestCase):
    def test_xml_must_declare_itself(self):
        self.assertTrue(looks_like_xml(b'<?xml version="1.0"?><net/>'))
        self.assertFalse(looks_like_xml(b'<net/>'))
        self.assertFalse(looks_like_xml(b'{"error": "not found"}'))

    def test_mp4_must_carry_an_ftyp_box(self):
        self.assertTrue(looks_like_mp4(b'\x00\x00\x00\x18ftypmp42' + b'\x00' * 8))
        self.assertFalse(looks_like_mp4(b'<html>404</html>'))

    def test_binary_payload_must_not_be_html_or_an_lfs_pointer(self):
        self.assertTrue(looks_like_binary(bytes(range(256)) * 4))
        self.assertFalse(looks_like_binary(b'<!DOCTYPE html><html>nope</html>'))
        self.assertFalse(looks_like_binary(b'version https://git-lfs.github.com/spec/v1\n'))
```

- [ ] **Step 8: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_verify_sources -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'tools.verify_sources'`.

- [ ] **Step 9: Implement `tools/verify_sources.py`**

Signatures the test pins:

```python
def looks_like_xml(payload: bytes) -> bool          # must start with b"<?xml"
def looks_like_mp4(payload: bytes) -> bool          # must contain b"ftyp" in bytes 4..8
def looks_like_binary(payload: bytes) -> bool       # reject HTML and git-lfs pointers

@dataclass(frozen=True)
class Source:
    id: str
    kind: str          # "xml" | "mp4" | "binary"
    path: Path

@dataclass(frozen=True)
class Result:
    source: Source
    ok: bool
    detail: str
```

Plus a `main()` that verifies every local file named in `config/models.json` against its recorded SHA-256 and content shape, printing `Source verification: N PASS, 0 FAIL` and exiting non-zero on any failure. It checks **content**, never HTTP status, because a 200 can still return an error page.

- [ ] **Step 10: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_verify_sources -v`
Expected: `ok` — proves a 200-with-error-body is rejected, which is the whole point.

- [ ] **Step 11: Populate `config/models.json` and copy the artifacts**

Copy from the existing verified local cache, never re-download: `yolo11n-fp16` (`yolo11n.xml` + `yolo11n.bin`), and the media clips. Record for each model `id`, `ir_path`, `sha256` (compute with `hashlib.sha256`), `license`, `provenance`, `input_shape: [1,3,640,640]`. Record for each clip its id, path, sha256, measured width/height/fps/duration via `cv2.VideoCapture`, `license`, and `distribution_status`.

Carried-over licence strings, to be verified by reading each source's terms, not assumed: Pexels License (commercial use, no attribution); Mixkit Free License (commercial use, no attribution); Open Model Zoo sample media (**redistribution status recorded as-is, not claimed as booth clearance**).

- [ ] **Step 12: Run source verification**

Run: `.venv\Scripts\python.exe tools\verify_sources.py`
Expected: every row PASS. Paste the output into `docs/qualification/M0-qualification.md` under "Assets".

- [ ] **Step 13: Commit**

```bash
git add requirements.txt tools/ config/models.json assets/scenarios/ tests/test_pins.py tests/test_verify_sources.py
git commit -m "M0: pinned environment and content-verified asset inventory"
```

---

### Task 2: Device discovery and the `DeviceStatus` contract

The foundation every later layer reports through. If discovery is wrong, every number in the product is wrong.

**Files:**
- Create: `app/__init__.py`, `app/contracts/__init__.py`, `app/contracts/device_status.py`
- Create: `app/inference/__init__.py`, `app/inference/devices.py`, `app/inference/cache.py`
- Test: `tests/test_device_status.py`, `tests/test_devices.py`, `tests/test_cache_key.py`

**Interfaces:**
- Consumes: `config/models.json` (Task 1)
- Produces:
  - `app.contracts.device_status.DeviceStatus` — frozen dataclass, fields `device_id: str`, `full_name: str`, `device_type: str`, `available: bool`, `validated: bool`, `ready: bool`, `runtime_version: str`, `driver_version: str | None`, `supported_model_ids: tuple[str, ...]`, `unavailable_reason: str | None`; methods `to_json() -> dict` and `from_json(payload: Mapping[str, Any]) -> DeviceStatus`
  - `app.inference.devices.DEVICE_IDS: Final = ("CPU", "GPU", "NPU")`
  - `app.inference.devices.TargetSpec` — frozen dataclass: `device_id: str`, `model_path: Path`, `input_shape: tuple[int, int, int, int]`
  - `app.inference.devices.CompiledTarget` — frozen dataclass: `device_id`, `model_path`, `model_hash`, `input_shape`, plus method `execution_devices_now(core: ov.Core, request: ov.InferRequest) -> tuple[str, ...]` that reads `EXECUTION_DEVICES` **live**, not cached at compile time
  - `app.inference.devices.DeviceRegistry` — `__init__(self, core: ov.Core)`, `available(self) -> tuple[str, ...]`, `full_name(self, device_id: str) -> str`, `describe(self) -> tuple[DeviceStatus, ...]`, `compile_target(self, spec: TargetSpec) -> CompiledTarget`

- [ ] **Step 1: Write the failing test `tests/test_device_status.py`**

```python
import unittest
from app.contracts.device_status import DeviceStatus, Validity

class DeviceStatusRoundTrip(unittest.TestCase):
    def test_round_trips_through_json(self):
        s = DeviceStatus(device_id="NPU", full_name="Intel(R) AI Boost",
                         device_type="NPU", available=True, validated=True, ready=True,
                         runtime_version="2026.4.0", driver_version="32.0.100.5540",
                         supported_model_ids=("yolo11n-fp16",), unavailable_reason=None)
        again = DeviceStatus.from_json(s.to_json())
        self.assertEqual(again, s)

    def test_unavailable_carries_a_reason(self):
        s = DeviceStatus(device_id="NPU", full_name="", device_type="NPU",
                         available=False, validated=False, ready=False,
                         runtime_version="2026.4.0", driver_version=None,
                         supported_model_ids=(), unavailable_reason="no PDH Neural engine")
        self.assertIsNotNone(s.unavailable_reason)
        self.assertFalse(s.available)

    def test_unavailable_is_a_distinct_state_from_zero(self):
        self.assertIsNot(Validity.UNAVAILABLE, Validity.STALE)
        self.assertNotEqual(Validity.UNAVAILABLE.value, "0")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_device_status -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.contracts.device_status'`.

- [ ] **Step 3: Implement `app/contracts/device_status.py`**

As pinned in Interfaces. `Validity` is a `str, Enum` with `VALID`, `STALE`, `UNAVAILABLE`. `from_json` coerces `supported_model_ids` back to a tuple so equality holds after a round trip.

- [ ] **Step 4: Write the failing test `tests/test_cache_key.py`**

```python
import unittest
from app.inference.cache import cache_key, cache_is_valid

BASE = dict(model_hash="a" * 64, device_id="NPU", runtime_version="2026.4.0",
            driver_version="32.0.100.5540", properties="THROUGHPUT")

class CacheKeyTracksEverythingThatCanInvalidateABlob(unittest.TestCase):
    def test_key_is_stable_for_identical_inputs(self):
        self.assertEqual(cache_key(**BASE), cache_key(**BASE))

    def test_a_driver_update_changes_the_key(self):
        # Review Focus #5: NPU blob compatibility is not guaranteed across drivers.
        other = dict(BASE, driver_version="32.0.100.9999")
        self.assertNotEqual(cache_key(**BASE), cache_key(other))

    def test_runtime_model_device_and_properties_each_change_the_key(self):
        for field, value in [("model_hash", "b" * 64), ("device_id", "GPU"),
                             ("runtime_version", "2025.1.0"),
                             ("properties", "LATENCY")]:
            self.assertNotEqual(cache_key(**BASE), cache_key(**dict(BASE, **{field: value})),
                                f"{field} does not affect the cache key")

    def test_a_blob_from_another_machine_state_is_rejected(self):
        recorded = dict(BASE)
        self.assertTrue(cache_is_valid(recorded, recorded))
        self.assertFalse(cache_is_valid(recorded, dict(BASE, driver_version="32.0.100.9999")))
```

- [ ] **Step 5: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_cache_key -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.inference.cache'`.

- [ ] **Step 6: Implement `app/inference/cache.py`**

```python
def cache_key(*, model_hash: str, device_id: str, runtime_version: str,
              driver_version: str, properties: str) -> str:
    """Identity of a compiled artifact. Any input change must invalidate the blob."""

def cache_is_valid(recorded: Mapping[str, str], current: Mapping[str, str]) -> bool:
    """True only when every key in `current` matches `recorded`."""
```

Both are pure and total. A compiled blob is never reused across a driver or runtime change, and
never distributed as universally compatible.

- [ ] **Step 7: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_cache_key -v`
Expected: `ok`.

- [ ] **Step 8: Write the failing test `tests/test_devices.py`**

```python
import unittest
from app.inference.devices import DeviceRegistry, DEVICE_IDS
import openvino.runtime as ov

class DeviceDiscovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = ov.Core()

    def test_registry_reports_exactly_the_three_demo_devices(self):
        self.assertEqual(DEVICE_IDS, ("CPU", "GPU", "NPU"))

    def test_all_three_devices_are_present_on_this_machine(self):
        reg = DeviceRegistry(self.core)
        for d in DEVICE_IDS:
            self.assertIn(d, reg.available(),
                          f"{d} not available; got {reg.available()}")

    def test_full_name_is_discovered_not_hardcoded(self):
        reg = DeviceRegistry(self.core)
        self.assertIn("288V", reg.full_name("CPU"))
        self.assertIn("Arc", reg.full_name("GPU"))
        self.assertIn("AI Boost", reg.full_name("NPU"))

    def test_describe_reports_the_discovered_names(self):
        for s in DeviceRegistry(self.core).describe():
            self.assertTrue(s.full_name, f"{s.device_id} has an empty full_name")
            self.assertTrue(s.available)
```

- [ ] **Step 6: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_devices -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.inference.devices'`.

- [ ] **Step 9: Implement `app/inference/devices.py`**

`DeviceRegistry.describe()` builds one `DeviceStatus` per id in `DEVICE_IDS`, marking `available` from `core.available_devices` and filling `unavailable_reason` with the OpenVINO error text when absent. `compile_target` calls `core.compile_model(core.read_model(str(spec.model_path)), spec.device_id)`, computes `model_hash` from the `.xml` file, and returns a `CompiledTarget`.

**Critical:** `CompiledTarget` must not capture `EXECUTION_DEVICES` at construction. Store the `InferRequest` and read the property in `execution_devices_now()`, so a caller can never display a placement that was true before the first inference ran.

- [ ] **Step 10: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_devices tests.test_cache_key tests.test_device_status -v`
Expected: `ok`, 10 tests. Paste the output into the M0 report under "Device discovery" — this is first-hand evidence that all three devices are enumerated.

- [ ] **Step 11: Commit**

```bash
git add app/ tests/
git commit -m "M0: device discovery, DeviceStatus contract, driver-aware cache keys"
```

---

### Task 3: PDH telemetry provider and the LUID-keyed instance map

The highest-risk task in M0. Windows exposes GPU engine counters per **process and adapter LUID**, and the NPU and GPU report overlapping physical device ids — so the LUID is the only safe tie-breaker. A wrong mapping puts a real, wrong number on the NPU card, which is worse than showing nothing.

**Files:**
- Create: `app/telemetry/__init__.py`, `app/telemetry/pdh.py`, `app/telemetry/instance_map.py`
- Create: `tools/probe_telemetry.py`, `config/telemetry_map.json`
- Test: `tests/test_instance_map.py`, `tests/test_pdh.py`

**Interfaces:**
- Consumes: `app.contracts.device_status` (Task 2)
- Produces:
  - `app.telemetry.instance_map.EngineInstanceRule` — frozen dataclass: `luid: str`, `device: str`, `engtype: str | None`
  - `app.telemetry.instance_map.InstanceMap` — `rules(self) -> tuple[EngineInstanceRule, ...]`, `rules_for(self, device: str) -> tuple[EngineInstanceRule, ...]`, `match(self, instance_name: str, pid: int) -> EngineInstanceRule | None`
  - `app.telemetry.instance_map.load_instance_map(path: Path) -> InstanceMap` — raises `ValueError` on an LUID mapped to two different devices
  - `app.telemetry.pdh.PdhCounterPath` — frozen dataclass: `category: str`, `instance: str`
  - `app.telemetry.pdh.PdhQuery` — `__init__(self, category: str, counters: Mapping[str, str])`, `instances(self) -> tuple[str, ...]`, `read(self, path: PdhCounterPath) -> float | None`, `close(self) -> None`; raises `PdhUnavailableError` from `__init__` when the category or counter cannot be opened
  - `tools/probe_telemetry.py` CLI writing `config/telemetry_map.json`

- [ ] **Step 1: Write the failing test `tests/test_instance_map.py`**

```python
import unittest
from app.telemetry.instance_map import (EngineInstanceRule, InstanceMap,
                                        parse_instance_name, load_instance_map)

GPU = "pid_1234_luid_0x00000000_0x00010E9E_phys_0_eng_5_engtype_Compute"
NPU = "pid_1234_luid_0x00000000_0x000112A0_phys_0_eng_0_engtype_Neural"

class InstanceNameParsing(unittest.TestCase):
    def test_parses_luid_and_engtype(self):
        r = parse_instance_name(GPU)
        self.assertEqual(r.luid, "0x00010E9E")
        self.assertEqual(r.engtype, "Compute")
        self.assertEqual(r.pid, 1234)

    def test_parses_the_neural_engine(self):
        self.assertEqual(parse_instance_name(NPU).engtype, "Neural")

    def test_rejects_a_name_without_a_luid(self):
        with self.assertRaises(ValueError):
            parse_instance_name("something else entirely")

class InstanceMapMatching(unittest.TestCase):
    def setUp(self):
        self.m = InstanceMap((
            EngineInstanceRule("0x00010E9E", "GPU", "Compute"),
            EngineInstanceRule("0x000112A0", "NPU", "Neural"),
        ))

    def test_gpu_and_npu_never_cross_match(self):
        self.assertEqual(self.m.match(GPU, 1234).device, "GPU")
        self.assertEqual(self.m.match(NPU, 1234).device, "NPU")

    def test_unknown_luid_matches_nothing_rather_than_guessing(self):
        unknown = "pid_1234_luid_0x00000000_0x00000000_phys_0_eng_5_engtype_Compute"
        self.assertIsNone(self.m.match(unknown, 1234))

    def test_rules_for_selects_one_device(self):
        self.assertEqual([r.engtype for r in self.m.rules_for("NPU")], ["Neural"])

class CollisionsAreFatal(unittest.TestCase):
    def test_one_luid_mapped_to_two_devices_is_rejected(self):
        import json, tempfile, pathlib
        payload = {"luids": [
            {"luid": "0x00010E9E", "device": "GPU"},
            {"luid": "0x00010E9E", "device": "NPU"},
        ]}
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "m.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_instance_map(p)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_instance_map -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.telemetry.instance_map'`.

- [ ] **Step 3: Implement `app/telemetry/instance_map.py`**

`parse_instance_name` extracts `pid`, `luid` (the second LUID word, normalized to `0x%08X` uppercase), and `engtype` from the documented `GPU Engine` instance format, and raises `ValueError` when no `luid_` field is present. `load_instance_map` reads `{"probed_on", "windows_build", "counter_source", "luids": [{"luid","device"}], "engines": [{"luid","device","engtype"}]}` and rejects any LUID appearing under two devices.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_instance_map -v`
Expected: `ok`. The collision test passing is what makes Review Focus #2 impossible to violate.

- [ ] **Step 5: Write the failing test `tests/test_pdh.py`**

```python
import unittest
from app.telemetry.pdh import PdhQuery, PdhUnavailableError, PdhCounterPath

class PdhDegradesHonestly(unittest.TestCase):
    def test_unknown_category_raises_rather_than_returning_zero(self):
        with self.assertRaises(PdhUnavailableError):
            PdhQuery("No Such Counter Category", {"x": "Utilization Percentage"})

    def test_missing_instance_reads_as_none_not_zero(self):
        q = PdhQuery("GPU Engine", {"u": "Utilization Percentage"})
        try:
            v = q.read(PdhCounterPath("GPU Engine", "pid_1_luid_bogus"))
            self.assertIsNone(v, "an unknown instance must read None, never 0")
        finally:
            q.close()
```

- [ ] **Step 6: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_pdh -v`
Expected: `ERROR` — `ModuleNotFoundError`.

- [ ] **Step 7: Implement `app/telemetry/pdh.py`**

`ctypes` bindings to `pdh.dll`: `PdhOpenQueryW`, `PdhAddEnglishCounterW`, `PdhGetFormattedCounterArrayW`, `PdhCloseQuery`. `__init__` raises `PdhUnavailableError` when the query handle is 0, wrapping the last error. `read` returns `None` when the instance is absent from the enumerated set, and a `float` in `[0, 100]` otherwise. `close` is idempotent.

**Never return `0.0` for a missing instance.** That single line is what separates "the NPU is idle" from "we cannot measure the NPU".

- [ ] **Step 8: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_pdh -v`
Expected: `ok`. If it errors with `PdhUnavailableError` on the *valid* category, the Windows install is not English — record that as an M0 blocker, do not work around it by substituting a number.

- [ ] **Step 9: Write `tools/probe_telemetry.py` and run it for real**

The probe: opens the `GPU Engine` query; enumerates every instance; resolves each LUID to a device by asking Windows once at operator time (WMI/CIM for the adapter and the AI Boost compute accelerator, matched by PCI device path, cross-checked so no LUID serves two devices); writes `config/telemetry_map.json`; prints a table of `luid -> device` and the engine types found.

**Record the outcome honestly.** If an `engtype_Neural` instance is found for the NPU LUID, the NPU card can show real utilization. If not, the probe must exit non-zero unless `--allow-missing-npu` is passed, and `npu_percent` stays `null` with the card reading "Utilization unavailable". Paste the probe's raw output into the M0 report either way — the negative result is equally valuable and is what makes the NPU card trustworthy.

- [ ] **Step 10: Commit**

```bash
git add app/telemetry/ tools/probe_telemetry.py config/telemetry_map.json tests/
git commit -m "M0: PDH telemetry provider with LUID-keyed instance map"
```

---

### Task 4: Preprocessing and letterbox coordinate recovery

Pure NumPy, no OpenVINO. Entirely unit-testable, and the source of the most plausible-looking wrong output in the whole product.

**Files:**
- Create: `app/inference/preprocess.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `app.inference.preprocess.Letterbox` — frozen dataclass: `scale: float`, `pad_x: int`, `pad_y: int`, `src_w: int`, `src_h: int`, `dst_w: int`, `dst_h: int`; methods `to_source(box: tuple[float,float,float,float]) -> tuple[float,float,float,float]` and `to_model(box: tuple[float,float,float,float]) -> tuple[float,float,float,float]`
  - `letterbox_params(src_w: int, src_h: int, dst_w: int, dst_h: int) -> Letterbox`
  - `preprocess_bgr(bgr: np.ndarray, dst_w: int, dst_h: int) -> tuple[np.ndarray, Letterbox]` — returns a `float32` NCHW RGB array scaled to `[0,1]` and the `Letterbox` used

- [ ] **Step 1: Write the failing test `tests/test_preprocess.py`**

```python
import unittest
import numpy as np
from app.inference.preprocess import letterbox_params, preprocess_bgr, Letterbox

class LetterboxGeometry(unittest.TestCase):
    def test_asymmetric_source_pads_asymmetrically(self):
        # 200x100 into a square must pad vertically only.
        lb = letterbox_params(200, 100, 640, 640)
        self.assertAlmostEqual(lb.scale, 3.2, places=5)
        self.assertEqual(lb.pad_x, 0)
        self.assertEqual(lb.pad_y, 160)

    def test_portrait_source_pads_horizontally(self):
        lb = letterbox_params(100, 200, 640, 640)
        self.assertEqual(lb.pad_x, 160)
        self.assertEqual(lb.pad_y, 0)

    def test_round_trip_recovers_source_coordinates(self):
        lb = letterbox_params(200, 100, 640, 640)
        src = (10.0, 20.0, 90.0, 80.0)
        for x1, y1, x2, y2 in [lb.to_model(src), lb.to_source(src)]:
            for a, b in zip((x1, y1, x2, y2), src):
                self.assertAlmostEqual(a, b, places=4)

    def test_corner_detection_survives_the_inverse_transform(self):
        # Review Focus #4: a centred object hides a missing pad/scale inverse.
        lb = letterbox_params(1920, 1080, 640, 640)
        corner_src = (1700.0, 980.0, 1900.0, 1070.0)
        back = lb.to_source(lb.to_model(corner_src))
        for a, b in zip(back, corner_src):
            self.assertAlmostEqual(a, b, places=3)

class PreprocessOutput(unittest.TestCase):
    def test_returns_float32_nchw_rgb_scaled_to_unit_range(self):
        bgr = np.full((100, 200, 3), 255, dtype=np.uint8)
        out, lb = preprocess_bgr(bgr, 640, 640)
        self.assertEqual(out.shape, (1, 3, 640, 640))
        self.assertEqual(out.dtype, np.float32)
        self.assertLessEqual(float(out.max()), 1.0)
        # BGR input of (0,0,255) is pure red, so channel 0 must be the max.
        blue = np.zeros((4, 4, 3), dtype=np.uint8); blue[..., 2] = 255
        out2, _ = preprocess_bgr(blue, 8, 8)
        self.assertGreater(float(out2[0, 0].max()), float(out2[0, 2].max()))

    def test_raises_on_a_three_dimensional_input(self):
        with self.assertRaises(ValueError):
            preprocess_bgr(np.zeros((10, 10), dtype=np.uint8), 640, 640)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_preprocess -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.inference.preprocess'`.

- [ ] **Step 3: Implement `app/inference/preprocess.py`**

`letterbox_params` computes `scale = min(dst_w/src_w, dst_h/src_h)` and centres the scaled image, so `pad_x`/`pad_y` are `int(round(...))` of the leftover. `to_model` maps source to model by `x*scale + pad_x`; `to_source` is its exact inverse. `preprocess_bgr` validates `ndim == 3 and shape[2] == 3`, converts BGR→RGB, resizes, pads with zeros, and returns `float32` NCHW in `[0,1]`.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_preprocess -v`
Expected: `ok`. The corner test is the one that matters; if it fails, the geometry is wrong in a way a demo would never reveal.

- [ ] **Step 5: Commit**

```bash
git add app/inference/preprocess.py tests/test_preprocess.py
git commit -m "M0: letterbox preprocessing with verified coordinate recovery"
```

---

### Task 5: Model I/O description and the decoder

Written against **observed** tensors, never an assumed layout. The specification warns explicitly that YOLO variants do not share one output layout.

**Files:**
- Create: `app/inference/decode.py`, `tools/dump_model_io.py`
- Create: `config/model_io/yolo11n-fp16.json`
- Test: `tests/test_decode.py`

**Interfaces:**
- Consumes: `app.inference.preprocess.Letterbox` (Task 4)
- Produces:
  - `app.inference.decode.ModelOutputSpec` — frozen dataclass: `names: tuple[str, ...]`, `shapes: dict[str, tuple[int, ...]]`, `layout: str` (one of `"yolo_v8_transposed"`, `"yolo_v8_passthrough"`, `"ssd"`, `"unknown"`), `num_classes: int`, `has_embedded_nms: bool`
  - `app.inference.decode.Detection` — frozen dataclass: `class_id: int`, `score: float`, `box: tuple[float, float, float, float]` in **source** pixels
  - `describe_outputs(model: ov.Model, num_classes: int) -> ModelOutputSpec`
  - `decode_detections(spec: ModelOutputSpec, outputs: Mapping[str, np.ndarray], letterbox: Letterbox, conf_threshold: float, iou_threshold: float) -> list[Detection]`
  - `nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]`

- [ ] **Step 1: Write the failing test `tests/test_decode.py`**

```python
import unittest
import numpy as np
from app.inference.decode import (ModelOutputSpec, decode_detections, nms)
from app.inference.preprocess import letterbox_params

def _one_box_tensor(layout, n=84, boxes=8400):
    """A single confident detection of class 0 at model coords (320,320,352,352)."""
    if layout == "yolo_v8_transposed":
        arr = np.zeros((1, n, boxes), dtype=np.float32); ax = 0
    else:
        arr = np.zeros((1, boxes, n), dtype=np.float32); ax = 1
    arr[0, :, 0] = np.arange(n, dtype=np.float32)
    arr[0, ax + 4, 0] = 320.0; arr[0, ax + 5, 0] = 320.0
    arr[0, ax + 6, 0] = 352.0; arr[0, ax + 7, 0] = 352.0
    arr[0, ax + 4 + 1, 0] = 0.9
    return {"output0": arr}

class LayoutIsNotAssumed(unittest.TestCase):
    def _decode(self, layout):
        spec = ModelOutputSpec(names=("output0",), shapes=(),
                               layout=layout, num_classes=80, has_embedded_nms=False)
        lb = letterbox_params(640, 640, 640, 640)
        return spec, decode_detections(spec, _one_box_tensor(layout),
                                       lb, 0.25, 0.45)

    def test_transposed_layout_yields_the_box(self):
        spec, dets = self._decode("yolo_v8_transposed")
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].class_id, 0)
        self.assertAlmostEqual(dets[0].box[0], 320.0, delta=1.0)

    def test_passthrough_layout_yields_the_same_box(self):
        spec, dets = self._decode("yolo_v8_passthrough")
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].class_id, 0)
        self.assertAlmostEqual(dets[0].box[0], 320.0, delta=1.0)

    def test_unknown_layout_yields_nothing_rather_than_guessing(self):
        spec = ModelOutputSpec(names=("output0",), shapes=(), layout="unknown",
                               num_classes=80, has_embedded_nms=False)
        lb = letterbox_params(640, 640, 640, 640)
        self.assertEqual(decode_detections(spec, _one_box_tensor("yolo_v8_passthrough"),
                                           lb, 0.25, 0.45), [])

    def test_boxes_are_returned_in_source_pixels(self):
        spec = ModelOutputSpec(names=("output0",), shapes=(),
                               layout="yolo_v8_transposed", num_classes=80,
                               has_embedded_nms=False)
        lb = letterbox_params(1280, 720, 640, 640)   # scale 0.5, pad_y 80
        dets = decode_detections(spec, _one_box_tensor("yolo_v8_transposed"),
                                 lb, 0.25, 0.45)
        self.assertEqual(len(dets), 1)
        src_x1, src_y1, _, _ = dets[0].box
        self.assertAlmostEqual(src_x1, 640.0, delta=2.0)   # 320 / 0.5
        self.assertAlmostEqual(src_y1, 480.0, delta=2.0)   # (320 - 80) / 0.5

class Nms(unittest.TestCase):
    def test_overlapping_boxes_are_suppressed(self):
        b = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [200, 200, 300, 300]], dtype=np.float32)
        s = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        self.assertEqual(nms(b, s, 0.45), [0, 2])

    def test_low_confidence_is_filtered_before_nms(self):
        spec = ModelOutputSpec(names=("output0",), shapes=(),
                               layout="yolo_v8_transposed", num_classes=80,
                               has_embedded_nms=False)
        t = _one_box_tensor("yolo_v8_transposed"); t["output0"][0, 5, 0] = 0.01
        lb = letterbox_params(640, 640, 640, 640)
        self.assertEqual(decode_detections(spec, t, lb, 0.25, 0.45), [])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_decode -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.inference.decode'`.

- [ ] **Step 3: Implement `app/inference/decode.py`**

`decode_detections` dispatches on `spec.layout`; `"unknown"` returns `[]` rather than guessing. Both YOLO layouts are normalised to a common `(boxes, 4+num_classes)` array before thresholding, NMS, and the `letterbox.to_source` inverse. `nms` is class-agnostic IoU suppression on the caller-filtered list.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_decode -v`
Expected: `ok`. Both layouts decoding identically is what makes Review Focus #3 testable.

- [ ] **Step 5: Run `tools/dump_model_io.py` and record the truth**

Write `dump_model_io.py` to read the pinned IR, print every input and output port name, shape, and element type, classify the layout, and write `config/model_io/yolo11n-fp16.json` with the observed values.

**Then read the output and record the result in the M0 report, including `has_embedded_nms`.** If the export *does* embed NMS, the CPU postprocessing path and the "detector export without embedded NMS" preference in §7.1 are both recorded as a deviation with its consequence, rather than quietly assumed away.

- [ ] **Step 6: Add the recorded-output test and run it**

Append to `tests/test_decode.py` a test that loads `config/model_io/yolo11n-fp16.json` and asserts `describe_outputs` on the real compiled model agrees with the recorded `layout`, `names`, and `num_classes`. If it disagrees, the recorded file is regenerated and the disagreement is reported — it is never resolved by editing the recorded file to match the code.

Run: `.venv\Scripts\python.exe -m unittest tests.test_decode -v`
Expected: `ok`, including the recorded-output test.

- [ ] **Step 7: Commit**

```bash
git add app/inference/decode.py tools/dump_model_io.py config/model_io/ tests/test_decode.py
git commit -m "M0: decoder driven by observed model I/O, not an assumed layout"
```

---

### Task 6: The annotated detection sanity fixture

Spec M0 requires "detection output passes a small annotated sanity fixture". The annotation must be a real human judgement about a real frame — never a value copied from the model's own output, which would make the test circular.

**Files:**
- Create: `tests/fixtures/retail_sanity/retail_000.jpg`, `tests/fixtures/retail_sanity/retail_sanity.json`
- Create: `tools/build_sanity_fixture.py`
- Test: `tests/test_sanity_fixture.py`

**Interfaces:**
- Consumes: `app.inference.preprocess` (Task 4), `app.inference.decode` (Task 5)
- Produces: `tests/fixtures/retail_sanity/retail_sanity.json` shaped `{"source_clip", "frame_index", "image", "expected": [{"class_id", "class_name", "min_score", "box_xyxy"}], "notes"}`, and `evaluate_fixture(spec: ModelOutputSpec, class_names: Sequence[str], conf_threshold: float, iou_threshold: float) -> FixtureOutcome` in `tools/build_sanity_fixture.py` with `FixtureOutcome` a frozen dataclass `passed: bool`, `matched: int`, `expected: int`, `details: tuple[str, ...]`

- [ ] **Step 1: Write the failing test `tests/test_sanity_fixture.py`**

```python
import json, unittest
from pathlib import Path
from tools.build_sanity_fixture import evaluate_fixture

FIX = Path("tests/fixtures/retail_sanity/retail_sanity.json")

class FixtureIsReal(unittest.TestCase):
    def test_fixture_exists_and_is_not_empty(self):
        self.assertTrue(FIX.exists(), "sanity fixture has not been built")
        payload = json.loads(FIX.read_text(encoding="utf-8"))
        self.assertTrue(payload["expected"], "fixture must annotate at least one object")
        for e in payload["expected"]:
            self.assertIn("class_name", e)
            self.assertEqual(len(e["box_xyxy"]), 4)

    def test_every_expected_box_lies_inside_the_image(self):
        import cv2
        payload = json.loads(FIX.read_text(encoding="utf-8"))
        img = cv2.imread(str(FIX.parent / payload["image"]))
        h, w = img.shape[:2]
        for e in payload["expected"]:
            x1, y1, x2, y2 = e["box_xyxy"]
            self.assertGreaterEqual(min(x1, y1), -w * 0.05)
            self.assertLessEqual(max(x2, y2), w * 1.05)
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_sanity_fixture -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'tools.build_sanity_fixture'`.

- [ ] **Step 3: Implement `tools/build_sanity_fixture.py` with `evaluate_fixture`**

`evaluate_fixture` runs the pinned model over the fixture image, decodes with the recorded spec, and reports pass only when every annotated expectation is matched by class id with `score >= min_score` and box IoU `>= 0.5` against the annotation. It returns a `FixtureOutcome`; it never mutates the annotation file.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_sanity_fixture -v`
Expected: `ERROR` for the missing fixture JSON — which is correct, because the fixture does not exist yet.

- [ ] **Step 5: Build the fixture by annotating a real frame**

Use `build_sanity_fixture.py` to extract a specific, fixed `frame_index` from the pinned retail clip at native resolution. Then **open the extracted image and look at it**, and write the annotation by hand: the class names actually visible, and boxes drawn by eye in source pixels. Record the frame index in the JSON so the fixture is reproducible.

Annotate two or three clearly present, unambiguous objects — for example a `person` and any product actually in view. Do not annotate an object you cannot see, and do not copy box coordinates from the detector's own output.

- [ ] **Step 6: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_sanity_fixture -v`
Expected: `ok`.

- [ ] **Step 7: Record the per-device result**

Run the fixture on CPU, GPU, and NPU, and paste all three outcomes into the M0 report. All three must match the same annotation; any device that disagrees is a finding to diagnose, not to paper over.

- [ ] **Step 8: Commit**

```bash
git add tests/fixtures/ tools/build_sanity_fixture.py tests/test_sanity_fixture.py
git commit -m "M0: hand-annotated detection sanity fixture, verified on all three devices"
```

---

### Task 7: The qualification runner

The deliverable that produces M0's evidence. Everything before it is a component; this is the measurement.

**Files:**
- Create: `tools/qualify_m0.py`
- Test: `tests/test_qualify_m0.py`

**Interfaces:**
- Consumes: `app.contracts.device_status` (Task 2), `app.inference.devices` (Task 2), `app.inference.preprocess` (Task 4), `app.inference.decode` (Task 5), `config/models.json` (Task 1)
- Produces:
  - `TargetResult` — frozen dataclass: `device_id: str`, `full_name: str`, `compile_seconds: float | None`, `warmup_seconds: float | None`, `requests: int`, `latency_ms_p50: float | None`, `latency_ms_p95: float | None`, `latency_ms_min: float | None`, `latency_ms_max: float | None`, `requests_per_second: float | None`, `execution_devices: tuple[str, ...]`, `rss_bytes: int | None`, `error: str | None`
  - `qualify_target(core: ov.Core, spec: TargetSpec, image: np.ndarray, num_requests: int) -> TargetResult`
  - `render_table(results: Sequence[TargetResult]) -> str`
  - CLI writing `logs/qualification/m0.json`

- [ ] **Step 1: Write the failing test `tests/test_qualify_m0.py`**

```python
import unittest
from tools.qualify_m0 import render_table, TargetResult

def _r(device, error=None, devices=("CPU",)):
    return TargetResult(device_id=device, full_name=device, compile_seconds=1.0,
                        warmup_seconds=0.5, requests=100, latency_ms_p50=10.0,
                        latency_ms_p95=12.0, latency_ms_min=9.0, latency_ms_max=20.0,
                        requests_per_second=100.0, execution_devices=devices,
                        rss_bytes=1, error=error)

class ReportIsHonest(unittest.TestCase):
    def test_missing_numbers_render_as_unavailable_never_zero(self):
        r = TargetResult(device_id="NPU", full_name="Intel(R) AI Boost",
                         compile_seconds=None, warmup_seconds=None, requests=0,
                         latency_ms_p50=None, latency_ms_p95=None, latency_ms_min=None,
                         latency_ms_max=None, requests_per_second=None,
                         execution_devices=(), rss_bytes=None,
                         error="compile failed: unsupported shape")
        table = render_table([r])
        self.assertIn("unavailable", table.lower())
        self.assertNotIn("0.00 ms", table)

    def test_error_is_shown_with_the_fallback_route(self):
        table = render_table([_r("NPU", error="compile failed, fell back to CPU",
                                 devices=("CPU",))])
        self.assertIn("compile failed", table)
        self.assertIn("CPU", table)

    def test_execution_devices_are_printed(self):
        self.assertIn("CPU", render_table([_r("NPU", devices=("NPU",))]))

    def test_a_failed_target_renders_nothing_that_looks_like_a_result(self):
        r = _r("NPU", error="unsupported shape on this driver", devices=())
        table = render_table([r])
        self.assertIn("unsupported shape", table)
        self.assertNotIn("100", table)
        self.assertNotIn("0.00", table)

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_qualify_m0 -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'tools.qualify_m0'`.

- [ ] **Step 3: Implement `tools/qualify_m0.py`**

`qualify_target` per device: read the model, time `compile_model`, create the infer request, run **one warmup inference**, then **read `EXECUTION_DEVICES` after that warmup**, then run `num_requests` (default 100) timed inferences with `time.perf_counter`, computing p50/p95/min/max and requests/s. Any exception is caught and returned as `error` with `execution_devices=()` — a failed target must never be rendered as a zero. `rss_bytes` comes from `psutil.Process().memory_info().rss`.

`render_table` prints one row per device with device, full name, compile s, warmup s, requests, p50, p95, rps, `EXECUTION_DEVICES`, RSS, and error. `None` renders as `unavailable`.

The runner also measures **startup time** — the wall time from process start to the first inference completing — and records **CPU-only fallback**: it runs the same 100 requests pinned to `CPU` and reports that result as the fallback baseline, so the M0 report can state the fallback works with evidence.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_qualify_m0 -v`
Expected: `ok`. The "never zero" test is the guard for the whole product's central claim.

- [ ] **Step 5: Run it on this machine for real**

Run: `.venv\Scripts\python.exe tools\qualify_m0.py --requests 100`
Expected: four rows — CPU, GPU, NPU, and the explicit CPU-only fallback baseline — each with 100 requests. **Paste the complete raw output into the M0 report.** If a device fails, the report records the failure and the diagnosed fallback; M0 does not pass on two devices.

- [ ] **Step 6: Commit**

```bash
git add tools/qualify_m0.py tests/test_qualify_m0.py
git commit -m "M0: qualification runner measuring compile, latency, placement and fallback"
```

---

### Task 8: The preflight gate

The single command that tells a presenter whether the machine is Ready, Limited, or Needs attention — spec §9.

**Files:**
- Create: `tools/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `app.inference.devices` (Task 2), `app.contracts.device_status` (Task 2), `tools.verify_sources` (Task 1)
- Produces:
  - `app/contracts/telemetry_availability.py` — frozen dataclass `TelemetryAvailability` with `provider: str`, `available: bool`, `metric_names: tuple[str, ...]`, `unavailable_reason: str | None`, `scope_note: str`; plus `to_json()`
  - `tools/preflight.py` — `Check` frozen dataclass `name: str`, `ok: bool`, `level: str` (`"PASS"`/`"WARN"`/`"FAIL"`), `detail: str`; `render_row(check: Check) -> str`; `render_summary(rows: Sequence[Check]) -> str`; `run_checks() -> list[Check]`; `main() -> int` printing `Summary: N PASS, 0 WARN, 0 FAIL — N total rows`

- [ ] **Step 1: Write the failing test `tests/test_preflight.py`**

```python
import unittest
from tools.preflight import Check, render_row, render_summary
from app.contracts.telemetry_availability import TelemetryAvailability

class PreflightIsReadable(unittest.TestCase):
    def test_every_row_has_a_level_and_a_plain_detail(self):
        rows = [Check("NPU model compile", True, "PASS", "100 requests, EXECUTION_DEVICES=(NPU,)")]
        out = render_row(rows[0])
        self.assertIn("PASS", out)
        self.assertIn("EXECUTION_DEVICES", out)

    def test_failed_check_is_fail_not_warn(self):
        self.assertEqual(Check("x", False, "PASS", "").level, "PASS")  # sanity on the dataclass
        self.assertNotEqual(Check("x", False, "WARN", "").level, "PASS")

    def test_summary_counts_match_the_rows(self):
        rows = [Check("a", True, "PASS", ""), Check("b", True, "PASS", ""),
                Check("c", False, "WARN", "")]
        self.assertIn("2 PASS", render_summary(rows))
        self.assertIn("1 WARN", render_summary(rows))
        self.assertIn("0 FAIL", render_summary(rows))

    def test_unavailable_telemetry_is_a_named_reason_not_a_zero(self):
        a = TelemetryAvailability(provider="pdh_gpu_engine", available=False,
                                  metric_names=(), scope_note="",
                                  unavailable_reason="no engtype_Neural instance for the AI Boost LUID")
        self.assertFalse(a.available)
        self.assertIn("Neural", a.unavailable_reason)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_preflight -v`
Expected: `ERROR` — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `app/contracts/telemetry_availability.py` and `tools/preflight.py`**

`run_checks()` covers: asset integrity via `verify_sources`; RAM and disk budget; each device's availability; a model compile per device; the sanity fixture per device; the CPU-only fallback; each telemetry provider; and the recorded `config/telemetry_map.json` freshness. A missing NPU counter is `WARN` with a plain reason, not `FAIL`, because §9 says a CPU-only package is still useful. Overall status is Ready / Limited / Needs attention.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_preflight -v`
Expected: `ok`.

- [ ] **Step 5: Run preflight on this machine**

Run: `.venv\Scripts\python.exe tools\preflight.py`
Expected: a row per check with no `FAIL`. Paste the whole table into the M0 report and into `bench_report.md`.

- [ ] **Step 6: Commit**

```bash
git add tools/preflight.py app/contracts/telemetry_availability.py tests/test_preflight.py
git commit -m "M0: preflight gate reporting Ready / Limited / Needs attention"
```

---

### Task 9: The M0 qualification report and content gate

The milestone's actual deliverable. Every claim in it must trace to pasted output from a task above.

**Files:**
- Create: `docs/qualification/M0-qualification.md`, `docs/qualification/M0-content-gate.md`, `docs/qualification/qualification-matrix.csv`, `bench_report.md`

**Interfaces:**
- Consumes: `logs/qualification/m0.json` (Task 7), preflight output (Task 8), probe output (Task 3), model I/O record (Task 5), asset inventory (Task 1)
- Produces: no code; the milestone gate

- [ ] **Step 1: Write `docs/qualification/qualification-matrix.csv`**

Header: `scenario,model_artifact,device,machine_sku,driver_version,runtime_version,status,evidence`.
Fill one row per cell actually tested in M0 — the retail scenario on the pinned detector across CPU/GPU/NPU plus the CPU-only fallback. Every **untested** cell is written explicitly as `not tested`, because spec §11 requires the matrix to exist and a successful CPU run is not evidence of NPU readiness.

- [ ] **Step 2: Write `docs/qualification/M0-qualification.md`**

Sections, in order: exact SKU / RAM / Windows build / driver versions / OpenVINO and Python versions / power scheme / AC status unavailable / monitor configuration; device discovery with the pasted `available_devices` and `FULL_DEVICE_NAME` output; model pin with the recorded I/O, including `has_embedded_nms` and any deviation from the "no embedded NMS" preference; per-device compile and latency with the pasted `qualify_m0.py` table; `EXECUTION_DEVICES` per device; the sanity fixture result per device; CPU-only fallback result; startup time and memory; **telemetry providers — what is available, what is not, and why**; the preflight table; and an explicit list of unmet gates and open blockers.

Rules for this document: a number appears only if a task above measured it; anything not run is written "not run"; the NPU utilization row states either the validated provider or `unavailable` with the reason.

- [ ] **Step 3: Write `docs/qualification/M0-content-gate.md`**

Answer spec M0's content gate from evidence: the retail clip's licence, owner, source URL, hash and distribution status; the detector weights' licence and provenance; **whether a clear shelf-to-basket clip with a visible transfer and no prolonged occlusion exists in the material actually available** — answered by looking at the footage, not by assuming; and the remaining footage requirements for the other three packs, with the measured clip count and duration against the spec's three-per-pack, 45–90 s requirement.

If no such clip exists, state it and commit to the documented reduced-scope fallback — "items entering checkout zone" as an explicitly labelled intermediate milestone — with the requested shelf-to-basket scenario left incomplete rather than claimed.

- [ ] **Step 4: Append the evidence to `bench_report.md`**

Add an M0 section with the pasted preflight table and the pasted qualification table.

- [ ] **Step 5: Run the full gate set and paste the results**

```powershell
.venv\Scripts\python.exe tools\verify_sources.py
.venv\Scripts\python.exe tools\preflight.py
.venv\Scripts\python.exe -m unittest discover -s tests
```

Expected: source verification all PASS; preflight with no FAIL; unittest all ok. Paste each into the report. If `app.main --selftest` is not yet implemented, write "not run — `app.main` arrives in M1" rather than omitting it.

- [ ] **Step 6: Commit**

```bash
git add docs/qualification/ bench_report.md
git commit -m "M0: qualification report, telemetry provenance and content gate"
```

---

## M0 exit criteria

M0 is complete when, and only when, all of the following are true and evidenced in
`docs/qualification/M0-qualification.md`:

1. CPU, GPU and NPU each compile the pinned detector and complete **at least 100 requests**, with `EXECUTION_DEVICES` confirming placement.
2. The hand-annotated sanity fixture passes on all three devices.
3. Latency, memory, and startup time are reported from real measurements.
4. Available telemetry providers are documented, including any provider that is unavailable and why.
5. CPU-only fallback is demonstrated with its own measurement.
6. `preflight.py` has no `FAIL`, and the full unit suite is green.
7. Every unmet gate and blocker is written down, including the M5 clip shortfall, the unreadable AC status, the monitor mismatch, the missing 1920×1080 canvas on this panel, and code signing.
8. The model and dependency pins are exact and recorded.

**Not claimed by M0, and stated as such in the report:** UI, routing switches, comparison runs, and the other three vertical packs. Those are M1 onward.

## Next milestone

M1 — one truthful retail stream: real detection overlays, an explicit CPU/GPU/NPU selector, tracking, ROI event logic, basic metrics, the "Recorded footage • live inference" label, and the model explanation. Its plan is written after M0 closes, so it builds on the `DeviceStatus`, `TargetSpec` and decoder interfaces this milestone actually produced.
