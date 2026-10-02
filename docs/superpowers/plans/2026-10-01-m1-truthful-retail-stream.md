# M1 — One Truthful Retail Stream — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A running Windows application that plays one retail clip, runs real OpenVINO inference on a device the presenter chooses, draws overlays that never drift from the frame they belong to, turns detections into retail business events, and shows measured metrics — with the mandatory "Recorded footage • live inference" label and a visible error/fallback state when the chosen device fails.

**Architecture:** A QML/Qt Quick UI in the main process, with inference in a spawned worker process (Approach C from the approved design), frames moved over `multiprocessing.shared_memory`, control/status over a local socket. The engine layer is pure and unit-tested: a tracker, an event state machine, zone geometry, and a frame-identity contract that the overlay depends on.

**Tech Stack:** Python 3.12.10, `openvino==2026.4.0`, `PySide6==6.11.2` (Qt Quick/QML), `numpy==2.5.3`, `opencv-python==5.0.0.93`, `psutil==7.2.2`. Tests use stdlib `unittest` — `pytest` is deliberately not added.

**Spec:** `docs/superpowers/specs/2026-10-01-intel-edge-ai-roadshow-design.md` (approved design) and `Intel-Edge-AI-Roadshow-Specification.md` §3, §4, §5, §6, §7, §10 M1. Executors read both.

## Global Constraints

Copied from the approved design, the specification, and `AGENTS.md`. Every task implicitly includes this section.

- Target machine: Intel Core Ultra 9 288V (Lunar Lake, Core Ultra Series 2), Windows 11 Pro build 26200. Show the discovered `FULL_DEVICE_NAME`; never a hardcoded family string; never present a Lunar Lake measurement as Panther Lake or Arrow Lake.
- Pinned exactly, no ranges: `openvino==2026.4.0`, `PySide6==6.11.2`, `numpy==2.5.3`, `opencv-python==5.0.0.93`, `psutil==7.2.2`, `PyYAML==6.0.3`.
- **Static shapes only.** Batch 1, 640×640, the pinned `yolo11n-fp16` IR.
- Explicit `compile_model(device)` targets only. No `AUTO`, no `HETERO`, no `MULTI`.
- **`EXECUTION_DEVICES`, re-read AFTER the first inference, is the only proof of placement.** It is inconsistent by device: NPU returns the bare string `'NPU'`, CPU `['CPU']`, GPU `['GPU.0']`. Always use `normalise_execution_devices` or `is_device`; **never read the property directly**.
- **Unavailable is `null`, never `0`.** Anything older than 2 s is `stale`. A missing value renders as `unavailable`, never as a number.
- No simulated metrics in release mode. A development banner is mandatory for any simulated value, and `app.main` refuses to start in release mode with a simulated provider registered.
- Never fabricate a number, a detection, a device name, or a licence. A CPU run is never evidence of NPU readiness.
- Layout tokens live in `app/theme.py`. No metric values or magic numbers in QML or UI code.
- Telemetry sampling runs on a background thread and reaches the UI only through Qt signals. Never touch widgets from a non-UI thread.
- Catch per-stream exceptions, mark that stream failed, keep the rest running, log a traceback, restart a failed worker once automatically.
- Standard user at runtime. No elevation. No network at runtime. No runtime downloads.
- Tests: stdlib `unittest`, run from the repo root. `pytest` is not available.

**Carried over from M0 — these are settled facts, not open questions.** Re-derive only where a step says to.

- The detector's box encoding is **`cx,cy,w,h`**, not `xyxy` (15/15 real anchors are inverted under `xyxy`). The output port is **unnamed**: tensors are filed under the port-index **string** key `'0'`; `infer` accepts an int index or the **compiled** model's port object and rejects the **IR** model's port object; `get_tensor(0)` raises — use the compiled port object or `get_output_tensor(0)`. `has_embedded_nms` is `False`, **derived** from the IR graph.
- The letterbox pad band is **114**, a measured decision: 43.75% of the model canvas. `preprocess_bgr(..., pad_value=114)`.
- Per-device p50 on this machine: CPU 24.07 ms, GPU 3.76 ms, NPU 5.27 ms. Treat as a reference, not a target to hit.
- `DeviceRegistry.describe()` costs **~3.3–3.8 s cold** (one memoised PowerShell PnP call). **Startup-only — it must never touch a per-frame or per-sample path.**
- GPU/NPU **utilisation is unavailable** in the product's runtime path (in-process PDH value reads fault with `0xC0000005`), and a machine-wide peak is **not attributable to this process**. The card says "Utilization unavailable" and shows inference activity instead. Do not invent a utilisation number.
- App-layer model facts live in `app/inference/pinned.py` (pure `json` + `pathlib`). Import from there; do not re-read `config/models.json` in `app/`.
- `tools/build_sanity_fixture.py` holds the only proven hand annotation: frame 0 of `assets/media/retail-checkout.mp4`, person `[645,0,1268,658]`, apple `[0,878,150,1062]`, `conf=0.25`, `iou=0.45`, `max_detections=6`.
- **The retail clip has no shelf-to-basket event.** 72 frames were reviewed: a shelf appears in `store_aisle_alt`, a basket in `retail_checkout`, and they never co-occur. `retail_checkout` is a self-checkout/till with the basket already full at frame 0. M0 committed to the reduced-scope intermediate milestone **"items entering the checkout zone"** as the labelled deliverable, with shelf-to-basket left incomplete. Task 9 owns that decision; do not quietly widen it.

## Review Focus

Failure modes the specification implies that no naive implementation would test. Each is pinned by a test in the owning task.

1. **A result arrives after a newer frame was already displayed, and the box paints onto the wrong frame.** Expect: the overlay is bound to a `frame_id`; a late or obsolete result is discarded, never drawn.
2. **A track is lost mid-transfer (occlusion) and the event machine completes the transfer anyway.** Expect: a track identity change cancels the candidate transfer; the event is marked uncertain or not counted.
3. **The same transfer fires twice — once per loop, or twice for one movement** because the dedup key is missing or too weak. Expect: a stable dedup key, one event per transfer per loop, and an explicit reset at loop restart.
4. **The metric widget shows `0` when a counter is unavailable**, which reads as "the engine is idle" rather than "we cannot measure". Expect: `null` renders as `unavailable`.
5. **The overlay shows a different device from the one actually executing** because the card renders the *requested* device. Expect: the displayed device is the confirmed `EXECUTION_DEVICES` value, and a mismatch is visibly a fallback.
6. **The worker wedges and the UI freezes** because inference ran on the UI thread. Expect: input acknowledgement stays under 200 ms with CPU-only work saturating the machine.

---

## File Structure

| File | Responsibility |
|---|---|
| `app/contracts/frame.py` | `FramePacket` — the frame-identity contract every layer agrees on |
| `app/contracts/metric_sample.py` | `MetricSample`, `Validity` — name/value/unit/scope/provider/window/staleness |
| `app/contracts/route_plan.py` | `RoutePlan` — generation, requested device, confirmed actual, transition state |
| `app/contracts/scenario_manifest.py` | `ScenarioManifest` — zones, event rules, class allowlist, sources |
| `app/engine/tracker.py` | IoU tracker with explicit reset and stable track ids |
| `app/engine/zones.py` | normalised ROI geometry, hit tests, dwell and line-crossing |
| `app/engine/events.py` | the retail event state machine (pure) |
| `app/engine/source.py` | clip reader producing `FramePacket`s with monotonic timestamps and loop ids |
| `app/engine/queue.py` | bounded frame queue, newest-frame policy, explicit drop counters |
| `app/engine/worker.py` | the spawned inference worker process entry point |
| `app/supervisor/runner.py` | owns the worker, routes, and restarts it once on failure |
| `app/telemetry/collector.py` | background 4 Hz sampling → Qt signals, per spec §8 |
| `app/ui/Main.qml`, `app/ui/*.qml` | the booth screen |
| `app/ui/bridge.py` | the QML-facing object exposing state as properties/signals |
| `app/main.py` | CLI, startup gate, `--selftest`, release-mode simulation refusal |
| `app/theme.py` | layout tokens only |
| `tools/run_retail_stream.py` | headless run for validation and CI-free soak |

---

### Task 1: `FramePacket` — the frame-identity contract

Everything downstream depends on this. If a box can be associated with the wrong frame, the product's central claim fails.

**Files:**
- Create: `app/contracts/frame.py`
- Test: `tests/test_frame_contract.py`

**Interfaces:**
- Consumes: nothing
- Produces: `FramePacket` — frozen dataclass: `session_id: str`, `scenario_id: str`, `stream_id: str`, `frame_id: int`, `source_pts: float`, `source_loop_id: int`, `captured_at: float` (monotonic), `decoded_at: float` (monotonic), `width: int`, `height: int`, `image: np.ndarray`; methods `age_at(now: float) -> float`, `is_from(loop_id: int) -> bool`, `to_json() -> dict` (metadata only, never the pixels)

- [ ] **Step 1: Write the failing test**

```python
import unittest
import numpy as np
from app.contracts.frame import FramePacket

def _p(**kw):
    base = dict(session_id="s1", scenario_id="retail", stream_id="A", frame_id=7,
                source_pts=0.35, source_loop_id=0, captured_at=10.0, decoded_at=10.002,
                width=64, height=48, image=np.zeros((48, 64, 3), np.uint8))
    base.update(kw)
    return FramePacket(**base)

class Identity(unittest.TestCase):
    def test_frame_id_is_unique_per_source_loop(self):
        self.assertEqual(_p().frame_id, 7)

    def test_two_frames_from_different_loops_are_distinguishable(self):
        a, b = _p(frame_id=7, source_loop_id=0), _p(frame_id=7, source_loop_id=1)
        self.assertNotEqual(a.source_loop_id, b.source_loop_id)
        self.assertFalse(a.is_from(1))

    def test_age_is_measured_from_capture_not_decode(self):
        self.assertAlmostEqual(_p(captured_at=10.0, decoded_at=10.9).age_at(12.0), 2.0, places=6)

    def test_age_never_goes_negative(self):
        self.assertGreaterEqual(_p(captured_at=10.0).age_at(9.0), 0.0)

    def test_to_json_omits_the_pixels(self):
        payload = _p().to_json()
        self.assertNotIn("image", payload)
        self.assertIn("frame_id", payload)

    def test_packet_is_immutable(self):
        with self.assertRaises(Exception):
            _p().frame_id = 9
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_frame_contract -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.contracts.frame'`.

- [ ] **Step 3: Implement `app/contracts/frame.py`**

`@dataclass(frozen=True)`. `age_at(now)` returns `max(0.0, now - captured_at)` — frame age is measured from **availability at application ingress**, i.e. capture, not decode, per spec §7.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_frame_contract -v`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add app/contracts/frame.py tests/test_frame_contract.py
git commit -m "M1: FramePacket identity contract"
```

---

### Task 2: Normalised zone geometry

**Files:**
- Create: `app/engine/zones.py`, `app/contracts/scenario_manifest.py`
- Test: `tests/test_zones.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `app.contracts.scenario_manifest.ScenarioManifest` — frozen dataclass: `id: str`, `version: int`, `title: str`, `business_line: str`, `class_allowlist: tuple[str, ...]`, `zones: tuple[Zone, ...]`, `event_rules: dict`, `source_ids: tuple[str, ...]`, `conf_threshold: float`, `iou_threshold: float`
  - `app.engine.zones.Zone` — frozen dataclass: `name: str`, `kind: str`, `roi: tuple[float, float, float, float]` (normalised x1,y1,x2,y2 in [0,1]); methods `contains(box: tuple[float, float, float, float]) -> bool`, `centroid_of(box) -> tuple[float, float]`, `to_pixels(width: int, height: int) -> tuple[int, int, int, int]`
  - `dwell_seconds_in_zone(zone: Zone, box: tuple[float, float, float, float], box_at: float, now: float) -> float`
  - `crossed_line(zone: Zone, previous: tuple[float, float], current: tuple[float, float], direction: int) -> bool`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from app.engine.zones import Zone, dwell_seconds_in_zone, crossed_line

SHELF = Zone("shelf", "shelf", (0.02, 0.22, 0.34, 0.55))
BASKET = Zone("basket", "shelf", (0.60, 0.30, 0.98, 0.70))
BELT = Zone(name="belt", kind="line", roi=(0.34, 0.18, 0.74, 0.58))

class Containment(unittest.TestCase):
    def test_zone_accepts_a_fully_contained_box(self):
        self.assertTrue(SHELF.contains((0.10, 0.30, 0.20, 0.40)))
        self.assertFalse(SHELF.contains((0.10, 0.30, 0.90, 0.40)))

    def test_a_box_touching_the_edge_is_inside(self):
        self.assertTrue(SHELF.contains((0.02, 0.22, 0.34, 0.55)))

    def test_pixels_round_to_the_source_frame(self):
        self.assertEqual(SHELF.to_pixels(1920, 1080), (38, 237, 652, 594))

    def test_roi_must_be_normalised_and_ordered(self):
        with self.assertRaises(ValueError):
            Zone("bad", "shelf", (0.9, 0.1, 0.2, 0.5))
        with self.assertRaises(ValueError):
            Zone("bad", "shelf", (0.1, 0.1, 1.2, 0.5))

class Dwell(unittest.TestCase):
    def test_dwell_accumulates_only_while_inside(self):
        # enters at t=1, still inside at t=3 -> 2 seconds
        self.assertAlmostEqual(dwell_seconds_in_zone(SHELF, (0.1, 0.3, 0.2, 0.4), 1.0, 3.0), 2.0)

    def test_dwell_never_goes_negative(self):
        self.assertEqual(dwell_seconds_in_zone(SHELF, (0.1, 0.3, 0.2, 0.4), 3.0, 1.0), 0.0)

class LineCrossing(unittest.TestCase):
    def test_a_left_to_right_crossing_is_detected(self):
        self.assertTrue(crossed_line(BELT, (0.30, 0.40), (0.80, 0.42), direction=1))

    def test_the_reverse_crossing_is_not_the_same_event(self):
        self.assertFalse(crossed_line(BELT, (0.30, 0.40), (0.80, 0.42), direction=-1))

    def test_a_movement_that_never_crosses_is_not_a_crossing(self):
        self.assertFalse(crossed_line(BELT, (0.40, 0.40), (0.50, 0.42), direction=1))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_zones -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.engine.zones'`.

- [ ] **Step 3: Implement `app/engine/zones.py` and `app/contracts/scenario_manifest.py`**

`Zone.__post_init__` validates that all four values are in `[0, 1]` and that `x1 < x2`, `y1 < y2` — an out-of-order or out-of-range ROI raises rather than silently misbehaving. `dwell_seconds_in_zone` returns `max(0.0, now - box_at)` only when the box is currently inside. `crossed_line` treats the ROI as a vertical band and compares the sign of the horizontal movement, so a return movement is a distinct event per spec §4.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_zones -v`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add app/engine/zones.py app/contracts/scenario_manifest.py tests/test_zones.py
git commit -m "M1: normalised zone geometry and the ScenarioManifest contract"
```

---

### Task 3: IoU tracker with explicit reset

**Files:**
- Create: `app/engine/tracker.py`
- Test: `tests/test_tracker.py`

**Interfaces:**
- Consumes: `app.contracts.frame.FramePacket` (Task 1)
- Produces: `Track` — frozen dataclass: `track_id: int`, `class_id: int`, `box: tuple[float, float, float, float]`, `first_seen: float`, `last_seen: float`, `hits: int`, `misses: int`, `confirmed: bool`; `Tracker(iou_threshold: float = 0.3, max_misses: int = 2, min_hits: int = 2)` with `update(detections: Sequence[Detection], now: float) -> tuple[Track, ...]`, `reset()`, `active_track_ids() -> tuple[int, ...]`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from app.engine.tracker import Tracker

def _d(x1, y1, x2, y2, cls=0):
    return type("D", (), {"class_id": cls, "score": 0.9, "box": (x1, y1, x2, y2)})()

class Tracking(unittest.TestCase):
    def test_a_stationary_object_keeps_one_id(self):
        t = Tracker()
        for f in range(4):
            t.update([_d(100, 100, 200, 200)], now=float(f) / 30)
        self.assertEqual(len(t.active_track_ids()), 1)

    def test_two_separate_objects_get_two_ids(self):
        t = Tracker()
        t.update([_d(0, 0, 50, 50), _d(400, 400, 450, 450)], now=0.0)
        t.update([_d(0, 0, 50, 50), _d(400, 400, 450, 450)], now=0.1)
        self.assertEqual(len(t.active_track_ids()), 2)

    def test_ids_survive_a_small_movement(self):
        t = Tracker()
        a = t.update([_d(100, 100, 200, 200)], now=0.0)[0]
        b = t.update([_d(105, 103, 205, 203)], now=0.1)[0]
        self.assertEqual(a.track_id, b.track_id)

    def test_a_track_is_lost_after_max_misses(self):
        t = Tracker(max_misses=2)
        t.update([_d(0, 0, 50, 50)], now=0.0)
        t.update([], now=0.1)
        t.update([], now=0.2)
        self.assertEqual(t.active_track_ids(), ())

    def test_reset_clears_every_id_so_a_loop_cannot_reuse_one(self):
        t = Tracker()
        t.update([_d(0, 0, 50, 50)], now=0.0)
        before = t.active_track_ids()
        t.reset()
        after = t.update([_d(0, 0, 50, 50)], now=1.0)
        self.assertNotEqual(after[0].track_id, before[0])

    def test_a_class_change_does_not_silently_reuse_the_id(self):
        t = Tracker()
        a = t.update([_d(0, 0, 50, 50, cls=0)], now=0.0)[0]
        b = t.update([_d(0, 0, 50, 50, cls=47)], now=0.1)[0]
        self.assertNotEqual(a.track_id, b.track_id)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_tracker -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.engine.tracker'`.

- [ ] **Step 3: Implement `app/engine/tracker.py`**

Greedy IoU matching, highest IoU first. A detection only matches a track of the **same `class_id`** — the last test pins this, because reusing an id across classes is how a person silently becomes a product. `reset()` clears the id counter as well as the tracks, so a new loop cannot inherit an old id; spec §9 requires an explicit reset after each source loop.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_tracker -v`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add app/engine/tracker.py tests/test_tracker.py
git commit -m "M1: IoU tracker with explicit reset and class-stable ids"
```

---

### Task 4: The retail event state machine

The product's business claim lives here. Spec §4 requires `visible_on_shelf → exited_shelf → entered_basket → candidate_transfer`, with the same stable track, minimum dwell, a bounded transition interval, and a dedup key; reverse movement produces a separate return event; **occlusion or lost track identity means uncertain or not counted, never a completed transfer**; and gaps are never filled with scripted detections.

**Files:**
- Create: `app/engine/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: `app.engine.zones.Zone` (Task 2), `app.engine.tracker.Track` (Task 3)
- Produces:
  - `EventRule` — frozen dataclass: `name: str`, `min_dwell_seconds: float`, `max_transition_seconds: float`, `classes: tuple[str, ...]`
  - `BusinessEvent` — frozen dataclass: `kind: str`, `class_name: str`, `track_id: int`, `source_loop_id: int`, `at_frame_id: int`, `at_time: float`, `dedup_key: str`, `uncertain: bool`
  - `RetailEventMachine(manifest: ScenarioManifest)` with `observe(track: Track, frame: FramePacket) -> tuple[BusinessEvent, ...]`, `on_track_lost(track_id: int, now: float)`, `reset(loop_id: int)`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from app.contracts.frame import FramePacket
from app.contracts.scenario_manifest import ScenarioManifest
from app.engine.events import RetailEventMachine, EventRule
from app.engine.tracker import Track
import numpy as np

SHELF = (0.02, 0.22, 0.34, 0.55)
BASKET = (0.60, 0.30, 0.98, 0.70)

def _manifest():
    return ScenarioManifest(
        id="retail", version=1, title="Retail / checkout zone",
        business_line="Items entering the checkout zone",
        class_allowlist=("person", "apple", "bottle", "cup"),
        zones=(), event_rules={
            "min_dwell_seconds": 0.5, "max_transition_seconds": 12.0,
            "shelf": SHELF, "basket": BASKET, "tracked_classes": ("person", "apple"),
        },
        source_ids=("retail_checkout",), conf_threshold=0.25, iou_threshold=0.45)

def _frame(fid, t, loop=0):
    return FramePacket(session_id="s", scenario_id="retail", stream_id="A",
                       frame_id=fid, source_pts=t, source_loop_id=loop,
                       captured_at=t, decoded_at=t, width=1920, height=1080,
                       image=np.zeros((8, 8, 3), np.uint8))

def _track(tid, box, t, loop=0, fid=0):
    return Track(track_id=tid, class_id=47, box=box, first_seen=t, last_seen=t,
                 hits=5, misses=0, confirmed=True)

class Transfer(unittest.TestCase):
    def _walk(self, machine, tid=1, loop=0, fid=0, drop_track_before=None):
        """Shelf -> out of shelf -> basket, emitting events as we go."""
        events = []
        steps = [
            (0.0, (50, 240, 300, 500)),    # inside shelf
            (0.6, (50, 240, 300, 500)),    # dwell satisfied
            (1.0, (700, 350, 900, 600)),   # in basket
            (1.4, (700, 350, 900, 600)),
        ]
        for t, box in steps:
            if drop_track_before is not None and t >= drop_track_before:
                machine.on_track_lost(tid, t)
            events += machine.observe(_track(tid, box, t, loop, fid), _frame(fid, t, loop))
            fid += 1
        return events

    def test_a_sustained_shelf_to_basket_move_is_a_candidate_transfer(self):
        m = RetailEventMachine(_manifest())
        kinds = [e.kind for e in self._walk(m)]
        self.assertIn("candidate_transfer", kinds)

    def test_minimum_dwell_is_required(self):
        m = RetailEventMachine(_manifest())
        m.observe(_track(1, (50, 240, 300, 500), 0.0), _frame(0, 0.0))
        m.observe(_track(1, (700, 350, 900, 600), 0.2), _frame(1, 0.2))
        self.assertEqual([e.kind for e in m.observe(_track(1, (700, 350, 900, 600), 0.4), _frame(2, 0.4))], [])

    def test_the_same_transfer_fires_once_not_once_per_frame(self):
        m = RetailEventMachine(_manifest())
        events = self._walk(m)
        self.assertEqual(len([e for e in events if e.kind == "candidate_transfer"]), 1)

    def test_a_transition_slower_than_the_maximum_does_not_count(self):
        m = RetailEventMachine(_manifest())
        m.observe(_track(1, (50, 240, 300, 500), 0.0), _frame(0, 0.0))
        m.observe(_track(1, (50, 240, 300, 500), 0.6), _frame(1, 0.6))
        late = m.observe(_track(1, (700, 350, 900, 600), 99.0), _frame(2, 99.0))
        self.assertEqual([e for e in late if e.kind == "candidate_transfer"], [])

    def test_losing_track_identity_cancels_the_transfer(self):
        # Review Focus #2: occlusion must not complete a transfer.
        m = RetailEventMachine(_manifest())
        m.observe(_track(1, (50, 240, 300, 500), 0.0), _frame(0, 0.0))
        m.observe(_track(1, (50, 240, 300, 500), 0.6), _frame(1, 0.6))
        m.on_track_lost(1, 0.8)
        m.observe(_track(1, (700, 350, 900, 600), 1.4), _frame(2, 1.4))
        self.assertEqual(m.events, [])

    def test_a_return_movement_is_a_separate_event_kind(self):
        m = RetailEventMachine(_manifest())
        self._walk(m)
        more = []
        for t, box in [(2.0, (700, 350, 900, 600)), (2.6, (700, 350, 900, 600)),
                       (3.0, (50, 240, 300, 500)), (3.4, (50, 240, 300, 500))]:
            more += m.observe(_track(1, box, t, 0, 20), _frame(20, t))
        self.assertIn("candidate_return", [e.kind for e in more])

    def test_a_new_loop_resets_counters_and_clears_events(self):
        # Review Focus #3: a rewind must not re-fire or leak old state.
        m = RetailEventMachine(_manifest())
        self._walk(m, loop=0)
        m.reset(loop_id=1)
        self.assertEqual(m.events, [])
        events = self._walk(m, loop=1, fid=1000)
        self.assertEqual(len([e for e in events if e.kind == "candidate_transfer"]), 1)
        self.assertEqual({e.source_loop_id for e in events}, {1})

    def test_events_carry_a_dedup_key_and_their_frame(self):
        m = RetailEventMachine(_manifest())
        ev = [e for e in self._walk(m) if e.kind == "candidate_transfer"][0]
        self.assertTrue(ev.dedup_key)
        self.assertGreaterEqual(ev.at_frame_id, 0)
        self.assertFalse(ev.uncertain)

    def test_an_unconfirmed_track_is_never_counted(self):
        m = RetailEventMachine(_manifest())
        weak = Track(track_id=9, class_id=47, box=(50, 240, 300, 500), first_seen=0.0,
                     last_seen=0.0, hits=1, misses=0, confirmed=False)
        m.observe(weak, _frame(0, 0.0))
        self.assertEqual(m.events, [])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_events -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.engine.events'`.

- [ ] **Step 3: Implement `app/engine/events.py`**

Per-track state: `first_seen_in_shelf`, `dwell_start`, `exited_shelf_at`, `last_seen`. Emit `candidate_transfer` only when the same `track_id` was observed in the shelf, dwell ≥ `min_dwell_seconds`, the gap between `exited_shelf_at` and basket entry ≤ `max_transition_seconds`, and the track was never lost in between. `dedup_key` is `f"{source_loop_id}:{track_id}:{class_name}:transfer"`. `on_track_lost` discards that track's pending state. `reset(loop_id)` clears every track's state, the event log, and the emitted set. The machine is **pure**: no clock, no OpenVINO, no I/O.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_events -v`
Expected: `ok`. If `test_a_return_movement_is_a_separate_event_kind` fails, the state machine is completing a second transfer instead of recording a return.

- [ ] **Step 5: Commit**

```bash
git add app/engine/events.py tests/test_events.py
git commit -m "M1: retail event state machine with dwell, dedup, occlusion and loop reset"
```

---

### Task 5: Clip source and the bounded frame queue

**Files:**
- Create: `app/engine/source.py`, `app/engine/queue.py`
- Test: `tests/test_source.py`, `tests/test_queue.py`

**Interfaces:**
- Consumes: `app.contracts.frame.FramePacket` (Task 1)
- Produces:
  - `ClipSource(path: Path, stream_id: str, session_id: str, scenario_id: str, target_fps: float = 15.0, source_fps: float | None = None)` with `read() -> FramePacket | None` (blocking, pacing to `target_fps` analysis rate from the source's own fps), `advance()`, `restart() -> int`, `position -> int`, `finished -> bool`
  - `BoundedFrameQueue(maxsize: int = 2)` with `put(frame: FramePacket) -> bool`, `get(timeout: float = 0.0) -> FramePacket | None`, `drops -> int`, `depth -> int`, `close()`

- [ ] **Step 1: Write the failing test**

```python
import unittest
import numpy as np
from app.contracts.frame import FramePacket
from app.engine.queue import BoundedFrameQueue

def _f(fid, t):
    return FramePacket(session_id="s", scenario_id="retail", stream_id="A", frame_id=fid,
                       source_pts=t, source_loop_id=0, captured_at=t, decoded_at=t,
                       width=8, height=8, image=np.zeros((8, 8, 3), np.uint8))

class QueueBounds(unittest.TestCase):
    def test_a_full_queue_drops_the_newest_and_counts_it(self):
        q = BoundedFrameQueue(maxsize=2)
        self.assertTrue(q.put(_f(1, 0.0)))
        self.assertTrue(q.put(_f(2, 0.1)))
        self.assertFalse(q.put(_f(3, 0.2)))
        self.assertEqual(q.drops, 1)
        self.assertEqual(q.depth, 2)

    def test_the_queue_never_exceeds_its_bound(self):
        q = BoundedFrameQueue(maxsize=2)
        for i in range(50):
            q.put(_f(i, i / 30))
        self.assertLessEqual(q.depth, 2)

    def test_get_returns_frames_in_order(self):
        q = BoundedFrameQueue(maxsize=2)
        q.put(_f(1, 0.0)); q.put(_f(2, 0.1))
        self.assertEqual(q.get().frame_id, 1)
        self.assertEqual(q.get().frame_id, 2)

    def test_get_on_an_empty_queue_returns_none_rather_than_blocking_forever(self):
        self.assertIsNone(BoundedFrameQueue().get(timeout=0.01))

    def test_drops_survive_a_get(self):
        q = BoundedFrameQueue(maxsize=1)
        q.put(_f(1, 0.0)); q.put(_f(2, 0.1))
        q.get()
        self.assertEqual(q.drops, 1)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_queue -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.engine.queue'`.

- [ ] **Step 3: Implement `app/engine/queue.py`**

`put` refuses when full and increments `drops` — the drop is **counted, never silent** (spec §6 requires explicit drop counters). `get` returns `None` on timeout. Add `tests/test_source.py` alongside it: assert that reading `assets/media/retail-checkout.mp4` yields rising `frame_id`, monotonic `captured_at`, the measured source fps (29.97) and 1920×1080 frames, that `restart()` increments `source_loop_id`, and that the analysis rate is `target_fps` rather than the source rate.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_queue tests.test_source -v`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add app/engine/source.py app/engine/queue.py tests/test_source.py tests/test_queue.py
git commit -m "M1: clip source with loop ids and a bounded newest-frame queue"
```

---

### Task 6: The inference worker process

**Files:**
- Create: `app/engine/worker.py`
- Test: `tests/test_worker.py`

**Interfaces:**
- Consumes: `app.inference.preprocess.preprocess_bgr` (M0), `app.inference.decode` (M0), `app.inference.pinned` (M0), `app.contracts.frame.FramePacket` (Task 1), `app.engine.queue.BoundedFrameQueue` (Task 5)
- Produces: `WorkerConfig` — frozen dataclass: `device_id: str`, `model_id: str`, `input_width: int`, `input_height: int`, `conf_threshold: float`, `iou_threshold: float`; `InferenceResult` — frozen dataclass: `frame: FramePacket`, `detections: tuple[Detection, ...]`, `latency_ms: float`, `execution_devices: tuple[str, ...]`, `error: str | None`; `run_worker(config: WorkerConfig, in_queue, out_queue) -> None` (the spawn entry point); `infer_once(request, frame, config) -> InferenceResult` (importable for tests)

- [ ] **Step 1: Write the failing test**

```python
import unittest
from app.contracts.frame import FramePacket
from app.engine.worker import infer_once, WorkerConfig
from app.inference.preprocess import preprocess_bgr
import numpy as np

def _frame(fid=0, loop=0, image=None):
    img = np.zeros((1080, 1920, 3), np.uint8) if image is None else image
    return FramePacket(session_id="s", scenario_id="retail", stream_id="A", frame_id=fid,
                       source_pts=fid / 30, source_loop_id=loop, captured_at=fid / 30,
                       decoded_at=fid / 30, width=1920, height=1080, image=img)

def _cfg(device="CPU"):
    return WorkerConfig(device_id=device, model_id="yolo11n-fp16", input_width=640,
                        input_height=640, conf_threshold=0.25, iou_threshold=0.45)

class RealInference(unittest.TestCase):
    def test_a_real_frame_produces_a_result_carrying_its_own_frame_id(self):
        cap = __import__("cv2").VideoCapture("assets/media/retail-checkout.mp4")
        ok, img = cap.read(); cap.release()
        self.assertTrue(ok)
        r = infer_once(None, _frame(11, image=img), _cfg())
        self.assertEqual(r.error, None)
        self.assertEqual(r.frame.frame_id, 11)
        self.assertGreater(r.latency_ms, 0.0)
        self.assertTrue(r.execution_devices)

    def test_placement_is_confirmed_not_requested(self):
        cap = __import__("cv2").VideoCapture("assets/media/retail-checkout.mp4")
        ok, img = cap.read(); cap.release()
        r = infer_once(None, _frame(0, image=img), _cfg("NPU"))
        self.assertTrue(r.execution_devices, "EXECUTION_DEVICES was not read after inference")

    def test_a_failure_is_returned_as_an_error_not_raised(self):
        r = infer_once(None, _frame(0), _cfg("NPU"), compile_failure=True)
        self.assertIsNotNone(r.error)
        self.assertEqual(r.execution_devices, ())
        self.assertIsNone(r.latency_ms or None) or self.assertGreaterEqual(r.latency_ms or 0, 0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_worker -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.engine.worker'`.

- [ ] **Step 3: Implement `app/engine/worker.py`**

`infer_once` preprocesses with `preprocess_bgr(frame.image, input_width, input_height)` — the pad default of 114 applies — then fetches the output tensor **by the compiled model's port object** or `get_output_tensor(0)`, decodes with the recorded `ModelOutputSpec`, and reads `EXECUTION_DEVICES` from the request's own compiled model **after** the inference. Wrap the whole body: on any exception return an `InferenceResult` with `error` set, `execution_devices=()` and no latency. That is spec §5's "if the target fails, show the failure" and AGENTS.md rule 12.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_worker -v`
Expected: `ok`. The NPU test needs the NPU plugin; if it is unavailable, the test must **skip with a stated reason**, not pass silently and not fail.

- [ ] **Step 5: Commit**

```bash
git add app/engine/worker.py tests/test_worker.py
git commit -m "M1: inference worker with post-inference placement proof and honest failure"
```

---

### Task 7: Metrics contract and the background collector

**Files:**
- Create: `app/contracts/metric_sample.py`, `app/telemetry/collector.py`
- Test: `tests/test_metric_sample.py`, `tests/test_collector.py`

**Interfaces:**
- Consumes: `psutil` (pinned)
- Produces:
  - `app.contracts.metric_sample.MetricSample` — frozen dataclass: `name: str`, `value: float | None`, `unit: str`, `provider: str`, `device_scope: str | None`, `process_scope: tuple[str, ...]`, `window_seconds: float`, `sampled_at: float`, `validity: Validity`; methods `is_stale(now: float) -> bool`, `to_json() -> dict`
  - `Validity` — `str, Enum`: `VALID`, `STALE`, `UNAVAILABLE`
  - `app.telemetry.collector.MetricCollector(process_pids: Sequence[int], interval: float = 0.25)` with `sample() -> tuple[MetricSample, ...]`, `system_cpu_percent() -> float | None`, `app_cpu_percent() -> float | None`, `stop()`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from app.contracts.metric_sample import MetricSample, Validity

def _s(value, validity=Validity.VALID, sampled_at=100.0, name="system_cpu_percent"):
    return MetricSample(name=name, value=value, unit="percent", provider="psutil",
                        device_scope=None, process_scope=("ui",), window_seconds=1.0,
                        sampled_at=sampled_at, validity=validity)

class UnavailableIsNotZero(unittest.TestCase):
    def test_review_focus_4_a_missing_value_is_null_not_zero(self):
        s = _s(None, Validity.UNAVAILABLE)
        self.assertIsNone(s.value)
        self.assertNotEqual(s.value, 0)
        self.assertEqual(s.validity, Validity.UNAVAILABLE)

    def test_a_genuine_zero_is_still_a_value(self):
        s = _s(0.0)
        self.assertEqual(s.value, 0.0)
        self.assertEqual(s.validity, Validity.VALID)

    def test_staleness_is_derived_from_the_timestamp(self):
        self.assertTrue(_s(1.0, sampled_at=100.0).is_stale(now=103.0))
        self.assertFalse(_s(1.0, sampled_at=100.0).is_stale(now=101.0))

    def test_an_unavailable_sample_is_never_stale_but_unavailable(self):
        s = _s(None, Validity.UNAVAILABLE, sampled_at=0.0)
        self.assertFalse(s.is_stale(now=10_000.0))
        self.assertEqual(s.validity, Validity.UNAVAILABLE)

    def test_json_keeps_none_as_null(self):
        import json
        self.assertIn("null", json.dumps(_s(None, Validity.UNAVAILABLE).to_json()))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_metric_sample -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.contracts.metric_sample'`.

- [ ] **Step 3: Implement both modules**

`MetricSample` is frozen; `is_stale` returns `False` for `UNAVAILABLE` so a permanently-missing counter reads as unavailable rather than degrading into "stale". `MetricCollector` measures **system** CPU across all logical cores and **app** CPU as the sum over the supplied PIDs normalised to 0–100% of total capacity — a process that died makes its contribution `None`, not zero. It samples at ~4 Hz and does no work other than sampling; **it must not call `DeviceRegistry.describe()`**, which costs 3.3–3.8 s.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_metric_sample tests.test_collector -v`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add app/contracts/metric_sample.py app/telemetry/collector.py tests/test_metric_sample.py tests/test_collector.py
git commit -m "M1: metric sample contract with null-not-zero and a background collector"
```

---

### Task 8: The supervisor, routing, and one-time worker restart

**Files:**
- Create: `app/supervisor/__init__.py`, `app/supervisor/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: Tasks 5, 6, 7
- Produces: `app.contracts.route_plan.RoutePlan` — frozen dataclass: `generation: int`, `requested: Mapping[str, str]`, `confirmed: Mapping[str, tuple[str, ...]]`, `state: str` (`"pending"`/`"active"`/`"failed"`), `errors: tuple[str, ...]`; `Supervisor(device_id: str = "CPU")` with `start()`, `stop()`, `submit(frame: FramePacket) -> bool`, `latest_result() -> InferenceResult | None`, `route_plan() -> RoutePlan`, `select_device(device_id: str) -> RoutePlan`, `samples() -> tuple[MetricSample, ...]`, `restart_count -> int`, signals: `result_ready(object)`, `state_changed(object)`, `error_raised(str)`

- [ ] **Step 1: Write the failing test**

```python
import unittest
import numpy as np
from app.contracts.frame import FramePacket
from app.supervisor.runner import Supervisor

def _frame(fid=0):
    return FramePacket(session_id="s", scenario_id="retail", stream_id="A", frame_id=fid,
                       source_pts=fid / 30, source_loop_id=0, captured_at=fid / 30,
                       decoded_at=fid / 30, width=1920, height=1080,
                       image=np.zeros((1080, 1920, 3), np.uint8))

class Routing(unittest.TestCase):
    def test_a_fresh_supervisor_reports_a_pending_route_then_a_confirmed_one(self):
        s = Supervisor(device_id="CPU")
        self.assertEqual(s.route_plan().state, "pending")
        s.start()
        s.submit(_frame(0))
        for _ in range(300):
            if s.route_plan().state == "active":
                break
            __import__("time").sleep(0.05)
        plan = s.route_plan()
        self.assertEqual(plan.state, "active")
        self.assertTrue(s.is_device("CPU"), plan)
        s.stop()

    def test_selecting_a_device_bumps_the_generation(self):
        s = Supervisor(device_id="CPU"); s.start()
        before = s.route_plan().generation
        after = s.select_device("CPU").generation
        self.assertGreater(after, before)
        s.stop()

    def test_a_failed_device_surfaces_the_fallback_and_never_claims_the_target(self):
        s = Supervisor(device_id="NPU", compile_failure=True)
        s.start()
        for _ in range(100):
            if s.route_plan().state in ("active", "failed"):
                break
            __import__("time").sleep(0.05)
        plan = s.route_plan()
        self.assertNotEqual(plan.confirmed.get("A", ()), ("NPU",))
        self.assertTrue(s.errors())
        s.stop()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_runner -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.supervisor.runner'`.

- [ ] **Step 3: Implement `app/supervisor/runner.py`**

`start()` spawns the worker with `multiprocessing` **spawn** semantics (Windows-safe; no `fork` assumptions, per the design's architecture note). `route_plan().state` is `"pending"` until the worker has reported a result with a confirmed `EXECUTION_DEVICES`, then `"active"`. `is_device(name)` reads the **confirmed** map, never the request. A worker that exits or reports `error` is restarted **once**; a second failure sets `state="failed"` and publishes the error, and the supervisor must not keep claiming the target device.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_runner -v`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add app/supervisor/ tests/test_runner.py
git commit -m "M1: supervisor with explicit route generations and one-time worker restart"
```

---

### Task 9: The QML booth screen

**Files:**
- Create: `app/theme.py`, `app/ui/Main.qml`, `app/ui/VideoTile.qml`, `app/ui/ComputeCard.qml`, `app/ui/PipelineStrip.qml`, `app/ui/Bridge.qml`, `app/ui/bridge.py`
- Test: `tests/test_ui_contract.py` (offscreen render assertions; no display needed)

**Interfaces:**
- Consumes: Tasks 7, 8
- Produces: `app.ui.bridge.UiBridge(QObject)` with properties `frame_ready`, `overlay_boxes`, `device_confirmed`, `device_requested`, `route_state`, `ai_fps`, `frame_age_p50_ms`, `frame_age_p95_ms`, `system_cpu_percent`, `app_cpu_percent`, `gpu_available`, `npu_available`, `event_counts`, `status_message`, `error_message`, and signals `frameArrived(object)`, `resultArrived(object)`, `routeChanged(object)`, `metricUpdated(object)`, `eventEmitted(object)`. Every optional numeric property is `null` when unavailable — never `0`.

- [ ] **Step 1: Write the failing test**

```python
import os, unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

class RequiredCopy(unittest.TestCase):
    def test_the_recorded_footage_label_exists_in_the_ui_sources(self):
        # Spec §4: displayed continuously. Its absence was a real gap in the
        # previous build, so it is pinned in source, not just reviewed by eye.
        from pathlib import Path
        qml = "\n".join(p.read_text(encoding="utf-8")
                       for p in Path("app/ui").glob("*.qml"))
        self.assertIn("Recorded footage", qml)
        self.assertIn("live inference", qml)

    def test_an_unavailable_metric_renders_as_unavailable_not_zero(self):
        from app.ui.bridge import UiBridge
        b = UiBridge()
        self.assertIsNone(b.system_cpu_percent)
        b.set_metrics(system_cpu_percent=None)
        self.assertEqual(b.display_text("system_cpu_percent"), "unavailable")

    def test_a_zero_metric_renders_as_a_number(self):
        from app.ui.bridge import UiBridge
        b = UiBridge()
        b.set_metrics(system_cpu_percent=0.0)
        self.assertNotEqual(b.display_text("system_cpu_percent"), "unavailable")

    def test_the_displayed_device_is_the_confirmed_one(self):
        from app.ui.bridge import UiBridge
        b = UiBridge()
        b.set_route(requested="NPU", confirmed=("CPU",), state="active")
        self.assertEqual(b.device_confirmed, "CPU")

    def test_no_magic_metric_numbers_live_in_qml(self):
        from pathlib import Path
        for p in Path("app/ui").glob("*.qml"):
            text = p.read_text(encoding="utf-8")
            self.assertNotIn("0.25", text, f"{p.name} hardcodes a threshold")
            self.assertNotIn("640", text, f"{p.name} hardcodes the input size")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_ui_contract -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.ui.bridge'`.

- [ ] **Step 3: Implement the UI**

1920×1080 logical canvas, ~68% video / 32% metrics, 12/66/22 vertical split. Hero video with boxes, labels, track ids and zone rectangles; two thumbnails; three compute cards (CPU always, GPU and NPU visibly disabled when their telemetry is unavailable, labelled "Utilization unavailable" rather than showing 0); a pipeline strip with per-stage device badges; a bottom control bar. **"Recorded footage • live inference" is a persistent visible label**, and the short explanation copy for YOLO detection, tracking, zone rules, CPU, GPU/NPU and metrics is reachable from hover and from a pinned panel. Type sizes per spec §3; every optional number routes through `bridge.display_text(name)`.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_ui_contract -v`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add app/theme.py app/ui/ tests/test_ui_contract.py
git commit -m "M1: QML booth screen with the recorded-footage label and null-not-zero rendering"
```

---

### Task 10: `app.main`, the self-test, and the headless run

**Files:**
- Create: `app/main.py`, `tools/run_retail_stream.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: everything above
- Produces: `app.main` CLI with `--scenario retail --device {CPU,GPU,NPU} --source loop --selftest --headless --seconds N --json PATH`; `--selftest` runs with no network and exits 0; **release mode refuses to start with a simulated provider registered**

- [ ] **Step 1: Write the failing test**

```python
import unittest

class SelfTestAndReleaseMode(unittest.TestCase):
    def test_selftest_exits_zero(self):
        import subprocess, sys
        r = subprocess.run([sys.executable, "-m", "app.main", "--selftest"],
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PASS", r.stdout)

    def test_release_mode_refuses_a_simulated_provider(self):
        from app.main import assert_no_simulation_in_release
        with self.assertRaises(SystemExit):
            assert_no_simulation_in_release(release=True, providers=("simulated_cpu",))
        assert_no_simulation_in_release(release=False, providers=("simulated_cpu",))
        assert_no_simulation_in_release(release=True, providers=("psutil",))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m unittest tests.test_main -v`
Expected: `ERROR` — `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Implement `app/main.py` and `tools/run_retail_stream.py`**

`--headless` runs the full pipeline with `QT_QPA_PLATFORM=offscreen` and writes a JSON run summary: frames submitted, frames analysed, drops, per-device placement, AI FPS, frame age p50/p95, system and app CPU, and the business events with their `dedup_key`s. `--selftest` exercises preprocess→infer→decode→track→event→render-metadata with no network and no display.

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_main -v`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tools/run_retail_stream.py tests/test_main.py
git commit -m "M1: application entry point, self-test and headless run summary"
```

---

### Task 11: The M1 qualification report

**Files:**
- Create: `docs/qualification/M1-qualification.md`, update `docs/qualification/qualification-matrix.csv`, append `bench_report.md`

**Interfaces:**
- Consumes: every tool above
- Produces: no code; the milestone gate

- [ ] **Step 1: Run the real demo and capture the evidence**

```powershell
.venv\Scripts\python.exe tools\run_retail_stream.py --scenario retail --device CPU --seconds 60 --json logs\m1-cpu.json
.venv\Scripts\python.exe tools\run_retail_stream.py --scenario retail --device GPU --seconds 60 --json logs\m1-gpu.json
.venv\Scripts\python.exe tools\run_retail_stream.py --scenario retail --device NPU --seconds 60 --json logs\m1-npu.json
.venv\Scripts\python.exe tools\preflight.py
.venv\Scripts\python.exe -m app.main --selftest
.venv\Scripts\python.exe -m unittest discover -s tests
```

Paste the **real** output of every command. If a device fails, record the failure and the fallback — do not substitute a working device.

- [ ] **Step 2: Write the report**

Sections: what M1 added; the routing evidence (`requested` vs `confirmed` per device, from `EXECUTION_DEVICES`); AI FPS and frame age per device; the business events actually emitted, with their `dedup_key`s and the loop boundaries; the visible error/fallback state; the "Recorded footage • live inference" label with a screenshot path; the metrics shown and each one's numerator, denominator, window, provider and scope; and an explicit **list of unmet gates**, carried forward from M0's two plus anything new.

Mark every M0 cell of the qualification matrix `passed` only if M1 re-measured it, and add the M1 row. A number that is unavailable reads `unavailable` — never `0`, never a plausible estimate.

- [ ] **Step 3: Run the whole gate set and paste the results**

```powershell
.venv\Scripts\python.exe tools\verify_sources.py
.venv\Scripts\python.exe tools\preflight.py
.venv\Scripts\python.exe -m app.main --selftest
.venv\Scripts\python.exe -m unittest discover -s tests
```

- [ ] **Step 4: Commit**

```bash
git add docs/qualification/ bench_report.md
git commit -m "M1: qualification report, routing evidence and measured event counts"
```

---

## M1 exit criteria

M1 is complete only when all of these hold, with pasted evidence in `docs/qualification/M1-qualification.md`:

1. One retail clip runs end to end with **real** detection overlays and no precomputed detections.
2. An explicit CPU/GPU/NPU selector exists, and the **selected and actual devices match** — proven by `EXECUTION_DEVICES` re-read after the first inference, never the request.
3. **Frame IDs prevent overlay drift**: a result whose `frame_id` is not the displayed frame is discarded, proven by a test.
4. Event tests cover **transfer, return, occlusion, duplicate prevention, and loop reset**, and the emitted events are reported with their `dedup_key`s.
5. The same clip works on **each** target validated in M0.
6. A runtime failure produces a **visible error/fallback state**, and the UI never claims the failed target.
7. "Recorded footage • live inference" is displayed, pinned by a test on the QML source.
8. Unavailable metrics render as `unavailable`, never `0`; a genuine `0` still renders as a number.
9. `verify_sources.py`, `preflight.py`, `--selftest` and the full unit suite are green, with real output pasted.

**Not claimed by M1:** routing switches (M2), the three-stream roadshow layout (M2), attract mode and the full booth layout (M3), validated GPU/NPU utilisation counters and the comparison runner (M4), the other three vertical packs (M5), packaging and soak (M6).

**Carried forward from M0 and still open after M1** — these are not M1's to close and must not be reported as closed:
- **GPU/NPU utilisation attribution** (unmet gate 2) — M4.
- **A real observed CPU fallback** (unmet gate 9) — M1 closes the *visible* failure state, but the fallback must still be *exercised for real* in M2's route-change matrix.
- **Retail shelf-to-basket** remains unmet; "items entering the checkout zone" is the labelled deliverable.
- **Code signing** (M6) and the **1920×1080 DPI gate** (M3, offscreen harness).
- **A second annotated frame with a non-square object**, so the `w`/`h` blind spot becomes detectable.

## Next milestone

M2 — three concurrent streams, the four GPU/NPU toggle combinations, bounded queues under load, route transitions, and 50 alternating warm route changes. Its plan is written after M1 closes, building on the `FramePacket`, `BoundedFrameQueue`, `RoutePlan` and `Supervisor` interfaces M1 actually produced.
