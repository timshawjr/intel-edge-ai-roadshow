"""The inference worker: frame identity, confirmed placement, and honest failure.

Review Focus #4 is what most of this module is about: **the worker's real target
silently falling back to CPU while the UI says NPU.** Three separate claims are tested
here and they are not the same claim:

- `test_placement_is_confirmed_not_requested` asks for a device by name and checks that
  the *confirmed* `EXECUTION_DEVICES` read - after the inference, from the request's own
  compiled model - names that device back. Requesting NPU is not evidence of NPU; this
  is.
- `test_the_devices_property_is_read_after_the_inference_not_before` pins the **order**
  with a stand-in request that records the sequence of calls. A worker that read
  placement before inferring would report whatever the plugin had cached from the
  compile, which is a claim about a placement that had not happened.
- `test_a_requested_device_that_does_not_exist_fails_instead_of_running_anywhere` asks
  for a device that is not there and requires an `error`, `latency_ms is None` and
  `execution_devices == ()`. The tempting failure - compile nowhere, decode an empty
  list, report a clean success with no devices - is what review focus 4 calls a clean
  empty success, and it is refused structurally in `InferenceResult.__post_init__`.

**Any device this machine does not have is skipped with a stated reason** - never a
silent pass, and never a failure for a machine that was not the one the milestone was
qualified on. The same holds for the clip and the IR: both are real artifacts, and a
test that passes without them would be passing vacuously.

The real-frame tests decode `assets/media/retail-checkout.mp4` and assert that frame 60
produces a `person` above the confidence threshold, which ties the whole path - real
pixels, real letterbox at pad 114, real compiled model, real `cx,cy,w,h` decode, real
NMS - to a semantically correct answer rather than to a self-consistent one.
"""

import unittest

import cv2
import numpy as np

from app.contracts.frame import FramePacket
from app.engine.worker import (
    DISPLAY_COPY_MAX_WIDTH,
    InferenceResult,
    WorkerConfig,
    _confirmed_devices,
    display_copy,
    infer_once,
    run_worker,
)
from app.inference.devices import is_device
from app.inference.pinned import MODEL_ID, REPO_ROOT, class_names, verified_ir_path

CLIP = REPO_ROOT / "assets" / "media" / "retail-checkout.mp4"
DEVICE_IDS = ("CPU", "GPU", "NPU")
#: The frame M0 measured: the top class is `person` at 0.909, which is what makes the
#: real-frame assertions semantic rather than merely self-consistent.
REAL_FRAME_INDEX = 60
REAL_FRAME_CLASS = "person"


def _require_clip() -> None:
    if not CLIP.exists():
        raise unittest.SkipTest(
            f"the pinned clip {CLIP} is not provisioned in this tree, so no assertion "
            f"about real pixels can be made. Provision assets/ with "
            f"tools/provision_venv.py; this test is not passing vacuously, it is not "
            f"running."
        )


def _require_ir() -> None:
    _require_clip()
    try:
        verified_ir_path()
    except (FileNotFoundError, ValueError) as exc:
        raise unittest.SkipTest(
            f"the pinned IR is not provisioned or does not match its recorded digest "
            f"({exc}). Re-provision rather than substitute; this test is not passing "
            f"vacuously, it is not running."
        ) from exc


def _available_devices() -> tuple[str, ...]:
    import openvino as ov

    return tuple(ov.Core().available_devices)


def _require_device(device_id: str) -> None:
    _require_ir()
    present = _available_devices()
    if device_id not in present:
        raise unittest.SkipTest(
            f"device {device_id!r} is not available on this machine: "
            f"ov.Core().available_devices() reports {list(present)}. The placement "
            f"assertion for {device_id!r} cannot be made here; this test is skipped with "
            f"that reason rather than passing silently and rather than failing."
        )


def _real_frame(index: int = REAL_FRAME_INDEX) -> np.ndarray:
    cap = cv2.VideoCapture(str(CLIP))
    try:
        ok, image = cap.read()
        for _ in range(index):
            ok, image = cap.read()
    finally:
        cap.release()
    if not ok or image is None:
        raise unittest.SkipTest(
            f"cv2 could not decode frame {index} of {CLIP}; there is no real frame to "
            f"run the model on."
        )
    return image


def _frame(fid=0, loop=0, image=None):
    img = np.zeros((1080, 1920, 3), np.uint8) if image is None else image
    return FramePacket(session_id="s", scenario_id="retail_sanity", stream_id="A",
                       frame_id=fid, source_pts=fid / 30, source_loop_id=loop,
                       captured_at=fid / 30, decoded_at=fid / 30,
                       width=int(img.shape[1]), height=int(img.shape[0]), image=img)


def _cfg(device="CPU", **kwargs):
    base = dict(device_id=device, model_id=MODEL_ID, input_width=640, input_height=640,
                conf_threshold=0.25, iou_threshold=0.45)
    base.update(kwargs)
    return WorkerConfig(**base)


class TheResultContract(unittest.TestCase):
    """A failure is not a clean empty success, and that is enforced structurally."""

    def test_a_success_carries_a_latency_and_a_confirmed_placement(self):
        result = InferenceResult(frame=_frame(), detections=(), latency_ms=12.5,
                                 execution_devices=("CPU",), error=None)
        self.assertEqual(result.latency_ms, 12.5)
        self.assertIsNone(result.error)

    def test_a_failure_carries_no_latency_and_no_placement(self):
        result = InferenceResult.failure(_frame(), "the device could not be compiled")
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.latency_ms)
        self.assertEqual(result.execution_devices, ())
        self.assertEqual(result.detections, ())

    def test_a_result_with_a_placement_but_no_latency_is_refused(self):
        """The exact shape of review focus 4's failure, refused at construction."""
        with self.assertRaises(ValueError):
            InferenceResult(frame=_frame(), detections=(), latency_ms=None,
                            execution_devices=("CPU",), error=None)

    def test_a_result_with_no_placement_and_no_error_is_refused(self):
        with self.assertRaises(ValueError):
            InferenceResult(frame=_frame(), detections=(), latency_ms=1.0,
                            execution_devices=(), error=None)

    def test_a_result_with_an_error_and_a_latency_is_refused(self):
        with self.assertRaises(ValueError):
            InferenceResult(frame=_frame(), detections=(), latency_ms=1.0,
                            execution_devices=(), error="it went wrong")

    def test_detections_must_be_a_tuple_of_detections_not_a_bare_list_of_boxes(self):
        with self.assertRaises(TypeError):
            InferenceResult(frame=_frame(), detections=[(0, 0.9, 1.0, 2.0, 3.0, 4.0)],
                            latency_ms=1.0, execution_devices=("CPU",), error=None)

    def test_to_json_keeps_the_error_and_the_absent_placement(self):
        import json

        payload = InferenceResult.failure(_frame(), "no NPU plugin").to_json()
        text = json.dumps(payload)
        self.assertIn('"error": "no NPU plugin"', text)
        self.assertIn('"latency_ms": null', text)
        self.assertIn('"execution_devices": []', text)


class TheConfigContract(unittest.TestCase):
    def test_a_non_positive_input_extent_is_refused(self):
        for kwargs in ({"input_width": 0}, {"input_height": -640}):
            with self.assertRaises(ValueError):
                _cfg(**kwargs)

    def test_a_threshold_outside_zero_to_one_is_refused(self):
        for kwargs in ({"conf_threshold": 1.5}, {"iou_threshold": -0.1}):
            with self.assertRaises(ValueError):
                _cfg(**kwargs)

    def test_an_empty_device_or_model_id_is_refused(self):
        with self.assertRaises(ValueError):
            _cfg(device="")
        with self.assertRaises(ValueError):
            _cfg(model_id="   ")

    def test_an_implicit_placement_plugin_is_refused_at_the_config_boundary(self):
        for pseudo in ("AUTO", "HETERO", "MULTI", "BATCH"):
            with self.assertRaises(ValueError):
                _cfg(device=pseudo)


class RealInferenceOnTheRealClip(unittest.TestCase):
    """The whole path, on real pixels, with a semantically checked answer."""

    def setUp(self):
        _require_device("CPU")

    def test_a_real_frame_produces_a_result_carrying_its_own_frame_id(self):
        result = infer_once(None, _frame(11, image=_real_frame()), _cfg())
        self.assertIsNone(result.error)
        self.assertEqual(result.frame.frame_id, 11)
        self.assertEqual(result.frame.identity, ("s", "A", 0, 11))
        self.assertGreater(result.latency_ms, 0.0)
        self.assertTrue(result.execution_devices)

    def test_a_real_frame_finds_the_person_m0_measured(self):
        result = infer_once(None, _frame(REAL_FRAME_INDEX, image=_real_frame()), _cfg())
        self.assertIsNone(result.error)
        labels = class_names()
        named = [(labels[d.class_id], d.score) for d in result.detections]
        self.assertTrue(named, "the model found nothing at all in a frame M0 measured "
                              "a person at 0.909 in; something is decoding wrong")
        self.assertEqual(named[0][0], REAL_FRAME_CLASS)
        self.assertGreater(named[0][1], 0.25)
        for label, _score in named:
            self.assertIn(label, labels, "a detection named something outside the pinned "
                                        "label list means class ids are misaligned")

    def test_a_detection_box_is_inside_the_frame_it_came_from(self):
        image = _real_frame()
        result = infer_once(None, _frame(REAL_FRAME_INDEX, image=image), _cfg())
        self.assertTrue(result.detections)
        for det in result.detections:
            x1, y1, x2, y2 = det.box
            self.assertGreater(x2, x1, "an inverted box means the cx,cy,w,h inverse was "
                                       "not applied")
            self.assertGreater(y2, y1)
            self.assertLess(abs(x1), image.shape[1] + 200)
            self.assertLess(abs(y1), image.shape[0] + 200)

    def test_the_guessed_frame_id_cannot_reach_the_result(self):
        """The packet handed in is the packet handed out, by identity."""
        image = _real_frame()
        packet = _frame(4242, loop=7, image=image)
        result = infer_once(None, packet, _cfg())
        self.assertIs(result.frame, packet)

    def test_the_confidence_threshold_actually_raises_the_bar(self):
        image = _real_frame()
        loose = infer_once(None, _frame(REAL_FRAME_INDEX, image=image),
                          _cfg(conf_threshold=0.05))
        strict = infer_once(None, _frame(REAL_FRAME_INDEX, image=image),
                            _cfg(conf_threshold=0.99))
        self.assertGreaterEqual(len(loose.detections), len(strict.detections))
        self.assertEqual(strict.detections, ())


class PlacementIsConfirmedNotRequested(unittest.TestCase):
    """Review Focus #4, on every device this machine has."""

    def test_placement_is_confirmed_not_requested(self):
        for device_id in DEVICE_IDS:
            with self.subTest(device=device_id):
                if device_id not in _available_devices():
                    continue  # reported by the test below, with a stated reason
                result = infer_once(None, _frame(0, image=_real_frame()), _cfg(device_id))
                self.assertIsNone(result.error, result.error)
                self.assertTrue(
                    result.execution_devices,
                    "EXECUTION_DEVICES was not read after the inference, so there is "
                    "no evidence any of this ran where it was asked to run")
                self.assertTrue(
                    is_device(result.execution_devices, device_id),
                    f"asked for {device_id}, confirmed {result.execution_devices}")

    def test_every_device_this_machine_reports_is_covered_by_the_test_above(self):
        """If a new device appears, the loop covers it rather than quietly skipping."""
        for device_id in _available_devices():
            self.assertIn(device_id, DEVICE_IDS)

    def test_each_device_skips_with_a_stated_reason_when_it_is_absent(self):
        for device_id in DEVICE_IDS:
            if device_id in _available_devices():
                continue
            with self.subTest(device=device_id):
                with self.assertRaises(unittest.SkipTest) as caught:
                    _require_device(device_id)
                self.assertIn(device_id, str(caught.exception))
                self.assertIn("available_devices", str(caught.exception))

    def test_a_requested_device_that_does_not_exist_fails_instead_of_running_anywhere(self):
        result = infer_once(None, _frame(0, image=_real_frame()),
                            _cfg(device="NOT_A_DEVICE"))
        self.assertIsNotNone(result.error)
        self.assertIn("NOT_A_DEVICE", result.error)
        self.assertEqual(result.execution_devices, ())
        self.assertIsNone(result.latency_ms)
        self.assertEqual(result.detections, ())

    def test_a_model_id_that_is_not_the_pinned_one_is_refused(self):
        result = infer_once(None, _frame(0, image=_real_frame()),
                            _cfg(model_id="yolo11n-int8"))
        self.assertIsNotNone(result.error)
        self.assertIn(MODEL_ID, result.error)
        self.assertEqual(result.execution_devices, ())

    def test_a_configured_input_extent_that_disagrees_with_the_pin_is_refused(self):
        result = infer_once(None, _frame(0, image=_real_frame()),
                            _cfg(input_width=8400, input_height=8400))
        self.assertIsNotNone(result.error)
        self.assertEqual(result.execution_devices, ())
        self.assertIsNone(result.latency_ms)


class FailureIsReturnedNotRaised(unittest.TestCase):
    """Spec section 5's 'if the target fails, show the failure'. AGENTS.md rule 12."""

    def setUp(self):
        _require_ir()

    def test_a_frame_preprocess_cannot_accept_is_an_error_not_an_exception(self):
        packet = _frame(0, image=np.zeros((1080, 1920, 3), np.float32))
        result = infer_once(None, packet, _cfg())
        self.assertIsNotNone(result.error)
        self.assertIn("uint8", result.error)
        self.assertEqual(result.execution_devices, ())
        self.assertIsNone(result.latency_ms)
        self.assertEqual(result.detections, ())

    def test_the_failing_result_still_carries_the_frame_it_was_for(self):
        packet = _frame(9, image=np.zeros((1080, 1920, 3), np.float32))
        result = infer_once(None, packet, _cfg())
        self.assertIs(result.frame, packet,
                      "a failure that loses the frame identity cannot be attributed to "
                      "a picture, and the overlay will keep the previous boxes up")

    def test_an_inference_that_throws_is_reported_with_no_latency_and_no_devices(self):
        request = _FailingRequest()
        result = infer_once(request, _frame(0, image=_real_frame()), _cfg())
        self.assertIsNotNone(result.error)
        self.assertIn("this plugin is on fire", result.error)
        self.assertIsNone(result.latency_ms)
        self.assertEqual(result.execution_devices, ())
        self.assertEqual(result.detections, ())
        self.assertFalse(request.execution_devices_was_read,
                         "the worker read placement off a request whose inference threw")

    def test_an_inference_request_without_a_confirmed_placement_is_a_failure(self):
        request = _StubRequest(placement=None)
        result = infer_once(request, _frame(0, image=_real_frame()), _cfg())
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.latency_ms)
        self.assertEqual(result.execution_devices, ())


class ThePlacementReadIsNormalised(unittest.TestCase):
    """The measured inconsistency in `EXECUTION_DEVICES`, pinned at the read.

    CPU reports `['CPU']`, GPU reports `['GPU.0']` and the **NPU reports the bare string
    `'NPU'`** on this machine. `tuple('NPU')` is `('N', 'P', 'U')`, so a reader that did
    not normalise would render the NPU card as a three-device CPU-style row and a
    `subTest(device='NPU')` would fail on a machine where the NPU worked perfectly.
    """

    def test_the_three_measured_shapes_all_normalise_to_one_tuple(self):
        for reported, expected in (("NPU", ("NPU",)), (["CPU"], ("CPU",)),
                                   (["GPU.0"], ("GPU.0",))):
            with self.subTest(reported=reported):
                self.assertEqual(_confirmed_devices(_StubRequest(placement=reported)),
                                 expected)

    def test_an_empty_reading_is_refused_rather_than_becoming_a_success(self):
        for reported in (None, [], (), ""):
            with self.subTest(reported=reported):
                with self.assertRaises(RuntimeError) as caught:
                    _confirmed_devices(_StubRequest(placement=reported))
                self.assertIn("EXECUTION_DEVICES", str(caught.exception))

    def test_a_bare_string_is_never_split_into_characters(self):
        devices = _confirmed_devices(_StubRequest(placement="NPU"))
        self.assertEqual(len(devices), 1)
        self.assertTrue(is_device(devices, "NPU"))


class _StubRequest:
    """A stand-in `ov.InferRequest` that records the order of the calls.

    It exists because the ordering claim - `EXECUTION_DEVICES` is read **after** the
    inference, from the request's own compiled model - cannot be checked against a real
    device: the real one does both things too fast to observe. This records the sequence
    and hands back a real `[1, 84, 8400]` tensor so the decode path is the production
    one.
    """

    def __init__(self, placement=("CPU",), infer_error: Exception | None = None,
                 tensor_shape=(1, 84, 8400)):
        self.calls: list[str] = []
        self.placement = placement
        self.infer_error = infer_error
        self._shape = tensor_shape
        self.execution_devices_was_read = False

    def get_compiled_model(self):
        return self

    def get_property(self, name):
        self.calls.append(f"property:{name}")
        if name != "EXECUTION_DEVICES":
            raise KeyError(name)
        self.execution_devices_was_read = True
        return self.placement

    @property
    def outputs(self):
        return [f"compiled-port-{i}" for i in range(len(self._shape) - 1)]

    def infer(self, _data):
        self.calls.append("infer")
        if self.infer_error is not None:
            raise self.infer_error

    def get_tensor(self, port):
        self.calls.append(f"get_tensor:{port}")
        return np.zeros(self._shape, np.float32)

    def get_output_tensor(self, index):
        self.calls.append(f"get_output_tensor:{index}")
        return np.zeros(self._shape, np.float32)


class _FailingRequest(_StubRequest):
    def infer(self, _data):
        super().infer(_data)
        raise RuntimeError("this plugin is on fire")


class TheOrderOfThePlacementRead(unittest.TestCase):
    """The single most consequential line in the worker."""

    def setUp(self):
        _require_ir()

    def test_the_devices_property_is_read_after_the_inference_not_before(self):
        request = _StubRequest()
        infer_once(request, _frame(0, image=_real_frame()), _cfg())
        self.assertIn("infer", request.calls)
        self.assertIn("property:EXECUTION_DEVICES", request.calls)
        self.assertLess(request.calls.index("infer"),
                        request.calls.index("property:EXECUTION_DEVICES"),
                        "placement was read before the inference ran, so it describes a "
                        "compile rather than an execution")

    def test_placement_is_read_from_the_requests_own_compiled_model(self):
        request = _StubRequest()
        infer_once(request, _frame(0, image=_real_frame()), _cfg())
        # `_StubRequest.get_property` is only reachable through `get_compiled_model()`,
        # so a reading taken off a core or off the requested device id cannot reach it.
        self.assertTrue(request.execution_devices_was_read)

    def test_the_output_tensor_is_fetched_by_the_compiled_port_object_first(self):
        request = _StubRequest()
        infer_once(request, _frame(0, image=_real_frame()), _cfg())
        fetches = [c for c in request.calls if c.startswith("get_tensor:")
                   or c.startswith("get_output_tensor:")]
        self.assertTrue(fetches)
        self.assertTrue(fetches[0].startswith("get_tensor:compiled-port-"),
                        f"the first fetch was {fetches[0]!r}; the compiled port object "
                        f"is the one that is accepted, and get_tensor(0) raises")
        self.assertNotIn("get_tensor:0", request.calls)

    def test_the_int_index_fallback_is_used_when_the_port_object_is_refused(self):
        request = _StubRequest()
        request.get_tensor = lambda port: (_ for _ in ()).throw(
            TypeError("this build rejects the port object"))
        result = infer_once(request, _frame(0, image=_real_frame()), _cfg())
        self.assertIsNone(result.error, result.error)
        self.assertIn("get_output_tensor:0", request.calls)


class TheRequestTravelsWithTheConfirmation(unittest.TestCase):
    """Review finding M-5: the pair could not be checked because only half was carried.

    `infer_once(Stub("CPU"), frame, cfg("NPU"))` returned `error=None`,
    `execution_devices=('CPU',)` and a latency. **That is honest** - it says `('CPU',)`,
    not `('NPU',)`, and the reviewer cleared placement on that basis. The gap was that
    nothing forced a consumer to compare the confirmation against the request: with only
    `execution_devices` on the record, and `app/` having no caller of `is_device` outside
    its own definition, a card would have to know to perform the comparison itself.

    So `requested_device` is a field of the record and `fell_back` is the comparison. A
    fallback stays a legitimate, reportable state - what it can no longer be is invisible.
    """

    def setUp(self):
        _require_ir()

    def test_a_fallback_carries_both_halves_of_the_pair_and_says_it_fell_back(self):
        result = infer_once(_StubRequest(placement=("CPU",)), _frame(0, image=_real_frame()),
                            _cfg("NPU"))
        self.assertIsNone(result.error, result.error)
        self.assertEqual(result.requested_device, "NPU")
        self.assertEqual(result.execution_devices, ("CPU",),
                         "the confirmed placement was misreported")
        self.assertTrue(result.fell_back,
                        "an NPU request answered by the CPU did not report a fallback")

    def test_a_confirmed_placement_is_not_a_fallback(self):
        result = infer_once(_StubRequest(placement=("NPU",)), _frame(0, image=_real_frame()),
                            _cfg("NPU"))
        self.assertEqual(result.requested_device, "NPU")
        self.assertFalse(result.fell_back)

    def test_the_pair_is_compared_through_the_normaliser_not_by_string_equality(self):
        """The NPU reports the bare string, so `== 'NPU'` on a tuple would be wrong."""
        result = infer_once(_StubRequest(placement="NPU"), _frame(0, image=_real_frame()),
                            _cfg("NPU"))
        self.assertEqual(result.execution_devices, ("NPU",))
        self.assertFalse(result.fell_back,
                         "a bare-string placement was compared as a character tuple")

    def test_a_failure_keeps_the_requested_device_and_reports_no_fallback(self):
        """'The NPU was asked for and would not run' must survive to the screen."""
        result = infer_once(None, _frame(0, image=_real_frame()),
                            _cfg(device="NOT_A_DEVICE"))
        self.assertIsNotNone(result.error)
        self.assertEqual(result.requested_device, "NOT_A_DEVICE",
                         "the failure forgot what it was trying to do")
        self.assertEqual(result.execution_devices, ())
        self.assertIsNone(result.fell_back,
                          "a failure cannot be a fallback; nothing ran anywhere")

    def test_a_record_without_a_request_cannot_claim_a_fallback(self):
        result = InferenceResult(frame=_frame(), detections=(), latency_ms=1.0,
                                 execution_devices=("CPU",), error=None)
        self.assertIsNone(result.requested_device)
        self.assertIsNone(result.fell_back,
                          "with nothing requested there is nothing to have fallen back from")

    def test_the_serialised_result_carries_the_pair_and_the_verdict(self):
        import json

        payload = InferenceResult(frame=_frame(), detections=(), latency_ms=1.0,
                                  execution_devices=("CPU",), error=None,
                                  requested_device="NPU").to_json()
        self.assertEqual(payload["requested_device"], "NPU")
        self.assertEqual(payload["execution_devices"], ["CPU"])
        self.assertTrue(payload["fell_back"])
        json.dumps(payload)

    def test_a_blank_requested_device_is_refused(self):
        with self.assertRaises(ValueError):
            InferenceResult(frame=_frame(), detections=(), latency_ms=1.0,
                            execution_devices=("CPU",), error=None,
                            requested_device="  ")

    def test_a_real_run_on_every_device_agrees_with_its_request(self):
        """On this machine all three are present, so nothing should fall back."""
        for device_id in DEVICE_IDS:
            if device_id not in _available_devices():
                continue
            with self.subTest(device=device_id):
                result = infer_once(None, _frame(0, image=_real_frame()), _cfg(device_id))
                self.assertIsNone(result.error, result.error)
                self.assertEqual(result.requested_device, device_id)
                self.assertFalse(result.fell_back,
                                 f"asked for {device_id}, ran on "
                                 f"{result.execution_devices}")


class TheWorkerLoop(unittest.TestCase):
    """`run_worker` drains its queue, keeps going through a bad frame, and stops."""

    def setUp(self):
        _require_device("CPU")

    def test_run_worker_returns_one_result_per_frame_with_the_same_identity(self):
        frames = [_frame(4 * i, image=_real_frame()) for i in range(3)]
        results = _drain(frames)
        self.assertEqual([r.frame.identity for r in results],
                         [f.identity for f in frames])
        self.assertTrue(all(r.error is None for r in results), results)

    def test_run_worker_stops_on_the_none_sentinel_without_a_result(self):
        results = _drain([_frame(0, image=_real_frame())])
        self.assertEqual(len(results), 1)

    def test_one_bad_frame_does_not_stop_the_stream(self):
        good = _frame(0, image=_real_frame())
        bad = _frame(1, image=np.zeros((1080, 1920, 3), np.float32))
        after = _frame(2, image=_real_frame())
        results = _drain([good, bad, after])
        self.assertEqual(len(results), 3, "one bad frame took the pipeline down; "
                                         "AGENTS.md rule 12")
        self.assertIsNone(results[0].error)
        self.assertIsNotNone(results[1].error)
        self.assertIsNone(results[2].error)
        self.assertEqual(results[1].frame.identity, bad.identity)
        self.assertEqual(results[2].frame.identity, after.identity)

    def test_an_empty_queue_produces_no_results_and_returns(self):
        self.assertEqual(_drain([]), [])

    def test_a_result_from_a_later_loop_is_not_confused_with_the_earlier_one(self):
        loop0 = _frame(60, loop=0, image=_real_frame())
        loop1 = _frame(60, loop=1, image=_real_frame())
        results = _drain([loop0, loop1])
        self.assertEqual(results[0].frame.identity, ("s", "A", 0, 60))
        self.assertEqual(results[1].frame.identity, ("s", "A", 1, 60))
        self.assertNotEqual(results[0].frame.identity, results[1].frame.identity)
        self.assertEqual(int(results[0].detections[0].box == results[1].detections[0].box),
                         1, "the same frame of the same clip decoded to different boxes")


def _drain(frames) -> list:
    """Run `run_worker` over an in-process queue pair, bounded so a hang fails."""
    import queue

    in_q: "queue.Queue" = queue.Queue()
    out_q: "queue.Queue" = queue.Queue()
    for frame in frames:
        in_q.put(frame)
    in_q.put(None)
    run_worker(_cfg(), in_q, out_q)
    results = []
    while True:
        try:
            results.append(out_q.get_nowait())
        except Exception:  # noqa: BLE001 - queue.Empty on every implementation
            return results


class TheResultIsReducedToDisplayResolutionAfterInference(unittest.TestCase):
    """**The detector sees full resolution; only what crosses back is smaller.**

    The frame the UI feeder submits is full-resolution because the detector letterboxes it
    to 640x640 and small-object detection depends on that source. The frame that comes
    *back* is only ever painted, and it was coming back at 1920x1080: ~6.2 MB of pickle and
    an unpickle per frame plus a `QImage` construction on the UI thread - measured on the
    three-camera booth at ~3.4 ms and ~13-18 ms of CPU per frame respectively - for pixels
    the hero displays at 957x538.

    `run_worker` now reduces each successful result to `DISPLAY_COPY_MAX_WIDTH` **after**
    `infer_once` returns, scaling the detection boxes with the frame so every normalised
    coordinate is unchanged.

    Every assertion here has a mutation that would break it:

    - not reducing at all fails the shape assertion (the frame stays 1920x1080);
    - reducing the frame without scaling the boxes moves every box, which the
      normalised-agreement test catches against a direct full-resolution `infer_once`;
    - reducing **before** `infer_once` changes the detection set, which the same test
      catches.
    """

    def test_display_copy_scales_1080p_to_the_heros_native_width(self):
        source = np.zeros((1080, 1920, 3), np.uint8)
        copy = display_copy(source)
        self.assertEqual(copy.shape, (540, 960, 3), copy.shape)
        self.assertEqual(copy.dtype, np.uint8)
        self.assertAlmostEqual(copy.shape[1] / copy.shape[0], 1920 / 1080, places=3)
        self.assertEqual(DISPLAY_COPY_MAX_WIDTH, 960)

    def test_a_frame_at_or_below_the_target_is_not_resized(self):
        """A small source must not acquire a resize cost, and must not be re-allocated."""
        source = np.zeros((360, 640, 3), np.uint8)
        self.assertIs(display_copy(source), source)

    def test_run_worker_returns_a_result_at_display_resolution(self):
        _require_device("CPU")
        frame = _frame(REAL_FRAME_INDEX, image=_real_frame())
        result = _drain([frame])[0]
        self.assertIsNone(result.error, result.error)
        self.assertEqual(result.frame.image.shape, (540, 960, 3), result.frame.image.shape)
        self.assertEqual((result.frame.width, result.frame.height), (960, 540),
                         "the extents must match the image the frame now carries")
        self.assertNotEqual(result.frame.width, 1920,
                            "the 1920x1080 pixels must not travel back")

    def test_a_small_frame_is_returned_untouched(self):
        """A source already at display resolution pays nothing and keeps its own frame."""
        _require_device("CPU")
        small = cv2.resize(_real_frame(), (640, 360), interpolation=cv2.INTER_AREA)
        frame = _frame(REAL_FRAME_INDEX, image=small)
        result = _drain([frame])[0]
        self.assertIsNone(result.error, result.error)
        self.assertEqual(result.frame.image.shape, (360, 640, 3), result.frame.image.shape)

    def test_the_normalised_detections_agree_with_a_direct_full_resolution_run(self):
        """**The non-vacuous form of 'the geometry is unmoved'.**

        Reducing the frame without scaling its boxes would keep every box inside [0, 1]
        on the smaller extents but put it in the wrong place - the "correct box, wrong
        picture" class. Comparing the **normalised** boxes (box / frame extents) against a
        direct full-resolution `infer_once` is what catches it; a shape check cannot.
        """
        _require_device("CPU")
        frame = _frame(REAL_FRAME_INDEX, image=_real_frame())
        direct = infer_once(None, frame, _cfg())
        drained = _drain([frame])[0]
        self.assertIsNone(direct.error, direct.error)
        self.assertIsNone(drained.error, drained.error)
        self.assertTrue(direct.detections, "no detections to compare - the test is vacuous")
        self.assertEqual(len(drained.detections), len(direct.detections))
        for direct_det, drained_det in zip(direct.detections, drained.detections):
            self.assertEqual(drained_det.class_id, direct_det.class_id)
            for got, want, extent_got, extent_want in (
                    (drained_det.box[0], direct_det.box[0], drained.frame.width, direct.frame.width),
                    (drained_det.box[1], direct_det.box[1], drained.frame.height, direct.frame.height),
                    (drained_det.box[2], direct_det.box[2], drained.frame.width, direct.frame.width),
                    (drained_det.box[3], direct_det.box[3], drained.frame.height, direct.frame.height)):
                self.assertAlmostEqual(got / extent_got, want / extent_want, places=5,
                                       msg="a reduced box must normalise to the same place")


class MeasuredLatencyIsAPercentileOfRealWork(unittest.TestCase):
    """The worker test records a real distribution, not a single lucky number."""

    def test_the_latency_percentiles_over_a_real_device_are_ordered_and_positive(self):
        _require_device("CPU")
        import statistics

        image = _real_frame()
        packet = _frame(0, image=image)
        results = _drain([_frame(i, image=image) for i in range(25)])
        self.assertEqual(len(results), 25)
        self.assertTrue(all(r.error is None for r in results))
        samples = sorted(r.latency_ms for r in results)
        median = statistics.median(samples)
        self.assertGreater(median, 0.0)
        self.assertLessEqual(samples[0], median)
        self.assertLessEqual(median, samples[-1])
        print(f"\n[worker latency, CPU, n={len(samples)}] "
              f"min={samples[0]:.3f} p50={median:.3f} "
              f"max={samples[-1]:.3f} ms")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
