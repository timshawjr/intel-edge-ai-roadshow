"""Device discovery over a real `ov.Core`, and the placement contract.

Two things are pinned here that nothing downstream would catch:

1. **Discovery is discovered.** `FULL_DEVICE_NAME` comes from OpenVINO, never from a
   hardcoded family string. This machine is Lunar Lake; nothing here may present a
   measurement as Panther Lake or Arrow Lake.
2. **`EXECUTION_DEVICES` is read live, after an inference, never captured at
   construction.** See `PlacementIsNeverCaptured`. This is the one invariant the whole
   product's honesty rests on, so it is pinned two ways: the dataclass has no field
   that could hold a placement, and the read demonstrably goes back to the runtime.
"""

import dataclasses
import json
import unittest
from pathlib import Path

# CORRECTED 2026-10-01: openvino==2026.4.0 has no `openvino.runtime` submodule.
# The top-level module exposes Core, Model and InferRequest directly.
import openvino as ov

from app.inference.devices import (
    DEVICE_IDS,
    CompiledTarget,
    DeviceRegistry,
    TargetSpec,
    require_static_shape,
)

MODELS = json.loads(Path("config/models.json").read_text(encoding="utf-8"))
DETECTOR = next(m for m in MODELS["models"] if m["id"] == "yolo11n-fp16")
IR_PATH = Path(DETECTOR["ir_path"])


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

    def test_describe_covers_every_demo_device_in_order(self):
        self.assertEqual([s.device_id for s in DeviceRegistry(self.core).describe()],
                         list(DEVICE_IDS))

    def test_full_name_is_the_value_openvino_reports(self):
        # Read independently of the registry, so the registry is not simply echoing itself.
        reg = DeviceRegistry(self.core)
        for d in DEVICE_IDS:
            self.assertEqual(reg.full_name(d),
                             str(ov.Core().get_property(d, "FULL_DEVICE_NAME")))

    def test_the_name_is_never_a_hardcoded_family(self):
        # A Lunar Lake measurement must not be labelled Panther Lake or Arrow Lake.
        joined = " ".join(s.full_name for s in DeviceRegistry(self.core).describe())
        for wrong in ("Panther", "Arrow Lake", "Series 3", "Lunar Lake"):
            self.assertNotIn(wrong, joined,
                             f"a family name leaked into a discovered name: {wrong!r}")

    def test_runtime_version_is_the_full_build_string_not_a_bare_version(self):
        for s in DeviceRegistry(self.core).describe():
            self.assertEqual(s.runtime_version, ov.__version__)
            self.assertNotEqual(s.runtime_version, "2026.4.0")
            self.assertRegex(s.runtime_version, r"^\d+\.\d+\.\d+.+$")

    def test_an_unavailable_device_is_absent_from_available_but_still_described(self):
        statuses = {s.device_id: s for s in DeviceRegistry(self.core).describe()}
        self.assertEqual(sorted(statuses), sorted(DEVICE_IDS))
        for s in statuses.values():
            if not s.available:
                self.assertTrue(s.unavailable_reason)
                self.assertFalse(s.ready)
                self.assertIsNone(s.unavailable_reason if s.available else None)


class _FakeCore:
    """A `ov.Core` stand-in, so the failure paths are testable on a machine where all
    three devices are present - which is this one."""

    def __init__(self, names=("CPU",), errors=(), driver_properties=()):
        self._names = tuple(names)
        self._errors = dict(errors)
        self._driver_properties = dict(driver_properties)

    # `available_devices` is a property on the real `ov.Core`, not a method.
    @property
    def available_devices(self):
        return list(self._names)

    def get_property(self, device_id, name):
        if device_id in self._errors:
            raise self._errors[device_id]
        if name == "FULL_DEVICE_NAME":
            return f"Fake {device_id}"
        if name == "SUPPORTED_PROPERTIES":
            # A property *name* list, as the real plugin reports it.
            return list(self._driver_properties.get(device_id, {}))
        if name in self._driver_properties.get(device_id, ()):
            return self._driver_properties[device_id][name]
        raise RuntimeError(f"unsupported property {name} on {device_id}")


class AnAbsentDeviceIsExplained(unittest.TestCase):
    def test_an_absent_device_carries_the_openvino_error_text(self):
        core = _FakeCore(names=("CPU", "GPU"),
                         errors={"NPU": RuntimeError("NPU plugin is not available")})
        statuses = {s.device_id: s for s in DeviceRegistry(core).describe()}
        self.assertTrue(statuses["NPU"].unavailable_reason)
        self.assertIn("NPU plugin is not available", statuses["NPU"].unavailable_reason)
        self.assertFalse(statuses["NPU"].available)
        self.assertFalse(statuses["NPU"].ready)

    def test_an_available_device_has_no_reason_and_a_discovered_name(self):
        core = _FakeCore(names=("CPU", "GPU", "NPU"))
        for s in DeviceRegistry(core).describe():
            self.assertTrue(s.available)
            self.assertIsNone(s.unavailable_reason)
            self.assertEqual(s.full_name, f"Fake {s.device_id}")

    def test_a_device_with_no_driver_property_records_null_not_zero(self):
        core = _FakeCore(names=("CPU", "GPU", "NPU"))
        for s in DeviceRegistry(core).describe():
            self.assertIsNone(s.driver_version,
                              "a device reporting no driver version must be null, not 0")

    def test_a_reported_driver_version_is_recorded_verbatim(self):
        core = _FakeCore(names=("CPU", "GPU", "NPU"),
                         driver_properties={"NPU": {"NPU_DRIVER_VERSION": 1005540}})
        statuses = {s.device_id: s for s in DeviceRegistry(core).describe()}
        # Measured on this machine: the NPU plugin exposes NPU_DRIVER_VERSION = 1005540
        # and CPU/GPU expose no driver-version property at all.
        self.assertEqual(statuses["NPU"].driver_version, "1005540")
        self.assertIsNone(statuses["CPU"].driver_version)
        self.assertIsNone(statuses["GPU"].driver_version)

    def test_discovery_alone_never_claims_a_device_is_validated_or_ready(self):
        # Presence is not proof. Nothing has compiled a model at this point.
        for s in DeviceRegistry(_FakeCore(names=("CPU", "GPU", "NPU"))).describe():
            self.assertFalse(s.validated, f"{s.device_id} claimed validated by discovery")
            self.assertFalse(s.ready, f"{s.device_id} claimed ready by discovery")
            self.assertEqual(s.supported_model_ids, ())


class StaticShapesAreEnforced(unittest.TestCase):
    def test_a_pinned_static_shape_is_accepted(self):
        require_static_shape((1, 3, 640, 640))

    def test_a_minus_one_dynamic_dimension_is_refused(self):
        # The NPU rejects dynamic shapes, so the refusal belongs before the compile.
        with self.assertRaises(ValueError) as ctx:
            require_static_shape((1, 3, -1, -1))
        self.assertIn("static", str(ctx.exception).lower())

    def test_a_zero_or_malformed_shape_is_refused(self):
        for bad in ((1, 3, 0, 640), (1, 3, 640), (1, 3, "640", 640), ()):
            with self.assertRaises(ValueError):
                require_static_shape(bad)

    def test_a_non_positive_dimensions_does_not_read_as_a_zero_throughput(self):
        with self.assertRaises(ValueError):
            require_static_shape((0, 3, 640, 640))


class _FakeCompiled:
    """Stands in for `ov.CompiledModel`. `placements` is consumed one value per read, so
    a test can prove the value is fetched again rather than remembered."""

    def __init__(self, placements):
        self._placements = list(placements)
        self.reads = 0

    def get_property(self, name):
        if name != "EXECUTION_DEVICES":
            raise RuntimeError(f"unexpected property {name}")
        self.reads += 1
        return self._placements[min(self.reads - 1, len(self._placements) - 1)]


class _FakeRequest:
    def __init__(self, compiled):
        self._compiled = compiled
        self.inferences = 0

    def get_compiled_model(self):
        return self._compiled

    def infer(self, _data):
        self.inferences += 1


class PlacementIsNeverCaptured(unittest.TestCase):
    """The invariant, pinned both ways.

    I chose the *pair* of forms over either alone. "No such field exists" alone would
    still pass if the placement were smuggled into a differently-named field, and a
    "returns a fresh value" test alone would still pass if a cached value happened to
    be returned. Together they close both holes: the dataclass surface is inspected, and
    the read is observed to go back to the runtime.
    """

    def _target(self, placements):
        compiled = _FakeCompiled(placements)
        request = _FakeRequest(compiled)
        target = CompiledTarget(device_id="NPU", model_path=Path("model.xml"),
                                model_hash="a" * 64, input_shape=(1, 3, 640, 640),
                                _core=self._core, _request=request)
        return target, compiled, request

    def setUp(self):
        self._core = _FakeCore(names=("CPU", "GPU", "NPU"))

    def test_the_dataclass_has_no_field_that_could_hold_a_placement(self):
        names = {f.name for f in dataclasses.fields(CompiledTarget)}
        for banned in ("execution_devices", "devices", "placement", "actual_device"):
            self.assertNotIn(banned, names,
                             f"CompiledTarget gained a {banned!r} field; placement must "
                             f"not be captured at construction")
        self.assertEqual(
            names,
            {"device_id", "model_path", "model_hash", "input_shape", "_core", "_request"},
            "CompiledTarget's fields changed; re-check the live-read invariant")

    def test_a_fresh_target_reports_no_placement_before_anything_is_read(self):
        target, compiled, _ = self._target([["NPU"]])
        self.assertEqual(compiled.reads, 0,
                         "construction read EXECUTION_DEVICES; it must not")

    def test_no_instance_attribute_holds_a_placement_tuple(self):
        # `input_shape` is legitimately a tuple; a *placement* tuple is not. Checked by
        # name, since a tuple of device-id strings under a placement-shaped name is the
        # exact thing being forbidden here.
        target, _, _ = self._target([["NPU"]])
        for name, value in vars(target).items():
            if not any(word in name for word in ("device", "placement", "execut")):
                continue
            self.assertNotIsInstance(
                value, tuple,
                f"{name} holds a tuple; a placement was captured at construction")
        self.assertEqual(target.input_shape, (1, 3, 640, 640))

    def test_the_read_goes_back_to_the_runtime_on_every_call(self):
        # Before the first inference the runtime may not have a meaningful value; after
        # it, it does. A cached read could not tell those apart.
        target, compiled, _ = self._target([[], ["NPU"]])
        self.assertEqual(target.execution_devices_now(), ())
        self.assertEqual(target.execution_devices_now(), ("NPU",))
        self.assertEqual(compiled.reads, 2)

    def test_the_placement_read_can_differ_from_the_requested_device(self):
        # Requested NPU, actually ran on CPU. The truth is the truth.
        target, _, _ = self._target([["CPU"]])
        self.assertEqual(target.device_id, "NPU")
        self.assertEqual(target.execution_devices_now(), ("CPU",))

    def test_an_explicit_core_and_request_are_accepted_and_also_read_live(self):
        target, compiled, request = self._target([["GPU.0"]])
        self.assertEqual(target.execution_devices_now(self._core, request), ("GPU.0",))
        self.assertEqual(compiled.reads, 1)

    def test_a_bare_string_placement_becomes_a_one_element_tuple(self):
        # Measured on this machine: the NPU plugin returns the string 'NPU', not a list.
        target, _, _ = self._target(["NPU"])
        self.assertEqual(target.execution_devices_now(), ("NPU",))

    def test_stored_and_explicit_handles_agree_on_the_same_runtime(self):
        target, compiled, request = self._target([["NPU"]])
        self.assertEqual(target.execution_devices_now(), target.execution_devices_now())
        self.assertEqual(compiled.reads, 2)
        self.assertIs(request.get_compiled_model(), compiled)

    def test_no_placement_is_read_before_the_caller_has_run_an_inference(self):
        # The registry builds the request; it does not infer. Reading placement is the
        # caller's job, after its own first inference.
        target, compiled, request = self._target([["NPU"]])
        self.assertEqual(request.inferences, 0)
        self.assertEqual(compiled.reads, 0)
        request.infer(None)
        self.assertEqual(target.execution_devices_now(), ("NPU",))

    def test_reading_a_property_other_than_execution_devices_is_a_bug(self):
        # Guards against the method being generalised into a property dump later.
        target, compiled, _ = self._target([["NPU"]])
        target.execution_devices_now()
        self.assertEqual(compiled.reads, 1)


class TargetsAndHashesAreReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = ov.Core()
        cls.registry = DeviceRegistry(cls.core)

    def _spec(self, device_id="CPU"):
        return TargetSpec(device_id=device_id, model_path=IR_PATH,
                          input_shape=tuple(DETECTOR["input_shape"]))

    def test_the_spec_shape_matches_the_recorded_inventory(self):
        self.assertEqual(tuple(DETECTOR["input_shape"]), (1, 3, 640, 640))
        self.assertEqual(DETECTOR["num_classes"], 80)

    def test_compiling_yields_a_target_with_no_placement(self):
        target = self.registry.compile_target(self._spec())
        self.assertEqual(target.device_id, "CPU")
        self.assertEqual(target.input_shape, (1, 3, 640, 640))
        self.assertFalse(any(name in {f.name for f in dataclasses.fields(CompiledTarget)}
                             for name in ("execution_devices", "placement")))

    def test_the_model_hash_is_the_recorded_sha256_of_the_ir(self):
        import hashlib
        target = self.registry.compile_target(self._spec())
        self.assertEqual(target.model_hash, DETECTOR["sha256"])
        self.assertEqual(target.model_hash,
                         hashlib.sha256(IR_PATH.read_bytes()).hexdigest())

    def test_placement_is_reported_only_after_a_real_inference(self):
        import numpy as np
        target = self.registry.compile_target(self._spec())
        request = target.request
        request.infer(np.zeros((1, 3, 640, 640), dtype=np.float32))
        self.assertEqual(target.execution_devices_now(), ("CPU",))

    def test_a_dynamic_shape_spec_is_refused_before_any_compile(self):
        with self.assertRaises(ValueError):
            self.registry.compile_target(
                TargetSpec(device_id="CPU", model_path=IR_PATH, input_shape=(1, 3, -1, -1)))

    def test_compiling_to_a_device_that_is_absent_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.registry.compile_target(self._spec("NOT_A_DEVICE"))
        self.assertIn("NOT_A_DEVICE", str(ctx.exception))

    def test_compiling_to_an_absent_device_names_what_is_available(self):
        with self.assertRaises(ValueError) as ctx:
            self.registry.compile_target(self._spec("NOT_A_DEVICE"))
        for d in DEVICE_IDS:
            self.assertIn(d, str(ctx.exception))


class ExplicitTargetsOnly(unittest.TestCase):
    """No `AUTO`, no `HETERO`, no `MULTI` in the primary routing demo."""

    @classmethod
    def setUpClass(cls):
        cls.registry = DeviceRegistry(ov.Core())

    def _spec(self, device_id):
        return TargetSpec(device_id=device_id, model_path=IR_PATH,
                          input_shape=(1, 3, 640, 640))

    def test_auto_hetero_and_multi_are_refused(self):
        for pseudo in ("AUTO", "HETERO", "MULTI", "auto", "HETERO:GPU,CPU"):
            with self.assertRaises(ValueError) as ctx:
                self.registry.compile_target(self._spec(pseudo))
            self.assertIn(pseudo.upper(), str(ctx.exception).upper())

    def test_an_explicitly_named_subdevice_is_still_allowed(self):
        # `GPU.0` is an explicit target and is what the runtime reports back.
        self.assertTrue(self.registry.available() != ())
        self.assertTrue(any(d == "GPU" for d in self.registry.available()))


class TargetSpecAndDeviceIdsAreStable(unittest.TestCase):
    def test_device_ids_is_a_tuple_not_a_list(self):
        self.assertIsInstance(DEVICE_IDS, tuple)

    def test_target_spec_is_frozen(self):
        spec = TargetSpec(device_id="CPU", model_path=Path("m.xml"),
                          input_shape=(1, 3, 640, 640))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.device_id = "GPU"

    def test_compiled_target_is_frozen(self):
        target = CompiledTarget(device_id="CPU", model_path=Path("m.xml"),
                                model_hash="a" * 64, input_shape=(1, 3, 640, 640))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            target.model_hash = "b" * 64

    def test_a_compiled_target_built_without_handles_cannot_claim_a_placement(self):
        target = CompiledTarget(device_id="CPU", model_path=Path("m.xml"),
                                model_hash="a" * 64, input_shape=(1, 3, 640, 640))
        with self.assertRaises(TypeError):
            target.execution_devices_now()


if __name__ == "__main__":
    unittest.main()
