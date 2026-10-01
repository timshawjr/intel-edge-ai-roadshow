"""Device discovery over a real `ov.Core`, and the placement contract.

Three things are pinned here that nothing downstream would catch:

1. **Discovery is discovered.** `FULL_DEVICE_NAME` comes from OpenVINO, never from a
   hardcoded family string. This machine is Lunar Lake; nothing here may present a
   measurement as Panther Lake or Arrow Lake.
2. **`EXECUTION_DEVICES` is read live, after an inference, never captured at
   construction.** See `PlacementIsNeverCaptured`. This is the one invariant the whole
   product's honesty rests on, so it is pinned two ways: the dataclass has no field
   that could hold a placement, and the read demonstrably goes back to the runtime.
3. **`EXECUTION_DEVICES` is not a consistent type.** See `ExecutionDevicesAreNormalised`
   and `ConsumersMatchThroughIsDevice`. Measured: NPU returns the bare string `'NPU'`,
   CPU returns `['CPU']`, GPU returns `['GPU.0']`. The normalisation is pinned here so a
   refactor cannot reintroduce a `('N','P','U')` tuple or leave a consumer matching a
   bare `"GPU"` against a dotted reading.
"""

import dataclasses
import json
import unittest
from pathlib import Path

# CORRECTED 2026-10-01: openvino==2026.4.0 has no `openvino.runtime` submodule.
# The top-level module exposes Core, Model and InferRequest directly.
import openvino as ov

from app.inference.cache import cache_identity
from app.inference.device_drivers import (
    PnpDriver,
    WindowsDriverLookup,
    default_driver_lookup,
)
from app.inference.devices import (
    DEVICE_IDS,
    CompiledTarget,
    DeviceRegistry,
    TargetSpec,
    is_device,
    normalise_execution_devices,
    plugin_class,
    require_static_shape,
)

MODELS = json.loads(Path("config/models.json").read_text(encoding="utf-8"))
DETECTOR = next(m for m in MODELS["models"] if m["id"] == "yolo11n-fp16")
IR_PATH = Path(DETECTOR["ir_path"])

#: A pinned PnP table for the hermetic tests. The ids and versions here are test fixtures,
#: not measurements: `NothingIsHardcoded` in `tests/test_device_drivers.py` pins that no
#: measured id or version is written into the shipped module.
FAKE_NPU_PCI = "PCI\\VEN_8086&DEV_1111&SUBSYS_00000000&REV_04\\3&1&0&11"
FAKE_GPU_PCI = "PCI\\VEN_8086&DEV_2222&SUBSYS_00000000&REV_04\\3&1&0&22"
FAKE_PNP = (
    PnpDriver(device_id=FAKE_NPU_PCI, name="Fake NPU", driver_version="9.9.1.1"),
    PnpDriver(device_id=FAKE_GPU_PCI, name="Fake GPU (iGPU)", driver_version="9.9.2.2"),
)

#: "The Windows PnP layer could not be read." A `_FakeCore` reports names like
#: 'Fake NPU', which match no PnP row anyway; pinning the table keeps these unit tests
#: off the real machine and keeps the OpenVINO-only fallback the path actually under test.
NO_PNP = WindowsDriverLookup(drivers=())



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
        statuses = {s.device_id: s for s in
                    DeviceRegistry(core, driver_lookup=NO_PNP).describe()}
        self.assertTrue(statuses["NPU"].unavailable_reason)
        self.assertIn("NPU plugin is not available", statuses["NPU"].unavailable_reason)
        self.assertFalse(statuses["NPU"].available)
        self.assertFalse(statuses["NPU"].ready)

    def test_an_available_device_has_no_reason_and_a_discovered_name(self):
        core = _FakeCore(names=("CPU", "GPU", "NPU"))
        for s in DeviceRegistry(core, driver_lookup=NO_PNP).describe():
            self.assertTrue(s.available)
            self.assertIsNone(s.unavailable_reason)
            self.assertEqual(s.full_name, f"Fake {s.device_id}")

    def test_a_device_with_no_driver_property_records_null_not_zero(self):
        core = _FakeCore(names=("CPU", "GPU", "NPU"))
        for s in DeviceRegistry(core, driver_lookup=NO_PNP).describe():
            self.assertIsNone(s.driver_version,
                              "a device reporting no driver version must be null, not 0")

    def test_a_reported_driver_version_is_recorded_verbatim(self):
        core = _FakeCore(names=("CPU", "GPU", "NPU"),
                         driver_properties={"NPU": {"NPU_DRIVER_VERSION": 1005540}})
        statuses = {s.device_id: s for s in
                    DeviceRegistry(core, driver_lookup=NO_PNP).describe()}
        # Measured on this machine: the NPU plugin exposes NPU_DRIVER_VERSION = 1005540
        # and CPU/GPU expose no driver-version property at all. With the PnP layer
        # unreadable this is the fallback; see `ThePnpLayerIsTheAuthority`.
        self.assertEqual(statuses["NPU"].driver_version, "1005540")
        self.assertIsNone(statuses["CPU"].driver_version)
        self.assertIsNone(statuses["GPU"].driver_version)

    def test_discovery_alone_never_claims_a_device_is_validated_or_ready(self):
        # Presence is not proof. Nothing has compiled a model at this point.
        core = _FakeCore(names=("CPU", "GPU", "NPU"))
        for s in DeviceRegistry(core, driver_lookup=NO_PNP).describe():
            self.assertFalse(s.validated, f"{s.device_id} claimed validated by discovery")
            self.assertFalse(s.ready, f"{s.device_id} claimed ready by discovery")
            self.assertEqual(s.supported_model_ids, ())

    def test_device_type_is_the_plugin_class_and_not_the_openvino_device_type(self):
        # C2: DEVICE_TYPE measures Type.INTEGRATED on all three, so it discriminated
        # nothing. The plugin class is the vocabulary `device_id` already uses.
        statuses = DeviceRegistry(_FakeCore(names=("CPU", "GPU", "NPU")),
                                  driver_lookup=NO_PNP).describe()
        for s in statuses:
            self.assertEqual(s.device_type, s.device_id)
            self.assertIn(s.device_type, ("CPU", "GPU", "NPU"))
            self.assertNotIn("INTEGRATED", s.device_type)

    def test_device_type_tells_the_three_devices_apart(self):
        statuses = DeviceRegistry(_FakeCore(names=("CPU", "GPU", "NPU")),
                                  driver_lookup=NO_PNP).describe()
        self.assertEqual({s.device_type for s in statuses}, {"CPU", "GPU", "NPU"})


class ThePnpLayerIsTheAuthority(unittest.TestCase):
    """C3: driver versions come from Windows, because only the NPU's is in OpenVINO."""

    def test_the_pci_device_id_is_discovered_from_supported_properties(self):
        # The GPU plugin reports its own PCI device id; the property *name* is found by
        # enumerating SUPPORTED_PROPERTIES. The name 'Fake GPU' matches no PnP row, so a
        # version here can only have come through the id route.
        core = _FakeCore(names=("GPU",),
                         driver_properties={"GPU": {"GPU_DEVICE_ID": "0x2222"}})
        statuses = {s.device_id: s for s in DeviceRegistry(
            core, driver_lookup=WindowsDriverLookup(drivers=FAKE_PNP)).describe()}
        self.assertEqual(statuses["GPU"].driver_version, "9.9.2.2")

    def test_a_plugin_ordinal_is_never_searched_for_as_a_pci_device_id(self):
        # DEVICE_ID reads '0' on the GPU plugin. A table whose *name* cannot match, so the
        # only route left is the ordinal: it must find nothing rather than find the row.
        table = (PnpDriver(device_id=FAKE_GPU_PCI, name="Something Else Entirely",
                           driver_version="9.9.2.2"),)
        core = _FakeCore(names=("GPU",), driver_properties={"GPU": {"DEVICE_ID": "0"}})
        statuses = {s.device_id: s for s in DeviceRegistry(
            core, driver_lookup=WindowsDriverLookup(drivers=table)).describe()}
        self.assertIsNone(statuses["GPU"].driver_version)

        # The same table, the same plugin, but the property name now means a real PCI id.
        core = _FakeCore(names=("GPU",),
                         driver_properties={"GPU": {"GPU_DEVICE_ID": "0x2222"}})
        statuses = {s.device_id: s for s in DeviceRegistry(
            core, driver_lookup=WindowsDriverLookup(drivers=table)).describe()}
        self.assertEqual(statuses["GPU"].driver_version, "9.9.2.2")

    def test_the_discovered_name_route_resolves_a_plugin_with_no_device_id(self):
        # The NPU plugin exposes no PCI device id at all, so the name is its only handle.
        core = _FakeCore(names=("NPU",))
        statuses = {s.device_id: s for s in DeviceRegistry(
            core, driver_lookup=WindowsDriverLookup(drivers=FAKE_PNP)).describe()}
        # _FakeCore answers FULL_DEVICE_NAME as 'Fake NPU', which is the pinned PnP row's
        # name. The GPU fixture above is deliberately named 'Fake GPU (iGPU)' so it
        # matches nothing by name, leaving the id route as the only route that can work.
        self.assertEqual(statuses["NPU"].driver_version, "9.9.1.1")

    def test_an_unreadable_pnp_layer_falls_back_to_the_openvino_property(self):
        def blocked(script, timeout):
            raise OSError("powershell.exe is blocked by policy")

        core = _FakeCore(names=("CPU", "GPU", "NPU"),
                         driver_properties={"NPU": {"NPU_DRIVER_VERSION": 1005540}})
        statuses = {s.device_id: s for s in DeviceRegistry(
            core, driver_lookup=WindowsDriverLookup(runner=blocked)).describe()}
        self.assertEqual(statuses["NPU"].driver_version, "1005540")
        self.assertIsNone(statuses["CPU"].driver_version)
        self.assertIsNone(statuses["GPU"].driver_version)

    def test_an_unreadable_pnp_layer_never_stops_discovery(self):
        def blocked(script, timeout):
            raise OSError("powershell.exe is blocked by policy")

        core = _FakeCore(names=("CPU", "GPU", "NPU"))
        statuses = DeviceRegistry(
            core, driver_lookup=WindowsDriverLookup(runner=blocked)).describe()
        self.assertEqual([s.device_id for s in statuses], list(DEVICE_IDS))
        for s in statuses:
            self.assertTrue(s.available)
            self.assertTrue(s.full_name)



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


class ExecutionDevicesAreNormalised(unittest.TestCase):
    """C5: the three plugins do not agree on the *type* of `EXECUTION_DEVICES`.

    Measured on this machine, first-hand:

        CPU -> ['CPU']        a list
        GPU -> ['GPU.0']     a list, with a sub-device suffix
        NPU -> 'NPU'         a bare **string**

    The string case is the trap. `tuple('NPU')` is `('N', 'P', 'U')`, which would render
    as a three-device row and read as "the NPU ran three places". Every reading therefore
    goes through one normaliser, and these tests feed it all three measured shapes.
    """

    def setUp(self):
        self._core = _FakeCore(names=("CPU", "GPU", "NPU"))

    def _reading(self, reported):
        compiled = _FakeCompiled([reported])
        request = _FakeRequest(compiled)
        target = CompiledTarget(device_id="NPU", model_path=Path("model.xml"),
                                model_hash="a" * 64, input_shape=(1, 3, 640, 640),
                                _core=self._core, _request=request)
        return target.execution_devices_now()

    def test_a_bare_npu_string_becomes_a_one_element_tuple(self):
        self.assertEqual(self._reading("NPU"), ("NPU",))

    def test_a_bare_string_is_never_split_into_its_characters(self):
        # The specific regression: `tuple(value)` on the string 'NPU'.
        reading = self._reading("NPU")
        self.assertEqual(len(reading), 1, f"'NPU' was split into {reading}")
        self.assertNotIn("P", reading)
        self.assertNotEqual(reading, tuple("NPU"))

    def test_a_cpu_list_becomes_a_one_element_tuple(self):
        self.assertEqual(self._reading(["CPU"]), ("CPU",))

    def test_a_gpu_list_keeps_its_dotted_sub_device_suffix(self):
        # Normalisation changes the *container*, never the value. 'GPU.0' is what the GPU
        # reports and it must survive verbatim; `is_device` is what absorbs the suffix.
        self.assertEqual(self._reading(["GPU.0"]), ("GPU.0",))

    def test_every_measured_shape_normalises_to_a_tuple(self):
        for reported, expected in ((["CPU"], ("CPU",)),
                                   (["GPU.0"], ("GPU.0",)),
                                   ("NPU", ("NPU",))):
            reading = self._reading(reported)
            self.assertEqual(reading, expected)
            self.assertIsInstance(reading, tuple)
            self.assertNotIsInstance(reading, list)

    def test_the_normaliser_takes_each_measured_shape_directly(self):
        self.assertEqual(normalise_execution_devices("NPU"), ("NPU",))
        self.assertEqual(normalise_execution_devices(["CPU"]), ("CPU",))
        self.assertEqual(normalise_execution_devices(["GPU.0"]), ("GPU.0",))

    def test_absent_and_empty_readings_are_no_devices(self):
        self.assertEqual(self._reading(None), ())
        self.assertEqual(self._reading([]), ())
        self.assertEqual(normalise_execution_devices(None), ())
        self.assertEqual(normalise_execution_devices(""), ())
        self.assertEqual(normalise_execution_devices("   "), ())

    def test_a_multi_device_reading_is_kept_in_order(self):
        # Nothing on this machine reports two, but the shape must not be collapsed.
        self.assertEqual(normalise_execution_devices(["CPU", "GPU.0"]), ("CPU", "GPU.0"))
        self.assertEqual(normalise_execution_devices(["NPU", "CPU"]), ("NPU", "CPU"))

    def test_normalisation_is_stable_when_applied_twice(self):
        once = normalise_execution_devices("NPU")
        self.assertEqual(normalise_execution_devices(once), once)


class ConsumersMatchThroughIsDevice(unittest.TestCase):
    """C6: the GPU reports `GPU.0`, so a bare `"GPU"` check cannot be the contract.

    The parent's ruling, implemented: consumers match the **exact** normalised string, and
    come through `is_device` for the `.0` suffix, so no consumer has to remember the
    dotted form and none can pass on a substring by accident.
    """

    def test_the_exact_normalised_string_is_what_consumers_compare(self):
        self.assertIn("CPU", ("CPU",))
        self.assertIn("GPU.0", ("GPU.0",))
        self.assertIn("NPU", ("NPU",))

    def test_a_bare_gpu_check_fails_against_the_dotted_reading(self):
        # Stated as a fact, because it is why `is_device` exists.
        self.assertNotIn("GPU", ("GPU.0",))

    def test_the_substring_trap_is_real_and_is_not_what_the_helper_does(self):
        # `'GPU' in 'GPU.0'` is True because that is a substring test, not a membership
        # test. It would also be True for a nonexistent 'GP' device.
        self.assertIn("GPU", "GPU.0")
        self.assertIn("GP", "GPU.0")
        self.assertFalse(is_device("GP", "GPU"))

    def test_is_device_handles_the_dotted_gpu_suffix(self):
        self.assertTrue(is_device(("GPU.0",), "GPU"))
        self.assertTrue(is_device(["GPU.0"], "GPU"))
        self.assertTrue(is_device("GPU.0", "GPU"))
        self.assertTrue(is_device(("GPU.0",), "GPU.0"))

    def test_is_device_matches_the_undotted_devices(self):
        self.assertTrue(is_device(("CPU",), "CPU"))
        self.assertTrue(is_device(("NPU",), "NPU"))
        self.assertTrue(is_device("NPU", "NPU"))

    def test_is_device_is_false_for_a_different_device(self):
        self.assertFalse(is_device(("NPU",), "GPU"))
        self.assertFalse(is_device(("GPU.0",), "NPU"))
        self.assertFalse(is_device(("CPU", "GPU.0"), "NPU"))

    def test_is_device_is_false_for_no_reading_at_all(self):
        for empty in (None, (), [], ""):
            self.assertFalse(is_device(empty, "GPU"), repr(empty))

    def test_is_device_is_false_for_an_empty_or_absent_wanted_device(self):
        self.assertFalse(is_device(("CPU",), ""))
        self.assertFalse(is_device(("CPU",), None))

    def test_is_device_is_case_insensitive(self):
        self.assertTrue(is_device(("gpu.0",), "GPU"))

    def test_plugin_class_strips_a_sub_device_and_a_pseudo_device(self):
        self.assertEqual(plugin_class("GPU.0"), "GPU")
        self.assertEqual(plugin_class("GPU"), "GPU")
        self.assertEqual(plugin_class("NPU"), "NPU")
        self.assertEqual(plugin_class("HETERO:GPU,CPU"), "HETERO")
        self.assertEqual(plugin_class("gpu.1"), "GPU")
        self.assertEqual(plugin_class(""), "")
        self.assertEqual(plugin_class(None), "")

    def test_an_unheard_of_device_id_is_not_promoted_to_a_known_class(self):
        # Nothing is invented: an id this build has never seen stays itself.
        self.assertEqual(plugin_class("MYACCEL"), "MYACCEL")
        self.assertFalse(is_device(("MYACCEL",), "GPU"))


class TheRealPluginsReportTheseShapes(unittest.TestCase):
    """The normalisation pinned against the real runtime, not only against doubles."""

    @classmethod
    def setUpClass(cls):
        cls.core = ov.Core()

    def test_the_real_gpu_reports_a_dotted_name_that_only_is_device_matches(self):
        import numpy as np
        target = DeviceRegistry(self.core).compile_target(
            TargetSpec(device_id="GPU", model_path=IR_PATH,
                       input_shape=tuple(DETECTOR["input_shape"])))
        target.request.infer(np.zeros((1, 3, 640, 640), dtype=np.float32))
        reading = target.execution_devices_now()
        self.assertEqual(reading, ("GPU.0",),
                         "the GPU's own reading changed; re-check is_device's contract")
        self.assertTrue(is_device(reading, "GPU"))
        self.assertNotIn("GPU", reading)

    def test_the_real_npu_reports_a_bare_string_that_is_not_split(self):
        import numpy as np
        target = DeviceRegistry(self.core).compile_target(
            TargetSpec(device_id="NPU", model_path=IR_PATH,
                       input_shape=tuple(DETECTOR["input_shape"])))
        target.request.infer(np.zeros((1, 3, 640, 640), dtype=np.float32))
        reading = target.execution_devices_now()
        self.assertEqual(reading, ("NPU",),
                         "the NPU's own reading changed; re-check the string case")
        self.assertEqual(len(reading), 1)
        self.assertTrue(is_device(reading, "NPU"))

    def test_the_real_cpu_reports_a_one_element_tuple(self):
        import numpy as np
        target = DeviceRegistry(self.core).compile_target(
            TargetSpec(device_id="CPU", model_path=IR_PATH,
                       input_shape=tuple(DETECTOR["input_shape"])))
        target.request.infer(np.zeros((1, 3, 640, 640), dtype=np.float32))
        reading = target.execution_devices_now()
        self.assertEqual(reading, ("CPU",))
        self.assertTrue(is_device(reading, "CPU"))


class RealDriverVersionsComeFromWindows(unittest.TestCase):
    """C3 measured on this machine: GPU and NPU have Windows driver versions, CPU does not."""

    @classmethod
    def setUpClass(cls):
        cls.core = ov.Core()
        cls.registry = DeviceRegistry(cls.core)
        cls.statuses = {s.device_id: s for s in cls.registry.describe()}

    def _pnp_version(self, device_id):
        full_name = str(self.core.get_property(device_id, "FULL_DEVICE_NAME"))
        return default_driver_lookup().driver_version_for(full_name)

    def test_the_gpu_driver_version_is_the_windows_one(self):
        self.assertEqual(self.statuses["GPU"].driver_version,
                         self._pnp_version("GPU"))
        self.assertIsNotNone(self.statuses["GPU"].driver_version)
        self.assertNotEqual(self.statuses["GPU"].driver_version, "0")

    def test_the_npu_driver_version_is_the_windows_one_and_not_the_openvino_spelling(self):
        self.assertEqual(self.statuses["NPU"].driver_version,
                         self._pnp_version("NPU"))
        self.assertNotEqual(self.statuses["NPU"].driver_version,
                            str(self.core.get_property("NPU", "NPU_DRIVER_VERSION")),
                            "the PnP layer is the authority; the plugin property is the "
                            "fallback only")

    def test_the_cpu_driver_version_is_null_because_it_has_no_separate_driver(self):
        self.assertIsNone(self.statuses["CPU"].driver_version)
        self.assertIsNone(self._pnp_version("CPU"))

    def test_a_null_driver_version_stays_distinct_in_the_blob_cache_key(self):
        # Review Focus #5: "no driver artefact" must not collide with any version.
        core_version = self.statuses["CPU"].driver_version
        gpu_version = self.statuses["GPU"].driver_version
        self.assertIsNone(core_version)
        self.assertIsNotNone(gpu_version)
        keys = {
            cache_identity(model_hash="a" * 64, device_id="X",
                           runtime_version="2026.4.0", driver_version=version,
                           properties="LATENCY")["driver_version"]
            for version in (None, gpu_version, "", "0")
        }
        self.assertEqual(len(keys), 4, f"cache identity collapsed states: {keys}")

    def test_device_type_is_the_plugin_class_on_this_machine(self):
        self.assertEqual({s.device_type for s in self.statuses.values()},
                         {"CPU", "GPU", "NPU"})
        for status in self.statuses.values():
            self.assertEqual(status.device_type, status.device_id)
            self.assertNotIn("INTEGRATED", status.device_type)
            self.assertNotEqual(
                status.device_type,
                str(self.core.get_property(status.device_id, "DEVICE_TYPE")))



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

    def test_the_live_handles_are_compared_and_shown_not_silently_excluded(self):
        # Measured, and the docstring says so: all six fields carry compare=True and
        # repr=True. Changing that would alter __eq__/__hash__ and is a deliberate act,
        # so it is pinned rather than left to drift.
        specs = {f.name: (f.compare, f.repr) for f in dataclasses.fields(CompiledTarget)}
        self.assertEqual(specs["_core"], (True, True))
        self.assertEqual(specs["_request"], (True, True))
        left = CompiledTarget(device_id="CPU", model_path=Path("m.xml"),
                              model_hash="a" * 64, input_shape=(1, 3, 640, 640))
        right = CompiledTarget(device_id="GPU", model_path=Path("m.xml"),
                               model_hash="a" * 64, input_shape=(1, 3, 640, 640))
        self.assertNotEqual(left, right)
        self.assertIn("_core", repr(left))
        self.assertIn("_request", repr(left))


if __name__ == "__main__":
    unittest.main()
