"""Driver versions read from the Windows PnP layer, and what happens when that fails.

OpenVINO exposes `NPU_DRIVER_VERSION` on the NPU plugin only. That is a property of the
*accessor*, not of the hardware: Windows versions the GPU driver too and says so in
`Win32_PnPSignedDriver`. Two things are pinned here, and the second matters more than the
first:

1. **The lookup works, and nothing about it is hardcoded.** A matching PCI device id
   returns the driver's `DriverVersion`; a non-matching one returns `None`; a lookup that
   cannot be performed at all returns `None` rather than raising.
2. **Every failure mode degrades to `None`.** No PowerShell, execution blocked by policy,
   a non-zero exit, unparsable output, no match, an ambiguous match, a driver that reports
   no version. A wrong driver version is a plausible-looking string, which is the single
   most dangerous thing this module could produce, so it must never produce one.

The whole class of pins is hermetic: `WindowsDriverLookup` takes the PnP table or the
enumeration runner, so no test here depends on the machine it runs on. One class
(`TheRealPnpLayerOnThisMachine`) deliberately does depend on the machine, because the
claim being pinned is that the lookup resolves this machine's real devices.
"""

import json
import re
import subprocess
import unittest

import openvino as ov

from app.inference.device_drivers import (
    PnpDriver,
    WindowsDriverLookup,
    _ENUMERATE_SCRIPT,
    _parse_enumeration,
    default_driver_lookup,
    driver_matches_device_id,
    looks_like_a_pci_device_id,
    normalise_device_name,
    windows_driver_version,
)

# A stand-in for the measured shape of `Win32_PnPSignedDriver` rows. The ids and versions
# are fabricated *for the test table only*; nothing here writes them into app code, and
# `NothingIsHardcoded` pins that.
NPU_PCI = "PCI\\VEN_8086&DEV_643E&SUBSYS_88F21043&REV_04\\3&11583659&0&58"
GPU_PCI = "PCI\\VEN_8086&DEV_64A0&SUBSYS_88F21043&REV_04\\3&11583659&0&10"
FAKE_TABLE = (
    PnpDriver(device_id=NPU_PCI, name="Intel(R) AI Boost", driver_version="32.0.100.5540"),
    PnpDriver(device_id=GPU_PCI, name="Intel(R) Arc(TM) 140V GPU (16GB)",
              driver_version="32.0.101.6737"),
    # A second, unrelated Intel function, so "the first PCI row" cannot pass by accident.
    PnpDriver(device_id="PCI\\VEN_8086&DEV_A838&SUBSYS_00000000&REV_10\\3&1&0&E0",
              name="PCI Express Root Port", driver_version="10.0.26100.9278"),
    # A driver with no version reported at all.
    PnpDriver(device_id="PCI\\VEN_8086&DEV_FFFF&SUBSYS_00000000&REV_10\\3&1&0&E1",
              name="Fabric With No Version", driver_version=None),
)

EMPTY_TABLE: list[PnpDriver] = []


def _lookup(**kwargs):
    """A lookup pinned to a table, so no subprocess and no machine state is involved."""
    kwargs.setdefault("drivers", FAKE_TABLE)
    return WindowsDriverLookup(**kwargs)


class _NeverRuns:
    """A runner that must never be called - used to prove a lookup short-circuits."""

    def __init__(self):
        self.calls = 0

    def __call__(self, script, timeout):
        self.calls += 1
        raise AssertionError("the enumeration must not run for this test")


class AMatchingDeviceIdReturnsItsVersion(unittest.TestCase):
    """The three pins the parent asked for, first: a match returns a version."""

    def test_a_matching_device_id_returns_the_signed_driver_version(self):
        self.assertEqual(windows_driver_version([GPU_PCI], lookup=_lookup()),
                         "32.0.101.6737")

    def test_a_non_matching_id_returns_none(self):
        self.assertIsNone(windows_driver_version(
            ["PCI\\VEN_8086&DEV_0000&SUBSYS_00000000"], lookup=_lookup()))

    def test_a_total_lookup_failure_returns_none_rather_than_raising(self):
        def explode(script, timeout):
            raise OSError("powershell.exe is blocked by policy")

        lookup = WindowsDriverLookup(runner=explode)
        self.assertEqual(lookup.drivers, ())
        self.assertIsNone(windows_driver_version([GPU_PCI], lookup=lookup))
        # Nothing matched, so nothing was invented - not "0", not "unknown".
        self.assertIsNone(lookup.driver_version_for("Intel(R) Arc(TM) 140V GPU (16GB)"))


class DeviceIdsMatchInEveryFormACallerMightHold(unittest.TestCase):
    """`GPU_DEVICE_ID` reads `'0x64a0'`, a hardware id reads `'PCI\\VEN_...&DEV_64A0'`,
    and a caller may hold the full PnP instance id. All three must resolve, or a caller
    has to know which spelling the PnP layer happens to use today."""

    def test_a_bare_hex_device_id_resolves(self):
        self.assertEqual(
            windows_driver_version(["0x64a0"], lookup=_lookup()), "32.0.101.6737")
        self.assertEqual(
            windows_driver_version(["64A0"], lookup=_lookup()), "32.0.101.6737")
        self.assertEqual(
            windows_driver_version(["0X64A0"], lookup=_lookup()), "32.0.101.6737")
        self.assertEqual(
            windows_driver_version(["DEV_64A0"], lookup=_lookup()), "32.0.101.6737")

    def test_a_hardware_id_prefix_resolves(self):
        self.assertEqual(
            windows_driver_version(["PCI\\VEN_8086&DEV_64A0"], lookup=_lookup()),
            "32.0.101.6737")

    def test_a_full_pnp_instance_id_resolves(self):
        self.assertEqual(windows_driver_version([GPU_PCI], lookup=_lookup()),
                         "32.0.101.6737")

    def test_matching_is_case_insensitive(self):
        self.assertEqual(
            windows_driver_version(["pci\\ven_8086&dev_64a0"], lookup=_lookup()),
            "32.0.101.6737")

    def test_a_hardware_id_carrying_a_different_vendor_does_not_match(self):
        # A `PCI\VEN_...&DEV_...` id is matched as a prefix, so the vendor is part of the
        # match. A wrong-vendor id cannot borrow this machine's GPU version.
        self.assertIsNone(windows_driver_version(
            ["PCI\\VEN_10DE&DEV_64A0"], lookup=_lookup()))
        self.assertIsNone(windows_driver_version(
            ["PCI\\VEN_8086&DEV_6430"], lookup=_lookup()))

    def test_a_bare_device_id_matches_the_device_id_field_alone(self):
        # Stated plainly, because it is a limitation and not a guarantee: a bare
        # `'0x64a0'` is matched against the `DEV_` field and nothing else, because that
        # is all a plugin property such as `GPU_DEVICE_ID` can hand over - it carries no
        # vendor. The two things that make it safe are pinned separately: the device-id
        # field was unique across all 28 rows read on this machine, and a bare id that
        # resolves to more than one distinct version is `None`, never a pick
        # (`AmbiguityIsNeverGuessed`).
        self.assertEqual(windows_driver_version(["64a0"], lookup=WindowsDriverLookup(
            drivers=[PnpDriver("PCI\\VEN_10DE&DEV_64A0", "Someone Else", "9.9.9")])),
            "9.9.9")

    def test_a_hint_and_a_name_that_resolve_to_different_drivers_is_none(self):
        # The conflict guard: if the plugin's own device id and its name point at rows
        # with different versions, neither is believed.
        lookup = _lookup()
        self.assertIsNone(lookup.driver_version_for(
            "Intel(R) AI Boost", device_id_hints=["0x64a0"]))


class EmptyAndUnusableInputIsNone(unittest.TestCase):
    def test_no_ids_is_none(self):
        self.assertIsNone(windows_driver_version([], lookup=_lookup()))

    def test_blank_and_none_ids_are_none(self):
        for bad in ([""], ["   "], [None]):
            self.assertIsNone(windows_driver_version(bad, lookup=_lookup()), bad)

    def test_a_driver_with_no_reported_version_is_none_not_zero(self):
        lookup = _lookup()
        self.assertIsNone(
            windows_driver_version(["PCI\\VEN_8086&DEV_FFFF"], lookup=lookup))
        self.assertNotEqual(
            windows_driver_version(["PCI\\VEN_8086&DEV_FFFF"], lookup=lookup), "0")

    def test_an_empty_table_is_none(self):
        self.assertIsNone(windows_driver_version([GPU_PCI], lookup=WindowsDriverLookup(
            drivers=EMPTY_TABLE)))


class AmbiguityIsNeverGuessed(unittest.TestCase):
    def test_two_candidate_ids_with_different_versions_is_none(self):
        self.assertIsNone(windows_driver_version([GPU_PCI, NPU_PCI], lookup=_lookup()))

    def test_two_rows_on_the_same_hardware_id_with_different_versions_is_none(self):
        table = list(FAKE_TABLE) + [
            PnpDriver("PCI\\VEN_8086&DEV_64A0&SUBSYS_00000000&REV_04\\3&9&0&99",
                      "Same Chip, Other Subsystem", "1.2.3.4"),
        ]
        self.assertIsNone(windows_driver_version(
            ["PCI\\VEN_8086&DEV_64A0"], lookup=WindowsDriverLookup(drivers=table)))

    def test_two_rows_on_the_same_hardware_id_agreeing_on_the_version_is_that_version(self):
        table = list(FAKE_TABLE) + [
            PnpDriver("PCI\\VEN_8086&DEV_64A0&SUBSYS_00000000&REV_04\\3&9&0&99",
                      "Same Chip, Other Subsystem", "32.0.101.6737"),
        ]
        self.assertEqual(
            windows_driver_version(["PCI\\VEN_8086&DEV_64A0"],
                                   lookup=WindowsDriverLookup(drivers=table)),
            "32.0.101.6737")


class EveryFailureDegradesToNone(unittest.TestCase):
    """No failure of the enumeration may raise out of this module."""

    def _assert_degrades(self, runner):
        lookup = WindowsDriverLookup(runner=runner)
        # These would all raise if any of the paths below propagated an exception.
        self.assertIsNone(lookup.driver_version_for("Intel(R) AI Boost"))
        self.assertIsNone(lookup.driver_version_for(
            "Intel(R) AI Boost", device_id_hints=["0x643e"]))
        self.assertIsNone(windows_driver_version([NPU_PCI], lookup=lookup))
        self.assertEqual(lookup.drivers, ())

    def test_a_missing_powershell_is_none(self):
        self._assert_degrades(
            lambda script, timeout: (_ for _ in ()).throw(FileNotFoundError("no such file")))

    def test_execution_blocked_by_policy_is_none(self):
        def blocked(script, timeout):
            raise subprocess.CalledProcessError(1, "powershell.exe")

        self._assert_degrades(blocked)

    def test_a_timeout_is_none(self):
        def slow(script, timeout):
            raise subprocess.TimeoutExpired("powershell.exe", timeout)

        self._assert_degrades(slow)

    def test_a_non_zero_exit_is_none(self):
        def failed(script, timeout):
            raise OSError("powershell.exe exited 1")

        self._assert_degrades(failed)

    def test_empty_stdout_is_none(self):
        self._assert_degrades(lambda script, timeout: "   \n")

    def test_unparsable_stdout_is_none(self):
        self._assert_degrades(lambda script, timeout: "Get-CimInstance : Invalid class")

    def test_a_json_payload_that_is_not_a_driver_table_is_none(self):
        self._assert_degrades(lambda script, timeout: json.dumps([1, 2, 3]))

    def test_a_row_without_a_device_id_is_skipped_not_invented(self):
        self.assertEqual(_parse_enumeration(json.dumps(
            [{"DeviceName": "Nameless", "DriverVersion": "1.2.3"}])), ())

    def test_a_missing_driver_version_becomes_none_not_an_empty_string(self):
        (driver,) = _parse_enumeration(json.dumps(
            [{"DeviceID": GPU_PCI, "DeviceName": "X", "DriverVersion": "  "}]))
        self.assertIsNone(driver.driver_version)


class TheEnumerationRunsOnceAndNotPerSample(unittest.TestCase):
    """The PnP read is a one-time discovery call, never a per-sample cost."""

    def test_the_enumeration_happens_once_however_many_lookups_follow(self):
        calls = []

        def runner(script, timeout):
            calls.append(script)
            return json.dumps([{"DeviceID": NPU_PCI, "DeviceName": "Intel(R) AI Boost",
                                "DriverVersion": "32.0.100.5540"}])

        lookup = WindowsDriverLookup(runner=runner)
        for _ in range(50):
            lookup.driver_version_for("Intel(R) AI Boost")
            lookup.driver_version_for("Intel(R) AI Boost", device_id_hints=["0x643e"])
            windows_driver_version([NPU_PCI], lookup=lookup)
        self.assertEqual(lookup.enumerations, 1)
        self.assertEqual(len(calls), 1)

    def test_a_pinned_table_never_enumerates(self):
        lookup = _lookup(runner=_NeverRuns())
        self.assertEqual(lookup.driver_version_for("Intel(R) AI Boost"),
                         "32.0.100.5540")
        self.assertEqual(lookup.enumerations, 0)

    def test_the_default_lookup_is_a_singleton_so_the_cost_is_paid_once_per_process(self):
        self.assertIs(default_driver_lookup(), default_driver_lookup())

    def test_the_enumeration_script_asks_only_for_the_three_fields_read_here(self):
        # Select-Object keeps the payload from dragging ~50 KB of CIM class metadata, and
        # the filter keeps it to PCI devices.
        self.assertIn("Select-Object DeviceID,DeviceName,DriverVersion", _ENUMERATE_SCRIPT)
        self.assertIn("Win32_PnPSignedDriver", _ENUMERATE_SCRIPT)
        self.assertIn("DeviceID LIKE 'PCI%'", _ENUMERATE_SCRIPT)


class DeviceNamesResolveWhenThereIsNoDeviceId(unittest.TestCase):
    """The NPU plugin exposes no PCI device id, so the name is the only handle it has."""

    def test_the_openvino_qualifier_suffix_is_dropped_but_the_vendor_marks_are_not(self):
        self.assertEqual(normalise_device_name("Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)"),
                         normalise_device_name("Intel(R) Arc(TM) 140V GPU (16GB)"))
        self.assertEqual(normalise_device_name("  intel(r)   ai   boost "),
                         "INTEL(R) AI BOOST")
        self.assertEqual(normalise_device_name("Intel(R) AI Boost"),
                         normalise_device_name("Intel(R) AI Boost"))

    def test_a_name_resolves_its_pci_device_id(self):
        lookup = _lookup()
        self.assertEqual(
            lookup.pci_device_ids_for("Intel(R) AI Boost"), (NPU_PCI,))
        self.assertEqual(
            lookup.pci_device_ids_for("Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)"),
            (GPU_PCI,))

    def test_the_name_route_returns_the_version(self):
        lookup = _lookup()
        self.assertEqual(lookup.driver_version_for("Intel(R) AI Boost"),
                         "32.0.100.5540")
        self.assertEqual(lookup.driver_version_for("Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)"),
                         "32.0.101.6737")

    def test_a_name_no_pnp_row_carries_resolves_to_nothing(self):
        # The CPU: 'Intel(R) Core(TM) Ultra 9 288V' is the processor, and no PnP signed
        # driver is named after it. Nothing is invented for it.
        lookup = _lookup()
        self.assertEqual(lookup.pci_device_ids_for("Intel(R) Core(TM) Ultra 9 288V"), ())
        self.assertIsNone(lookup.driver_version_for("Intel(R) Core(TM) Ultra 9 288V"))

    def test_an_empty_name_resolves_to_nothing(self):
        self.assertEqual(_lookup().pci_device_ids_for(""), ())

    def test_a_device_id_hint_and_a_name_agree_on_one_row(self):
        lookup = _lookup()
        self.assertEqual(
            lookup.pci_device_ids_for("Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)",
                                      device_id_hints=["0x64a0"]),
            (GPU_PCI,))
        self.assertEqual(len(lookup.pci_device_ids_for(
            "Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)", device_id_hints=["0x64a0"])), 1)


class PciDeviceIdShapes(unittest.TestCase):
    """`looks_like_a_pci_device_id` decides which plugin properties are even searched."""

    def test_real_spellings_are_accepted(self):
        for good in (GPU_PCI, NPU_PCI, "PCI\\VEN_8086&DEV_64A0", "DEV_64A0", "0x64a0",
                     "64A0", "0X64a0"):
            self.assertTrue(looks_like_a_pci_device_id(good), good)

    def test_a_plugin_ordinal_is_rejected(self):
        # Measured: the GPU plugin's DEVICE_ID reads '0'. Searching for it would be a
        # plausible-looking lookup that can only ever match nothing.
        for bad in ("0", "", "   ", None, 0, "GPU", "Intel(R) Arc(TM) 140V GPU"):
            self.assertFalse(looks_like_a_pci_device_id(bad), repr(bad))

    def test_the_device_id_field_is_read_out_of_a_pnp_hardware_id(self):
        self.assertEqual(PnpDriver(GPU_PCI, "x", None).pci_device_id, "64A0")
        self.assertEqual(PnpDriver("PCI\\VEN_8086&DEV_643E", "x", None).pci_device_id,
                         "643E")
        self.assertEqual(PnpDriver("no hardware id here", "x", None).pci_device_id, "")

    def test_a_blank_candidate_matches_nothing(self):
        self.assertFalse(driver_matches_device_id(FAKE_TABLE[0], ""))
        self.assertFalse(driver_matches_device_id(FAKE_TABLE[0], "   "))


class NothingIsHardcoded(unittest.TestCase):
    """No PCI device id and no version string is written into the shipped module.

    The whole point of this module is that the ids and the versions are *read*. A
    hardcoded `32.0.100.5540` would survive a driver update and be wrong, which is the
    exact failure `app.inference.cache` exists to prevent.
    """

    @classmethod
    def setUpClass(cls):
        import pathlib
        source = pathlib.Path(
            "app/inference/device_drivers.py").read_text(encoding="utf-8")
        # Strip comments and docstrings so the prose that *reports* the measurement is
        # not mistaken for a constant that *supplies* it.
        cls.code = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
        cls.code = "\n".join(
            line for line in cls.code.splitlines()
            if not line.lstrip().startswith("#")
        )

    def test_no_measured_pci_device_id_appears_in_the_code(self):
        for token in ("643E", "64A0", "A838", "88F21043", "11583659"):
            self.assertNotIn(token, self.code,
                             f"{token!r} is a measured PCI id and must not be code")

    def test_no_measured_driver_version_appears_in_the_code(self):
        for token in ("32.0.100.5540", "32.0.101.6737", "1005540"):
            self.assertNotIn(token, self.code,
                             f"{token!r} is a measured driver version and must not be code")

    def test_no_processor_family_is_written_into_the_code(self):
        for wrong in ("Panther", "Arrow Lake", "Lunar Lake", "Lunar", "Core Ultra"):
            self.assertNotIn(wrong, self.code,
                             f"a family name leaked into code: {wrong!r}")


class TheRealPnpLayerOnThisMachine(unittest.TestCase):
    """The pins that deliberately depend on this machine.

    These assert a *measurement*, so they are the ones a different laptop would fail -
    which is the point. A failure here means the lookup stopped resolving real hardware,
    not that the expectation needs loosening.
    """

    @classmethod
    def setUpClass(cls):
        cls.core = ov.Core()
        cls.lookup = default_driver_lookup()

    def test_the_pnp_table_is_read_from_windows(self):
        drivers = self.lookup.drivers
        self.assertTrue(drivers, "no PnP signed drivers were read on this machine")
        self.assertTrue(any(d.device_id.upper().startswith("PCI\\") for d in drivers))
        for driver in drivers:
            self.assertIsNotNone(driver.driver_version,
                                 f"{driver.device_id} has no DriverVersion")

    def test_the_gpu_and_the_npu_are_resolved_by_device_id(self):
        for device_id in ("GPU", "NPU"):
            full_name = str(self.core.get_property(device_id, "FULL_DEVICE_NAME"))
            ids = self.lookup.pci_device_ids_for(full_name)
            self.assertEqual(len(ids), 1,
                             f"{device_id} ({full_name!r}) matched {ids}")
            self.assertIsNotNone(self.lookup.driver_version_for_ids(ids))

    def test_the_npu_driver_version_agrees_with_openvino(self):
        # Cross-check, not a coincidence: the PnP row and the NPU plugin's own property
        # describe the same driver. '32.0.100.5540' minus its leading '32.0.' and dots
        # is the '1005540' the plugin reports.
        full_name = str(self.core.get_property("NPU", "FULL_DEVICE_NAME"))
        from_pnp = self.lookup.driver_version_for(full_name)
        from_openvino = str(self.core.get_property("NPU", "NPU_DRIVER_VERSION"))
        self.assertIsNotNone(from_pnp)
        tail = from_pnp.split(".", 2)[2] if from_pnp.count(".") >= 2 else ""
        self.assertEqual(tail.replace(".", ""), from_openvino)

    def test_the_cpu_has_no_pnp_driver_and_reports_none(self):
        full_name = str(self.core.get_property("CPU", "FULL_DEVICE_NAME"))
        self.assertEqual(self.lookup.pci_device_ids_for(full_name), (),
                         "the CPU unexpectedly matched a PnP driver; re-read C3")
        self.assertIsNone(self.lookup.driver_version_for(full_name))


if __name__ == "__main__":
    unittest.main()
