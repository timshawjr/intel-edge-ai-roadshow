"""M6 Task 2: the known-good environment, read from the machine and never guessed.

The file this suite protects is `docs/install/KNOWN-GOOD-environment.json`. It is only
worth anything if every value came from the machine it describes. So the tests do four
things:

1. **Read-through, not constants.** A fake inventory is injected and the manifest must
   echo it; Python and OpenVINO versions are compared against the interpreters that are
   actually running.
2. **Unavailable is ``None`` with a reason.** With the Windows inventory unavailable,
   every dependent field is ``None`` and `unavailable` names it. It is never ``0``,
   never a plausible default, and never a remembered value.
3. **Three SKU records stay three.** This machine reports `Win32_ComputerSystem` model
   `NUC14LNK` (family `LN`), hostname `NUC14PROAI`, and PCI subsystem `88F21043`. Those
   disagree, there is no defensible single SKU, and the collector must not collapse them
   into one or invent a fourth.
4. **No battery means AC status is unavailable.** `Win32_Battery` returns zero instances
   on this machine, so `ac_status` is ``None`` with that reason rather than a guessed
   `True`.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import socket
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

from tools import environment_manifest as em

REPO = Path(__file__).resolve().parents[1]
KNOWN_GOOD = REPO / "docs" / "install" / "KNOWN-GOOD-environment.json"


def _fake_windows() -> dict:
    """A complete inventory with values that appear nowhere in the implementation."""
    return {
        "computer_system": {
            "model": "FAKE-MODEL-9000",
            "name": "FAKE-HOST-42",
            "manufacturer": "Fake Systems",
            "system_family": "ZZ",
            "total_physical_memory": 123456789,
        },
        "product": {
            "uuid": "11111111-2222-3333-4444-555555555555",
            "version": "FAKE-BIOS-1",
            "vendor": "Fake Systems",
            "identifying_number": "FAKE-SERIAL",
        },
        "cpu": {
            "name": "Fake CPU 9000",
            "manufacturer": "Fake",
            "cores": 2,
            "logical_processors": 4,
            "max_clock_mhz": 1234,
        },
        "gpu": {
            "name": "Fake GPU",
            "driver_version": "1.2.3.4",
            "driver_date": "2026-01-02",
            "pnp_id": r"PCI\VEN_8086&DEV_0001&SUBSYS_AABBCCDD&REV_01\FAKE",
            "adapter_ram": 1024,
        },
        "npu": {
            "name": "Fake NPU",
            "pnp_id": r"PCI\VEN_8086&DEV_0002&SUBSYS_AABBCCDD&REV_01\FAKE",
            "status": "OK",
            "manufacturer": "Fake",
        },
        "gpu_driver": {
            "version": "1.2.3.4",
            "date": "2026-01-02",
            "provider": "Fake",
            "inf": "oem99.inf",
        },
        "npu_driver": {
            "version": "5.6.7.8",
            "date": "2026-02-03",
            "provider": "Fake",
            "inf": "oem98.inf",
        },
        "os": {
            "caption": "Fake Windows",
            "version": "10.0.99999",
            "build_number": "99999",
            "architecture": "64-bit",
            "total_visible_memory_kb": 120000,
        },
        "registry": {
            "product_name": "Fake Windows",
            "display_version": "99H9",
            "current_build": 99999,
            "ubr": 4242,
            "edition_id": "FakeEdition",
        },
        "bios": {
            "manufacturer": "Fake",
            "version": "FAKE-BIOS-1",
            "release_date": "2026-03-04",
        },
        "monitor_id": {
            "instance_name": "DISPLAY\\FAKE\\0",
            "manufacturer": "FAK",
            "product": "F001",
            "serial": "123456789",
            "year": 2026,
            "week": 7,
        },
        "monitor": {
            "name": "Fake Monitor 27",
            "pnp_id": "DISPLAY\\FAK0001\\0",
        },
        "battery": {"count": 0, "statuses": [], "charges": []},
        "power": "Power Scheme GUID: 11111111-2222-3333-4444-555555555555  (Fake Plan)",
    }


class TheEnvironmentIsReadFromThisMachine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = em.collect_environment()

    # -- 1. read-through ---------------------------------------------------------------

    def test_windows_fields_are_the_injected_inventory_not_constants(self):
        with mock.patch.object(em, "_read_windows", return_value=_fake_windows()):
            manifest = em.collect_environment()
        self.assertEqual(manifest.hostname, "FAKE-HOST-42")
        self.assertEqual(manifest.machine_id, "11111111-2222-3333-4444-555555555555")
        self.assertEqual(manifest.cpu.name, "Fake CPU 9000")
        self.assertEqual(manifest.gpu.name, "Fake GPU")
        self.assertEqual(manifest.npu.name, "Fake NPU")
        self.assertEqual(manifest.ram_bytes, 123456789)
        self.assertEqual(manifest.windows.build_number, "99999")
        self.assertEqual(manifest.power_plan.name, "Fake Plan")
        self.assertEqual(manifest.monitor.name, "Fake Monitor 27")

    def test_python_and_openvino_are_the_running_ones(self):
        self.assertEqual(self.manifest.python_version, platform.python_version())
        self.assertEqual(self.manifest.python_executable, sys.executable)
        self.assertEqual(
            self.manifest.openvino_version,
            importlib.metadata.version("openvino"),
        )

    def test_hostname_and_windows_version_match_independent_reads(self):
        self.assertEqual(self.manifest.hostname.lower(), socket.gethostname().lower())
        self.assertEqual(self.manifest.windows.version, platform.version())
        self.assertEqual(self.manifest.windows.build_number, platform.version().split(".")[-1])

    # -- 2. the three SKU records ------------------------------------------------------

    def test_three_conflicting_sku_records_are_preserved_as_three(self):
        records = self.manifest.sku_records
        self.assertEqual(len(records), 3,
                         "this machine has no single defensible SKU; collapsing the "
                         "records would invent one")
        self.assertEqual(len({record.source for record in records}), 3)
        values = [record.value for record in records]
        self.assertIn("NUC14LNK", values[0])
        self.assertEqual(values[1].lower(), socket.gethostname().lower())
        self.assertRegex(values[2], r"^[0-9A-F]{8}$")
        self.assertGreaterEqual(len(set(values)), 2,
                                "the records are supposed to disagree; if they agree, "
                                "this test no longer proves the collector preserves a "
                                "conflict")

    def test_the_sku_records_name_their_sources(self):
        model, hostname, pci = self.manifest.sku_records
        self.assertIn("Win32_ComputerSystem.Model", model.source)
        self.assertIn("hostname", hostname.source.lower())
        self.assertIn("PCI", pci.source)
        self.assertTrue(self.manifest.sku_note,
                        "the manifest must say why there is no winner")

    # -- 3. AC status is unavailable, with its reason ---------------------------------

    def test_ac_status_is_none_because_this_machine_has_no_battery(self):
        self.assertIsNone(self.manifest.ac_status)
        self.assertEqual(self.manifest.battery_instances, 0)
        reason = self.manifest.unavailable["ac_status"]
        self.assertIn("Win32_Battery", reason)
        self.assertIn("0", reason)

    def test_ac_status_is_never_reported_as_a_boolean_without_a_battery(self):
        self.assertIsNot(self.manifest.ac_status, False,
                         "False means 'on battery'; it is not a stand-in for unknown")
        self.assertIsNot(self.manifest.ac_status, True)

    # -- 4. unavailable fields are None with a reason, never a guess -------------------

    def test_unreadable_windows_inventory_leaves_none_and_a_reason(self):
        with mock.patch.object(
            em, "_read_windows", side_effect=RuntimeError("injected inventory failure")
        ):
            manifest = em.collect_environment()
        for field in ("cpu", "gpu", "npu", "windows", "ram_bytes", "power_plan",
                      "monitor", "machine_id", "hostname"):
            self.assertIsNone(getattr(manifest, field), field)
            self.assertIn(field, manifest.unavailable)
            self.assertIn("injected inventory failure", manifest.unavailable[field])
        self.assertIsNotNone(manifest.python_version,
                             "the interpreter is still readable when CIM is not")
        self.assertIsNotNone(manifest.openvino_version)

    def test_a_missing_monitor_is_none_with_its_own_reason(self):
        raw = _fake_windows()
        del raw["monitor_id"]
        del raw["monitor"]
        with mock.patch.object(em, "_read_windows", return_value=raw):
            manifest = em.collect_environment()
        self.assertIsNone(manifest.monitor)
        self.assertIn("monitor", manifest.unavailable)

    def test_a_missing_power_plan_is_none_with_its_own_reason(self):
        raw = _fake_windows()
        raw["power"] = ""
        with mock.patch.object(em, "_read_windows", return_value=raw):
            manifest = em.collect_environment()
        self.assertIsNone(manifest.power_plan)
        self.assertIn("power_plan", manifest.unavailable)

    def test_serialising_an_unavailable_field_keeps_none_and_the_reason(self):
        with mock.patch.object(
            em, "_read_windows", side_effect=RuntimeError("injected inventory failure")
        ):
            payload = json.loads(json.dumps(em.collect_environment().as_dict()))
        self.assertIsNone(payload["ac_status"])
        self.assertIsNone(payload["cpu"])
        self.assertIn("cpu", payload["unavailable"])
        self.assertIn("ac_status", payload["unavailable"])

    # -- 5. the committed known-good file is a real collection -------------------------

    def test_the_committed_known_good_file_was_collected_from_this_machine(self):
        self.assertTrue(KNOWN_GOOD.is_file())
        stored = json.loads(KNOWN_GOOD.read_text(encoding="utf-8"))
        self.assertEqual(stored["machine_id"], self.manifest.machine_id,
                         "KNOWN-GOOD-environment.json was not written from this machine; "
                         "re-run tools/environment_manifest.py on the demo machine")
        self.assertEqual(len(stored["sku_records"]), 3)
        self.assertIsNone(stored["ac_status"])
        self.assertIn("ac_status", stored["unavailable"])
        uuid.UUID(stored["machine_id"])
        self.assertRegex(stored["power_plan"]["guid"],
                         r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                         r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
        self.assertIn("environment_manifest.py", stored["collector"])


class TheCollectorDoesNotNeedWindowsToExist(unittest.TestCase):
    def test_collect_never_raises_when_the_platform_has_no_powershell(self):
        with mock.patch.object(em, "_read_windows",
                               side_effect=FileNotFoundError("no powershell")):
            manifest = em.collect_environment()
        self.assertIsNone(manifest.cpu)
        self.assertIn("cpu", manifest.unavailable)
        self.assertEqual(len(manifest.sku_records), 3)


if __name__ == "__main__":
    unittest.main()
