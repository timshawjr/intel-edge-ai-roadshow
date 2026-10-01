"""LUID-keyed matching of Windows `GPU Engine` PDH instances to devices.

Windows reports GPU engine counters per **process and adapter LUID**, and the NPU and the
GPU report overlapping physical device ids - so a naive mapping puts a real but wrong
number on the NPU's card. The LUID is the only unambiguous tie-breaker, and the pins here
are the structural guarantee that a swap cannot be committed silently:

1. `parse_instance_name` takes the **second** LUID word and refuses to guess. A name with
   no `luid_` field raises `ValueError` rather than yielding a plausible `0x00000000`.
2. An unknown LUID matches **nothing** (`None`), rather than falling back to the first rule.
3. One LUID mapped to two devices is a **hard error** on load, not a warning.

The LUIDs here are the ones measured on this machine under the probe; they appear in this
test as fixture data only and are read from `config/telemetry_map.json` at runtime, never
from a constant in application code.
"""

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
