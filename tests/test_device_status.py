"""The DeviceStatus contract: what every later layer reports through.

If discovery is wrong, every number in the whole product is wrong. These tests pin
the two properties the specification cares about most: a round trip through JSON must
preserve the record exactly, and an unavailable device must say *why* rather than
being rendered as a zero.
"""

import json
import unittest

from app.contracts.device_status import DeviceStatus, Validity, mark_validated


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


class TheRecordSurvivesRealJson(unittest.TestCase):
    """`to_json` is only useful if `json.dumps` accepts it, so prove that it does."""

    def _status(self):
        return DeviceStatus(
            device_id="GPU",
            full_name="Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)",
            device_type="Type.INTEGRATED",
            available=True, validated=True, ready=True,
            runtime_version="2026.4.0-22959-99c81491cc3-releases/2026/4",
            driver_version=None,
            supported_model_ids=("yolo11n-fp16",),
            unavailable_reason=None,
        )

    def test_to_json_is_accepted_by_json_dumps_and_reads_back_identical(self):
        s = self._status()
        text = json.dumps(s.to_json())
        self.assertEqual(DeviceStatus.from_json(json.loads(text)), s)

    def test_the_tuple_field_is_a_list_on_the_wire_and_a_tuple_in_the_contract(self):
        payload = self._status().to_json()
        self.assertIsInstance(payload["supported_model_ids"], list)
        self.assertEqual(
            DeviceStatus.from_json(payload).supported_model_ids, ("yolo11n-fp16",))

    def test_every_field_is_present_in_the_payload(self):
        payload = self._status().to_json()
        for name in ("device_id", "full_name", "device_type", "available", "validated",
                     "ready", "runtime_version", "driver_version",
                     "supported_model_ids", "unavailable_reason"):
            self.assertIn(name, payload)

    def test_a_missing_field_fails_closed_rather_than_defaulting(self):
        payload = self._status().to_json()
        del payload["driver_version"]
        with self.assertRaises(ValueError) as ctx:
            DeviceStatus.from_json(payload)
        self.assertIn("driver_version", str(ctx.exception))


class UnavailableIsNeverASilentZero(unittest.TestCase):
    """A device that cannot be used must be constructible only with a stated reason."""

    def _unavailable(self, **overrides):
        base = dict(device_id="NPU", full_name="", device_type="Type.INTEGRATED",
                    available=False, validated=False, ready=False,
                    runtime_version="2026.4.0-22959-99c81491cc3-releases/2026/4",
                    driver_version=None, supported_model_ids=(),
                    unavailable_reason="not reported by ov.Core().available_devices()")
        base.update(overrides)
        return base

    def test_an_unavailable_device_with_no_reason_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            DeviceStatus(**self._unavailable(unavailable_reason=None))
        self.assertIn("unavailable_reason", str(ctx.exception))

    def test_an_unavailable_device_with_a_blank_reason_is_rejected(self):
        with self.assertRaises(ValueError):
            DeviceStatus(**self._unavailable(unavailable_reason="   "))

    def test_an_available_device_cannot_carry_a_reason(self):
        with self.assertRaises(ValueError) as ctx:
            DeviceStatus(device_id="NPU", full_name="Intel(R) AI Boost",
                         device_type="Type.INTEGRATED", available=True, validated=True,
                         ready=True,
                         runtime_version="2026.4.0-22959-99c81491cc3-releases/2026/4",
                         driver_version="1005540", supported_model_ids=("yolo11n-fp16",),
                         unavailable_reason="something is odd")
        self.assertIn("unavailable_reason", str(ctx.exception))

    def test_a_missing_driver_version_is_none_and_not_zero(self):
        s = DeviceStatus(**self._unavailable())
        self.assertIsNone(s.driver_version)
        self.assertNotEqual(s.driver_version, 0)

    def test_the_two_unavailability_values_are_different_strings(self):
        self.assertNotEqual(Validity.UNAVAILABLE.value, Validity.STALE.value)
        self.assertNotEqual(Validity.VALID.value, Validity.UNAVAILABLE.value)
        for member in Validity:
            self.assertNotEqual(member.value, "0")
            self.assertTrue(member.value)
            # `str, Enum` so a value survives a JSON round trip as itself.
            self.assertEqual(member, member.value)


class ValidationIsASeparateClaimFromDiscovery(unittest.TestCase):
    """`available` is a discovery fact. `validated` and `ready` are earned later."""

    def _discovered(self):
        return DeviceStatus(device_id="NPU", full_name="Intel(R) AI Boost",
                            device_type="Type.INTEGRATED", available=True,
                            validated=False, ready=False,
                            runtime_version="2026.4.0-22959-99c81491cc3-releases/2026/4",
                            driver_version="1005540", supported_model_ids=(),
                            unavailable_reason=None)

    def test_a_discovered_device_is_not_yet_validated_or_ready(self):
        s = self._discovered()
        self.assertTrue(s.available)
        self.assertFalse(s.validated)
        self.assertFalse(s.ready)
        self.assertEqual(s.supported_model_ids, ())

    def test_mark_validated_records_the_models_and_promotes_ready(self):
        out = mark_validated(self._discovered(), ["yolo11n-fp16", "yolo11n-fp16"])
        self.assertTrue(out.validated)
        self.assertTrue(out.ready)
        self.assertEqual(out.supported_model_ids, ("yolo11n-fp16",))
        self.assertEqual(out.runtime_version, self._discovered().runtime_version)

    def test_mark_validated_does_not_mutate_the_original(self):
        before = self._discovered()
        mark_validated(before, ["yolo11n-fp16"])
        self.assertEqual(before, self._discovered())
        self.assertEqual(before.supported_model_ids, ())

    def test_an_unavailable_device_is_never_promoted_to_ready(self):
        unavailable = DeviceStatus(device_id="NPU", full_name="", device_type="Type.INTEGRATED",
                                  available=False, validated=False, ready=False,
                                  runtime_version="2026.4.0-22959-99c81491cc3-releases/2026/4",
                                  driver_version=None, supported_model_ids=(),
                                  unavailable_reason="no Neural engine counter")
        out = mark_validated(unavailable, ["yolo11n-fp16"])
        self.assertFalse(out.ready)
        self.assertFalse(out.available)
        self.assertEqual(out.unavailable_reason, "no Neural engine counter")


if __name__ == "__main__":
    unittest.main()
