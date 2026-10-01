"""The compiled-blob cache key.

Review Focus #5: a compiled blob is reused after a driver update. NPU blob-cache
compatibility is not guaranteed across driver versions, so the key must carry every
input that can change the artifact. These tests pin that each one does, and that a
recorded blob from a different machine state is rejected rather than reused.
"""

import unittest

from app.inference.cache import cache_identity, cache_is_valid, cache_key

BASE = dict(model_hash="a" * 64, device_id="NPU", runtime_version="2026.4.0",
            driver_version="32.0.100.5540", properties="THROUGHPUT")


class CacheKeyTracksEverythingThatCanInvalidateABlob(unittest.TestCase):
    def test_key_is_stable_for_identical_inputs(self):
        self.assertEqual(cache_key(**BASE), cache_key(**BASE))

    def test_a_driver_update_changes_the_key(self):
        # Review Focus #5: NPU blob compatibility is not guaranteed across drivers.
        # NOTE: the plan wrote `cache_key(other)` here. `cache_key` is keyword-only, so
        # that call is a TypeError. The plan's own test is corrected, not the signature.
        other = dict(BASE, driver_version="32.0.100.9999")
        self.assertNotEqual(cache_key(**BASE), cache_key(**other))

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


class TheFullBuildStringIsWhatGoesInTheKey(unittest.TestCase):
    """`runtime_version` is the full OpenVINO build string, so two builds that share a
    short version do not share a blob."""

    def test_two_different_builds_of_one_version_differ(self):
        a = dict(BASE, runtime_version="2026.4.0-22959-99c81491cc3-releases/2026/4")
        b = dict(BASE, runtime_version="2026.4.0-23001-aaaabbbbcccc-releases/2026/4")
        self.assertEqual(a["runtime_version"].split("-")[0], b["runtime_version"].split("-")[0])
        self.assertNotEqual(cache_key(**a), cache_key(**b))

    def test_the_identity_is_built_from_the_short_version_too(self):
        # A truncated record is still accepted as an identity; it is simply a coarser one.
        self.assertEqual(cache_identity(**dict(BASE, runtime_version="2026.4.0")),
                         cache_identity(**dict(BASE, runtime_version="2026.4.0")))


class AnUnreportedDriverVersionIsItsOwnState(unittest.TestCase):
    """`driver_version=None` means "this device reports no driver version". That is not
    the same machine state as a recorded empty string, and neither is `0`."""

    def test_none_is_distinct_from_an_empty_string_and_from_zero(self):
        none = cache_key(**dict(BASE, driver_version=None))
        self.assertNotEqual(none, cache_key(**dict(BASE, driver_version="")))
        self.assertNotEqual(none, cache_key(**dict(BASE, driver_version="0")))

    def test_the_key_is_stable_for_none(self):
        self.assertEqual(cache_key(**dict(BASE, driver_version=None)),
                         cache_key(**dict(BASE, driver_version=None)))

    def test_a_blob_recorded_without_a_driver_version_is_rejected_once_one_appears(self):
        recorded = cache_identity(**dict(BASE, driver_version=None))
        current = cache_identity(**dict(BASE, driver_version="1005540"))
        self.assertFalse(cache_is_valid(recorded, current))


class ValidityIsCheckedFieldByField(unittest.TestCase):
    def test_a_partial_current_mapping_does_not_validate(self):
        # A caller that forgets a field must get False, never a silent pass.
        current = dict(BASE)
        del current["properties"]
        self.assertFalse(cache_is_valid(BASE, current))

    def test_an_empty_current_mapping_does_not_validate(self):
        self.assertFalse(cache_is_valid(BASE, {}))

    def test_every_identity_field_is_compared(self):
        for field, value in [("model_hash", "b" * 64), ("device_id", "GPU"),
                             ("runtime_version", "2025.1.0"),
                             ("driver_version", "32.0.100.9999"),
                             ("properties", "LATENCY")]:
            self.assertFalse(
                cache_is_valid(cache_identity(**BASE), cache_identity(**dict(BASE, **{field: value}))),
                f"{field} is not compared by cache_is_valid")


class TheIdentityIsCompleteByConstruction(unittest.TestCase):
    def test_the_identity_has_exactly_the_five_key_inputs(self):
        self.assertEqual(set(cache_identity(**BASE)),
                         {"model_hash", "device_id", "runtime_version",
                          "driver_version", "properties"})

    def test_identity_survives_a_json_round_trip(self):
        import json
        identity = cache_identity(**BASE)
        self.assertEqual(json.loads(json.dumps(identity)), identity)

    def test_the_key_is_a_hex_digest_of_a_canonical_encoding(self):
        key = cache_key(**BASE)
        self.assertEqual(len(key), 64)
        int(key, 16)  # raises if it is not hex

    def test_a_delimiter_cannot_forge_a_collision(self):
        # Naive "a|b|c" joining collides these two. A canonical encoding does not.
        a = dict(BASE, device_id="NPU|32.0.100.5540", properties="THROUGHPUT")
        b = dict(BASE, device_id="NPU", properties="THROUGHPUT|32.0.100.5540")
        self.assertNotEqual(cache_key(**a), cache_key(**b))


if __name__ == "__main__":
    unittest.main()
