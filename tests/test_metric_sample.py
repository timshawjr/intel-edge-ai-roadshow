"""`MetricSample`: the record that makes "unavailable" unrepresentable as `0`.

Review Focus #3 is *"unavailable rendered as 0"*, and it is the failure a booth hits
hardest: a card that reads **NPU 0%** tells a visitor the NPU is idle, and if the truth
is that no counter exists it tells them the NPU is broken. Those are opposite claims
about the same hardware and only one of them may be rendered.

So the distinction is not a convention this layer follows, it is a shape:

- `value is None` **and** `validity is UNAVAILABLE` are the same fact, and one of them
  without the other is refused at construction. There is no way to build a sample that
  says "we could not measure this" and carries a number anyway.
- A genuine `0.0` is a perfectly good value and stays `VALID`. The refusal must not be so
  eager that an idle NPU reads as unavailable - that is the same mistake pointing the
  other way, and a card that says "unavailable" on an idle device is just as wrong.
- `unavailable_reason` is required when `validity is UNAVAILABLE` and forbidden
  otherwise, exactly as `DeviceStatus` and `TelemetryAvailability` already do. A blank
  reason is indistinguishable from no reason.
- `to_json()` keeps `None` as JSON `null`. A number substituted at serialisation time is
  the same lie one layer further out.

`is_stale` is **derived from the timestamp**, not from the recorded `validity`: staleness
is a fact about the clock, and a `validity` field that could assert it independently
would be a second, drifting copy of the same fact. An `UNAVAILABLE` sample is never
stale - a counter that has never been readable does not go stale, it stays unavailable,
and letting it decay into "stale" would eventually get it rendered as a last-known
number, which is a measurement nobody took.
"""

import json
import unittest

from app.contracts.metric_sample import MetricSample, Validity


def _s(value, validity=Validity.VALID, sampled_at=100.0, name="system_cpu_percent",
       **kwargs):
    base = dict(unit="percent", provider="psutil", device_scope=None,
                process_scope=("ui",), window_seconds=1.0)
    base.update(kwargs)
    return MetricSample(name=name, value=value, validity=validity,
                        sampled_at=sampled_at, **base)


class UnavailableIsNotZero(unittest.TestCase):
    def test_review_focus_3_a_missing_value_is_null_not_zero(self):
        s = _s(None, Validity.UNAVAILABLE, unavailable_reason="no counter exists")
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
        s = _s(None, Validity.UNAVAILABLE, sampled_at=0.0,
               unavailable_reason="no counter exists")
        self.assertFalse(s.is_stale(now=10_000.0))
        self.assertEqual(s.validity, Validity.UNAVAILABLE)

    def test_json_keeps_none_as_null(self):
        text = json.dumps(_s(None, Validity.UNAVAILABLE,
                             unavailable_reason="no counter exists").to_json())
        self.assertIn("null", text)


class TheTwoStatesCannotBeConfused(unittest.TestCase):
    """The pairing is enforced, so no code path can build the confusing shape."""

    def test_a_number_cannot_be_carried_as_unavailable(self):
        with self.assertRaises(ValueError):
            _s(0.0, Validity.UNAVAILABLE, unavailable_reason="no counter exists")

    def test_a_number_cannot_be_carried_as_unavailable_even_with_no_reason(self):
        """The pairing is enforced on the value, not merely via the reason.

        The first version of the test above passed a reason, so it would also have passed
        against an implementation that only checked the *reason* pairing and let a
        number ride along beside `UNAVAILABLE` with `unavailable_reason=None`. Both
        halves of "unavailable" have to hold independently: the reason says why, and the
        absent value says that there is nothing.
        """
        with self.assertRaises(ValueError):
            _s(0.0, Validity.UNAVAILABLE, unavailable_reason=None)
        with self.assertRaises(ValueError):
            _s(12.5, Validity.UNAVAILABLE, unavailable_reason=None)

    def test_an_unavailable_sample_must_say_why(self):
        for bad in (None, "", "   "):
            with self.assertRaises(ValueError):
                _s(None, Validity.UNAVAILABLE, unavailable_reason=bad)

    def test_a_measured_sample_cannot_carry_an_unavailable_reason(self):
        with self.assertRaises(ValueError):
            _s(1.0, Validity.VALID, unavailable_reason="but actually not")

    def test_a_stale_sample_cannot_carry_an_unavailable_reason_either(self):
        with self.assertRaises(ValueError):
            _s(1.0, Validity.STALE, unavailable_reason="but actually not")

    def test_a_none_value_cannot_claim_to_be_valid(self):
        with self.assertRaises(ValueError):
            _s(None, Validity.VALID)

    def test_a_none_value_cannot_claim_to_be_stale(self):
        with self.assertRaises(ValueError):
            _s(None, Validity.STALE, unavailable_reason="x")

    def test_the_record_is_frozen(self):
        s = _s(1.0)
        with self.assertRaises(Exception):
            s.value = 2.0  # type: ignore[misc]

    def test_the_enum_survives_a_json_round_trip_as_itself(self):
        self.assertEqual(json.loads(json.dumps(Validity.UNAVAILABLE)), "unavailable")
        self.assertEqual(json.loads(json.dumps(Validity.VALID)), "valid")


class IdentityAndScope(unittest.TestCase):
    def test_an_empty_name_unit_or_provider_is_refused(self):
        for field in ("name", "unit", "provider"):
            with self.assertRaises(ValueError):
                _s(1.0, **{field: "  "})

    def test_process_scope_is_normalised_to_a_tuple(self):
        s = _s(1.0, process_scope=["ui", "worker"])
        self.assertEqual(s.process_scope, ("ui", "worker"))

    def test_an_empty_entry_in_the_process_scope_is_refused(self):
        with self.assertRaises(ValueError):
            _s(1.0, process_scope=["ui", ""])

    def test_an_empty_device_scope_is_refused_because_absent_is_none(self):
        with self.assertRaises(ValueError):
            _s(1.0, device_scope="")

    def test_a_device_scoped_sample_names_the_device(self):
        s = _s(12.0, name="npu_neural_percent", device_scope="NPU")
        self.assertEqual(s.device_scope, "NPU")

    def test_a_non_positive_window_is_refused(self):
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                _s(1.0, window_seconds=bad)

    def test_a_negative_timestamp_is_refused(self):
        with self.assertRaises(ValueError):
            _s(1.0, sampled_at=-1.0)

    def test_a_non_finite_timestamp_is_refused_because_nan_never_compares_true(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                _s(1.0, sampled_at=bad)

    def test_a_non_finite_value_is_refused(self):
        with self.assertRaises(ValueError):
            _s(float("nan"))

    def test_the_json_payload_carries_every_field_and_lists_the_tuple(self):
        payload = _s(1.0, name="app_cpu_percent", process_scope=("ui", "worker"),
                     device_scope=None).to_json()
        self.assertEqual(payload["name"], "app_cpu_percent")
        self.assertEqual(payload["process_scope"], ["ui", "worker"])
        self.assertEqual(payload["unit"], "percent")
        self.assertEqual(payload["provider"], "psutil")
        self.assertIsNone(payload["device_scope"])
        self.assertEqual(payload["validity"], "valid")
        json.dumps(payload)  # must be serialisable as-is


class ClockBehaviour(unittest.TestCase):
    def test_a_sample_exactly_at_its_window_edge_is_not_yet_stale(self):
        self.assertFalse(_s(1.0, sampled_at=100.0, window_seconds=3.0).is_stale(now=103.0))

    def test_a_clock_that_reads_earlier_does_not_make_a_sample_fresh_forever(self):
        # now < sampled_at is a clock disagreement, not "very fresh"; it is reported as
        # stale rather than as a negative age, which is what `max(0, ...)` would give.
        s = _s(1.0, sampled_at=100.0)
        self.assertTrue(s.is_stale(now=50.0))

    def test_the_effective_validity_degrades_to_stale_and_never_to_a_number(self):
        fresh = _s(1.0, sampled_at=100.0, window_seconds=1.0)
        self.assertEqual(fresh.effective_validity(now=100.5), Validity.VALID)
        self.assertEqual(fresh.effective_validity(now=200.0), Validity.STALE)
        self.assertEqual(fresh.value, 1.0, "ageing a sample does not change its value; "
                                           "the validity is what changes")

    def test_the_effective_validity_of_an_unavailable_sample_stays_unavailable(self):
        s = _s(None, Validity.UNAVAILABLE, sampled_at=0.0,
               unavailable_reason="no counter exists")
        self.assertEqual(s.effective_validity(now=10_000.0), Validity.UNAVAILABLE)

    def test_a_recorded_stale_validity_is_honoured(self):
        s = _s(1.0, Validity.STALE, sampled_at=100.0, window_seconds=1000.0)
        self.assertEqual(s.effective_validity(now=100.1), Validity.STALE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
