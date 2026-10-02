"""The qualification runner's report, and the guard that it cannot flatter a device.

This is the module whose output goes in the M0 report, and the product's central claim is
that every number on screen was measured. So the tests here are not about arithmetic -
they are about the cases where a report is *tempting* to be kind:

- A device that failed. `TargetResult` carries `error` and a pile of `None`s for exactly
  that state, and `render_table` has to print `unavailable` where a number would go. A
  NPU that will not compile must read as a NPU that will not compile, never as a NPU
  that ran in 0.00 ms and found nothing.
- A device that silently fell back. `EXECUTION_DEVICES` is the only proof of placement
  (AGENTS.md rule 5), so a row that requested NPU and reports `('CPU',)` has to be
  visible as a fallback in the same glance as its error text.

Both are pinned here with synthetic results, so they hold on a machine with no NPU.
Whether the *real* devices pass is `tools/qualify_m0.py --requests 100`, and the M0
report carries its output verbatim.
"""

import unittest

from tools.qualify_m0 import (
    FALLBACK_MARKER,
    UNAVAILABLE,
    TargetResult,
    percentile,
    render_table,
)


def _r(device, error=None, devices=("CPU",), requests=100, p50=10.0):
    return TargetResult(device_id=device, full_name=device, compile_seconds=1.0,
                        warmup_seconds=0.5, requests=requests, latency_ms_p50=p50,
                        latency_ms_p95=12.0, latency_ms_min=9.0, latency_ms_max=20.0,
                        requests_per_second=100.0, execution_devices=devices,
                        rss_bytes=1, error=error)


class ReportIsHonest(unittest.TestCase):
    def test_missing_numbers_render_as_unavailable_never_zero(self):
        r = TargetResult(device_id="NPU", full_name="Intel(R) AI Boost",
                         compile_seconds=None, warmup_seconds=None, requests=0,
                         latency_ms_p50=None, latency_ms_p95=None, latency_ms_min=None,
                         latency_ms_max=None, requests_per_second=None,
                         execution_devices=(), rss_bytes=None,
                         error="compile failed: unsupported shape")
        table = render_table([r])
        self.assertIn("unavailable", table.lower())
        self.assertNotIn("0.00 ms", table)

    def test_error_is_shown_with_the_fallback_route(self):
        table = render_table([_r("NPU", error="compile failed, fell back to CPU",
                                 devices=("CPU",))])
        self.assertIn("compile failed", table)
        self.assertIn("CPU", table)

    def test_execution_devices_are_printed(self):
        self.assertIn("CPU", render_table([_r("NPU", devices=("NPU",))]))

    def test_a_failed_target_renders_nothing_that_looks_like_a_result(self):
        r = _r("NPU", error="unsupported shape on this driver", devices=())
        table = render_table([r])
        self.assertIn("unsupported shape", table)
        self.assertNotIn("100", table)
        self.assertNotIn("0.00", table)


class FallbackIsVisible(unittest.TestCase):
    """A row that asked for one device and ran on another is a fallback, and says so.

    This is the case AGENTS.md rule 5 exists for. `EXECUTION_DEVICES` is the only proof
    of placement, and the failure mode is not a wrong number - it is a *right* number
    attached to the wrong device's name, which reads as a successful NPU run.
    """

    def test_a_silent_fallback_is_labelled_a_fallback(self):
        r = _r("NPU", error=None, devices=("CPU",), requests=100, p50=9.0)
        self.assertIn(FALLBACK_MARKER, render_table([r]).lower())

    def test_a_correctly_placed_row_is_not_labelled_a_fallback(self):
        table = render_table([_r("CPU", devices=("CPU",))]).lower()
        self.assertNotIn(FALLBACK_MARKER, table)

    def test_the_gpu_dot_zero_suffix_still_counts_as_placed_on_gpu(self):
        # Measured: the GPU plugin reports ['GPU.0'], the CPU ['CPU'] and the NPU the
        # bare string 'NPU'. A substring test would call 'GPU' a match on 'GPU.0' and
        # also on 'GP', so the accessor is is_device, not `in`.
        on_gpu = render_table([_r("GPU", devices=("GPU.0",))]).lower()
        on_cpu = render_table([_r("GPU", devices=("CPU",))]).lower()
        self.assertNotIn(FALLBACK_MARKER, on_gpu)
        self.assertIn(FALLBACK_MARKER, on_cpu)


class FailedTargetCannotBeStamped(unittest.TestCase):
    """A failure is rendered as a failure, whatever the record happens to carry.

    Note what is *not* asserted here: that a `TargetResult` carrying both an `error` and
    a set of measurements cannot be constructed. The brief's own
    `test_error_is_shown_with_the_fallback_route` builds exactly that - a failed NPU with
    a populated `requests` and a `('CPU',)` placement - and requires the error to be
    visible, so refusing the construction would break a specified test. The invariant is
    therefore enforced where it actually matters, in the renderer: an errored row's
    numbers are suppressed no matter what the row holds. A half-populated failure is
    still constructible; it can never be *printed* as a result.
    """

    def test_an_errored_row_hides_a_populated_latency(self):
        row = _r("NPU", error="compile failed", devices=("NPU",), p50=1.0)
        table = render_table([row])
        self.assertIn("compile failed", table)
        self.assertIn(UNAVAILABLE, table)
        self.assertNotIn("1.00 ms", table)

    def test_an_errored_row_hides_a_populated_request_count(self):
        table = render_table([_r("NPU", error="compile failed", devices=())])
        self.assertNotIn("100", table)

    def test_an_errored_row_still_names_its_fallback_route(self):
        # The error text and the placement reading are evidence, so they survive the
        # suppression. A failure that hides why it failed is not honest, it is quiet.
        table = render_table([_r("NPU", error="compile failed, fell back to CPU",
                                 devices=("CPU",))])
        self.assertIn("compile failed", table)
        self.assertIn("CPU", table)

    def test_a_zero_latency_is_refused_rather_than_rounded_or_kept(self):
        # 0.0 ms is not a fast inference, it is an unmeasured one. A caller that wants
        # to record "too fast to measure" has to say so in the error, not in a number.
        with self.assertRaises(ValueError):
            _r("NPU", p50=0.0)

    def test_a_negative_request_count_is_refused(self):
        with self.assertRaises(ValueError):
            _r("NPU", requests=-1)

    def test_an_empty_error_string_is_refused(self):
        # Unavailable is a stated reason. A blank reason is the worst of both worlds:
        # the row reads as a failure that cannot say why.
        with self.assertRaises(ValueError):
            _r("NPU", error="   ")

    def test_a_plain_failure_reports_no_measurements_and_no_placement(self):
        from tools.qualify_m0 import _failed
        row = _failed("NPU", "compile failed")
        self.assertEqual(row.requests, 0)
        self.assertEqual(row.execution_devices, ())
        for name in ("compile_seconds", "warmup_seconds", "latency_ms_p50",
                     "latency_ms_p95", "latency_ms_min", "latency_ms_max",
                     "requests_per_second", "rss_bytes"):
            self.assertIsNone(getattr(row, name), f"{name} must be unmeasured")


class PercentileIsHonest(unittest.TestCase):
    """p50/p95 are computed from the samples, not chosen.

    The temptation these guard against is picking the percentile that makes a device
    look good. `percentile` is a plain nearest-rank order statistic with no
    interpolation and no smoothing, so p95 of 100 samples is the 95th of the 100
    samples and nothing else.
    """

    def test_p50_of_one_to_one_hundred_is_fifty(self):
        self.assertEqual(percentile(list(range(1, 101)), 50.0), 50)

    def test_p50_of_an_odd_count_is_the_middle_sample(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 50.0), 2)

    def test_p95_of_one_to_one_hundred_is_ninety_five(self):
        self.assertEqual(percentile(list(range(1, 101)), 95.0), 95)

    def test_an_empty_sample_set_has_no_percentile(self):
        with self.assertRaises(ValueError):
            percentile([], 50.0)

    def test_an_out_of_range_percentile_is_refused(self):
        with self.assertRaises(ValueError):
            percentile([1.0, 2.0], 101.0)


if __name__ == "__main__":
    unittest.main()
