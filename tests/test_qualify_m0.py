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
        """The placement column carries what the runtime actually reported.

        The brief's version of this test asserted `"CPU" in table` for a row whose
        `device_id` *and* `execution_devices` are both `NPU`. There is no `CPU` anywhere
        in that record, so the assertion could only pass because the legend mentions the
        three device ids - and it could not fail while the legend stood, which is the
        signature of a vacuous test. The name says what is being tested, so the name is
        what the assertion now checks: the reported value appears in the row.
        """
        table = render_table([_r("NPU", devices=("NPU",))])
        self.assertIn("NPU", table)
        # And the device that was asked for is named, so the row cannot be mistaken for
        # one that silently fell back to the CPU.
        self.assertNotIn(FALLBACK_MARKER, table.lower())

    def test_an_empty_execution_devices_reading_is_not_rendered_as_a_device(self):
        self.assertIn(UNAVAILABLE, render_table([_r("NPU", error="x", devices=())]))

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


class AGuardRaisesRatherThanFailingTheRun(unittest.TestCase):
    """A bad argument is a **failed row**, not an exception out of the tool.

    `qualify_target`'s contract says any exception is caught and returned as `error`. The
    first version broke that contract for its own argument guard: `entry_started` was
    assigned inside the `try` and `del`'d in the `finally`, so when the `num_requests`
    guard raised, the `finally` raised `UnboundLocalError: cannot access local variable
    'entry_started'` and **discarded the `_failed(...)` return**. Measured: both
    `num_requests=0` and `num_requests=-5` raised instead of returning a row.

    That is the worst shape a bug can take here. The caller asked an impossible question,
    the tool was supposed to answer "no, and here is why", and instead the process died -
    so a mistyped flag on a booth takes down the qualification run rather than printing
    one row saying the request count was invalid.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import openvino as ov
        except ImportError:
            raise unittest.SkipTest("openvino is not importable")
        if "CPU" not in tuple(ov.Core().available_devices):
            raise unittest.SkipTest("no CPU plugin on this machine")

    def _spec(self):
        from app.inference.devices import TargetSpec
        from tools.build_sanity_fixture import ir_path, pinned_input_shape
        return TargetSpec(device_id="CPU", model_path=ir_path(),
                          input_shape=pinned_input_shape())

    def test_zero_requests_returns_a_failed_row(self):
        import openvino as ov
        from tools.qualify_m0 import qualify_target
        result = qualify_target(ov.Core(), self._spec(), _frame(), 0)
        self.assertTrue(result.failed)
        self.assertIn("num_requests", result.error)
        self.assertEqual(result.requests, 0)
        self.assertEqual(result.execution_devices, ())
        self.assertIsNone(result.latency_ms_p50)

    def test_a_negative_request_count_returns_a_failed_row(self):
        import openvino as ov
        from tools.qualify_m0 import qualify_target
        result = qualify_target(ov.Core(), self._spec(), _frame(), -5)
        self.assertTrue(result.failed)
        self.assertIn("num_requests", result.error)
        self.assertEqual(result.requests, 0)

    def test_a_non_integer_request_count_returns_a_failed_row(self):
        import openvino as ov
        from tools.qualify_m0 import qualify_target
        result = qualify_target(ov.Core(), self._spec(), _frame(), "100")
        self.assertTrue(result.failed)

    def test_a_failed_row_renders_without_a_startup_figure(self):
        """A row that never ran must not carry a time to its first inference.

        `to_first_inference_s` was recorded unconditionally, so a device absent from
        `available_devices` - which skips `qualify_target` entirely - got a row reading
        about two millionths of a second. That is a *time to first inference* for a row
        where no inference happened, and it sits in the same JSON as three honest rows
        where a reader has no way to tell them apart.
        """
        from tools.qualify_m0 import TargetResult, _row_seconds
        failed = TargetResult(device_id="NPU", full_name="", compile_seconds=None,
                              warmup_seconds=None, requests=0, latency_ms_p50=None,
                              latency_ms_p95=None, latency_ms_min=None,
                              latency_ms_max=None, requests_per_second=None,
                              execution_devices=(), rss_bytes=None,
                              error="not in available_devices")
        self.assertIsNone(_row_seconds(failed, 2.7e-06))
        ok = _r("CPU", devices=("CPU",))
        self.assertAlmostEqual(_row_seconds(ok, 1.25), 1.25)


class FullNameIsNotSilentlyTruncated(unittest.TestCase):
    """A cut string must say it was cut.

    The first version truncated `full_name` at 34 characters with no marker, so
    `Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)` printed as
    `Intel(R) Arc(TM) 140V GPU (16GB) (` and a reader could not tell a cut string from
    the whole name. That trailing `(iGPU)` is the difference between an integrated and a
    discrete GPU, which is the first thing a visitor asks about this hardware.
    """

    def test_a_name_that_fits_is_left_alone(self):
        from tools.qualify_m0 import _column
        self.assertEqual(_column("Intel(R) AI Boost"), "Intel(R) AI Boost")

    def test_a_name_that_does_not_fit_is_marked(self):
        from tools.qualify_m0 import FULL_NAME_WIDTH, _column
        long_name = "Intel(R) Arc(TM) 140V GPU (16GB) (iGPU) " + "x" * 40
        self.assertGreater(len(long_name), FULL_NAME_WIDTH)
        cut = _column(long_name)
        self.assertTrue(cut.endswith("..."), f"{cut!r} does not say it was cut")
        self.assertLessEqual(len(cut), FULL_NAME_WIDTH)
        self.assertLess(len(cut), len(long_name))

    def test_this_machines_gpu_name_keeps_its_qualifier(self):
        from tools.qualify_m0 import FULL_NAME_WIDTH, _column
        name = "Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)"
        self.assertLessEqual(len(name), FULL_NAME_WIDTH,
                             "the column is narrower than this machine's real GPU name, "
                             "so the integrated/discrete qualifier is being hidden")
        self.assertEqual(_column(name), name)

    def test_the_truncated_row_still_carries_the_whole_name_in_the_json(self):
        long_name = "Intel(R) Arc(TM) 140V GPU (16GB) (iGPU) with a very long suffix"
        row = _r("GPU", devices=("GPU.0",))
        row = type(row)(**{**{f: getattr(row, f) for f in row.__dataclass_fields__},
                           "full_name": long_name})
        payload = row.as_dict()
        self.assertEqual(payload["full_name"], long_name,
                         "the JSON record must carry the untruncated name, since that is "
                         "what a reader checks the table against")


def _frame():
    """The pinned fixture frame, so a real inference is what gets timed."""
    from tools.qualify_m0 import _fixture_image
    return _fixture_image()


if __name__ == "__main__":
    unittest.main()
