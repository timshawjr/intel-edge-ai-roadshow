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

import time
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


class PlacementIsReadFromTheRequestNotAssumed(unittest.TestCase):
    """The `EXECUTION_DEVICES` read itself is the untested part (F7).

    Everything above the placement line had a test and the placement line did not.
    `qualify_target` was only ever called in the suite with an *invalid* `num_requests`,
    so its whole happy path - compile, warmup, timed loop - was exercised zero times. A
    mutant that replaced

        normalise_execution_devices(
            request.get_compiled_model().get_property("EXECUTION_DEVICES"))

    with the silent fallback `placement = (device_id,)` left all 449 tests green. That is
    the worst mutant in this file: it fabricates a *clean NPU row* for a run that never
    touched an NPU, which is exactly what AGENTS.md rule 5 exists to forbid, and it would
    be invisible in the one table the M0 report leads with.

    So two tests, from opposite directions:

    - **Hermetic, no hardware.** A fake core whose `get_property` reports a *different*
      device than the one requested. `qualify_target` must return a row that is **not**
      placed on the requested device and that renders `[fallback]`. The fake reports
      `'CPU'`; if the mutant is installed, the row claims to be on `'NPU'` and is
      rendered as a clean success. This runs on a machine with no NPU at all, which is
      the point: the guard must not need an NPU to fire.
    - **Real, CPU, `num_requests=1`.** The happy path end to end against the pinned
      model, asserting the placement is the device that was asked for. This is the first
      and only test that runs `qualify_target` successfully, so it is also what stops a
      later edit from breaking the timed loop in a way nothing notices.
    """

    def _fake_spec(self):
        from app.inference.devices import TargetSpec
        from tools.build_sanity_fixture import pinned_input_shape
        return TargetSpec(device_id="NPU", model_path="fake.xml",
                          input_shape=pinned_input_shape())

    class _FakeCompiledModel:
        """`ov.CompiledModel`'s shape: `create_infer_request` on the compiled model, and
        `EXECUTION_DEVICES` read back from it through the request's own compiled model."""

        def __init__(self, reported):
            self._reported = reported
            self.infer_calls = 0
            self.request = PlacementIsReadFromTheRequestNotAssumed._FakeRequest(self)

        def get_property(self, name):
            assert name == "EXECUTION_DEVICES", f"unexpected property {name!r}"
            return self._reported

        def create_infer_request(self):
            return self.request

    class _FakeRequest:
        def __init__(self, compiled):
            self._compiled = compiled

        def infer(self, feeds):
            # A real, if tiny, cost. Without it the timed loop measures a few hundred
            # nanoseconds and the fake row renders `0.00 ms` in every latency column -
            # which is the exact number this project must never show, appearing in a test
            # that has nothing to do with it.
            time.sleep(0.002)
            self._compiled.infer_calls += 1

        def get_compiled_model(self):
            return self._compiled

    class _FakeCore:
        """A core that compiles instantly and reports whatever it was built with."""

        def __init__(self, reported=("CPU",)):
            self.compiled = PlacementIsReadFromTheRequestNotAssumed._FakeCompiledModel(
                reported)

        def get_property(self, device_id, name):
            return "Intel(R) AI Boost"

        def read_model(self, path):
            return path

        def compile_model(self, model, device_id):
            return self.compiled

    def test_a_core_that_reports_another_device_gives_a_fallback_row(self):
        from tools.qualify_m0 import qualify_target
        import numpy as np
        core = self._FakeCore(reported=("CPU",))
        result = qualify_target(core, self._fake_spec(), np.zeros((8, 8, 3), np.uint8), 2)
        self.assertFalse(result.failed, f"the happy path failed: {result.error}")
        self.assertEqual(result.execution_devices, ("CPU",),
                         "the placement must be what the compiled model reported, not "
                         "what was asked for")
        self.assertFalse(result.placed_on_requested_device)
        table = render_table([result]).lower()
        self.assertIn(FALLBACK_MARKER, table,
                      "a row that ran on the wrong device must be flagged, or a NPU row "
                      "carrying CPU numbers reads as a clean NPU result")
        self.assertIn("asked for npu", table.lower())

    def test_a_core_that_reports_the_requested_device_gives_a_clean_row(self):
        from tools.qualify_m0 import qualify_target
        import numpy as np
        core = self._FakeCore(reported=("NPU",))
        result = qualify_target(core, self._fake_spec(), np.zeros((8, 8, 3), np.uint8), 2)
        self.assertFalse(result.failed, f"the happy path failed: {result.error}")
        self.assertTrue(result.placed_on_requested_device)
        self.assertNotIn(FALLBACK_MARKER, render_table([result]).lower())

    def test_the_bare_string_shape_is_normalised_the_way_the_npu_reports_it(self):
        """Measured: the NPU plugin returns the bare string `'NPU'`, not `['NPU']`.

        A silent `tuple()` on that would yield `('N', 'P', 'U')` and render as a
        three-device row. `normalise_execution_devices` exists for exactly this, so the
        fake core reports the measured shape.
        """
        from tools.qualify_m0 import qualify_target
        import numpy as np
        core = self._FakeCore(reported="NPU")
        result = qualify_target(core, self._fake_spec(), np.zeros((8, 8, 3), np.uint8), 2)
        self.assertEqual(result.execution_devices, ("NPU",))
        self.assertTrue(result.placed_on_requested_device)

    def test_the_placement_read_happens_after_the_first_inference(self):
        """AGENTS.md rule 5: the reading must follow a real inference, not precede one.

        Before the warmup, `EXECUTION_DEVICES` is a claim about a placement that has not
        happened. The fake compiled model counts the inferences, so this asserts the order
        rather than the comment describing it.
        """
        from tools.qualify_m0 import qualify_target
        import numpy as np
        core = self._FakeCore(reported=("NPU",))
        seen_at_read = []

        class _Counting(PlacementIsReadFromTheRequestNotAssumed._FakeCompiledModel):
            def get_property(self, name):
                seen_at_read.append(self.infer_calls)
                return super().get_property(name)

        core.compiled = _Counting("NPU")
        qualify_target(core, self._fake_spec(), np.zeros((8, 8, 3), np.uint8), 3)
        self.assertTrue(seen_at_read, "EXECUTION_DEVICES was never read at all")
        self.assertGreaterEqual(seen_at_read[0], 1,
                                "placement was read before any inference ran")

    def test_a_real_cpu_run_reports_the_device_it_was_asked_for(self):
        """The happy path, for real, against the pinned model.

        `num_requests=1` plus one warmup: this is the shortest run that is genuinely a
        measurement, and it is the only test in the suite that executes
        `qualify_target`'s full body - compile, warmup, timed loop, placement read, rss.
        """
        import openvino as ov
        from app.inference.devices import TargetSpec
        from tools.build_sanity_fixture import ir_path, pinned_input_shape
        from tools.qualify_m0 import qualify_target

        if "CPU" not in tuple(ov.Core().available_devices):
            self.skipTest("no CPU plugin on this machine")
        spec = TargetSpec(device_id="CPU", model_path=ir_path(),
                          input_shape=pinned_input_shape())
        result = qualify_target(ov.Core(), spec, _frame(), 1)
        self.assertFalse(result.failed, f"a real CPU run failed: {result.error}")
        self.assertEqual(result.execution_devices, ("CPU",))
        self.assertTrue(result.placed_on_requested_device)
        self.assertEqual(result.requests, 1)
        for name in ("compile_seconds", "warmup_seconds", "latency_ms_p50",
                     "latency_ms_p95", "latency_ms_min", "latency_ms_max",
                     "requests_per_second", "rss_bytes"):
            self.assertIsNotNone(getattr(result, name), f"{name} was not measured")


class UnavailableIsNeverZeroForAnyRow(unittest.TestCase):
    """The `unavailable` guard on the row that carries M0's headline numbers (F8).

    `_fmt`'s whole job is `None -> "unavailable"`, and the brief's own test only covered
    it on a **failed** row, where `render_table` short-circuits every numeric column to
    `UNAVAILABLE` without calling `_fmt` at all. So the guard had no test: changing
    `return UNAVAILABLE` to `return "0.00" + suffix` left the suite green.

    The mutant is reachable and it is the single thing AGENTS.md rule 15 forbids. A
    `TargetResult` with `error=None` and `latency_ms_p95=None` is a row that rendered
    `PASS` and printed `0.00 ms` in the p95 column of the table the M0 report leads
    with. `TargetResult` now refuses to be built in that state (see the constructor), and
    the render guard is pinned independently below so that either half failing is
    visible.
    """

    def _success_with_a_hole(self, **overrides):
        kwargs = dict(device_id="NPU", full_name="Intel(R) AI Boost", compile_seconds=1.69,
                      warmup_seconds=0.01, requests=100, latency_ms_p50=5.27,
                      latency_ms_p95=8.54, latency_ms_min=4.09, latency_ms_max=17.51,
                      requests_per_second=168.83, execution_devices=("NPU",),
                      rss_bytes=503975936, error=None)
        kwargs.update(overrides)
        return kwargs

    def test_a_non_failed_row_with_a_missing_p95_cannot_be_constructed(self):
        with self.assertRaises(ValueError) as caught:
            TargetResult(**self._success_with_a_hole(latency_ms_p95=None))
        self.assertIn("latency_ms_p95", str(caught.exception))

    def test_a_non_failed_row_with_no_measurements_at_all_cannot_be_constructed(self):
        with self.assertRaises(ValueError):
            TargetResult(device_id="NPU", full_name="Intel(R) AI Boost",
                         compile_seconds=None, warmup_seconds=None, requests=100,
                         latency_ms_p50=None, latency_ms_p95=None, latency_ms_min=None,
                         latency_ms_max=None, requests_per_second=None,
                         execution_devices=("NPU",), rss_bytes=None, error=None)

    def test_a_non_failed_row_with_zero_requests_cannot_be_constructed(self):
        """Minor 17: `requests == 0` on a success is a row that measured nothing."""
        with self.assertRaises(ValueError) as caught:
            TargetResult(**self._success_with_a_hole(requests=0))
        self.assertIn("requests", str(caught.exception))

    def test_the_formatter_itself_never_returns_zero_for_a_missing_value(self):
        """`_fmt` in isolation, which is what the mutant changed.

        `render_table` cannot cover this on its own: it only reaches `_fmt` on a
        non-failed row, and those can no longer carry a `None` - so if the guard ever
        moves again, only this assertion sees it.
        """
        from tools.qualify_m0 import _fmt, _fmt_bytes
        self.assertEqual(_fmt(None, " ms"), UNAVAILABLE)
        self.assertEqual(_fmt_bytes(None), UNAVAILABLE)
        self.assertNotIn("0.00", _fmt(None, " ms"))

    def test_a_row_that_cannot_carry_a_hole_still_renders_unavailable_if_one_arrives(self):
        """The renderer guard, reached through a record the constructor no longer allows.

        The renderer is the last line of defence and is deliberately not dependent on the
        constructor: a hand-built object with `object.__setattr__` (or a future field) can
        still carry a `None`, and `unavailable` is what must appear.
        """
        row = TargetResult(**self._success_with_a_hole())
        object.__setattr__(row, "latency_ms_p95", None)
        table = render_table([row])
        self.assertIn(UNAVAILABLE, table)
        self.assertNotIn("0.00 ms", table)


def _frame():
    """The pinned fixture frame, so a real inference is what gets timed."""
    from tools.qualify_m0 import _fixture_image
    return _fixture_image()


if __name__ == "__main__":
    unittest.main()
