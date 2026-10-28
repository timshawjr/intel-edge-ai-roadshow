"""The sustained paired trials in the suite: does a device comparison mean anything?

`tools/sustained_trials.py` produced the M4 Task 4 measurement. This runs **the same code**
for the properties that have no excuse - a fallback reported as its request, a trial that
carries a number it did not measure, a run that is not interleaved, a record that cannot be
read back - and one **real** CPU trial in a fresh process, because the child-spawn path is
where a measurement tool is most likely to be quietly wrong.

## What is asserted, and what is only recorded

The measurement itself is a **result, not a gate**: this suite does not assert that the GPU
is faster than the NPU, or that any device reaches a rate. Those are the numbers the report
quotes, with their conditions, and a threshold here would turn a machine's current state
into a test that fails on a warm afternoon. What *is* asserted is everything that has no
excuse:

- a fallback - NPU asked for, CPU confirmed - cannot be built into a `TrialResult` under
  the requested device's name, and the table prints the placement that ran;
- every required figure is present on a measured row and **absent** on an unmeasured one;
  no zero stands in for "not measured";
- `EXECUTION_DEVICES` is normalised through all three measured plugin shapes (bare `'NPU'`,
  `['CPU']`, `['GPU.0']`), and two readings in one window are refused;
- the repeats are interleaved with a rotating order, and every `(condition, repeat,
  device)` appears exactly once;
- a percentile is the nearest-rank statistic `app.main` uses, not an interpolation;
- one real 8 s CPU trial in a fresh process confirms its placement from
  `EXECUTION_DEVICES`, produces a latency distribution, and still emits the counter-band
  presence event.

## Cost

The fake-runner tests are instant. The one real trial is a **shared 8 s CPU window in a
fresh process**, and the class measured **14.2 s** for its three tests on 2026-10-27. It is
skipped - loudly, with the reason - on a machine that cannot run a CPU worker. The full
27-trial campaign is a tool run, not a test: it measured 758.8 s (12 min 39 s) and belongs
in `docs/qualification/`, not in a red/green.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools import sustained_trials as st  # noqa: E402
from tools.sustained_trials import (  # noqa: E402
    CONDITION_BOOTH,
    CONDITION_SOURCE,
    CONDITION_SUSTAINED,
    TrialResult,
    TrialSpec,
    format_table,
    interleaved_order,
    percentile,
    placement_from_readings,
    run_paired_trials,
    run_trial,
)

DEVICES = ("CPU", "GPU", "NPU")


def _result(**overrides) -> TrialResult:
    """A measured `sustained` row, valid by construction, with any field overridden.

    Built from real field names so a rename breaks the tests rather than silently leaving a
    default in place.
    """
    base = dict(
        condition=CONDITION_SUSTAINED, device="GPU", requested="GPU",
        confirmed=("GPU.0",), fell_back=False, repeat=0, seconds=20.0, frames=6000,
        throughput_fps=300.0, latency_p50_ms=3.1, latency_p95_ms=3.7, latency_max_ms=9.6,
        latency_samples=6000, frame_age_p50_ms=None, frame_age_p95_ms=None,
        frame_age_samples=0, drops=None, rss_start=400.0, rss_end=410.0, feed_fps=None,
        warmup_s=3.0, cold30_p50_ms=3.2,
    )
    base.update(overrides)
    return TrialResult(**base)


def _pipeline_result(**overrides) -> TrialResult:
    base = dict(
        condition=CONDITION_BOOTH, device="NPU", requested="NPU", confirmed=("NPU",),
        fell_back=False, repeat=0, seconds=20.0, frames=300, throughput_fps=14.99,
        latency_p50_ms=6.4, latency_p95_ms=7.1, latency_max_ms=12.0, latency_samples=300,
        frame_age_p50_ms=31.0, frame_age_p95_ms=44.0, frame_age_samples=300, drops=0,
        rss_start=500.0, rss_end=505.0, feed_fps=14.985, warmup_s=3.0,
    )
    base.update(overrides)
    return TrialResult(**base)


def _payload(**overrides) -> dict:
    """A child's JSON payload for a measured pipeline trial."""
    base = {
        "condition": CONDITION_BOOTH, "device": "GPU", "requested": "GPU",
        "confirmed": ["GPU.0"], "fell_back": False, "repeat": 1, "seconds": 20.02,
        "frames": 300, "throughput_fps": 14.98, "latency_p50_ms": 4.4,
        "latency_p95_ms": 5.2, "latency_max_ms": 9.7, "latency_samples": 300,
        "frame_age_p50_ms": 31.2, "frame_age_p95_ms": 45.5, "frame_age_samples": 300,
        "drops": 0, "rss_start": 812.3, "rss_end": 815.0, "feed_fps": 14.985,
        "warmup_s": 3.0, "cold30_p50_ms": None, "events": ["1:8:person:presence:x"],
        "event_counts": {"zone_occupancy": 1}, "errors": [],
    }
    base.update(overrides)
    return base


class TrialResultRefusals(unittest.TestCase):
    """A record that cannot hold a bad number is a record nobody has to check.

    Each test here constructs the fault the M4 brief names and asserts the record refuses
    it. The fallback case is the one the whole milestone exists for.
    """

    maxDiff = None

    def test_a_fallback_cannot_be_reported_as_its_requested_target(self):
        """NPU asked for, CPU confirmed, `fell_back=False` - the shape review focus #4 names."""
        with self.assertRaises(ValueError) as caught:
            _pipeline_result(device="NPU", requested="NPU", confirmed=("CPU",),
                             fell_back=False)
        self.assertIn("requested target", str(caught.exception))

    def test_a_fell_back_row_that_names_its_request_is_refused(self):
        """The inverse: a row claiming a fallback while confirming the requested device."""
        with self.assertRaises(ValueError) as caught:
            _pipeline_result(device="NPU", requested="NPU", confirmed=("NPU",),
                             fell_back=True)
        self.assertIn("which is the requested device", str(caught.exception))

    def test_an_unconfirmed_row_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            _pipeline_result(confirmed=(), fell_back=False)
        self.assertIn("never proved", str(caught.exception))

    def test_a_confirmed_row_without_a_fallback_decision_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            _pipeline_result(fell_back=None)
        self.assertIn("must say whether it fell", str(caught.exception))

    def test_percentiles_must_be_ordered(self):
        with self.assertRaises(ValueError):
            _result(latency_p50_ms=9.0, latency_p95_ms=4.0)
        with self.assertRaises(ValueError):
            _result(latency_p50_ms=3.0, latency_p95_ms=9.0, latency_max_ms=4.0)

    def test_frame_age_belongs_only_to_a_pipeline_condition(self):
        """`sustained` has no queue, so a frame age there is a number about nothing."""
        with self.assertRaises(ValueError) as caught:
            _result(frame_age_p50_ms=30.0, frame_age_p95_ms=40.0)
        self.assertIn("no queue", str(caught.exception))
        with self.assertRaises(ValueError):
            _result(drops=0)
        with self.assertRaises(ValueError):
            _pipeline_result(drops=None)
        with self.assertRaises(ValueError):
            _pipeline_result(frame_age_p50_ms=None)
        # And the pipeline row that does carry all three is accepted.
        self.assertEqual(_pipeline_result().drops, 0)

    def test_a_measured_row_missing_a_figure_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            _result(seconds=None)
        self.assertIn("['seconds']", str(caught.exception))

    def test_a_throughput_without_a_distribution_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            _result(latency_samples=0)
        self.assertIn("half a result", str(caught.exception))

    def test_an_unmeasured_row_cannot_carry_a_measurement_or_a_placement(self):
        with self.assertRaises(ValueError) as caught:
            _result(unmeasured="child exited 3", confirmed=(), fell_back=None, frames=5)
        self.assertIn("no measurements", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            _result(unmeasured="child exited 3", confirmed=("GPU.0",), fell_back=False)
        self.assertIn("cannot carry a placement", str(caught.exception))
        # The honest empty row is accepted.
        empty = _result(
            unmeasured="child exited 3", confirmed=(), fell_back=None, seconds=None,
            frames=None, throughput_fps=None, latency_p50_ms=None, latency_p95_ms=None,
            latency_max_ms=None, latency_samples=0, rss_start=None, rss_end=None,
            cold30_p50_ms=None)
        self.assertEqual(empty.unmeasured, "child exited 3")
        self.assertEqual(empty.confirmed, ())
        self.assertIsNone(empty.seconds)

    def test_a_zero_resident_set_is_not_a_measurement(self):
        with self.assertRaises(ValueError) as caught:
            _result(rss_start=0.0)
        self.assertIn("resident set", str(caught.exception))

    def test_the_measurements_are_frozen(self):
        row = _result()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            row.throughput_fps = 900.0  # type: ignore[misc]

    def test_placement_reads_the_confirmed_device_never_the_label(self):
        row = _result(device="GPU", requested="GPU", confirmed=("GPU.0",))
        self.assertEqual(row.placement, "GPU.0")
        self.assertEqual(_result(device="GPU", requested="GPU", confirmed=("CPU",),
                                 fell_back=True).placement, "CPU (fell back from GPU)")


class PlacementConfirmation(unittest.TestCase):
    """`EXECUTION_DEVICES` in all three measured shapes, and the comparisons it allows."""

    def test_each_measured_shape_normalises(self):
        """Bare `'NPU'`, `['CPU']` and `['GPU.0']` all match the device that was asked for."""
        for requested, reading in (("CPU", ["CPU"]), ("GPU", ["GPU.0"]), ("NPU", "NPU")):
            with self.subTest(requested=requested, reading=reading):
                confirmed, fell_back = placement_from_readings(requested, [reading])
                self.assertFalse(fell_back)
                self.assertTrue(st.is_device(confirmed, requested))

    def test_a_fallback_reading_is_reported_as_what_ran(self):
        confirmed, fell_back = placement_from_readings("NPU", [["CPU"]])
        self.assertEqual(confirmed, ("CPU",))
        self.assertTrue(fell_back)
        self.assertNotIn("NPU", confirmed)

    def test_two_readings_in_one_window_are_refused(self):
        with self.assertRaises(ValueError) as caught:
            placement_from_readings("GPU", [["GPU.0"], ["CPU"]])
        self.assertIn("changed during one measured window", str(caught.exception))

    def test_no_reading_is_a_refusal_not_an_empty_placement(self):
        with self.assertRaises(ValueError) as caught:
            placement_from_readings("GPU", [])
        self.assertIn("only proved by an inference", str(caught.exception))

    def test_an_empty_property_reading_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            placement_from_readings("NPU", ["", "NPU"])
        self.assertIn("empty", str(caught.exception))

    def test_the_same_reading_repeated_is_one_placement(self):
        confirmed, fell_back = placement_from_readings("NPU", ["NPU"] * 300)
        self.assertEqual(confirmed, ("NPU",))
        self.assertFalse(fell_back)


class PercentileRules(unittest.TestCase):
    """Nearest rank, no interpolation, and the same answer the application gives."""

    def test_it_agrees_with_the_application_own_percentile(self):
        from app.main import _percentile

        values = [4.0, 1.0, 3.0, 2.0, 9.0, 7.0, 8.0, 6.0, 5.0, 0.5]
        for fraction in (0.5, 0.95, 1.0):
            with self.subTest(fraction=fraction):
                self.assertEqual(percentile(values, fraction),
                                 _percentile(values, fraction))

    def test_no_sample_between_two_taken_values_is_invented(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(percentile(values, 0.50), 2.0)   # not 2.5
        self.assertEqual(percentile(values, 0.95), 4.0)
        self.assertEqual(percentile(list(range(1, 21)), 0.95), 19.0)

    def test_an_empty_sequence_has_no_percentile(self):
        self.assertIsNone(percentile([], 0.5))

    def test_a_fraction_outside_the_range_is_refused(self):
        for fraction in (0.0, -0.1, 1.5):
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    percentile([1.0], fraction)


class Interleaving(unittest.TestCase):
    """The anti-drift rule, asserted rather than described."""

    def test_every_device_once_per_round_and_the_order_rotates(self):
        order = interleaved_order(DEVICES, 3)
        self.assertEqual(len(order), 9)
        for repeat in range(3):
            devices = [device for round_index, device in order if round_index == repeat]
            self.assertEqual(sorted(devices), sorted(DEVICES))
            # Round r starts at devices[r]: the trial that runs first changes every round.
            self.assertEqual(devices[0], DEVICES[repeat % len(DEVICES)])
        # No device runs twice in a row across a round boundary.
        flat = [device for _repeat, device in order]
        self.assertFalse(any(a == b for a, b in zip(flat, flat[1:])))

    def test_a_duplicate_device_is_refused(self):
        with self.assertRaises(ValueError):
            interleaved_order(("CPU", "GPU", "CPU"), 2)

    def test_a_float_or_zero_repeat_count_is_refused(self):
        with self.assertRaises(TypeError):
            interleaved_order(DEVICES, 2.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            interleaved_order(DEVICES, 0)

    def test_an_implicit_placement_spec_is_refused(self):
        for device in ("AUTO", "HETERO:GPU,CPU", "MULTI:GPU,CPU", "BATCH:GPU"):
            with self.subTest(device=device):
                with self.assertRaises(ValueError):
                    TrialSpec(condition=CONDITION_BOOTH, device=device, seconds=1.0,
                              warmup_s=0.5, repeat=0, order=1, feed_fps=15.0)


class PairedRun(unittest.TestCase):
    """`run_paired_trials` with an injected runner: the schedule and the rows it keeps."""

    def _runner(self, *, fail_on=None):
        calls: list[TrialSpec] = []

        def runner(spec: TrialSpec) -> TrialResult:
            calls.append(spec)
            if fail_on is not None and spec.device == fail_on:
                raise RuntimeError("the worker did not compile")
            if spec.device == "NPU":
                # A legitimate fallback: asked for the NPU, the runtime ran it on the CPU.
                return _pipeline_result(device=spec.device, requested="NPU",
                                        confirmed=("CPU",), fell_back=True,
                                        repeat=spec.repeat)
            return _pipeline_result(device=spec.device, requested=spec.device,
                                    confirmed=("CPU",) if spec.device == "CPU" else ("GPU.0",),
                                    fell_back=False, repeat=spec.repeat)
        return runner, calls

    def test_the_run_is_interleaved_and_every_pair_is_measured_once(self):
        runner, calls = self._runner()
        results = run_paired_trials(devices=DEVICES,
                                    conditions=(CONDITION_BOOTH, CONDITION_SOURCE),
                                    repeats=2, seconds=1.0, warmup_s=0.5, runner=runner,
                                    progress=lambda _line: None)
        self.assertEqual(len(results), 12)
        pairs = [(spec.condition, spec.repeat, spec.device) for spec in calls]
        self.assertEqual(len(pairs), len(set(pairs)))
        expected = interleaved_order(DEVICES, 2)
        for condition in (CONDITION_BOOTH, CONDITION_SOURCE):
            used = [(spec.repeat, spec.device) for spec in calls
                    if spec.condition == condition]
            self.assertEqual(tuple(used), expected)
        self.assertTrue(all(result.condition in (CONDITION_BOOTH, CONDITION_SOURCE)
                            for result in results))

    def test_a_fallback_row_keeps_the_placement_that_ran(self):
        runner, _calls = self._runner()
        results = run_paired_trials(devices=("NPU",), conditions=(CONDITION_BOOTH,),
                                    repeats=1, seconds=1.0, warmup_s=0.5, runner=runner,
                                    progress=lambda _line: None)
        row = results[0]
        self.assertEqual(row.device, "NPU")
        self.assertEqual(row.confirmed, ("CPU",))
        self.assertIs(row.fell_back, True)
        table = format_table([row])
        self.assertIn("FALLBACK from NPU", table)
        npu_row = [line for line in table.splitlines() if line.startswith("    booth")][0]
        cells = npu_row.split()
        # cond rep device requested confirmed fell_back ...
        self.assertEqual(cells[2], "NPU")     # the label
        self.assertEqual(cells[4], "CPU")     # what EXECUTION_DEVICES said
        self.assertEqual(cells[5], "yes")

    def test_a_failed_trial_is_recorded_and_the_run_continues(self):
        runner, _calls = self._runner(fail_on="GPU")
        results = run_paired_trials(devices=DEVICES, conditions=(CONDITION_BOOTH,),
                                    repeats=1, seconds=1.0, warmup_s=0.5, runner=runner,
                                    progress=lambda _line: None)
        self.assertEqual(len(results), 3)
        failed = [row for row in results if row.unmeasured is not None]
        self.assertEqual([row.device for row in failed], ["GPU"])
        self.assertIn("did not compile", failed[0].unmeasured)
        self.assertEqual(failed[0].confirmed, ())
        self.assertIsNone(failed[0].frames)
        table = format_table(results)
        self.assertIn("ERROR", table)
        # The measured rows around it are untouched.
        self.assertEqual(sum(1 for row in results if row.unmeasured is None), 2)

    def test_the_table_names_absence_rather_than_zero(self):
        table = format_table([_result(), _pipeline_result()])
        rows = [line for line in table.splitlines()[2:]]
        sustained_line = [line for line in rows if line.strip().startswith("sustained")][0]
        pipeline_line = [line for line in rows if line.strip().startswith("booth")][0]
        header = table.splitlines()[0].split()
        self.assertEqual(sustained_line.split()[header.index("age_p50")], "n/a")
        self.assertEqual(sustained_line.split()[header.index("age_p95")], "n/a")
        self.assertEqual(sustained_line.split()[header.index("drops")], "n/a")
        # The pipeline row's drops is a measured count, and its frame age a measured figure:
        # an absent value and a zero must not look the same in the table.
        self.assertEqual(pipeline_line.split()[header.index("drops")], "0")
        self.assertNotEqual(pipeline_line.split()[header.index("age_p50")], "n/a")


class Payloads(unittest.TestCase):
    """A child's JSON, refused when incomplete and read back exactly when complete."""

    def test_a_payload_missing_a_figure_is_refused(self):
        payload = _payload()
        del payload["frames"]
        with self.assertRaises(ValueError) as caught:
            TrialResult.from_payload(payload)
        self.assertIn("frames", str(caught.exception))

    def test_a_measured_payload_round_trips(self):
        row = TrialResult.from_payload(_payload())
        self.assertEqual(row.confirmed, ("GPU.0",))
        self.assertFalse(row.fell_back)
        self.assertEqual(row.counts, {"zone_occupancy": 1})
        self.assertEqual(row.events, ("1:8:person:presence:x",))
        again = TrialResult.from_payload(json.loads(json.dumps(row.as_dict())))
        self.assertEqual(again.as_dict(), row.as_dict())

    def test_an_unmeasured_payload_builds_a_row_with_no_numbers(self):
        row = TrialResult.from_payload({
            "condition": CONDITION_BOOTH, "device": "NPU", "requested": "NPU",
            "repeat": 2, "warmup_s": 3.0,
            "unmeasured": "RuntimeError: the worker did not compile"})
        self.assertIsNotNone(row.unmeasured)
        self.assertEqual(row.confirmed, ())
        self.assertIsNone(row.fell_back)
        self.assertIsNone(row.seconds)
        self.assertIn("n/a", format_table([row]))

    def test_a_record_can_be_rebuilt_from_the_child_payload_of_a_real_shape(self):
        """The shape the child writes, including the keys the dataclass does not carry."""
        payload = _payload(**{"settle_s": 1.4, "analysis_fps": 14.985, "wall_s": 30.1,
                              "placement_readings": 300, "input_shape": [1, 3, 640, 640]})
        row = TrialResult.from_payload(payload)
        self.assertEqual(row.frames, 300)
        self.assertEqual(row.feed_fps, 14.985)


class RealTrialOnThisMachine(unittest.TestCase):
    """One real CPU trial, in a fresh process, through the child-spawn path.

    The spawn path is where a measurement tool is most likely to be quietly wrong: a
    listener that never fires, a payload the parent cannot read, a placement read before
    the inference. This runs the smallest trial that exercises all of it.

    **It does not assert a rate.** What is asserted is that a frame was analysed, that the
    placement was confirmed from `EXECUTION_DEVICES`, that the latency distribution is
    present and ordered, and that the window really lasted what it says.
    """

    @classmethod
    def setUpClass(cls):
        reason = cls._why_not()
        if reason:
            raise unittest.SkipTest(f"cannot run a CPU trial on this machine: {reason}. "
                                    f"The child-spawn path is therefore UNMEASURED here.")
        # 8 s, not 2 s, on purpose. The measured window starts at frame 0 of a fresh loop, and
        # the first counter-band presence on this clip arrives a few seconds in: 2 s windows
        # legitimately contain no event (measured), while every 20 s window in the task's own
        # campaign contained both. 8 s is long enough for the event to be a guard rather than
        # a coin toss, and short enough that this test is not the campaign.
        spec = TrialSpec(condition=CONDITION_BOOTH, device="CPU", seconds=8.0,
                         warmup_s=0.5, repeat=0, order=1, feed_fps=15.0)
        cls.result = run_trial(spec, timeout=180.0)

    @staticmethod
    def _why_not() -> str | None:
        try:
            import openvino as ov
        except ImportError:
            return "openvino is not importable"
        from app.engine.retail_config import RETAIL_CHECKOUT_CLIP
        from app.inference.pinned import REPO_ROOT, verified_ir_path

        if not (REPO_ROOT / RETAIL_CHECKOUT_CLIP).exists():
            return "the pinned clip is not provisioned"
        try:
            verified_ir_path()
        except Exception as exc:  # noqa: BLE001 - a skip reason, never a pass
            return f"the pinned IR is unusable: {exc}"
        if "CPU" not in tuple(ov.Core().available_devices):
            return f"the CPU is not in available_devices: {list(ov.Core().available_devices)}"
        return None

    def test_the_placement_is_confirmed_from_execution_devices(self):
        self.assertEqual(self.result.confirmed, ("CPU",))
        self.assertIs(self.result.fell_back, False)
        self.assertEqual(self.result.requested, "CPU")
        self.assertIsNone(self.result.unmeasured)
        self.assertEqual(self.result.errors, ())

    def test_the_window_is_sustained_and_carries_a_distribution(self):
        row = self.result
        self.assertGreaterEqual(row.seconds, 7.9)
        self.assertLess(row.seconds, 10.0)
        self.assertGreater(row.frames, 0)
        self.assertGreater(row.throughput_fps, 0.0)
        self.assertGreater(row.latency_p50_ms, 0.0)
        self.assertLessEqual(row.latency_p50_ms, row.latency_p95_ms)
        self.assertLessEqual(row.latency_p95_ms, row.latency_max_ms)
        # Every analysed frame contributes a latency and an age: the distribution covers
        # the window rather than a sample of it.
        self.assertEqual(row.latency_samples, row.frames)
        self.assertEqual(row.frame_age_samples, row.frames)
        # A frame age measured from `captured_at` through decode and transport cannot be
        # zero. A zero here would mean the clock behind it is not the pipeline's.
        self.assertGreater(row.frame_age_p50_ms, 0.0)
        self.assertGreater(row.frame_age_p95_ms, 0.0)
        self.assertIsNotNone(row.drops)
        self.assertGreater(row.rss_start, 0.0)
        self.assertGreater(row.rss_end, 0.0)

    def test_the_stream_still_emits_only_presence_events(self):
        """`event_counts` is the application's own counter, carried and not recomputed.

        A speed-up that altered the detection stream would be a regression, not a win, so
        the trial must still see the counter-band presence event. What is *not* asserted is
        a track id or a frame id: those are assigned by the tracker and differ between a
        15 fps pass and a 29.97 fps pass (measured on this machine), while the count, the
        kind and the zone did not.
        """
        self.assertGreaterEqual(len(self.result.events), 1)
        self.assertEqual(sum(self.result.counts.values()), len(self.result.events))
        for kind in self.result.counts:
            self.assertEqual(kind, "zone_occupancy")
        for key in self.result.events:
            self.assertIn("person:presence:counter_band", key)


class TheToolWritesNoRecordOfItsOwn(unittest.TestCase):
    """The record is written by the entry point, from rows it was handed."""

    def test_the_record_carries_every_row_and_the_interleaving(self):
        rows = [_pipeline_result(), _result()]
        record = st.build_record(rows, seconds=20.0, warmup_s=3.0, repeats=2,
                                 census={"clean": True}, wall_s=99.5)
        self.assertEqual(len(record["results"]), 2)
        self.assertEqual(record["conditions"]["order"],
                         [f"rep{repeat}:{device}"
                          for repeat, device in interleaved_order(st.DEVICE_IDS, 2)])
        self.assertEqual(record["census"], {"clean": True})
        # Every row is JSON round-trippable, and None survives as None.
        json.dumps(record)
        self.assertIsNone(record["results"][1]["frame_age_p50_ms"])

    def test_the_default_record_path_is_a_qualification_artefact(self):
        self.assertEqual(Path(st.OUTPUT_JSON).parent,
                         Path(st.REPO_ROOT) / "docs" / "qualification")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
