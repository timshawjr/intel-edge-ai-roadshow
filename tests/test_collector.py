"""The background collector: 4 Hz, cheap, and unable to say "zero" when it means
"nothing".

Three things this module is tested for, all of them consequences of review focus #3:

1. **`psutil`'s first `cpu_percent()` returns `0.0`, and that zero is not a measurement.**
   Measured on this machine: `psutil.cpu_percent(interval=None)` returns `0.0` on the
   first call after import and a real value on the second, because there is no previous
   sample to compute a delta from. A collector that reported that first call would put
   "CPU 0%" on a booth screen for a machine running flat out. So the collector primes at
   construction and reports `UNAVAILABLE` - never `0.0` - for a reading it has not had
   the window for.

2. **A process that died makes its contribution `None`, not zero.** `psutil` raises
   `NoSuchProcess` for a pid that has gone. A dead worker process is exactly the state a
   booth operator needs to see, and rendering its CPU as 0% says "idle" instead of
   "gone".

3. **The collector cannot afford `DeviceRegistry.describe()`.** Measured on this
   machine at **3.3-3.8 s**, for one memoised Windows PnP call. At 4 Hz the collector's
   entire budget is 250 ms per sample. So `describe()` is structurally off the module -
   asserted here by reading the module's own source for the name, and by holding the
   measured per-sample cost far below the 3 s that call would take.

**`rss` is host process memory and is labelled as such.** `device_scope` is `None` on
every memory sample and the unit is `bytes`. No accelerator-memory claim is available
from `psutil`, and none is invented.
"""

import os
import time
import unittest
from unittest import mock

import psutil

from app.contracts.metric_sample import MetricSample, Validity
from app.telemetry.collector import (
    MAX_REASONED_PROCESSES,
    MetricCollector,
    _TrackedProcess,
    cpu_capacity,
)

THIS_PID = os.getpid()


def _by_name(samples):
    return {s.name: s for s in samples}


def _spin(seconds: float = 0.4) -> None:
    """Burn CPU for a while, so a CPU percentage has something to report."""
    end = time.perf_counter() + seconds
    total = 0
    while time.perf_counter() < end:
        total += 1
    return total


class _SteppingClock:
    """A monotonic clock that advances a fixed amount on every read.

    `MetricCollector` reads its clock exactly once per `sample()` (plus once in
    `__init__`), so stepping on read makes the measured window exactly `step` - which is
    what a test needs when it wants to assert something about the *pids* rather than
    about the clock.

    **This exists because the real clock cannot be relied on for that.** Measured on this
    machine: 20000 consecutive `time.monotonic()` calls returned the same value, so its
    resolution is the ~15.6 ms Windows tick, and any two samples closer together than
    one tick measure a window of exactly `0.0`.
    """

    def __init__(self, step: float = 0.25, start: float = 1000.0) -> None:
        self.now = float(start)
        self.step = float(step)

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


class SystemCpu(unittest.TestCase):
    def setUp(self):
        self.collector = MetricCollector([], interval=0.25, background=False)
        self.addCleanup(self.collector.stop)

    def test_the_system_cpu_sample_is_measured_and_in_range(self):
        self.collector.sample()  # establish the window
        _spin(0.25)
        samples = _by_name(self.collector.sample())
        cpu = samples["system_cpu_percent"]
        self.assertEqual(cpu.validity, Validity.VALID)
        self.assertIsNotNone(cpu.value)
        self.assertGreaterEqual(cpu.value, 0.0)
        self.assertLessEqual(cpu.value, 100.0,
                             "system CPU is normalised across all logical cores, so it "
                             "cannot exceed 100; above that it is a per-core sum that "
                             "has been labelled wrongly")
        self.assertEqual(cpu.unit, "percent")
        self.assertEqual(cpu.provider, "psutil")
        self.assertIsNone(cpu.device_scope, "this belongs to the machine, not to a device")

    def test_a_first_call_with_no_window_is_unavailable_not_zero(self):
        """The measured `psutil` trap, at the boundary it happens at."""
        collector = MetricCollector([], interval=0.25, background=False)
        self.addCleanup(collector.stop)
        samples = _by_name(collector.sample())
        cpu = samples["system_cpu_percent"]
        self.assertIsNone(cpu.value)
        self.assertEqual(cpu.validity, Validity.UNAVAILABLE)
        self.assertIn("psutil", cpu.unavailable_reason)
        self.assertNotEqual(cpu.value, 0)

    def test_the_window_is_the_measured_interval_not_a_hardcoded_number(self):
        self.collector.sample()
        time.sleep(0.3)
        cpu = _by_name(self.collector.sample())["system_cpu_percent"]
        self.assertAlmostEqual(cpu.window_seconds, 0.3, delta=0.15,
                               msg="the declared window is not the measured elapsed time")
        self.assertNotEqual(cpu.window_seconds, self.collector.interval,
                            "the window is being reported as the requested interval "
                            "rather than as what elapsed")

    def test_a_busy_machine_reports_a_non_zero_system_cpu(self):
        self.collector.sample()
        _spin(0.5)
        value = self.collector.system_cpu_percent()
        self.assertIsNotNone(value)
        self.assertGreater(value, 0.0,
                           "a core was spinning for half a second and the system "
                           "reported idle; the first-call zero trap is leaking in")


class AppCpu(unittest.TestCase):
    def setUp(self):
        self.collector = MetricCollector([THIS_PID], interval=0.25, background=False,
                                         process_labels=("ui",))
        self.addCleanup(self.collector.stop)

    def test_the_app_cpu_is_a_measured_number_normalised_across_all_cores(self):
        self.collector.sample()
        _spin(0.4)
        samples = _by_name(self.collector.sample())
        cpu = samples["app_cpu_percent"]
        self.assertEqual(cpu.validity, Validity.VALID)
        self.assertIsNotNone(cpu.value)
        self.assertGreater(cpu.value, 0.0,
                           "this process was spinning; a reported idle app CPU is the "
                           "normalisation dividing by the wrong capacity")
        # Normalised against the whole machine, so a busy app can never read above the
        # machine's own 100%, however many processes are summed in.
        self.assertLessEqual(cpu.value, 100.0 * cpu_capacity(),
                             "app CPU is normalised to the whole machine's capacity")
        self.assertEqual(cpu.process_scope, ("ui",))

    def test_app_cpu_percent_is_the_measurement_not_a_cache_read(self):
        """The direct accessor takes its own reading, so it can be called standalone."""
        self.collector.sample()
        _spin(0.4)
        direct = self.collector.app_cpu_percent()
        self.assertIsNotNone(direct)
        self.assertGreater(direct, 0.0)

    def test_app_cpu_percent_is_a_share_of_the_machine_and_not_a_per_core_sum(self):
        """The normalisation is the claim, so it is pinned on both sides.

        `psutil` measures a process against a **single** core, so a process using four of
        this machine's cores reads near 400. `app_cpu_percent` is a share of the whole
        machine, so it is bounded by 100. An implementation that forgot the division would
        pass every "is it positive" test and print a 400% application CPU figure beside a
        system figure that cannot exceed 100 - two different scales in one column.

        The per-core reading is **injected** rather than produced by spinning threads,
        because the GIL means pure-Python threads cannot take this process over 100 on a
        multi-core machine - measured here at 97.7 with three spinners. A test that
        depends on beating the GIL to prove an arithmetic claim is a test that will
        quietly stop proving anything on a faster or busier machine. What is being
        tested is the arithmetic, so the arithmetic gets its input directly, and the real
        end-to-end reading is pinned separately by
        `test_a_busy_machine_reports_a_non_zero_system_cpu`.
        """
        collector = MetricCollector([THIS_PID], interval=0.25, background=False,
                                    process_labels=("ui",))
        self.addCleanup(collector.stop)
        collector.sample()  # first sample is the declared-unavailable one
        with mock.patch.object(_TrackedProcess, "cpu_percent", lambda self: 400.0):
            aggregate = collector.app_cpu_percent()
        self.assertAlmostEqual(aggregate, 400.0 / cpu_capacity(), places=9,
                               msg="app CPU is not the per-core reading divided by the "
                                   "machine's logical capacity")
        self.assertLessEqual(aggregate, 100.0,
                             "app CPU is a share of the whole machine, so it cannot "
                             "exceed 100 whatever the per-core reading said")
        self.assertGreater(400.0 / cpu_capacity(), 1.0)

    def test_an_app_reading_of_every_core_is_one_hundred_percent(self):
        """The end of the scale, so the division's denominator is the machine's."""
        collector = MetricCollector([THIS_PID], interval=0.25, background=False,
                                    process_labels=("ui",))
        self.addCleanup(collector.stop)
        collector.sample()
        with mock.patch.object(_TrackedProcess, "cpu_percent",
                               lambda self: 100.0 * cpu_capacity()):
            self.assertAlmostEqual(collector.app_cpu_percent(), 100.0, places=9)

    def test_one_dead_pid_among_several_still_makes_the_whole_total_unavailable(self):
        """The partial-sum case, with every member's health known."""
        collector = MetricCollector([THIS_PID, THIS_PID], interval=0.25,
                                    background=False,
                                    process_labels=("ui", "worker"))
        self.addCleanup(collector.stop)
        collector.sample()
        readings = {"ui": 400.0, "worker": None}

        def read(self):
            return readings[self.label]

        with mock.patch.object(_TrackedProcess, "cpu_percent", read):
            self.assertIsNone(collector.app_cpu_percent(),
                              "the aggregate reported the sum of the pids that "
                              "answered, which understates by exactly the one that "
                              "could not be measured")

    def test_a_dead_pid_makes_the_app_cpu_unavailable_not_zero(self):
        # A pid that cannot exist. psutil raises NoSuchProcess for it, which is a
        # statement about the process, not a CPU percentage.
        collector = MetricCollector([THIS_PID, 0x7FFFFFFF], interval=0.25,
                                    background=False,
                                    process_labels=("ui", "worker"))
        self.addCleanup(collector.stop)
        collector.sample()
        _spin(0.3)
        samples = _by_name(collector.sample())
        cpu = samples["app_cpu_percent"]
        self.assertIsNone(cpu.value, "a dead process was summed as zero")
        self.assertEqual(cpu.validity, Validity.UNAVAILABLE)
        self.assertIn("worker", cpu.unavailable_reason)

    def test_a_dead_pid_makes_the_app_rss_unavailable_too(self):
        """Not just the CPU: a total that quietly skips a dead member is also a lie."""
        collector = MetricCollector([THIS_PID, 0x7FFFFFFF], interval=0.25,
                                    background=False,
                                    process_labels=("ui", "worker"))
        self.addCleanup(collector.stop)
        collector.sample()
        _spin(0.2)
        rss = _by_name(collector.sample())["app_rss_bytes"]
        self.assertIsNone(rss.value)
        self.assertEqual(rss.validity, Validity.UNAVAILABLE)
        self.assertIn("worker", rss.unavailable_reason)

    def test_each_process_gets_its_own_sample_and_a_dead_one_is_its_own_unavailable(self):
        collector = MetricCollector([THIS_PID, 0x7FFFFFFF], interval=0.25,
                                    background=False,
                                    process_labels=("ui", "worker"))
        self.addCleanup(collector.stop)
        collector.sample()
        _spin(0.3)
        samples = _by_name(collector.sample())
        self.assertEqual(samples["rss:ui"].validity, Validity.VALID)
        self.assertIsNotNone(samples["rss:ui"].value)
        self.assertEqual(samples["rss:worker"].validity, Validity.UNAVAILABLE)
        self.assertIsNone(samples["rss:worker"].value)
        self.assertNotEqual(samples["rss:worker"].value, 0)

    def test_an_empty_pid_list_leaves_the_app_cpu_unavailable_rather_than_zero(self):
        collector = MetricCollector([], interval=0.25, background=False)
        self.addCleanup(collector.stop)
        samples = _by_name(collector.sample())
        cpu = samples["app_cpu_percent"]
        self.assertIsNone(cpu.value)
        self.assertEqual(cpu.validity, Validity.UNAVAILABLE)
        self.assertIsNone(samples["app_rss_bytes"].value)


class ExplainingAnErrorCannotTakeTheSamplerDown(unittest.TestCase):
    """The fault is **constructed**, not observed. See the note on the class above it.

    An earlier version of these tests claimed the exploding-`__str__` `psutil.Error` had
    been measured over 199 pids on this machine. It had not: measured, 187 of pids
    1..199 failed to resolve, 12 resolved, and all 12 raised cleanly-rendering errors.
    Nothing raised a `MemoryError` from `__str__`. The hazard is still worth a guard, so
    the tests below **inject** the fault and prove the guard is load-bearing.
    """

    def test_an_error_whose_own_str_raises_is_still_reported_as_a_reason(self):
        class Exploding(psutil.Error):
            def __str__(self):
                raise MemoryError("building this message is not survivable")

        def explode(self, interval=None):
            # Raised by a call, not handed to the mock as an instance: `side_effect`
            # holding the exception makes unittest's own reporting call its broken
            # `__str__`, which is the very thing under test here.
            raise Exploding("pid on fire")

        # Patched on `psutil.Process`, not on `_TrackedProcess.cpu_percent`: replacing the
        # tracked method would remove the `except psutil.Error` that this test exists to
        # exercise. The failure has to come from *inside* the real method.
        with mock.patch.object(psutil.Process, "cpu_percent", explode):
            collector = MetricCollector([THIS_PID], interval=0.25, background=False,
                                        process_labels=("ui",))
            self.addCleanup(collector.stop)
            collector.sample()
            samples = _by_name(collector.sample())
        cpu = samples["cpu:ui"]
        self.assertIsNone(cpu.value)
        self.assertEqual(cpu.validity, Validity.UNAVAILABLE)
        self.assertIn("Exploding", cpu.unavailable_reason,
                      "the exception type is what a reader needs; the message is a "
                      "nicety that may be unavailable")

    def test_a_read_failure_is_recorded_even_though_the_handle_resolved(self):
        """A process whose handle opens but whose counters do not read.

        This is not hypothetical. Measured over pids 1..199 on this machine: 187 failed
        to resolve, **12 resolved**, and of those 12, **9 then raised** `NoSuchProcess`
        from `cpu_percent()`. So the handle existing and the handle being readable are
        two different facts here. A version that only recorded a failure from resolving
        the handle produced samples with `value=None` and no reason, and `MetricSample`
        correctly refused to build them - an absent reading with no stated reason, which
        is the state this layer exists to prevent.
        """
        collector = MetricCollector([THIS_PID], interval=0.25, background=False,
                                    process_labels=("ui",))
        self.addCleanup(collector.stop)
        collector.sample()
        with mock.patch.object(psutil.Process, "cpu_percent",
                               side_effect=psutil.NoSuchProcess(THIS_PID)):
            samples = _by_name(collector.sample())
        cpu = samples["cpu:ui"]
        self.assertIsNone(cpu.value)
        self.assertEqual(cpu.validity, Validity.UNAVAILABLE)
        self.assertIn("NoSuchProcess", cpu.unavailable_reason)

    def test_many_tracked_pids_still_produce_a_complete_and_honest_sample_set(self):
        """199 pids, every counter unreadable, and the sample set is still whole.

        **The fault is injected, and the window is stepped, because the test was
        vacuous without both.** Two separate reasons, both verified by mutation:

        - On the real `time.monotonic()` - measured coarse on this machine, 20000
          consecutive calls returning the same value - two samples microseconds apart
          measure a window of exactly `0.0`, which makes the *whole* set unavailable with
          the window reason. Every sample was then unavailable because of the clock, so
          the test passed whether or not a dead pid's reason was recorded.
        - Even with a real window, the pids that *resolve and then fail to read* are a
          small and transient population: measured over pids 1..199, only 12 resolve at
          all. Every other pid fails in `handle()`, which records its reason on the
          unmutated path, so a test leaning on the real process table stays green with
          the read-failure recording deleted.

        So `THIS_PID` is tracked to guarantee at least one handle has resolved and cached,
        and the read failure is injected on the real `psutil.Process.cpu_percent`.

        **Only `cpu_percent` is made to fail; `memory_info` is left working.** That is
        deliberate and it is what makes this test discriminating. `sample()` builds its
        per-process reasons from `tracked.failure` *after* every reading has been taken,
        so if both counters failed, the surviving `rss_bytes` failure would set the
        reason and mask a deleted recording in `cpu_percent` - which is exactly how the
        first attempt at this test passed with the recording removed. Failing one counter
        and not the other produces the shape that matters anyway: **a partial reading**,
        where the CPU is unavailable and the memory beside it is a real number.
        """
        clock = _SteppingClock(step=0.25)
        pids = [THIS_PID] + [pid for pid in range(1, 200) if pid != THIS_PID]
        collector = MetricCollector(pids, interval=0.25, background=False, clock=clock)
        self.addCleanup(collector.stop)
        collector.sample()  # the declared-unavailable first sample; resolves handles

        def dead_cpu(self, interval=None):
            raise psutil.NoSuchProcess(self.pid)

        with mock.patch.object(psutil.Process, "cpu_percent", dead_cpu):
            samples = collector.sample()

        by_name = _by_name(samples)
        self.assertEqual(len(samples), 3 + 2 * len(pids))
        self.assertAlmostEqual(by_name["system_cpu_percent"].window_seconds, 0.25,
                               places=9)

        cpu_samples = [s for s in samples if s.name.startswith("cpu:")]
        rss_samples = [s for s in samples if s.name.startswith("rss:")]
        self.assertEqual(len(cpu_samples), len(pids))
        self.assertEqual(len(rss_samples), len(pids))
        for sample in cpu_samples:
            self.assertEqual(sample.validity, Validity.UNAVAILABLE, sample.name)
            self.assertIsNone(sample.value,
                              f"{sample.name} carries a number for a process whose CPU "
                              f"could not be read")
            reason = sample.unavailable_reason or ""
            self.assertIn("NoSuchProcess", reason,
                          f"{sample.name} is unavailable for a reason that does not name "
                          f"the failure")
            self.assertNotIn("no measurable window elapsed", reason,
                             "this sample was unavailable because of the clock, not "
                             "because of the pid, so it pins nothing about dead pids")

        # The partial reading, asserted rather than assumed: beside a CPU that could not
        # be read, the memory of a pid that *can* be read is a genuine number. If this
        # were empty the whole set would be unavailable and the test would prove nothing
        # about the shape it claims to be checking.
        self.assertTrue([s for s in rss_samples if s.validity is Validity.VALID],
                        "no readable pid produced a real memory reading, so this set "
                        "cannot distinguish a partial reading from a uniform absence")
        self.assertIsNone(by_name["app_cpu_percent"].value)
        self.assertEqual(by_name["app_cpu_percent"].validity, Validity.UNAVAILABLE)


class ARecordedReasonDoesNotDescribeItselfRecursively(unittest.TestCase):
    """The reason must be the same after the tenth read as after the first.

    `handle()` re-raises the recorded reason as the `msg` of a fresh `NoSuchProcess`, and
    `cpu_percent`/`rss_bytes` record every failure. A version that re-recorded
    unconditionally therefore built its reason out of itself, one layer per reading -
    measured here as

        NoSuchProcess: NoSuchProcess: NoSuchProcess: NoSuchProcess: process PID not
        found (pid=1) (pid=1) (pid=1) (pid=1)

    four times a second, per unreadable process. Two readings of the same failure are
    not two failures, and a reason that grows without bound on the 4 Hz path is a way to
    lose the process while trying to explain it.
    """

    DEAD_PID = 0x7FFFFFFF

    def test_the_reason_is_identical_after_many_reads(self):
        tracked = _TrackedProcess(self.DEAD_PID, "worker")
        self.assertIsNone(tracked.cpu_percent())
        first = tracked.failure
        self.assertIsNotNone(first)
        for _ in range(50):
            self.assertIsNone(tracked.cpu_percent())
            self.assertIsNone(tracked.rss_bytes())
        self.assertEqual(tracked.failure, first,
                         "the recorded reason changed without the failure changing")
        self.assertEqual(first.count("NoSuchProcess"), 1,
                         f"the reason has described itself recursively: {first!r}")

    def test_the_reason_stays_short_over_many_samples_of_a_whole_collector(self):
        """The end-to-end shape, and the one that actually cost this process memory."""
        clock = _SteppingClock(step=0.25)
        collector = MetricCollector(list(range(1, 200)), interval=0.25,
                                    background=False, clock=clock)
        self.addCleanup(collector.stop)
        collector.sample()
        for _ in range(6):
            samples = collector.sample()
        for sample in samples:
            if sample.unavailable_reason:
                self.assertLess(len(sample.unavailable_reason), 4_000,
                                f"{sample.name} carries a {len(sample.unavailable_reason):,}"
                                f"-character reason")
                self.assertLessEqual(sample.unavailable_reason.count("NoSuchProcess"),
                                     2 * MAX_REASONED_PROCESSES + 2,
                                     f"{sample.name} has compounded its own reason")


class TheReasonIsBounded(unittest.TestCase):
    """A reason that grows with the failure can lose the process it is explaining.

    Measured on this machine, before the bound existed: a collector tracking 199 pids
    grew this process's RSS 56 MB -> 95 MB -> 206 MB -> 647 MB -> 2.4 GB across ten
    samples, and the eleventh tick raised `MemoryError`. Handles, thread count,
    `psutil._pmap` and the tracked Python object count were all flat throughout, and a
    control loop doing the same `psutil` reads without the collector did not move RSS at
    all - so the growth was the collector's own unbounded reason strings, not the OS.
    After the bound: RSS flat at 44.1 MB across the same ten samples.
    """

    def setUp(self):
        self.clock = _SteppingClock(step=0.25)
        self.pids = [THIS_PID] + [p for p in range(1, 200) if p != THIS_PID]
        self.collector = MetricCollector(self.pids, interval=0.25, background=False,
                                         clock=self.clock)
        self.addCleanup(self.collector.stop)
        self.collector.sample()

    def test_a_long_list_of_dead_processes_is_summarised_with_a_count_not_named_in_full(self):
        samples = self.collector.sample()
        app_cpu = _by_name(samples)["app_cpu_percent"]
        self.assertIsNone(app_cpu.value)
        reason = app_cpu.unavailable_reason or ""
        named = sorted(t.label for t in self.collector._tracked if t.failure)
        self.assertGreater(len(named), MAX_REASONED_PROCESSES,
                           "this test needs more failures than the bound allows")
        # The first clause of the reason names at most the bound, and says how many it
        # left out...
        opening = reason.split(" could not be read")[0]
        self.assertLessEqual(len(opening.split(",")), MAX_REASONED_PROCESSES)
        self.assertIn(f"and {len(named) - MAX_REASONED_PROCESSES} more", opening)
        for label in named[:MAX_REASONED_PROCESSES]:
            self.assertIn(label, reason)
        self.assertNotIn(named[MAX_REASONED_PROCESSES], reason,
                         "the reason names every process; it is unbounded")
        self.assertLess(len(reason), 4_000,
                        f"the aggregate reason is {len(reason):,} characters long")

    def test_the_per_process_samples_still_carry_every_reason(self):
        """Bounding the summary must not lose the evidence it summarises."""
        samples = self.collector.sample()
        by_name = _by_name(samples)
        self.assertEqual(len([s for s in samples if s.name.startswith("cpu:")]),
                         len(self.pids))
        for tracked in self.collector._tracked:
            sample = by_name[f"cpu:{tracked.label}"]
            if tracked.failure:
                self.assertIn(tracked.failure.split(":")[0], sample.unavailable_reason)

    def test_a_short_list_of_dead_processes_is_named_in_full_with_no_count(self):
        """Two dead processes is not a long list, and 'and 0 more' is noise."""
        collector = MetricCollector([THIS_PID, THIS_PID, 0x7FFFFFFF], interval=0.25,
                                    background=False, clock=_SteppingClock(step=0.25),
                                    process_labels=("ui", "supervisor", "worker"))
        self.addCleanup(collector.stop)
        collector.sample()
        reason = _by_name(collector.sample())["app_cpu_percent"].unavailable_reason or ""
        self.assertIn("worker", reason)
        self.assertNotIn("more", reason)


class TheRecordedReasonIsRereadCleanly(unittest.TestCase):
    """The cached reason is re-raised through `handle()`, so its rendering matters.

    `psutil.NoSuchProcess`'s signature is `(pid, name=None, msg=None)`. Passing the
    recorded reason as the **second positional** argument therefore sets `name`, and the
    exception renders as
    `"process no longer exists (pid=5, name='NoSuchProcess: process no longer exists
    (pid=5)')"` - the reason quoted inside a reason, which reads as a second, different
    failure rather than as the one that was recorded.
    """

    def test_the_reread_reason_does_not_quote_itself_as_a_process_name(self):
        tracked = _TrackedProcess(0x7FFFFFFF, "worker")
        self.assertIsNone(tracked.cpu_percent(), "an unreadable pid did not report None")
        self.assertIsNotNone(tracked.failure)
        with self.assertRaises(psutil.NoSuchProcess) as caught:
            tracked.handle()
        rendered = str(caught.exception)
        self.assertNotIn("name=", rendered,
                         f"the recorded reason was passed as the process name: {rendered}")
        self.assertIn("NoSuchProcess", rendered)
        self.assertIn(str(0x7FFFFFFF), rendered,
                      "the pid is missing from the re-raised reason")


class TheCoarseClockIsNotAPermittedZero(unittest.TestCase):
    """`time.monotonic()` on Windows has tick resolution, and a zero window is real.

    Measured on this machine: 20000 consecutive `time.monotonic()` calls returned the
    same value, so the clock advances in ~15.6 ms steps. Two samples closer together than
    one tick therefore measure a window of exactly `0.0`. The collector must report that
    as *unavailable with a stated reason* - never as `interval` substituted for a window
    that did not elapse, and never as a number.
    """

    def test_a_window_of_exactly_zero_is_unavailable_and_says_so(self):
        frozen = _SteppingClock(step=0.0)
        collector = MetricCollector([THIS_PID], interval=0.25, background=False,
                                    clock=frozen)
        self.addCleanup(collector.stop)
        collector.sample()  # first sample: unavailable because it is the first
        samples = _by_name(collector.sample())
        cpu = samples["system_cpu_percent"]
        self.assertIsNone(cpu.value, "a window of zero produced a percentage anyway")
        self.assertEqual(cpu.validity, Validity.UNAVAILABLE)
        self.assertIn("no measurable window elapsed", cpu.unavailable_reason)
        self.assertNotEqual(cpu.window_seconds, collector.interval,
                            "the requested interval was substituted for a window that "
                            "did not elapse")

    def test_a_real_window_is_reported_as_elapsed_rather_than_as_the_request(self):
        clock = _SteppingClock(step=0.37)
        collector = MetricCollector([THIS_PID], interval=0.25, background=False,
                                    clock=clock)
        self.addCleanup(collector.stop)
        collector.sample()
        cpu = _by_name(collector.sample())["system_cpu_percent"]
        self.assertAlmostEqual(cpu.window_seconds, 0.37, places=9)
        self.assertEqual(cpu.validity, Validity.VALID)
        self.assertIsNotNone(cpu.value)


class RssIsHostMemory(unittest.TestCase):
    """`rss` is process memory. It is never accelerator memory."""

    def setUp(self):
        self.collector = MetricCollector([THIS_PID], interval=0.25, background=False,
                                         process_labels=("ui",))
        self.addCleanup(self.collector.stop)

    def test_rss_is_in_bytes_scoped_to_the_process_and_to_no_device(self):
        self.collector.sample()
        _spin(0.1)
        samples = _by_name(self.collector.sample())
        rss = samples["rss:ui"]
        self.assertEqual(rss.unit, "bytes")
        self.assertIsNone(rss.device_scope)
        self.assertEqual(rss.process_scope, ("ui",))
        self.assertEqual(rss.provider, "psutil")
        self.assertGreater(rss.value, 0)

    def test_no_sample_ever_claims_a_device_scope(self):
        self.collector.sample()
        for sample in self.collector.sample():
            self.assertIsNone(sample.device_scope,
                             f"{sample.name} claims device_scope="
                             f"{sample.device_scope!r}; psutil measures host process "
                             f"state and knows nothing about an accelerator")

    def test_the_memory_names_say_host_and_not_gpu(self):
        self.collector.sample()
        for sample in self.collector.sample():
            lowered = sample.name.lower()
            self.assertNotIn("gpu", lowered)
            self.assertNotIn("npu", lowered)
            self.assertNotIn("vram", lowered)


class NothingUnmeasurableBecomesZero(unittest.TestCase):
    def test_no_unavailable_sample_in_a_real_run_carries_a_number(self):
        collector = MetricCollector([THIS_PID, 0x7FFFFFFF], interval=0.25,
                                    background=False,
                                    process_labels=("ui", "worker"))
        self.addCleanup(collector.stop)
        for _ in range(3):
            for sample in collector.sample():
                if sample.validity is Validity.UNAVAILABLE:
                    self.assertIsNone(sample.value)
                    self.assertTrue(sample.unavailable_reason)
                else:
                    self.assertIsNotNone(sample.value)
                self.assertIsInstance(sample, MetricSample)
                sample.to_json()  # must survive serialisation as-is


class TheCollectorIsCheap(unittest.TestCase):
    """3.3-3.8 s for one `DeviceRegistry.describe()`. The budget is 250 ms."""

    def test_a_sample_costs_far_less_than_the_device_registry_description(self):
        collector = MetricCollector([THIS_PID], interval=0.25, background=False,
                                    process_labels=("ui",))
        self.addCleanup(collector.stop)
        collector.sample()  # warm the Process handles
        timings = []
        for _ in range(30):
            started = time.perf_counter()
            collector.sample()
            timings.append((time.perf_counter() - started) * 1000.0)
        timings.sort()
        median = timings[len(timings) // 2]
        self.assertLess(median, 50.0,
                        f"median sample cost {median:.3f} ms; DeviceRegistry.describe() "
                        f"costs 3.3-3.8 s on this machine and cannot be on this path")
        print(f"\n[collector sample cost, n=30] p50={median:.3f} ms "
              f"min={timings[0]:.3f} max={timings[-1]:.3f}")

    def test_the_module_imports_nothing_that_could_take_seconds(self):
        """The bound is on what the module *imports*, read from the live module.

        Not a source-text search: the module's own docstring names
        `DeviceRegistry.describe()` to record why it is absent, so grepping the source
        would fail on the documentation of the constraint it is documenting. The
        constraint that matters is structural - which modules are actually in
        `sys.modules` with `app.telemetry.collector` in the frame.
        """
        import ast
        import pathlib

        import app.telemetry.collector as collector

        source = pathlib.Path(collector.__file__).read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for module in sorted(imported):
            root = module.split(".")[0]
            self.assertNotIn(
                root, ("openvino", "cv2", "PySide6"),
                f"app/telemetry/collector.py imports {module!r}; the 4 Hz path has no "
                f"business loading a runtime, a video library or a GUI toolkit")
        for module in sorted(imported):
            self.assertFalse(
                module.startswith("app.inference"),
                f"app/telemetry/collector.py imports {module!r}; device compilation "
                f"belongs on the startup path, and EXECUTION_DEVICES is read by the "
                f"worker, not by the sampler")
            self.assertFalse(
                module.startswith("app.telemetry.pdh"),
                f"app/telemetry/collector.py imports {module!r}; the PDH reader opens "
                f"Windows counters and is owned by the device cards, not by this sampler")

    def test_the_collector_module_does_not_hold_a_device_registry(self):
        import app.telemetry.collector as collector

        for name in dir(collector):
            self.assertNotIn("device_registry", name.lower())
            self.assertNotIn("DeviceRegistry", name)

    def test_the_readings_come_from_psutil_and_are_not_synthesised(self):
        """A spy on the primitive the collector actually calls."""
        collector = MetricCollector([THIS_PID], interval=0.25, background=False,
                                    process_labels=("ui",))
        self.addCleanup(collector.stop)
        real = psutil.cpu_percent
        calls = []

        def spy(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        collector.sample()
        with mock.patch.object(psutil, "cpu_percent", spy):
            collector.sample()
        self.assertTrue(calls, "the system CPU reading did not go through psutil")


class TheBackgroundThread(unittest.TestCase):
    def test_the_thread_samples_repeatedly_and_stop_ends_it(self):
        collector = MetricCollector([THIS_PID], interval=0.02,
                                     process_labels=("ui",))
        self.addCleanup(collector.stop)
        self.assertTrue(collector.running)
        deadline = time.monotonic() + 5.0
        while collector.samples_taken < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(collector.samples_taken, 3)
        self.assertIsNotNone(collector.latest())
        collector.stop()
        self.assertFalse(collector.running)
        taken = collector.samples_taken
        time.sleep(0.2)
        self.assertEqual(collector.samples_taken, taken, "the thread kept sampling "
                                                         "after stop()")

    def test_stop_is_idempotent(self):
        collector = MetricCollector([THIS_PID], interval=0.02)
        collector.stop()
        collector.stop()
        self.assertFalse(collector.running)

    def test_a_collector_with_no_background_thread_says_so(self):
        collector = MetricCollector([], background=False)
        self.addCleanup(collector.stop)
        self.assertFalse(collector.running)
        collector.sample()
        self.assertIsNotNone(collector.latest())

    def test_a_background_collector_is_off_by_default_for_the_reading_path(self):
        # The default is a running thread, which is what a booth wants; the reading path
        # is `sample()`, and it must work whether or not a thread is running.
        collector = MetricCollector([], interval=0.05)
        self.assertTrue(collector.running)
        self.addCleanup(collector.stop)
        self.assertEqual(len(collector.sample()), len(collector.latest()))


class Refusals(unittest.TestCase):
    def test_a_non_positive_interval_is_refused(self):
        for bad in (0.0, -0.25):
            with self.assertRaises(ValueError):
                MetricCollector([], interval=bad)

    def test_a_non_integer_or_non_positive_pid_is_refused(self):
        for bad in ("1234", 1.5, -1, True):
            with self.assertRaises(ValueError):
                MetricCollector([bad])

    def test_a_process_label_is_required_when_labels_are_given_for_some_pids(self):
        with self.assertRaises(ValueError):
            MetricCollector([THIS_PID, THIS_PID], process_labels=("ui",))

    def test_a_blank_process_label_is_refused(self):
        with self.assertRaises(ValueError):
            MetricCollector([THIS_PID], process_labels=("  ",))

    def test_cpu_capacity_is_a_positive_int(self):
        self.assertIsInstance(cpu_capacity(), int)
        self.assertGreater(cpu_capacity(), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
