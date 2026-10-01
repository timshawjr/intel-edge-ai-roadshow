"""PDH reads, and the one behaviour the whole telemetry layer exists to protect.

**A missing instance is `None`, never `0.0`.** A person at the booth who sees "NPU 0%"
concludes the NPU is idle, and - if the truth is that no counter exists - concludes the
NPU is broken. Those are opposite claims about the hardware, and only one of them can be
rendered. So:

- A category or counter that cannot be opened raises `PdhUnavailableError` from
  `__init__`. It is never downgraded to a reading of zero.
- `read` returns `None` for an instance that is not in the enumerated set.
- `close` is idempotent, so a `finally:` block can always run it.

**Two kinds of test, deliberately.** The tests named
`RealPdhOnThisMachine` open a **real** `pdh.dll` query against the real `GPU Engine`
category, because what they pin is a fact about this machine: the counter *path* that adds
(the earlier bug), the five-parameter wildcard expansion, the two-call array pattern, the
NULL-handle guard, and the fact that a missing instance reads `None`. The tests named
`ValueReadDecisionLogic` inject a substitute `PdhApi`, because the real
`PdhGetFormattedCounterValue` terminates the process on this build - measured, with
controls - and a test that can take the test runner down proves nothing. See the
`app.telemetry.pdh` module docstring for the measurements.

Per the plan, if `test_missing_instance_reads_as_none_not_zero` raises
`PdhUnavailableError` on the *valid* `GPU Engine` category, the Windows install is not
English and that is an M0 blocker to record - never to work around by substituting a
number.
"""

import ctypes
import unittest

from app.telemetry import pdh
from app.telemetry.pdh import (
    GPU_ENGINE_COUNTER,
    GPU_ENGINE_COUNTER_PATH,
    GPU_ENGINE_COUNTER_SET,
    MAX_VALUE_READ_REFUSALS,
    PdhApi,
    PdhCounterPath,
    PdhQuery,
    PdhUnavailableError,
)

PERF_DOUBLE = 5
PDH_INVALID_DATA = 0xC0000BC6
PDH_NO_COUNTER = 0xC0000BC0
PDH_MORE_DATA = 0x800007D2


def live_query(**kwargs):
    """A real query over the real category, with priming switched off so tests are fast.

    `prime_seconds=0` is honest here: these tests never read a value, and a prime is two
    `PdhCollectQueryData` calls with a wait between them, which belongs on the setup path
    of the application, not in a unit test.
    """
    kwargs.setdefault("prime_seconds", 0.0)
    return PdhQuery(GPU_ENGINE_COUNTER_SET, {"u": GPU_ENGINE_COUNTER}, **kwargs)


class PdhDegradesHonestly(unittest.TestCase):
    """The plan's Step 8 tests, unchanged in substance."""

    def test_unknown_category_raises_rather_than_returning_zero(self):
        with self.assertRaises(PdhUnavailableError):
            PdhQuery("No Such Counter Category", {"x": GPU_ENGINE_COUNTER},
                     prime_seconds=0.0)

    def test_missing_instance_reads_as_none_not_zero(self):
        q = live_query()
        try:
            v = q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, "pid_1_luid_bogus"))
            self.assertIsNone(v, "an unknown instance must read None, never 0")
        finally:
            q.close()


class RealPdhOnThisMachine(unittest.TestCase):
    """Facts about this machine, proven against the real `pdh.dll`."""

    def test_the_counter_path_constant_is_the_one_that_adds(self):
        """The bug this whole file exists over: the *set* name is not a counter.

        Measured on this machine, same interpreter, same process:

            PdhAddEnglishCounterW(q, "GPU Engine", 0, &c)                        -> NULL handle
            PdhAddEnglishCounterW(q, "\\GPU Engine(*)\\Utilization Percentage",0,&c) -> rc 0

        The first is a counter set, not a counter, and it leaves the handle NULL. Every
        call made after a failed add was writing through that NULL, which is the access
        violation the earlier implementation reported as "PDH does not work here".
        """
        self.assertEqual(GPU_ENGINE_COUNTER_PATH,
                         r"\GPU Engine(*)\Utilization Percentage")
        q = live_query()
        try:
            self.assertEqual(q.category, GPU_ENGINE_COUNTER_SET)
            # The wildcard counter opened with a real, non-zero handle.
            self.assertTrue(all(h for h in q._handles.values()),
                            "the correct counter path must produce a usable handle")
        finally:
            q.close()

    def test_adding_the_counter_set_name_fails_and_leaves_a_null_handle(self):
        """Pinned at the native seam: a wrong name is *loudly* diagnosable.

        If this ever starts returning a handle, the NULL-handle crash is reachable again.
        """
        api = pdh.pdh_api()
        self.assertIsNotNone(api, "pdh.dll must be loadable for this machine's test")
        status, handle = api.open_query()  # type: ignore[union-attr]
        self.assertEqual(status, 0)
        try:
            add_status, add_handle = api.add_counter(status, GPU_ENGINE_COUNTER_SET)  # type: ignore[union-attr]
            self.assertNotEqual(add_status, 0,
                                f"{GPU_ENGINE_COUNTER_SET!r} is a counter SET, not a counter")
            self.assertEqual(add_handle, 0,
                             "a failed add leaves the handle NULL; that NULL is what the "
                             "earlier implementation then wrote through")
            self.assertEqual(add_status, PDH_NO_COUNTER)
        finally:
            api.close_query(status)  # type: ignore[union-attr]

    def test_a_query_whose_counter_cannot_be_added_raises_with_the_real_status(self):
        with self.assertRaises(PdhUnavailableError) as caught:
            PdhQuery(GPU_ENGINE_COUNTER_SET, {"u": "No Such Counter"}, prime_seconds=0.0)
        message = str(caught.exception)
        self.assertIn("PdhAddEnglishCounterW failed", message)
        self.assertIn("0x", message, "the real PDH status must be in the message")

    def test_a_null_counter_handle_is_refused_even_when_the_status_is_success(self):
        """A `(0, 0)` add must raise, not produce a query that writes through NULL."""
        api = _StubApi(add_results=[(0, 0)])
        with self.assertRaises(PdhUnavailableError) as caught:
            PdhQuery(GPU_ENGINE_COUNTER_SET, {"u": GPU_ENGINE_COUNTER},
                     api=api, prime_seconds=0.0)
        self.assertIn("NULL", str(caught.exception))
        self.assertEqual(api.closed, 1, "the half-open query must be closed, not leaked")

    def test_instances_are_the_real_names_from_the_five_parameter_expansion(self):
        """`instances()` used to return `()`: the expansion was bound with four arguments."""
        q = live_query()
        try:
            names = q.instances()
            self.assertTrue(names, f"expansion returned nothing; last_status="
                                    f"0x{q.last_status:08X}")
            for name in names:
                self.assertNotIn("\\", name, "instances are bare names, not full paths")
                self.assertNotIn("(", name)
                self.assertIn("luid_", name, f"not a GPU Engine instance name: {name!r}")
        finally:
            q.close()

    def test_instances_for_luid_selects_the_owned_subset_and_leaves_others_alone(self):
        """The read strategy: only the LUIDs the caller names, and no device judgement.

        `app.telemetry.pdh` filters on a LUID and nothing else. Deciding that a LUID is
        the NPU is `app.telemetry.instance_map`'s job, on facts recorded by the probe - a
        reader that guessed here could put a real number on the wrong card.
        """
        q = live_query()
        try:
            if not q.instances():
                self.skipTest("no live GPU Engine instances to select from")
            third = "0x00011232"  # the third adapter on this machine, left UNATTRIBUTED
            luid = pdh.parse_instance_name(q.instances()[0]).luid
            subset = q.instances_for_luid(luid)
            self.assertTrue(subset)
            for name in subset:
                self.assertEqual(pdh.parse_instance_name(name).luid, luid)
            unattributed = q.instances_for_luid(third)
            for name in unattributed:
                # Returned, not dropped: whether it is any device's is the map's call.
                self.assertEqual(pdh.parse_instance_name(name).luid, third)
            self.assertEqual(q.instances_for_luid(luid, engtype="NoSuchEngine"), ())
        finally:
            q.close()

    def test_the_two_call_array_pattern_uses_a_real_buffer(self):
        """Probe with a NULL buffer, then read with a buffer sized in **bytes**.

        The earlier implementation bound four parameters instead of six and passed a NULL
        `ItemBuffer` on a *sized* call, which is the combination that faulted the process.
        This exercises the whole pattern against the real `pdh.dll`.
        """
        q = live_query()
        try:
            # Measured: unprimed, the probe reports "no buffer filled" with 0 bytes. That is
            # a distinct status from the primed one and must not be read as "no items".
            unprimed_size, unprimed_items, unprimed_text = q.formatted_array_size()
            self.assertEqual(unprimed_size, 0)
            self.assertIn(PDH_MORE_DATA_UNPRIMED_HEX, unprimed_text)

            self.assertTrue(q.prime(), "the array pattern needs a sample on the counter")
            size, expected, status_text = q.formatted_array_size()
            self.assertGreater(size, 0, f"array size probe returned {status_text}")
            self.assertGreater(expected, 0)
            self.assertIn(PDH_MORE_DATA_HEX, status_text)
            status_text, written, expected, items = q.formatted_array_probe(max_items=8)
            self.assertEqual(
                status_text, "0x00000000 (Windows supplies no text for this PDH status)"
            )
            self.assertEqual(written, 8)
            self.assertEqual(len(items), 8)
        finally:
            q.close()

    def test_close_is_idempotent_and_leaves_reads_as_none(self):
        q = live_query()
        q.close()
        q.close()
        self.assertIsNone(q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, "anything")))
        self.assertTrue(any("closed" in r for r in q.value_read_refusals))

    def test_lookup_index_by_name_works_which_proves_ctypes_is_fine(self):
        api = pdh.pdh_api()
        self.assertIsNotNone(api)
        status, index = api.lookup_index(GPU_ENGINE_COUNTER_SET)  # type: ignore[union-attr]
        self.assertEqual(status, 0)
        self.assertGreater(index, 0)


PDH_MORE_DATA_HEX = f"0x{PDH_MORE_DATA:08X}"
PDH_MORE_DATA_UNPRIMED_HEX = f"0x{pdh.PDH_MORE_DATA_UNPRIMED:08X}"


# --------------------------------------------------------------------------------------
# The value read. The real one terminates the process on this machine, so the decision
# logic is pinned against a substitute. What is pinned is the *decision*, not the machine.
# --------------------------------------------------------------------------------------


class _StubApi:
    """A `PdhApi` substitute: records every call, returns scripted results.

    Handles are small integers, so `0` really is a NULL handle and the reader's guard is
    exercised rather than bypassed.
    """

    def __init__(self, *, add_results=None, add_default=(0, 0x9990), value=None,
                 instances=(), collect_status=0):
        self.add_results = list(add_results or [(0, 0x1000)])
        self.add_default = add_default
        self.value = value  # (status, cstatus, ctype, double)
        self.instances = list(instances)
        self.collect_status = collect_status
        self.calls: list[tuple] = []
        self.closed = 0
        self._next_handle = 0x2000

    def _record(self, name, *args):
        self.calls.append((name, *args))

    def open_query(self):
        self._record("open_query")
        return 0, 0x100

    def close_query(self, query):
        self._record("close_query", query)
        self.closed += 1
        return 0

    def add_counter(self, query, path):
        self._record("add_counter", query, path)
        if self.add_results:
            return self.add_results.pop(0)
        if self.add_default[0] == 0:
            self._next_handle += 0x10
            return 0, self._next_handle
        return self.add_default

    def remove_counter(self, counter):
        self._record("remove_counter", counter)
        return 0

    def collect(self, query):
        self._record("collect", query)
        return self.collect_status

    def counter_timebase(self, counter):
        return 0, 10_000_000

    def lookup_index(self, name):
        return 0, 5802

    def expand_wildcard(self, path):
        self._record("expand_wildcard", path)
        return 0, tuple(
            f"\\\\GPU Engine({name})\\Utilization Percentage" for name in self.instances
        )

    def formatted_value(self, counter):
        self._record("formatted_value", counter)
        if self.value is None:
            raise AssertionError(
                "the native value read was reached; on this machine that call ends the "
                "process, so a test must never be allowed to trigger it by accident"
            )
        return self.value

    def formatted_array_size(self, counter, dw_format=0):
        self._record("formatted_array_size", counter, dw_format)
        return PDH_MORE_DATA, 16 * 8

    def formatted_array(self, counter, items, dw_format=0):
        self._record("formatted_array", counter, items, dw_format)
        return 0, items, tuple(0 for _ in range(items))


def stubbed_query(*, value=None, instances=(), enable_value_reads=True, **kwargs):
    api = _StubApi(value=value, instances=instances)
    kwargs.setdefault("prime_seconds", 0.0)
    q = PdhQuery(GPU_ENGINE_COUNTER_SET, {"u": GPU_ENGINE_COUNTER},
                 api=api, enable_value_reads=enable_value_reads, **kwargs)
    return q, api


LIVE = "pid_4242_luid_0x00000000_0x000112A0_phys_0_eng_0_engtype_Neural"


class ValueReadDecisionLogic(unittest.TestCase):
    def test_a_good_read_is_a_float_in_range_and_distinguishable_from_none(self):
        q, api = stubbed_query(value=(0, 0, PERF_DOUBLE, 47.0), instances=[LIVE])
        try:
            q.prime()
            q.watch([LIVE])
            v = q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, LIVE))
            self.assertIsNotNone(v, "a dwCStatus of 0 must produce a number, not None")
            self.assertIs(type(v), float, "read returns a plain float")
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 100.0)
            self.assertEqual(v, 47.0)
            self.assertIsNot(v, None)
        finally:
            q.close()

    def test_a_non_zero_dw_cstatus_is_no_data_not_a_zero(self):
        """`PDH_INVALID_DATA` is what an unprimed rate counter returns. It is not zero."""
        q, api = stubbed_query(
            value=(0, PDH_INVALID_DATA, PERF_DOUBLE, 0.0), instances=[LIVE]
        )
        try:
            q.prime()
            q.watch([LIVE])
            v = q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, LIVE))
            self.assertIsNone(v, "a non-zero dwCStatus must read None, never 0.0")
            self.assertTrue(
                any("dwCStatus" in r and "0xC0000BC6" in r for r in q.value_read_refusals),
                f"the refusal must name the real status: {q.value_read_refusals}",
            )
        finally:
            q.close()

    def test_a_non_zero_call_status_is_no_data_not_a_zero(self):
        q, api = stubbed_query(value=(0xC0000BBD, 0, PERF_DOUBLE, 0.0), instances=[LIVE])
        try:
            q.prime()
            q.watch([LIVE])
            self.assertIsNone(q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, LIVE)))
        finally:
            q.close()

    def test_a_missing_instance_is_none_never_zero(self):
        q, api = stubbed_query(
            value=(0, 0, PERF_DOUBLE, 47.0), instances=["pid_1_luid_0x00000000_0xDEAD"]
        )
        try:
            q.prime()
            q.watch([LIVE])
            v = q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, LIVE))
            self.assertIsNone(v, "an instance that is not there must read None, never 0.0")
            self.assertNotEqual(v, 0.0)
            self.assertTrue(any("not in the enumerated set" in r
                                for r in q.value_read_refusals))
        finally:
            q.close()

    def test_reading_an_unwatched_instance_is_none_and_never_reaches_the_native_call(self):
        q, api = stubbed_query(value=(0, 0, PERF_DOUBLE, 47.0), instances=[LIVE])
        try:
            q.prime()
            self.assertIsNone(q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, LIVE)))
            self.assertFalse(
                [c for c in api.calls if c[0] == "formatted_value"],
                "a wildcard counter cannot be read for one instance; the native call must "
                "not be reached with a handle that is not the instance's",
            )
            self.assertTrue(any("no concrete counter open" in r
                                for r in q.value_read_refusals))
        finally:
            q.close()

    def test_an_unprimed_query_reads_none_not_zero(self):
        q, api = stubbed_query(value=(0, 0, PERF_DOUBLE, 47.0), instances=[LIVE])
        try:
            q.watch([LIVE])
            self.assertFalse(q.primed)
            self.assertIsNone(q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, LIVE)))
            self.assertTrue(any("rate counter" in r for r in q.value_read_refusals))
        finally:
            q.close()

    def test_value_reads_off_by_default_never_reach_the_native_call(self):
        """The measured state of this machine, and the reason the reader cannot crash."""
        q, api = stubbed_query(value=(0, 0, PERF_DOUBLE, 47.0), instances=[LIVE],
                               enable_value_reads=None)
        try:
            self.assertFalse(q.value_reads)
            q.prime()
            q.watch([LIVE])
            self.assertIsNone(q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, LIVE)))
            self.assertFalse([c for c in api.calls if c[0] == "formatted_value"])
            self.assertFalse([c for c in api.calls if c[0] == "expand_wildcard"],
                             "with reads off the 4 Hz path must not touch enumeration either")
            self.assertTrue(any("not usable on this machine" in r
                                for r in q.value_read_refusals))
        finally:
            q.close()

    def test_an_out_of_range_reading_is_refused_not_clamped(self):
        q, api = stubbed_query(value=(0, 0, PERF_DOUBLE, 137.2), instances=[LIVE])
        try:
            q.prime()
            q.watch([LIVE])
            v = q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, LIVE))
            self.assertIsNone(v, "137.2% clamped to 100.0% is a number the counter never gave")
            self.assertTrue(any("137.2" in r for r in q.value_read_refusals))
        finally:
            q.close()

    def test_a_non_double_counter_type_is_refused_rather_than_reinterpreted(self):
        q, api = stubbed_query(value=(0, 0, 0x800, 12.0), instances=[LIVE])
        try:
            q.prime()
            q.watch([LIVE])
            self.assertIsNone(q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, LIVE)))
            self.assertTrue(any("CType" in r for r in q.value_read_refusals))
        finally:
            q.close()

    def test_watch_reports_what_it_opened_and_survives_a_failure(self):
        good = LIVE
        bad = "pid_7_luid_0x00000000_0x000112A0_phys_0_eng_1_engtype_Neural"
        api = _StubApi(add_results=[(0, 0x1000), (0, 0x2000), (PDH_NO_COUNTER, 0)],
                       add_default=(PDH_NO_COUNTER, 0), instances=[good, bad])
        q = PdhQuery(GPU_ENGINE_COUNTER_SET, {"u": GPU_ENGINE_COUNTER},
                     api=api, enable_value_reads=True, prime_seconds=0.0)
        try:
            opened = q.watch([good, bad, "pid_8_missing"])
            self.assertIn(good, opened)
            self.assertNotIn(bad, opened, "a counter that did not open is not watched")
            self.assertEqual(set(q.watched()), {good})
            self.assertTrue(any(bad in r for r in q.value_read_refusals))
        finally:
            q.close()

    def test_read_many_keeps_a_missing_instance_at_none(self):
        present = LIVE
        absent = "pid_9_luid_0x00000000_0x000112A0_phys_0_eng_9_engtype_Neural"
        q, api = stubbed_query(value=(0, 0, PERF_DOUBLE, 12.5), instances=[present])
        try:
            q.prime()
            q.watch([present, absent])
            out = q.read_many([present, absent])
            self.assertEqual(out[present], 12.5)
            self.assertIsNone(out[absent])
            self.assertNotEqual(out[absent], 0.0)
        finally:
            q.close()

    def test_an_empty_query_and_an_empty_category_are_refused(self):
        for kwargs in ({"category": "GPU Engine", "counters": {}},
                       {"category": "  ", "counters": {"u": "x"}}):
            with self.assertRaises(PdhUnavailableError):
                PdhQuery(api=_StubApi(), prime_seconds=0.0, **kwargs)


class RefusalLogIsBounded(unittest.TestCase):
    """The refusal log must not be a process-lifetime leak.

    Measured on this machine: the 4 Hz path (`collect()` + `read_many`) over 40 owned
    instances appends 160 refusals a second, so 60 s produced 9600 entries holding
    8,841,600 characters, and they survived `close()`. M6's gate is a four-hour soak with no
    continuing memory growth, so the log is now bounded to `MAX_VALUE_READ_REFUSALS` and
    keeps the **most recent** refusal, which is the one a diagnostic actually needs.

    Every assertion here goes through the public surface - the module constant,
    `len(query.value_read_refusals)`, indexing it, and `query.refusal_count` - so none of it
    depends on how the log is stored.
    """

    def _hammer(self, query, instance, ticks):
        """Drive `ticks` reads through the public `read` surface, like the 4 Hz path does."""
        for _ in range(ticks):
            query.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, instance))
        return query

    def test_the_bound_is_named_and_positive(self):
        self.assertIsInstance(MAX_VALUE_READ_REFUSALS, int)
        self.assertGreater(MAX_VALUE_READ_REFUSALS, 0)
        # A bound that still holds a useful current symptom, not a token one.
        self.assertGreaterEqual(MAX_VALUE_READ_REFUSALS, 64)
        # Exported, so a caller can size a diagnostic buffer against it without importing
        # the private name off the class.
        self.assertIn("MAX_VALUE_READ_REFUSALS", pdh.__all__)

    def test_many_more_refusals_than_the_bound_still_hold_only_the_bound(self):
        q, api = stubbed_query(enable_value_reads=None, instances=[LIVE])
        try:
            attempts = MAX_VALUE_READ_REFUSALS * 50  # 50x the bound
            self._hammer(q, LIVE, attempts)
            self.assertEqual(q.refusal_count, attempts,
                             "every refusal must be counted, bound or not")
            self.assertEqual(len(q.value_read_refusals), MAX_VALUE_READ_REFUSALS,
                             f"the log must never exceed the bound; "
                             f"{attempts} refusals were appended")
        finally:
            q.close()

    def test_the_most_recent_refusal_survives_and_the_oldest_is_dropped(self):
        """Dropping the *oldest* is the useful direction: a current symptom is the diagnosis.

        Driven entirely through the public `read` surface. The absent-instance refusal
        names the instance, so distinct instances give distinct strings and the log's two
        ends can be told apart by reading them rather than by counting.
        """
        def absent(pid):
            return f"pid_{pid}_luid_0x00000000_0x000112A0_phys_0_eng_0_engtype_Neural"

        # Value reads are ON here so the refusal is the per-instance one, which names the
        # instance. With reads off - this machine's default - every refusal is the same
        # constant string and the log's two ends could not be told apart by content.
        q, api = stubbed_query(enable_value_reads=True, instances=[LIVE])
        try:
            q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, absent(1)))
            first = list(q.value_read_refusals)
            self.assertEqual(len(first), 1)
            self.assertIn("pid_1_", first[0])

            # Far past the bound, all still through the public surface.
            for i in range(MAX_VALUE_READ_REFUSALS * 4):
                q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, absent(i + 1000)))
            final = q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, absent(99999)))
            self.assertIsNone(final)

            self.assertEqual(len(q.value_read_refusals), MAX_VALUE_READ_REFUSALS)
            self.assertNotIn(first[0], list(q.value_read_refusals),
                             "the oldest refusal must be dropped once the bound is hit")
            self.assertIn("pid_99999_", q.value_read_refusals[-1],
                          "the newest refusal must be the one just recorded; got "
                          f"{q.value_read_refusals[-1][:80]!r}")
        finally:
            q.close()

    def test_the_bound_holds_with_the_4hz_read_many_shape(self):
        """The shape that actually leaked: `read_many` over N instances, repeatedly."""
        owned = [f"pid_{100 + i}_luid_0x00000000_0x000112A0_phys_0_eng_0_engtype_Neural"
                 for i in range(40)]
        q, api = stubbed_query(enable_value_reads=None, instances=owned)
        try:
            ticks = MAX_VALUE_READ_REFUSALS // len(owned) + 5
            for _ in range(ticks):
                out = q.read_many(owned)
            self.assertTrue(all(v is None for v in out.values()))
            self.assertEqual(q.refusal_count, ticks * len(owned))
            self.assertEqual(len(q.value_read_refusals), MAX_VALUE_READ_REFUSALS,
                             f"{ticks * len(owned)} refusals from read_many must be bounded")
        finally:
            q.close()

    def test_close_leaves_a_bounded_log_and_keeps_the_post_mortem(self):
        """`close()` must not clear the log: the post-mortem is when it is read.

        The log is already bounded, so keeping it past `close()` costs a few kilobytes and
        is the only way a caller can still find out *why* a reading was refused after the
        query is gone. Pinned here so a future "tidy up on close" does not quietly take it.
        """
        q, api = stubbed_query(enable_value_reads=None, instances=[LIVE])
        self._hammer(q, LIVE, MAX_VALUE_READ_REFUSALS * 3)
        before = len(q.value_read_refusals)
        newest = q.value_read_refusals[-1]
        q.close()
        self.assertEqual(len(q.value_read_refusals), before,
                         "close() must not truncate the log; it is already bounded")
        self.assertLessEqual(len(q.value_read_refusals), MAX_VALUE_READ_REFUSALS)
        self.assertEqual(q.value_read_refusals[-1], newest)
        self.assertEqual(q.refusal_count, MAX_VALUE_READ_REFUSALS * 3,
                         "the count must survive close() too")

    def test_refusals_from_every_path_are_counted_and_bounded_together(self):
        """One log, one bound: a path that bypassed `_refuse` would show up here.

        Each read must take a *different* refusal path, and each must be recorded, so this
        also pins that the paths are distinguishable - four identical reasons would mean a
        caller is being told "unavailable" for four indistinguishable reasons.
        """
        # Enumerated but deliberately not watched, so it reaches the "no concrete counter"
        # refusal rather than the earlier "not in the enumerated set" one.
        unwatched = "pid_9_luid_0x00000000_0x00010E9E_phys_0_eng_5_engtype_Compute"
        q, api = stubbed_query(
            value=(0, PDH_INVALID_DATA, PERF_DOUBLE, 0.0), instances=[LIVE, unwatched]
        )
        try:
            q.prime()
            q.watch([LIVE])
            present = q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, LIVE))       # dwCStatus
            absent = q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, "pid_0_absent"))
            not_watched = q.read(PdhCounterPath(GPU_ENGINE_COUNTER_SET, unwatched))
            other = q.read(PdhCounterPath("Other", LIVE))
            for value, label in ((present, "dwCStatus"), (absent, "absent"),
                                 (not_watched, "unwatched"), (other, "category")):
                self.assertIsNone(value, f"{label} must read None")
            self.assertEqual(q.refusal_count, 4)
            self.assertEqual(len(q.value_read_refusals), 4)
            self.assertIn("dwCStatus", q.value_read_refusals[0])
            self.assertIn("not in the enumerated set", q.value_read_refusals[1])
            self.assertIn("no concrete counter open", q.value_read_refusals[2])
            self.assertIn("is not this query's category", q.value_read_refusals[3])
            self.assertEqual(len(set(q.value_read_refusals)), 4,
                             "four different paths must give four distinguishable reasons")
        finally:
            q.close()

    def test_a_fresh_query_starts_with_no_refusals_and_a_zero_count(self):
        q, api = stubbed_query(enable_value_reads=None, instances=[LIVE])
        try:
            self.assertEqual(len(q.value_read_refusals), 0)
            self.assertEqual(q.refusal_count, 0)
        finally:
            q.close()


class FormatHelpers(unittest.TestCase):
    def test_format_pdh_status_is_numeric_and_says_windows_gave_no_text(self):
        text = pdh.format_pdh_status(0xC0000BC0)
        self.assertIn("0xC0000BC0", text)
        self.assertIn("no text", text)

    def test_the_fmt_countervalue_layout_is_what_pdhtypes_it_as(self):
        self.assertEqual(ctypes.sizeof(pdh._PdhFmtCountervalue), 16)
        self.assertEqual(pdh.FMT_COUNTERVALUE_SIZE, 16)

    def test_wildcard_and_instance_paths_are_built_consistently(self):
        self.assertEqual(pdh.wildcard_path("GPU Engine", "Utilization Percentage"),
                         r"\GPU Engine(*)\Utilization Percentage")
        self.assertEqual(
            pdh.instance_path("GPU Engine", LIVE),
            rf"\GPU Engine({LIVE})\Utilization Percentage",
        )

    def test_a_missing_pdh_dll_degrades_to_an_error_not_a_crash(self):
        saved, saved_err = pdh._API, pdh._API_ERROR
        try:
            pdh._API, pdh._API_ERROR = None, "simulated: pdh.dll is not loadable here"
            self.assertFalse(pdh.pdh_available()[0])
            with self.assertRaises(PdhUnavailableError) as caught:
                PdhQuery(GPU_ENGINE_COUNTER_SET, {"u": GPU_ENGINE_COUNTER}, prime_seconds=0.0)
            self.assertIn("simulated", str(caught.exception))
        finally:
            pdh._API, pdh._API_ERROR = saved, saved_err

    def test_the_api_is_constructible_from_any_object_with_the_pdh_exports(self):
        """`PdhApi` binds by name, so the binding itself is testable without pdh.dll."""
        pdh.PdhApi(_NullLibrary())
        self.assertTrue(True)


class _NullLibrary:
    """Stands in for `pdh.dll`: names `PdhApi` can bind, and nothing else."""

    class _Fn:
        def __init__(self):
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return 0

    def __init__(self):
        for name in (
            "PdhOpenQueryW", "PdhCloseQuery", "PdhAddEnglishCounterW", "PdhRemoveCounter",
            "PdhCollectQueryData", "PdhGetCounterTimeBase", "PdhLookupPerfIndexByNameW",
            "PdhExpandWildCardPathW", "PdhGetFormattedCounterArrayW",
            "PdhGetFormattedCounterValue", "PdhGetRawCounterValue",
        ):
            setattr(self, name, self._Fn())


if __name__ == "__main__":
    unittest.main()
