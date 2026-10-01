"""PDH reads, and the one behaviour the whole telemetry layer exists to protect.

**A missing instance is `None`, never `0.0`.** A person at the booth who sees "NPU 0%"
concludes the NPU is idle, and - if the truth is that no counter exists - concludes the
NPU is broken. Those are opposite claims about the hardware, and only one of them can be
rendered. So:

- A category or counter that cannot be opened raises `PdhUnavailableError` from
  `__init__`. It is never downgraded to a reading of zero.
- `read` returns `None` for an instance that is not in the enumerated set.
- `close` is idempotent, so a `finally:` block can always run it.

These tests open a **real** `pdh.dll` query for the real `GPU Engine` category, because the
claim being pinned is about this machine's counters and a fake query would prove nothing.
The first test is hermetic: no such counter category exists on any Windows install, so it
must raise everywhere.

Per the plan, if `test_missing_instance_reads_as_none_not_zero` raises
`PdhUnavailableError` on the *valid* `GPU Engine` category, the Windows install is not
English and that is an M0 blocker to record - never to work around by substituting a
number.
"""

import unittest
from app.telemetry.pdh import PdhQuery, PdhUnavailableError, PdhCounterPath


class PdhDegradesHonestly(unittest.TestCase):
    def test_unknown_category_raises_rather_than_returning_zero(self):
        with self.assertRaises(PdhUnavailableError):
            PdhQuery("No Such Counter Category", {"x": "Utilization Percentage"})

    def test_missing_instance_reads_as_none_not_zero(self):
        q = PdhQuery("GPU Engine", {"u": "Utilization Percentage"})
        try:
            v = q.read(PdhCounterPath("GPU Engine", "pid_1_luid_bogus"))
            self.assertIsNone(v, "an unknown instance must read None, never 0")
        finally:
            q.close()
