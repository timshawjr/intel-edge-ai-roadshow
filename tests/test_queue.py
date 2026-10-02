"""The bounded frame queue, and the two things it must never do silently.

Review Focus #2 is the reason this module exists: **a full queue that silently
discards frames.** The analysis is then quietly behind the video, the overlay
smoothly animates against stale boxes, and nothing anywhere says so. Spec section 6
requires explicit drop counters, so:

- `put` **refuses** a frame when the queue is full, returns `False`, and increments
  `drops`. It never blocks and it never silently overwrites: a newest-frame queue
  would be a different class of loss - it would make the latency number meaningless,
  because the depth would stop being a depth.
- `drops` is monotonic and survives `get`. A `get` is not a repair of the record;
  the frame that was refused is still a frame the booth never analysed.

Review Focus #1 is why `put` type-checks its argument. **The queue carries
`FramePacket`s, not detection lists.** A bare list of boxes has no identity, so a
result could not be bound to the frame it came from, which is exactly the "correct
box, wrong picture" failure `FramePacket` was written to make unrepresentable.
"""

import threading
import time
import unittest

import numpy as np

from app.contracts.frame import FramePacket
from app.engine.queue import BoundedFrameQueue


def _f(fid, t=0.0, loop=0):
    return FramePacket(session_id="s", scenario_id="retail", stream_id="A", frame_id=fid,
                       source_pts=t, source_loop_id=loop, captured_at=t, decoded_at=t,
                       width=8, height=8, image=np.zeros((8, 8, 3), np.uint8))


class QueueBounds(unittest.TestCase):
    """The brief's five tests, plus the bound under real contention."""

    def test_a_full_queue_drops_the_newest_and_counts_it(self):
        q = BoundedFrameQueue(maxsize=2)
        self.assertTrue(q.put(_f(1, 0.0)))
        self.assertTrue(q.put(_f(2, 0.1)))
        self.assertFalse(q.put(_f(3, 0.2)))
        self.assertEqual(q.drops, 1)
        self.assertEqual(q.depth, 2)

    def test_the_queue_never_exceeds_its_bound(self):
        q = BoundedFrameQueue(maxsize=2)
        for i in range(50):
            q.put(_f(i, i / 30))
        self.assertLessEqual(q.depth, 2)

    def test_get_returns_frames_in_order(self):
        q = BoundedFrameQueue(maxsize=2)
        q.put(_f(1, 0.0)); q.put(_f(2, 0.1))
        self.assertEqual(q.get().frame_id, 1)
        self.assertEqual(q.get().frame_id, 2)

    def test_get_on_an_empty_queue_returns_none_rather_than_blocking_forever(self):
        self.assertIsNone(BoundedFrameQueue().get(timeout=0.01))

    def test_drops_survive_a_get(self):
        q = BoundedFrameQueue(maxsize=1)
        q.put(_f(1, 0.0)); q.put(_f(2, 0.1))
        q.get()
        self.assertEqual(q.drops, 1)


class TheBoundIsStructural(unittest.TestCase):
    """These fail if the bound is advisory rather than enforced."""

    def test_accepted_plus_dropped_accounts_for_every_put(self):
        q = BoundedFrameQueue(maxsize=3)
        accepted = 0
        for i in range(200):
            accepted += 1 if q.put(_f(i, i / 30)) else 0
            # Drain every other frame so the queue does not simply saturate once.
            if i % 2:
                self.assertIsNotNone(q.get())
        self.assertEqual(accepted + q.drops, 200)
        self.assertLessEqual(q.depth, 3)

    def test_a_refused_frame_leaves_the_queue_untouched(self):
        q = BoundedFrameQueue(maxsize=1)
        q.put(_f(7, 0.7))
        self.assertFalse(q.put(_f(8, 0.8)))
        # The survivor is the frame that was accepted, not the newest one. A queue that
        # evicted the oldest would hand the worker a different picture than it counted.
        self.assertEqual(q.get().frame_id, 7)
        self.assertEqual(q.drops, 1)

    def test_a_zero_or_negative_maxsize_is_refused_rather_than_clamped(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                BoundedFrameQueue(maxsize=bad)

    def test_a_bool_maxsize_is_refused_because_a_bool_is_an_int_in_python(self):
        with self.assertRaises(TypeError):
            BoundedFrameQueue(maxsize=True)


class FrameIdentityIsNotOptional(unittest.TestCase):
    """Review Focus #1, enforced at the queue's door."""

    def test_put_refuses_anything_that_is_not_a_frame_packet(self):
        q = BoundedFrameQueue(maxsize=2)
        for impostor in ([(0, 0.9, 10, 20, 20, 20)], None, "frame-1", 7,
                         (np.zeros((8, 8, 3), np.uint8),)):
            with self.assertRaises(TypeError):
                q.put(impostor)
        self.assertEqual(q.depth, 0)
        self.assertEqual(q.drops, 0, "a malformed put is a bug, not a dropped frame")

    def test_a_get_carries_the_whole_packet_not_its_pixels(self):
        q = BoundedFrameQueue(maxsize=1)
        packet = _f(11, 0.5, loop=2)
        q.put(packet)
        got = q.get()
        self.assertIs(got, packet)
        self.assertEqual(got.identity, ("s", "A", 2, 11))


class DropCountingUnderContention(unittest.TestCase):
    """The counter has to survive being hammered, or it is decoration."""

    def test_concurrent_producers_never_break_the_bound_or_the_accounting(self):
        q = BoundedFrameQueue(maxsize=4)
        threads_count, per_thread = 4, 250
        accepted = []
        lock = threading.Lock()
        barrier = threading.Barrier(threads_count)

        def producer(base):
            local = 0
            barrier.wait()
            for i in range(per_thread):
                if q.put(_f(base * per_thread + i, 0.0)):
                    local += 1
            with lock:
                accepted.append(local)

        producers = [threading.Thread(target=producer, args=(n,)) for n in range(threads_count)]
        for t in producers:
            t.start()
        for t in producers:
            t.join(60.0)

        total = threads_count * per_thread
        self.assertEqual(sum(accepted) + q.drops, total,
                         "every put is either accepted-and-counted or dropped-and-counted")
        self.assertLessEqual(q.depth, 4)
        self.assertEqual(sum(accepted), threads_count * per_thread - q.drops)

    def test_a_blocked_getter_is_released_by_close_rather_than_waiting_out_its_timeout(self):
        q = BoundedFrameQueue(maxsize=1)
        seen: list[object] = []

        def waiter():
            seen.append(q.get(timeout=30.0))

        t = threading.Thread(target=waiter)
        t.start()
        # Let the getter actually reach the wait before closing underneath it.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and q.waiting == 0:
            time.sleep(0.005)
        self.assertGreater(q.waiting, 0,
                           "the getter never blocked; this test proves nothing")
        q.close()
        t.join(10.0)
        self.assertFalse(t.is_alive(), "close() did not wake the blocked getter")
        self.assertEqual(seen, [None])

    def test_get_after_close_on_an_empty_queue_is_none(self):
        q = BoundedFrameQueue(maxsize=1)
        q.close()
        self.assertIsNone(q.get(timeout=5.0))


class CloseIsNotADrop(unittest.TestCase):
    """A refused put after shutdown must not inflate the lag counter."""

    def test_put_after_close_is_refused_without_counting_a_drop(self):
        q = BoundedFrameQueue(maxsize=4)
        q.put(_f(1, 0.0))
        q.close()
        self.assertFalse(q.put(_f(2, 0.1)))
        self.assertEqual(q.drops, 0,
                         "a frame refused because the consumer is gone is a shutdown, "
                         "not the analysis falling behind")
        self.assertTrue(q.closed)

    def test_frames_already_queued_survive_close(self):
        q = BoundedFrameQueue(maxsize=2)
        q.put(_f(1, 0.0))
        q.put(_f(2, 0.1))
        q.close()
        self.assertEqual(q.get().frame_id, 1)
        self.assertEqual(q.get().frame_id, 2)

    def test_close_is_idempotent(self):
        q = BoundedFrameQueue()
        q.close()
        q.close()
        self.assertTrue(q.closed)


class TheTimeoutIsMeasuredAgainstAFineClock(unittest.TestCase):
    """Review finding I-3: `get(timeout)` overshot, and the claim about it was false.

    The deadline used `time.monotonic`, which on this machine has ~16 ms resolution -
    measured, 20 000 consecutive calls returned a **single** distinct value. A deadline of
    "now + 1 ms" could not be represented. The default is now `time.perf_counter`.

    **Switching the clock did not fix the overshoot, and this test says so rather than
    claiming otherwise.** With `perf_counter`, `get` on an empty queue still measures
    14.89 ms p50 for a 1 ms timeout. The remaining overshoot is `Condition.wait` itself,
    which has the same Windows timer granularity - measured with no queue involved,
    `Condition.wait(0.001)` -> 15.24 ms p50 and `Event.wait(0.001)` -> 15.09 ms, while
    `time.sleep(0.001)` was accurate to 1.75 ms. Slicing cannot help; every wait has the
    same granularity.

    So this class pins three things: the clock is fine-grained, the overshoot is bounded
    and understood, and the brief's actual requirement - `None` rather than blocking
    forever - holds.
    """

    TIMEOUT = 0.001

    def test_the_default_clock_is_perf_counter_not_monotonic(self):
        import time

        self.assertIs(BoundedFrameQueue()._clock, time.perf_counter,
                      "the queue's default clock is not the one with ~16 ms resolution")

    def test_the_two_clocks_measured_on_this_machine(self):
        """Why the default changed. Both numbers measured here, not quoted."""
        import time

        coarse = len({time.monotonic() for _ in range(4000)})
        fine = len({time.perf_counter() for _ in range(4000)})
        self.assertLess(coarse, 100,
                        "time.monotonic() is no longer coarse here; this test and the "
                        "docstring that motivates it should both be revisited")
        self.assertGreater(fine, 3000,
                           "time.perf_counter() is not finer than monotonic() here")
        print(f"\n[clock resolution, distinct values in 4000 calls] "
              f"monotonic={coarse} perf_counter={fine}")

    def test_the_overshoot_is_bounded_and_is_the_condition_wait_not_the_clock(self):
        """Measured, not asserted against an aspiration.

        The bound below is the Windows timer granularity (~15.6 ms), because
        `Condition.wait` cannot do better here. If this test starts failing after a
        platform change, the number to update is this bound and the docstring's table -
        not the code.
        """
        overshoots = []
        for _ in range(20):
            q = BoundedFrameQueue()
            t0 = time.perf_counter()
            self.assertIsNone(q.get(timeout=self.TIMEOUT))
            overshoots.append((time.perf_counter() - t0) * 1000.0)
        overshoots.sort()
        median = overshoots[len(overshoots) // 2]
        self.assertLess(median, 30.0,
                        f"a {self.TIMEOUT * 1000:.0f} ms timeout took {median:.3f} ms; "
                        f"Condition.wait on Windows cannot resolve a sub-tick deadline, "
                        f"and 30 ms is the measured bound for it")
        print(f"\n[queue get(timeout={self.TIMEOUT * 1000:.0f}ms) overshoot, n=20] "
              f"p50={median:.3f} ms max={overshoots[-1]:.3f}")

    def test_the_shortest_timeout_still_returns_none_and_never_blocks_forever(self):
        """The brief's actual requirement, which held before and must still hold."""
        q = BoundedFrameQueue()
        self.assertIsNone(q.get(timeout=0.0))
        self.assertIsNone(q.get(timeout=0.001))
        self.assertIsNone(q.get(timeout=0.02))

    def test_a_frozen_injected_clock_blocks_and_that_is_documented(self):
        """A clock that does not advance has no deadline to reach.

        This is correct behaviour for a caller-supplied clock - the deadline is only
        meaningful in terms of the clock being used - but it was an easy way to hang a
        test, so it is pinned: `close()` is what releases such a getter.
        """
        frozen = _FrozenClock()
        q = BoundedFrameQueue(clock=frozen)
        t = threading.Thread(target=lambda: q.get(timeout=0.05), daemon=True)
        t.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and q.waiting == 0:
            time.sleep(0.005)
        self.assertEqual(q.waiting, 1)
        self.assertTrue(t.is_alive(), "a frozen clock cannot expire a deadline")
        q.close()
        t.join(5.0)
        self.assertFalse(t.is_alive(), "close() did not release the getter")


class _FrozenClock:
    def __call__(self) -> float:
        return 1000.0


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
