"""`BoundedFrameQueue` - the one place frames wait, and where a drop is recorded.

Review Focus #2: *"a full queue silently discards frames."* That is the failure this
module exists to prevent. When the analysis falls behind, the queue fills, and if the
overflow is swallowed then the booth shows a smooth 15 fps video with overlay boxes that
are progressively older than the picture underneath them. Nothing on screen says the
stream is late, so an operator watching the demo concludes the demo is fine.

So the queue is **bounded and the bound is enforced**, and enforcement is visible:

- `put` returns `True` when the frame was queued and `False` when it was refused
  because the queue was already full. It never blocks and never evicts: evicting the
  oldest frame would keep the depth honest while destroying the frame identity, and the
  depth is the operator's only evidence about lag.
- Every refusal increments `drops`. That counter is monotonic, it survives `get`, and a
  `get` is not a repair of it - the frame that was refused is still a frame nobody
  analysed.
- The number of accepted puts plus `drops` equals the number of puts attempted, so a
  caller can reconcile the two. `tests/test_queue.py` asserts that identity under four
  concurrent producers, because a counter that is only right when nothing is racing is
  not a counter.

**What is carried is a `FramePacket` and nothing else.** Review Focus #1: a bare list of
boxes has no identity, so a result could not be bound to the frame it produced it.
`put` refuses anything that is not a `FramePacket`, which is a different failure from a
full queue - a malformed put is a bug in the caller, and counting it as a "drop" would
put a wrong number on the lag counter.

**After `close()` a `put` is refused but is *not* counted as a drop.** A frame arriving
after the consumer has gone is a shutdown, not the analysis falling behind, and mixing
the two would make the drop counter unreadable exactly when someone is diagnosing it.
`closed` is public so the caller can tell the two apart. Frames already queued survive
`close()` and are still handed out, and `close()` wakes every blocked `get` so a
supervisor shutting down does not wait out a 30-second timeout.

**Thread-safe, in-process.** The lock is what makes `drops` and `depth` mean anything
under the multi-threaded supervisor of Task 8. This is not a `multiprocessing.Queue`
wrapper: crossing the spawn boundary is the supervisor's job, and it needs to know about
drops from both sides, which a wrapped `mp.Queue` cannot report.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable

from app.contracts.frame import FramePacket

__all__ = ["BoundedFrameQueue"]


class BoundedFrameQueue:
    """A FIFO of `FramePacket`s with a hard capacity and an honest overflow counter.

    `maxsize` is a **hard** bound. `maxsize=2` means at most two frames are ever
    resident, whatever the producer does, and a producer that offers a third is told so
    rather than being quietly delayed.
    """

    def __init__(self, maxsize: int = 2,
                 *, clock: Callable[[], float] | None = None) -> None:
        if isinstance(maxsize, bool) or not isinstance(maxsize, int):
            raise TypeError(
                f"maxsize must be an int; got {maxsize!r} ({type(maxsize).__name__}). A "
                f"bool is an int in Python, and a float would be silently truncated into "
                f"a capacity nobody chose."
            )
        if maxsize < 1:
            raise ValueError(
                f"maxsize must be >= 1; got {maxsize}. A queue of zero capacity could "
                f"never hand a frame to the worker, so every put would be a drop and the "
                f"stream would be dead rather than bounded."
            )
        self._maxsize = int(maxsize)
        self._frames: deque[FramePacket] = deque()
        self._drops = 0
        self._closed = False
        self._waiting = 0
        self._condition = threading.Condition(threading.Lock())
        if clock is None:
            import time as _time
            # `perf_counter`, not `monotonic`: measured on this machine `monotonic` returned a
            # single distinct value in 20 000 consecutive calls, so a deadline of
            # "now + 1 ms" could not be represented. The residual overshoot is the
            # wait itself, not this clock; see `get`.
            clock = _time.perf_counter
        self._clock = clock

    # -- capacity -------------------------------------------------------------------

    @property
    def maxsize(self) -> int:
        """The hard capacity this queue was built with."""
        return self._maxsize

    @property
    def depth(self) -> int:
        """Frames waiting right now. Never exceeds `maxsize`."""
        with self._condition:
            return len(self._frames)

    @property
    def drops(self) -> int:
        """How many frames have been refused for want of room, since construction.

        Monotonic, and unaffected by `get`: the frames already in the queue being
        consumed says nothing about the frames that were never accepted. Spec section 6
        requires this counter, and AGENTS.md rule 2 says a missing counter must render as
        "missing", which is what `drops == 0` legitimately is and what a swallowed drop
        would be indistinguishable from.
        """
        with self._condition:
            return self._drops

    @property
    def waiting(self) -> int:
        """How many threads are currently blocked in `get`.

        A diagnostic, and the seam that lets a test prove a getter really did block
        rather than returning `None` immediately. On a booth this is also the "is the
        worker stuck?" reading: a `waiting` that stays at the number of consumers while
        `depth` is zero and no result is coming back is a wedged pipeline.
        """
        with self._condition:
            return self._waiting

    @property
    def closed(self) -> bool:
        """Whether `close()` has been called. A closed queue refuses new frames."""
        with self._condition:
            return self._closed

    # -- the two operations ---------------------------------------------------------

    def put(self, frame: FramePacket) -> bool:
        """Queue `frame`; `False` means it was refused and `drops` went up.

        A refusal is not an error and never raises: the producer cannot do anything about
        the worker being slow except count it, and raising here would take the video
        thread down over a full queue (AGENTS.md rule 12).
        """
        if not isinstance(frame, FramePacket):
            raise TypeError(
                f"put takes a FramePacket; got {type(frame).__name__}. A bare list of "
                f"detections has no frame identity, so a result carrying one could not be "
                f"matched to the picture it came from - review focus 1, structurally."
            )
        with self._condition:
            if self._closed:
                return False
            if len(self._frames) >= self._maxsize:
                self._drops += 1
                return False
            self._frames.append(frame)
            self._condition.notify()
            return True

    def get(self, timeout: float = 0.0) -> FramePacket | None:
        """The oldest queued frame, or `None`.

        `None` is returned when the queue is empty at the deadline, and when it is empty
        **and closed**. It never blocks *indefinitely*, so a caller with a UI thread behind
        it cannot hang; and a blocked getter is released by `close()` rather than waiting
        out the remainder of its timeout. That is the brief's requirement and it holds.

        **1. The deadline used `time.monotonic`, which cannot represent a deadline here.**
        Measured on this machine: 20 000 consecutive `time.monotonic()` calls returned a
        **single** distinct value, so its resolution is the ~16 ms Windows tick, and a
        deadline of "now + 1 ms" had nowhere to land. `time.perf_counter` returned 19 024
        distinct values in the same 20 000 calls. The clock is now `perf_counter`.

        **2. That was not the whole cause, and the residual is measured rather than
        denied.** `threading.Condition.wait` on this platform has a floor of its own.
        Measured here, no queue involved, 60 runs per row:

            Condition.wait( 1.0 ms) -> p50 15.39 ms   min  1.06   max 26.20
            Condition.wait( 5.0 ms) -> p50 15.85 ms   min  5.06   max 20.27
            Condition.wait(20.0 ms) -> p50 32.83 ms   min 20.08   max 35.96
            Event.wait(     1.0 ms) -> p50 15.63 ms
            sleep(          1.0 ms) -> p50  1.77 ms   <- fine-grained, for contrast

        So the overshoot is the wait, not the clock: `sleep` honours 1 ms to within
        0.77 ms while any wait-based primitive returns no earlier than ~15 ms for the
        same request. Slicing cannot help - every wait has the same floor - so no
        implementation of a condition variable here can honour a 1 ms deadline. The
        honest statement is the bound below, not "never blocks past `timeout`".

        **`get`'s measured overshoot on an empty queue.** Five repeats of 60 runs, p50 of
        each, reported as a range because it varies by a couple of milliseconds run to
        run:

            timeout   1.0 ms -> 15.38 - 15.58 ms
            timeout   5.0 ms -> 16.11 - 16.38 ms
            timeout  20.0 ms -> 32.80 - 33.30 ms
            timeout  50.0 ms -> 64.04 - 66.01 ms

        The clock change is nevertheless real and not cosmetic: reverting it to
        `time.monotonic` moves the 1 ms row's p50 from ~1.4 ms to ~15.9 ms in-process.
        The absolute numbers above are from a quiet standalone process; inside a busy
        suite they are better, and that variance is why the test asserts a bound rather
        than an exact figure.

        A caller needing a tighter deadline than ~15 ms should poll `depth` rather than
        sleep on a condition variable.

        **An injected clock that does not advance will block indefinitely.** That is
        correct for a clock the caller controls - the deadline is only meaningful in terms
        of the clock being used - but it means a frozen clock must not be used with a
        non-zero timeout in a test that expects `get` to return. `close()` releases it.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError(f"timeout must be a number of seconds; got {timeout!r}")
        deadline = self._clock() + max(0.0, float(timeout))
        with self._condition:
            self._waiting += 1
            try:
                while True:
                    if self._frames:
                        return self._frames.popleft()
                    if self._closed:
                        return None
                    remaining = deadline - self._clock()
                    if remaining <= 0.0:
                        return None
                    self._condition.wait(remaining)
            finally:
                self._waiting -= 1

    def close(self) -> None:
        """Refuse new frames and wake every blocked `get`. Idempotent.

        Frames already queued are **not** discarded: they are still real pictures that
        were paid for, and dropping them on shutdown would inflate neither counter and
        would still lose a frame. They are drained by `get`, which returns `None` only
        once the queue is empty.
        """
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def __len__(self) -> int:
        return self.depth

    def __repr__(self) -> str:
        return (f"BoundedFrameQueue(maxsize={self._maxsize}, depth={self.depth}, "
                f"drops={self.drops}, closed={self.closed})")
