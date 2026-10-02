"""The clip source: frame identity across loops, and an analysis rate that is not the
source rate.

Two things this module is tested for, both of which the brief asked for and both of
which have a specific way of being got wrong:

1. **`restart()` must make loop 0 frame 37 and loop 1 frame 37 different frames.** The
   pictures are identical - it is the same clip - and `frame_id` alone cannot tell them
   apart. The test asserts both halves of that: the *identity tuples differ*, and the
   *pixels are the same*. A source that faked a different `frame_id` on a restart would
   pass the first and fail the second; one that left the loop id at 0 passes the second
   and fails the first.

2. **The analysis rate is `target_fps`, not the source's 29.97.** Decoding a 1080p clip
   at 30 fps and analysing every frame is the demo's budget gone; the booth runs the
   analysis at 15. Because the source frame grid is discrete, 15 out of 29.97 is
   quantised - `step = round(29.97 / 15) = 2` source frames per analysis frame, so the
   achieved rate is **14.985**, not exactly 15. The module therefore exposes
   `analysis_fps` (achieved) alongside `target_fps` (requested), and the tests measure
   both the arithmetic and a real wall-clock rate.

Every measured clip fact - extent, frame count, fps - is read from `config/models.json`
through `app.engine.retail_config`, never hardcoded here, and cross-checked against what
`cv2.VideoCapture` reports for the file on disk. If the clip were re-provisioned, the
cross-check would fail rather than quietly describing the new file with the old numbers.

The clip itself is required. A missing or unreadable clip **skips** these tests with the
reason stated; it does not pass them vacuously, and it does not fail on a machine that
was never provisioned with the assets.
"""

import time
import unittest
from pathlib import Path

from app.engine.retail_config import (
    RETAIL_CHECKOUT_CLIP,
    RETAIL_CHECKOUT_ID,
    clip_measurements,
)
from app.engine.source import ClipSource, recorded_media_for
from app.inference.pinned import REPO_ROOT

CLIP = REPO_ROOT / RETAIL_CHECKOUT_CLIP
RECORDED = clip_measurements(RETAIL_CHECKOUT_ID)

#: A deliberately absurd target for the tests that have to reach the end of the clip.
#: `step` becomes `round(29.97 / 0.01) = 2997`, so the whole 3387-frame clip is crossed
#: in two reads rather than two minutes - and every one of those frames is still really
#: grabbed and decoded by the source, so the loop path is the production path.
COARSE_FPS = 0.01


class _VirtualClock:
    """A monotonic clock that advances only when something sleeps on it.

    Injected into `ClipSource` so the pacing *schedule* can be asserted exactly, in
    microseconds, rather than approximately and in milliseconds. It is not a mock of the
    thing under test: `ClipSource` computes the deadlines, and this only records what it
    asked for and moves time forward by exactly that much.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        seconds = float(seconds)
        self.slept.append(seconds)
        self.now += max(0.0, seconds)


def _source(**kwargs) -> ClipSource:
    """A source on the real clip, on a virtual clock unless a test asks for real time."""
    clock = _VirtualClock()
    kwargs.setdefault("clock", clock)
    kwargs.setdefault("sleeper", clock.sleep)
    return ClipSource(CLIP, stream_id="A", session_id="s",
                      scenario_id="retail_sanity", **kwargs)


def _require_clip() -> None:
    if not CLIP.exists():
        raise unittest.SkipTest(
            f"the pinned clip {CLIP} is not provisioned in this tree, so nothing about "
            f"its measured extents, rate or length can be asserted here. Provision "
            f"assets/ with tools/provision_venv.py. This test is not passing vacuously; "
            f"it is not running."
        )


def _exhaust(source: ClipSource) -> list:
    """Every packet the source will give before it declares itself finished."""
    packets = []
    while True:
        packet = source.read()
        if packet is None:
            return packets
        packets.append(packet)


class MeasuredIdentityOfTheClip(unittest.TestCase):
    """The source reports the recorded measurements, and agrees with the container."""

    def setUp(self):
        _require_clip()

    def test_the_total_frame_count_is_the_recorded_one_and_the_container_agrees(self):
        with _source() as source:
            self.assertEqual(source.total_frames, int(RECORDED["measured_frames"]))
            self.assertEqual(source.container_total_frames, source.total_frames)

    def test_the_source_fps_is_the_recorded_one_and_the_container_agrees(self):
        with _source() as source:
            self.assertAlmostEqual(source.source_fps, float(RECORDED["measured_fps"]), places=6)
            self.assertAlmostEqual(source.source_fps, source.container_source_fps, places=6)

    def test_the_recorded_duration_is_frames_over_fps(self):
        implied = int(RECORDED["measured_frames"]) / float(RECORDED["measured_fps"])
        self.assertAlmostEqual(implied, float(RECORDED["measured_duration_s"]), places=3)

    def test_frames_carry_the_recorded_extents(self):
        with _source() as source:
            packet = source.read()
        self.assertEqual(packet.width, int(RECORDED["measured_width"]))
        self.assertEqual(packet.height, int(RECORDED["measured_height"]))
        self.assertEqual(packet.image.shape[:2], (packet.height, packet.width))
        self.assertEqual(packet.image.dtype.name, "uint8")

    def test_every_frame_matches_its_own_declared_extents(self):
        with _source() as source:
            for _ in range(4):
                packet = source.read()
                self.assertEqual(packet.image.shape[1], packet.width)
                self.assertEqual(packet.image.shape[0], packet.height)

    def test_recorded_media_for_finds_the_pinned_clip_by_path(self):
        entry = recorded_media_for(CLIP)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["id"], RETAIL_CHECKOUT_ID)
        self.assertEqual(recorded_media_for(Path("nowhere/at/all.mp4")), None)


class IdentityAcrossReads(unittest.TestCase):
    def setUp(self):
        _require_clip()

    def test_frame_ids_rise_and_captures_are_monotonic(self):
        with _source(target_fps=15.0) as source:
            packets = [source.read() for _ in range(6)]
        ids = [p.frame_id for p in packets]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(set(ids)), len(ids), "a repeat here would break identity")
        captures = [p.captured_at for p in packets]
        self.assertEqual(captures, sorted(captures))
        for p in packets:
            self.assertGreaterEqual(p.decoded_at, p.captured_at)

    def test_every_packet_is_the_identity_it_claims(self):
        with _source() as source:
            packet = source.read()
            self.assertEqual(packet.identity, ("s", "A", 0, packet.frame_id))
            self.assertEqual(packet.session_id, "s")
            self.assertEqual(packet.stream_id, "A")
            self.assertEqual(packet.scenario_id, "retail_sanity")

    def test_source_pts_is_the_frame_index_over_the_measured_fps(self):
        with _source(target_fps=15.0) as source:
            packets = [source.read() for _ in range(3)]
            rate = source.source_fps
        for p in packets:
            self.assertAlmostEqual(p.source_pts, p.frame_id / rate, places=12)
        pts = [p.source_pts for p in packets]
        self.assertEqual(pts, sorted(pts))
        # One analysis frame is exactly `step` source frames, so consecutive
        # presentation timestamps advance by step/source_fps and by nothing else.
        self.assertAlmostEqual(pts[1] - pts[0], 2 / rate, places=12)


class TheAnalysisRateIsNotTheSourceRate(unittest.TestCase):
    """The brief's `target_fps`, the source's 29.97, and the number in between."""

    def setUp(self):
        _require_clip()

    def test_the_step_is_two_at_fifteen_fps_over_a_twenty_nine_nine_seven_fps_clip(self):
        with _source(target_fps=15.0) as source:
            self.assertEqual(source.source_fps, float(RECORDED["measured_fps"]))
            self.assertEqual(source.step, 2)
            self.assertEqual(source.target_fps, 15.0)
            # The honest reading: 15 was requested, the discrete frame grid delivers
            # 29.97002997 / 2 = 14.985. A module that reported 15.000 here would be
            # reporting the request, not the rate.
            self.assertAlmostEqual(source.analysis_fps, source.source_fps / 2, places=9)
            self.assertAlmostEqual(source.analysis_fps, 15.0, delta=0.02)
            self.assertLess(source.analysis_fps, source.source_fps)

    def test_reads_skip_source_frames_rather_than_analysing_every_one(self):
        with _source(target_fps=15.0) as source:
            ids = [source.read().frame_id for _ in range(5)]
        self.assertEqual(ids, [0, 2, 4, 6, 8])

    def test_the_pacing_schedule_is_exactly_one_over_target_fps(self):
        clock = _VirtualClock()
        with _source(target_fps=15.0, clock=clock, sleeper=clock.sleep) as source:
            for _ in range(4):
                source.read()
        # The first frame is delivered immediately; a booth should not stare at a black
        # screen for a pacing interval before the first picture.
        self.assertEqual(len(clock.slept), 3)
        for delay in clock.slept:
            self.assertAlmostEqual(delay, 1.0 / 15.0, places=12)

    def test_the_achieved_rate_on_a_real_clock_is_near_the_target_and_below_the_source(self):
        source = ClipSource(CLIP, stream_id="A", session_id="s",
                            scenario_id="retail_sanity", target_fps=15.0)
        self.addCleanup(source.close)
        count = 9
        started = time.perf_counter()
        for _ in range(count):
            self.assertIsNotNone(source.read())
        elapsed = time.perf_counter() - started
        source.close()
        achieved = (count - 1) / elapsed
        self.assertLess(achieved, float(RECORDED["measured_fps"]) * 0.75,
                        "the analysis rate is the source rate; nothing is being skipped")
        self.assertAlmostEqual(achieved, 15.0, delta=1.5)

    def test_a_target_at_the_source_rate_reads_every_frame(self):
        clock = _VirtualClock()
        with _source() as probe:
            at_source = ClipSource(CLIP, stream_id="A", session_id="s",
                                   scenario_id="retail_sanity",
                                   target_fps=probe.source_fps,
                                   clock=clock, sleeper=clock.sleep)
        with at_source:
            self.assertEqual(at_source.step, 1)
            self.assertEqual([at_source.read().frame_id for _ in range(4)], [0, 1, 2, 3])

    def test_a_target_faster_than_the_source_still_steps_by_at_least_one(self):
        with _source(target_fps=240.0) as fast:
            self.assertEqual(fast.step, 1)
            self.assertAlmostEqual(fast.analysis_fps, fast.source_fps, places=9)

    def test_advance_moves_the_position_without_decoding_or_pacing(self):
        clock = _VirtualClock()
        with _source(target_fps=15.0, clock=clock, sleeper=clock.sleep) as source:
            self.assertEqual(source.position, 0)
            self.assertEqual(source.advance(), 2)
            self.assertEqual(source.advance(), 4)
            self.assertEqual(clock.slept, [], "advance() paced, so it is a read in disguise")
            self.assertEqual(source.position, 4)
            # The skipped frames were not decoded, so the next read is frame 4.
            self.assertEqual(source.read().frame_id, 4)


class TheLoopIsVisible(unittest.TestCase):
    """Review Focus #1, at the source: two loops of one clip are two sets of frames."""

    def setUp(self):
        _require_clip()

    def test_reading_to_the_end_marks_finished_and_does_not_wrap_by_itself(self):
        with _source(target_fps=COARSE_FPS) as source:
            packets = _exhaust(source)
            self.assertGreater(len(packets), 1)
            self.assertTrue(source.finished)
            self.assertGreaterEqual(source.position, source.total_frames)
            self.assertEqual(source.loop_id, 0)
            self.assertLess(max(p.frame_id for p in packets), source.total_frames)
            self.assertIsNone(source.read(), "the source wrapped without being asked to")
            self.assertEqual(source.loop_id, 0)
            self.assertIsNone(source.read())

    def test_reading_past_the_end_does_not_sleep_on_the_pacing_deadline(self):
        clock = _VirtualClock()
        with _source(target_fps=COARSE_FPS, clock=clock, sleeper=clock.sleep) as source:
            _exhaust(source)
            slept = len(clock.slept)
            for _ in range(5):
                self.assertIsNone(source.read())
            self.assertEqual(len(clock.slept), slept,
                             "an exhausted source still paces, so a polling supervisor "
                             "runs at that rate instead of returning immediately")

    def test_restart_returns_the_new_loop_id_and_the_packets_carry_it(self):
        with _source(target_fps=15.0) as source:
            first = [source.read() for _ in range(3)]
            self.assertEqual(source.restart(), 1)
            second = [source.read() for _ in range(3)]
            self.assertEqual(source.restart(), 2)
            self.assertEqual(source.loop_id, 2)
            self.assertEqual([p.frame_id for p in first], [p.frame_id for p in second])
        self.assertEqual([p.source_loop_id for p in first], [0, 0, 0])
        self.assertEqual([p.source_loop_id for p in second], [1, 1, 1])

    def test_the_same_frame_in_two_loops_has_two_identities_and_the_same_pixels(self):
        with _source(target_fps=COARSE_FPS) as source:
            loop0 = _exhaust(source)
            source.restart()
            loop1 = _exhaust(source)
        for a, b in zip(loop0, loop1):
            self.assertEqual(a.frame_id, b.frame_id)
            self.assertEqual(a.source_pts, b.source_pts)
            self.assertNotEqual(
                a.identity, b.identity,
                "two loops of one clip must not share an identity, or an overlay binds "
                "to loop 0's detections and paints them onto loop 1's pixels")
            self.assertEqual(int(a.image.sum()), int(b.image.sum()),
                             "the pixels are the same clip rewound; if they are not, "
                             "this test is comparing two different pictures")

    def test_restart_rewinds_the_decoder_not_just_the_index(self):
        with _source() as reference:
            first_frame_sum = int(reference.read().image.sum())
        with _source(target_fps=15.0) as source:
            for _ in range(5):
                self.assertIsNotNone(source.read())
            self.assertEqual(source.restart(), 1)
            self.assertEqual(source.position, 0)
            self.assertFalse(source.finished)
            rewound = source.read()
        self.assertEqual(rewound.source_loop_id, 1)
        self.assertEqual(rewound.frame_id, 0)
        self.assertEqual(int(rewound.image.sum()), first_frame_sum,
                         "the decoder was not rewound, so 'frame 0 of loop 1' is "
                         "whatever the decoder happened to be sitting on")


class Refusals(unittest.TestCase):
    """A source that cannot start must say so, not present an empty stream."""

    def test_a_missing_clip_is_refused_at_construction(self):
        missing = REPO_ROOT / "assets" / "media" / "not-a-real-clip.mp4"
        with self.assertRaises(FileNotFoundError) as caught:
            ClipSource(missing, stream_id="A", session_id="s", scenario_id="retail_sanity")
        self.assertIn("not-a-real-clip.mp4", str(caught.exception))

    def test_a_non_positive_target_fps_is_refused(self):
        _require_clip()
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                ClipSource(CLIP, stream_id="A", session_id="s",
                           scenario_id="retail_sanity", target_fps=bad)

    def test_a_non_positive_explicit_source_fps_is_refused(self):
        _require_clip()
        with self.assertRaises(ValueError):
            ClipSource(CLIP, stream_id="A", session_id="s", scenario_id="retail_sanity",
                       source_fps=0.0)

    def test_an_empty_identifier_is_refused(self):
        _require_clip()
        base = {"stream_id": "A", "session_id": "s", "scenario_id": "retail_sanity"}
        for name in ("stream_id", "session_id", "scenario_id"):
            for blank in ("", "   "):
                with self.assertRaises(ValueError):
                    ClipSource(CLIP, **{**base, name: blank})

    def test_reading_after_close_raises_rather_than_claiming_the_clip_finished(self):
        _require_clip()
        source = _source()
        source.read()
        source.close()
        with self.assertRaises(RuntimeError):
            source.read()

    def test_close_is_idempotent(self):
        _require_clip()
        source = _source()
        source.close()
        source.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
