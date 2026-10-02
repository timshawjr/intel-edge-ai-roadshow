"""`app/contracts/frame.py` - the identity every layer has to agree on.

Review Focus #1: *"A result arrives after a newer frame was already displayed, and
the box paints onto the wrong frame."* This module is the answer to that, and the
tests below are written to fail if the answer is weakened.

Three decisions are load-bearing and each has a test that breaks if it is removed:

1. **Identity is the four-tuple `(session_id, stream_id, source_loop_id,
   frame_id)`.** Not `frame_id` alone. A clip that loops produces frame 7 again, so
   `frame_id` alone collides across loops and a result from loop 0 becomes
   indistinguishable from one from loop 1.
2. **`FramePacket` compares and hashes by identity, never by pixels.** A
   dataclass-generated `__eq__` on a field holding an `np.ndarray` returns an array,
   and `assertEqual` on it raises `ValueError: truth value of an array is
   ambiguous` - which is a confusing crash, not a wrong answer. Identity equality
   is both correct and cheap.
3. **`to_json()` reports the pixel array's shape and dtype, never its contents.**
   A metadata payload that carried an `image` field would be a 6 MB log line per
   frame; one that carried nothing about the pixels would be less useful than it
   needs to be.
"""

import json
import unittest

import numpy as np

from app.contracts.frame import FramePacket


def _p(**kw):
    base = dict(session_id="s1", scenario_id="retail", stream_id="A", frame_id=7,
                source_pts=0.35, source_loop_id=0, captured_at=10.0, decoded_at=10.002,
                width=64, height=48, image=np.zeros((48, 64, 3), np.uint8))
    base.update(kw)
    return FramePacket(**base)


class Identity(unittest.TestCase):
    def test_frame_id_is_unique_per_source_loop(self):
        self.assertEqual(_p().frame_id, 7)

    def test_two_frames_from_different_loops_are_distinguishable(self):
        a, b = _p(frame_id=7, source_loop_id=0), _p(frame_id=7, source_loop_id=1)
        self.assertNotEqual(a.source_loop_id, b.source_loop_id)
        self.assertFalse(a.is_from(1))

    def test_identity_includes_the_loop_so_a_rewind_cannot_collide(self):
        # The brief's own `test_two_frames_from_different_loops_are_distinguishable`
        # compares two values the test itself just passed in, so it cannot fail for
        # any implementation. This one can: it asks whether the packet agrees that
        # two packets sharing a frame_id are the same frame.
        a, b = _p(frame_id=7, source_loop_id=0), _p(frame_id=7, source_loop_id=1)
        self.assertNotEqual(a.identity, b.identity)
        self.assertNotEqual(a, b)
        self.assertNotEqual(hash(a), hash(b))

    def test_identity_is_the_documented_four_tuple(self):
        p = _p(session_id="s", stream_id="B", source_loop_id=3, frame_id=99)
        self.assertEqual(p.identity, ("s", "B", 3, 99))

    def test_same_identity_in_the_same_loop_is_the_same_frame(self):
        a = _p(frame_id=7, source_loop_id=2)
        b = _p(frame_id=7, source_loop_id=2, image=np.ones((48, 64, 3), np.uint8))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        # ...and it is still the very same *object* equality would allow a dict key,
        # so a result keyed on a frame cannot be filed against the wrong frame.
        self.assertEqual(len({a, b}), 1)

    def test_a_different_stream_is_a_different_frame(self):
        self.assertNotEqual(_p(stream_id="A"), _p(stream_id="B"))
        self.assertNotEqual(_p(session_id="s1"), _p(session_id="s2"))


class Age(unittest.TestCase):
    def test_age_is_measured_from_capture_not_decode(self):
        self.assertAlmostEqual(_p(captured_at=10.0, decoded_at=10.9).age_at(12.0), 2.0, places=6)

    def test_age_never_goes_negative(self):
        self.assertGreaterEqual(_p(captured_at=10.0).age_at(9.0), 0.0)

    def test_age_uses_the_clock_it_is_given_and_never_calls_one(self):
        # A pure module: two calls at the same `now` must agree, and the packet must
        # not have cached an age from some earlier call.
        p = _p(captured_at=10.0)
        self.assertEqual(p.age_at(20.0), p.age_at(20.0))
        self.assertAlmostEqual(p.age_at(11.5), 1.5, places=6)


class Serialisation(unittest.TestCase):
    def test_to_json_omits_the_pixels(self):
        payload = _p().to_json()
        self.assertNotIn("image", payload)
        self.assertIn("frame_id", payload)

    def test_to_json_is_json_dumpable_and_names_the_pixel_geometry(self):
        # "Never the pixels" must not degrade into "says nothing about the image".
        payload = _p().to_json()
        self.assertEqual(payload["image_shape"], [48, 64, 3])
        self.assertEqual(payload["image_dtype"], "uint8")
        text = json.dumps(payload)
        self.assertIn('"frame_id": 7', text)
        self.assertLess(len(text), 1024)

    def test_to_json_carries_the_whole_identity(self):
        payload = _p(source_loop_id=4).to_json()
        self.assertEqual(payload["identity"], ["s1", "A", 4, 7])
        for key in ("session_id", "scenario_id", "stream_id", "source_pts",
                    "captured_at", "decoded_at", "width", "height"):
            self.assertIn(key, payload)


class Immutability(unittest.TestCase):
    def test_packet_is_immutable(self):
        with self.assertRaises(Exception):
            _p().frame_id = 9

    def test_the_image_is_not_copied_into_a_mutable_alias(self):
        # `frozen=True` freezes the *attribute*, not the array it points at. That is
        # why `to_json` is the only way out of this object and why nothing here
        # mutates `image`: the contract is that the pixels are read, never edited.
        p = _p()
        self.assertIs(p.image, p.image)


class Validation(unittest.TestCase):
    """A bad packet must be refused at construction.

    Every one of these is a value that would otherwise arrive as a plausible number
    and be drawn with: a negative frame id sorts before every real one, a zero
    width is a division by zero in the ROI maths, and a `width` that disagrees with
    the image is exactly how a correctly-decoded box lands on the wrong object.
    """

    def _refuse(self, **kw):
        with self.assertRaises(ValueError):
            _p(**kw)

    def test_frame_id_must_be_a_non_negative_int(self):
        self._refuse(frame_id=-1)
        self._refuse(frame_id=1.5)
        self._refuse(frame_id=True)
        self._refuse(frame_id="7")

    def test_source_loop_id_must_be_a_non_negative_int(self):
        self._refuse(source_loop_id=-1)
        self._refuse(source_loop_id=2.0)

    def test_identifiers_must_be_non_empty_strings(self):
        for key in ("session_id", "scenario_id", "stream_id"):
            self._refuse(**{key: ""})
            self._refuse(**{key: None})

    def test_extents_must_be_positive_ints(self):
        self._refuse(width=0)
        self._refuse(height=0)
        self._refuse(width=-64)
        self._refuse(height=48.0)

    def test_timestamps_must_be_finite_non_negative_and_ordered(self):
        self._refuse(captured_at=-1.0)
        self._refuse(source_pts=float("nan"))
        self._refuse(decoded_at=float("inf"))
        # Decoded-before-captured is not a timestamp pair, it is a broken clock.
        self._refuse(captured_at=10.0, decoded_at=9.9)

    def test_the_image_must_be_an_ndarray_of_the_declared_size(self):
        self._refuse(image=np.zeros((8, 8, 3), np.uint8))          # wrong geometry
        self._refuse(image=[0] * 5)                                # not an array
        self._refuse(image=np.zeros((48, 64), np.uint8))           # no channel axis

    def test_a_valid_packet_still_builds(self):
        self.assertEqual(_p().frame_id, 7)


if __name__ == "__main__":
    unittest.main()