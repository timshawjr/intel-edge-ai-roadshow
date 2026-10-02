"""`FramePacket` - the identity of one frame, agreed on by every layer.

Review Focus #1: *"A result arrives after a newer frame was already displayed, and
the box paints onto the wrong frame."* A clip loops. Frame 7 of loop 0 and frame 7
of loop 1 are different pictures carrying the same `frame_id`, so an overlay that
binds to `frame_id` alone will happily paint loop 0's detections onto loop 1's
pixels the first time the clip rewinds. This module makes that expression
impossible rather than merely unlikely:

1. **`identity` is the four-tuple `(session_id, stream_id, source_loop_id,
   frame_id)`.** That is the whole identity; a result carries the packet it came
   from, and the overlay compares identities rather than trusting arrival order.
2. **Equality and hashing are identity, not pixels.** The default dataclass
   `__eq__` on a field holding an `np.ndarray` returns an *array*, so
   `assertEqual(packet_a, packet_b)` raises `ValueError: truth value of an array
   with more than one element is ambiguous` - a crash at the exact place a caller
   is trying to ask "is this the frame on screen?". Defining `__eq__` on the
   identity tuple makes that question answerable and makes `FramePacket` usable as
   a dict key, which is how a stale result gets recognised as stale.
3. **`age_at` measures from `captured_at`, not `decoded_at`.** Frame age is the
   delay between the picture being available at application ingress and being
   looked at (spec §7). Decode time is a separate, also-reported fact; using it
   would understate the age of every frame by the decode duration and would make
   the stall invisible.

**Validation is at construction, and it is aggressive on purpose.** A packet with
`width=0` divides by zero downstream; a packet whose declared `width`/`height`
disagree with `image.shape` is precisely the "correct box, wrong picture" failure
this contract exists to catch, and it is far cheaper to refuse it here than to
debug it on a booth screen. Every refusal names the field and the value, because
the alternative - a plausible wrong number reaching the UI - is the failure mode
this repository exists to prevent (AGENTS.md rule 1, rule 2).

This module is pure: `numpy` for the array it carries and nothing else. No clock is
read here - `now` is always a parameter, so every age in the product is a value
somebody passed in and can be reproduced in a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["FramePacket"]

#: A source timestamp may legitimately be 0.0 (the first frame) but never negative
#: and never non-finite. `nan` in particular would propagate silently: every
#: comparison against it is False, so a frame with `nan` age would look fresh for
#: ever.
def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be an int; got {value!r} ({type(value).__name__}). A float or "
            f"a bool here is a value that would be silently truncated."
        )
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {number}")
    return number


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a str; got {value!r} ({type(value).__name__})")
    if not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string; got {value!r}. An empty identifier "
            f"cannot be told apart from a missing one."
        )
    return value


def _require_seconds(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number of seconds; got {value!r}")
    number = float(value)
    # `nan` fails every comparison, so it must be refused explicitly rather than
    # by a `>=` test, which it would pass by accident in some formulations.
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0.0; got {number}")
    return number


@dataclass(frozen=True)
class FramePacket:
    """One frame, with everything needed to say *which* frame it is.

    `source_pts` is the container's own presentation timestamp, in seconds. It is
    not used for ageing - see `age_at` - and is carried because it is the only
    frame-time that means the same thing across decoders.

    `captured_at` and `decoded_at` are both monotonic-clock seconds, taken at
    application ingress and after decode respectively. `decoded_at >= captured_at`
    is enforced: the reverse is a broken clock, and a negative decode duration
    would render as a negative latency.

    `image` is `np.ndarray`, `uint8`, `(height, width, channels)` BGR as `cv2`
    hands it over. It is validated against `width`/`height` so that a caller cannot
    carry a 1920x1080 declaration beside a 640x640 letterbox and have every ROI
    land in the wrong place.
    """

    session_id: str
    scenario_id: str
    stream_id: str
    frame_id: int
    source_pts: float
    source_loop_id: int
    captured_at: float
    decoded_at: float
    width: int
    height: int
    image: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id",
                           _require_text("session_id", self.session_id))
        object.__setattr__(self, "scenario_id",
                           _require_text("scenario_id", self.scenario_id))
        object.__setattr__(self, "stream_id",
                           _require_text("stream_id", self.stream_id))
        object.__setattr__(self, "frame_id",
                           _require_int("frame_id", self.frame_id, minimum=0))
        object.__setattr__(self, "source_loop_id",
                           _require_int("source_loop_id", self.source_loop_id, minimum=0))
        object.__setattr__(self, "source_pts",
                           _require_seconds("source_pts", self.source_pts))
        captured = _require_seconds("captured_at", self.captured_at)
        decoded = _require_seconds("decoded_at", self.decoded_at)
        if decoded < captured:
            raise ValueError(
                f"decoded_at ({decoded}) must be >= captured_at ({captured}); the other "
                f"way round is a broken clock, and it would render as a negative decode "
                f"latency"
            )
        object.__setattr__(self, "captured_at", captured)
        object.__setattr__(self, "decoded_at", decoded)
        width = _require_int("width", self.width, minimum=1)
        height = _require_int("height", self.height, minimum=1)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        image = self.image
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"image must be a numpy array; got {type(image).__name__}. A list or a "
                f"None here would be a decode that did not happen."
            )
        if image.ndim != 3:
            raise ValueError(
                f"image must be (height, width, channels); got shape {image.shape!r}"
            )
        if image.shape[:2] != (height, width):
            raise ValueError(
                f"image is {image.shape[0]}x{image.shape[1]} but the packet declares "
                f"{height}x{width}. The declared extents are what every ROI is scaled "
                f"against, so a disagreement here draws correct boxes on the wrong "
                f"picture."
            )

    @property
    def identity(self) -> tuple[str, str, int, int]:
        """`(session_id, stream_id, source_loop_id, frame_id)`.

        The identity of the frame, and the only thing that distinguishes frame 7 of
        loop 0 from frame 7 of loop 1. An overlay binds to this, not to arrival
        order and not to `frame_id`.
        """
        return (self.session_id, self.stream_id, self.source_loop_id, self.frame_id)

    def age_at(self, now: float) -> float:
        """Seconds since this frame was available at ingress, at `now`.

        Measured from **`captured_at`**, per spec §7: frame age is how long the
        picture has been waiting to be shown. Clamped at zero rather than allowed
        to go negative - a negative age is not "very fresh", it is a clock
        disagreement, and rendering it as a large negative would be worse than
        rendering zero.

        `now` is a parameter. This module reads no clock.
        """
        return max(0.0, float(now) - self.captured_at)

    def is_from(self, loop_id: int) -> bool:
        """Whether this frame came from `loop_id` of the source.

        A result from a previous loop must be discarded on sight; this is the
        predicate that decides it.
        """
        return self.source_loop_id == loop_id

    def to_json(self) -> dict[str, Any]:
        """Metadata only. **Never the pixels.**

        `image_shape` and `image_dtype` are included because "this frame was
        1920x1080 BGR" is a fact worth logging and "these are the bytes" is not -
        a single 1920x1080 frame serialises to about 6 MB, and a log line per frame
        is how a run becomes unopenable.
        """
        return {
            "session_id": self.session_id,
            "scenario_id": self.scenario_id,
            "stream_id": self.stream_id,
            "frame_id": self.frame_id,
            "source_pts": self.source_pts,
            "source_loop_id": self.source_loop_id,
            "captured_at": self.captured_at,
            "decoded_at": self.decoded_at,
            "decode_latency_s": self.decoded_at - self.captured_at,
            "width": self.width,
            "height": self.height,
            "image_shape": [int(v) for v in self.image.shape],
            "image_dtype": str(self.image.dtype),
            "identity": list(self.identity),
        }

    def __eq__(self, other: object) -> bool:
        """Identity equality. Never pixel equality.

        The dataclass default would compare `image` with `==` and hand back an
        array, so `packet_a == packet_b` would raise rather than answer. Two
        packets with the same identity *are* the same frame as far as anything
        downstream is concerned; the pixels behind them were produced once, by the
        decoder that made them.
        """
        if not isinstance(other, FramePacket):
            return NotImplemented
        return self.identity == other.identity

    def __hash__(self) -> int:
        """Hash of the identity, so a packet can key a dict of in-flight results.

        This is what makes a late result identifiable: the overlay holds the
        displayed identity, and any result whose identity is not in the displayed
        set is obsolete and is dropped.
        """
        return hash(self.identity)