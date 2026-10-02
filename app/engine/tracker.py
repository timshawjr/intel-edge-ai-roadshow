"""`app/engine.tracker` - IoU tracking with class-stable ids and an explicit reset.

Review Focus #4: *"a track id is silently reused across classes, so a person becomes
a product."* A detector returns a flat list of boxes with no identity, so identity
has to be manufactured here. Two rules make the manufactured identity trustworthy:

1. **A detection only ever matches a track of the same `class_id`.** Matching is
   greedy by IoU **within each class**, never across the whole set. Without the class
   partition, a large `dining_table` box overlapping a `person` would out-rank the
   person and claim its id, and every downstream count would then be a person
   labelled as furniture. This is not a rare race - a person standing at a counter
   is the exact geometry that triggers it, and it is the geometry in the retail
   footage.
2. **`reset()` clears the id counter as well as the tracks.** A clip that loops
   presents the same object again; if the counter restarted, the new track would
   carry the old track's id and a pending event in the event machine would be
   completed by an object that merely looks like the one that started it. Spec §9
   requires an explicit reset per source loop, and this is what it is protecting.

**Ids are unique for the lifetime of one `Tracker`, never reused.** That is what
makes `Track.track_id` usable as the identity half of an event `dedup_key`, which is
the whole basis of Review Focus #3.

`confirmed` is a claim about evidence, and it is what the event machine requires
before counting anything: a track is confirmed at `min_hits` matches and stays
confirmed thereafter. A track that has drifted (many consecutive misses) is reported
with `confirmed=False` even though it was once confirmed, because a box the tracker
is guessing at is not evidence of a present object.

Pure: no clock - `now` is a parameter - no numpy (the geometry is 4-tuples), no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

__all__ = ["Track", "Tracker", "iou"]


def _require_threshold(name: str, value: Any, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number; got {value!r}")
    number = float(value)
    if number != number or not minimum <= number <= maximum:
        raise ValueError(
            f"{name} must be within [{minimum}, {maximum}]; got {value!r}"
        )
    return number


def _require_count(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int; got {value!r} ({type(value).__name__})")
    if value < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}; got {value}. "
            + ("A max_misses of 0 would delete every track the frame it was created, "
               "so no track could ever be confirmed." if name == "max_misses" else "")
        )
    return value


def iou(a: tuple[float, float, float, float],
        b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two `xyxy` boxes. `0.0` when they cannot overlap.

    Degenerate boxes - zero or negative area - score `0.0` and therefore never match.
    This is the same decision M0's `nms` made and for the same reason: a zero-area
    box makes `union` zero, and an implementation that divides anyway raises, while
    one that guards the division and returns `0.0` would match *every* box at
    threshold 0. Either way a degenerate detection becomes a track that follows
    whatever else is on screen.
    """
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


@dataclass(frozen=True)
class Track:
    """One tracked object, as of one observation.

    `first_seen` and `last_seen` are the values of `now` passed to `update` on the
    frames this track was created and last matched. They are not wall-clock times and
    nothing in this module reads a clock.

    `hits` counts matched observations; `misses` counts consecutive frames in which
    this track was **not** matched. `confirmed` is this track's own verdict on
    whether `hits` is enough evidence to count it.
    """
    track_id: int
    class_id: int
    box: tuple[float, float, float, float]
    first_seen: float
    last_seen: float
    hits: int
    misses: int
    confirmed: bool

    def __post_init__(self) -> None:
        for name, value in (("track_id", self.track_id), ("class_id", self.class_id),
                            ("hits", self.hits), ("misses", self.misses)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Track.{name} must be an int; got {value!r}")
        if self.track_id < 1:
            raise ValueError(
                f"Track.track_id must be >= 1; got {self.track_id}. Ids start at 1 so "
                f"that 0 can never be mistaken for a track, and so an event dedup_key "
                f"containing it is still a real key."
            )
        if self.class_id < 0:
            raise ValueError(f"Track.class_id must be >= 0; got {self.class_id}")
        if self.hits < 1:
            raise ValueError(
                f"Track.hits must be >= 1; got {self.hits}. A track with no hits has "
                f"never been observed, so it is not a track."
            )
        if self.misses < 0:
            raise ValueError(f"Track.misses must be >= 0; got {self.misses}")
        if not isinstance(self.confirmed, bool):
            raise ValueError(
                f"Track.confirmed must be a bool; got {self.confirmed!r}. It is the "
                f"event machine's gate on counting anything, so a truthy non-bool must "
                f"not be read as evidence."
            )
        box = tuple(float(v) for v in self.box)
        if len(box) != 4:
            raise ValueError(f"Track.box must be (x1, y1, x2, y2); got {self.box!r}")
        object.__setattr__(self, "box", box)
        object.__setattr__(self, "first_seen", float(self.first_seen))
        object.__setattr__(self, "last_seen", float(self.last_seen))

    @property
    def is_current(self) -> bool:
        """Whether this track was matched by the most recent `update`.

        The event machine reads this rather than `misses`, because a coasted track's
        `box` is the last one seen and is not a statement about the present.
        """
        return self.misses == 0

    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class Tracker:
    """Greedy IoU tracker. Same class matches same class, and ids are never reused.

    `iou_threshold=0.3` and `max_misses=2` are the defaults the plan specifies.
    `min_hits=2` is the confirmation gate: a single detection is a candidate, not
    evidence, and the event machine counts nothing that has not reached it.

    Matching is greedy by descending IoU, computed **per class**. Ties break on the
    lower `track_id`, so the outcome does not depend on the order detections happened
    to arrive in - which matters because that order is whatever the decoder's
    score-sorted list gave, and a score sort is not stable across frames.
    """

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 2,
                 min_hits: int = 2) -> None:
        self._iou_threshold = _require_threshold("iou_threshold", iou_threshold,
                                                 minimum=0.0, maximum=1.0)
        self._max_misses = _require_count("max_misses", max_misses, minimum=1)
        self._min_hits = _require_count("min_hits", min_hits, minimum=1)
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._frames = 0

    @property
    def iou_threshold(self) -> float:
        return self._iou_threshold

    @property
    def max_misses(self) -> int:
        return self._max_misses

    @property
    def min_hits(self) -> int:
        return self._min_hits

    @property
    def frames_seen(self) -> int:
        """How many `update` calls this tracker has processed. Never a clock value."""
        return self._frames

    def update(self, detections: Sequence[Any], now: float) -> tuple[Track, ...]:
        """Fold one frame's detections in and return the tracks, live ones first.

        `detections` is any sequence of objects with `.class_id`, `.box` and
        `.score` - `app.inference.decode.Detection` satisfies this, and so does a
        plain stand-in in a test. `now` is a parameter; this method reads no clock.

        Returns **every live track** - current ones first, then coasted ones, each
        group ascending by `track_id`. Coasted tracks are included deliberately: the
        event machine needs to know a track still exists (or has just died) to decide
        whether a pending transfer survives. A caller that draws from the returned
        sequence without filtering on `is_current` will draw a coasted box, which is
        why the flag is on the record rather than implied by membership.

        Raises on a non-finite `now` rather than storing it: a `nan` timestamp makes
        every later dwell and transition-interval comparison `False`, which would
        silently stop every event rather than crash.
        """
        if isinstance(now, bool) or not isinstance(now, (int, float)):
            raise ValueError(f"now must be a real number of seconds; got {now!r}")
        moment = float(now)
        if moment != moment or moment in (float("inf"), float("-inf")):
            raise ValueError(
                f"now must be finite; got {now!r}. A non-finite timestamp compares "
                f"False against everything, so every dwell and transition check would "
                f"silently fail and the machine would emit nothing."
            )
        if isinstance(detections, (str, bytes)):
            raise ValueError(
                f"detections must be a sequence of detections; got {detections!r}"
            )
        items = list(detections)

        self._frames += 1
        # Snapshot before mutating: a matched track's box is replaced, and a track
        # matched twice would otherwise be scored against its own updated position.
        candidates = list(self._tracks.values())
        pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(candidates):
            for det_index, detection in enumerate(items):
                if int(detection.class_id) != track.class_id:
                    # Review Focus #4. Not a threshold, not a tie-break: a person and
                    # a table are different things even when the boxes overlap almost
                    # completely, which is what a person at a counter looks like.
                    continue
                overlap = iou(track.box, tuple(float(v) for v in detection.box))
                if overlap >= self._iou_threshold:
                    pairs.append((overlap, track_index, det_index))

        # Highest overlap first; ties on the lower track id, so the assignment does
        # not depend on the order the decoder emitted detections in.
        pairs.sort(key=lambda p: (-p[0], candidates[p[1]].track_id, p[2]))
        matched_tracks: dict[int, Any] = {}
        matched_detections: dict[int, Any] = {}
        for _overlap, track_index, det_index in pairs:
            if track_index in matched_tracks or det_index in matched_detections:
                continue
            matched_tracks[track_index] = items[det_index]
            matched_detections[det_index] = candidates[track_index]

        updated: dict[int, Track] = {}
        for track_index, track in enumerate(candidates):
            detection = matched_tracks.get(track_index)
            if detection is None:
                misses = track.misses + 1
                if misses >= self._max_misses:
                    # Dropped on the `max_misses`-th consecutive miss, so
                    # `max_misses=2` means "lost after 2 absent frames" and survives
                    # one. The off-by-one matters: `>` would drop on the 3rd and a
                    # track would coast for one frame longer than the configured
                    # tolerance, which is a silently different tracker.
                    #
                    # The id is **not** returned to the pool. See `reset`.
                    continue
                updated[track.track_id] = replace(
                    track, misses=misses, confirmed=track.confirmed and misses == 0
                )
                continue
            hits = track.hits + 1
            updated[track.track_id] = replace(
                track,
                box=tuple(float(v) for v in detection.box),
                last_seen=moment,
                hits=hits,
                misses=0,
                confirmed=track.confirmed or hits >= self._min_hits,
            )

        seen = set(matched_detections)
        for det_index, detection in enumerate(items):
            if det_index in seen:
                continue
            track_id = self._next_id
            self._next_id += 1
            updated[track_id] = Track(
                track_id=track_id,
                class_id=int(detection.class_id),
                box=tuple(float(v) for v in detection.box),
                first_seen=moment,
                last_seen=moment,
                hits=1,
                misses=0,
                confirmed=self._min_hits <= 1,
            )

        self._tracks = updated
        # Current tracks first, then coasted ones; ascending `track_id` within each
        # group. Putting the matches first is what makes `update(...)[0]` mean "the
        # thing this frame is actually about" for the common single-object case, and
        # it is a total order, so the same detections in a different arrival order
        # give the same sequence.
        return tuple(sorted(updated.values(), key=lambda t: (t.misses > 0, t.track_id)))

    def active_track_ids(self) -> tuple[int, ...]:
        """Ids of every live track, ascending. Includes coasted and unconfirmed ones.

        "Live" means not yet dropped for exceeding `max_misses`. It deliberately does
        **not** mean "confirmed": the event machine needs to know a tentative track
        exists so it can build up dwell on it, and a tracker that hid unconfirmed
        tracks would make that impossible.
        """
        return tuple(sorted(self._tracks))

    def tracks(self) -> tuple[Track, ...]:
        """Every live track, current first then coasted, ascending by `track_id`."""
        return tuple(self._tracks[track_id] for track_id in sorted(self._tracks))

    def current_tracks(self) -> tuple[Track, ...]:
        """Only the tracks matched by the most recent `update`, ascending by `track_id`.

        **This is the accessor an overlay or an event machine should use.** The
        failure it avoids is silent and expensive: a coasted track's `box` is the last
        one *seen*, so drawing from `update()`'s return value without filtering draws
        a plausible, correctly-placed rectangle from a recent frame on top of an
        object that has since left. Nothing about that rectangle looks wrong.

        `update()` still returns coasted tracks on purpose - the event machine needs
        to know a track is coasting so it can decide whether a pending transfer
        survives an occlusion - which is exactly why the filtering has to be a named,
        tested call rather than something every consumer remembers to do.
        """
        return tuple(track for track in self.tracks() if track.is_current)

    def get(self, track_id: int) -> Track | None:
        """The live track with this id, or `None` if there is none.

        `None` is a real answer, not a default: it means this tracker has no such
        track, which is what `on_track_lost` in the event machine is told when a track
        dies and its pending state must be discarded.
        """
        return self._tracks.get(track_id)

    def reset(self) -> None:
        """Drop every track **and** advance past every id ever issued.

        The counter is deliberately not rewound to 1. Spec §9 requires an explicit
        reset at each source loop; if ids restarted, the first track of loop 1 would
        carry the id the last track of loop 0 had, and any pending transfer in the
        event machine would be completed by a different object that happens to look
        like it. `frames_seen` is cleared too, so a post-reset frame count describes
        the new loop only.
        """
        self._tracks = {}
        self._frames = 0
        # `_next_id` keeps counting. It is not reset to 1, on purpose.