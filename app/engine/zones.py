"""`app.engine.zones` - normalised zone geometry, hit tests, dwell, line crossings.

**Why normalised, and why validated hard.** An ROI is `(x1, y1, x2, y2)` with all
four values in `[0, 1]`, relative to the frame's own width and height. The
alternative - pixel coordinates - hard-codes a resolution into operator-authored
JSON, and the one clip that varies (`store_aisle_alt` at 720x404 against
`retail_checkout` at 1920x1080) would then need a different config per clip.

**A bad ROI raises. That is the whole point of this module.** The dangerous
failure is not a crash, it is a zone that silently contains nothing: every hit test
returns false, every event count is zero, and the booth screen says "no items
moved", which is a false claim about a shop (AGENTS.md rules 1 and 2). So
`Zone.__post_init__` refuses out-of-range, inverted, degenerate, non-numeric and
`nan` ROIs, and says which one. Three details are load-bearing and each is pinned
by a test that would fail if it were removed:

- **`nan` is refused explicitly.** Every comparison against `nan` is `False`, so a
  range check written as `not 0.0 <= v <= 1.0` lets `nan` straight through. This is
  a real hole in the obvious implementation, not a hypothetical one.
- **`to_pixels` truncates, it does not round.** The brief's expected value for
  `(0.02, 0.22, 0.34, 0.55)` at 1920x1080 is `(38, 237, 652, 594)`, which is
  `int()`, not `round()` - `0.22 * 1080 = 237.6` and `round` would say 238. The
  truncation is also the correct choice: the far edge of a drawn zone should not be
  reported one pixel outside the ROI the operator wrote.
- **`contains` is full containment, not "centre is in the zone".** A straddling
  object is not in the zone. Counting it as present is how a shelf and a basket
  both claim one item and the counter doubles.

**On zone names.** `name` here is the state machine's **semantic slot**
(`"shelf"`, `"basket"`, `"belt"`, `"counter_band"`), and the generic names are
deliberate: one machine serves every scenario, and the *scenario manifest* is what
says which of these the footage actually contains.

Read `app/engine/retail_config.py` before assuming a retail clip has a shelf.
`retail_checkout` is a self-checkout with **no shelf anywhere in it**, and measured
over all 3387 frames **no product-transfer event is detectable in it at all** - the
pack that ships is people-in-zone presence over a `counter_band` zone.

An earlier revision of this note named a two-zone product route for this clip. That
claim was **withdrawn** when the rectangles were measured and found to hold a
stationary apple and two `suitcase` detections. It is deliberately described rather
than quoted here: `tests/test_retail_measurements.py` fails on any shipped source
containing the withdrawn wording, so a correction that quoted it would keep failing
its own check.

`"shelf"` is a **fixture name** used by the unit tests to exercise the geometry over
synthetic boxes. It is not a claim that any provisioned footage contains a shelf.

`dwell_seconds_in_zone` and `crossed_line` are free functions rather than methods
because they are predicates about a *pair* of observations - the zone is the same
object every frame, the box is not.

Pure: no numpy, no clock, no I/O. `now` is always a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Zone", "crossed_line", "dwell_seconds_in_zone"]

#: Kinds this module gives meaning to. `area` zones answer `contains`; `line` zones
#: answer `crossed_line`. A `"shelf"`/`"basket"` style name is a **name**, not a
#: kind - the brief's own fixture is `Zone("shelf", "shelf", ...)`, where the first
#: field is the slot and the second is its kind, and an area zone may legitimately
#: be described as `shelf`. Kinds outside this set are refused rather than stored,
#: because `kind` decides which predicate applies and an unknown one would have to
#: be guessed at.
ZONE_KINDS: frozenset[str] = frozenset({"area", "line", "shelf", "basket", "checkout"})

#: Every direction `crossed_line` will accept. Deliberately not `-1, 0, 1`: zero
#: must never mean "either way" (AGENTS.md rule 15 - `0` standing in for a missing
#: value), and spec §4 counts a return movement as a *separate* event.
LINE_DIRECTIONS: tuple[int, int] = (1, -1)


def _require_unit_coordinate(name: str, value: Any) -> float:
    """A normalised coordinate: a real number in `[0, 1]`, `nan` refused explicitly."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} must be a real number in [0, 1]; got {value!r} "
            f"({type(value).__name__})"
        )
    number = float(value)
    if number != number:  # NaN: every comparison is False, so range checks pass it
        raise ValueError(
            f"{name} must be a real number in [0, 1]; got {value!r} (nan). A NaN "
            f"coordinate would compare False against every bound and pass an "
            f"ordinary range check."
        )
    if not 0.0 <= number <= 1.0:
        raise ValueError(
            f"{name} must be within [0, 1]; got {number}. A ROI outside the frame is a "
            f"config error, and clamping it would move the zone somewhere the operator "
            f"did not draw it - silently, and wrongly."
        )
    return number


def _require_real(name: str, value: Any) -> float:
    """Any finite real number. Order and range are the caller's business."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} must be a real number; got {value!r} ({type(value).__name__})"
        )
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite real number; got {value!r}")
    return number


def _unpack(name: str, values: Any, arity: int) -> tuple[float, ...]:
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of {arity} numbers; got {values!r}") from exc
    if len(items) != arity:
        raise ValueError(f"{name} must have exactly {arity} values; got {items!r}")
    return tuple(_require_real(f"{name}[{i}]", v) for i, v in enumerate(items))


def _require_box(name: str, box: Any) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _unpack(name, box, 4)
    return (x1, y1, x2, y2)


def _require_point(name: str, point: Any) -> tuple[float, float]:
    x, y = _unpack(name, point, 2)
    return (x, y)


@dataclass(frozen=True)
class Zone:
    """One named region of a frame, in normalised coordinates.

    `name` is the semantic slot the event machine keys per-track state under.
    `kind` is one of `ZONE_KINDS` and decides which predicate answers questions
    about this zone: `contains` for a region, `crossed_line` for a band.

    Immutable, so a zone built once from a manifest cannot be edited between the
    event machine reading it and the drawer drawing it.
    """

    name: str
    kind: str
    roi: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                f"a zone needs a non-empty name; got {self.name!r}. The name is the key "
                f"per-track state is held under, so an unnamed zone is unreachable "
                f"rather than merely untidy."
            )
        if self.kind not in ZONE_KINDS:
            raise ValueError(
                f"zone {self.name!r} kind must be one of {sorted(ZONE_KINDS)}; got "
                f"{self.kind!r}. `kind` decides whether this zone is tested for "
                f"containment or crossed, so an unrecognised kind would have to be "
                f"guessed at - and a wrong guess invents events."
            )
        x1, y1, x2, y2 = _require_box(f"zone {self.name!r} roi", self.roi)
        for label, value in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
            _require_unit_coordinate(f"zone {self.name!r} roi {label}", value)
        if x1 >= x2:
            raise ValueError(
                f"zone {self.name!r} roi ({x1}, {y1}, {x2}, {y2}) is out of order: x1 "
                f"({x1}) must be < x2 ({x2}). An inverted ROI is a config typo, and "
                f"accepting it yields a zone that is empty - a silent always-zero "
                f"counter."
            )
        if y1 >= y2:
            raise ValueError(
                f"zone {self.name!r} roi ({x1}, {y1}, {x2}, {y2}) is out of order: y1 "
                f"({y1}) must be < y2 ({y2})."
            )
        object.__setattr__(self, "roi", (x1, y1, x2, y2))

    def contains(self, box: tuple[float, float, float, float]) -> bool:
        """Whether `box` is **entirely** inside this zone, edges inclusive.

        A zone contains its own ROI, which is what makes a box exactly on the
        operator's boundary count as inside rather than depending on a strict `<`.

        A box whose own edges are inverted returns `False` rather than raising: an
        inverted detection box is a fact about the upstream decode, and the answer
        to "is this thing in the zone?" when the thing has no interior is no.
        """
        x1, y1, x2, y2 = _require_box("box", box)
        if x1 > x2 or y1 > y2:
            return False
        zx1, zy1, zx2, zy2 = self.roi
        return x1 >= zx1 and y1 >= zy1 and x2 <= zx2 and y2 <= zy2

    def contains_point(self, point: tuple[float, float]) -> bool:
        """Whether a single `(x, y)` lies inside this zone, edges inclusive.

        This is the test a **presence** rule uses, and it is not interchangeable with
        `contains`. Measured on `retail_checkout`: a person's box is routinely clipped
        by the frame border (`x1 = -0.001`, `x2 = 0.999`), so **full containment can
        never hold for a person in this footage** - a presence rule written on
        `contains` reports that nobody is ever anywhere, which is a false statement
        about a shop. Where an object *is* cannot be decided by a box that runs off
        the edge of the picture; where its **centre** is can.

        A point exactly on an edge is inside, consistent with `contains`.
        """
        x, y = _require_point("point", point)
        zx1, zy1, zx2, zy2 = self.roi
        return zx1 <= x <= zx2 and zy1 <= y <= zy2

    def centroid_of(self, box: tuple[float, float, float, float]) -> tuple[float, float]:
        """The centre of `box`, in the same normalised coordinates.

        This is what a line-crossing test compares, not containment: a moving
        object is inside a band for a frame or two either side of the crossing, so
        a containment test would fire on the way in *and* on the way out.
        """
        x1, y1, x2, y2 = _require_box("box", box)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """The ROI as an integer `(x1, y1, x2, y2)` on a `width` x `height` frame.

        **Truncated, not rounded**, and clamped into the frame. The brief's measured
        expectation for `(0.02, 0.22, 0.34, 0.55)` at 1920x1080 is
        `(38, 237, 652, 594)`: `0.22 * 1080 = 237.6` and `0.34 * 1920 = 652.8`,
        both of which `round` would take up to 238 and 653. Truncation keeps the
        drawn rectangle inside the ROI the operator wrote.

        `width`/`height` must be positive ints. A zero extent would divide by zero
        inside a draw call, and clamping it to `(0, 0, 0, 0)` would draw a zone that
        is not there.
        """
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int; got {value!r}")
            if value < 1:
                raise ValueError(
                    f"{name} must be >= 1; got {value}. A frame with no pixels in it "
                    f"cannot carry a zone."
                )
        zx1, zy1, zx2, zy2 = self.roi
        return (
            min(width - 1, max(0, int(zx1 * width))),
            min(height - 1, max(0, int(zy1 * height))),
            min(width - 1, max(0, int(zx2 * width))),
            min(height - 1, max(0, int(zy2 * height))),
        )


def dwell_seconds_in_zone(zone: Zone, box: tuple[float, float, float, float],
                          box_at: float, now: float) -> float:
    """Seconds `box` has been in `zone`, given it was seen there at `box_at`.

    `now - box_at` when `box` is inside the zone right now; `0.0` otherwise. Two
    deliberate choices:

    - **Zero, not the running total, once the box leaves.** The caller accumulates
      across frames; the moment the box is outside, this contributes nothing.
      Returning the old total would keep growing for an object that has gone.
    - **Clamped at zero**, so a `now` earlier than `box_at` reads as zero. A negative
      duration is not a measurement.

    `now` is a parameter. This function reads no clock.
    """
    if not isinstance(zone, Zone):
        raise ValueError(f"zone must be a Zone; got {type(zone).__name__}")
    if not zone.contains(box):
        return 0.0
    return max(0.0, _require_real("now", now) - _require_real("box_at", box_at))


def crossed_line(zone: Zone, previous: tuple[float, float], current: tuple[float, float],
                 direction: int) -> bool:
    """Whether a centroid moved across `zone`'s band in `direction`.

    A `"line"` zone's ROI is a **vertical segment**: `x1`..`x2` is the band,
    `y1`..`y2` is the extent along which a crossing counts, and the band **centre** is
    the line. Both endpoints of the movement must lie within `y1`..`y2`, so a
    horizontal movement at the far bottom of the frame does not count as crossing a
    line drawn near the top.

    That `y` test is deliberately **strict at the boundary**: a movement whose start
    is outside `y1`..`y2` does not fire, even if it ends inside. The alternative -
    intersecting the movement segment with the band - would fire for an object
    arriving from below the line as well as one crossing it, which is the same
    one-event-two-reasons problem `crossed_line` exists to avoid. The looser rule is
    future work and would need its own measured test.

    A crossing is a change of side:

        crossed_line(BELT, (0.30, 0.40), (0.80, 0.42), direction=1)   -> True
        crossed_line(BELT, (0.30, 0.40), (0.80, 0.42), direction=-1)  -> False
        crossed_line(BELT, (0.40, 0.40), (0.50, 0.42), direction=1)   -> False

    The third case is the one worth stating: both endpoints are inside the band, so
    nothing was crossed. A test that only checked the first two would pass an
    implementation that returned `direction == 1` unconditionally - which is
    exactly the defect `test_the_reverse_crossing_is_not_the_same_event` cannot
    see, since it asserts a `False` that a constant `True` fails but a constant
    `False` would also pass. `test_a_backwards_movement_is_detected_when_asked_for`
    is the positive case in the other direction and it does catch it.

    `direction` must be `1` or `-1`; `0` raises. Spec §4 requires a return movement
    to be a distinct event, so "either way" is not a value this function may accept.
    """
    if not isinstance(zone, Zone):
        raise ValueError(f"zone must be a Zone; got {type(zone).__name__}")
    if zone.kind != "line":
        raise ValueError(
            f"crossed_line applies to a zone of kind 'line'; zone {zone.name!r} is "
            f"{zone.kind!r}. A movement across an area zone is a change of containment, "
            f"not a line crossing, and answering one with the other would invent events."
        )
    if isinstance(direction, bool) or not isinstance(direction, int) \
            or direction not in LINE_DIRECTIONS:
        raise ValueError(
            f"direction must be 1 (forwards) or -1 (backwards); got {direction!r}. It "
            f"is not optional: spec section 4 counts a return movement as a separate "
            f"event, so a direction of 0 meaning 'either way' would merge a transfer "
            f"and its own return into one event."
        )
    previous_x, previous_y = _require_point("previous", previous)
    current_x, current_y = _require_point("current", current)
    zx1, zy1, zx2, zy2 = zone.roi
    # The band is a segment, not an infinite line: both endpoints must lie within its
    # y extent. Reading `zy1`/`zy2` and discarding them - which is what this function
    # did before a review caught it - would let a movement at the bottom of the frame
    # cross a line drawn at the top.
    if not (zy1 <= previous_y <= zy2 and zy1 <= current_y <= zy2):
        return False
    centre = (zx1 + zx2) / 2.0
    was_left = previous_x < centre
    now_left = current_x < centre
    if was_left == now_left:
        return False
    moved_forwards = not now_left
    return moved_forwards if direction == 1 else not moved_forwards