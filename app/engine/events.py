"""`app/engine/events.py` - the retail event state machine. The product's claim.

Spec §4: `visible_on_shelf -> exited_shelf -> entered_basket -> candidate_transfer`,
with the same stable track, a minimum dwell, a bounded transition interval and a
dedup key. A reverse movement is a **separate** return event. Occlusion or a lost
track identity means **uncertain or not counted, never a completed transfer**. Gaps
are never filled with assumed detections.

This is where Review Focus #2 and #3 are answered, so most of the design is about
what the machine must *refuse* to do:

- **A lost track cancels; it never completes.** `on_track_lost` discards that track's
  pending state entirely. Re-observing the same `track_id` afterwards starts from
  nothing, because after an occlusion we cannot know it is the same object - the
  tracker's ids are never reused, so a re-observed id means a caller replayed, not a
  new object, and treating it as a continuation would be a guess. The cancellation
  is **counted** (`abandoned_transfers`, `abandoned_reasons`) so a booth can say "one
  transfer was dropped by an occlusion" instead of showing a smaller number with no
  explanation - a silently short count is the dishonest version of the same fact.
- **Dwell is measured between two real observations** - `last_seen_in_shelf` minus
  `first_seen_in_shelf`. Never interpolated, never carried across a gap, never
  estimated from `source_pts`. A dwell that nobody observed is not a dwell.
- **The transition interval is bounded from the last origin-zone observation**, so a
  track seen in the origin at t=0 and in the destination at t=99 does not transfer
  even though it satisfied dwell.
- **A return needs a transfer behind it.** `candidate_return` means "back where it
  came from"; without a counted transfer in this loop there is nothing to return
  from, so an origin->destination->origin jiggle inside one dwell window emits
  nothing at all.
- **An unconfirmed track is never counted, and refusing it does not poison its id.**
  A track the tracker has not confirmed is not evidence, but it is also not a
  rejected candidate - there is nothing to un-do, so a track that reaches
  confirmation later still transfers.

**The machine is pure.** Time is `frame.captured_at`; no clock is read, no file is
opened, nothing is written. Class names come from an injected label tuple, defaulting
to the pinned model's via `app.inference.pinned`, and a `class_id` outside that tuple
**raises** rather than being reported as `"unknown"` - a fabricated class name is a
fabricated measurement, and it would render on the booth as a label the model never
produced.

**Which zones the footage actually supports.** Read
`docs/qualification/M0-content-gate.md` §3-§4 before reading this module as a claim
about `retail_checkout`:

| slot | in `retail_checkout`? | what it is |
|---|---|---|
| `basket` | **yes** | the black shopping basket, bottom-left, already full at frame 0 |
| `checkout` | **yes** | the recessed scanner plate / scale, centre of frame |
| `shelf` | **no** | there is no shelf, shelving or shelf stock anywhere in the clip |

The observed movement is **basket -> scanner plate**, several times, in the dense
pass at frames 1500-1560, 1740-1800, 2000, 2200, 2400 and 2600. So the shipped
retail configuration is `basket` -> `checkout`, and the committed deliverable is
**"Items entering the checkout zone"** (reduced scope). The names `"shelf"` and
`"basket"` used in `tests/test_events.py` are this machine's **semantic test
fixtures** - generic slots that exercise the geometry without asserting the footage
contains them. The machine has no hardcoded slot names: both come from
`manifest.event_rules`, which is why `test_a_basket_to_checkout_move_is_a_candidate_transfer`
can drive the configuration that will actually ship. Shelf-to-basket remains **unmet**
and must not be reported as closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from app.contracts.frame import FramePacket
from app.contracts.scenario_manifest import ScenarioManifest
from app.engine.tracker import Track
from app.engine.zones import Zone

__all__ = ["BusinessEvent", "EventRule", "RetailEventMachine"]

#: Keys in `event_rules` that name which ROI is the origin and which the
#: destination. Present so a pack whose footage has no `shelf` can say so in data
#: instead of the code carrying a hardcoded slot that the footage does not have.
ORIGIN_SLOT_KEY = "origin_slot"
DESTINATION_SLOT_KEY = "destination_slot"

#: The only kinds this machine emits. `uncertain` is deliberately absent as a kind:
#: spec §4 offers "uncertain **or** not counted", and this milestone takes *not
#: counted* - an event whose evidence is incomplete is not emitted at all. The
#: `uncertain` field on `BusinessEvent` stays so the contract has room for a later
#: milestone to mark rather than drop, and every event emitted here is `False`.
EVENT_KINDS: frozenset[str] = frozenset({"candidate_transfer", "candidate_return"})

#: Default slot names in `event_rules` for the origin and destination of a transfer.
#: These are **defaults, not requirements**: `event_rules` may carry
#: `origin_slot`/`destination_slot` naming any two ROI entries, which is how the
#: retail pack describes `basket` -> `checkout` for footage that contains no shelf.
#: The machine holds no opinion about what a zone is called.
DEFAULT_ORIGIN_SLOT = "shelf"
DEFAULT_DESTINATION_SLOT = "basket"


@dataclass(frozen=True)
class EventRule:
    """The thresholds and classes a transfer is judged against, from the manifest.

    `min_dwell_seconds` and `max_transition_seconds` are **requirements**, not
    tunables chosen for a good-looking count. Both are validated non-negative with a
    strictly positive transition interval, because a zero transition interval would
    mean "only a track that teleports transfers", which is a rule that can never fire
    and would read as a shop where nothing happens.
    """

    name: str
    min_dwell_seconds: float
    max_transition_seconds: float
    classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"an event rule needs a non-empty name; got {self.name!r}")
        for field_name, value in (("min_dwell_seconds", self.min_dwell_seconds),
                                  ("max_transition_seconds", self.max_transition_seconds)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a real number; got {value!r}")
            number = float(value)
            if number != number or number in (float("inf"), float("-inf")) or number < 0.0:
                raise ValueError(
                    f"{field_name} must be a finite, non-negative number of seconds; "
                    f"got {value!r}"
                )
            object.__setattr__(self, field_name, number)
        if self.max_transition_seconds <= 0.0:
            raise ValueError(
                f"max_transition_seconds must be > 0; got {self.max_transition_seconds}. "
                f"Zero would mean only a teleporting track transfers, which is a rule "
                f"that can never fire and reads as a shop where nothing happens."
            )
        classes = tuple(self.classes)
        if not classes:
            raise ValueError(
                "classes must not be empty; a rule that tracks nothing can never emit "
                "an event, and would look like a detector that finds nothing"
            )
        for entry in classes:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(f"classes entries must be non-empty strings; got {entry!r}")
        object.__setattr__(self, "classes", classes)


@dataclass(frozen=True)
class BusinessEvent:
    """One counted business event, on one frame of one loop.

    `dedup_key` is `f"{source_loop_id}:{track_id}:{class_name}:{kind_suffix}"`. It is
    the whole of Review Focus #3: stable across replays of the same frames, distinct
    for two different objects in the same loop, and distinct again after a loop, so a
    consumer that has already shown a key can skip it without guessing.

    `at_frame_id` and `source_loop_id` are carried so a caller can check the event
    against the frame on screen. An event whose loop is not the one being displayed
    is from a previous pass of the clip and must not be drawn over the current frame.

    `uncertain` is always `False` for anything this machine emits - see the module
    note. It is on the record so that a future milestone can mark rather than drop.
    """
    kind: str
    class_name: str
    track_id: int
    source_loop_id: int
    at_frame_id: int
    at_time: float
    dedup_key: str
    uncertain: bool = False

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(
                f"event kind must be one of {sorted(EVENT_KINDS)}; got {self.kind!r}. An "
                f"unrecognised kind is a bug, and rendering it would put a word on the "
                f"screen that no rule in this module can produce."
            )
        for name, value in (("class_name", self.class_name),):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{name} must be a non-empty string; got {value!r}. An unnamed class "
                    f"on a retail event is a fabricated label."
                )
        for name, value in (("track_id", self.track_id),
                            ("source_loop_id", self.source_loop_id),
                            ("at_frame_id", self.at_frame_id)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int; got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0; got {value}")
        if self.track_id < 1:
            raise ValueError(
                f"track_id must be >= 1; got {self.track_id}. Ids start at 1 so 0 can "
                f"never stand in for an unknown track, and a dedup_key containing it "
                f"would still look like a real key."
            )
        if not isinstance(self.dedup_key, str) or not self.dedup_key.strip():
            raise ValueError(
                f"dedup_key must be a non-empty string; got {self.dedup_key!r}. It is "
                f"the mechanism that stops the same transfer being counted twice."
            )
        if isinstance(self.at_time, bool) or not isinstance(self.at_time, (int, float)):
            raise ValueError(f"at_time must be a real number; got {self.at_time!r}")
        if float(self.at_time) != float(self.at_time):
            raise ValueError(f"at_time must be finite; got {self.at_time!r} (nan)")
        if not isinstance(self.uncertain, bool):
            raise ValueError(
                f"uncertain must be a bool; got {self.uncertain!r}"
            )


@dataclass
class _Progress:
    """Per-track state. Mutable, and discarded wholesale on any interruption."""

    track_id: int
    class_id: int
    #: First and most recent observation **inside the origin zone**, both from real
    #: frames. Dwell is the difference; nothing else may set them. `first_` is set
    #: once per visit and never overwritten, or dwell would restart every frame and
    #: no track could ever satisfy a minimum.
    first_seen_in_origin: float | None = None
    last_seen_in_origin: float | None = None
    #: Set on the first observation **outside** the origin after having been in it.
    #: The transition interval is measured from `last_seen_in_origin` - the last frame
    #: that positively placed the object in the origin - so a track that was invisible
    #: for a long stretch cannot transfer by being briefly seen in the destination
    #: afterwards. An observed exit time would have been 99.0 in that case and would
    #: have produced a 0.0 s "transition" across 98 s of nothing.
    exited_origin: bool = False
    #: First and most recent observation inside the destination zone.
    first_seen_in_destination: float | None = None
    last_seen_in_destination: float | None = None
    #: Whether a transfer has already been counted for this track in this loop. Gates
    #: `candidate_return`; without it there is nothing to return from.
    transferred: bool = False
    returned: bool = False

    def has_origin_dwell(self, minimum: float) -> bool:
        if self.first_seen_in_origin is None or self.last_seen_in_origin is None:
            return False
        return (self.last_seen_in_origin - self.first_seen_in_origin) >= minimum

    def destination_dwell(self, minimum: float) -> bool:
        if (self.first_seen_in_destination is None
                or self.last_seen_in_destination is None):
            return False
        return (self.last_seen_in_destination - self.first_seen_in_destination) >= minimum


class RetailEventMachine:
    """The state machine. Pure: no clock, no OpenVINO, no I/O.

    `class_names` is the label list to map a track's `class_id` to a name. It
    defaults to the pinned model's via `app.inference.pinned`, which reads
    `labels.txt`; tests pass one explicitly so they do not need the model
    provisioned. A `class_id` past the end of the list **raises** - see the module
    note.
    """

    def __init__(self, manifest: ScenarioManifest,
                 class_names: tuple[str, ...] | None = None) -> None:
        if not isinstance(manifest, ScenarioManifest):
            raise ValueError(
                f"manifest must be a ScenarioManifest; got {type(manifest).__name__}"
            )
        rules = manifest.event_rules
        for key in ("min_dwell_seconds", "max_transition_seconds", "tracked_classes"):
            if key not in rules:
                raise ValueError(
                    f"event_rules has no {key!r}; keys are {sorted(rules)}. Without it "
                    f"the machine would have to invent a threshold, and an invented "
                    f"threshold decides what counts as a business event."
                )
        self._rule = EventRule(
            name="retail_transfer",
            min_dwell_seconds=rules["min_dwell_seconds"],
            max_transition_seconds=rules["max_transition_seconds"],
            classes=tuple(rules["tracked_classes"]),
        )
        allowlist = set(manifest.class_allowlist)
        undeclared = sorted(set(self._rule.classes) - allowlist)
        if undeclared:
            raise ValueError(
                f"tracked_classes {undeclared} are not in class_allowlist "
                f"{sorted(allowlist)}. A class the pack says it does not show cannot "
                f"be tracked; accepting it would put a class name on the screen that "
                f"the scenario never claimed to detect."
            )
        self._zones, self._origin_name, self._destination_name = self._build_zones(rules)
        self._class_names = tuple(class_names) if class_names is not None else None
        self._manifest = manifest
        self._progress: dict[int, _Progress] = {}
        self._events: list[BusinessEvent] = []
        self._emitted: set[str] = set()
        self._loop_id: int | None = None
        self._abandoned = 0
        self._abandoned_reasons: dict[str, int] = {}

    @staticmethod
    def _build_zones(rules: Mapping[str, Any]) -> tuple[dict[str, Zone], str, str]:
        """Build the `Zone`s named by the rule slots; return them and the route.

        The slot names come from `event_rules` (`origin_slot`/`destination_slot`,
        defaulting to `shelf`/`basket`), so a pack whose footage has no shelf declares
        `basket`/`checkout` and this function builds those. **The machine holds no
        opinion about what a zone is called** - the only two names it will invent are
        the defaults above, and a pack that wants different ones says so in its own
        config.

        Every ROI is validated by `Zone` here, at load, with the slot's name in the
        message - the difference between a loud failure at startup and boxes in the
        wrong place for a whole session.
        """
        named_keys = {ORIGIN_SLOT_KEY, DESTINATION_SLOT_KEY}
        roi_slots = {
            key: value for key, value in rules.items()
            if key not in named_keys
            and isinstance(value, (tuple, list)) and len(value) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
        }
        origin_name = rules.get(ORIGIN_SLOT_KEY, DEFAULT_ORIGIN_SLOT)
        destination_name = rules.get(DESTINATION_SLOT_KEY, DEFAULT_DESTINATION_SLOT)
        for role, slot in ((ORIGIN_SLOT_KEY, origin_name),
                           (DESTINATION_SLOT_KEY, destination_name)):
            if not isinstance(slot, str) or not slot.strip():
                raise ValueError(f"{role} must name an ROI entry; got {slot!r}")
            if slot not in roi_slots:
                raise ValueError(
                    f"{role} is {slot!r}, which is not an ROI in event_rules; "
                    f"ROI-shaped entries are {sorted(roi_slots)}. A transfer needs an "
                    f"origin and a destination, and a missing one is a config error "
                    f"rather than an event that cannot happen."
                )
        if origin_name == destination_name:
            raise ValueError(
                f"origin_slot and destination_slot are both {origin_name!r}. A transfer "
                f"from a zone to itself has no direction and would fire on any "
                f"dwell in that zone."
            )
        zones = {slot: Zone(name=slot, kind="area", roi=tuple(roi))
                 for slot, roi in roi_slots.items()}
        return zones, origin_name, destination_name

    # -- read-only views ------------------------------------------------------

    @property
    def manifest(self) -> ScenarioManifest:
        return self._manifest

    @property
    def rule(self) -> EventRule:
        """The thresholds and classes this machine judges against."""
        return self._rule

    @property
    def zones(self) -> Mapping[str, Zone]:
        """The route slots, by name. Read-only."""
        return MappingProxyType(self._zones)

    @property
    def origin(self) -> Zone:
        """The zone an item departs from. Named by `event_rules`, not by this module."""
        return self._zones[self._origin_name]

    @property
    def destination(self) -> Zone:
        """The zone an item arrives in. Named by `event_rules`."""
        return self._zones[self._destination_name]

    @property
    def origin_name(self) -> str:
        return self._origin_name

    @property
    def destination_name(self) -> str:
        return self._destination_name

    @property
    def events(self) -> list[BusinessEvent]:
        """Events counted since the last `reset`. A copy, so callers cannot edit it."""
        return list(self._events)

    @property
    def abandoned_transfers(self) -> int:
        """Transfers **not** counted because their evidence was interrupted.

        Zero is a real measurement here - the machine did decide - and it is only
        meaningful next to `events`, which is why both are exposed. A count of zero
        never means "we did not look".
        """
        return self._abandoned

    @property
    def abandoned_reasons(self) -> Mapping[str, int]:
        """Why each abandoned transfer was dropped, by reason. Read-only."""
        return MappingProxyType(dict(self._abandoned_reasons))

    @property
    def loop_id(self) -> int | None:
        """The source loop this machine is currently following, or `None` before the
        first observation. Explicit: "not yet started" is not `0`."""
        return self._loop_id

    def state_of(self, track_id: int) -> str:
        """A one-word description of a track's progress, for diagnostics and tests.

        One of `unseen`, `in_origin`, `left_origin`, `transferred`, `returned`. It
        exists so the state machine's position can be inspected without inferring it
        from a list of events - which is how a machine that is stuck becomes
        indistinguishable from one that is working.
        """
        progress = self._progress.get(track_id)
        if progress is None:
            return "unseen"
        if progress.returned:
            return "returned"
        if progress.transferred:
            return "transferred"
        if progress.first_seen_in_origin is not None and not progress.exited_origin:
            return "in_origin"
        if progress.exited_origin:
            return "left_origin"
        return "unseen"

    # -- observation ----------------------------------------------------------

    def observe(self, track: Track, frame: FramePacket) -> tuple[BusinessEvent, ...]:
        """Fold one track observation in; return the events it produced (0 or 1).

        Order of the checks is the design, and each refusal is deliberate:

        1. **Wrong loop, or unstarted.** A frame from a loop the machine was not told
           about raises. Adopting it silently would let one loop's pending state leak
           into the next (Review Focus #3, other direction). Spec §9 requires an
           explicit `reset` per loop.
        2. **Class not tracked.** A person walking the route produces nothing, and no
           state is kept - there is no pending transfer to poison.
        3. **Not confirmed.** Not evidence. No state kept, nothing counted.
        4. **Track not current.** A coasted track's box is the last one *seen*; using
           it now would place the object somewhere nobody observed it.
        5. **Unmapped class id.** Raises rather than becoming `"unknown"`.

        `now` is `frame.captured_at`. This method reads no clock.
        """
        if not isinstance(track, Track):
            raise ValueError(f"track must be a Track; got {type(track).__name__}")
        if not isinstance(frame, FramePacket):
            raise ValueError(f"frame must be a FramePacket; got {type(frame).__name__}")

        if self._loop_id is None:
            self._loop_id = frame.source_loop_id
        elif frame.source_loop_id != self._loop_id:
            raise ValueError(
                f"frame is from source loop {frame.source_loop_id} but this machine is "
                f"following loop {self._loop_id}. Call reset(loop_id=...) at each source "
                f"loop (spec section 9); adopting the new loop silently would carry one "
                f"loop's pending transfers into the next."
            )

        class_name = self._class_name_for(track.class_id)
        now = frame.captured_at

        if class_name not in self._rule.classes:
            return ()
        if not track.confirmed:
            return ()
        if not track.is_current:
            return ()

        normalised = _normalise(track.box, frame)
        in_origin = self.origin.contains(normalised)
        in_destination = self.destination.contains(normalised)

        progress = self._progress.get(track.track_id)
        if progress is None or progress.class_id != track.class_id:
            progress = _Progress(track_id=track.track_id, class_id=track.class_id)
            self._progress[track.track_id] = progress

        if in_origin:
            # Re-entering the origin. If a transfer had been counted for this track,
            # that is a **return** - and it is emitted here, before the episode is
            # cleared, because clearing first would forget there was anything to
            # return from.
            emitted: tuple[BusinessEvent, ...] = ()
            if progress.transferred and not progress.returned:
                emitted = self._maybe_emit_return(progress, class_name, frame, now)

            # A genuinely new visit starts the dwell clock again. `first_` is only
            # carried over while the track has stayed continuously in the origin:
            # overwriting it every frame - the obvious way to write this - restarts
            # the clock on every frame and no track can ever satisfy a minimum dwell.
            continuous = (progress.first_seen_in_origin is not None
                          and not progress.exited_origin)
            if not continuous:
                progress.first_seen_in_origin = now
                progress.first_seen_in_destination = None
                progress.last_seen_in_destination = None
                progress.transferred = False
                progress.returned = False
            progress.last_seen_in_origin = now
            progress.exited_origin = False
            return emitted

        if progress.first_seen_in_origin is not None and not progress.exited_origin:
            # First observation outside the origin, having been in it.
            progress.exited_origin = True

        if in_destination:
            if progress.first_seen_in_destination is None:
                progress.first_seen_in_destination = now
            progress.last_seen_in_destination = now

        return self._maybe_emit_transfer(progress, class_name, frame, now, in_destination)

    def _maybe_emit_transfer(self, progress: _Progress, class_name: str,
                             frame: FramePacket, now: float,
                             in_destination: bool) -> tuple[BusinessEvent, ...]:
        """The forward leg: origin -> destination.

        Requires, all of it measured from real frames: the track was observed in the
        origin, dwell in the origin reached `min_dwell_seconds`, it is being observed
        in the destination **now**, it is confirmed and current (checked by the
        caller), and the interval between the last origin observation and this one is
        within `max_transition_seconds`.
        """
        rule = self._rule
        if not in_destination or progress.transferred or progress.returned:
            return ()
        if not progress.exited_origin or progress.last_seen_in_origin is None:
            return ()
        if not progress.has_origin_dwell(rule.min_dwell_seconds):
            return ()
        if (now - progress.last_seen_in_origin) > rule.max_transition_seconds:
            self._abandon("transition_too_slow", progress.track_id)
            return ()
        progress.transferred = True
        return self._emit("candidate_transfer", progress, class_name, frame, now)

    def _maybe_emit_return(self, progress: _Progress, class_name: str,
                           frame: FramePacket, now: float) -> tuple[BusinessEvent, ...]:
        """The backward leg: destination -> origin, and only after a counted transfer.

        Without a transfer behind it there is nothing to return from, so an
        origin->destination->origin jiggle inside one dwell window emits nothing -
        which is the difference between "the shopper put it back" and "the tracker
        jitters".
        """
        rule = self._rule
        if not progress.destination_dwell(rule.min_dwell_seconds):
            return ()
        if progress.last_seen_in_destination is None:
            return ()
        if (now - progress.last_seen_in_destination) > rule.max_transition_seconds:
            self._abandon("return_too_slow", progress.track_id)
            return ()
        progress.returned = True
        return self._emit("candidate_return", progress, class_name, frame, now)

    def _emit(self, kind: str, progress: _Progress, class_name: str,
              frame: FramePacket, now: float) -> tuple[BusinessEvent, ...]:
        suffix = "transfer" if kind == "candidate_transfer" else "return"
        key = f"{frame.source_loop_id}:{progress.track_id}:{class_name}:{suffix}"
        if key in self._emitted:
            # Replayed frames of an already-counted transfer. Skipping is the whole
            # point of the key: the same movement, seen again, is not a second event.
            return ()
        event = BusinessEvent(
            kind=kind,
            class_name=class_name,
            track_id=progress.track_id,
            source_loop_id=frame.source_loop_id,
            at_frame_id=frame.frame_id,
            at_time=now,
            dedup_key=key,
            uncertain=False,
        )
        self._emitted.add(key)
        self._events.append(event)
        return (event,)

    def _abandon(self, reason: str, track_id: int) -> None:
        self._abandoned += 1
        self._abandoned_reasons[reason] = self._abandoned_reasons.get(reason, 0) + 1
        # The episode is dead either way: a track that has been away too long is not
        # going to become a countable transfer by waiting longer, and keeping the
        # pending state would let a much later frame complete it.
        progress = self._progress.get(track_id)
        if progress is not None:
            progress.exited_origin = False
            progress.first_seen_in_origin = None
            progress.last_seen_in_origin = None
            progress.first_seen_in_destination = None
            progress.last_seen_in_destination = None
            progress.transferred = False

    # -- interruption and loop boundaries -------------------------------------

    def on_track_lost(self, track_id: int, now: float) -> None:
        """Discard a track's pending state. **It never completes a transfer.**

        Review Focus #2. `now` is the time the loss was observed and is recorded so
        the caller can report when; nothing is inferred from it. If the track had a
        pending, dwell-satisfied origin episode, that is counted as abandoned -
        visibly, because a shorter count with no explanation is the dishonest version
        of the same fact.

        Idempotent: losing the same id twice abandons nothing the second time, because
        its state is already gone.
        """
        if isinstance(track_id, bool) or not isinstance(track_id, int):
            raise ValueError(f"track_id must be an int; got {track_id!r}")
        if track_id < 1:
            raise ValueError(f"track_id must be >= 1; got {track_id}")
        if isinstance(now, bool) or not isinstance(now, (int, float)):
            raise ValueError(f"now must be a real number; got {now!r}")
        progress = self._progress.pop(track_id, None)
        if progress is None:
            return ()
        had_pending = (
            progress.first_seen_in_origin is not None
            and progress.last_seen_in_origin is not None
            and progress.has_origin_dwell(self._rule.min_dwell_seconds)
            and not progress.transferred
        )
        if had_pending:
            self._abandoned += 1
            self._abandoned_reasons["track_lost"] = (
                self._abandoned_reasons.get("track_lost", 0) + 1
            )

    def reset(self, loop_id: int) -> None:
        """Start a new source loop: clear all state, the event log and the dedup set.

        `loop_id` is the loop now being followed. It is required, not defaulted to 0,
        because a reset that did not name the loop could not be checked against the
        frames that arrive afterwards - and "the machine thinks it is on loop 1 while
        the source is on loop 0" is exactly the leak `reset` exists to prevent.

        Track ids are **not** cleared from the tracker's counter here; that is the
        tracker's job (`Tracker.reset`) and it deliberately does not rewind either.
        """
        if isinstance(loop_id, bool) or not isinstance(loop_id, int) or loop_id < 0:
            raise ValueError(
                f"loop_id must be a non-negative int; got {loop_id!r}. It names the loop "
                f"now being followed, and 'loop 0 or whatever' is not a value the frames "
                f"that follow can be checked against."
            )
        self._progress = {}
        self._events = []
        self._emitted = set()
        self._abandoned = 0
        self._abandoned_reasons = {}
        self._loop_id = loop_id

    # -- names ----------------------------------------------------------------

    def _class_name_for(self, class_id: int) -> str:
        """The label for `class_id`, or a refusal.

        Never `"unknown"`. An unmappable class id is a disagreement between the
        detector and the label list, and rendering it as a class name would put a
        word on the screen that the model never produced.
        """
        names = self._class_names
        if names is None:
            from app.inference.pinned import class_names as pinned_class_names
            names = pinned_class_names()
            self._class_names = names
        if not 0 <= class_id < len(names):
            raise ValueError(
                f"class_id {class_id} is outside the label list of {len(names)} classes "
                f"(the pinned model is {names and len(names)} wide). Refusing rather "
                f"than labelling it 'unknown': a class name the model did not produce "
                f"is a fabricated measurement on a booth screen."
            )
        return names[class_id]


def _normalise(box: tuple[float, float, float, float],
               frame: FramePacket) -> tuple[float, float, float, float]:
    """A track's source-pixel box as normalised coordinates for this frame.

    The normalisation is against **this frame's** extents, read from the packet and
    not from a config file - a box scaled by anything other than the frame it was
    measured on is a box in the wrong place, which is the one arithmetic mistake the
    overlay cannot make visible.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    width = float(frame.width)
    height = float(frame.height)
    return (x1 / width, y1 / height, x2 / width, y2 / height)