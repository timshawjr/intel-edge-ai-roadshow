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
  is **counted** (`abandoned_episodes`, `abandoned_reasons`) so a booth can say "one
  event was dropped by an occlusion" instead of showing a smaller number with no
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

**The machine is pure.** Time is `frame.captured_at`; no clock is read and **no file is
opened after construction**. Class names are resolved and cross-checked against the pin
**once, in `__init__`**. That is a correction: an earlier revision imported
`app.inference.pinned` lazily on first observation, and `pinned.class_names()` does
`Path.read_text` on `labels.txt` - so the first frame of a live stream performed a
filesystem read, and the "no file is opened" claim above was false. A review caught it
by spying on `pathlib.Path.read_text`, which spying on `builtins.open` does not see.
`tests/test_events.py::Purity::test_observe_opens_no_file_after_construction` now pins
it. `observe()` reads a list of strings and a dict; that is all.

An unmappable `class_id` is **counted and skipped**, not raised: `class_id` comes from
the model, so an id past the end of the label list is a fact about the model, and
killing a live stream over it would be worse than reporting it
(`skipped_unmapped_class`). The name is never fabricated.

**Which zones the footage actually supports.** Read `app/engine/retail_config.py` -
it holds the measurements - and `docs/qualification/M0-content-gate.md` §3-§4, before
reading this module as a claim about `retail_checkout`:

| slot | in `retail_checkout`? | what it is |
|---|---|---|
| `basket` | yes, as an object | the black shopping basket, bottom-left, already full at frame 0 |
| `counter_band` | **yes, and this is the shipped zone** | the full-width band `y` 0.20..0.60 in which a person stands at the counter |
| `shelf` | **no** | there is no shelf, shelving or shelf stock anywhere in the clip |

**No product-transfer event is detectable on this footage. The shipped pack is
people-in-zone presence**, declared in `retail_config.py` as
`presence_zone`/`min_presence_seconds` with **no transfer route at all**. It is not
"items entering the checkout zone" - measured over all 3387 frames, that produces zero
events - and it is not shelf-to-basket. Both remain **unmet** and must not be reported
as closed.

An earlier revision of this note claimed the observed movement was "basket -> scanner
plate" and backed it with a test that drove a `basket` -> `checkout` transfer. That test
was deleted in the review fix and the claim with it: the rectangles it drove held a
stationary apple and two `suitcase` detections. The numbers are in
`retail_config.py`. The deleted test is deliberately not named here -
`tests/test_retail_measurements.py` fails on any shipped docstring that cites a test
which does not exist, and a correction that names the wrong identifier would trip it.

The transfer legs themselves - `candidate_transfer`, `candidate_return` - are real,
tested and stay: they are spec §4 behaviour and another scenario may have footage for
them. They simply have **no events on this clip**, which is a fact about the footage
and not about the code. `tests/test_events.py` exercises them with **semantic fixtures**
(`shelf`/`basket` slots over synthetic boxes), which say nothing about `retail_checkout`.
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

#: A presence rule: "anything tracked that stays in `presence_zone` for
#: `min_presence_seconds` is a `zone_occupancy` event". This is what the shipped
#: retail pack uses, because that is the only business statement the provisioned
#: footage supports - see the module note.
PRESENCE_ZONE_KEY = "presence_zone"
MIN_PRESENCE_KEY = "min_presence_seconds"

#: Sentinel for "this track has no presence record at all", distinct from a record
#: holding `None` ("this track's stay has been counted"). The two must not be
#: confused: conflating them would either abandon an already-counted stay or keep a
#: lost track's dwell alive.
_NEVER: object = object()


@dataclass
class _Stay:
    """One continuous presence episode: when it began, and whether it was counted.

    `counted` is the whole reason this is a record and not a bare float. Without it
    `on_track_lost` cannot distinguish a stay that had already produced a
    `zone_occupancy` event from one that had not, and has to guess - which made it
    report an abandonment for an event that had already been emitted.
    """

    entered_at: float
    counted: bool = False

#: The only kinds this machine emits. `uncertain` is deliberately absent as a kind:
#: spec §4 offers "uncertain **or** not counted", and this milestone takes *not
#: counted* - an event whose evidence is incomplete is not emitted at all. The
#: `uncertain` field on `BusinessEvent` stays so the contract has room for a later
#: milestone to mark rather than drop, and every event emitted here is `False`.
EVENT_KINDS: frozenset[str] = frozenset({
    "candidate_transfer", "candidate_return", "zone_occupancy",
})

#: A transfer route is declared by naming both slots. There are **no default zone
#: names in this module**: `shelf` and `basket` appeared here in an earlier revision
#: as implicit defaults, which meant a pack got a transfer route it never asked for
#: just by having ROIs called `shelf` and `basket`, and a pack whose footage has no
#: shelf had no way to say so without inventing a slot name. A route is now explicit or
#: absent.


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

    `dedup_key` is `f"{source_loop_id}:{track_id}:{class_name}:{kind_suffix}"`, where
    the suffix is `transfer`, `return`, or `presence:<zone_name>`. It is the whole of
    Review Focus #3: stable across replays of the same frames, distinct for two
    different objects in the same loop, and distinct again after a loop, so a consumer
    that has already shown a key can skip it without guessing.

    **One consequence, stated rather than left to be discovered:** the key carries no
    episode discriminator, so a track that leaves a zone and comes back **within the
    same loop** produces the same key and is counted **once**. That is deliberate and it
    is uniform across all three kinds; a future need to count repeated round trips
    requires a per-episode discriminator here and in `_emit_keyed`, not a change to any
    caller's expectations. `test_a_second_stay_in_the_same_loop_is_suppressed_by_the_dedup_key`
    pins it.

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


def _seconds(rules: Mapping[str, Any], key: str, *, required: bool = True) -> float | None:
    """A dwell/interval threshold from `event_rules`, validated.

    `required=False` returns `None` for a key that is absent, which is how a
    presence-only pack omits the transfer thresholds instead of having to invent
    them. An invented threshold decides what counts as a business event, so it is
    never invented.
    """
    if key not in rules:
        if required:
            raise ValueError(
                f"event_rules has no {key!r}; keys are {sorted(rules)}. Without it the "
                f"machine would have to invent a threshold, and an invented threshold "
                f"decides what counts as a business event."
            )
        return None
    value = rules[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a real number of seconds; got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")) or number < 0.0:
        raise ValueError(f"{key} must be a finite, non-negative number of seconds; got {value!r}")
    return number


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
    """The state machine. Pure: no clock, no OpenVINO, **no I/O**.

    `class_names` is the label list used to turn a track's `class_id` into a name, and
    it is **resolved and validated here, once, at construction**. It defaults to the
    pinned model's via `app.inference.pinned.class_names()`, which reads `labels.txt`
    - that read happens exactly once, during `__init__`, and never again. After
    construction `observe()` opens nothing. (It used to import `pinned` lazily on
    first use, which meant a filesystem read on the first frame of a live stream; a
    review caught it by spying on `Path.read_text`, which `builtins.open` spying does
    not see, and the "no file is opened" claim in this module's docstring was
    therefore false.)

    Every startup failure this module can have happens in `__init__`, where it is
    diagnosable: a missing `labels.txt`, a label list that disagrees with the pinned
    class count, an undeclared tracked class, a malformed ROI.

    Two rules the machine applies, each from a review finding:

    - **An unmappable `class_id` does not kill the stream.** `class_id` comes from the
      model, so a detector emitting an id past the end of the label list is a fact
      about the model, not a bug in the caller. The correct invariant is a *startup*
      one - `pinned.labels_match_pin()` - and it is checked in `__init__`. At
      observation time an unmappable id is an unusable observation: it yields no
      event and increments `skipped_unmapped_class`, so the screen can read "N
      skipped: class not in label list" instead of the stream dying repeatedly
      (AGENTS.md rule 12). It never becomes the fabricated name `"unknown"`.
    - **Only current tracks are events.** A coasted track's box is the last one seen;
      counting it as present would place an object where nobody observed it. Use
      `Tracker.current_tracks()` to feed this machine.
    """

    def __init__(self, manifest: ScenarioManifest,
                 class_names: tuple[str, ...] | None = None) -> None:
        if not isinstance(manifest, ScenarioManifest):
            raise ValueError(
                f"manifest must be a ScenarioManifest; got {type(manifest).__name__}"
            )
        rules = manifest.event_rules
        if "tracked_classes" not in rules:
            raise ValueError(
                f"event_rules has no 'tracked_classes'; keys are {sorted(rules)}. "
                f"Without it the machine would have to invent a class list, and an "
                f"invented list decides what counts as a business event."
            )
        self._tracked_classes = frozenset(rules["tracked_classes"])
        declares_route = ORIGIN_SLOT_KEY in rules or DESTINATION_SLOT_KEY in rules
        # The transfer thresholds belong to the transfer rule. A presence-only pack
        # declares none of them, and must not have to invent values for a rule it does
        # not use.
        self._rule = (
            EventRule(
                name="retail_transfer",
                min_dwell_seconds=_seconds(rules, "min_dwell_seconds"),
                max_transition_seconds=_seconds(rules, "max_transition_seconds"),
                classes=tuple(rules["tracked_classes"]),
            )
            if declares_route else None
        )
        allowlist = set(manifest.class_allowlist)
        undeclared = sorted(set(self._tracked_classes) - allowlist)
        if undeclared:
            raise ValueError(
                f"tracked_classes {undeclared} are not in class_allowlist "
                f"{sorted(allowlist)}. A class the pack says it does not show cannot "
                f"be tracked; accepting it would put a class name on the screen that "
                f"the scenario never claimed to detect."
            )
        self._zones, self._origin_name, self._destination_name = self._build_zones(rules)
        self._presence_zone_name = rules.get(PRESENCE_ZONE_KEY)
        self._min_presence = _seconds(rules, MIN_PRESENCE_KEY, required=False)
        if self._presence_zone_name is None and self._min_presence is not None:
            raise ValueError(
                f"event_rules sets {MIN_PRESENCE_KEY!r} but names no "
                f"{PRESENCE_ZONE_KEY!r}. A minimum presence with no zone to be present "
                f"in is a threshold that can never be met."
            )
        if self._presence_zone_name is not None:
            if self._presence_zone_name not in self._zones:
                raise ValueError(
                    f"{PRESENCE_ZONE_KEY} is {self._presence_zone_name!r}, which is not "
                    f"an ROI in event_rules; zones are {sorted(self._zones)}."
                )
            if self._min_presence is None:
                raise ValueError(
                    f"event_rules names {PRESENCE_ZONE_KEY!r} but no "
                    f"{MIN_PRESENCE_KEY!r}; the machine would have to invent a dwell "
                    f"requirement, and an invented one decides what counts as presence."
                )
        if self._origin_name is None and self._presence_zone_name is None:
            raise ValueError(
                f"event_rules configures no rule: neither a transfer route "
                f"({ORIGIN_SLOT_KEY!r}/{DESTINATION_SLOT_KEY!r}) nor a presence rule "
                f"({PRESENCE_ZONE_KEY!r}). A machine that can never emit is a counter "
                f"that reads zero forever, which on a booth screen says 'nothing is "
                f"happening'."
            )
        self._class_names = self._resolve_class_names(class_names)
        self._manifest = manifest
        self._progress: dict[int, _Progress] = {}
        self._presence: dict[int, tuple[float, str]] = {}
        self._events: list[BusinessEvent] = []
        self._emitted: set[str] = set()
        self._loop_id: int | None = None
        self._abandoned = 0
        self._abandoned_reasons: dict[str, int] = {}
        self._skipped_unmapped = 0

    def _resolve_class_names(self, class_names: tuple[str, ...] | None) -> tuple[str, ...]:
        """The label list, read and cross-checked **here and nowhere else**.

        The cross-check against the pinned class count is **unconditional** whenever
        `pinned.num_classes()` is available - it runs for an injected list too. It did
        not before, and that was a real hole: a machine built with
        `class_names=("person",)` was accepted silently, and then every `class_id` above
        0 became a `skipped_unmapped_class` at observation time instead of a startup
        failure. That is the failure mode this whole refactor exists to remove -
        `class_id` comes from the model, so the label list has to match the model
        whether or not the caller supplied it.

        If the pin is unreadable (an unprovisioned tree) the check is skipped rather
        than failed, and that is stated: a boot-time module must not refuse to import
        because the model is not on disk yet.
        """
        from app.inference.pinned import class_names as pinned_class_names
        from app.inference.pinned import num_classes

        if class_names is not None:
            names = tuple(str(n) for n in class_names)
            if not names:
                raise ValueError(
                    "class_names is empty. Every observation's class_id is resolved "
                    "against this list, and an empty list would make every frame "
                    "unusable while reporting nothing wrong."
                )
        else:
            names = tuple(pinned_class_names())

        try:
            declared = int(num_classes())
        except Exception:  # noqa: BLE001 - an unprovisioned tree, not a fault to report
            return names
        if len(names) != declared:
            raise ValueError(
                f"the label list has {len(names)} names but the pinned model declares "
                f"{declared} classes. Refusing to build the machine: with a list of the "
                f"wrong width an in-range class_id can be given the wrong name, and a "
                f"wrong class name on a retail event is a fabricated measurement. "
                f"Re-provision rather than substitute."
            )
        return names

    @staticmethod
    def _build_zones(rules: Mapping[str, Any]) -> tuple[dict[str, Zone], str | None, str | None]:
        """Build the `Zone`s named by the rule slots; return them and the route.

        A transfer route is declared by naming **both** ends, in `event_rules`, as
        `origin_slot` and `destination_slot`. There are **no default zone names**:
        an earlier revision defaulted to `shelf`/`basket`, which meant a pack acquired
        a transfer route it never asked for merely by having ROIs with those names,
        and a pack whose footage has no shelf had no way to say so. Both keys are
        required together; declaring one and not the other raises, because the machine
        would otherwise have to invent the missing end and an invented zone is a claim
        about the footage nobody made. `tests/test_events.py::
        test_the_default_zone_names_are_gone_from_the_app_layer` pins the absence.

        A transfer route is **optional**. A presence-only pack declares
        `presence_zone` and no origin/destination, and gets a machine that emits
        `zone_occupancy` only. That is what the shipped retail pack does, because the
        footage supports no product transfer.

        Every ROI is validated by `Zone` here, at load, with the slot's name in the
        message - the difference between a loud failure at startup and boxes in the
        wrong place for a whole session.
        """
        named_keys = {ORIGIN_SLOT_KEY, DESTINATION_SLOT_KEY, PRESENCE_ZONE_KEY}
        roi_slots = {
            key: value for key, value in rules.items()
            if key not in named_keys
            and isinstance(value, (tuple, list)) and len(value) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
        }
        declares_route = ORIGIN_SLOT_KEY in rules or DESTINATION_SLOT_KEY in rules
        if not declares_route:
            return ({slot: Zone(name=slot, kind="area", roi=tuple(roi))
                     for slot, roi in roi_slots.items()}, None, None)
        origin_name = rules.get(ORIGIN_SLOT_KEY)
        destination_name = rules.get(DESTINATION_SLOT_KEY)
        for role, slot in ((ORIGIN_SLOT_KEY, origin_name),
                           (DESTINATION_SLOT_KEY, destination_name)):
            if slot is None:
                raise ValueError(
                    f"event_rules declares one end of a transfer route but not the "
                    f"other: {sorted(k for k in (ORIGIN_SLOT_KEY, DESTINATION_SLOT_KEY) if k in rules)}. "
                    f"Both are required, because a half-declared route would have to "
                    f"invent the missing end - and an invented zone is a claim about "
                    f"the footage nobody made."
                )
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
    def rule(self) -> EventRule | None:
        """The transfer thresholds, or `None` in a presence-only pack.

        `None` is the honest answer for a pack that declares no route: there is no
        transfer to threshold, and inventing `min_dwell_seconds` for it would be a
        number on a screen describing a rule that does not exist.
        """
        return self._rule

    @property
    def tracked_classes(self) -> frozenset[str]:
        """The class names this machine will emit events for."""
        return self._tracked_classes

    @property
    def zones(self) -> Mapping[str, Zone]:
        """The route slots, by name. Read-only."""
        return MappingProxyType(self._zones)

    @property
    def origin(self) -> Zone | None:
        """The zone an item departs from, or `None` in a presence-only pack.

        Named by `event_rules`, never by this module. `None` is an honest answer for a
        pack that declares no transfer route - the shipped retail pack is presence-only
        because the footage supports no product transfer.
        """
        return self._zones[self._origin_name] if self._origin_name else None

    @property
    def destination(self) -> Zone | None:
        """The zone an item arrives in, or `None` in a presence-only pack."""
        return self._zones[self._destination_name] if self._destination_name else None

    @property
    def origin_name(self) -> str | None:
        return self._origin_name

    @property
    def destination_name(self) -> str | None:
        return self._destination_name

    @property
    def presence_zone(self) -> Zone | None:
        """The zone whose sustained occupancy is an event, or `None` if unconfigured."""
        if not self._presence_zone_name:
            return None
        return self._zones[self._presence_zone_name]

    @property
    def min_presence_seconds(self) -> float | None:
        return self._min_presence

    @property
    def skipped_unmapped_class(self) -> int:
        """Observations dropped because their `class_id` had no name.

        **Non-fatal by design.** `class_id` comes from the model, so an id past the
        end of the label list is a fact about the model, and killing the stream over it
        would take down a booth for something the label list - not the caller - is
        wrong about (AGENTS.md rule 12). The real invariant is checked once at startup,
        against the pin.

        Exposed so the screen can say "N skipped: class not in label list" instead of
        showing a count that is quietly short.
        """
        return self._skipped_unmapped

    @property
    def events(self) -> list[BusinessEvent]:
        """Events counted since the last `reset`. A copy, so callers cannot edit it."""
        return list(self._events)

    @property
    def abandoned_episodes(self) -> int:
        """Episodes that **ended without producing an event**, and why.

        "Episode" rather than "transfer", because this counter is incremented by two
        different things and a screen-facing name that said "transfer" while counting
        presence stays was wrong in the configuration that actually ships - the
        presence-only retail pack has no transfer in it at all, yet this reads
        non-zero. It was renamed for that reason.

        What it counts:
        - a transfer episode that reached `min_dwell_seconds` and was then lost before
          arriving in the destination (`track_lost`), and
        - a presence stay that was lost before it reached `min_presence_seconds`.

        It does **not** count a stay that had already emitted its event; nothing is
        retracted. `abandoned_reasons` says which.

        Zero is a real measurement - the machine did decide - and it is only meaningful
        next to `events`. A count of zero never means "we did not look".
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
        2. **Unmapped class id.** Counts `skipped_unmapped_class` and returns nothing.
           It does **not** raise: `class_id` comes from the model, so this is a fact
           about the model, and a per-frame exception would take a live stream down
           repeatedly for something the caller cannot fix (AGENTS.md rule 12). The
           matching invariant - label list length against the pinned class count - is
           checked once in `__init__`.
        3. **Class not tracked.** A person walking a route produces nothing, and no
           state is kept - there is no pending transfer to poison.
        4. **Not confirmed.** Not evidence. No state kept, nothing counted.
        5. **Track not current.** A coasted track's box is the last one *seen*; using
           it now would place the object somewhere nobody observed it.

        `now` is `frame.captured_at`. **This method reads no clock and opens no
        file** - the label list was resolved in `__init__`.
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

        now = frame.captured_at
        if not 0 <= track.class_id < len(self._class_names):
            self._skipped_unmapped += 1
            return ()
        class_name = self._class_names[track.class_id]

        if class_name not in self._tracked_classes:
            return ()
        if not track.confirmed:
            return ()
        if not track.is_current:
            return ()

        normalised = _normalise(track.box, frame)
        emitted: tuple[BusinessEvent, ...] = ()
        if self._presence_zone_name:
            emitted += self._maybe_emit_presence(track, class_name, frame, now,
                                                 normalised)
        if self._origin_name is not None:
            emitted += self._observe_route(track, class_name, frame, now, normalised)
        return emitted

    def _maybe_emit_presence(self, track: Track, class_name: str, frame: FramePacket,
                             now: float,
                             normalised: tuple[float, float, float, float],
                             ) -> tuple[BusinessEvent, ...]:
        """Sustained presence in the configured zone is itself the event.

        The test is on the object's **centre**, via `Zone.contains_point`, not on
        `Zone.contains`. That is measured, not stylistic: on `retail_checkout` a
        person's box is routinely clipped by the frame border (`x1 = -0.001`,
        `x2 = 0.999`), so full containment is never true and a presence rule written on
        it would report an empty shop.

        Dwell is `now - entered_at`, elapsed wall-clock time between real
        observations. Sampling the clip changes the resolution of the measurement, not
        the elapsed time, so a guard that samples measures the same presence the
        product would at full rate.

        Emitted **once per continuous stay** per track: the `dedup_key` carries the
        zone and the loop, and leaving the zone starts a new stay.
        """
        zone = self._zones[self._presence_zone_name]
        if not zone.contains_point(zone.centroid_of(normalised)):
            self._presence.pop(track.track_id, None)
            return ()
        record = self._presence.setdefault(track.track_id, _Stay(now))
        entered_at = record.entered_at
        if (now - entered_at) < self._min_presence:
            return ()
        # `record.counted` is what lets `on_track_lost` tell a stay that was already
        # counted from one that had not. An earlier version stored that fact as a bare
        # `None` in the same dict, which was indistinguishable from "no record" and made
        # the two cases impossible to separate; `on_track_lost` then counted an
        # already-counted stay as abandoned, so a stay that matured at t=2.0 and was
        # lost at t=3.5 reported an abandonment for an event that had already been
        # emitted. A `_Stay` record carries both facts and nothing overloads `None`.
        record.counted = True
        key = (f"{frame.source_loop_id}:{track.track_id}:{class_name}"
               f":presence:{self._presence_zone_name}")
        return self._emit_keyed("zone_occupancy", key, track.track_id, class_name,
                                frame, now)

    def _observe_route(self, track: Track, class_name: str, frame: FramePacket,
                       now: float, normalised: tuple[float, float, float, float],
                       ) -> tuple[BusinessEvent, ...]:
        """The origin -> destination -> origin legs. Absent in a presence-only pack."""
        origin = self._zones[self._origin_name]
        destination = self._zones[self._destination_name]
        in_origin = origin.contains(normalised)
        in_destination = destination.contains(normalised)

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
        return self._emit_keyed(kind, key, progress.track_id, class_name, frame, now)

    def _emit_keyed(self, kind: str, key: str, track_id: int, class_name: str,
                    frame: FramePacket, now: float) -> tuple[BusinessEvent, ...]:
        """Append an event unless its `dedup_key` has already been emitted this loop.

        This is also where a **second round trip by the same track in one loop** is
        suppressed. That is a deliberate, documented consequence of the key rather
        than of the state machine: a track that leaves the zone and comes back within
        the same loop produces the *same* key and is therefore counted once. If a
        product later needs repeated round trips counted separately, the key needs a
        per-episode discriminator - which is a change to this method and to
        `BusinessEvent.dedup_key`, not to the caller's expectations.
        """
        if key in self._emitted:
            # Replayed frames of an already-counted event. Skipping is the whole point
            # of the key: the same movement, seen again, is not a second event.
            return ()
        event = BusinessEvent(
            kind=kind,
            class_name=class_name,
            track_id=track_id,
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
        # A presence stay is interrupted by the loss too. Leaving it in place would
        # let a track that vanished mid-dwell come back under the same id and be
        # credited with time nobody observed it standing there. A stay that has
        # already been *counted* is not abandoned - it was counted, and nothing is
        # being retracted here.
        stay = self._presence.pop(track_id, _NEVER)
        # Only a stay that had **not** yet produced an event counts as abandoned. A
        # stay already counted produced its `zone_occupancy`; nothing is being
        # retracted here, and reporting an abandonment for it would put a number on the
        # screen that does not describe anything that was withheld. `_Stay.counted`
        # makes the two cases separable, which they were not when this held a bare
        # timestamp.
        had_uncounted_stay = isinstance(stay, _Stay) and not stay.counted
        had_pending = (
            progress is not None
            and progress.first_seen_in_origin is not None
            and progress.last_seen_in_origin is not None
            and self._rule is not None
            and progress.has_origin_dwell(self._rule.min_dwell_seconds)
            and not progress.transferred
        )
        if had_pending or had_uncounted_stay:
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
        self._presence = {}
        self._events = []
        self._emitted = set()
        self._abandoned = 0
        self._abandoned_reasons = {}
        self._skipped_unmapped = 0
        self._loop_id = loop_id

    # -- names ----------------------------------------------------------------
    #
    # There is deliberately no `_class_name_for` any more. It used to exist, and it
    # lazily imported `app.inference.pinned` - which reads `labels.txt` - on the
    # first observation, which put a filesystem read on the first frame of a live
    # stream and made this module's "no file is opened" claim false. A review caught
    # it by spying on `pathlib.Path.read_text`; spying on `builtins.open` does not
    # see it. The label list is resolved once in `__init__` (`_resolve_class_names`)
    # and an id with no entry is counted and skipped, never named.


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