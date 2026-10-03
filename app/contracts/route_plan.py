"""`RoutePlan` - what was asked for, what actually ran, and how sure we are.

Review Focus #5 is *"the overlay shows a different device from the one actually
executing, because the card renders the requested device"*, and this record exists so
that the two cannot be confused by a consumer that has only one of them:

1. **`requested` and `confirmed` are separate maps, keyed the same way.** `requested`
   is what a caller asked for; `confirmed` is what `EXECUTION_DEVICES` said **after an
   inference ran**, normalised through `app.inference.devices.normalise_execution_devices`.
   A plan with `requested == {"A": "NPU"}` and `confirmed == {"A": ("CPU",)}` is a
   *fallback that actually happened*, and it is a state this record can express - which
   is the point. There is no field on this record that holds "the device we hope is
   running".
2. **`confirmed` carries no default for a stream that has not proved anything.** A
   stream with no entry is *absent*, not `("CPU",)`, not `()`. Reading `confirmed` for a
   stream is how a caller avoids reaching for `requested` instead.
3. **`state` is a three-value enum, not a bool.** `"pending"` means no inference has
   come back with a confirmed placement yet, and it is a state a screen must be able to
   draw: "moving..." is the honest thing to show, "NPU" is not. `"failed"` means the
   supervisor gave up on this generation and **must not keep claiming the target**.

**`generation` is the anti-stale filter.** Every device selection bumps it. A result or
a plan from an older generation describes a routing decision that has been superseded,
and spec section 7 requires it to be discarded on sight - the same shape as the
frame-identity contract, applied to routing. `Supervisor.select_device` returns the new
plan so a caller can compare generations without a second call.

**Mappings are wrapped read-only.** `frozen=True` does not freeze a `dict`, and a
`RoutePlan` whose `requested` map could be edited after the fact would render a
different device than the one the plan was built from. This record is therefore
unhashable, which is the same trade `ScenarioManifest` and `ModelOutputSpec` make.

## M2: three streams, and why `generation` is the **highest** generation

One `Supervisor` now owns three streams, and three separate failures have to be
expressible on one plan: stream A proving `CPU`, stream C failing, stream B still pending.
The three fields that used to carry the whole story - `generation`, `state`, `errors` -
are each therefore read as follows, and the aggregate rule is stated rather than left to
be inferred:

1. **`generation` is the highest generation across the streams.** A single stream keeps the
   exact M1 meaning. With three, it is the **age of the newest routing decision on the
   plan** and nothing else: a stream whose route did not change keeps its own, older
   generation, and a consumer that wants to know whether *this* stream moved must ask the
   stream - `stream_generation(stream_id)`. Using the plan's generation as a
   discard-by-generation filter would therefore discard valid results for the two streams
   that did not move.
2. **`state` is `failed` only when *every* stream failed**, and `active` when *at least
   one* stream has proved a placement. Neither is a claim about any particular stream, and
   a consumer rendering "Camera C" must read `stream_states` / `confirmed_devices("C")`.
3. **`errors` is the concatenation of every stream's stated reasons, stream order.** A
   failure on one stream is never collapsed away, and `stream_states` says which stream
   is in which state so a card can show it next to the right camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Mapping

__all__ = ["ROUTE_STATES", "RoutePlan", "StreamStats"]

#: The three states. Named rather than a free string so that "pending" and "active"
#: cannot end up as two spellings of one value, and so a caller that switches on the
#: state gets a `KeyError` from a renamed value rather than a silent fallthrough.
ROUTE_STATES: Final[tuple[str, ...]] = ("pending", "active", "failed")


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str; got {type(value).__name__}")
    if not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string; got {value!r}. An empty device id or "
            f"error cannot be told apart from a missing one, and both fields exist so "
            f"that a caller is never left guessing what ran."
        )
    return value


def _require_devices(name: str, value: Any) -> tuple[str, ...]:
    """A confirmed placement as a tuple of device names.

    A tuple, not a list, and never defaulted: an **empty** tuple here means "this stream
    has confirmed nothing", which is different from "this stream is not in the map at
    all" and different again from `("CPU",)`.
    """
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be a sequence of device names; got {type(value).__name__}. "
            f"The NPU plugin reports EXECUTION_DEVICES as a bare string, so accepting "
            f"one here would let a three-character string stand in for a device."
        )
    try:
        entries = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be iterable: {exc}") from exc
    return tuple(_require_text(f"{name} entry", str(entry)) for entry in entries)


@dataclass(frozen=True)
class RoutePlan:
    """One routing generation: requested per stream, confirmed per stream, and state."""

    generation: int
    requested: Mapping[str, str]
    confirmed: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    state: str = "pending"
    errors: tuple[str, ...] = ()
    #: Per-stream state. **M2 addition.** Absent streams have not declared one, which is
    #: the same convention `confirmed` uses: an unmentioned stream is unproven, not
    #: pending-and-fine.
    stream_states: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise TypeError(
                f"generation must be an int; got {self.generation!r} "
                f"({type(self.generation).__name__})"
            )
        if self.generation < 0:
            raise ValueError(
                f"generation must be >= 0; got {self.generation}. Generation 0 is the "
                f"first plan a supervisor ever builds, so a negative one is a broken "
                f"counter rather than an earlier plan."
            )
        if self.state not in ROUTE_STATES:
            raise ValueError(
                f"state must be one of {list(ROUTE_STATES)}; got {self.state!r}. An "
                f"unrecognised state is refused here rather than reaching a screen as a "
                f"word no code in this repository can produce."
            )
        if not isinstance(self.requested, Mapping):
            raise TypeError(
                f"requested must be a mapping of stream_id -> device id; got "
                f"{type(self.requested).__name__}"
            )
        if not isinstance(self.confirmed, Mapping):
            raise TypeError(
                f"confirmed must be a mapping of stream_id -> (device, ...); got "
                f"{type(self.confirmed).__name__}"
            )
        if not isinstance(self.stream_states, Mapping):
            raise TypeError(
                f"stream_states must be a mapping of stream_id -> state; got "
                f"{type(self.stream_states).__name__}"
            )
        object.__setattr__(self, "requested", MappingProxyType({
            _require_text("requested key", str(key)):
                _require_text("requested value", value)
            for key, value in self.requested.items()
        }))
        object.__setattr__(self, "confirmed", MappingProxyType({
            _require_text("confirmed key", str(key)):
                _require_devices("confirmed value", value)
            for key, value in self.confirmed.items()
        }))
        for key, state in self.stream_states.items():
            if state not in ROUTE_STATES:
                raise ValueError(
                    f"stream_states[{key!r}] = {state!r} is not one of {list(ROUTE_STATES)}. "
                    f"An unrecognised per-stream state would reach a screen as a word no "
                    f"code in this repository can produce."
                )
        object.__setattr__(self, "stream_states", MappingProxyType({
            _require_text("stream_states key", str(key)): str(state)
            for key, state in self.stream_states.items()
        }))
        if isinstance(self.errors, (str, bytes)):
            raise TypeError(
                f"errors must be a sequence of stated reasons; got "
                f"{type(self.errors).__name__}. A single reason is one-element, never "
                f"the reason itself."
            )
        object.__setattr__(self, "errors",
                           tuple(_require_text("errors entry", str(e)) for e in self.errors))

        unrequested = sorted(set(self.confirmed) - set(self.requested))
        if unrequested:
            raise ValueError(
                f"confirmed carries streams {unrequested} that requested does not "
                f"mention. Placement is reported for a route that was requested; a "
                f"confirmed stream with no request behind it is a device that ran "
                f"because somebody asked it to, and nothing here did."
            )
        unstated = sorted(set(self.stream_states) - set(self.requested))
        if unstated:
            raise ValueError(
                f"stream_states carries streams {unstated} that requested does not "
                f"mention. A per-stream state for a route nobody requested describes "
                f"work this plan does not account for."
            )
        if self.state == "failed" and not self.errors:
            raise ValueError(
                "a RoutePlan in state 'failed' must carry at least one stated reason. "
                "A failed route with no reason is a card showing an error with nothing "
                "to say, which is the same as showing no error at all."
            )

    # -- reads a consumer should use rather than reaching for `requested` ------------

    def confirmed_devices(self, stream_id: str) -> tuple[str, ...]:
        """The confirmed `EXECUTION_DEVICES` for `stream_id`, or `()` if unproven.

        `()` means **nothing has proved where this stream ran**, which is exactly what a
        caller must render as "pending" rather than as a device. This accessor exists so
        that the correct lookup is a named, tested call.
        """
        return tuple(self.confirmed.get(str(stream_id), ()))

    def requested_device(self, stream_id: str) -> str | None:
        """The device this stream was asked to run on, or `None` if it was not asked."""
        return self.requested.get(str(stream_id))

    def is_confirmed(self, stream_id: str, device_id: str) -> bool:
        """Whether `stream_id` proved it ran on `device_id`.

        Compared through `app.inference.devices.is_device`, because the readings are
        inconsistent on this machine: NPU reports the bare string `'NPU'`, CPU `['CPU']`
        and GPU `['GPU.0']`. `'GPU' in ('GPU.0',)` is `False`, and `'GPU' in 'GPU.0'` is
        `True` - a substring test on an unnormalised string passes on `'GP'` too. So no
        consumer has to remember the dotted spelling and none can pass by accident.
        """
        from app.inference.devices import is_device

        return is_device(self.confirmed_devices(stream_id), device_id)

    def fell_back(self, stream_id: str) -> bool:
        """Whether a stream ran somewhere other than where it was asked to run.

        `False` when the confirmation matches the request, or when either is absent -
        an unproven route has not fallen back, it has *not been shown*, and reporting
        `False` there would read as "it is running where you asked".
        """
        wanted = self.requested_device(stream_id)
        if wanted is None:
            return False
        confirmed = self.confirmed_devices(stream_id)
        if not confirmed:
            return False
        from app.inference.devices import is_device

        return not is_device(confirmed, wanted)

    def stream_state(self, stream_id: str) -> str | None:
        """This stream's own state, or `None` when it declared none.

        `self.state` is the **aggregate** across streams - `failed` only when every
        stream failed, `active` when at least one proved a placement - and it is the
        answer to "is anything wrong", not "what is Camera C doing". This accessor is
        the per-stream question, and it exists so that the correct lookup is a named,
        tested call rather than a comparison against the aggregate.

        `None` means unproven, which is not the same as `"pending"` by accident: a
        stream absent from `stream_states` has not declared anything, and a consumer
        must draw it as unknown rather than as healthy.
        """
        return self.stream_states.get(str(stream_id))

    def to_json(self) -> dict[str, Any]:
        """A `json.dumps`-acceptable payload. Mappings become objects, tuples lists."""
        return {
            "generation": self.generation,
            "requested": dict(self.requested),
            "confirmed": {k: list(v) for k, v in self.confirmed.items()},
            "state": self.state,
            "errors": list(self.errors),
            "stream_states": dict(self.stream_states),
        }


@dataclass(frozen=True)
class StreamStats:
    """One stream's accounting, independent of the other two.

    Review focus #4 is *"a toggle mid-stream dropping frames silently"*. The M1
    supervisor already reconciled `accepted + refused + drops == attempts` in the
    aggregate; with three streams the aggregate identity can hold while one stream's is
    wrong, so the identity is stated **per stream** here and `frame_attempts` is a field
    rather than something a caller reconstructs.

    ## Why `frame_attempts` is here and the brief did not list it

    The brief's field list is the brief's, and it is implemented. `frame_attempts` is
    **added**, because without it `frames_accepted + frames_refused + drops` has nothing
    to be checked against on this record: the whole point of the per-stream invariant is
    that somebody can make the check, and a record that cannot be checked is a record
    that is hoped for. The M1 aggregate counter already existed under this name, so the
    vocabulary is not new.

    `frames_analysed` and `generation` are added for the same reason: "how many frames did
    this stream actually get analysed" and "which routing generation is this stream on"
    are both questions a per-stream record exists to answer, and `RoutePlan.generation`
    cannot answer the second (it is the highest across streams).

    ## What each number counts, and the three that must never be merged

    - `frames_accepted` - frames the stream's bounded queue took.
    - `frames_refused` - frames `submit` refused *with a reason*, in
      `frames_refused_reasons`: `not_running`, `closed`, `failed`, `transitioning`,
      `unknown_stream`. A refusal is the pipeline not being there; it is not lag.
    - `drops` - frames refused for want of room. The pipeline falling behind.

    So `frames_accepted + frames_refused + drops == frame_attempts`, **for this stream
    alone**. `frames_analysed` is a separate population entirely - frames that came back
    with a placement - and is deliberately outside that identity.

    `state` is this stream's own state, never the aggregate: with three streams a single
    `failed` on Camera C is a fact about Camera C, and collapsing it into one verdict is
    the mistake `RoutePlan.stream_states` exists to prevent.
    """

    stream_id: str
    requested_device: str
    confirmed_devices: tuple[str, ...] = ()
    frames_accepted: int = 0
    frames_refused: int = 0
    frames_refused_reasons: Mapping[str, int] = field(default_factory=dict)
    drops: int = 0
    state: str = "pending"
    generations_seen: tuple[int, ...] = ()
    frame_attempts: int = 0
    frames_analysed: int = 0
    generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", _require_text("stream_id", self.stream_id))
        object.__setattr__(self, "requested_device",
                           _require_text("requested_device", self.requested_device))
        object.__setattr__(self, "confirmed_devices",
                           _require_devices("confirmed_devices", self.confirmed_devices))
        object.__setattr__(self, "state", str(self.state))
        if self.state not in ROUTE_STATES:
            raise ValueError(
                f"StreamStats.state must be one of {list(ROUTE_STATES)}; got {self.state!r}"
            )
        for name in ("frames_accepted", "frames_refused", "drops", "frame_attempts",
                     "frames_analysed", "generation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an int; got {value!r} ({type(value).__name__})"
                )
            if value < 0:
                raise ValueError(f"{name} must be >= 0; got {value}. A negative count is a "
                                 f"broken counter, not an observation.")
        if self.generation < 0:
            raise ValueError(f"generation must be >= 0; got {self.generation}")
        object.__setattr__(self, "generations_seen",
                           tuple(int(g) for g in self.generations_seen))
        if not isinstance(self.frames_refused_reasons, Mapping):
            raise TypeError(
                f"frames_refused_reasons must be a mapping of reason -> count; got "
                f"{type(self.frames_refused_reasons).__name__}. A refusal without a reason "
                f"is the failure this field exists to prevent."
            )
        object.__setattr__(self, "frames_refused_reasons", MappingProxyType({
            _require_text("frames_refused_reasons key", str(key)): int(count)
            for key, count in self.frames_refused_reasons.items()
        }))

    def reconciles(self) -> bool:
        """Whether `accepted + refused + drops == attempts` holds **for this stream**.

        A named method rather than a convention, because the identity is the thing a
        test asserts and a test should not have to re-derive the arithmetic to check it.
        With three streams it is entirely possible for the aggregate to reconcile while
        one stream's does not, and that is the case this method is here to catch.
        """
        return (self.frames_accepted + self.frames_refused + self.drops
                == self.frame_attempts)

    def fell_back(self) -> bool:
        """Whether this stream ran somewhere other than where it was asked to run.

        `False` when nothing has confirmed a placement: an unproven route has not fallen
        back, it has not been shown, and reporting `False` there would render as "it is
        running where you asked". Compared through `app.inference.devices.is_device`
        because `EXECUTION_DEVICES` is inconsistent on this machine (`'GPU.0'`, and the
        NPU's bare string `'NPU'`).
        """
        if not self.confirmed_devices:
            return False
        from app.inference.devices import is_device

        return not is_device(self.confirmed_devices, self.requested_device)

    def to_json(self) -> dict[str, Any]:
        """A `json.dumps`-acceptable payload, including whether the identity holds."""
        return {
            "stream_id": self.stream_id,
            "requested_device": self.requested_device,
            "confirmed_devices": list(self.confirmed_devices),
            "frames_accepted": self.frames_accepted,
            "frames_refused": self.frames_refused,
            "frames_refused_reasons": dict(self.frames_refused_reasons),
            "drops": self.drops,
            "state": self.state,
            "generations_seen": list(self.generations_seen),
            "frame_attempts": self.frame_attempts,
            "frames_analysed": self.frames_analysed,
            "generation": self.generation,
            "reconciles": self.reconciles(),
            "fell_back": self.fell_back(),
        }