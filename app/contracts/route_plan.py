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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Mapping

__all__ = ["ROUTE_STATES", "RoutePlan"]

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

    def to_json(self) -> dict[str, Any]:
        """A `json.dumps`-acceptable payload. Mappings become objects, tuples lists."""
        return {
            "generation": self.generation,
            "requested": dict(self.requested),
            "confirmed": {k: list(v) for k, v in self.confirmed.items()},
            "state": self.state,
            "errors": list(self.errors),
        }