"""`TelemetryAvailability` - what a telemetry provider can and cannot do on this machine.

**Why this is a contract and not a boolean in the UI.** A card that shows `0%` and a card
that shows `Utilization unavailable` are opposite claims about the hardware, and only one of
them may be true. This record makes the difference unrepresentable: `available=False` cannot
be constructed without a non-empty `unavailable_reason`, exactly as `DeviceStatus` does for
an absent device. There is no field in which "no reading" can be spelled as a number.

The same pairing rule as `app.contracts.device_status`:

- `available=True` **cannot** carry an `unavailable_reason`. An available provider has
  nothing to explain, and a reason sitting next to `available=True` is a contradiction a
  later reader has to resolve.
- `available=False` **requires** a non-empty `unavailable_reason`. Unavailable is a stated
  reason, never a blank and never a zero.
- `metric_names` is normalised to a tuple so the record cannot be mutated into a list after
  the fact, and it is **not** required to be empty or non-empty: a provider can be
  unavailable for every metric it would have carried, which is the state this milestone
  actually measured.

**`scope_note` is the scope rule, not a caveat.** Spec section 8 requires each provider's
scope and aggregation to be documented and forbids summing engines into a bogus total, so
the note travels with the record rather than living in a comment somewhere. It is allowed to
be empty, because an empty scope note is a documentation gap and refusing to construct the
record would hide the gap rather than report it.

`to_json()` mirrors `DeviceStatus.to_json`: the tuple field becomes a list so the payload is
`json.dumps`-acceptable, and every field is carried so a reader can check the screen against
the record.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Any

__all__ = ["TelemetryAvailability"]


@dataclass(frozen=True)
class TelemetryAvailability:
    """One telemetry provider's availability, with the reason when it is unavailable.

    `provider` is the provider's own name as the spec's telemetry table spells it, e.g.
    ``"pdh_gpu_engine"``. `metric_names` are the metrics this provider would carry, e.g.
    ``("gpu_compute_percent", "npu_neural_percent")`` - **named whether or not they can be
    read**, because "we know which metric this provider owns and we cannot read it" is the
    honest state, and an empty `metric_names` would read as "this provider owns nothing".

    `unavailable_reason` is `None` only when `available` is `True`. `scope_note` states how
    the metrics are scoped and aggregated, and is allowed to be empty.
    """

    provider: str
    available: bool
    metric_names: tuple[str, ...]
    unavailable_reason: str | None
    scope_note: str

    def __post_init__(self) -> None:
        provider = str(self.provider or "").strip()
        if not provider:
            raise ValueError(
                "a telemetry availability record needs a provider name; an unnamed "
                "provider cannot be disabled, enabled or explained"
            )
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "metric_names", tuple(
            str(name) for name in (self.metric_names or ())
        ))
        object.__setattr__(self, "scope_note", str(self.scope_note or ""))
        if not isinstance(self.available, bool):
            raise TypeError(
                f"{provider}: available must be a bool; got {self.available!r}. The whole "
                f"point of this record is that 'unavailable' and 'zero' cannot be confused, "
                f"and a non-bool is how that confusion starts."
            )
        if self.available and self.unavailable_reason is not None:
            raise ValueError(
                f"{provider}: available=True cannot carry unavailable_reason="
                f"{self.unavailable_reason!r}. An available provider has nothing to "
                f"explain, and a reason beside available=True is a contradiction the next "
                f"reader has to guess at."
            )
        if not self.available and not (self.unavailable_reason or "").strip():
            raise ValueError(
                f"{provider}: available=False requires a non-empty unavailable_reason. "
                f"Unavailable is a stated reason, never a blank and never a zero - a card "
                f"showing 0% and a card showing 'Utilization unavailable' are opposite "
                f"claims about the same hardware, and only one of them may be shown."
            )

    def to_json(self) -> dict[str, Any]:
        """A `json.dumps`-acceptable payload. The tuple field becomes a list."""
        payload: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            payload[f.name] = list(value) if isinstance(value, tuple) else value
        return payload

    def metric_names_as_tuple(self, names: Iterable[str]) -> tuple[str, ...]:
        """Normalise a caller's metric list to the tuple this field holds.

        Exists so a caller building the record from a list does not have to remember to
        convert, and so the conversion is one tested function rather than a `tuple(...)`
        sprinkled through every construction site.
        """
        return tuple(str(name) for name in names)
