"""`MetricSample` - one reading, with the fact of whether it is a reading.

Review Focus #3: *"unavailable rendered as 0."* A card that shows **NPU 0%** tells a
visitor the NPU is idle; if the truth is that no counter exists, it tells them the NPU
is broken. Those are opposite claims about the same hardware and only one of them may be
put on screen. So the difference is made **structural** here, once, for every producer in
the application:

1. **`value is None` and `validity is UNAVAILABLE` are the same fact and one without the
   other is refused.** There is no construction of this record that says "we could not
   measure this" and carries a number anyway. Every provider that reaches for a default
   hits a `ValueError` at the boundary rather than at the booth screen.
2. **A genuine `0.0` stays `VALID`.** The refusal must not become indiscriminate: an idle
   NPU at a measured 0.0% is a *reading*, and calling it unavailable is the same error
   pointing the other way. `tests/test_metric_sample.py` pins that directly, because a
   contract that cannot say "zero" cannot say anything.
3. **`unavailable_reason` is required when unavailable and forbidden otherwise.** The
   same pairing rule as `app.contracts.device_status.DeviceStatus` and
   `app.contracts.telemetry_availability.TelemetryAvailability`, for the same reason: an
   unexplained absence and a blank are the same shape on a screen.
4. **`to_json()` keeps `None` as JSON `null`.** A number substituted at serialisation
   time would be the same lie one layer further out, and it is exactly where a log file
   becomes a source of claims nobody can trace back to a measurement.

`is_stale` is **derived from the timestamp and the window**, never read from
`validity`. Staleness is a fact about the clock, and letting the record assert it
separately gives two copies of one fact that drift apart. An `UNAVAILABLE` sample is
never stale - a counter that was never readable does not decay into "stale", because
"stale" invites a reader to show the last known number, which is a measurement nobody
took. `effective_validity(now)` is the display-side helper that degrades a real reading
to `STALE` and leaves an absent one absent.

`Validity` is **imported from `app.contracts.device_status`, not redeclared.** The enum
the brief describes - `VALID`, `STALE`, `UNAVAILABLE` - is byte-for-byte the one that
module already defines for `DeviceStatus`, and two identical `str` enums would not
compare equal to each other: a caller reading `sample.validity == device_status_validity`
would get `False` for the same word. One enum, two records.

**This module is pure.** No `psutil`, no clock, no device, no I/O - `now` is always a
parameter, so every age in the product is a value somebody passed in and can reproduce.
The collector that produces these records is `app.telemetry.collector`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from app.contracts.device_status import Validity

__all__ = ["MetricSample", "Validity"]


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str; got {type(value).__name__}")
    if not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string; got {value!r}. An empty name, unit or "
            f"provider cannot be told apart from a missing one, and this record exists "
            f"so that they can be told apart."
        )
    return value


def _require_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number; got {value!r}")
    number = float(value)
    # `nan` fails every comparison, so it is refused by name rather than by a `>=` test
    # it would slip through. An age of `nan` would look fresh for ever.
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return number


@dataclass(frozen=True)
class MetricSample:
    """One metric, one reading, and the evidence that it is a reading.

    `name` is what the UI and the log call it, e.g. `"app_cpu_percent"`. `unit` is
    `"percent"`, `"bytes"` or `"ms"`. `provider` is who measured it, e.g. `"psutil"`,
    `"pdh_gpu_engine"` - a provider name is carried rather than implied, because the
    scope rules are per provider (spec section 8) and a reader needs to know which set of
    rules applied.

    **`device_scope`** is the device a reading belongs to, or `None` for a reading that
    belongs to no device - which is the honest answer for a metric that is *not* a device
    metric. `rss` is the sharpest case: it is **host process resident memory** and never
    accelerator memory, so `device_scope` is `None` and the unit is `bytes` of this
    process. Labelling it `NPU` because the process is the one driving the NPU would be
    the single most misleading number this application could publish, and no provider is
    permitted to make that claim from here.

    **`process_scope`** is the labelled set of processes a reading covers, e.g.
    `("ui", "worker")`. It is normalised to a tuple so a caller cannot mutate it into a
    list afterwards, and it may be empty - a reading about the whole machine belongs to
    no process.

    `window_seconds` is the window the reading is over: for a rate like CPU percent, the
    interval between the two reads it was computed from; for a gauge like `rss`, how
    often the gauge is read. Either way it is **carried**, because "NPU 41%" and "NPU
    41% averaged over 4 hours" are different claims and only the first is on the card.

    `sampled_at` is monotonic-clock seconds, matching `FramePacket.captured_at`, so a
    metric reading and the frame it was measured beside can be placed on one timeline.
    """
    name: str
    value: float | None
    unit: str
    provider: str
    device_scope: str | None
    process_scope: tuple[str, ...]
    window_seconds: float
    sampled_at: float
    validity: Validity
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "unit", _require_text("unit", self.unit))
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        if not isinstance(self.validity, Validity):
            raise TypeError(
                f"{self.name}: validity must be a Validity; got "
                f"{type(self.validity).__name__}. A free string here is how "
                f"'unavailable' and 'UNAVAILABLE' end up as two different states."
            )
        if self.device_scope is not None:
            object.__setattr__(self, "device_scope",
                               _require_text("device_scope", self.device_scope))
        if not isinstance(self.process_scope, (tuple, list)):
            raise TypeError(
                f"{self.name}: process_scope must be a sequence of labels; got "
                f"{type(self.process_scope).__name__}"
            )
        labels = tuple(_require_text("process_scope entry", str(label))
                       for label in self.process_scope)
        object.__setattr__(self, "process_scope", labels)

        window = _require_finite("window_seconds", self.window_seconds)
        if window <= 0.0:
            raise ValueError(
                f"{self.name}: window_seconds must be > 0; got {window}. A reading with "
                f"no window says nothing about what it averages over, which is half the "
                f"claim a rate makes."
            )
        object.__setattr__(self, "window_seconds", window)
        sampled_at = _require_finite("sampled_at", self.sampled_at)
        if sampled_at < 0.0:
            raise ValueError(
                f"{self.name}: sampled_at must be >= 0 on the monotonic clock; got "
                f"{sampled_at}"
            )
        object.__setattr__(self, "sampled_at", sampled_at)

        reason = self.unavailable_reason
        if reason is not None:
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"{self.name}: unavailable_reason must be a non-empty stated reason "
                    f"or None; got {reason!r}. A blank reason is indistinguishable from "
                    f"no reason, which is the absence this record exists to report."
                )
        if self.value is None:
            if self.validity is not Validity.UNAVAILABLE:
                raise ValueError(
                    f"{self.name}: value=None must carry validity=Validity.UNAVAILABLE, "
                    f"not {self.validity!r}. There is no state in which this record says "
                    f"'no measurement' and also claims to be valid or stale."
                )
            if not reason:
                raise ValueError(
                    f"{self.name}: validity=UNAVAILABLE requires a non-empty "
                    f"unavailable_reason. A card showing 0% and a card showing "
                    f"'unavailable' are opposite claims about the same hardware, and "
                    f"only one of them may be shown - so the reason has to travel with "
                    f"the absence."
                )
            object.__setattr__(self, "value", None)
            return

        measured = _require_finite(f"{self.name}.value", self.value)
        object.__setattr__(self, "value", measured)
        if self.validity is Validity.UNAVAILABLE:
            raise ValueError(
                f"{self.name}: validity=UNAVAILABLE cannot carry value={measured!r}. An "
                f"unavailable reading is a number nobody measured, and a number beside "
                f"this validity is the shape that renders as a fact."
            )
        if reason is not None:
            raise ValueError(
                f"{self.name}: validity={self.validity.value!r} cannot carry "
                f"unavailable_reason={reason!r}. A measured reading has nothing to "
                f"explain, and a reason beside it is a contradiction the next reader has "
                f"to resolve."
            )

    def is_stale(self, now: float) -> bool:
        """Whether this reading has aged past its own window, at `now`.

        Derived from `sampled_at` and `window_seconds`; the record's `validity` is not
        consulted, because it cannot have been set from the clock without a second copy
        of the same fact drifting out of step.

        - `UNAVAILABLE` is never stale. A counter that was never readable does not go
          stale, it stays unavailable - and "stale" invites a reader to show the last
          known number, which is a measurement nobody took.
        - `now < sampled_at` is a clock disagreement, not "very fresh". It reads as
          stale rather than as a negative age, because a negative age rendered on a card
          would be a number nobody measured either.
        """
        if self.validity is Validity.UNAVAILABLE:
            return False
        age = float(now) - self.sampled_at
        if age < 0.0:
            return True
        return age > self.window_seconds

    def effective_validity(self, now: float) -> Validity:
        """What should be rendered for this sample at `now`.

        `UNAVAILABLE` stays `UNAVAILABLE`; a recorded `STALE` stays `STALE`; a `VALID`
        reading that has aged past its window degrades to `STALE`. **The value is never
        touched** - ageing changes what the number means, not what the number is.
        """
        if self.validity is not Validity.VALID:
            return self.validity
        return Validity.STALE if self.is_stale(now) else Validity.VALID

    def to_json(self) -> dict[str, Any]:
        """A `json.dumps`-acceptable payload. The tuple field becomes a list.

        **`None` stays `None`, and so serialises as `null`.** Nothing in this method
        substitutes a number for an absent reading; that substitution is the failure
        this module exists to prevent, and doing it here would put it one layer further
        from the place where it can be caught.
        """
        payload: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, tuple):
                payload[f.name] = list(value)
            elif isinstance(value, Validity):
                payload[f.name] = value.value
            else:
                payload[f.name] = value
        return payload
