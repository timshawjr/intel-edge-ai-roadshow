"""`DeviceStatus` - the device record every later layer reports through.

Two rules in this module exist because of how badly a wrong device record shows up:

1. **Unavailable is a stated reason, never a zero.** `available=False` cannot be
   constructed without a non-empty `unavailable_reason`, and `available=True` cannot
   carry one. A device that cannot be used always says why.
2. **`available` is not `validated`.** Presence in `ov.Core().available_devices()` is a
   discovery fact. Whether the device has actually compiled and run a pinned model is a
   separate, later-earned fact. `describe()` reports the first and never claims the
   second, so an empty `supported_model_ids` means "not established yet", never
   "supports nothing".

`device_type` is the OpenVINO plugin class - `CPU`, `GPU` or `NPU` - which is the same
vocabulary as `device_id`, the field that actually identifies the device. It is
deliberately not a processor family and never a hardcoded one; this machine is Lunar Lake,
and nothing here may present a measurement as Panther Lake or Arrow Lake.

DEVIATION, 2026-10-01, on the parent's ruling: this field used to be the stringified
OpenVINO `DEVICE_TYPE` property, which measured `Type.INTEGRATED` for CPU, GPU and NPU
alike and therefore discriminated nothing while looking as though it did.

This module is pure. It does not import OpenVINO: `app.inference.devices` supplies the
measured `runtime_version` and `driver_version`, so the contract can be unit-tested on a
machine with no devices at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any

__all__ = ["DeviceStatus", "Validity", "mark_validated"]


class Validity(str, Enum):
    """How much a displayed value can be trusted.

    A `str` enum so a value survives a JSON round trip as itself. Deliberately never
    `"0"`: an unavailable reading and a zero reading mean opposite things to a booth
    visitor, and conflating them is how a person concludes the NPU is broken.
    """

    VALID = "valid"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DeviceStatus:
    """One device, as discovered and as earned.

    `runtime_version` holds the **full** OpenVINO build string, e.g.
    `2026.4.0-22959-99c81491cc3-releases/2026/4`, not a truncated `2026.4.0`. It feeds
    the blob cache key in `app.inference.cache`, and a fuller string invalidates more
    safely than a shorter one.

    `driver_version` is the value the device's Windows signed driver reports, verbatim,
    read from the PnP layer by `app.inference.device_drivers`. It is `None` when the
    device has no separately versioned driver - the CPU, on this machine - and it is
    never a number invented here and never `0`.
    """

    device_id: str
    full_name: str
    device_type: str
    available: bool
    validated: bool
    ready: bool
    runtime_version: str
    driver_version: str | None
    supported_model_ids: tuple[str, ...]
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        if self.available and self.unavailable_reason is not None:
            raise ValueError(
                f"{self.device_id}: available=True cannot carry unavailable_reason="
                f"{self.unavailable_reason!r}; an available device has nothing to explain"
            )
        if not self.available and not (self.unavailable_reason or "").strip():
            raise ValueError(
                f"{self.device_id}: available=False requires a non-empty "
                f"unavailable_reason; unavailable is a stated reason, never a zero"
            )
        if self.available and not self.full_name:
            raise ValueError(
                f"{self.device_id}: an available device must carry the discovered "
                f"full_name, never an empty one"
            )
        if not self.available and not self.validated:
            if self.ready:
                raise ValueError(
                    f"{self.device_id}: ready=True requires validated=True"
                )

    def to_json(self) -> dict[str, Any]:
        """A `json.dumps`-acceptable payload. The tuple field becomes a list."""
        payload: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            payload[f.name] = list(value) if isinstance(value, tuple) else value
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "DeviceStatus":
        """Rebuild a record. Fails closed on a missing field rather than defaulting."""
        known = {f.name for f in fields(cls)}
        missing = sorted(known - set(payload))
        if missing:
            raise ValueError(
                f"{cls.__name__}.from_json: missing required field(s) {missing}"
            )
        kwargs = {name: payload[name] for name in known}
        models = kwargs["supported_model_ids"]
        kwargs["supported_model_ids"] = (
            tuple(models) if isinstance(models, (list, tuple)) else (models,)
        )
        return cls(**kwargs)


def mark_validated(
    status: DeviceStatus, model_ids: Sequence[str]
) -> DeviceStatus:
    """Return a copy of `status` recording that `model_ids` were validated on it.

    This is the only way `validated`, `ready` and `supported_model_ids` become true.
    `ready` is `available and validated`: an unavailable device is never promoted,
    and its `unavailable_reason` is carried through untouched.
    """
    unique = tuple(dict.fromkeys(model_ids))
    return replace(
        status,
        validated=True,
        ready=status.available,
        supported_model_ids=unique,
    )
