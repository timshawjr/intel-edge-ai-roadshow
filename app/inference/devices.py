"""Device discovery and the compiled-target contract, over `ov.Core`.

The invariant this module exists to protect: **`EXECUTION_DEVICES`, re-read after the
first inference, is the only proof of placement.** A requested device preference is
never evidence of execution, so `CompiledTarget` holds no placement field at all — it
stores the live `Core` and `InferRequest` and reads the property on demand, inside
`execution_devices_now()`. There is no attribute a caller could read that was captured
before an inference ran.

Explicit `compile_model` targets only. `AUTO`, `HETERO`, `MULTI` and `BATCH` are
refused here, in the one place that could introduce them.

Measured on this machine, and the reason the reads below are written defensively:

- `ov.Core().available_devices()` -> `['CPU', 'GPU', 'NPU']`
- `FULL_DEVICE_NAME` CPU -> `'Intel(R) Core(TM) Ultra 9 288V'`
- `FULL_DEVICE_NAME` GPU -> `'Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)'`
- `FULL_DEVICE_NAME` NPU -> `'Intel(R) AI Boost'`
- `ov.__version__` -> `'2026.4.0-22959-99c81491cc3-releases/2026/4'`, a **build
  string**. `DeviceStatus.runtime_version` records it verbatim; the cache key in
  `app.inference.cache` depends on the whole thing.
- `EXECUTION_DEVICES` is a list on CPU (`['CPU']`) and GPU (`['GPU.0']`) but a bare
  string on NPU (`'NPU'`), so the read is normalised to a tuple.
- `core.get_property(compiled_model, ...)` is a `TypeError` in 2026.4.0. The property
  is read from the compiled model itself, or from the request's compiled model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import openvino as ov

from app.contracts.device_status import DeviceStatus

__all__ = [
    "DEVICE_IDS",
    "CompiledTarget",
    "DeviceRegistry",
    "TargetSpec",
    "require_static_shape",
]

#: The three devices the booth demonstrates. `NPU` is deliberately last: it is the one
#: most likely to be absent on a machine that is not this one, and the discovery code
#: below must say so honestly rather than drop it silently.
DEVICE_IDS: Final[tuple[str, ...]] = ("CPU", "GPU", "NPU")

#: Plugin prefixes that hand placement back to the runtime. Refused: the presenter must
#: always know where each job ran, and these hide it. `BATCH` is included because it is
#: the same class of implicit splitting.
FORBIDDEN_TARGETS: Final[frozenset[str]] = frozenset({"AUTO", "HETERO", "MULTI", "BATCH"})

EXECUTION_DEVICES: Final[str] = "EXECUTION_DEVICES"
FULL_DEVICE_NAME: Final[str] = "FULL_DEVICE_NAME"
SUPPORTED_PROPERTIES: Final[str] = "SUPPORTED_PROPERTIES"
#: The only device-specific driver-version property OpenVINO exposes, and only on the
#: NPU plugin. CPU and GPU expose none, so their `driver_version` is `null`.
NPU_DRIVER_VERSION: Final[str] = "NPU_DRIVER_VERSION"


def require_static_shape(shape: tuple[int, ...]) -> None:
    """Raise unless `shape` is a concrete, positive, fully four-dimensional shape.

    The NPU rejects dynamic shapes, so the refusal has to happen before the compile
    rather than surfacing as an opaque plugin error. A non-positive dimension is also
    refused, because `-1` is how a dynamic dimension arrives and `0` must never be
    mistaken for a measured extent.
    """
    if len(shape) != 4:
        raise ValueError(f"expected a 4-dimensional [N,C,H,W] shape, got {shape!r}")
    for axis, extent in enumerate(shape):
        if not isinstance(extent, int) or isinstance(extent, bool):
            raise ValueError(
                f"shape axis {axis} must be a concrete int; {extent!r} is not. "
                f"The NPU rejects dynamic shapes, so the extent must be static."
            )
        if extent <= 0:
            raise ValueError(
                f"shape axis {axis} is {extent}; a static shape must be positive, "
                f"and a dynamic shape (-1) is refused because the NPU rejects it."
            )


def _refuse_pseudo_device(device_id: str) -> None:
    """Raise if `device_id` is an implicit-placement plugin rather than a real device."""
    head = device_id.split(":", 1)[0].split(".", 1)[0].strip().upper()
    if head in FORBIDDEN_TARGETS:
        raise ValueError(
            f"refusing to compile to {device_id!r}: this demo uses explicit "
            f"compile_model targets only. {head} hides where the work actually ran, and "
            f"EXECUTION_DEVICES re-read after the first inference is the only proof of "
            f"placement. Compile to one of {DEVICE_IDS} instead."
        )


@dataclass(frozen=True)
class TargetSpec:
    """A request to compile one pinned model for one explicit device at one static shape."""

    device_id: str
    model_path: Path
    input_shape: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        _refuse_pseudo_device(self.device_id)
        require_static_shape(tuple(self.input_shape))


@dataclass(frozen=True)
class CompiledTarget:
    """A compiled model plus the live handles needed to prove where it runs.

    **No placement is captured here.** There is no `execution_devices` field, and the
    private `_core`/`_request` fields are live runtime objects, not recorded values.
    `dataclasses.fields()` is asserted in the tests to contain no placement-shaped name,
    and a fresh target is asserted to have performed zero property reads.

    The handles are excluded from `__eq__`, `__hash__` and `__repr__` so two targets
    compiled from the same spec on the same device compare equal, which is what the
    cache key and the report both mean by "the same target".
    """

    device_id: str
    model_path: Path
    model_hash: str
    input_shape: tuple[int, int, int, int]
    _core: Any = None
    _request: Any = None

    def __post_init__(self) -> None:
        if not self.model_hash:
            raise ValueError("model_hash must be the recorded SHA-256 of the IR")

    def execution_devices_now(
        self,
        core: "ov.Core | None" = None,
        request: "ov.InferRequest | None" = None,
    ) -> tuple[str, ...]:
        """`EXECUTION_DEVICES`, read live right now.

        Readable only once a first inference has run: before that, OpenVINO may have no
        meaningful value, and a value captured then would be a claim about a placement
        that had not happened. Both arguments are optional and default to the handles
        this target was built with, so the common call is `target.execution_devices_now()`
        and it still cannot return anything recorded at construction time.

        Returns a tuple, normalising the measured inconsistency that the NPU plugin
        reports a bare string while CPU and GPU report a list. Raises `TypeError` when
        no live handle is available, rather than returning an empty tuple that would
        read as "it ran nowhere".
        """
        live_core = core if core is not None else self._core
        live_request = request if request is not None else self._request
        if live_core is None and live_request is None:
            raise TypeError(
                f"{self.device_id}: this CompiledTarget has no live ov.Core or "
                f"ov.InferRequest, so EXECUTION_DEVICES cannot be read. Compile it "
                f"through DeviceRegistry.compile_target, or pass core and request."
            )
        if live_request is not None:
            # The request's own compiled model, so the reading belongs to the handle the
            # caller is actually going to infer on.
            value = live_request.get_compiled_model().get_property(EXECUTION_DEVICES)
        else:
            value = live_core.get_property(self.device_id, EXECUTION_DEVICES)
        if isinstance(value, str):
            # Measured: the NPU plugin returns 'NPU', not ['NPU'].
            return (value,)
        if value is None:
            return ()
        return tuple(value)

    @property
    def request(self) -> Any:
        """The live `ov.InferRequest`. Callers run their own first inference on it."""
        return self._request


def _hash_ir(model_path: Path) -> str:
    """SHA-256 over the IR `.xml`, computed here rather than copied from a manifest."""
    return hashlib.sha256(Path(model_path).read_bytes()).hexdigest()


class DeviceRegistry:
    """Discovers the devices and compiles explicit targets for them."""

    def __init__(self, core: "ov.Core") -> None:
        self._core = core

    @property
    def core(self) -> "ov.Core":
        return self._core

    @property
    def runtime_version(self) -> str:
        """The full OpenVINO build string, e.g. `2026.4.0-22959-99c81491cc3-releases/2026/4`.

        Not a bare `2026.4.0`: this value is part of the blob cache key, and two builds
        sharing a short version are not the same runtime.
        """
        return str(ov.__version__)

    def available(self) -> tuple[str, ...]:
        """Every device the runtime reports, discovered rather than assumed."""
        return tuple(self._core.available_devices)

    def full_name(self, device_id: str) -> str:
        """The discovered `FULL_DEVICE_NAME`.

        Never a hardcoded family string. This machine is Lunar Lake (Core Ultra Series
        2); a measurement from it must not be presented as Panther Lake or Arrow Lake.
        """
        return str(self._core.get_property(device_id, FULL_DEVICE_NAME))

    def _driver_version(self, device_id: str) -> str | None:
        """The device's own driver version, or `None` where it reports none.

        `None`, never `0` and never a guess. On this machine only the NPU plugin
        exposes `NPU_DRIVER_VERSION`; CPU and GPU have no driver-version property, so
        their `DeviceStatus.driver_version` is `null`.
        """
        try:
            supported = self._core.get_property(device_id, SUPPORTED_PROPERTIES)
        except Exception:  # noqa: BLE001 - a missing property is an absent version
            return None
        if NPU_DRIVER_VERSION not in supported:
            return None
        try:
            return str(self._core.get_property(device_id, NPU_DRIVER_VERSION))
        except Exception:  # noqa: BLE001
            return None

    def describe(self) -> tuple[DeviceStatus, ...]:
        """One `DeviceStatus` per id in `DEVICE_IDS`, in that order.

        `available` comes from `ov.Core().available_devices`. When a device is absent
        the record carries the OpenVINO error text as `unavailable_reason`, so the UI can
        say what went wrong instead of rendering a zero.

        `validated` and `ready` are left `False` and `supported_model_ids` empty on
        purpose. Presence in `available_devices` is a discovery fact, not proof that a
        model has been compiled and run; only `mark_validated()` after a real inference
        may set them.
        """
        present = self.available()
        statuses: list[DeviceStatus] = []
        for device_id in DEVICE_IDS:
            if device_id in present:
                statuses.append(DeviceStatus(
                    device_id=device_id,
                    full_name=self.full_name(device_id),
                    device_type=self._device_type(device_id),
                    available=True,
                    validated=False,
                    ready=False,
                    runtime_version=self.runtime_version,
                    driver_version=self._driver_version(device_id),
                    supported_model_ids=(),
                    unavailable_reason=None,
                ))
            else:
                statuses.append(DeviceStatus(
                    device_id=device_id,
                    full_name="",
                    device_type=device_id,
                    available=False,
                    validated=False,
                    ready=False,
                    runtime_version=self.runtime_version,
                    driver_version=None,
                    supported_model_ids=(),
                    unavailable_reason=self._unavailable_reason(device_id, present),
                ))
        return tuple(statuses)

    def _device_type(self, device_id: str) -> str:
        """`DEVICE_TYPE` as OpenVINO reports it, stringified.

        Measured: `Type.INTEGRATED` for CPU, GPU and NPU alike, so it carries no
        discriminating information. The plugin name in `device_id` is the label that
        does. It is recorded because it was measured, not because it is useful.
        """
        try:
            return str(self._core.get_property(device_id, "DEVICE_TYPE"))
        except Exception:  # noqa: BLE001
            return device_id

    def _unavailable_reason(self, device_id: str, present: tuple[str, ...]) -> str:
        """The OpenVINO error text for the missing device, plus what *is* there.

        Asking for the property is what produces the runtime's own wording, so the
        message the presenter reads is the plugin's and not this module's invention.
        """
        try:
            self._core.get_property(device_id, FULL_DEVICE_NAME)
        except Exception as exc:  # noqa: BLE001 - the error text is the point
            detail = f"{type(exc).__name__}: {exc}".strip()
        else:
            detail = f"not reported by ov.Core().available_devices()"
        return (f"device {device_id!r} is not available on this machine: {detail}. "
                f"ov.Core().available_devices() reported "
                f"{list(present) if present else 'no devices at all'}.")

    def compile_target(self, spec: TargetSpec) -> CompiledTarget:
        """Compile `spec` for its explicit device and return a `CompiledTarget`.

        Creates the infer request but does **not** run an inference: the caller owns the
        first inference, and only after it may it read `execution_devices_now()`. Any
        placement this returns is read live, never remembered.
        """
        _refuse_pseudo_device(spec.device_id)
        require_static_shape(tuple(spec.input_shape))
        present = self.available()
        if spec.device_id not in present:
            raise ValueError(
                f"cannot compile to {spec.device_id!r}: not available on this machine. "
                f"Available devices are {list(present)}."
            )
        model = self._core.read_model(str(spec.model_path))
        compiled = self._core.compile_model(model, spec.device_id)
        request = compiled.create_infer_request()
        return CompiledTarget(
            device_id=spec.device_id,
            model_path=Path(spec.model_path),
            model_hash=_hash_ir(spec.model_path),
            input_shape=tuple(spec.input_shape),
            _core=self._core,
            _request=request,
        )
