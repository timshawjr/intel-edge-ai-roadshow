"""Cross-layer contracts. Pure data: no OpenVINO, no Qt, no I/O."""

from app.contracts.device_status import (
    DeviceStatus,
    Validity,
    mark_validated,
)

__all__ = ["DeviceStatus", "Validity", "mark_validated"]
