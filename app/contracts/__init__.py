"""Cross-layer contracts. Pure data: no OpenVINO, no Qt, no I/O."""

from app.contracts.device_status import (
    DeviceStatus,
    Validity,
    mark_validated,
)
from app.contracts.frame import FramePacket
from app.contracts.scenario_manifest import ScenarioManifest

__all__ = ["DeviceStatus", "FramePacket", "ScenarioManifest", "Validity",
           "mark_validated"]