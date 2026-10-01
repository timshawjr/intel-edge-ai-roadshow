"""Telemetry: PDH counter reads and the LUID-keyed instance map that gives them a device.

Two modules, deliberately separate:

- `app.telemetry.instance_map` is **pure**. It matches a `GPU Engine` PDH instance name to
  a device using facts recorded once by `tools/probe_telemetry.py`. It spawns nothing, so
  nothing on the 4 Hz sampling path can pay for a subprocess.
- `app.telemetry.pdh` is the **ctypes** reader for `pdh.dll`, and it spawns nothing either.
  A missing instance reads as `None`, never `0.0`.

The separation is the design: the LUID -> device mapping is resolved at operator time,
where CIM/WMI is acceptable, and the runtime only reads the recorded answer.
"""

from app.telemetry.instance_map import (
    EngineInstanceRule,
    InstanceMap,
    ParsedInstanceName,
    load_instance_map,
    parse_instance_name,
)

__all__ = [
    "EngineInstanceRule",
    "InstanceMap",
    "ParsedInstanceName",
    "load_instance_map",
    "parse_instance_name",
]
