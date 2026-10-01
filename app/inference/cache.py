"""Identity of a compiled artifact, and the gate that decides whether to reuse one.

Review Focus #5: **a compiled blob is never reused across a driver or runtime change.**
NPU blob-cache compatibility is not guaranteed across driver versions, and a blob is
never treated as universally compatible with any other machine.

Both functions here are pure and total: no I/O, no OpenVINO, no environment lookups.
The caller supplies every input as an argument, so the identity of an artifact is a
function of five values and nothing else.

- `driver_version` is the value the target device reports, verbatim, and may be `None`
  when the device exposes no driver-version property. `None` is encoded distinctly from
  `""` and from `"0"`, so a blob recorded while the driver version was unknown is
  rejected the moment one becomes known.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

__all__ = ["IDENTITY_FIELDS", "cache_identity", "cache_is_valid", "cache_key"]

#: The complete set of inputs that define a compiled artifact. Adding a field here is
#: what forces every caller to supply it, so an identity can never be partial.
IDENTITY_FIELDS: tuple[str, ...] = (
    "model_hash",
    "device_id",
    "runtime_version",
    "driver_version",
    "properties",
)

# `None` and `""` are different machine states, and neither is `"0"`.
_ABSENT = "\x00<absent>"


def cache_identity(
    *,
    model_hash: str,
    device_id: str,
    runtime_version: str,
    driver_version: str | None,
    properties: str,
) -> dict[str, str]:
    """The complete recorded identity of one compiled artifact, as plain strings.

    `runtime_version` is the **full** OpenVINO build string
    (`2026.4.0-22959-99c81491cc3-releases/2026/4`), not a truncated `2026.4.0`: two
    builds that share a short version are not the same runtime, and reusing a blob
    across them is exactly the failure this module exists to prevent.
    """
    return {
        "model_hash": str(model_hash),
        "device_id": str(device_id),
        "runtime_version": str(runtime_version),
        "driver_version": _ABSENT if driver_version is None else str(driver_version),
        "properties": str(properties),
    }


def cache_key(
    *,
    model_hash: str,
    device_id: str,
    runtime_version: str,
    driver_version: str | None,
    properties: str,
) -> str:
    """Identity of a compiled artifact. Any input change must invalidate the blob.

    A SHA-256 over a canonical JSON encoding, not a delimited join: a join would let a
    value containing the delimiter forge a collision between two different identities.
    """
    identity = cache_identity(
        model_hash=model_hash,
        device_id=device_id,
        runtime_version=runtime_version,
        driver_version=driver_version,
        properties=properties,
    )
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cache_is_valid(recorded: Mapping[str, str], current: Mapping[str, str]) -> bool:
    """True only when `current` is a complete identity and every field matches.

    The plan words this as "every key in `current` matches `recorded`", which is
    vacuously true of an empty or partial `current`. That reading turns a caller bug -
    forgetting a field - into a silent pass, so `current` is additionally required to
    carry every field in `IDENTITY_FIELDS`. A key absent from `recorded` is likewise a
    mismatch, not a pass.
    """
    if set(current) != set(IDENTITY_FIELDS):
        return False
    for name in IDENTITY_FIELDS:
        if name not in recorded or recorded[name] != current[name]:
            return False
    return True
