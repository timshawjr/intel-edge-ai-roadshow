"""Per-device Windows driver versions, read from the PnP layer.

`ov.Core().get_property(device, "NPU_DRIVER_VERSION")` is exposed by the **NPU plugin
only**. Measured on this machine with OpenVINO 2026.4.0:

- CPU reports no driver-version property at all.
- GPU reports no driver-version property at all.
- NPU reports `1005540`.

That is a property of the *accessor*, not of the hardware. Windows versions the GPU
driver too, and says so. `Win32_PnPSignedDriver`, measured on this machine:

    NPU  PCI\\VEN_8086&DEV_643E   Intel(R) AI Boost          32.0.100.5540
    GPU  PCI\\VEN_8086&DEV_64A0   Intel(R) Arc(TM) 140V GPU  32.0.101.6737

`1005540` is `32.0.100.5540` with the leading `32.0.` and the dots removed, so the NPU
plugin's property and the PnP row describe the same driver. The two agreeing is a live
cross-check, and it is pinned as one.

The **CPU has no separate driver artefact** - the processor is not a PnP function with a
signed driver - so the lookup finds nothing for it and its `driver_version` stays `None`.
That is the correct answer, not a gap, and it is pinned as one.

**Nothing here is hardcoded.** Neither the PCI device ids nor the version strings appear
in this module; both are read. A device is located from OpenVINO by two discovered
handles, in order:

1. A PCI device id the plugin reports itself. The GPU plugin's `SUPPORTED_PROPERTIES`
   includes `GPU_DEVICE_ID`, which reads `'0x64a0'` here - the `DEV_64A0` field of the PnP
   row above. The property name is discovered by enumerating `SUPPORTED_PROPERTIES`, not
   by a list written down here.
2. The device's `FULL_DEVICE_NAME`, matched against the PnP `DeviceName` after trailing
   parenthesised groups are dropped (`'Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)'` ->
   `'Intel(R) Arc(TM) 140V GPU'`). This is what resolves the NPU, whose plugin exposes no
   device id at all. Verified for both on this machine.

**Why a subprocess.** The pinned dependency set has no `pywin32`, so the enumeration is one
`powershell.exe` / CIM call. It is made at most once per lookup, lazily, and memoised. It
is not on the telemetry path and must never be called per sample - `enumerations` is
observable so a test can prove that. It runs as a standard user and needs no elevation.

**Every failure is `None`:** no PowerShell on the box, execution blocked by policy,
non-zero exit, unparsable output, no match, or an ambiguous match. It never raises and
never returns a plausible-looking string it did not read. `None` is distinct from `"0"`
and from `""` all the way into the blob cache key in `app.inference.cache`.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "PnpDriver",
    "WindowsDriverLookup",
    "default_driver_lookup",
    "driver_matches_device_id",
    "looks_like_a_pci_device_id",
    "normalise_device_name",
    "windows_driver_version",
]

#: Measured ~3.0 s on this machine, cold. Generous enough that a loaded booth laptop does
#: not fail the lookup, tight enough that a hung PowerShell is not waited on forever.
POWERSHELL_TIMEOUT_S: Final[float] = 20.0

#: Resolved rather than assumed, but this is where Windows 11 puts it.
POWERSHELL_EXE: Final[str] = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"

#: One call, one shape. `-Property`-free so `Select-Object` keeps the payload to the three
#: fields this module reads; the default projection drags ~50 KB of CIM class metadata.
_ENUMERATE_SCRIPT: Final[str] = (
    "Get-CimInstance -ClassName Win32_PnPSignedDriver "
    "-Filter \"DeviceID LIKE 'PCI%'\" "
    "| Select-Object DeviceID,DeviceName,DriverVersion "
    "| ConvertTo-Json -Compress"
)

#: The PCI device-id field of a PnP hardware id, e.g. `DEV_64A0` -> `64A0`.
_DEVICE_ID_FIELD: Final[re.Pattern[str]] = re.compile(r"DEV_([0-9A-Fa-f]{4})")

#: A PnP instance id, a `DEV_`-prefixed id, or a bare 4-hex-digit PCI device id. The
#: GPU plugin's `DEVICE_ID` reads `'0'`, which is a plugin ordinal rather than a PCI
#: device id, and is rejected here rather than searched for.
_PCI_DEVICE_ID_HINT: Final[re.Pattern[str]] = re.compile(
    r"PCI\\.*|DEV_.*|0X[0-9A-F]{4}|[0-9A-F]{4}", re.IGNORECASE
)

#: Signature of an injected enumeration: takes the script and the timeout, returns stdout.
Runner = Callable[[str, float], str]


@dataclass(frozen=True)
class PnpDriver:
    """One row of `Win32_PnPSignedDriver`, reduced to the three fields used here."""

    device_id: str
    name: str
    driver_version: str | None

    @property
    def pci_device_id(self) -> str:
        """The `DEV_` field of the hardware id, e.g. `64A0`, or `""` if there is none."""
        match = _DEVICE_ID_FIELD.search(self.device_id)
        return match.group(1).upper() if match else ""


def _run_powershell(script: str, timeout: float) -> str:
    """Run `script` and return stdout. Raises on any failure; callers degrade to `None`."""
    completed = subprocess.run(
        [POWERSHELL_EXE, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        # No console flash behind the booth app. `CREATE_NO_WINDOW` is Windows-only.
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        raise OSError(
            f"powershell.exe exited {completed.returncode}: "
            f"{(completed.stderr or '').strip()[:200]}"
        )
    return completed.stdout


def looks_like_a_pci_device_id(value: object) -> bool:
    """Whether `value` could be a PCI device id, as opposed to a plugin-local ordinal.

    Accepts a full PnP instance id (`PCI\\VEN_8086&DEV_64A0&SUBSYS_...\\3&...`), a
    `DEV_`-prefixed id, and a bare id such as `'0x64a0'` or `'64A0'`. Rejects `'0'`, which
    is what the GPU plugin reports for `DEVICE_ID`.
    """
    text = str(value or "").strip()
    return bool(text) and _PCI_DEVICE_ID_HINT.fullmatch(text) is not None


def _bare_device_id(text: str) -> str:
    """`0x64a0`, `64A0` and `DEV_64A0` all normalise to `64A0`."""
    upper = text.strip().upper()
    for prefix in ("DEV_", "0X"):
        if upper.startswith(prefix):
            upper = upper[len(prefix):]
    return upper


def driver_matches_device_id(driver: PnpDriver, candidate: str) -> bool:
    """Whether `driver` is the PnP signed driver for the PCI device `candidate`.

    Two forms are accepted, because both are things a caller can actually hold: a PnP
    instance id or hardware-id prefix, which must be a prefix of the driver's own id; and
    a bare PCI device id, which must equal the driver's `DEV_` field.
    """
    text = str(candidate or "").strip()
    if not text:
        return False
    if driver.device_id.upper().startswith(text.upper()):
        return True
    field = driver.pci_device_id
    return bool(field) and field == _bare_device_id(text)


def normalise_device_name(name: str) -> str:
    """A device name reduced to something two layers can be compared on.

    Trailing parenthesised groups are dropped, because OpenVINO appends qualifiers that
    the PnP friendly name does not carry: `'Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)'` and
    `'Intel(R) Arc(TM) 140V GPU (16GB)'` both reduce to
    `'INTEL(R) ARC(TM) 140V GPU'`. Interior `(R)` and `(TM)` are kept, because they are
    part of the vendor's name and dropping them would make the match looser than it needs
    to be. Case and runs of whitespace are collapsed.
    """
    text = str(name or "").strip()
    while True:
        cut = text.rfind("(")
        if cut > 0 and text.endswith(")"):
            text = text[:cut].strip()
            continue
        return " ".join(text.split()).upper()


def _parse_enumeration(stdout: str) -> tuple[PnpDriver, ...]:
    """The JSON payload from one enumeration into `PnpDriver` rows."""
    text = str(stdout or "").strip()
    if not text:
        return ()
    payload = json.loads(text)
    if isinstance(payload, dict):
        # A single object, not a one-element array.
        payload = [payload]
    drivers: list[PnpDriver] = []
    for record in payload:
        if not isinstance(record, dict):
            continue
        device_id = str(record.get("DeviceID") or "").strip()
        if not device_id:
            continue
        version = str(record.get("DriverVersion") or "").strip()
        drivers.append(PnpDriver(
            device_id=device_id,
            name=str(record.get("DeviceName") or "").strip(),
            driver_version=version or None,
        ))
    return tuple(drivers)


class WindowsDriverLookup:
    """Finds the signed Windows driver for a device, once, and remembers the answer.

    Construct with `drivers=` to pin the PnP table and stay hermetic - that is how the
    tests exercise matching without a subprocess. Construct with `runner=` to substitute
    the enumeration itself. Construct with neither to use the real thing, once.
    """

    def __init__(
        self,
        drivers: Iterable[PnpDriver] | None = None,
        *,
        runner: Runner | None = None,
        timeout: float = POWERSHELL_TIMEOUT_S,
    ) -> None:
        # `None` means "not looked up yet"; `()` means "looked up, found nothing".
        self._drivers: tuple[PnpDriver, ...] | None = (
            None if drivers is None else tuple(drivers)
        )
        self._runner = runner
        self._timeout = timeout
        #: Observable, so a test can prove the enumeration happened once and not per
        #: sample. Nothing in the telemetry path may read this after the first fill.
        self.enumerations = 0

    @property
    def drivers(self) -> tuple[PnpDriver, ...]:
        """The PnP signed-driver table, enumerated on first use and memoised."""
        if self._drivers is None:
            self.enumerations += 1
            self._drivers = self._enumerate()
        return self._drivers

    def _enumerate(self) -> tuple[PnpDriver, ...]:
        runner = self._runner or _run_powershell
        try:
            stdout = runner(_ENUMERATE_SCRIPT, self._timeout)
        except Exception:  # noqa: BLE001 - an unreadable PnP table is "no versions"
            return ()
        try:
            return _parse_enumeration(stdout)
        except Exception:  # noqa: BLE001
            return ()

    def pci_device_ids_for(
        self,
        full_name: str,
        *,
        device_id_hints: Sequence[str] = (),
    ) -> tuple[str, ...]:
        """PnP device ids for the device OpenVINO discovered, in match order.

        `device_id_hints` are PCI device ids the plugin reported for itself. `full_name`
        is its `FULL_DEVICE_NAME`. Empty means the lookup found no PnP device, which is
        the correct and expected outcome for the CPU.
        """
        found: list[str] = []

        def add(driver: PnpDriver) -> None:
            if driver.device_id not in found:
                found.append(driver.device_id)

        for hint in device_id_hints:
            for driver in self.drivers:
                if driver_matches_device_id(driver, hint):
                    add(driver)
        target = normalise_device_name(full_name)
        if target:
            for driver in self.drivers:
                if normalise_device_name(driver.name) == target:
                    add(driver)
        return tuple(found)

    def driver_version_for(
        self,
        full_name: str,
        *,
        device_id_hints: Sequence[str] = (),
    ) -> str | None:
        """The signed driver's `DriverVersion` for this device, or `None`.

        `None` when the device has no PnP driver (the CPU), when the PnP layer could not
        be read, when the driver reports no version, or when the candidate ids match more
        than one distinct version - an ambiguity is `None`, never a coin flip.
        """
        return self.driver_version_for_ids(
            self.pci_device_ids_for(full_name, device_id_hints=device_id_hints)
        )

    def driver_version_for_ids(
        self, pci_device_ids: Sequence[str]
    ) -> str | None:
        """The signed driver's `DriverVersion` for any of these PCI device ids."""
        versions: set[str] = set()
        for candidate in pci_device_ids or ():
            for driver in self.drivers:
                if driver.driver_version and driver_matches_device_id(driver, candidate):
                    versions.add(driver.driver_version)
        return versions.pop() if len(versions) == 1 else None


_DEFAULT_LOOKUP: WindowsDriverLookup | None = None


def default_driver_lookup() -> WindowsDriverLookup:
    """The process-wide lookup, so the enumeration is paid once per process, not per call."""
    global _DEFAULT_LOOKUP
    if _DEFAULT_LOOKUP is None:
        _DEFAULT_LOOKUP = WindowsDriverLookup()
    return _DEFAULT_LOOKUP


def windows_driver_version(
    pci_device_ids: Sequence[str], *, lookup: WindowsDriverLookup | None = None
) -> str | None:
    """The Windows `DriverVersion` for the first unambiguous PnP match, else `None`.

    `None` - never a raised exception and never a plausible-looking string - when no id
    matches, when more than one distinct version matches, or when the PnP layer cannot be
    read at all.
    """
    active = lookup if lookup is not None else default_driver_lookup()
    return active.driver_version_for_ids(pci_device_ids)
