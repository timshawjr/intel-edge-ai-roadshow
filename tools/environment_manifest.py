#!/usr/bin/env python3
"""Collect the M6 known-good environment **from the machine**, never from memory.

Everything that describes the machine is read at run time:

- Windows inventory from CIM (`Win32_ComputerSystem`, `Win32_ComputerSystemProduct`,
  `Win32_Processor`, `Win32_VideoController`, `Win32_PnPEntity`, `Win32_PnPSignedDriver`,
  `Win32_OperatingSystem`, `Win32_DesktopMonitor`, `WmiMonitorID`, `Win32_Battery`),
- the active power scheme from `powercfg /getactivescheme`,
- Python from `sys`/`platform`,
- OpenVINO from the installed distribution metadata and `openvino.Core()`.

**A field that cannot be read is `None` with a reason in `unavailable`.** It is never
`0`, never `False`, and never a remembered value. This machine has no `Win32_Battery`
instance at all, so `ac_status` is `None` with that reason rather than a guessed `True`.

**There is no single SKU.** The machine reports three SKU-bearing records that disagree:
the `Win32_ComputerSystem` model (`NUC14LNK`, family `LN`), the hostname
(`NUC14PROAI`), and the PCI subsystem ID (`88F21043`). They are preserved as three
records. The hostname is not a model, the PCI subsystem is not a marketing SKU, and this
module does not promote one of them to "the" SKU.

Usage::

    python tools/environment_manifest.py --output docs/install/KNOWN-GOOD-environment.json
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata as metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final[Path] = REPO_ROOT / "docs" / "install" / "KNOWN-GOOD-environment.json"
COLLECTOR: Final[str] = (
    "tools/environment_manifest.py: PowerShell CIM (Win32_* + WmiMonitorID) + "
    "powercfg + sys/platform + importlib.metadata + openvino.Core().available_devices"
)
SKU_NOTE: Final[str] = (
    "This machine reports three SKU-bearing records that disagree: the "
    "Win32_ComputerSystem model with its system family, the hostname, and the PCI "
    "subsystem ID. No single SKU is defensible, so all three are preserved and none is "
    "promoted to 'the' SKU. The hostname is not a model; the PCI subsystem ID is not a "
    "marketing name."
)

_SUBSYSTEM_RE: Final[re.Pattern[str]] = re.compile(r"SUBSYS_([0-9A-Fa-f]{8})", re.IGNORECASE)
_POWER_RE: Final[re.Pattern[str]] = re.compile(
    r"Power Scheme GUID:\s*([0-9a-fA-F-]{36})\s*\(([^)]+)\)"
)


@dataclass(frozen=True)
class CpuRecord:
    name: str
    manufacturer: str | None
    cores: int | None
    logical_processors: int | None
    max_clock_mhz: int | None


@dataclass(frozen=True)
class DriverRecord:
    version: str | None
    date: str | None
    provider: str | None
    inf: str | None
    reason: str | None


@dataclass(frozen=True)
class GpuRecord:
    name: str
    pnp_id: str | None
    driver_version: str | None
    driver_date: str | None
    driver_provider: str | None
    driver_inf: str | None
    adapter_ram_reported_bytes: int | None
    adapter_ram_note: str
    current_width: int | None
    current_height: int | None
    current_refresh_hz: int | None


@dataclass(frozen=True)
class NpuRecord:
    name: str
    manufacturer: str | None
    pnp_id: str | None
    status: str | None
    driver_version: str | None
    driver_date: str | None
    driver_provider: str | None
    driver_inf: str | None


@dataclass(frozen=True)
class WindowsRecord:
    caption: str | None
    version: str | None
    build_number: str | None
    ubr: int | None
    edition_id: str | None
    display_version: str | None
    registry_product_name: str | None
    product_name_note: str | None
    architecture: str | None


@dataclass(frozen=True)
class PowerPlanRecord:
    guid: str
    name: str


@dataclass(frozen=True)
class MonitorRecord:
    name: str | None
    pnp_id: str | None
    manufacturer: str | None
    product: str | None
    serial: str | None
    year: int | None
    week: int | None
    current_width: int | None
    current_height: int | None
    current_refresh_hz: int | None


@dataclass(frozen=True)
class SkuRecord:
    source: str
    value: str | None
    detail: str | None
    reason: str | None


@dataclass(frozen=True)
class EnvironmentManifest:
    machine_id: str | None
    hostname: str | None
    cpu: CpuRecord | None
    gpu: GpuRecord | None
    npu: NpuRecord | None
    drivers: Mapping[str, DriverRecord]
    windows: WindowsRecord | None
    ram_bytes: int | None
    power_plan: PowerPlanRecord | None
    ac_status: bool | None
    battery_instances: int | None
    monitor: MonitorRecord | None
    python_version: str
    python_executable: str
    openvino_version: str | None
    openvino_available_devices: tuple[str, ...] | None
    sku_records: tuple[SkuRecord, ...]
    sku_note: str
    unavailable: Mapping[str, str]
    generated_at_utc: str
    collector: str

    def as_dict(self) -> dict[str, Any]:
        return _plain(self)


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _powershell() -> str:
    if os.name != "nt":
        raise RuntimeError("this is not Windows; CIM/PowerShell inventory is unavailable")
    found = shutil.which("powershell.exe") or shutil.which("powershell")
    if found:
        return found
    fallback = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("powershell.exe was not found")


#: One CIM read. It deliberately performs no writes and changes no policy; a standard
#: user can run it. ``-ExecutionPolicy`` is not passed: this must not weaken a security
#: setting to make the collector run.
_WINDOWS_SCRIPT: Final[str] = r"""
$ErrorActionPreference = 'SilentlyContinue'
$OutputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

function Date-Text($value) {
    if ($null -eq $value) { return $null }
    try { return ([datetime]$value).ToString('yyyy-MM-dd') } catch { return [string]$value }
}
function Code-Text($codes) {
    if ($null -eq $codes) { return $null }
    $text = -join ($codes | Where-Object { $_ -ne 0 } | ForEach-Object { [char]$_ })
    if ([string]::IsNullOrWhiteSpace($text)) { return $null }
    return $text
}

$cs = Get-CimInstance Win32_ComputerSystem
$csp = Get-CimInstance Win32_ComputerSystemProduct
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$os = Get-CimInstance Win32_OperatingSystem
$gpu = Get-CimInstance Win32_VideoController |
    Where-Object { $_.Name -match 'Intel|Arc|Graphics' } | Select-Object -First 1
if ($null -eq $gpu) { $gpu = Get-CimInstance Win32_VideoController | Select-Object -First 1 }
$npu = Get-CimInstance Win32_PnPEntity |
    Where-Object { $_.Name -match 'AI Boost|Neural|VPU' } | Select-Object -First 1
$signed = @(Get-CimInstance Win32_PnPSignedDriver)
$gpuDriver = $signed | Where-Object { $_.DeviceID -eq $gpu.PNPDeviceID } | Select-Object -First 1
$npuDriver = $signed | Where-Object { $_.DeviceID -eq $npu.PNPDeviceID } | Select-Object -First 1
$bios = Get-CimInstance Win32_BIOS
$reg = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
$monitor = Get-CimInstance Win32_DesktopMonitor | Select-Object -First 1
$monitorId = Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorID | Select-Object -First 1
$battery = @(Get-CimInstance Win32_Battery)
$power = (powercfg /getactivescheme 2>&1 | Out-String).Trim()

[ordered]@{
    computer_system = [ordered]@{
        model = $cs.Model; name = $cs.Name; manufacturer = $cs.Manufacturer
        system_family = $cs.SystemFamily; total_physical_memory = $cs.TotalPhysicalMemory
    }
    product = [ordered]@{
        uuid = $csp.UUID; version = $csp.Version; vendor = $csp.Vendor
        identifying_number = $csp.IdentifyingNumber
    }
    cpu = [ordered]@{
        name = $cpu.Name; manufacturer = $cpu.Manufacturer; cores = $cpu.NumberOfCores
        logical_processors = $cpu.NumberOfLogicalProcessors
        max_clock_mhz = $cpu.MaxClockSpeed
    }
    gpu = [ordered]@{
        name = $gpu.Name; pnp_id = $gpu.PNPDeviceID
        driver_version = $gpuDriver.DriverVersion
        driver_date = Date-Text $gpuDriver.DriverDate
        driver_provider = $gpuDriver.DriverProviderName; driver_inf = $gpuDriver.InfName
        adapter_ram = $gpu.AdapterRAM
        current_width = $gpu.CurrentHorizontalResolution
        current_height = $gpu.CurrentVerticalResolution
        current_refresh_hz = $gpu.CurrentRefreshRate
    }
    npu = [ordered]@{
        name = $npu.Name; manufacturer = $npu.Manufacturer; pnp_id = $npu.PNPDeviceID
        status = $npu.Status; driver_version = $npuDriver.DriverVersion
        driver_date = Date-Text $npuDriver.DriverDate
        driver_provider = $npuDriver.DriverProviderName; driver_inf = $npuDriver.InfName
    }
    gpu_driver = [ordered]@{
        version = $gpuDriver.DriverVersion; date = Date-Text $gpuDriver.DriverDate
        provider = $gpuDriver.DriverProviderName; inf = $gpuDriver.InfName
    }
    npu_driver = [ordered]@{
        version = $npuDriver.DriverVersion; date = Date-Text $npuDriver.DriverDate
        provider = $npuDriver.DriverProviderName; inf = $npuDriver.InfName
    }
    os = [ordered]@{
        caption = $os.Caption; version = $os.Version; build_number = $os.BuildNumber
        architecture = $os.OSArchitecture
        total_visible_memory_kb = $os.TotalVisibleMemorySize
    }
    registry = [ordered]@{
        product_name = $reg.ProductName; display_version = $reg.DisplayVersion
        current_build = $reg.CurrentBuild; ubr = $reg.UBR; edition_id = $reg.EditionID
    }
    bios = [ordered]@{
        manufacturer = $bios.Manufacturer; version = $bios.SMBIOSBIOSVersion
        release_date = Date-Text $bios.ReleaseDate
    }
    monitor = [ordered]@{
        name = $monitor.Name; pnp_id = $monitor.PNPDeviceID
    }
    monitor_id = [ordered]@{
        instance_name = $monitorId.InstanceName
        manufacturer = Code-Text $monitorId.ManufacturerName
        product = Code-Text $monitorId.ProductCodeID
        serial = Code-Text $monitorId.SerialNumberID
        year = $monitorId.YearOfManufacture; week = $monitorId.WeekOfManufacture
    }
    battery = [ordered]@{
        count = $battery.Count
        statuses = @($battery | ForEach-Object { [int]$_.BatteryStatus })
        charges = @($battery | ForEach-Object { [int]$_.EstimatedChargeRemaining })
    }
    power = $power
} | ConvertTo-Json -Depth 8 -Compress
"""


def _read_windows() -> dict[str, Any]:
    """Run one CIM inventory and return its JSON object. Raises on any failure."""
    executable = _powershell()
    completed = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", _WINDOWS_SCRIPT],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"PowerShell inventory exited {completed.returncode}: "
            f"{(completed.stderr or '').strip()[:400]}"
        )
    payload = (completed.stdout or "").strip()
    if not payload:
        raise RuntimeError("PowerShell inventory returned no output")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"PowerShell inventory was not JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise RuntimeError("PowerShell inventory did not return an object")
    return parsed


def _read_openvino() -> tuple[str | None, tuple[str, ...] | None, str | None]:
    """Installed version and available devices, or ``None`` with a reason."""
    try:
        version = metadata.version("openvino")
    except metadata.PackageNotFoundError:
        return None, None, "the openvino distribution is not installed in this interpreter"
    try:
        import openvino as ov

        devices = tuple(str(device) for device in ov.Core().available_devices)
    except Exception as error:  # noqa: BLE001 - a device probe must not break collection
        return version, None, f"openvino.Core() failed: {type(error).__name__}: {error}"
    return version, devices, None


def _parse_power_plan(text: Any) -> tuple[PowerPlanRecord | None, str | None]:
    if not text:
        return None, "powercfg returned no active scheme"
    match = _POWER_RE.search(str(text))
    if match is None:
        return None, f"powercfg output did not contain a scheme GUID: {str(text)[:200]!r}"
    guid, name = match.group(1), match.group(2).strip()
    try:
        uuid.UUID(guid)
    except ValueError:
        return None, f"powercfg returned a malformed GUID: {guid!r}"
    return PowerPlanRecord(guid=guid.lower(), name=name), None


def _subsystem(pnp_id: Any) -> str | None:
    match = _SUBSYSTEM_RE.search(str(pnp_id or ""))
    return match.group(1).upper() if match else None


def _driver(raw: Mapping[str, Any], key: str, reason: str | None) -> DriverRecord:
    record = raw.get(key) if isinstance(raw.get(key), Mapping) else {}
    version = _text(record.get("version"))
    date = _text(record.get("date"))
    provider = _text(record.get("provider"))
    inf = _text(record.get("inf"))
    problem = None
    if version is None and date is None:
        problem = reason or "the signed-driver record was not returned"
    return DriverRecord(version=version, date=date, provider=provider, inf=inf,
                        reason=problem)


def _sku_records(raw: Mapping[str, Any], reason: str | None) -> tuple[SkuRecord, ...]:
    """Exactly three records, always, whether or not the machine answered."""
    computer = raw.get("computer_system") if isinstance(raw.get("computer_system"), Mapping) else {}
    gpu = raw.get("gpu") if isinstance(raw.get("gpu"), Mapping) else {}
    npu = raw.get("npu") if isinstance(raw.get("npu"), Mapping) else {}

    model = _text(computer.get("model"))
    family = _text(computer.get("system_family"))
    model_value = " ".join(part for part in (model, family) if part) or None
    hostname = _text(computer.get("name"))
    gpu_subsystem = _subsystem(gpu.get("pnp_id"))
    npu_subsystem = _subsystem(npu.get("pnp_id"))
    subsystem = gpu_subsystem or npu_subsystem

    detail_reason = reason or "the machine returned no value"
    return (
        SkuRecord(
            source="Win32_ComputerSystem.Model + SystemFamily",
            value=model_value,
            detail=f"model={model!r}; system_family={family!r}",
            reason=None if model_value else detail_reason,
        ),
        SkuRecord(
            source="Win32_ComputerSystem.Name (hostname)",
            value=hostname,
            detail="the machine's hostname; this is not a model or a SKU",
            reason=None if hostname else detail_reason,
        ),
        SkuRecord(
            source="PCI SUBSYS in GPU/NPU PNPDeviceID",
            value=subsystem,
            detail=(
                f"gpu={gpu_subsystem!r}; npu={npu_subsystem!r}; the last four hex "
                f"digits are the subsystem vendor ID (1043 = ASUSTeK)"
            ),
            reason=None if subsystem else detail_reason,
        ),
    )


def collect_environment() -> EnvironmentManifest:
    """Read the machine. Never raises; unavailable fields are ``None`` with a reason."""
    unavailable: dict[str, str] = {}
    inventory_error: str | None = None
    try:
        raw = _read_windows()
    except Exception as error:  # noqa: BLE001 - collection must survive a probe failure
        raw = {}
        inventory_error = f"{type(error).__name__}: {error}"

    def missing(field: str, detail: str | None = None) -> None:
        unavailable[field] = detail or (
            f"could not read {field} from the machine: {inventory_error}"
            if inventory_error
            else f"the machine returned no value for {field}"
        )

    computer = raw.get("computer_system") if isinstance(raw.get("computer_system"), Mapping) else {}
    product = raw.get("product") if isinstance(raw.get("product"), Mapping) else {}
    cpu_raw = raw.get("cpu") if isinstance(raw.get("cpu"), Mapping) else {}
    gpu_raw = raw.get("gpu") if isinstance(raw.get("gpu"), Mapping) else {}
    npu_raw = raw.get("npu") if isinstance(raw.get("npu"), Mapping) else {}
    os_raw = raw.get("os") if isinstance(raw.get("os"), Mapping) else {}
    registry = raw.get("registry") if isinstance(raw.get("registry"), Mapping) else {}
    monitor_raw = raw.get("monitor") if isinstance(raw.get("monitor"), Mapping) else {}
    monitor_id = raw.get("monitor_id") if isinstance(raw.get("monitor_id"), Mapping) else {}
    battery = raw.get("battery") if isinstance(raw.get("battery"), Mapping) else {}
    inventory_reason = inventory_error or "the machine returned no value"

    machine_id = _text(product.get("uuid"))
    if machine_id is None:
        missing("machine_id", f"Win32_ComputerSystemProduct.UUID is unreadable: {inventory_reason}")
    else:
        machine_id = machine_id.upper()

    hostname = _text(computer.get("name"))
    if hostname is None:
        missing("hostname", f"Win32_ComputerSystem.Name is unreadable: {inventory_reason}")

    cpu = None
    if _text(cpu_raw.get("name")):
        cpu = CpuRecord(
            name=_text(cpu_raw.get("name")) or "",
            manufacturer=_text(cpu_raw.get("manufacturer")),
            cores=_int(cpu_raw.get("cores")),
            logical_processors=_int(cpu_raw.get("logical_processors")),
            max_clock_mhz=_int(cpu_raw.get("max_clock_mhz")),
        )
        if cpu.cores is None:
            missing("cpu.cores", f"Win32_Processor.NumberOfCores is unreadable: {inventory_reason}")
        if cpu.logical_processors is None:
            missing("cpu.logical_processors",
                    f"Win32_Processor.NumberOfLogicalProcessors is unreadable: {inventory_reason}")
        if cpu.max_clock_mhz is None:
            missing("cpu.max_clock_mhz",
                    f"Win32_Processor.MaxClockSpeed is unreadable: {inventory_reason}")
    else:
        missing("cpu", f"Win32_Processor.Name is unreadable: {inventory_reason}")

    gpu = None
    if _text(gpu_raw.get("name")):
        gpu = GpuRecord(
            name=_text(gpu_raw.get("name")) or "",
            pnp_id=_text(gpu_raw.get("pnp_id")),
            driver_version=_text(gpu_raw.get("driver_version")),
            driver_date=_text(gpu_raw.get("driver_date")),
            driver_provider=_text(gpu_raw.get("driver_provider")),
            driver_inf=_text(gpu_raw.get("driver_inf")),
            adapter_ram_reported_bytes=_int(gpu_raw.get("adapter_ram")),
            adapter_ram_note=(
                "Win32_VideoController.AdapterRAM is a 32-bit property and can be "
                f"capped; it reported {_int(gpu_raw.get('adapter_ram'))!r} here. The "
                f"adapter name is {_text(gpu_raw.get('name'))!r}; do not read either as "
                f"a measured VRAM figure."
            ),
            current_width=_int(gpu_raw.get("current_width")),
            current_height=_int(gpu_raw.get("current_height")),
            current_refresh_hz=_int(gpu_raw.get("current_refresh_hz")),
        )
        if gpu.driver_version is None:
            missing("gpu.driver_version",
                    f"Win32_PnPSignedDriver for the GPU is unreadable: {inventory_reason}")
        if gpu.current_width is None or gpu.current_height is None:
            missing("gpu.current_resolution",
                    f"Win32_VideoController resolution is unreadable: {inventory_reason}")
    else:
        missing("gpu", f"Win32_VideoController.Name is unreadable: {inventory_reason}")

    npu = None
    if _text(npu_raw.get("name")):
        npu = NpuRecord(
            name=_text(npu_raw.get("name")) or "",
            manufacturer=_text(npu_raw.get("manufacturer")),
            pnp_id=_text(npu_raw.get("pnp_id")),
            status=_text(npu_raw.get("status")),
            driver_version=_text(npu_raw.get("driver_version")),
            driver_date=_text(npu_raw.get("driver_date")),
            driver_provider=_text(npu_raw.get("driver_provider")),
            driver_inf=_text(npu_raw.get("driver_inf")),
        )
        if npu.driver_version is None:
            missing("npu.driver_version",
                    f"Win32_PnPSignedDriver for the NPU is unreadable: {inventory_reason}")
    else:
        missing("npu", f"no AI Boost/Neural/VPU PnP entity was returned: {inventory_reason}")

    drivers: dict[str, DriverRecord] = {
        "gpu": _driver(raw, "gpu_driver", inventory_error),
        "npu": _driver(raw, "npu_driver", inventory_error),
    }
    if drivers["gpu"].version is None:
        missing("drivers.gpu", drivers["gpu"].reason or inventory_reason)
    if drivers["npu"].version is None:
        missing("drivers.npu", drivers["npu"].reason or inventory_reason)

    windows = None
    if _text(os_raw.get("caption")) or _text(os_raw.get("version")):
        caption = _text(os_raw.get("caption"))
        registry_product_name = _text(registry.get("product_name"))
        product_name_note = None
        if caption and registry_product_name and caption != registry_product_name:
            product_name_note = (
                f"Win32_OperatingSystem.Caption ({caption!r}) and the registry "
                f"ProductName ({registry_product_name!r}) disagree on this machine; both "
                f"are recorded and neither is corrected."
            )
        windows = WindowsRecord(
            caption=caption,
            version=_text(os_raw.get("version")),
            build_number=_text(os_raw.get("build_number")),
            ubr=_int(registry.get("ubr")),
            edition_id=_text(registry.get("edition_id")),
            display_version=_text(registry.get("display_version")),
            registry_product_name=registry_product_name,
            product_name_note=product_name_note,
            architecture=_text(os_raw.get("architecture")),
        )
        if windows.ubr is None:
            missing("windows.ubr",
                    f"registry UBR is unreadable: {inventory_reason}")
    else:
        missing("windows", f"Win32_OperatingSystem is unreadable: {inventory_reason}")

    ram_bytes = _int(computer.get("total_physical_memory"))
    if ram_bytes is None:
        missing("ram_bytes", f"Win32_ComputerSystem.TotalPhysicalMemory is unreadable: {inventory_reason}")

    power_plan, power_reason = _parse_power_plan(raw.get("power"))
    if power_plan is None:
        missing("power_plan", f"{power_reason}; {inventory_reason}" if inventory_error else power_reason)

    battery_instances = _int(battery.get("count"))
    ac_status: bool | None = None
    if battery_instances is None:
        missing("battery_instances", f"Win32_Battery instance count is unreadable: {inventory_reason}")
        missing("ac_status", f"Win32_Battery instance count is unreadable: {inventory_reason}")
    elif battery_instances == 0:
        missing(
            "ac_status",
            "Win32_Battery returned 0 instances on this machine: there is no battery, so "
            "AC/battery status cannot be read and is None rather than a guess",
        )
    else:
        statuses = [value for value in battery.get("statuses", []) if isinstance(value, int)]
        if 2 in statuses:
            ac_status = True
        elif statuses:
            ac_status = False
        else:
            missing("ac_status", "Win32_Battery reported no BatteryStatus value")

    monitor = None
    monitor_name = _text(monitor_raw.get("name"))
    manufacturer = _text(monitor_id.get("manufacturer"))
    product_code = _text(monitor_id.get("product"))
    if monitor_name or manufacturer or product_code:
        monitor = MonitorRecord(
            name=monitor_name,
            pnp_id=_text(monitor_raw.get("pnp_id")),
            manufacturer=manufacturer,
            product=product_code,
            serial=_text(monitor_id.get("serial")),
            year=_int(monitor_id.get("year")),
            week=_int(monitor_id.get("week")),
            current_width=_int(gpu_raw.get("current_width")),
            current_height=_int(gpu_raw.get("current_height")),
            current_refresh_hz=_int(gpu_raw.get("current_refresh_hz")),
        )
        if monitor.current_width is None or monitor.current_height is None:
            missing("monitor.current_resolution",
                    f"the current display mode is unreadable: {inventory_reason}")
    else:
        missing("monitor", f"Win32_DesktopMonitor/WmiMonitorID are unreadable: {inventory_reason}")

    openvino_version, devices, openvino_reason = _read_openvino()
    if openvino_version is None:
        missing("openvino_version", openvino_reason)
    if devices is None:
        missing("openvino_available_devices", openvino_reason or "no device list was returned")

    return EnvironmentManifest(
        machine_id=machine_id,
        hostname=hostname,
        cpu=cpu,
        gpu=gpu,
        npu=npu,
        drivers=drivers,
        windows=windows,
        ram_bytes=ram_bytes,
        power_plan=power_plan,
        ac_status=ac_status,
        battery_instances=battery_instances,
        monitor=monitor,
        python_version=platform.python_version(),
        python_executable=sys.executable,
        openvino_version=openvino_version,
        openvino_available_devices=devices,
        sku_records=_sku_records(raw, inventory_error),
        sku_note=SKU_NOTE,
        unavailable=unavailable,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        collector=COLLECTOR,
    )


def write_manifest(path: Path, manifest: EnvironmentManifest) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--print", action="store_true", dest="print_only")
    args = parser.parse_args(argv)
    manifest = collect_environment()
    if args.print_only:
        print(json.dumps(manifest.as_dict(), indent=2, sort_keys=False))
    else:
        print(f"wrote {write_manifest(args.output, manifest)}")
        print(f"unavailable: {', '.join(sorted(manifest.unavailable)) or 'none'}")
        print(f"sku records: {len(manifest.sku_records)} (preserved, no winner)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
