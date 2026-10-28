"""The LUID-to-PCI-function link DXGI exposes, over `ctypes`. No `pywin32`, no WMI.

**Why this module exists.** A `GPU Engine` counter instance identifies its adapter only by
a LUID:

    pid_6588_luid_0x00000000_0x000113B4_phys_0_eng_5_engtype_Compute

`config/telemetry_map.json` recorded LUID -> device on 2026-10-01, and those LUIDs are
assigned per boot: measured on 2026-10-27, the same machine reports `0x000113B4` for the
Arc GPU and `0x000117B6` for the NPU, while the file still says `0x00010E9E` and
`0x000112A0`. So a runtime that matched against the recorded file would find nothing on
this boot, and a runtime that guessed "the LUID next to the NPU's" is the mistake the
instance map exists to prevent.

DXGI gives the missing link directly. `IDXGIAdapter::GetDesc` returns
`DXGI_ADAPTER_DESC{Description, VendorId, DeviceId, ..., AdapterLuid}`, so the Arc adapter
is identified as **Intel, PCI device `0x64A0`, LUID `0x000113B4`** - the same
`VEN_8086&DEV_64A0` that `Win32_VideoController` and the driver row report. That is an
identity from the driver, not a taxonomy inference.

Measured on this machine, 2026-10-27, two adapters:

    0  'Intel(R) Arc(TM) 140V GPU (16GB)'  vendor 0x8086  device 0x64A0  luid 0x113B4
    1  'Microsoft Basic Render Driver'     vendor 0x1414  device 0x008C  luid 0x1174C

The second is Windows' software renderer: it reports `3D` engine instances for many
processes and must never be presented as the GPU. `DxgiAdapter.software` is true for
vendor `0x1414` (Microsoft), which is how the provider excludes it. If more than one
**hardware** adapter is present the provider refuses rather than picking one.

**The NPU is not a DXGI adapter.** It has no display function, so DXGI does not enumerate
it; the NPU is identified from the one adapter LUID that exposes an `engtype_Neural` engine
(see `app.telemetry.accelerator`). A GPU answer never implies an NPU answer.

Failure is `()`, never a fabricated adapter: this module is called once at construction,
and an unreachable `dxgi.dll` means the GPU's LUID is unknown, so the GPU figure is
unavailable with a stated reason.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any, Final

__all__ = ["DXGI_DLL", "DxgiAdapter", "adapters", "hardware_adapters"]

DXGI_DLL: Final[str] = "dxgi.dll"

#: `IID_IDXGIFactory` = {7B7166EC-21C7-44AE-B21A-C9AE321AE369}. `CreateDXGIFactory1` with
#: the *Factory1* IID returned `E_NOINTERFACE` (0x80004002) on this machine; the base
#: factory IID works, and `EnumAdapters`/`GetDesc` are enough for the LUID.
_MAX_ADAPTERS: Final[int] = 16

#: Microsoft's vendor id. The Basic Render Driver (WARP) reports it, and it is not a
#: physical GPU: measured description 'Microsoft Basic Render Driver', device 0x008C.
_SOFTWARE_VENDOR: Final[int] = 0x1414


class _Luid(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", ctypes.c_long)]


class _AdapterDesc(ctypes.Structure):
    """`DXGI_ADAPTER_DESC`. Measured: the description is 128 UTF-16 characters, and
    `AdapterLuid` follows three `SIZE_T` memory fields."""

    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _Luid),
    ]


class _Guid(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]


@dataclass(frozen=True)
class DxgiAdapter:
    """One enumerated adapter: its driver name, PCI ids and LUID.

    `luid` is the **low word** of `AdapterLuid`, spelled `0x%08X`, which is the second
    word of a `GPU Engine` instance name. `raw_luid` keeps the full 64-bit value for a
    report. `software` is true only for Microsoft's renderer.
    """

    description: str
    vendor_id: int
    device_id: int
    luid: str
    raw_luid: int

    @property
    def software(self) -> bool:
        return self.vendor_id == _SOFTWARE_VENDOR


def _bind(lib: Any) -> Any:
    """`CreateDXGIFactory` with explicit argtypes/restype. Raises on a missing export."""
    create = lib.CreateDXGIFactory
    create.argtypes = [ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    return create


def adapters() -> tuple[DxgiAdapter, ...]:
    """Every DXGI adapter, or `()` when DXGI cannot be reached here.

    Never raises: a machine with no `dxgi.dll`, a failed factory, or a failed enumeration
    is a machine whose GPU LUID is unknown, and the caller turns that into a stated reason.
    """
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        return ()
    try:
        lib = loader(DXGI_DLL)
        create = _bind(lib)
    except (OSError, AttributeError):
        return ()
    factory = ctypes.c_void_p()
    iid = _Guid(0x7B7166EC, 0x21C7, 0x44AE,
                (ctypes.c_ubyte * 8)(0xB2, 0x1A, 0xC9, 0xAE, 0x32, 0x1A, 0xE3, 0x69))
    if create(ctypes.byref(iid), ctypes.byref(factory)) != 0 or not factory.value:
        return ()
    vtable = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    # IDXGIObject contributes 7 slots (IUnknown 0-2, SetPrivateData..GetParent 3-6), so
    # IDXGIFactory::EnumAdapters is 7; IDXGIAdapter::GetDesc is 8 on the adapter's own
    # vtable (EnumOutputs 7, GetDesc 8, CheckInterfaceSupport 9). Measured working.
    enum = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.UINT,
                            ctypes.POINTER(ctypes.c_void_p))(vtable[7])
    release = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
    found: list[DxgiAdapter] = []
    try:
        for index in range(_MAX_ADAPTERS):
            adapter = ctypes.c_void_p()
            if enum(factory, index, ctypes.byref(adapter)) != 0 or not adapter.value:
                break
            try:
                desc = _AdapterDesc()
                adapter_vtable = ctypes.cast(
                    adapter, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
                )[0]
                get_desc = ctypes.CFUNCTYPE(
                    ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_AdapterDesc)
                )(adapter_vtable[8])
                status = get_desc(adapter, ctypes.byref(desc))
                if status != 0:
                    continue
                raw = ((desc.AdapterLuid.HighPart & 0xFFFFFFFF) << 32
                       | (desc.AdapterLuid.LowPart & 0xFFFFFFFF))
                found.append(DxgiAdapter(
                    description=str(desc.Description or ""),
                    vendor_id=int(desc.VendorId),
                    device_id=int(desc.DeviceId),
                    luid=f"0x{desc.AdapterLuid.LowPart & 0xFFFFFFFF:08X}",
                    raw_luid=raw,
                ))
            finally:
                release(adapter)
    finally:
        release(factory)
    return tuple(found)


def hardware_adapters(items: tuple[DxgiAdapter, ...] | None = None) -> tuple[DxgiAdapter, ...]:
    """The adapters that are not Microsoft's software renderer.

    **Empty means unknown**, not "no GPU": a machine where DXGI failed and a machine with
    only the software renderer are different facts, and the caller has the original tuple
    to tell them apart if it needs to.
    """
    pool = adapters() if items is None else items
    return tuple(a for a in pool if not a.software)
