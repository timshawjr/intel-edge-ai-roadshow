# GPU and NPU utilisation: per-process attribution, measured

**M4 Task 5 — closing M0's unmet gate 2.**
Machine: `NUC14ProAI`, Windows 11 Pro 10.0.26200. GPU driver **32.0.101.6737**, NPU driver
**32.0.100.5540**. Measured 2026-10-27.

---

## 1. The question, and the answer

M0 recorded that a GPU/NPU utilisation figure could not be put on the booth screen, for two
separate reasons:

1. the in-process PDH value read terminated the process with `0xC0000005`, and
2. the routes that did work (`Get-Counter`, CIM) returned a **machine-wide peak**, which is
   not attributable to this application. A 62% / 68% figure was withdrawn because this
   application's PID appeared in **none of the 469 GPU Engine instances** M0 measured in one
   query, and the busiest engines belonged to unrelated `pythonw.exe` processes.

**The answer is: the figure can be attributed, for both the GPU and the NPU, and this
milestone ships it.** The counter is `\GPU Engine(*)\Utilization Percentage`, which is
per-process **by construction** — every instance name carries `pid_<n>`, the adapter LUID
and the engine type:

```
pid_13728_luid_0x00000000_0x000113B4_phys_0_eng_5_engtype_Compute    97.06-100.0 %
pid_15624_luid_0x00000000_0x000117B6_phys_0_eng_0_engtype_Neural      63.76-66.08 %
```

Reason 1 was not a property of PDH. It was a **three-argument binding of a four-argument
function** in `app/telemetry/pdh.py`. With the documented signature the same call returns
real values in this process, and the same module's array reader returns names and
percentages together. Section 4 records the two measurements side by side, because the
difference between them is the whole of that finding.

Reason 2 still stands, and is why the provider **never** returns a machine-wide figure: it
filters every instance by its own `pid_` field to this application's processes and by the
adapter LUID to the identified device. The 62%/68% figure is **not** reinstated.

---

## 2. What "attributed" means here

A figure is attributable, and this provider publishes it, only when all of the following
hold:

- **Per process.** The instance's own `pid_` field must be in this application's process
  set (the UI process and its workers). There is no code path that reads an instance
  belonging to another process, and no path that reads an aggregate.
- **Per identified adapter.** The GPU is the **one hardware adapter DXGI enumerates**;
  Microsoft's Basic Render Driver is excluded by vendor id `0x1414`. The NPU is the **one
  adapter LUID exposing an `engtype_Neural` engine**. Zero or several candidates is `None`
  with a reason — never "the nearest one".
- **A stated denominator and meaning.** `Utilization Percentage` is the fraction of the
  interval between two counter samples during which **that engine** was busy. The figure
  published is the **busiest engine** of that device for this application: *"at least one
  engine of this device was busy for this application for X% of the interval"*. It is
  **never a sum**: two engines can run at once, so a sum can exceed 100 and stops meaning
  time.
- **Demonstrably not another process's.** The control in section 3 shows the figure move
  with this process's own load, and a second busy process not appear under our PID.
- **`None` is never `0`.** A process with no engine instance on a device has not been
  observed using it: the sample is `UNAVAILABLE` with that reason. A process with an
  instance at `0.0%` has a measured zero, and `0.0%` is published as a reading.

---

## 3. The control: idle → busy, and a distractor

`tools/probe_utilisation.py` spawns a worker process, leaves it idle for 5 s, then drives
OpenVINO inference on `GPU` (and separately `NPU`) for 15 s. A provider samples **that
worker's pid**. A second provider samples **the probe's own pid** at the same time. The
machine is asserted clean before any worker starts.

Command and output (2026-10-27, this machine):

```
$ .venv\Scripts\python.exe tools\probe_utilisation.py
machine: NUC14ProAI  Windows-11-10.0.26200-SP0
GPU driver: 32.0.101.6737   NPU driver: 32.0.100.5540
  adapter 0x000113B4  Intel(R) Arc(TM) 140V GPU (16GB)  PCI 0x8086:0x64A0
  adapter 0x0001174C  Microsoft Basic Render Driver  PCI 0x1414:0x008C  [software]

baseline: no worker started; the machine must be clean
  349 live instances, busiest 0.00% -> CLEAN

GPU: worker idle 5s, then busy 15s
  idle  n=5 min=0.0 median=0.0 max=0.0 absent=0
  busy  n=26 min=97.06 median=97.19 max=100.0 absent=0
  probe's own pid while that worker was busy: n=28 max=0.0 absent=0
     100.00%  pid_13728_luid_0x00000000_0x000113B4_phys_0_eng_5_engtype_Compute
       1.42%  pid_13116_luid_0x00000000_0x000113B4_phys_0_eng_0_engtype_3D
       0.00%  pid_11516_luid_0x00000000_0x000113B4_phys_0_eng_0_engtype_3D

NPU: worker idle 5s, then busy 15s
  idle  n=5 min=0.0 median=0.0 max=0.0 absent=0
  busy  n=26 min=63.76 median=64.86 max=66.08 absent=0
  probe's own pid while that worker was busy: n=28 max=0.0 absent=0
      66.97%  pid_15624_luid_0x00000000_0x000117B6_phys_0_eng_0_engtype_Neural
       2.79%  pid_13116_luid_0x00000000_0x000113B4_phys_0_eng_0_engtype_3D

  verdict GPU: busy=True idle=True distractor=True -> PASS
  verdict NPU: busy=True idle=True distractor=True -> PASS
VERDICT: attributed - the figure tracks this process
```

This is the run committed at `docs/qualification/m4-utilisation-control.json`
(`measured_at` 2026-10-27T15:29:24). Repeated runs measured a GPU busy median of
97.19-97.28% and an NPU busy median of 64.61-65.06%, with idle 0.0% every time: the *levels*
are stable, the exact median depends on host scheduling, which is why the card quotes this
committed run rather than a lifetime figure.

What that output establishes, and why each part is needed:

| observation | what it proves |
|---|---|
| idle median **0.0%** on the same pid, busy median **97.19%** GPU / **64.86%** NPU | the figure **moves when this process's load moves**. A stale, cached or machine-wide number would not. |
| the worker's `engtype_Compute` row at **100.00%** while the probe's own pid reads **0.0%** | the busy worker is **not attributed to the probe**. A machine-wide maximum or sum would have put ~97% on the probe's card. |
| baseline busiest engine **0.00%**, no worker running, 349 live instances | the machine was clean before the measurement; an idle reading could not be confused with unrelated load. |
| the 3D engine on the **same** LUID (`pid_13116`, 1.42%) | another process on the **same adapter** is present in the sample and excluded by the pid filter. Filtering by adapter alone would have included it. |
| the software renderer's LUID `0x0001174C` is absent from the GPU split | the GPU is the Arc, not the Basic Render Driver that reports 3D instances for many processes. |

The worker was **idle** for the first samples with engine instances already present, and
read exactly `0.0`. That is the case a `None`-vs-`0` decision has to get right: those
samples are **measured zeros**, not absences. The absences are the other case — before the
worker opens an engine, its pid does not appear at all, and the provider returns `None`
with the reason *"this application has no GPU engine instance in this counter sample"*.

---

## 4. Every route tried, what it returned, and its failure mode

### Route 1 — Windows Performance Counters, in-process through `pdh.dll` (shipped)

**Counter set / path.** `GPU Engine` is a counter **set**, not a counter:
`PdhAddEnglishCounterW(query, "GPU Engine", …)` returns `0xC0000BC0` and a NULL handle. The
addable path is `\GPU Engine(*)\Utilization Percentage`. Value entry points:
`PdhGetFormattedCounterValue` (one counter) and `PdhGetFormattedCounterArrayW` (all
instances, names and values together).

**What it returned in M0.** `PdhGetFormattedCounterValue` killed the process with
`0xC0000005` — including on `\Memory\Available Bytes` and
`\Processor(_Total)\% Processor Time`, which is what made it look like a PDH defect.

**The actual failure mode, measured 2026-10-27.** The binding passed **three** arguments to
a **four**-argument function:

```
old: PdhGetFormattedCounterValue(hCounter, pValue, lpType)              -> 0xC0000005
     ("access violation reading 0xFFFFFFFFFFFFFFFF" — the exact M0 message)
new: PdhGetFormattedCounterValue(hCounter, PDH_FMT_DOUBLE, lpType, pValue)
     -> rc = 0, CStatus = 0, \Memory\Available Bytes = 25,560,748,032.0
```

Both are pinned by tests that run the call in a **child** process, so the failure is
observable and a native fault cannot take the suite down
(`tests/test_pdh.py::RealPdhValueRead`). The array reader had the same class of error: a
`PDH_FMT_COUNTERVALUE_ITEM_W` is **24 bytes** (8-byte name pointer + 16-byte value), and it
was walked at a 16-byte stride. Measured with the real stride: **335 items**, `rc = 0`,
`pid_11860_…_eng_5_engtype_Compute` at **97.91%**, names paired with values.

**Per-engine sum.** The instances are per **engine**, not per device. On this machine the
Arc exposes `3D`, `Compute`, `Copy`, `GSC`, `VideoDecode`, `VideoProcessing`; the NPU
exposes `Neural`. Summing engines **double-counts parallel work** and can exceed 100, so the
provider publishes the busiest engine and says so. Measured during a GPU worker's busy
phase the sum and the maximum were both ~98 because only one engine was active; they are
not interchangeable in general, and the code does not rely on that coincidence.

The first manual read of the busy worker after the binding fix returned **97.91%** on
`pid_6588_…_eng_5_engtype_Compute`; the committed idle/busy control (section 3) measured a
busy median of **97.19%** (97.06-100.0%) and an NPU median of **64.86%** (63.76-66.08%).
Both are measurements of the same mechanism; the control is the number the card text
quotes.

**Verdict: usable, shipped.** In-process, stdlib `ctypes`, no subprocess, no new dependency.
Cost: **0.37-0.84 ms** for 335 items per sample (n=30: p50 0.38, p95 0.72), inside a
250 ms budget. The provider's `samples()` (one array read, device resolution and the pid
filter for both devices) measured **1.77-3.38 ms** (n=30: p50 1.81, p95 2.83).

### Route 2 — PDH through a helper process

**Why it was considered.** If the in-process call were genuinely unusable, a helper could
read the counter in a process that is allowed to die.

**What it was not needed for.** The in-process call works once bound correctly (route 1), so
no helper is shipped. A helper would add a second process on the 4 Hz path, a spawn cost,
and a new failure mode (a helper that dies is a silent absence), for no gain.

**Failure mode if used.** A helper returns *whatever it was told to return*; the
attribution rule would have to be implemented inside it and audited there. Keeping the
filter in this process, next to the pid set, is the stronger arrangement.

### Route 3 — `Get-Counter` / CIM

**What it returns.** Also **per-PID**: `Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine`
and `Get-Counter '\GPU Engine(*)\Utilization Percentage'` expose the same instance names
(`pid_<n>_luid_…_engtype_…`). Measured: 335 instances grouped by pid. So this is route 1
with different plumbing, not a different attribution story.

**Failure mode.** It is not a Python API here: the only stdlib route is spawning
PowerShell, which costs a process launch per sample and depends on script execution being
allowed under the demo machine's policy. **Measured cost on this machine** (three calls
each): `Get-Counter '\GPU Engine(*)\Utilization Percentage'` **1099-1458 ms** per call, and
`Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine` **329-335 ms**
warm, **3185 ms** cold. Both are outside the 250 ms sampling budget, and a
process-per-sample is not a telemetry path. It was used as an **independent cross-check** of
the counter values, not as the shipped provider.

**Verdict: correct but unusable at 4 Hz; not shipped.**

### Route 4 — Intel-specific counters

**What was looked for.** Counter sets and CIM classes whose names contain `Intel`, `NPU`,
`VPU`, `Neural` or `Accel`, and every GPU counter set on the machine.

**What exists** (`Get-Counter -ListSet '*'`, measured):

```
GPU Engine                       The running time of each gpu engine
GPU Adapter Memory               The memory usage of each adapter
GPU Local Adapter Memory         The local memory usage of each adapter
GPU Non Local Adapter Memory     The non local memory usage of each adapter
GPU Process Memory               The memory usage of each process
RemoteFx Root GPU Management     Displays RemoteFx Root GPU Management Counters
```

No Intel-named counter set and no NPU/VPU-specific counter set exists on this build. There
is **no** per-process Intel counter to prefer over `GPU Engine`.

**Verdict: does not exist here; nothing to ship.**

### Route 5 — NPU utilisation specifically

**What exists.** The NPU has no counter set of its own, but it **does** appear in `GPU
Engine` as `engtype_Neural` instances, one per process, under its own adapter LUID. Measured
on this machine: LUID `0x000117B6` exposes `pid_…_eng_0_engtype_Neural` for many processes,
and an NPU inference worker's own instance read **63.76-66.08%** (median **64.86%**) while
idle it read **0.0%**.

**Failure mode it avoids.** The NPU is **not** a DXGI adapter, so the GPU answer cannot
identify it and is not allowed to imply one. It is identified independently, as the one
adapter exposing a Neural engine. If zero or two adapters expose one, the NPU is `None`
with the reason.

**Denominator.** The Neural engine's busy fraction of the sample interval. The measured
~65% under a tight inference loop is the engine's own busy time, not a claim that the NPU is
incapable of more: the loop spends host time submitting and copying between inferences.

**Verdict: usable, shipped.**

---

## 5. How the devices are identified (and why the recorded file is not enough)

`config/telemetry_map.json` recorded LUID → device on **2026-10-01**: GPU `0x00010E9E`,
NPU `0x000112A0`, third adapter `0x00011232`. **Those LUIDs are stale.** Windows assigns
adapter LUIDs per boot; this boot reports:

```
$ .venv\Scripts\python.exe -c "from app.telemetry.dxgi import adapters; print(adapters())"
DxgiAdapter(description='Intel(R) Arc(TM) 140V GPU (16GB)', vendor_id=32902,
            device_id=25760, luid='0x000113B4', raw_luid=70580)
DxgiAdapter(description='Microsoft Basic Render Driver', vendor_id=5140,
            device_id=140, luid='0x0001174C', raw_luid=71500)
```

A runtime that matched the recorded file would find nothing, and a runtime that guessed
would be the mistake the instance map exists to prevent. DXGI's
`IDXGIAdapter::GetDesc` returns `Description`, `VendorId`, `DeviceId` and `AdapterLuid`
**together** — so the Arc is identified as Intel `8086:64A0` at LUID `0x000113B4`, the same
PCI function `Win32_VideoController` and the driver table report. That supersedes
`direct_luid_link.available: false` in the recorded file ("no CIM or Win32 source joins a
LUID to a PCI function"), which is now known to be wrong.

`config/telemetry_map.json` is verified data and **was not modified** by this task. The
runtime no longer depends on it for attribution; `tools/preflight.py` reports the stale
LUIDs and the DXGI link explicitly.

---

## 6. What is still unavailable, and stays unavailable

These are `None` with a reason, never `0`:

| figure | why it is not published |
|---|---|
| GPU utilisation when this application has no engine instance on the Arc | no instance to read; an absence of evidence is not an idle device |
| NPU utilisation when no adapter exposes a Neural engine, or two do | the NPU cannot be identified without guessing |
| GPU utilisation when DXGI reports zero or several hardware adapters | the Arc cannot be picked out; a nearest match is a guess |
| **any** machine-wide GPU/NPU utilisation | not attributable to this application; this is the failure mode the project exists to prevent |
| per-process VRAM | no counter read here reports per-process accelerator memory; `psutil` RSS is host memory and is never relabelled |
| NPU driver-level utilisation outside the Windows counter | no separate NPU counter set exists on this build |

## 7. Why the 62% / 68% figure remains withdrawn

That figure was a machine-wide peak. This application's PID appeared in **none of the 469
GPU Engine instances** in M0's query, and the busiest engines belonged to unrelated
`pythonw.exe` processes. Nothing
about route 1 changes that finding: the new figure is a **different claim**, produced by
filtering instances to this application's pids. The old figure is not reinstated, and the
cards do not quote a peak.

## 8. Reproduction

```powershell
# The control: idle -> busy on GPU and NPU, with a distractor pid. Writes
# docs/qualification/m4-utilisation-control.json and exits 0 only if all three
# assertions pass for both devices.
.venv\Scripts\python.exe tools\probe_utilisation.py

# The value-read control (old binding faults, new binding returns a value) and the
# array reader, in a child process so a fault is observable.
.venv\Scripts\python.exe -m unittest tests.test_pdh.RealPdhValueRead -v

# The provider's decision logic and the collector wiring.
.venv\Scripts\python.exe -m unittest tests.test_collector.AcceleratorAttribution -v
```

The preflight gate reports the live split and the instance count in the row
`Telemetry: per-process GPU/NPU utilisation`.

## 9. What shipped

- `app/telemetry/pdh.py` — corrected the four-argument value read, the 24-byte array item
  stride, and added `PdhQuery.read_array()` (one collect + one array read per sample).
  `VALUE_READS_ENABLED_BY_DEFAULT` is `True`; a disabled query still refuses with a reason.
- `app/telemetry/dxgi.py` — DXGI adapter enumeration: LUID, PCI ids, description, software
  flag.
- `app/telemetry/accelerator.py` — `AcceleratorUtilisation`, which filters to this
  application's pids and identified device and produces the two `MetricSample`s.
- `app/telemetry/collector.py` — an optional `extra_samples` producer, measured on the
  telemetry thread and published with each set. The collector still imports no PDH code.
- `app/supervisor/runner.py` — builds the provider over the same pid set as the collector
  and closes it on rebuild/stop.
- `app/ui/bridge.py` — availability derived from the sample, per-device reasons on the
  cards, and the measured-method text.
- `tools/probe_utilisation.py`, `docs/qualification/m4-utilisation-control.json` — the
  control and its evidence.
- Tests in `tests/test_pdh.py`, `tests/test_collector.py`, `tests/test_ui_contract.py`.
