# Presenter guide — Intel Edge AI Roadshow

This is the guide for the person standing at the screen. It is checked against the code and
the qualification record by `tests/test_presenter_guide.py`, so every command below is a
command the application accepts, and every number is one this project measured.

Read the **three open gates** before the doors open. A presenter who says the screen reads
at three metres, or that the shelf-to-basket story is running, will be contradicted by the
room. Qualifying the demo is part of the demo.

The package is **unsigned**. There is no certificate on this machine and no signature on
the bundle; read `docs/install/SIGNING.md` before you tell anyone what the package is.
Nothing in this guide asks you to weaken a security control.

## 1. Before the doors open

The self-test opens no window and no network, and exits 0 when every stage works:

```powershell
.venv\Scripts\python.exe -m app.main --selftest
```

The booth screen, with the release gate on (the default; `--release` makes it explicit):

```powershell
.venv\Scripts\python.exe -m app.main --windowed --release
```

The three-camera layout — one camera per available device, each with its own clip, decode
thread, tracker and event machine — names the first camera's device with `--device`:

```powershell
.venv\Scripts\python.exe -m app.main --windowed --streams 3 --device NPU
```

A presenter-driven move is `--select`, repeatable, and it changes a badge only after a
placement is **confirmed**:

```powershell
.venv\Scripts\python.exe -m app.main --windowed --select A=GPU
```

A headless run is for measurement, not for the booth. It writes the summary you can read
afterwards:

```powershell
.venv\Scripts\python.exe -m app.main --headless --device GPU --seconds 20 --json logs\run.json
.venv\Scripts\python.exe -m app.main --headless --scenario retail --source once --device NPU --seconds 30 --json logs\once.json
```

### `--streams` is refused in headless, on purpose

`--streams` is not silently ignored by `--headless`: it is refused before a run starts. The
application says it **is not supported by `--headless`**, because the headless run builds
one supervisor for one stream and its summary is single-stream, so accepting the flag would
run one camera while claiming three. Use `--windowed` for the multi-camera layout.

```text
--streams 3 is not supported by --headless: the headless run builds one supervisor for one
stream and its summary is single-stream, so accepting this flag would run one camera while
claiming 3. Use --windowed for the multi-camera layout, or omit --streams.
```

So a command like the one below exits 2 and runs nothing.

```powershell
.venv\Scripts\python.exe -m app.main --headless --streams 3
```

### A fault-injection aid, for rehearsal only

`--fail-stream` makes one camera's worker really die before compiling, so the failure path
can be observed on the live screen. It is a **QUALIFICATION AID, not a booth feature**: it
injects the same fault `Supervisor`'s own test seam injects, so what you see is the real
detect-a-dead-worker path and not a mock. Run it in a rehearsal, not while a visitor is
watching.

```powershell
.venv\Scripts\python.exe -m app.main --windowed --streams 3 --fail-stream C
```

### The developer-only escape hatch

`--no-release` runs without the simulated-provider gate. It exists for a developer who
needs a simulated provider on their own machine, and it is deliberately awkward to type.
It is not a booth command:

```powershell
.venv\Scripts\python.exe -m app.main --no-release --windowed
```

## 2. Recorded mode

`--play-record` plays a saved `BenchmarkRecord` JSON instead of running the pipeline. No
camera, model or device is opened on this path. The screen says, in full and for as long as
the recording plays, `Recorded demonstration — metrics are not live`. Every figure carries
the recording's own timestamp and offset, and the live device controls are **disabled** — a
real click emits nothing, so the recording cannot be operated as a live screen. Leave the
recording to move a camera.

```powershell
.venv\Scripts\python.exe -m app.main --play-record docs\qualification\m5-records\candidate.json
```

A playback is refused while live work is running, and it is refused with `--headless`,
`--selftest`, `--select` or `--streams` rather than quietly combined with them.

## 3. The three open gates

These are **not** closed. They belong in what you say to a customer, not only in the
qualification report.

**Failed — content: 3 of 12 clips, 0 rights-cleared.** The specification asks for four
scenario packs of three 45–90 s clips each, twelve in all, with a recorded redistribution
position. What is provisioned is one pack and three clips, only one of them inside the
45–90 s band, and none rights-cleared: no licence text is embedded and no licence file sits
beside any clip. The shelf-to-basket scenario is **not provisioned**. Do not describe the
demo as a shelf-to-basket demonstration.

**Failed at 3 m — legibility on the monitor actually attached.** On the actual DELL
S3422DWG (80 x 33 cm, a 2293x960 desktop at 150 %, 0.310 mm per logical pixel), the 44 px
body text is 16.8′ of cap height at 2 m and 11.2′ at 3 m; its x-height is 12.2′ at 2 m and
8.1′ at 3 m. It passes at 2 m on cap height and **fails at 3 m**; the mixed-case sentences
fail at both distances. A 55-inch 1080p panel would give 22.9′ / 16.6′ at 3 m, but that is
a **procurement requirement**, not the delivered result. If the booth has this monitor,
say the screen is read at 2 m.

**Missed — the 2 s warm transition.** Changing a camera's device is a warm transition, and
the target is 2 s. M2 measured p50 **2.238 s**; the committed full-deadline file's p50 is
**2.261 s** and the committed 2 s-gate file's p50 is **2.628 s**. **37 of 50** toggles were
over 2 s at the real 2.0 s default, with 25 of 50 over at the widened 30 s deadline. The
target is **missed**; do not tell a visitor the switch is instant.

## 4. What the numbers mean, and what they do not

**NPU latency has two figures, and they are not interchangeable.** The NPU's inference is
**5.50–5.57 ms cold** — the first 30 inferences in a fresh process — against **3.96–4.01 ms
settled** in a 20 s sustained window. The **5.2–5.9 ms** figure quoted in earlier documents
is the **cold** number. Never show it beside a sustained figure or a warmed-up screen.

**Throughput at the booth is set by the source, not by the device.** The booth feed runs at
15 fps, and at 15 fps all three devices delivered 300 of 300 frames. So the booth's own
stream **ranks nothing** between CPU, GPU and NPU: throughput is set by the source. The
only condition that orders the two accelerators is isolated sustained inference — GPU
307–310 fps against NPU 246–249 fps — and that is **not** a statement about the booth
stream, which never asks for more than 15 fps.

**The retail event is people-in-zone presence, 2 per clip.** It is not a people count and
not a dwell measurement. The screen can say that a person was present in a zone; it cannot
say how many people were there or how long they stayed.

## 5. When something goes wrong

Run the self-test first; it is the gate that works with no window and no network. The
preflight table is the next read:

```powershell
.venv\Scripts\python.exe -m app.main --selftest
.venv\Scripts\python.exe tools\preflight.py
```

`preflight` is expected to report **17 PASS, 1 WARN, 0 FAIL** on this machine. The WARN is
the recorded LUID-to-device row, not a device failure: Windows assigns adapter LUIDs per
boot and the runtime resolves the split from DXGI.

If a camera fails, the supervisor keeps the other cameras running and the failed camera's
card shows its own reason. That is the intended behaviour, not a crash to hide. For a
comparison of two saved runs, use the module form — the tool's own docstring prints a form
that does not run, because it omits `-m` and `tools/` is therefore not on `sys.path`:

```powershell
python -m tools.compare_records docs\qualification\m5-records\baseline.json docs\qualification\m5-records\candidate.json
```

Do **not** use the docstring's script path; it fails with
`ModuleNotFoundError: No module named 'app'`. The working form is `python -m
tools.compare_records`.

To check the footage inventory and its rights position:

```powershell
.venv\Scripts\python.exe -m tools.content_gate
```

It exits 1 while the content gate is unmet. That is the correct exit code for this tree,
not a fault to work around.

## 6. What you may and may not say

- **May say:** three cameras run at once, each on a named device, each with its own clip
  and its own event machine; a placement badge changes only after the placement is
  confirmed; the self-test passes with no network.
- **May say:** at the booth's 15 fps the devices all deliver the source's frames; the
  interesting difference is latency, not throughput.
- **May say:** the retail event is people-in-zone presence, 2 per clip.
- **Must not say:** the screen reads at 3 m on this monitor; the content gate is met; the
  shelf-to-basket scenario is provisioned; the 2 s transition target is met; the NPU
  sustains 5.2–5.9 ms; one device is faster than another on the booth stream; the package
  is signed or certified.
- **Must not do:** disable Defender, an ASR rule, AppLocker or the execution policy to make
  the unsigned package run. If a step needs elevation, it is an **install step** for a
  human, not a runtime step, and the application itself must run as a standard user.

The signing position, the exact commands a human with a certificate would run, and what
the absence costs are in [install/SIGNING.md](install/SIGNING.md). The offline bundle and
its SHA-256 manifest are documented in [install/SETUP.md](install/SETUP.md).
