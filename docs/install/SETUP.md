# Setup: the offline bundle

This directory documents how to take the M6 bundle to the demo machine, verify it, install
it **without network access**, and run it.

## What the bundle is, and what it is not

The bundle is a **reproducible offline source bundle**:

- the application source under `app/`,
- the pinned wheels under `wheelhouse/`,
- the verified model files under `assets/models/`,
- **only the clip the application uses**, `assets/media/retail-checkout.mp4`,
- the scenario manifest and the JSON config the application reads,
- `provision_offline.py`, a stdlib-only installer that verifies the manifest first,
- `bundle-manifest.json`, which records the **SHA-256 of every payload file**.

It is **unsigned**. There is no Authenticode signature on this bundle and no certificate
was available to create one. It is not a signed installer and it is not a frozen
executable. The manifest records `"signed": false` and a `signature_note` saying the same
thing. Nothing in this package should be described to a customer as signed or certified.

Windows may show a SmartScreen or "unknown publisher" warning. **Do not** disable
Defender, ASR rules, AppLocker or the execution policy to work around it. The signing step
a human with a certificate would run is documented in the M6 signing document; until that
step is performed, the package is unsigned.

## Before you start

The bundle is built for **CPython 3.12 on 64-bit Windows**. The pinned wheels carry the
tags that prove it (`cp312` and `abi3` wheels for `win_amd64`). The demo machine must have
a CPython 3.12 interpreter available. Installing Python is an **install step**: a human
runs the installer with the privileges it asks for. Nothing in this bundle requires
elevation at runtime.

No network is used by the install or by the application. The only package source is the
`wheelhouse/` directory in this bundle; `provision_offline.py` passes `--no-index` to pip.

## 1. Verify the manifest

From a copy of the bundle, with any CPython 3.12 (the provisioner is stdlib only):

```powershell
python provision_offline.py --verify-only
```

It recomputes the SHA-256 of every file in `bundle-manifest.json` and prints
`MANIFEST PASS: <n> files match their SHA-256`, plus the manifest's own unsigned note. If
any byte differs, it names the file and exits non-zero. Do not install a bundle that does
not pass.

## 2. Install into a fresh virtual environment

```powershell
python provision_offline.py --selftest
```

That command:

1. verifies the manifest again,
2. creates `.venv` beside `provision_offline.py`,
3. runs `pip install --no-index --only-binary=:all: --find-links wheelhouse -r requirements.txt`,
4. prints `pip freeze`,
5. runs `.venv\Scripts\python.exe -m app.main --selftest`.

`--no-index` means a package index is never consulted, even if one is configured. The
wheelhouse contains the pinned requirements **and the dependencies those requirements
declare** (`openvino-telemetry`, `PySide6-Essentials`, `PySide6-Addons`, `shiboken6`). A
wheel outside that closure, or a missing pin, makes `tools/build_bundle.py` refuse to
build the bundle in the first place.

If a wheel is missing on the machine that built the bundle, the bundle was not built: the
builder fails rather than omitting it. Rebuild on a machine with the pinned environment,
then copy the completed bundle. **Do not** substitute a different version, a different
interpreter, or a pre-existing environment.

## 3. Run

Self-test (no window, no network):

```powershell
.venv\Scripts\python.exe -m app.main --selftest
```

The booth screen:

```powershell
.venv\Scripts\python.exe -m app.main --windowed
```

The command line is the same one the qualification record used. `--selftest` is the gate
to run before a booth; it exits 0 and prints `PASS` when every stage works.

## If verification fails

A manifest mismatch means the bytes are not the bytes that were built. Re-copy the bundle
from the source media and verify again. Do not edit a bundled file and do not regenerate
the manifest by hand: the manifest is written by `tools/build_bundle.py` from the files it
staged, and a hand-edited manifest is not evidence of anything.

After the installer creates `.venv`, that directory is generated state and is not in the
manifest. Run `--verify-only` **before** installation, as above.

## What is not in the bundle

- **A signature or certificate.** See the note above and the signing document.
- **A frozen executable.** The project permits no new dependency, so there is no packaging
  runtime to freeze with. The bundle installs source into a venv.
- **The two clips the application does not use.** The bundle carries the one clip the code
  names (`assets/media/retail-checkout.mp4`), not the other provisioned media.
- **Elevation or a security-control change.** Neither is required, and neither is offered.
