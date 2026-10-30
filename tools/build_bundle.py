#!/usr/bin/env python3
"""Build the M6 reproducible offline bundle and its per-file SHA-256 manifest.

**What this produces, and what it is not.** There is no new dependency in this project,
so there is no frozen executable and no installer. The deliverable is a *source bundle*:
the application source, the pinned wheels, the verified model files, the one clip the
application uses, an offline provisioning script, the setup guide, and a manifest that
records the SHA-256 of every payload file. It is **unsigned**. It carries no Authenticode
signature and no certificate was available to make one. The manifest says so in
``signed`` and ``signature_note``; nothing here fabricates a signature or a hash.

**Why the wheelhouse is a closure, not six wheels.** ``requirements.txt`` pins six
distributions. Those distributions declare their own dependencies
(``openvino-telemetry``, ``PySide6-Essentials``, ``PySide6-Addons``, ``shiboken6``), and
an offline install cannot resolve them from an index. The builder therefore reads the
direct pins, walks the dependencies the **installed pinned environment declares**, and
requires the wheelhouse to contain exactly that closure: every direct pin at its pinned
version, every declared dependency present, and nothing unrelated. A wheel that is not
in the closure (for example ``requests``) fails the build, and a missing pin fails it
too.

**Nothing is silently dropped.** Files are staged into a temporary directory, hashed
from the staged bytes, and only then renamed into place. If a single file cannot be read
or hashed the build raises :class:`BundleError` and leaves no destination behind.

Usage::

    python tools/build_bundle.py --destination dist/M6-bundle --wheelhouse dist/wheelhouse
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Iterable, Mapping

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REQUIREMENTS: Final[Path] = REPO_ROOT / "requirements.txt"
MANIFEST_NAME: Final[str] = "bundle-manifest.json"
SCHEMA_VERSION: Final[int] = 1
BUNDLE_FORMAT: Final[str] = "reproducible-offline-source-bundle"

#: The honest description of this artifact. It is deliberately not a claim of signing.
SIGNATURE_NOTE: Final[str] = (
    "UNSIGNED. This bundle carries no Authenticode signature and no certificate was "
    "available to create one. It is a reproducible offline source bundle (source, pinned "
    "wheels, verified models, the used clip, a provisioning script and this SHA-256 "
    "manifest), not a signed installer. Windows may warn that the publisher is unknown; "
    "do not weaken Defender, ASR, AppLocker or the execution policy to get around that. "
    "The signing step a human with a certificate would run is documented in "
    "docs/install/SETUP.md."
)

#: The manifest is written last and is not listed in its own ``files`` mapping.
_WHEEL_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)(?:-(?P<build>\d[^-]*))?"
    r"-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^-]+)\.whl$"
)
_REQUIREMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^=<>!;\s]+)$"
)


class BundleError(RuntimeError):
    """The bundle cannot be built as specified. Never downgraded to a warning."""


@dataclass(frozen=True)
class BundleManifest:
    """The manifest written beside the payload, and returned to the builder's caller."""

    files: Mapping[str, str]
    python_version: str
    platform: str
    signed: bool
    signature_note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "bundle_format": BUNDLE_FORMAT,
            "generated_by": "tools/build_bundle.py",
            "python_version": self.python_version,
            "platform": self.platform,
            "signed": self.signed,
            "signature_note": self.signature_note,
            "files": {relative: self.files[relative] for relative in sorted(self.files)},
        }


@dataclass(frozen=True)
class VerificationReport:
    """What a verification pass found. ``ok`` is the conjunction of the empty lists."""

    ok: bool
    checked: int
    missing: tuple[str, ...]
    mismatched: tuple[str, ...]
    unexpected: tuple[str, ...]


def canonical_name(name: str) -> str:
    """PEP 503 normalisation, so ``PySide6_Addons`` and ``pyside6-addons`` agree."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def parse_requirements(path: Path = REQUIREMENTS) -> dict[str, str]:
    """Return ``{canonical name: version}`` for every line; refuse anything but ``==``."""
    parsed: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise BundleError(f"cannot read {path}: {error}") from error
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _REQUIREMENT_RE.match(line)
        if match is None:
            raise BundleError(
                f"{path.name}:{number}: {line!r} is not an exact == pin; the wheelhouse "
                f"cross-check cannot be honest about a range"
            )
        name = canonical_name(match.group("name"))
        if name in parsed:
            raise BundleError(f"{path.name}:{number}: {name} is pinned twice")
        parsed[name] = match.group("version")
    if not parsed:
        raise BundleError(f"{path} contains no pins")
    return parsed


def wheel_identity(path: Path) -> tuple[str, str]:
    """Return ``(canonical distribution, version)`` parsed from a PEP 427 file name."""
    match = _WHEEL_RE.match(path.name)
    if match is None:
        raise BundleError(f"{path.name!r} is not a PEP 427 wheel file name")
    return canonical_name(match.group("name")), match.group("version")


def _marker_applies(marker: str) -> bool:
    """Evaluate the small marker grammar the pinned distributions actually use.

    ``extra == ...`` is optional and never part of the runtime closure. Simple
    ``python_version``/``sys_platform``/``os_name`` clauses are evaluated; an unknown
    clause is treated as applying, which fails closed by demanding the dependency be
    present rather than silently omitting it.
    """
    marker = marker.strip()
    if not marker:
        return True
    if "extra" in marker:
        return False

    def clause(part: str) -> bool | None:
        part = part.strip()
        match = re.match(
            r'^(python_version|python_full_version|sys_platform|os_name|platform_system)'
            r'\s*(==|!=|<=|>=|<|>)\s*[\'"]([^\'"]+)[\'"]$',
            part,
        )
        if match is None:
            return None
        key, operator, wanted = match.groups()
        if key.startswith("python"):
            actual = platform.python_version()
            left: Any = tuple(int(piece) for piece in actual.split(".")[: len(wanted.split("."))])
            right: Any = tuple(int(piece) for piece in wanted.split("."))
        elif key == "sys_platform":
            left, right = sys.platform, wanted
        elif key == "os_name":
            left, right = ("nt" if sys.platform.startswith("win") else "posix"), wanted
        else:
            left, right = platform.system(), wanted
        return {
            "==": left == right,
            "!=": left != right,
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
        }[operator]

    for part in re.split(r"\s+and\s+", marker):
        verdict = clause(part)
        if verdict is False:
            return False
        if verdict is None:
            return True
    return True


def _declared_dependencies(distribution: str) -> Iterable[tuple[str, str | None]]:
    """Yield ``(canonical name, None)`` for the runtime requirements a dist declares."""
    for spec in metadata.requires(distribution) or ():
        base, _, marker = spec.partition(";")
        if not _marker_applies(marker):
            continue
        base = base.split("[", 1)[0].strip()
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", base)
        if match is None:
            continue
        yield canonical_name(match.group(1)), None


def dependency_closure(requirements: Mapping[str, str]) -> dict[str, str]:
    """The direct pins plus the dependencies the pinned environment declares.

    The installed metadata is the ground truth for "declared": a dependency the pinned
    environment does not have cannot be provisioned offline, so it is an error rather
    than an omission.
    """
    closure: dict[str, str] = {}
    queue: list[tuple[str, str | None]] = list(requirements.items())
    while queue:
        name, pinned = queue.pop(0)
        if name in closure:
            if pinned is not None and closure[name] != pinned:
                raise BundleError(f"{name} is required at two versions")
            continue
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as error:
            raise BundleError(
                f"{name} is pinned but is not installed in the interpreter building the "
                f"bundle, so its dependencies cannot be resolved offline"
            ) from error
        if pinned is not None and distribution.version != pinned:
            raise BundleError(
                f"{name} is pinned at {pinned} but {distribution.version} is installed"
            )
        closure[name] = distribution.version
        queue.extend(_declared_dependencies(name))
    return closure


def validate_wheelhouse(
    wheelhouse: Path, requirements: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Refuse a wheelhouse that is not exactly the pinned closure."""
    requirements = requirements or parse_requirements()
    if not wheelhouse.is_dir():
        raise BundleError(f"wheelhouse {wheelhouse} is not a directory")
    found: dict[str, Path] = {}
    for path in sorted(wheelhouse.glob("*.whl")):
        name, version = wheel_identity(path)
        if name in found:
            raise BundleError(f"two wheels for {name}: {found[name].name} and {path.name}")
        found[name] = path
    if not found:
        raise BundleError(f"wheelhouse {wheelhouse} contains no wheels")

    closure = dependency_closure(requirements)
    extra = sorted(set(found) - set(closure))
    if extra:
        raise BundleError(
            "wheelhouse contains wheels that are not pinned in requirements.txt or "
            f"declared by a pinned distribution: {', '.join(extra)}"
        )
    missing = sorted(set(closure) - set(found))
    if missing:
        raise BundleError(
            "wheelhouse is missing a pinned requirement or a dependency it declares: "
            + ", ".join(missing)
        )
    for name, version in closure.items():
        _, wheel_version = wheel_identity(found[name])
        if wheel_version != version:
            raise BundleError(
                f"{name}: the pinned environment has {version} but the wheelhouse has "
                f"{wheel_version}"
            )
    return {name: wheel_identity(found[name])[1] for name in sorted(found)}


#: The provisioning script is generated into the bundle. It is stdlib only, verifies the
#: manifest before touching the machine, and passes ``--no-index`` so pip cannot reach a
#: package index even if a proxy is configured.
PROVISION_SCRIPT: Final[str] = '''#!/usr/bin/env python3
"""Verify this bundle, then provision it offline into a venv beside it.

The only source of packages is ``wheelhouse/``. ``--no-index`` is passed to pip, so a
network index is not consulted even if one is configured. This script does not change
any security setting and does not need elevation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
MANIFEST = BUNDLE / "bundle-manifest.json"
REQUIREMENTS = BUNDLE / "requirements.txt"
WHEELHOUSE = BUNDLE / "wheelhouse"
VENV = BUNDLE / ".venv"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> bool:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems = []
    for relative, expected in data["files"].items():
        path = BUNDLE / relative
        if not path.is_file():
            problems.append(f"{relative}: missing")
            continue
        actual = sha256_of(path)
        if actual != expected:
            problems.append(f"{relative}: {actual} != {expected}")
    if problems:
        for problem in problems:
            print(f"MANIFEST FAIL {problem}", file=sys.stderr)
        return False
    print(f"MANIFEST PASS: {len(data['files'])} files match their SHA-256", flush=True)
    print(f"SIGNED: {data['signed']} - {data['signature_note']}", flush=True)
    return True


def venv_python(directory: Path) -> Path:
    if os.name == "nt":
        return directory / "Scripts" / "python.exe"
    return directory / "bin" / "python"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--selftest", action="store_true",
                        help="run python -m app.main --selftest after installing")
    args = parser.parse_args(argv)

    if not verify():
        return 1
    if args.verify_only:
        return 0

    if VENV.exists() and args.recreate:
        print(f"removing {VENV}", flush=True)
        shutil.rmtree(VENV)
    if not VENV.exists():
        print(f"creating venv at {VENV} from {sys.executable}", flush=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
    python = venv_python(VENV)
    if not python.exists():
        print(f"venv interpreter not found at {python}", file=sys.stderr)
        return 1

    environment = dict(os.environ)
    environment["PIP_NO_INDEX"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PIP_NO_INPUT"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"

    command = [
        str(python), "-m", "pip", "install",
        "--no-index",
        "--only-binary=:all:",
        "--find-links", str(WHEELHOUSE),
        "-r", str(REQUIREMENTS),
    ]
    print("installing offline:", " ".join(command), flush=True)
    install = subprocess.run(command, cwd=BUNDLE, env=environment)
    if install.returncode != 0:
        print("offline install failed; do not substitute a different version or a "
              "pre-existing environment", file=sys.stderr)
        return install.returncode

    print("--- pip freeze ---", flush=True)
    subprocess.run([str(python), "-m", "pip", "freeze"], cwd=BUNDLE,
                   env=environment)
    print("--- end pip freeze ---", flush=True)

    if args.selftest:
        print("--- python -m app.main --selftest ---", flush=True)
        return subprocess.run([str(python), "-m", "app.main", "--selftest"],
                              cwd=BUNDLE, env=environment).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _config_paths() -> list[str]:
    """Model and scenario payload, read from the verified config rather than guessed."""
    import json as _json

    config = _json.loads((REPO_ROOT / "config" / "models.json").read_text(encoding="utf-8"))
    paths: list[str] = []
    for model in config.get("models", []):
        for key in ("ir_path", "weights_path", "labels_path"):
            value = model.get(key)
            if value:
                paths.append(value)
    scenarios = REPO_ROOT / "assets" / "scenarios"
    paths.extend(
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted(scenarios.rglob("manifest.json"))
    )
    return sorted(set(paths))


def _used_clip() -> str:
    """The clip the application itself names, so the bundle follows the code."""
    if str(REPO_ROOT) not in sys.path:
        # Running ``python tools/build_bundle.py`` puts tools/ on sys.path, not the repo.
        sys.path.insert(0, str(REPO_ROOT))
    from app.engine.retail_config import RETAIL_CHECKOUT_CLIP

    return RETAIL_CHECKOUT_CLIP


def source_paths() -> list[str]:
    """Every payload file this repository contributes, as POSIX relative paths."""
    paths: set[str] = {
        "requirements.txt",
        "docs/install/SETUP.md",
        "config/models.json",
        "config/telemetry_map.json",
    }
    paths.update(
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted((REPO_ROOT / "config" / "model_io").glob("*.json"))
    )
    paths.update(_config_paths())
    paths.add(_used_clip())
    for path in sorted((REPO_ROOT / "app").rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        paths.add(relative.as_posix())
    missing = sorted(path for path in paths if not (REPO_ROOT / path).is_file())
    if missing:
        raise BundleError("the source tree is incomplete: " + ", ".join(missing))
    return sorted(paths)


def _hash_file(path: Path) -> str:
    """SHA-256 of the bytes on disk. Raises rather than returning a made-up value."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise BundleError(f"refusing to omit {path}: cannot hash it ({error})") from error
    return digest.hexdigest()


def _stage(destination: Path, wheelhouse: Path) -> tuple[Path, Path]:
    """Copy the source and wheels into a temporary sibling; return ``(staging, final)``."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".building-",
                                    dir=destination.parent))
    try:
        for relative in source_paths():
            source = REPO_ROOT / relative
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        provision = staging / "provision_offline.py"
        provision.write_text(PROVISION_SCRIPT, encoding="utf-8", newline="\n")

        wheels = staging / "wheelhouse"
        wheels.mkdir()
        for path in sorted(wheelhouse.glob("*.whl")):
            shutil.copy2(path, wheels / path.name)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging, destination


def build_bundle(destination: Path, wheelhouse: Path) -> BundleManifest:
    """Build the bundle at ``destination`` from ``wheelhouse``; return its manifest.

    The wheelhouse is validated **before** anything is copied. The payload is hashed in
    a staging directory and renamed into place, so a failure leaves no half-bundle.
    """
    destination = Path(destination)
    wheelhouse = Path(wheelhouse)
    validate_wheelhouse(wheelhouse)

    staging, final = _stage(destination, wheelhouse)
    try:
        files: dict[str, str] = {}
        for path in sorted(staging.rglob("*")):
            if not path.is_file() or path.name == MANIFEST_NAME:
                continue
            relative = path.relative_to(staging).as_posix()
            try:
                files[relative] = _hash_file(path)
            except BundleError:
                raise
            except OSError as error:
                raise BundleError(
                    f"refusing to omit {relative}: cannot hash it ({error})"
                ) from error
        manifest = BundleManifest(
            files=files,
            python_version=platform.python_version(),
            platform=platform.platform(),
            signed=False,
            signature_note=SIGNATURE_NOTE,
        )
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest.as_dict(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if final.exists():
            shutil.rmtree(final)
        staging.rename(final)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_manifest(destination: Path) -> BundleManifest:
    """Read and shape-check ``bundle-manifest.json`` from a built bundle."""
    destination = Path(destination)
    try:
        data = json.loads((destination / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read {destination / MANIFEST_NAME}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        raise BundleError(f"{MANIFEST_NAME} has no files mapping")
    if data.get("signed") is not False:
        raise BundleError(
            f"{MANIFEST_NAME} does not record signed=false; refusing to treat an "
            f"unverified signing claim as this bundle's status"
        )
    return BundleManifest(
        files={str(key): str(value) for key, value in data["files"].items()},
        python_version=str(data.get("python_version")),
        platform=str(data.get("platform")),
        signed=False,
        signature_note=str(data.get("signature_note", "")),
    )


def verify_bundle(destination: Path) -> VerificationReport:
    """Recompute every digest and report missing, changed and unlisted files."""
    destination = Path(destination)
    manifest = read_manifest(destination)
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected in manifest.files.items():
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise BundleError(f"manifest path escapes the bundle: {relative!r}")
        path = destination / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual = _hash_file(path)
        if actual != expected:
            mismatched.append(relative)
    listed = set(manifest.files)
    unexpected = sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
        and path.relative_to(destination).as_posix() not in listed
    )
    return VerificationReport(
        ok=not (missing or mismatched or unexpected),
        checked=len(manifest.files),
        missing=tuple(missing),
        mismatched=tuple(mismatched),
        unexpected=tuple(unexpected),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--wheelhouse", required=True, type=Path)
    parser.add_argument("--verify", type=Path, metavar="DESTINATION",
                        help="verify an existing bundle instead of building one")
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            report = verify_bundle(args.verify)
            print(f"checked {report.checked} files; ok={report.ok}")
            for label, values in (("missing", report.missing),
                                  ("mismatched", report.mismatched),
                                  ("unexpected", report.unexpected)):
                for value in values:
                    print(f"{label}: {value}", file=sys.stderr)
            return 0 if report.ok else 1
        manifest = build_bundle(args.destination, args.wheelhouse)
        print(f"built {args.destination}: {len(manifest.files)} files, "
              f"signed={manifest.signed}")
        print(manifest.signature_note)
        return 0
    except BundleError as error:
        print(f"bundle error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
