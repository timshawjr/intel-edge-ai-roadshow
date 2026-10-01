#!/usr/bin/env python3
"""One-time provisioning of ``.venv`` from ``requirements.txt``.

**This is the only step in the project that needs network access.** Everything
after it runs offline: the pinned dependencies are installed once here, and the
application never reaches the network at runtime.

Usage::

    python tools/provision_venv.py

The venv is created at ``<repo>/.venv`` from the running interpreter, then
``pip install -r requirements.txt`` is executed. On success the resolved
``pip freeze`` is printed so the provisioned set can be recorded in the
qualification report.

A venv is not relocatable: ``pyvenv.cfg`` and ``Scripts/*.exe`` embed absolute
paths. This script always creates the venv in this tree and never copies one in
from elsewhere. If ``pip install`` fails, that is a blocker to report - do not
substitute a different version or a different interpreter.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args), cwd=ROOT, text=True, capture_output=True, check=False
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="delete an existing .venv first (a venv is not relocatable)",
    )
    args = parser.parse_args(argv)

    if not REQUIREMENTS.exists():
        print(f"missing {REQUIREMENTS}; cannot provision", file=sys.stderr)
        return 1

    if VENV_DIR.exists():
        if not args.recreate:
            print(f"{VENV_DIR} already exists; pass --recreate to rebuild it")
        else:
            print(f"removing {VENV_DIR}")
            shutil.rmtree(VENV_DIR)

    if not VENV_DIR.exists():
        print(f"creating venv at {VENV_DIR} from {sys.executable}")
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV_DIR)

    python = venv_python(VENV_DIR)
    if not python.exists():
        print(f"venv interpreter not found at {python}", file=sys.stderr)
        return 1

    # No pip self-upgrade: the brief did not ask for one, and an unpinned pip
    # upgrade is an unrequested variable in a reproducible environment. The pip
    # that ships with the venv is recorded below instead.
    pip_version = run(str(python), "-m", "pip", "--version")
    print(f"pip in the new venv: {pip_version.stdout.strip() or pip_version.stderr.strip()}")

    print(f"installing pinned dependencies from {REQUIREMENTS.name} (needs network)")
    install = run(str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS))
    print(install.stdout)
    if install.returncode != 0:
        print(install.stderr, file=sys.stderr)
        print(
            "pip install failed. Do NOT substitute a different version, a "
            "different Python, or a pre-existing venv. Report the blocker with "
            "the pip error above.",
            file=sys.stderr,
        )
        return install.returncode

    freeze = run(str(python), "-m", "pip", "freeze")
    print("--- pip freeze ---")
    print(freeze.stdout, end="")
    print("--- end pip freeze ---")
    return freeze.returncode


if __name__ == "__main__":
    raise SystemExit(main())
