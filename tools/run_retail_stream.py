#!/usr/bin/env python3
r"""Run the retail stream headless and print where the work actually ran.

    .venv\Scripts\python.exe tools\run_retail_stream.py --device CPU --seconds 8

A thin wrapper around ``python -m app.main --headless``. **It computes nothing.** Every
figure it prints was measured by ``app.main.run_headless``; re-deriving any of it here
would give a booth operator two numbers for the same thing and no way to tell which one
the application stands behind (the same argument ``tools/preflight.py`` makes about not
re-timing ``tools/qualify_m0.py``'s table).

## Why the wrapper exists at all

``python -m app.main --headless`` is one flag away and needs no wrapper. This exists so
that the operator-facing entry point has a name that says what it does, and so the two
things a headless run must never do happen **here, once**:

1. **``QT_QPA_PLATFORM=offscreen``** is set if the caller has not set it. A headless run
   that opened a window would be a claim about a display that was never meant to be used;
   the application still supports ``--windowed`` for the case where a display is wanted.
2. **``--json`` gets a default path.** A run whose summary only ever reached a scrollback
   buffer is a run nobody can check afterwards. The default lands in the repository's
   ``artifacts/`` directory with the device in the name, and the path is printed.

## What it refuses to do

- **It does not invent a device.** ``--device`` accepts exactly ``CPU``, ``GPU`` or
  ``NPU``. There is no ``AUTO``, no ``HETERO`` and no ``MULTI``: those hide where the work
  ran, which is the one fact this run exists to establish.
- **It does not fall back silently.** If the requested device cannot take the work, the
  summary's ``device.confirmed`` says what ran instead and ``device.fell_back`` is
  ``true``. The wrapper prints both and does not retry on another device - a retry would
  turn "the NPU did not run this" into a number about the CPU that nobody asked for.
- **It does not treat a fallback as a failure.** Exit code ``0`` means the run happened
  and its summary was written. Exit code ``2`` means it did not run and nothing was
  written, and the reason is on stderr.

Exit codes
    ``0``  the run happened and the summary was written.
    ``2``  the run was refused; nothing was written; the reason is on stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Where a summary lands when the caller names no path. Under `artifacts/`, which is
#: where a run that produced evidence belongs; a file written beside the source would be
#: picked up by the next test run's discovery.
ARTIFACT_DIR = ROOT / "artifacts"

#: The only devices this may be asked for. `AUTO`/`HETERO`/`MULTI` are absent by
#: construction rather than by convention - see the module note.
DEVICES = ("CPU", "GPU", "NPU")


def build_parser() -> argparse.ArgumentParser:
    """The wrapper's own flags. Everything else belongs to `app.main`."""
    parser = argparse.ArgumentParser(
        prog="tools/run_retail_stream.py",
        description="Run the retail stream headless and report the confirmed placement.",
    )
    parser.add_argument("--device", default="CPU", choices=DEVICES,
                        help="the explicitly requested compile target.")
    parser.add_argument("--seconds", type=float, default=8.0,
                        help="how long the run lasts. The default is long enough for the "
                             "NPU's first inferences to settle, measured at 5.92-5.99 ms "
                             "each after a multi-second compile.")
    parser.add_argument("--source", default="loop", choices=("loop", "once"),
                        help="loop the clip, or stop at the end of one pass.")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="where to write the summary. Defaults to "
                             "artifacts/headless-<device>-<utc timestamp>.json.")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the one-line verdict instead of the whole "
                             "summary.")
    return parser


def default_json_path(device: str, now: datetime) -> Path:
    """A timestamped path under `artifacts/`. UTC, because two machines must not collide."""
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    return ARTIFACT_DIR / f"headless-{device}-{stamp}.json"


def build_command(target: Path, args: argparse.Namespace) -> list[str]:
    """The exact `python -m app.main` invocation. Built here so it can be printed."""
    return [sys.executable, "-m", "app.main",
            "--headless",
            "--device", args.device,
            "--seconds", str(args.seconds),
            "--source", args.source,
            "--json", str(target)]


def summarise(summary: dict) -> str:
    """One line naming the confirmed placement. Never a claim about anything unmeasured.

    Prints `unconfirmed` when `confirmed` is empty. That is the case where a run
    submitted frames and analysed none - a figure printed as a device name there would be
    the requested device wearing the confirmed one's label.
    """
    device = summary.get("device", {})
    confirmed = device.get("confirmed") or []
    fell_back = device.get("fell_back")
    if not confirmed:
        placement = "unconfirmed"
    elif fell_back:
        placement = f"{'/'.join(confirmed)} (fell back from {device.get('requested')})"
    else:
        placement = "/".join(confirmed)
    age = summary.get("frame_age_ms", {})
    fps = summary.get("ai_fps")
    # **Every absent figure says so.** The first version defaulted
    # `summary.get("frames_analysed", 0)`, which prints `0 analysed` for a summary that
    # does not carry the key at all - a fabricated zero in the one line an operator is
    # most likely to read off a scrollback buffer.
    analysed = _number(summary, "frames_analysed")
    submitted = _number(summary, "frames_submitted")
    refused = _number(summary, "frames_refused")
    drops = _number(summary, "drops")
    return (f"{placement}: {analysed} analysed of {submitted} submitted, "
            f"{refused} refused, {drops} dropped, "
            f"ai_fps={fps if fps is not None else 'unmeasured'}, "
            f"frame_age p50={age.get('p50')} ms p95={age.get('p95')} ms, "
            f"events={json.dumps(summary.get('event_counts', {}), sort_keys=True)}")


def _number(summary: dict, key: str) -> str:
    """The figure, or `unreported` when the summary does not carry it."""
    value = summary.get(key)
    return str(value) if isinstance(value, (int, float)) else "unreported"


def main(argv: list[str] | None = None) -> int:
    """Run it. Returns the child's exit code; ``2`` when the run did not happen."""
    args = build_parser().parse_args(argv)
    now = datetime.now(timezone.utc)
    target = Path(args.json_path) if args.json_path else default_json_path(args.device, now)

    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["PYTHONPATH"] = str(ROOT)

    command = build_command(target, args)
    print(f"[run] {' '.join(command)}", flush=True)
    print(f"[run] QT_QPA_PLATFORM={env['QT_QPA_PLATFORM']}", flush=True)

    # stdout is captured rather than inherited so `--quiet` is real: `app.main` prints the
    # whole summary itself, so an inherited stdout would make `--quiet` a flag that
    # changes nothing. stderr is inherited - a refusal has to reach the operator.
    completed = subprocess.run(command, cwd=str(ROOT), env=env,
                               stdout=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        print(f"[run] the run did not happen; nothing was written to {target}",
              file=sys.stderr, flush=True)
        return completed.returncode

    if not target.exists():
        # `app.main --headless` exited 0 without writing. Reported rather than papered
        # over: a missing summary means the evidence this tool exists to produce is not
        # on disk, and saying so is the only honest response.
        print(f"[run] the run exited 0 but wrote no summary to {target}",
              file=sys.stderr, flush=True)
        return 2

    summary = json.loads(target.read_text(encoding="utf-8"))
    print(f"[run] {summarise(summary)}", flush=True)
    print(f"[run] summary written to {target}", flush=True)
    if not args.quiet:
        # The child's own stdout, verbatim, so this wrapper adds a verdict line and never
        # reformats a figure the application measured.
        sys.stdout.write(completed.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover - the operator entry point
    sys.exit(main())