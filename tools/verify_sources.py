#!/usr/bin/env python3
"""Content-verify every local artifact named in ``config/models.json``.

An HTTP status is never accepted as proof of anything here. A 200 can carry an
HTML error page, a JSON error object, or a Git LFS pointer instead of the bytes
that were asked for. This tool therefore checks *content*:

* ``xml``    - the payload must start with an XML declaration and look like
               OpenVINO IR (a ``<net>`` or ``<ie>`` root).
* ``mp4``    - the payload must carry an ``ftyp`` box in bytes 4..8.
* ``binary`` - the payload must not be HTML, a Git LFS pointer, or a text error
               body.

Each entry is additionally checked against the SHA-256 recorded in
``config/models.json``, so a silently swapped artifact fails even when its shape
is plausible. Paths in the config are repo-relative; this tool runs offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "models.json"

_LFS_PREFIX = b"version https://git-lfs.github.com/spec/"
_ERROR_MARKERS = (
    b"not found",
    b"access denied",
    b"forbidden",
    b"error",
    b"exception",
    b"traceback",
)


@dataclass(frozen=True)
class Source:
    id: str
    kind: str  # "xml" | "mp4" | "binary"
    path: Path


@dataclass(frozen=True)
class Result:
    source: Source
    ok: bool
    detail: str


def looks_like_xml(payload: bytes) -> bool:
    """True only when the payload opens with an XML declaration."""

    head = payload[:4096].lstrip(b"\xef\xbb\xbf").lstrip()
    return head.startswith(b"<?xml")


def looks_like_mp4(payload: bytes) -> bool:
    """True only when the payload carries an MP4 ``ftyp`` box."""

    return len(payload) >= 12 and payload[4:8] == b"ftyp"


def looks_like_binary(payload: bytes) -> bool:
    """True when the payload is not HTML, a Git LFS pointer, or a text error."""

    if not payload:
        return False
    head = payload[:4096]
    lowered = head.lstrip().lower()
    if lowered.startswith(b"<!doctype html") or lowered.startswith(b"<html") or b"<html" in head[:1024].lower():
        return False
    if payload.startswith(_LFS_PREFIX):
        return False
    try:
        text = head[:1024].decode("ascii")
    except UnicodeDecodeError:
        return True  # genuinely binary, as weights are
    return not any(marker in text.lower().encode("ascii") for marker in _ERROR_MARKERS)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head(path: Path, count: int = 65536) -> bytes:
    with path.open("rb") as handle:
        return handle.read(count)


def verify(source: Source, expected_sha256: str | None = None) -> Result:
    path = source.path if source.path.is_absolute() else ROOT / source.path
    if not path.exists():
        return Result(source, False, f"file not found: {path}")
    size = path.stat().st_size
    if size == 0:
        return Result(source, False, "file is empty")

    head = _head(path)
    if source.kind == "xml":
        if not looks_like_xml(head):
            return Result(source, False, f"no XML declaration in first bytes ({size} bytes)")
        if b"<net" not in head and b"<ie" not in head:
            return Result(source, False, "XML does not have an OpenVINO IR root")
        shape = "xml+OpenVINO-IR"
    elif source.kind == "mp4":
        if not looks_like_mp4(head):
            return Result(source, False, f"no MP4 ftyp box in bytes 4..8 ({size} bytes)")
        shape = "mp4+ftyp"
    elif source.kind == "binary":
        if not looks_like_binary(head):
            return Result(source, False, f"payload is HTML, a Git LFS pointer, or a text error ({size} bytes)")
        shape = "binary"
    else:
        return Result(source, False, f"unsupported source kind {source.kind!r}")

    if expected_sha256 is None:
        return Result(source, False, f"{shape}, but no sha256 is recorded for this entry")

    actual = sha256_of(path)
    if actual != expected_sha256:
        return Result(
            source,
            False,
            f"{shape}, sha256 mismatch: recorded {expected_sha256}, actual {actual}",
        )
    return Result(source, True, f"{shape}, {size} bytes, sha256 {actual}")


def collect_sources(config: dict[str, Any]) -> list[tuple[Source, str | None]]:
    pairs: list[tuple[Source, str | None]] = []
    for model in config.get("models", []):
        pairs.append((Source(model["id"], "xml", Path(model["ir_path"])), model.get("sha256")))
        weights = model.get("weights_path")
        if weights:
            pairs.append(
                (Source(f"{model['id']}:weights", "binary", Path(weights)), model.get("weights_sha256"))
            )
    for media in config.get("media", []):
        pairs.append((Source(media["id"], "mp4", Path(media["path"])), media.get("sha256")))
    return pairs


def print_table(results: Iterable[Result]) -> None:
    results = list(results)
    width = max((len(r.source.id) for r in results), default=8)
    print(f"{'STATUS':6}  {'ID':<{width}}  {'KIND':8}  DETAIL")
    print("-" * (26 + width))
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"{status:6}  {result.source.id:<{width}}  {result.source.kind:8}  {result.detail}")
        if not result.ok:
            print(
                f"       {'':<{width}}  remediation: re-fetch or re-provision the pinned "
                f"artifact and re-run this tool; never substitute a different one"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH, help="path to config/models.json"
    )
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    results = [verify(source, sha) for source, sha in collect_sources(config)]
    print_table(results)
    failures = sum(not result.ok for result in results)
    print(f"\nSource verification: {len(results) - failures} PASS, {failures} FAIL")
    if failures:
        print(
            "A status-only check is not accepted; every failure above failed its "
            "content signature or hash check.",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
