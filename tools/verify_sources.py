#!/usr/bin/env python3
"""Content-verify every local artifact named in ``config/models.json``.

An HTTP status is never accepted as proof of anything here. A 200 can carry an
HTML error page, a plain-text refusal, or a Git LFS pointer instead of the bytes
that were asked for. This tool therefore checks *content*:

* ``xml``    - the payload must start with an XML declaration and look like
               OpenVINO IR (a ``<net>`` or ``<ie>`` root).
* ``mp4``    - the payload must carry an ``ftyp`` box in bytes 4..8.
* ``binary`` - the payload must not be HTML, a Git LFS pointer, XML, or a
               text-shaped body.
* ``text``   - the payload must decode as UTF-8 and have exactly the declared
               number of lines.

**The SHA-256 recorded in ``config/models.json`` is the gate.** The shape checks
above are necessary but not sufficient: they cannot tell one valid IR from
another. A file whose bytes hash to the recorded value is the artifact that was
verified; a file whose bytes do not is not, whatever shape it has. An entry with
no recorded hash fails closed rather than reading as verified.

Paths in the config are repo-relative; this tool runs offline.
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
_UTF8_BOM = b"\xef\xbb\xbf"
#: A weight payload is not text. Measured on the pinned FP16 weights
#: (assets/models/yolo11n-fp16/yolo11n.bin) the printable-ASCII proportion of the
#: first 64 KiB is 0.5443, and of the first 4 KiB it is 0.6875. A threshold of
#: 0.90 is comfortably above both and still rejects every plain-text error body,
#: which is ~1.0 printable.
_PRINTABLE_RATIO_LIMIT = 0.90
_PRINTABLE_SAMPLE = 65536


@dataclass(frozen=True)
class Source:
    id: str
    kind: str  # "xml" | "mp4" | "binary" | "text"
    path: Path


@dataclass(frozen=True)
class Result:
    source: Source
    ok: bool
    detail: str


def _printable_ratio(payload: bytes) -> float:
    if not payload:
        return 1.0
    printable = sum(
        1 for b in payload if 32 <= b <= 126 or b in (9, 10, 13)
    )
    return printable / len(payload)


def looks_like_xml(payload: bytes) -> bool:
    """True only when the payload opens with an XML declaration.

    The BOM is stripped with an explicit ``startswith``, not ``lstrip(charset)``:
    ``bytes.lstrip`` treats its argument as a *set* of bytes to remove, so
    ``b"\\xbf\\xbf<?xml"`` would otherwise pass.
    """

    head = payload[:4096]
    if head.startswith(_UTF8_BOM):
        head = head[len(_UTF8_BOM):]
    return head.lstrip().startswith(b"<?xml")


def looks_like_mp4(payload: bytes) -> bool:
    """True only when the payload carries an MP4 ``ftyp`` box."""

    return len(payload) >= 12 and payload[4:8] == b"ftyp"


def looks_like_binary(payload: bytes) -> bool:
    """True when the payload is plausibly binary rather than a served text page.

    Rejects, in order: an empty payload, HTML, a Git LFS pointer, an XML
    document, and any payload whose printable-ASCII proportion exceeds
    ``_PRINTABLE_RATIO_LIMIT``. The last rule is what catches the error bodies
    that the named-shape rules miss - "rate limit exceeded", an access refusal,
    a permission denial - all of which are entirely printable text.
    """

    if not payload:
        return False
    head = payload[:_PRINTABLE_SAMPLE]
    lowered = head.lstrip().lower()
    if lowered.startswith(b"<!doctype html") or lowered.startswith(b"<html") or b"<html" in head[:1024].lower():
        return False
    if payload.startswith(_LFS_PREFIX):
        return False
    if looks_like_xml(payload):
        return False
    return _printable_ratio(head) <= _PRINTABLE_RATIO_LIMIT


def looks_like_text(payload: bytes, expected_lines: int | None) -> tuple[bool, str]:
    """True when the payload decodes as UTF-8 and has the expected line count."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        return False, f"does not decode as UTF-8: {exc}"
    lines = text.splitlines()
    if expected_lines is None:
        return True, f"utf-8, {len(lines)} lines (no line count declared)"
    if len(lines) != expected_lines:
        return False, f"expected {expected_lines} lines, found {len(lines)}"
    return True, f"utf-8, {len(lines)} lines"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_head(path: Path, count: int = 65536) -> bytes:
    with path.open("rb") as handle:
        return handle.read(count)


def verify(
    source: Source,
    expected_sha256: str | None = None,
    expected_lines: int | None = None,
) -> Result:
    path = source.path if source.path.is_absolute() else ROOT / source.path
    if not path.exists():
        return Result(source, False, f"file not found: {path}")
    size = path.stat().st_size
    if size == 0:
        return Result(source, False, "file is empty")

    if source.kind == "text":
        ok, shape_detail = looks_like_text(_read_head(path, _PRINTABLE_SAMPLE * 64), expected_lines)
        if not ok:
            return Result(source, False, f"text: {shape_detail} ({size} bytes)")
        shape = f"text+{shape_detail}"
    else:
        head = _read_head(path)
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
                return Result(
                    source,
                    False,
                    f"payload is HTML, XML, a Git LFS pointer, or a text error body "
                    f"(printable ratio {_printable_ratio(head[:_PRINTABLE_SAMPLE]):.3f}) "
                    f"({size} bytes)",
                )
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


def collect_sources(config: dict[str, Any]) -> list[tuple[Source, str | None, int | None]]:
    """Every recorded artifact in the config, as (source, sha256, expected_lines)."""

    rows: list[tuple[Source, str | None, int | None]] = []
    for model in config.get("models", []):
        rows.append((Source(model["id"], "xml", Path(model["ir_path"])), model.get("sha256"), None))
        weights = model.get("weights_path")
        if weights:
            rows.append(
                (
                    Source(f"{model['id']}:weights", "binary", Path(weights)),
                    model.get("weights_sha256"),
                    None,
                )
            )
        labels = model.get("labels_path")
        if labels:
            rows.append(
                (
                    Source(f"{model['id']}:labels", "text", Path(labels)),
                    model.get("labels_sha256"),
                    model.get("num_classes"),
                )
            )
    for media in config.get("media", []):
        rows.append((Source(media["id"], "mp4", Path(media["path"])), media.get("sha256"), None))
    return rows


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
                f"{'':8}  {'':<{width}}  remediation: re-fetch or re-provision the pinned "
                f"artifact and re-run this tool; never substitute a different one"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH, help="path to config/models.json"
    )
    args = parser.parse_args(argv)

    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"config unreadable: {args.config} ({exc.strerror or exc})", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"config unreadable: {args.config} (invalid JSON: {exc})", file=sys.stderr)
        return 1

    results = [verify(source, sha, lines) for source, sha, lines in collect_sources(config)]
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
