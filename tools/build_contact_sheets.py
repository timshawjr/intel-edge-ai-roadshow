"""Build the contact sheets the M0 footage review was read from.

Run from the repo root:

    .venv\\Scripts\\python.exe tools\\build_contact_sheets.py
    .venv\\Scripts\\python.exe tools\\build_contact_sheets.py --clip retail_checkout

Writes one JPEG per (clip, pass) under `logs/contact_sheets/` and prints the exact frame
indices it decoded. **It does not write into `assets/`, `config/` or `tests/`.**

**Why this tool exists at all.** The M0 content gate answers "is there a shelf-to-basket
clip in the material we actually shipped?" by *looking at decoded frames* - and until now
that was done by a throwaway harness whose code was not in the repository. So the one part
of M0 a reader cannot check is the part that decided the scope of the whole milestone: the
content gate promised the method was reproduced in `bench_report.md`, and it was
reproduced nowhere. `docs/qualification/M0-content-gate.md` cited frame numbers that no
one could re-derive. This tool is that derivation.

The frame lists below are **the lists the M0 report cites**, transcribed into
`CLIP_PASSES`. They are data, not parameters invented by this tool: change them and the
sheets stop being the evidence the report names. If a future pass needs different frames,
that is a new pass with a new name, and the report must cite that name.

**A contact sheet is evidence and it is weak evidence.** It is a grid of small frames -
a person looking at one can see a shelf, a basket, a room. It cannot show an event
sequence or an occlusion, which is why the dense passes exist and why the report's
conclusions are phrased about *what appears in the frames looked at*, never about what the
clip contains. Nothing here is annotated by the model: there is no detector in this file,
and a detection drawn on a twelve-frame grid is a claim this tool has no standing to make.
The tool's entire output is pixels plus the frame indices that produced them.

Exit code is 1 when any seek was not honoured, so a sheet that silently shows the wrong
frames cannot be reported as a clean build.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODELS_JSON = REPO_ROOT / "config" / "models.json"

#: Sheets are written here. `logs/` is gitignored, which is deliberate: these are derived
#: artifacts of a decode, not source, and a reader re-runs the tool rather than trusting a
#: committed JPEG of unknown provenance. The frame lists in the reports are the record.
OUTPUT_DIR = REPO_ROOT / "logs" / "contact_sheets"

#: The frame index lists the M0 content gate and `bench_report.md` §9 cite. 12 evenly
#: spaced frames per clip plus a denser pass over the regions where an event would be
#: visible. `queue_alt` has no dense pass: at 10 fps over 139.4 s the evenly spaced grid
#: already covers the whole clip and the report records no event to zoom into.
CLIP_PASSES: dict[str, dict[str, list[int]]] = {
    "retail_checkout": {
        "contact": [0, 308, 616, 923, 1231, 1539, 1847, 2155, 2463, 2770, 3078, 3386],
        "dense": [1400, 1500, 1560, 1620, 1680, 1740, 1800, 2000, 2200, 2400, 2600, 2900],
    },
    "store_aisle_alt": {
        "contact": [0, 356, 713, 1069, 1425, 1782, 2138, 2495, 2851, 3207, 3564, 3920],
        "dense": [900, 1000, 1069, 1140, 1200, 2400, 2495, 2560, 3100, 3207, 3270, 3400],
    },
    "queue_alt": {
        "contact": [0, 127, 253, 380, 507, 633, 760, 886, 1013, 1140, 1266, 1393],
    },
}

#: Grid width in tiles. 4 columns holds a 12-frame pass with the sheet still roughly
#: screen-shaped; 6 columns would give the same pixels in a wider, shorter sheet, and a
#: shelf edge is a horizontal feature that benefits from the height.
COLUMNS = 4

#: Longest edge of one tile. The clips range from 720x404 to 1920x1080 and the tile is
#: the same size for all of them, because comparing a 1920-wide tile with a 720-wide one
#: tells a reader more about the sheet than about the footage. 640 is large enough that a
#: basket at 720x404 survives downscaling and small enough that a 1080p sheet stays
#: viewable on one screen.
TILE_WIDTH = 640

#: JPEG quality. Lossy, and recorded, for the same reason the fixture JPEG's quality is:
#: a tile that cannot be read because it was saved at quality 40 defeats the purpose of
#: stamping it.
JPEG_QUALITY = 92


def media_entries() -> list[dict]:
    """The pinned media, in `config/models.json` order. Nothing here is discovered."""
    manifest = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    return list(manifest["media"])


def _stamp(tile: np.ndarray, index: int, timestamp_s: float) -> np.ndarray:
    """A frame index and timestamp, drawn on the tile.

    Without the stamp a grid of twelve frames is a grid of twelve frames, and "the basket
    is already full at frame 0" is unfalsifiable. `cv2.putText` is used because it is in
    the already-pinned `opencv-python`; a TrueType font would be a dependency this project
    should not take on at M0 to draw six characters.
    """
    label = f"frame {index}  t={timestamp_s:.2f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (width, height), baseline = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(tile, (0, 0), (width + 8, height + baseline + 6), (0, 0, 0), -1)
    cv2.putText(tile, label, (4, height + 2), font, scale, (255, 255, 255), thickness,
                cv2.LINE_AA)
    return tile


def build_sheet(clip_path: Path, indices: list[int]) -> tuple[np.ndarray, list[str]]:
    """Decode `indices` from `clip_path` and lay them out on one grid.

    `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES)` seeks by decoding from the preceding
    keyframe, so this is a **seek-and-decode**, not a frame counter. Two measured details
    of this runtime, both of which the first version of this function got wrong:

    - **`CAP_PROP_POS_FRAMES` read *after* `read()` is the position of the NEXT frame,
      not the one just delivered.** Measured on `retail_checkout.mp4`: set to 1400, read,
      then `get` returns 1401. Reading it after the read therefore reports every single
      seek as off by one and stamps every tile with the wrong index. The position is read
      **between** the set and the read, which is the frame about to be delivered.
    - A seek that genuinely cannot be honoured still leaves a position that is not the
      requested one, so **the position actually decoded is what gets stamped on the tile**
      and any difference is reported as a warning. The reader sees the frame the file
      really delivered, and the number on the tile is the number in the report.

    A frame that cannot be decoded is reported by index rather than silently dropped:
    silent blackness in one cell of a twelve-cell grid is, to a person scanning it,
    indistinguishable from a dark corner of the scene.
    """
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open {clip_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    warnings: list[str] = []
    try:
        tiles: list[np.ndarray] = []
        for requested in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, requested)
            actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
            ok, frame = capture.read()
            if not ok or frame is None:
                warnings.append(f"frame {requested}: decode returned no frame")
                continue
            scale = TILE_WIDTH / frame.shape[1]
            tile = cv2.resize(frame, (TILE_WIDTH, max(1, int(frame.shape[0] * scale))))
            tile = _stamp(tile, actual, actual / fps if fps > 0 else 0.0)
            tiles.append(tile)
            if actual != requested:
                warnings.append(
                    f"frame {requested}: the seek landed on {actual}, which is what this "
                    f"tile shows and what is stamped on it")
        if not tiles:
            raise RuntimeError(f"no frame of {clip_path.name} could be decoded")
        return _grid(tiles), warnings
    finally:
        capture.release()


def _grid(tiles: list[np.ndarray]) -> np.ndarray:
    """Lay tiles out `COLUMNS` wide, centring a short final row on a blank row.

    A ragged last row is a misalignment, and an aligned grid is what lets a reader say
    "the third tile of the second row" instead of counting. Tiles are padded to the
    tallest and widest in the grid rather than to their own size, so a sheet never has a
    half-height cell.
    """
    height = max(tile.shape[0] for tile in tiles)
    width = max(tile.shape[1] for tile in tiles)
    padded: list[np.ndarray] = []
    for tile in tiles:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:tile.shape[0], :tile.shape[1]] = tile
        padded.append(canvas)
    rows: list[np.ndarray] = []
    for start in range(0, len(padded), COLUMNS):
        chunk = padded[start:start + COLUMNS]
        while len(chunk) < COLUMNS:
            chunk.append(np.zeros((height, width, 3), dtype=np.uint8))
        rows.append(np.hstack(chunk))
    return np.vstack(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the contact sheets the M0 footage review cites.")
    parser.add_argument("--clip", default=None,
                        help="only this clip id (default: every pinned clip)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    entries = media_entries()
    if args.clip is not None:
        entries = [e for e in entries if e["id"] == args.clip]
        if not entries:
            known = [e["id"] for e in media_entries()]
            raise SystemExit(f"no clip {args.clip!r} in config/models.json; have {known}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    total_warnings = 0
    for entry in entries:
        clip_id = entry["id"]
        passes = CLIP_PASSES.get(clip_id)
        if not passes:
            print(f"SKIP  {clip_id}: no CLIP_PASSES entry, so this tool has no frame list "
                  f"to decode. The M0 report cites no sheet for this clip.")
            continue
        clip_path = REPO_ROOT / entry["path"]
        if not clip_path.is_file():
            print(f"SKIP  {clip_id}: {clip_path} is absent. Provision the tree first; "
                  f"this tool never downloads anything.")
            continue
        for pass_name, indices in passes.items():
            sheet, warnings = build_sheet(clip_path, indices)
            target = args.output_dir / f"{clip_id}_{pass_name}.jpg"
            cv2.imwrite(str(target), sheet,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            written += 1
            total_warnings += len(warnings)
            print(f"WROTE {target.relative_to(REPO_ROOT)}  {len(indices)} frames "
                  f"{indices}  {sheet.shape[1]}x{sheet.shape[0]}")
            for warning in warnings:
                print(f"      WARN  {warning}")
    print(f"\n{written} sheet(s) written, {total_warnings} warning(s). The frame lists "
          f"above are the ones docs/qualification/M0-content-gate.md cites; this tool "
          f"invents no frames of its own.")
    return 1 if total_warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())