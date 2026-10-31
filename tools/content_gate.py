"""The M5 content gate: what footage exists, measured from the files, and what is missing.

**The requirement, and the decision this module implements.** Spec section 4 asks for
**three 45-90 s clips per pack, four packs, twelve clips in total**. The material actually
provisioned is shorter and less of it. This module does not round that up, trim a clip to
fit a band, or invent a clip that is not on disk: it measures every file with
`cv2.VideoCapture`, records what it measured, and states the gap.

## What is measured, and what cannot be

- **Duration, resolution, fps and frame count** come from `cv2.VideoCapture` on the file,
  not from `assets/scenarios/*/manifest.json`. A manifest can say 45 s; the file is what a
  decoder will actually play.
- **The SHA-256 and byte count** are read from the file, so a re-encoded clip with the same
  name is a different clip and the inventory says so.
- **Whether the application uses it** is read from the application's own source: a clip is
  "used" when its relative path or file name appears in `app/**/*.py`. That is a fact about
  the code that will run, not about a manifest's wish list.
- **Licence and rights status are NOT properties of the bytes.** A video file does not
  carry a grant, and scanning for the word "license" in a container would produce a claim
  with nothing behind it. So each clip is recorded as **unverified** unless a licence file
  sits beside it, and even then the status is "licence file present, clearance not
  established". A sidecar records provenance; it is not a clearance, and this module never
  reports one as cleared. This is the honest measurement, and it is why rights is an unmet
  gate rather than a checkbox.
- **A licence sidecar is clip-scoped.** It is named `<clip>.<suffix>` (for example
  `people-detection.mp4.LICENSE`) or `<stem>.<suffix>`. A bare `LICENSE` in the folder is
  not used: one file cannot be attributed to every clip beside it, and treating it that way
  is exactly the assumption this gate exists to refuse.

## The shortfall this module exists to report

Provisioned: **9 clips, 1 pack, 4 clips inside the 45-90 s band**. Required: **12 clips,
4 packs, every clip inside the band**. The verdicts below carry those numbers; nothing in
this module turns a shortfall into a pass.

**The sub-45 s clips are a recorded deviation, not an in-band result.** The operator chose
to provision six Intel sample-videos clips, three of which measure under 45 s
(`bottle-detection.mp4`, `car-detection.mp4`, `classroom.mp4`). They are inventoried like
every other clip and the duration verdict counts them as **outside the band**. A product
decision to use a shorter clip does not move the specification's boundary.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
MEDIA_DIR: Final[Path] = REPO_ROOT / "assets" / "media"
SCENARIOS_DIR: Final[Path] = REPO_ROOT / "assets" / "scenarios"
APP_DIR: Final[Path] = REPO_ROOT / "app"

VIDEO_SUFFIXES: Final[tuple[str, ...]] = (
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
)

#: Spec section 4 / design section 5.
SPEC_PACKS: Final[int] = 4
SPEC_CLIPS_PER_PACK: Final[int] = 3
SPEC_MIN_SECONDS: Final[float] = 45.0
SPEC_MAX_SECONDS: Final[float] = 90.0

#: Sidecar names a licence could plausibly live in, checked beside each clip.
LICENCE_SIDECARS: Final[tuple[str, ...]] = (
    "LICENSE", "LICENCE", "LICENSE.txt", "LICENCE.txt", "license.txt", "licence.txt",
)


@dataclass(frozen=True)
class ContentClip:
    """One video file, measured from the bytes on disk."""

    path: str
    sha256: str
    bytes: int
    width: int
    height: int
    fps: float
    frames: int
    duration_s: float
    used_by_app: bool
    used_in: tuple[str, ...]
    licence: str
    rights_status: str

    @property
    def inside_duration_band(self) -> bool:
        """Whether the **measured** duration is in the spec's 45-90 s band."""
        return SPEC_MIN_SECONDS <= self.duration_s <= SPEC_MAX_SECONDS

    @property
    def duration_shortfall_s(self) -> float:
        """Seconds short of the band, or `0.0` when it is inside it.

        A clip longer than the band is over, not short; `duration_excess_s` records that.
        Keeping the two separate is what stops "out of band" becoming one number that
        hides which way it missed.
        """
        return max(0.0, SPEC_MIN_SECONDS - self.duration_s)

    @property
    def duration_excess_s(self) -> float:
        return max(0.0, self.duration_s - SPEC_MAX_SECONDS)

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path, "sha256": self.sha256, "bytes": self.bytes,
            "width": self.width, "height": self.height, "fps": self.fps,
            "frames": self.frames, "duration_s": self.duration_s,
            "inside_duration_band": self.inside_duration_band,
            "duration_shortfall_s": self.duration_shortfall_s,
            "duration_excess_s": self.duration_excess_s,
            "used_by_app": self.used_by_app, "used_in": list(self.used_in),
            "licence": self.licence, "rights_status": self.rights_status,
        }


@dataclass(frozen=True)
class GateVerdict:
    """One specification requirement, what is provisioned, and the gap."""

    requirement: str
    required: str
    provisioned: str
    met: bool
    gap: str

    def to_json(self) -> dict[str, Any]:
        return {"requirement": self.requirement, "required": self.required,
                "provisioned": self.provisioned, "met": self.met, "gap": self.gap}


@dataclass(frozen=True)
class ContentInventory:
    """Every clip under `assets/media/` and `assets/scenarios/`, as measured."""

    scenario: str
    clips: tuple[ContentClip, ...]
    packs_provisioned: int
    packs_required: int = SPEC_PACKS
    clips_required_per_pack: int = SPEC_CLIPS_PER_PACK
    min_duration_s: float = SPEC_MIN_SECONDS
    max_duration_s: float = SPEC_MAX_SECONDS

    @property
    def clips_required_total(self) -> int:
        return self.packs_required * self.clips_required_per_pack

    @property
    def clips_in_band(self) -> tuple[ContentClip, ...]:
        return tuple(clip for clip in self.clips if clip.inside_duration_band)

    @property
    def clips_used_by_app(self) -> tuple[ContentClip, ...]:
        return tuple(clip for clip in self.clips if clip.used_by_app)

    @property
    def clips_with_unverified_rights(self) -> tuple[ContentClip, ...]:
        """Every clip whose rights are *not* recorded as cleared.

        A licence sidecar does **not** clear a clip: its status reads "licence file
        present; clearance not established". Counting only `"not cleared"` here would
        silently reclassify every sidecarred clip as cleared, which would upgrade a
        provenance record into a clearance. Only a literal `cleared` status counts.
        """
        return tuple(clip for clip in self.clips
                     if not clip.rights_status.startswith("cleared"))

    def verdicts(self) -> tuple[GateVerdict, ...]:
        """One verdict per requirement. **Nothing here is rounded in the requirement's
        favour**: a count is the measured count, a gap names the missing items."""
        clip_count = len(self.clips)
        in_band = len(self.clips_in_band)
        cleared = clip_count - len(self.clips_with_unverified_rights)
        with_sidecar = sum(1 for clip in self.clips
                           if clip.rights_status.startswith("licence file present"))
        without_sidecar = sum(1 for clip in self.clips
                              if clip.rights_status.startswith("not cleared"))
        return (
            GateVerdict(
                requirement="three 45-90 s clips per pack",
                required=f"{SPEC_CLIPS_PER_PACK} clips per pack, "
                         f"{SPEC_MIN_SECONDS:.0f}-{SPEC_MAX_SECONDS:.0f} s each",
                provisioned=(f"{clip_count} clip(s), {len(self.clips_in_band)} inside "
                             f"the band"),
                met=(clip_count >= self.clips_required_total
                     and in_band == clip_count),
                gap=("missing "
                     f"{max(0, self.clips_required_total - clip_count)} clip(s) against "
                     f"the four-pack requirement; "
                     f"{clip_count - in_band} provisioned clip(s) outside the band"),
            ),
            GateVerdict(
                requirement="four scenario packs",
                required=f"{self.packs_required} packs",
                provisioned=f"{self.packs_provisioned} pack(s)",
                met=self.packs_provisioned >= self.packs_required,
                gap=f"missing {max(0, self.packs_required - self.packs_provisioned)} "
                    f"pack(s)",
            ),
            GateVerdict(
                requirement="every clip inside the 45-90 s band",
                required=f"{clip_count} of {clip_count} inside the band",
                provisioned=f"{in_band} of {clip_count} inside the band",
                met=(clip_count > 0 and in_band == clip_count),
                gap=", ".join(
                    f"{clip.path} is {clip.duration_excess_s:g} s over "
                    f"({clip.duration_s:.5f} s)" if clip.duration_excess_s
                    else f"{clip.path} is {clip.duration_shortfall_s:g} s short"
                    for clip in self.clips if not clip.inside_duration_band)
                    or "none",
            ),
            GateVerdict(
                requirement="rights-cleared footage",
                required=f"{clip_count} cleared clip(s)",
                provisioned=f"{cleared} cleared clip(s)",
                met=cleared == clip_count and clip_count > 0,
                gap=(f"{len(self.clips_with_unverified_rights)} clip(s) are not cleared: "
                     f"{with_sidecar} carry a licence sidecar (status 'licence file "
                     f"present; clearance not established'), {without_sidecar} have no "
                     f"licence file beside them. A sidecar records provenance, not "
                     f"clearance; clearance is an operator/legal dependency"),
            ),
        )

    @property
    def met(self) -> bool:
        return all(verdict.met for verdict in self.verdicts())

    def to_json(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "requirement": {
                "packs": self.packs_required,
                "clips_per_pack": self.clips_required_per_pack,
                "clips_total": self.clips_required_total,
                "min_duration_s": self.min_duration_s,
                "max_duration_s": self.max_duration_s,
            },
            "clips": [clip.to_json() for clip in self.clips],
            "verdicts": [verdict.to_json() for verdict in self.verdicts()],
            "met": self.met,
        }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measured(path: Path) -> tuple[int, int, float, int, float]:
    """`(width, height, fps, frames, duration_s)` from the decoder, not a manifest."""
    import cv2  # local: the gate needs a decoder, not the rest of the application

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cv2.VideoCapture cannot open {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"{path}: the decoder reports a {width}x{height} frame")
    if fps <= 0:
        raise RuntimeError(f"{path}: the decoder reports {fps} fps")
    if frames <= 0:
        raise RuntimeError(f"{path}: the decoder reports {frames} frames")
    # **The measured duration is frames / fps**, the same arithmetic for every clip, so
    # the comparison is between like and like. It is not rounded.
    return width, height, fps, frames, frames / fps


def _usage(path: Path) -> tuple[bool, tuple[str, ...]]:
    """Which application sources name this clip.

    Read from `app/`, because the question is "will the booth run it", not "does a pack
    manifest mention it". A clip that only a manifest names is provisioned, not used.
    """
    if not APP_DIR.exists():
        return False, ()
    needles = {path.name, _relative(path)}
    used_in = []
    for source in sorted(APP_DIR.rglob("*")):
        if source.suffix not in (".py", ".qml", ".json"):
            continue
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(needle in text for needle in needles):
            used_in.append(source.resolve().relative_to(REPO_ROOT).as_posix())
    return bool(used_in), tuple(used_in)


def _sidecar(path: Path) -> Path | None:
    """The clip-scoped licence sidecar beside `path`, or `None`.

    Accepted names are `<clip>.<suffix>` (`people-detection.mp4.LICENSE`) and
    `<stem>.<suffix>` (`people-detection.LICENSE`). A bare `LICENSE` in the directory is
    **not** accepted: it cannot be attributed to one clip, and treating it as covering
    every clip in the folder is the unearned assumption this gate refuses.
    """
    for name in LICENCE_SIDECARS:
        for candidate in (path.with_name(f"{path.name}.{name}"),
                          path.with_name(f"{path.stem}.{name}")):
            if candidate.exists():
                return candidate
    return None


def _rights(path: Path) -> tuple[str, str]:
    """`(licence, rights_status)` from what is **beside the file**, never assumed."""
    sidecar = _sidecar(path)
    if sidecar is not None:
        text = sidecar.read_text(encoding="utf-8", errors="replace").strip()
        first = text.splitlines()[0] if text else ""
        return (first or "licence file present but empty",
                "licence file present; clearance not established")
    return ("unverified - no licence text is embedded in the file",
            "not cleared - no licence file sits beside the clip")


def _video_files(*directories: Path) -> list[Path]:
    found: list[Path] = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                found.append(path)
    return found


def _pack_count(scenarios_dir: Path = SCENARIOS_DIR) -> int:
    """Packs with a committed manifest. A directory with no manifest is not a pack."""
    if not scenarios_dir.exists():
        return 0
    return sum(1 for path in scenarios_dir.glob("*/manifest.json") if path.is_file())


def content_inventory(scenario: str = "retail_sanity", *,
                      media_dir: Path | None = None,
                      scenarios_dir: Path | None = None) -> ContentInventory:
    """Measure every clip and return the inventory and its gate verdicts.

    `scenario` labels the inventory; the clip set is **everything on disk** under
    `assets/media/` and `assets/scenarios/`, because a clip that exists but belongs to no
    pack is exactly the kind of production the gate must not lose.

    The two directories are parameters so a test can prove that **adding or removing a
    file changes the inventory**: a committed "declared" table that cannot fail on an
    added clip is a table nobody is maintaining.
    """
    media = MEDIA_DIR if media_dir is None else Path(media_dir)
    scenarios = SCENARIOS_DIR if scenarios_dir is None else Path(scenarios_dir)
    clips = []
    for path in _video_files(media, scenarios):
        width, height, fps, frames, duration = _measured(path)
        used, used_in = _usage(path)
        licence, rights = _rights(path)
        clips.append(ContentClip(
            path=_relative(path), sha256=_sha256(path), bytes=path.stat().st_size,
            width=width, height=height, fps=fps, frames=frames, duration_s=duration,
            used_by_app=used, used_in=used_in, licence=licence, rights_status=rights))
    return ContentInventory(scenario=str(scenario), clips=tuple(clips),
                            packs_provisioned=_pack_count(scenarios))


def main(argv: list[str] | None = None) -> int:
    """Print the inventory, the verdicts and the gap. Exit 1 while the gate is unmet."""
    inventory = content_inventory("retail_sanity")
    print(f"M5 content gate - scenario {inventory.scenario}")
    print(f"required: {inventory.clips_required_total} clips "
          f"({inventory.packs_required} packs x {inventory.clips_required_per_pack}), "
          f"each {inventory.min_duration_s:.0f}-{inventory.max_duration_s:.0f} s")
    print(f"provisioned: {len(inventory.clips)} clip(s), "
          f"{inventory.packs_provisioned} pack(s)")
    print()
    header = (f"{'clip':52s} {'resolution':11s} {'fps':>7s} {'frames':>6s} "
              f"{'duration s':>10s} {'band':>5s} {'app':>4s}")
    print(header)
    print("-" * len(header))
    for clip in inventory.clips:
        print(f"{clip.path:52s} {clip.width}x{clip.height:<6d} {clip.fps:7.3f} "
              f"{clip.frames:6d} {clip.duration_s:10.5f} "
              f"{'yes' if clip.inside_duration_band else 'NO':>5s} "
              f"{'yes' if clip.used_by_app else 'no':>4s}")
    print()
    for verdict in inventory.verdicts():
        print(f"[{'MET' if verdict.met else 'NOT MET'}] {verdict.requirement}")
        print(f"    required:    {verdict.required}")
        print(f"    provisioned: {verdict.provisioned}")
        print(f"    gap:         {verdict.gap}")
    print()
    print("rights:")
    for clip in inventory.clips:
        print(f"  {clip.path}: {clip.licence} / {clip.rights_status}")
    print()
    print("VERDICT: " + ("MET" if inventory.met else "NOT MET - the shortfall is reported, "
                                                 "not rounded up"))
    return 0 if inventory.met else 1


if __name__ == "__main__":  # pragma: no cover - the operator's entry point
    sys.exit(main())
