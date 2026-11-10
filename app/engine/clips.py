"""The clip catalogue: every provisioned clip, measured once, selectable by id.

`config/clips.json` is the committed record this module reads. It exists because the
application used to name one clip in code (`RETAIL_CHECKOUT_CLIP`) while nine sat in
`assets/media/`; a presenter could not select the other eight, and a pack could not say
which clips belonged to it. The catalogue makes the selection a data fact rather than a
`Path` literal on the run path.

**Measured, not declared.** `tests/test_clip_catalogue.py` re-reads every file with
`cv2.VideoCapture`, re-hashes it with SHA-256, and fails if any field here has drifted.
Nothing in this module opens a video: at run time it reads a JSON record and answers
"which file does `--clip <id>` mean". The measurement lives in the test, where it can
actually fail.

**Licence status is not upgraded here.** A clip with a clip-scoped `*.LICENSE` sidecar
carries `CC BY 4.0` and the repository URL from that sidecar; a clip without one carries
`unverified`. `distribution_status` says what that means - a sidecar records provenance,
not clearance. No record in this module can say `cleared`, because no file on disk grants
that.

**No network and no detector.** Pure `json` + `pathlib`, so a UI or a self-test can ask
what clips exist on a machine where the model is not provisioned yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from app.inference.pinned import REPO_ROOT

__all__ = [
    "CLIPS_JSON",
    "ClipRecord",
    "Selection",
    "SelectionError",
    "clip_ids",
    "default_selection",
    "load_catalogue",
    "pack_ids",
    "resolve_selection",
]

CLIPS_JSON: Final[Path] = REPO_ROOT / "config" / "clips.json"

#: Spec section 4 / design section 5, restated where the catalogue is read. A duration is
#: in the band when it is **measured** between these two seconds, inclusive.
SPEC_MIN_SECONDS: Final[float] = 45.0
SPEC_MAX_SECONDS: Final[float] = 90.0


class SelectionError(ValueError):
    """A requested clip or pack cannot be resolved, with a reason a presenter can read."""


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string; got {value!r}")
    return value


def _require_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}; got {value!r}")
    return value


def _require_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number; got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")) or number <= 0.0:
        raise ValueError(f"{name} must be a finite number > 0; got {value!r}")
    return number


@dataclass(frozen=True)
class ClipRecord:
    """One provisioned clip, as the catalogue measured it.

    `in_duration_band` is stored **and checked** against the measured duration rather
    than trusted: a hand-edited `true` next to a 30 s duration would be a claim the file
    does not support, and the whole point of the catalogue is that the numbers are the
    file's.
    """

    id: str
    path: str
    pack: str
    sha256: str
    bytes: int
    measured_width: int
    measured_height: int
    measured_fps: float
    measured_frames: int
    measured_duration_s: float
    license: str
    license_source: str
    distribution_status: str
    in_duration_band: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "path", _require_text("path", self.path))
        object.__setattr__(self, "pack", _require_text("pack", self.pack))
        digest = _require_text("sha256", self.sha256).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{self.id}: sha256 must be 64 hex characters; got {self.sha256!r}")
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "bytes", _require_int("bytes", self.bytes))
        object.__setattr__(self, "measured_width",
                           _require_int("measured_width", self.measured_width))
        object.__setattr__(self, "measured_height",
                           _require_int("measured_height", self.measured_height))
        object.__setattr__(self, "measured_fps",
                           _require_number("measured_fps", self.measured_fps))
        object.__setattr__(self, "measured_frames",
                           _require_int("measured_frames", self.measured_frames))
        duration = _require_number("measured_duration_s", self.measured_duration_s)
        object.__setattr__(self, "measured_duration_s", duration)
        object.__setattr__(self, "license", _require_text("license", self.license))
        object.__setattr__(self, "license_source",
                           _require_text("license_source", self.license_source))
        object.__setattr__(self, "distribution_status",
                           _require_text("distribution_status", self.distribution_status))
        if not isinstance(self.in_duration_band, bool):
            raise ValueError(f"{self.id}: in_duration_band must be a bool")
        measured_in_band = SPEC_MIN_SECONDS <= duration <= SPEC_MAX_SECONDS
        if self.in_duration_band != measured_in_band:
            raise ValueError(
                f"{self.id}: in_duration_band is {self.in_duration_band} but the measured "
                f"duration {duration} s is {'inside' if measured_in_band else 'outside'} "
                f"the {SPEC_MIN_SECONDS:g}-{SPEC_MAX_SECONDS:g} s band. The flag must "
                f"follow the measurement, not a wish."
            )

    @property
    def absolute_path(self) -> Path:
        """The file the record names, rooted at the repository."""
        return REPO_ROOT / self.path

    def exists(self) -> bool:
        return self.absolute_path.is_file()


@dataclass(frozen=True)
class Selection:
    """A resolved `--clip` / `--pack` pair: one clip, and the pack it runs under."""

    clip: ClipRecord
    pack_id: str

    def __post_init__(self) -> None:
        _require_text("pack_id", self.pack_id)
        if self.clip.pack != self.pack_id:
            raise ValueError(
                f"clip {self.clip.id!r} belongs to pack {self.clip.pack!r}, not "
                f"{self.pack_id!r}"
            )


def _payload() -> dict[str, Any]:
    try:
        return json.loads(CLIPS_JSON.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(
            f"{CLIPS_JSON} could not be read: {exc}. The catalogue is the only list of "
            f"selectable clips; without it the application would have to guess a path."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{CLIPS_JSON} is not valid JSON: {exc}") from exc


def load_catalogue() -> tuple[ClipRecord, ...]:
    """Every record in `config/clips.json`, validated, in catalogue order.

    Order is meaningful: the first clip of a pack is that pack's default, and the
    catalogue's own `default_clip` names the application's default. Nothing here checks
    that the files exist, because `--selftest` must work on an unprovisioned machine;
    `resolve_selection(require_files=True)` is the check a run makes.
    """
    payload = _payload()
    entries = payload.get("clips")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{CLIPS_JSON} has no 'clips' list")
    records = tuple(ClipRecord(**entry) for entry in entries)
    ids = [record.id for record in records]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"{CLIPS_JSON} has duplicate clip id(s) {duplicates}")
    return records


def clip_ids() -> tuple[str, ...]:
    """The ids `--clip` accepts, in catalogue order."""
    return tuple(record.id for record in load_catalogue())


def pack_ids() -> tuple[str, ...]:
    """The pack ids the catalogue assigns clips to, in first-appearance order."""
    seen: list[str] = []
    for record in load_catalogue():
        if record.pack not in seen:
            seen.append(record.pack)
    return tuple(seen)


def _by_id(catalogue: tuple[ClipRecord, ...]) -> dict[str, ClipRecord]:
    return {record.id: record for record in catalogue}


def _canonical_clip_id(wanted: str) -> str:
    """Accept `retail_checkout` as well as `retail-checkout`, without hiding a typo.

    The catalogue's ids are the file stems. The underscore spelling is what
    `config/models.json` used for its media ids, so an operator who has read that file
    will type it; translating only `_` to `-` keeps the catalogue's ids authoritative
    while not punishing the older spelling.
    """
    text = str(wanted).strip()
    if text in _by_id(load_catalogue()):
        return text
    return text.replace("_", "-")


def resolve_selection(clip: str | None = None, pack: str | None = None, *,
                      scenario: str | None = None,
                      require_files: bool = False) -> Selection:
    """Resolve the CLI's selection to exactly one clip and one pack.

    Rules, in order:
    - `--clip` names a clip; its pack is the catalogue's assignment unless `--pack` also
      names a different one, which is refused rather than silently overridden.
    - `--pack` with no `--clip` selects that pack's first clip (the pack manifest's
      default).
    - neither selects `default_clip` - `retail-checkout` - so the default invocation is
      unchanged from M1.
    - `--scenario` is the legacy spelling. `retail` means `retail_sanity`; any other pack
      id is accepted as an alias for `--pack` only when `--pack` is absent.
    """
    catalogue = load_catalogue()
    index = _by_id(catalogue)

    if pack is None and scenario not in (None, "", "retail"):
        pack = str(scenario)
    if pack == "retail":
        pack = "retail_sanity"

    if clip is not None:
        wanted = _canonical_clip_id(clip)
        record = index.get(wanted)
        if record is None:
            raise SelectionError(
                f"--clip {clip!r} is not in the catalogue; ids are "
                f"{sorted(index)}"
            )
        if pack is not None and pack != record.pack:
            raise SelectionError(
                f"--clip {record.id!r} belongs to pack {record.pack!r}, but --pack "
                f"{pack!r} was also given. Ask for one or the other; a clip cannot run "
                f"under a pack it is not assigned to."
            )
        selected_pack = record.pack
    elif pack is not None:
        matches = [record for record in catalogue if record.pack == pack]
        if not matches:
            raise SelectionError(
                f"--pack {pack!r} has no clips; packs are {list(pack_ids())}"
            )
        record = matches[0]
        selected_pack = pack
    else:
        default_id = _canonical_clip_id(str(_payload().get("default_clip") or ""))
        record = index.get(default_id)
        if record is None:
            raise SelectionError(
                f"the catalogue's default_clip {default_id!r} is not one of its clips; "
                f"ids are {sorted(index)}"
            )
        selected_pack = record.pack

    if require_files and not record.exists():
        raise SelectionError(
            f"clip {record.id!r} is catalogued at {record.path} but that file is not "
            f"provisioned on this machine"
        )
    return Selection(clip=record, pack_id=selected_pack)


def default_selection() -> Selection:
    """The default invocation: `retail-checkout` under `retail_sanity`."""
    return resolve_selection()
