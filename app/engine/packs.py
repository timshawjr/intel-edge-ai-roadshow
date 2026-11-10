"""Scenario packs, loaded from their committed manifests and tied to the clip catalogue.

Each `assets/scenarios/<pack>/manifest.json` is the pack's runtime configuration now, not
only an inventory. The loader turns its `zones` and `event_rules` into the same
`ScenarioManifest` the event machine already consumes, and checks that every clip it names
is a real catalogue record assigned to that pack.

**The honest limitation lives in the data.** A pack declares `unexpressible_rules`, a
`scope_note` and an `explanation`, and this module refuses to send a pack whose
user-visible strings claim a rule the event machine cannot compute. The machine's only
event kinds are `zone_occupancy`, `candidate_transfer` and `candidate_return`; there is no
line-crossing kind and no dwell-event kind. A pack that mentions "line crossing" or
"dwell" as something it *delivers* is a false claim on a booth screen, so the loader
rejects it rather than letting the string reach the UI.

**`zones_for_event_rules()` is the validation path.** `Zone` validates every ROI at load,
naming the slot, and the loader calls the manifest's own method so a malformed ROI in a
pack fails here rather than drawing a rectangle in the wrong place for a session.

Pure: `json` + the contract modules. No detector, no decoder, no Qt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from app.contracts.scenario_manifest import ScenarioManifest
from app.engine.clips import ClipRecord, Selection, load_catalogue, resolve_selection
from app.engine.zones import Zone
from app.inference.pinned import REPO_ROOT

__all__ = [
    "Pack",
    "PackError",
    "available_packs",
    "default_pack",
    "load_all_packs",
    "load_pack",
    "pack_for_clip",
    "pack_ids",
    "selection_for",
]

SCENARIOS_DIR: Final[Path] = REPO_ROOT / "assets" / "scenarios"

#: Event-rule keys that are not ROIs. Anything else whose value is a 4-number sequence is
#: treated as an ROI slot, exactly as `RetailEventMachine._build_zones` does.
_ROUTE_KEYS: Final[frozenset[str]] = frozenset({
    "origin_slot", "destination_slot", "min_dwell_seconds", "max_transition_seconds",
})
_PRESENCE_KEYS: Final[frozenset[str]] = frozenset({
    "presence_zone", "min_presence_seconds",
})
_SCALAR_KEYS: Final[frozenset[str]] = _ROUTE_KEYS | _PRESENCE_KEYS | {"tracked_classes"}

#: Words that describe a rule the engine does not compute. They are allowed in the
#: `unexpressible_rules` list and in explanatory prose that says they are NOT delivered;
#: the check below is about a pack claiming to deliver them. Kept as a separate,
#: explicitly-named list so a future event kind can remove one with a stated change.
_UNSUPPORTED_DELIVERY_WORDS: Final[tuple[tuple[str, str], ...]] = (
    ("line crossing", "no line-crossing event kind"),
    ("line-crossing", "no line-crossing event kind"),
    ("dwell event", "no dwell-event kind"),
    ("dwell measurement", "the event carries no duration"),
)


class PackError(ValueError):
    """A pack manifest cannot be loaded, with a reason a presenter can read."""


@dataclass(frozen=True)
class Pack:
    """One scenario pack: its manifest, its clips, and the words the booth shows."""

    id: str
    manifest: ScenarioManifest
    clips: tuple[ClipRecord, ...]
    business_metric: str
    explanation: str
    delivered_event: str
    content_assignment: str
    unexpressible_rules: tuple[str, ...]

    @property
    def default_clip(self) -> ClipRecord:
        """The pack's default clip: its first declared clip, in catalogue order."""
        return self.clips[0]

    @property
    def clip_ids(self) -> tuple[str, ...]:
        return tuple(record.id for record in self.clips)

    def clip(self, clip_id: str) -> ClipRecord:
        for record in self.clips:
            if record.id == clip_id:
                return record
        raise PackError(
            f"pack {self.id!r} has no clip {clip_id!r}; its clips are {list(self.clip_ids)}"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "clip_ids": list(self.clip_ids),
            "default_clip": self.default_clip.id,
            "business_metric": self.business_metric,
            "explanation": self.explanation,
            "delivered_event": self.delivered_event,
            "content_assignment": self.content_assignment,
            "unexpressible_rules": list(self.unexpressible_rules),
            "manifest": self.manifest.to_json(),
        }


def _payload(pack_id: str) -> dict[str, Any]:
    path = SCENARIOS_DIR / str(pack_id) / "manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PackError(
            f"scenario pack {pack_id!r} has no readable manifest at {path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise PackError(f"{path} is not valid JSON: {exc}") from exc


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PackError(f"manifest has no non-empty {key!r}")
    return value


def _check_delivery_words(pack_id: str, *texts: str) -> None:
    """Refuse a user-visible string that claims a rule the machine cannot compute.

    A string may *name* the limitation ("line crossing is NOT delivered"); what it may not
    do is present the rule as the thing being measured. The check is deliberately simple
    and quoted: the false form is a claim, and the loader cannot parse English. So it
    requires every unsupported phrase to be preceded, in the same string, by a negation -
    "NOT", "no ", "cannot", "is not", "not computed", "not delivered" or "does not".
    """
    lowered = " ".join(texts).lower()
    for phrase, reason in _UNSUPPORTED_DELIVERY_WORDS:
        start = 0
        while True:
            at = lowered.find(phrase, start)
            if at < 0:
                break
            context = lowered[max(0, at - 120): at + len(phrase) + 120]
            negated = any(token in context for token in (
                "not", "no ", "cannot", "never", "unexpressible", "does not",
            ))
            if not negated:
                raise PackError(
                    f"pack {pack_id!r} mentions {phrase!r} without saying it is not "
                    f"delivered ({reason}). A string that reads as a delivered rule the "
                    f"engine does not compute is the false claim this loader exists to "
                    f"stop."
                )
            start = at + len(phrase)


def _manifest_from_payload(pack_id: str, payload: dict[str, Any],
                    clips: tuple[ClipRecord, ...]) -> ScenarioManifest:
    zones_payload = payload.get("zones") or []
    zones: list[Zone] = []
    for entry in zones_payload:
        if not isinstance(entry, dict):
            raise PackError(f"pack {pack_id!r}: every zone must be an object; got {entry!r}")
        zones.append(Zone(name=str(entry.get("name") or ""),
                          kind=str(entry.get("kind") or "area"),
                          roi=tuple(entry.get("roi") or ())))
    event_rules = payload.get("event_rules")
    if not isinstance(event_rules, dict):
        raise PackError(f"pack {pack_id!r} has no event_rules object")
    # JSON gives lists; the manifest contract and the event machine use tuples. Normalise
    # once, at load, so a test that compares `tracked_classes` to a tuple is comparing
    # against the same shape the shipped hardcoded pack produced.
    normalised_rules: dict[str, Any] = {}
    for key, value in event_rules.items():
        if isinstance(value, list):
            normalised_rules[key] = tuple(value)
        else:
            normalised_rules[key] = value

    manifest = ScenarioManifest(
        id=pack_id,
        version=int(payload.get("version") or 1),
        title=_text(payload, "title"),
        business_line=_text(payload, "business_line"),
        class_allowlist=tuple(payload.get("class_allowlist") or ()),
        zones=tuple(zones),
        event_rules=normalised_rules,
        source_ids=tuple(record.id for record in clips),
        conf_threshold=float(payload.get("conf_threshold", 0.25)),
        iou_threshold=float(payload.get("iou_threshold", 0.45)),
        scope_note=str(payload.get("scope_note") or ""),
    )
    # Build every ROI through the one supported path. A bad ROI fails at load, naming
    # the slot, before the machine or the drawer ever reads it.
    built = manifest.zones_for_event_rules()
    rules = manifest.event_rules

    tracked = tuple(rules.get("tracked_classes") or ())
    if not tracked:
        raise PackError(f"pack {pack_id!r} event_rules has no tracked_classes")
    unknown = sorted(set(tracked) - set(manifest.class_allowlist))
    if unknown:
        raise PackError(
            f"pack {pack_id!r} tracks {unknown}, which are not in class_allowlist "
            f"{sorted(manifest.class_allowlist)}"
        )

    declares_route = "origin_slot" in rules or "destination_slot" in rules
    if declares_route:
        for key in ("origin_slot", "destination_slot", "min_dwell_seconds",
                    "max_transition_seconds"):
            if key not in rules:
                raise PackError(f"pack {pack_id!r} declares a transfer route but no {key!r}")
        for key in ("origin_slot", "destination_slot"):
            if str(rules[key]) not in built:
                raise PackError(
                    f"pack {pack_id!r}: {key} is {rules[key]!r}, which is not an ROI"
                )
        if str(rules["origin_slot"]) == str(rules["destination_slot"]):
            raise PackError(f"pack {pack_id!r}: origin_slot and destination_slot are the same")

    if "presence_zone" in rules:
        name = str(rules["presence_zone"])
        if name not in built:
            raise PackError(
                f"pack {pack_id!r}: presence_zone {name!r} is not an ROI in event_rules"
            )
        if "min_presence_seconds" not in rules:
            raise PackError(
                f"pack {pack_id!r} names presence_zone but no min_presence_seconds"
            )
    if not declares_route and "presence_zone" not in rules:
        raise PackError(
            f"pack {pack_id!r} configures no rule: neither a transfer route nor a "
            f"presence_zone; a machine that can never emit would read zero forever"
        )

    for key, value in rules.items():
        if key in _SCALAR_KEYS:
            continue
        if isinstance(value, (tuple, list)) and len(value) == 4:
            if key not in built:
                raise PackError(f"pack {pack_id!r}: ROI slot {key!r} did not validate")
            continue
        raise PackError(
            f"pack {pack_id!r} event_rules has an unrecognised scalar {key!r}={value!r}"
        )
    return manifest


def load_pack(pack_id: str) -> Pack:
    """Load one pack, validate its geometry and tie it to the catalogue."""
    pack_id = str(pack_id)
    payload = _payload(pack_id)
    declared_id = _text(payload, "id")
    if declared_id != pack_id:
        raise PackError(
            f"{SCENARIOS_DIR / pack_id / 'manifest.json'} declares id {declared_id!r}, "
            f"not {pack_id!r}"
        )

    raw_clips = payload.get("clip_ids")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise PackError(f"pack {pack_id!r} has no clip_ids list; a pack with no clip has nothing to play")

    catalogue = {record.id: record for record in load_catalogue()}
    clips: list[ClipRecord] = []
    for clip_id in raw_clips:
        record = catalogue.get(str(clip_id))
        if record is None:
            raise PackError(
                f"pack {pack_id!r} names clip {clip_id!r}, which is not in "
                f"config/clips.json"
            )
        if record.pack != pack_id:
            raise PackError(
                f"pack {pack_id!r} names clip {record.id!r}, which the catalogue assigns "
                f"to {record.pack!r}"
            )
        clips.append(record)
    if len({record.id for record in clips}) != len(clips):
        raise PackError(f"pack {pack_id!r} lists a clip more than once")

    manifest = _manifest_from_payload(pack_id, payload, tuple(clips))
    business_metric = _text(payload, "business_metric")
    explanation = _text(payload, "explanation")
    delivered_event = _text(payload, "delivered_event")
    content_assignment = _text(payload, "content_assignment")
    raw_unexpressible = payload.get("unexpressible_rules") or []
    if not isinstance(raw_unexpressible, list):
        raise PackError(f"pack {pack_id!r}: unexpressible_rules must be a list")
    unexpressible = tuple(str(item) for item in raw_unexpressible)
    if not unexpressible:
        raise PackError(
            f"pack {pack_id!r} does not state any unexpressible rule. If none of the "
            f"specification's rules are missing, say so with an empty list only after "
            f"checking the engine's kinds; today every spec pack has one or more."
        )

    # **Each user-visible string is checked on its own.** Joining them let a negation in
    # one sentence excuse a false delivery claim in another: the smart pack's explanation
    # says "NOT a line crossing" and its delivered-event sentence could then claim one
    # without the guard noticing.
    for text in (manifest.business_line, business_metric, explanation, delivered_event,
                 manifest.scope_note, content_assignment):
        _check_delivery_words(pack_id, text)
    return Pack(id=pack_id, manifest=manifest, clips=tuple(clips),
                business_metric=business_metric, explanation=explanation,
                delivered_event=delivered_event,
                content_assignment=content_assignment,
                unexpressible_rules=unexpressible)


def available_packs() -> tuple[str, ...]:
    """Pack directories that carry a manifest, sorted for a stable CLI."""
    if not SCENARIOS_DIR.is_dir():
        return ()
    return tuple(sorted(
        path.parent.name for path in SCENARIOS_DIR.glob("*/manifest.json")
        if path.is_file()
    ))


def pack_ids() -> tuple[str, ...]:
    """The pack ids `--pack` accepts: manifests on disk, in catalogue order.

    Catalogue order is used where every manifest is assigned a clip; a manifest with no
    clip is refused by `load_pack`, so it cannot appear here as a selectable pack.
    """
    from app.engine.clips import pack_ids as catalogue_pack_ids

    ordered = catalogue_pack_ids()
    return tuple(pack_id for pack_id in ordered if pack_id in available_packs())


def load_all_packs() -> tuple[Pack, ...]:
    return tuple(load_pack(pack_id) for pack_id in available_packs())


def default_pack() -> Pack:
    """The pack the default invocation runs: `retail_sanity`."""
    from app.engine.clips import default_selection

    return load_pack(default_selection().pack_id)


def pack_for_clip(clip: str) -> Pack:
    selection = resolve_selection(clip=clip)
    return load_pack(selection.pack_id)


def selection_for(clip: str | None = None, pack: str | None = None, *,
                  scenario: str | None = None,
                  require_files: bool = False) -> tuple[Selection, Pack]:
    """Resolve the CLI selection once, and return it with its validated pack."""
    selection = resolve_selection(clip=clip, pack=pack, scenario=scenario,
                                  require_files=require_files)
    return selection, load_pack(selection.pack_id)
