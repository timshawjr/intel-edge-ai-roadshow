"""`ScenarioManifest` - what a scenario pack claims, and what it is allowed to claim.

This is the config a scenario ships with, and it exists so that the honesty of a
pack is a **data structure that can be read**, not a sentence in a slide deck. Two
fields do that work:

- **`business_line`** - the sentence the booth shows. For `retail_sanity` this is
  **"People present at the checkout counter"**. An earlier revision of this docstring
  recorded a reduced-scope sentence about items entering the checkout zone; that claim
  was **withdrawn**, because measured over all 3387 frames of `retail_checkout` **no
  product-transfer event is detectable in it at all**. The withdrawn sentence is
  deliberately not quoted here: `tests/test_retail_measurements.py` fails on any
  shipped source containing it, so a correction that quoted it would keep failing its
  own check. See `app/engine/retail_config.py` for the measurements and
  `docs/qualification/M0-content-gate.md` §3-§4 for the content gate. A pack that
  overstates its own claim here would put a false statement on a screen at a trade
  show, which is the one failure this whole module exists to prevent.
- **`scope_note`** - free text carried alongside it, so a reduced-scope label can
  ride with the data and be exported with it rather than living in a presenter
  script that can drift. Optional: a pack with no reduction has nothing to say.

**Immutability is enforced past `frozen=True`.** `event_rules` is a `dict`, and a
frozen dataclass does not freeze the dict it points at - `manifest.event_rules[
"min_dwell_seconds"] = 0.0` would otherwise succeed and silently change what the
event machine requires. It is wrapped in a `MappingProxyType`, which is also why
this record is unhashable. That is the same trade M0's `ModelOutputSpec` makes and
for the same reason.

Everything here is pure: no clock, no I/O, no OpenVINO, and the class-name
allowlist is checked for internal consistency rather than against
`config/models.json` - reading the pin is `app.inference.pinned`'s job, and this
module must stay importable on a machine with no model provisioned.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from app.engine.zones import Zone

__all__ = ["ScenarioManifest"]


def _require_threshold(name: str, value: Any) -> float:
    """A confidence or IoU threshold, in `[0, 1]`.

    The pinned pack uses `conf=0.25, iou=0.45` - the same pair M0's fixture was
    evaluated at, fixed before the model was run. A manifest that disagrees is not
    a stylistic difference; it means the acceptance set was built at one operating
    point and the demo runs at another.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number; got {value!r}")
    number = float(value)
    if number != number or not 0.0 <= number <= 1.0:
        raise ValueError(
            f"{name} must be within [0, 1]; got {value!r}. Outside that range a "
            f"threshold either suppresses everything or nothing, and both look like a "
            f"working detector."
        )
    return number


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string; got {value!r}. An empty title is what "
            f"a placeholder looks like, and it is indistinguishable on screen from a "
            f"title nobody wrote."
        )
    return value


@dataclass(frozen=True)
class ScenarioManifest:
    """One scenario pack: what it shows, which classes it names, and how it judges.

    `zones` is a tuple of `Zone`, so an ROI carried here was validated by `Zone`
    itself. A bad ROI fails at **load**, with the zone's name in the message, rather
    than producing boxes in the wrong place at run time - which is the whole reason
    `Zone.__post_init__` refuses out-of-order and out-of-range geometry.

    `event_rules` is the machine's configuration: `min_dwell_seconds`,
    `max_transition_seconds`, the zone slots named by key (`"shelf"`, `"basket"`
    for the semantic test fixture; `"basket"`, `"checkout"` for the footage that
    actually exists), and `tracked_classes`. Its zone entries are **raw ROI tuples**,
    not `Zone` objects, and that is deliberate: the manifest is what a JSON config
    loads into, so a loader builds `Zone`s from it and the validation happens in one
    place. `zones_for_event_rules()` does that building and is the only supported way
    to get them.
    """

    id: str
    version: int
    title: str
    business_line: str
    class_allowlist: tuple[str, ...]
    zones: tuple[Zone, ...]
    event_rules: Mapping[str, Any]
    source_ids: tuple[str, ...]
    conf_threshold: float
    iou_threshold: float
    scope_note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        if isinstance(self.version, bool) or not isinstance(self.version, int) \
                or self.version < 1:
            raise ValueError(
                f"version must be a positive int; got {self.version!r}. Version 0 is "
                f"not a draft, it is an unlabelled record."
            )
        object.__setattr__(self, "title", _require_text("title", self.title))
        object.__setattr__(self, "business_line",
                           _require_text("business_line", self.business_line))
        object.__setattr__(self, "conf_threshold",
                           _require_threshold("conf_threshold", self.conf_threshold))
        object.__setattr__(self, "iou_threshold",
                           _require_threshold("iou_threshold", self.iou_threshold))

        allowlist = tuple(self.class_allowlist)
        if not allowlist:
            raise ValueError(
                "class_allowlist must not be empty. An empty allowlist means the pack "
                "names no class, which would be a detector with nothing to say."
            )
        for name in allowlist:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"class_allowlist entries must be non-empty strings; got {name!r}"
                )
        object.__setattr__(self, "class_allowlist", allowlist)

        sources = tuple(self.source_ids)
        if not sources:
            raise ValueError("source_ids must not be empty; a pack with no clip has nothing to play")
        for source in sources:
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"source_ids entries must be non-empty strings; got {source!r}")
        object.__setattr__(self, "source_ids", sources)

        zones = tuple(self.zones)
        for zone in zones:
            if not isinstance(zone, Zone):
                raise ValueError(
                    f"zones must hold Zone objects, each of which validated its own ROI; "
                    f"got {zone!r} ({type(zone).__name__})"
                )
        names = [z.name for z in zones]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate zone name(s) {duplicates}. Zone names key per-track event "
                f"state, so two zones sharing a name would share and corrupt it."
            )
        object.__setattr__(self, "zones", zones)

        if not isinstance(self.event_rules, Mapping):
            raise ValueError(
                f"event_rules must be a mapping; got {type(self.event_rules).__name__}"
            )
        rules = dict(self.event_rules)
        # Read-only: `frozen=True` does not reach inside a dict, so a caller could
        # otherwise change the dwell requirement after the machine read it.
        object.__setattr__(self, "event_rules", MappingProxyType(rules))
        object.__setattr__(self, "scope_note",
                           self.scope_note if isinstance(self.scope_note, str) else "")

    def zones_for_event_rules(self) -> dict[str, Zone]:
        """Build `Zone`s for the ROI entries in `event_rules`, validating each.

        The one supported path from this record to usable geometry. A malformed ROI
        in a config file raises here, **naming the slot**, at load - which is the
        difference between a loud failure at startup and boxes drawn in the wrong
        place for a whole demo session.
        """
        built: dict[str, Zone] = {zone.name: zone for zone in self.zones}
        for key, value in self.event_rules.items():
            if not isinstance(value, (tuple, list)) or len(value) != 4:
                continue
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                # Overwrites a `zones` entry of the same name, deliberately: the event
                # rule is the more specific statement of where that slot is.
                built[key] = Zone(name=key, kind="area", roi=tuple(value))
        return built

    def to_json(self) -> dict[str, Any]:
        """A `json.dumps`-acceptable payload. Zones become ROI tuples again."""
        payload: dict[str, Any] = {}
        for name in ("id", "version", "title", "business_line", "class_allowlist",
                     "zones", "event_rules", "source_ids", "conf_threshold",
                     "iou_threshold", "scope_note"):
            value = getattr(self, name)
            if name == "zones":
                payload[name] = [
                    {"name": z.name, "kind": z.kind, "roi": list(z.roi)} for z in value
                ]
            elif isinstance(value, tuple):
                payload[name] = list(value)
            elif name == "event_rules":
                payload[name] = {
                    k: (list(v) if isinstance(v, tuple) else v) for k, v in value.items()
                }
            else:
                payload[name] = value
        return payload