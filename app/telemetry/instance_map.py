"""LUID-keyed matching of Windows `GPU Engine` PDH instances to devices.

**Why the LUID and nothing else.** Windows enumerates GPU-engine performance counters per
*process* and per *adapter*, and the instance name carries both:

    pid_1234_luid_0x00000000_0x00010E9E_phys_0_eng_5_engtype_Compute
    pid_1234_luid_0x00000000_0x000112A0_phys_0_eng_0_engtype_Neural

The `luid_` field is **two** words. The **second** word identifies the adapter and is the
only unambiguous way to tell the NPU from the GPU on this machine, where the two report
overlapping physical device ids. A mapping that keys on anything else - on physical device
id, on enumeration order, on "the only Neural engine I saw" - can put a real but wrong
number on the NPU's card, and a wrong number is worse than no number: a booth visitor who
sees "NPU 0%" concludes the NPU is broken.

**Never guess.** `parse_instance_name` raises `ValueError` on a name with no `luid_` field
rather than returning a plausible `0x00000000`. `InstanceMap.match` returns `None` for an
unknown LUID rather than falling back to the first rule. A missing instance is `None`,
never `0.0`; that distinction is the whole point of this layer.

**A collision is fatal, structurally.** The one-LUID-two-devices check lives in
`InstanceMap.__post_init__`, not only in `load_instance_map`, so it holds for a map built
by any route. `load_instance_map` therefore propagates it, and so does a hand-constructed
map. This is what makes a GPU/NPU swap impossible to commit silently.

**This module is pure.** It spawns nothing, opens no PDH query, and reads no registry. The
LUID -> device mapping is resolved once, at operator time, by `tools/probe_telemetry.py`
and recorded in `config/telemetry_map.json`; the 4 Hz sampling path only reads that file
and matches strings. See `app.telemetry.pdh` for the counter reads themselves.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from app.inference.devices import plugin_class

__all__ = [
    "ENGINE_TYPES",
    "EngineInstanceRule",
    "InstanceMap",
    "ParsedInstanceName",
    "device_for_instance",
    "load_instance_map",
    "normalise_luid",
    "parse_instance_name",
]

#: The engine types this demo cares about, as Windows spells them. Recorded for the
#: report and for the probe's summary; **not** a filter. An engine type outside this set is
#: still reported rather than dropped, because a name Windows actually produced is evidence
#: and hiding it would be the same mistake as inventing it.
ENGINE_TYPES: Final[tuple[str, ...]] = (
    "Compute", "Neural", "VideoDecode", "3D", "Copy", "VideoProcessing",
)

#: `luid_<word0>_<word1>`. Windows writes both words as `0x%08X`. The digits are accepted
#: 1-8 long and the `0x` prefix is optional, because a differently-formatted LUID is still
#: a real LUID and dropping the instance would hide evidence; it is normalised, not refused.
_LUID_FIELD: Final[re.Pattern[str]] = re.compile(
    r"luid_(?:0[xX])?([0-9A-Fa-f]{1,8})_(?:0[xX])?([0-9A-Fa-f]{1,8})"
)

#: `pid_<n>`, anchored at the start: `GPU Engine` instances lead with the owning process.
_PID_FIELD: Final[re.Pattern[str]] = re.compile(r"^\s*pid_(\d+)")

#: `engtype_<Name>`. Word characters only, so the documented numeric `3D` parses and a
#: trailing `_phys_0` cannot be swallowed into the engine type.
_ENGINE_TYPE_FIELD: Final[re.Pattern[str]] = re.compile(r"engtype_([A-Za-z0-9]+)")


def normalise_luid(value: object) -> str:
    """A LUID word as the single canonical spelling `0x%08X` upper-case.

    `0x00010e9e`, `0X00010E9E`, `10e9e` and `0x10E9E` all become `0x00010E9E`. Both the
    recorded map and the counter's instance name go through this, so a casing difference
    can never split one adapter into two rules.

    `ValueError` on anything that is not a hex LUID word - including `None` and `""` -
    because an unidentifiable adapter is exactly the case this module refuses to guess at.
    """
    text = str(value if value is not None else "").strip()
    if text[:2].lower() == "0x":
        text = text[2:]
    if not text or len(text) > 8 or re.fullmatch(r"[0-9A-Fa-f]{1,8}", text) is None:
        raise ValueError(
            f"not a LUID word: {value!r}. Expected 1-8 hex digits, optionally "
            f"'0x'-prefixed, as Windows writes in a 'GPU Engine' instance name."
        )
    return f"0x{int(text, 16):08X}"


@dataclass(frozen=True)
class ParsedInstanceName:
    """The three facts a `GPU Engine` instance name carries that matter here.

    `luid` is the **second** LUID word - the adapter - normalised by `normalise_luid`.
    `pid` and `engtype` are `None` when the name does not carry them, which is an absence
    of evidence and is never defaulted to `0` or to a guessed engine type.
    """

    pid: int | None
    luid: str
    engtype: str | None


def parse_instance_name(name: str) -> ParsedInstanceName:
    """Pull `pid`, the adapter `luid` and `engtype` out of a `GPU Engine` instance name.

    Raises `ValueError` when there is no `luid_` field. This is the structural guard
    against a plausible-looking zero: an unidentifiable adapter is not `0x00000000`, it is
    unknown, and the only honest answer to unknown here is to refuse rather than hand back
    a LUID that might collide with a real one.

    `pid` and `engtype` are `None` when absent, because they are not the tie-breaker and
    their absence does not make the adapter unidentifiable.
    """
    text = str(name or "")
    luid_field = _LUID_FIELD.search(text)
    if luid_field is None:
        raise ValueError(
            f"no 'luid_' field in PDH instance name {name!r}. A 'GPU Engine' instance "
            f"name is expected to be "
            f"'pid_<pid>_luid_0x<high>_0x<adapter>_phys_<n>_eng_<n>_engtype_<Type>'. "
            f"Refusing to guess an adapter: the LUID is the only tie-breaker between the "
            f"GPU and the NPU on this machine."
        )
    pid_field = _PID_FIELD.search(text)
    engine_field = _ENGINE_TYPE_FIELD.search(text)
    return ParsedInstanceName(
        pid=int(pid_field.group(1)) if pid_field is not None else None,
        luid=normalise_luid(luid_field.group(2)),
        engtype=engine_field.group(1) if engine_field is not None else None,
    )


@dataclass(frozen=True)
class EngineInstanceRule:
    """One recorded LUID -> (device, engine type) fact, as the probe observed it.

    `engtype` is `None` when the probe saw the adapter but no engine instance of a
    particular type for it. That is how a device with **no** `Neural` engine is recorded:
    present adapter, absent engine. The NPU is then `null`, never `0`.
    """

    luid: str
    device: str
    engtype: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "luid", normalise_luid(self.luid))
        device = plugin_class(self.device)
        if not device:
            raise ValueError(
                f"LUID {self.luid!r} has no device; a rule with no device would match "
                f"every instance of that adapter and attribute it to nothing"
            )
        object.__setattr__(self, "device", device)
        if self.engtype is not None and not str(self.engtype).strip():
            raise ValueError(
                f"LUID {self.luid!r} has an empty engtype; absent is None, never an "
                f"empty string that would read as a named engine"
            )


class InstanceMap:
    """The recorded adapter facts, and matching that can only ever be right or absent.

    **A collision is a hard error.** Two rules claiming the same LUID for different
    devices is a data-integrity failure, so it is refused at construction - by
    `load_instance_map`, by a probe-written file, and by a hand-built map alike. The check
    is here rather than in the loader so that no route into an `InstanceMap` can bypass it.
    """

    def __init__(self, rules: Iterable[EngineInstanceRule] = ()) -> None:
        by_luid: dict[str, EngineInstanceRule] = {}
        ordered: list[EngineInstanceRule] = []
        for rule in rules:
            if not isinstance(rule, EngineInstanceRule):
                raise ValueError(
                    f"InstanceMap takes EngineInstanceRule rows, got "
                    f"{type(rule).__name__}: {rule!r}"
                )
            existing = by_luid.get(rule.luid)
            if existing is not None and existing.device != rule.device:
                raise ValueError(
                    f"LUID {rule.luid} is mapped to two devices: {existing.device!r} and "
                    f"{rule.device!r}. The NPU and the GPU report overlapping physical "
                    f"device ids on this machine, so the LUID is the only tie-breaker - "
                    f"if it points at both, one of the two rows is wrong and the readings "
                    f"would be real but misattributed. Refusing to load."
                )
            if existing is None:
                by_luid[rule.luid] = rule
                ordered.append(rule)
            elif existing.engtype is None and rule.engtype is not None:
                # Same adapter, same device, and this row knows more. The recorded file's
                # `luids` section carries no `engtype` while its `engines` section does, so
                # without this merge the thinner row would win by arriving first and
                # `rules_for("NPU")` would report no engine type - losing the measured
                # evidence that the NPU has a `Neural` engine. Position in `rules()` is
                # preserved, so the load order still reads the same.
                ordered[ordered.index(existing)] = rule
                by_luid[rule.luid] = rule
        self._rules: tuple[EngineInstanceRule, ...] = tuple(ordered)
        self._by_luid: dict[str, EngineInstanceRule] = by_luid

    def rules(self) -> tuple[EngineInstanceRule, ...]:
        """Every recorded rule, in the order it was loaded."""
        return self._rules

    def rules_for(self, device: str) -> tuple[EngineInstanceRule, ...]:
        """The rules for one device, in load order.

        `device` goes through `plugin_class`, so `'NPU'`, `'npu'` and a reported
        `'NPU.0'` all select the same rules. An unknown device is an empty tuple, never a
        fallback to some other device's rules.
        """
        wanted = plugin_class(device)
        if not wanted:
            return ()
        return tuple(rule for rule in self._rules if rule.device == wanted)

    def match(self, instance_name: str, pid: int) -> EngineInstanceRule | None:
        """The rule for this whole instance - adapter **and** process - or `None`.

        The adapter is selected by the LUID alone; the supplied `pid` is a cross-check
        that the instance belongs to the process the caller is asking about. Both must
        agree, and either failure is `None`:

        - an unparsable name, or a LUID this map has never seen -> `None`. It is not
          matched to the nearest rule, because the nearest rule is a guess.
        - an instance whose own `pid_` is not `pid` -> `None`. The two sources disagree
          about which process this engine is, so the instance cannot be attributed at
          all. `None` renders as "unavailable"; returning the adapter anyway would put a
          number on a device on the strength of a disagreement.

        For a device-wide figure - the number a device card should show - call
        `device_for_instance` per instance, which resolves by LUID and ignores the process
        entirely. Do not pass this process's own pid for every instance: that returns
        `None` for every foreign engine, which is safe but useless.

        `None` never becomes `0.0` anywhere downstream. That is the difference between
        "the NPU is idle" and "we cannot measure the NPU", and a booth visitor reads them
        as opposite claims about the hardware.
        """
        try:
            parsed = parse_instance_name(instance_name)
        except ValueError:
            # The 4 Hz sampling path must not die on one odd instance name. The outcome is
            # still honest: an instance that cannot be read is not a reading of zero.
            return None
        if parsed.pid is None or int(pid) != parsed.pid:
            return None
        return self._by_luid.get(parsed.luid)

    def device_for_instance(self, instance_name: str) -> str | None:
        """The device a `GPU Engine` instance belongs to, by LUID, ignoring the process.

        This is the device-wide path: an adapter's utilisation is the sum over every
        process using it, and no single process's pid may select the device. `None` for an
        unparsable or unmapped instance - never `""`, never `"GPU"` as a default.
        """
        try:
            parsed = parse_instance_name(instance_name)
        except ValueError:
            return None
        rule = self._by_luid.get(parsed.luid)
        return rule.device if rule is not None else None

    def __len__(self) -> int:
        return len(self._rules)

    def __repr__(self) -> str:
        return f"InstanceMap({list(self._rules)!r})"


def _rule_from_record(record: Any, *, section: str) -> EngineInstanceRule:
    """One `{"luid", "device"[, "engtype"]}` record from the recorded map."""
    if not isinstance(record, dict):
        raise ValueError(
            f"telemetry_map.json {section}[] entries must be objects, got "
            f"{type(record).__name__}: {record!r}"
        )
    luid = record.get("luid")
    device = record.get("device")
    if not str(luid or "").strip():
        raise ValueError(
            f"telemetry_map.json {section}[] entry {record!r} has no 'luid'. A rule with "
            f"no adapter cannot be matched to anything."
        )
    if not str(device or "").strip():
        raise ValueError(
            f"telemetry_map.json {section}[] entry {record!r} has no 'device'. A rule "
            f"with no device would attribute an adapter to nothing."
        )
    return EngineInstanceRule(
        luid=luid, device=device, engtype=record.get("engtype")
    )


def load_instance_map(path: Path) -> InstanceMap:
    """Load the probe's recorded `config/telemetry_map.json`.

    Expected shape, as written by `tools/probe_telemetry.py`:

        {"probed_on": ..., "windows_build": ..., "counter_source": ...,
         "luids":   [{"luid": "0x00010E9E", "device": "GPU"}],
         "engines": [{"luid": "0x000112A0", "device": "NPU", "engtype": "Neural"}]}

    Both sections are loaded, and the one-LUID-two-devices check runs across **both**, so
    a map that is internally consistent only because it split the adapter across sections
    is still refused. `engines` entries are kept as their own rules so `rules_for` can
    report which engine types a device actually has; a LUID in `luids` with no
    corresponding `engines` entry is still a known adapter with no known engine, which is
    the correct record of a device we could not measure.

    Raises `ValueError` - never a silently empty map - on a missing file, malformed JSON, a
    non-object payload, a missing `luids` key, a malformed record, or a collision. A
    well-formed map with no rules is allowed and means "no adapter was identified": the
    cards read unavailable, which is honest.

    The file is only **read** here. `probed_on` and `windows_build` are metadata about the
    machine the probe ran on, and they are not validated against the running machine - a
    stale map is a fact the operator sees in the report, not something to be papered over
    by rejecting the file at load time.
    """
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"cannot read the recorded telemetry map at {target}: {exc}. Run "
            f"'.venv\\Scripts\\python.exe tools\\probe_telemetry.py' to produce it; the "
            f"runtime only reads what the probe recorded."
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{target} is not valid JSON: {exc}. Refusing to load a telemetry map that "
            f"cannot be read - an unreadable map must not look like an empty one."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"{target} must hold a JSON object, got {type(payload).__name__}. Refusing "
            f"to load: an unreadable map must not look like an empty one."
        )
    if "luids" not in payload:
        raise ValueError(
            f"{target} has no 'luids' key. Expected the shape written by "
            f"tools/probe_telemetry.py: "
            f"{{'probed_on', 'windows_build', 'counter_source', 'luids', 'engines'}}."
        )

    rules: list[EngineInstanceRule] = []
    for section in ("luids", "engines"):
        entries: Sequence[Any] = payload.get(section) or ()
        if not isinstance(entries, (list, tuple)):
            raise ValueError(
                f"{target}: {section!r} must be a list, got {type(entries).__name__}"
            )
        for record in entries:
            rules.append(_rule_from_record(record, section=section))
    # The collision check happens in InstanceMap.__post_init__ for every rule at once, so
    # the guard holds across both sections rather than within each.
    return InstanceMap(rules)
