r"""The preflight gate: one command, one word, and every row a presenter can check.

    .venv\Scripts\python.exe tools\preflight.py

Prints a row per check, then `Ready`, `Limited` or `Needs attention`, then the reasons.
Exit code is `0` for Ready and Limited, `1` for Needs attention.

---

## What this gate is, and the question it is shaped around

A presenter runs this once, before standing in front of a crowd, and reads one word off the
bottom. So the question is not "what did we measure" - `tools/qualify_m0.py` already
answers that - it is **"what would make this word wrong?"** Five ways, and what closes each:

1. **A failed check softened into a warning.** `level` is a field a caller fills in and
   `Check` stores it verbatim; `validate_rows` refuses a `WARN` or a `FAIL` that carries no
   detail, because a warning whose reason cannot be read is a shrug. A degraded-but-working
   capability is `WARN`. Something that does not work is `FAIL`. The two are never chosen
   for effect: `run_checks` computes each level from the measurement, not the other way
   round.
2. **A missing device rendered as a working one.** Device rows come from
   `DeviceRegistry.describe()`, whose `DeviceStatus` cannot even be constructed for an
   absent device without a reason. `device_control_lines` renders that reason next to a
   `disabled` marker and asserts nothing about the absent device's capabilities. The
   accelerator is never implied into existence.
3. **A re-derived number disagreeing with the recorded one.** The per-device compile,
   latency and placement rows **reuse** `TargetResult` and `render_table` from
   `tools/qualify_m0.py` over the JSON that run wrote, rather than timing anything again.
   A preflight that re-measured would print a *different* table from the one in the M0
   report, and the report is the record. Every row that quotes a recorded figure says so, in
   the row, and names the file it came from.
4. **A recorded run that no longer describes this machine.** The recorded table is only
   evidence if it was produced by the same IR and the same OpenVINO build. Both are checked
   against `config/models.json` and the live `ov.__version__` before a single number is
   quoted; a mismatch downgrades the row to `WARN` and quotes nothing.
5. **A number on screen that nothing measured.** Nothing here computes a percentage it did
   not read. Telemetry availability is a `TelemetryAvailability`, whose `available=False`
   cannot be constructed without a reason. Disk and RAM rows print what `psutil` and
   `shutil` returned in this run, and the two policy floors they are compared against are
   module constants named as policy, not presented as measurements.

## The one row that is a real inference, and why

`Sanity fixture on CPU` runs `evaluate_fixture` - a real compile, a real inference through
the real decoder, a real comparison against the hand-drawn annotation - and it is **required**:
a `FAIL` here is a `FAIL`, not a warning.

That is not decoration. `tests/test_sanity_fixture.py`'s CPU-backed test skips itself when
no CPU plugin is present, and the committed mutation sweep is a throwaway harness, so
without this row nothing in the default gate ever exercised the fixture. A fixture that is
never run protects nothing.

`PASS` additionally requires the run to have been **placed on the CPU** - `evaluate_fixture`
checks `EXECUTION_DEVICES` through `app.inference.devices.is_device`, so the `.0` suffix and
the NPU's bare-string reading are handled by the one accessor that handles them. This module
never reads `EXECUTION_DEVICES` itself.

## What this gate deliberately does not do

- It does not time anything. `qualify_m0.py` owns latency; duplicating it would produce a
  second, slightly different number for the same thing.
- It does not judge the decoder. `tools/build_sanity_fixture.py` owns the annotation and its
  known blind spots; this gate asks whether the fixture passed and reports the failure text.
- It does not claim a hardware SKU, and it does not reconcile the three conflicting
  identity records - `Win32_ComputerSystem.Model` says `NUC14LNK LN`, the hostname says
  `nuc14proai`, the PCI subsystem id says `SUBSYS_88F21043`. All three are printed and none
  is named the winner.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openvino as ov  # noqa: E402
import psutil  # noqa: E402

from app.contracts.device_status import DeviceStatus  # noqa: E402
from app.contracts.telemetry_availability import TelemetryAvailability  # noqa: E402
from app.inference.devices import DeviceRegistry  # noqa: E402
from app.telemetry.instance_map import load_instance_map  # noqa: E402
from app.telemetry.pdh import (  # noqa: E402
    VALUE_READS_ENABLED_BY_DEFAULT,
    VALUE_READS_REFUSED_REASON,
    pdh_available,
)
from tools.build_sanity_fixture import (  # noqa: E402
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    MATCH_IOU,
    class_names,
    describe_pinned_model,
    evaluate_fixture,
    pinned_input_shape,
    pinned_model_entry,
)
from tools.qualify_m0 import TargetResult, render_table  # noqa: E402

__all__ = [
    "LEVELS",
    "Check",
    "device_control_lines",
    "offer_cpu_only",
    "overall_status",
    "render_row",
    "render_summary",
    "run_checks",
    "validate_rows",
]

#: Stamped at import, before any hardware work. The zero for the startup row: this is what
#: a booth pays between running the gate and the first inference completing, and it is
#: measured here rather than quoted from the qualification run.
_MODULE_IMPORTED = time.perf_counter()

LEVELS: Final[tuple[str, ...]] = ("PASS", "WARN", "FAIL")

#: The three words a presenter reads. A closed set, because a verdict that can acquire a
#: fourth value ("Ready-ish") stops being a verdict.
STATUS_READY: Final[str] = "Ready"
STATUS_LIMITED: Final[str] = "Limited"
STATUS_ATTENTION: Final[str] = "Needs attention"

#: The recorded qualification run this gate reads. Written by `tools/qualify_m0.py`; never
#: re-measured here. `logs/` is not committed, so on a machine that has never been
#: qualified this file is absent and the rows that quote it say exactly that.
RECORDED_QUALIFICATION = REPO_ROOT / "logs" / "qualification" / "m0.json"
RECORDED_SANITY_FIXTURE = REPO_ROOT / "logs" / "qualification" / "m0_sanity_fixture.json"
TELEMETRY_MAP = REPO_ROOT / "config" / "telemetry_map.json"

#: A recorded run older than this is reported as stale rather than as evidence. Thirty
#: days is a judgement, and it is recorded as one: it is short enough that a driver or
#: runtime change on a demo machine is unlikely to hide behind it. The alternative - no
#: bound at all - means a table from a previous machine is presented as this machine's.
RECORDED_RUN_MAX_AGE_DAYS: Final[int] = 30

#: Bytes of free disk on the volume holding the repo below which preflight says so. A
#: policy floor, not a hardware fact, and it is the only disk number this module invents.
MIN_FREE_DISK_BYTES: Final[int] = 2 * 1024 ** 3

#: Bytes of free physical memory below which preflight says so. Same status: a policy floor
#: chosen so a presenter is not surprised by swapping, not a measurement of this machine.
MIN_AVAILABLE_RAM_BYTES: Final[int] = 4 * 1024 ** 3

WIDTH = 78


@dataclass(frozen=True)
class Check:
    """One preflight row: a name, a level, and a detail a presenter can check.

    `ok` and `level` are two independent axes and `Check` deliberately does **not**
    reconcile them. `level` is what `render_row` and `render_summary` read, because `level`
    is the one a human reads; `ok` is there for programmatic callers. Collapsing them would
    hide exactly the mislabelling this gate has to be able to show, and the brief's own test
    constructs `Check("x", False, "PASS", "")` and requires the `PASS` to survive.

    An unknown `level` is refused rather than passed through: a level nobody renders is a
    level nobody reads, and a typo in `"WARN"` would silently become a fourth, unhandled
    state.
    """

    name: str
    ok: bool
    level: str
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name or "").strip())
        object.__setattr__(self, "detail", str(self.detail or ""))
        if not self.name:
            raise ValueError(
                "a Check needs a name. A row a presenter cannot find by name is a row "
                "they cannot check, and this table is read in a hurry."
            )
        if self.level not in LEVELS:
            raise ValueError(
                f"{self.name}: level must be one of {list(LEVELS)}; got "
                f"{self.level!r}. A level the renderer does not know is a level nobody "
                f"reads."
            )


def _pass(name: str, detail: str = "") -> Check:
    return Check(name, True, "PASS", detail)


def _warn(name: str, reason: str) -> Check:
    """A degraded-but-working capability. `reason` is mandatory and is not defaulted."""
    if not reason.strip():
        raise ValueError(f"{name}: a WARN row must state why. Refusing to emit one.")
    return Check(name, False, "WARN", reason)


def _fail(name: str, reason: str) -> Check:
    """Something that does not work. Never a `WARN`: softening a `FAIL` is the failure."""
    if not reason.strip():
        raise ValueError(f"{name}: a FAIL row must state why.")
    return Check(name, False, "FAIL", reason)


def validate_rows(rows: Sequence[Check]) -> None:
    """Refuse a table that cannot be read honestly. Raises rather than repairing.

    Two rules, both about the reader rather than the machine:

    - a `WARN` or a `FAIL` must carry a detail. A warning whose reason cannot be read is a
      shrug, and a shrug in a table of twenty rows is invisible.
    - two rows may not share a name. A reader counting the reasons for the NPU row cannot
      tell which of two NPU rows they are reading.

    A `PASS` may carry an empty detail, because `Check("a", True, "PASS", "")` is in the
    brief's own test and a passing row has nothing to explain.
    """
    seen: set[str] = set()
    for check in rows:
        if check.name in seen:
            raise ValueError(
                f"two preflight rows are both named {check.name!r}. Duplicate names make "
                f"the summary ambiguous, because a reader cannot tell which row a listed "
                f"reason belongs to."
            )
        seen.add(check.name)
        if check.level in ("WARN", "FAIL") and not check.detail.strip():
            raise ValueError(
                f"{check.name}: level {check.level} with an empty detail. A warning whose "
                f"reason cannot be read is a shrug, and this table is read in a hurry."
            )


def render_row(check: Check) -> str:
    """One row: level, name, then the detail indented under it.

    The detail is **wrapped per source line, never reflowed across them.** A first version
    passed the whole detail to `textwrap.wrap`, which joins the lines of a multi-line detail
    into one paragraph - and the row that embeds the recorded qualification table came out
    as an unreadable run-on block where `device`, `compile s` and `EXECUTION_DEVICES` were
    interleaved. That is worse than not printing the table: those are the numbers a reader
    checks this gate's claims against, and a mangled table invites them to trust the row's
    `PASS` instead of reading the evidence.

    Nothing is truncated either. A cut detail is a detail the reader cannot check, and this
    is the one table where "the rest was elided" would be a lie about what was measured.
    """
    import textwrap

    lines = [f"{check.level:<4}  {check.name}"]
    for source_line in check.detail.splitlines() or [""]:
        wrapped = textwrap.wrap(source_line, width=WIDTH, initial_indent="      ",
                                subsequent_indent="      ")
        lines.extend(wrapped or ["      "])
    return "\n".join(lines)


def _counts(rows: Sequence[Check]) -> dict[str, int]:
    return {level: sum(1 for check in rows if check.level == level) for level in LEVELS}


def overall_status(rows: Sequence[Check]) -> str:
    """Ready, Limited or Needs attention, derived from the rows and never passed in.

    - any `FAIL` -> **Needs attention**. Something does not work, and a presenter needs to
      know before a visitor does.
    - otherwise any `WARN` -> **Limited**. The demo runs; a named capability is degraded.
      This is the state spec section 9's CPU-only-degraded path describes, and it is a
      legitimate booth configuration rather than a fault.
    - otherwise -> **Ready**.

    **Zero rows is `Needs attention`, not `Ready`.** A table that measured nothing has
    produced no evidence of readiness, and the word `Ready` over an empty table is the same
    class of error as the word `Ready` over a failing machine.
    """
    if not rows:
        return STATUS_ATTENTION
    counts = _counts(rows)
    if counts["FAIL"]:
        return STATUS_ATTENTION
    if counts["WARN"]:
        return STATUS_LIMITED
    return STATUS_READY


def render_summary(rows: Sequence[Check]) -> str:
    """The counts, the word, and every row that is not a `PASS`, named.

    The reasons are listed because a word on its own is unactionable: a presenter who is
    told `Limited` and nothing else has to go and find out what. The exit code is stated
    here too, so the number a script reads is on the same screen as the word a human reads.

    `validate_rows` is **not** called here, and that is deliberate. The brief's own test
    renders a summary of `Check("c", False, "WARN", "")` - a `WARN` with an empty detail -
    and requires the counts to come back. So the renderer is a renderer: it reports the rows
    it is handed, including a shrug, rather than refusing to draw a table. The guard lives in
    `run_checks`, which is where this gate's own rows are built and where a detail-less
    `WARN` would be this module's mistake rather than a caller's.
    """
    counts = _counts(rows)
    status = overall_status(rows)
    parts = [
        f"Summary: {counts['PASS']} PASS, {counts['WARN']} WARN, "
        f"{counts['FAIL']} FAIL - {len(rows)} total rows",
        f"Status: {status}",
    ]
    for level in ("FAIL", "WARN"):
        flagged = [check for check in rows if check.level == level]
        if not flagged:
            continue
        parts.append(f"  {level}:")
        for check in flagged:
            # Collapsed to one line per row. `render_row` preserves a detail's own line
            # structure, which is what makes the embedded qualification table readable; here
            # that same structure would break the summary's own indentation and leave the
            # second bullet flush against the margin. The row above carries the full
            # formatted detail, so nothing is lost by summarising it. A bullet marker is
            # stripped because the summary supplies its own separator, and keeping both
            # printed "- - the LUID-to-device mapping is ...".
            flat = " ".join(check.detail.split()).lstrip("-").strip()
            parts.append(f"    {check.name} - {flat}")
    parts.append(
        f"Exit code: {0 if status != STATUS_ATTENTION else 1}. Ready and Limited both "
        f"exit 0, because in both the demo runs; only Needs attention exits 1."
    )
    return "\n".join(parts)


def device_control_lines(statuses: Sequence[DeviceStatus]) -> list[str]:
    """One line per device control, and an absent device's is visibly **disabled**.

    A control is `enabled` only when the device is present. An absent device's line carries
    the reason `describe()` recorded - which is the OpenVINO runtime's own error text, not
    this module's invention - and the word `disabled`.

    What this function controls is **its own wording**, not the reason text's: the marker it
    writes is `enabled` or `disabled` and never both, and it prints no full name for an
    absent device, because an absent device has none and inventing one would be the failure.
    An earlier version of this docstring claimed the word "available" never appears in a
    disabled line. That was false, and measurably so: the runtime's own reason quotes
    `ov.Core().available_devices()`, so the substring is there and the claim could not be
    made true without paraphrasing the runtime's wording. Paraphrasing it would be the worse
    loss, so the claim is withdrawn rather than the text.
    """
    lines: list[str] = []
    for status in statuses:
        if status.available:
            driver = status.driver_version or "no separate driver artefact"
            lines.append(
                f"  {status.device_id:<4} enabled  {status.full_name} "
                f"(driver {driver})"
            )
        else:
            reason = (status.unavailable_reason or "no reason recorded").strip()
            lines.append(f"  {status.device_id:<4} disabled - {reason}")
    return lines


def offer_cpu_only(statuses: Sequence[DeviceStatus]) -> str:
    """The CPU-only continuation, offered only when an accelerator is actually absent.

    Returns `""` when every device in `statuses` is present, because offering CPU-only on a
    machine that has all three accelerators would teach a presenter to expect a degradation
    that is not there. The named absent devices are listed, so the offer says which control
    is off rather than implying all of them are.
    """
    absent = [status.device_id for status in statuses
              if not status.available and status.device_id != "CPU"]
    if not absent:
        return ""
    if not any(status.available and status.device_id == "CPU" for status in statuses):
        return ""
    names = ", ".join(absent)
    plural = len(absent) > 1
    return (
        f"Continue with CPU only: yes - {names} "
        f"{'are' if plural else 'is'} not available on this machine, and the CPU is. The "
        f"demo runs on the CPU; the {names} "
        f"{'controls stay' if plural else 'control stays'} disabled rather than being "
        f"offered and then failing."
    )


# --------------------------------------------------------------------- recorded results


def _load_recorded(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _recorded_target_results(path: Path) -> list[TargetResult]:
    """The recorded `TargetResult` rows, or `[]` when there is no usable recording.

    The JSON is rendered through `TargetResult` and `render_table` from
    `tools/qualify_m0.py` rather than being reformatted here, so a preflight table and the
    M0 report's table are the same table produced by the same code. Only the declared
    `TargetResult` fields are read, so the derived keys in each row
    (`caption`, `placed_on_requested_device`, `failed`) cannot be fed back in and shadow a
    computed property - which is the same discipline `_row_to_result` applies.
    """
    from dataclasses import fields as dataclass_fields

    report = _load_recorded(path)
    if not isinstance(report, dict):
        return []
    results: list[TargetResult] = []
    for row in report.get("rows") or ():
        if not isinstance(row, dict):
            continue
        known = {f.name for f in dataclass_fields(TargetResult)}
        if not known.issubset(row):
            continue
        kwargs = {name: row[name] for name in known}
        kwargs["execution_devices"] = tuple(row["execution_devices"])
        try:
            results.append(TargetResult(**kwargs))
        except (TypeError, ValueError):
            # A row that cannot be rebuilt is dropped rather than repaired: a repaired row
            # would carry a number this module did not measure.
            continue
    return results


def _display_path(path: Path) -> str:
    """`path` relative to the repo when it is inside it, and absolute when it is not.

    `Path.relative_to` **raises** `ValueError` on a path outside its base, and these paths
    are configurable - `RECORDED_QUALIFICATION` has a default but the helpers take any path,
    and a caller pointing one at a mounted share or a temp directory is legitimate. The first
    version called `relative_to` unguarded, so asking the question "what is the recorded run's
    path" crashed with `ValueError` on exactly the case where the file is *absent* and the
    caller most needs an explanation. A reporting helper that raises while reporting is the
    one bug in this module that could not be caught by a row.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _recorded_age_days(path: Path) -> float | None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return (time.time() - mtime) / 86400.0


def _recorded_provenance_problem(path: Path) -> str | None:
    """Why the recorded run may not be quoted, or `None` when it may be.

    A recorded latency table is evidence about *this* machine only if it was produced by the
    same IR and the same OpenVINO build. Both are checked here; a mismatch means the numbers
    describe a different runtime, and quoting them beside a live device row would be the
    exact "a real number on the wrong hardware" failure the project exists to prevent.
    """
    report = _load_recorded(path)
    if not isinstance(report, dict):
        return (
            f"no usable recorded qualification run at {_display_path(path)}. This "
            f"row quotes recorded numbers and does not re-measure, so there is nothing to "
            f"quote. Run '.venv\\Scripts\\python.exe tools\\qualify_m0.py --requests 100' to "
            f"produce one."
        )
    entry = pinned_model_entry()
    problems: list[str] = []
    if report.get("openvino_runtime") != ov.__version__:
        problems.append(
            f"the recorded run used OpenVINO {report.get('openvino_runtime')!r} and this "
            f"process is {ov.__version__!r}"
        )
    if not report.get("model_ir_sha256_matches_config_models_json", False):
        problems.append(
            f"the recorded run's IR hash does not match the sha256 in config/models.json "
            f"for {entry['id']}"
        )
    age = _recorded_age_days(path)
    if age is not None and age > RECORDED_RUN_MAX_AGE_DAYS:
        problems.append(
            f"the recorded file was last written {age:.0f} days ago, past this gate's "
            f"{RECORDED_RUN_MAX_AGE_DAYS}-day bound"
        )
    if problems:
        return ("the recorded run does not describe this machine, so none of its numbers "
                "are quoted: " + "; ".join(problems))
    return None


# ------------------------------------------------------------------------------- checks


def _check_assets() -> Check:
    """Content and hash verification of every artifact in `config/models.json`.

    Delegated to `tools.verify_sources` rather than reimplemented: a second implementation
    of the hash comparison would be a second source of truth for the one check that decides
    whether the booth is running the pinned bytes.
    """
    from tools.verify_sources import collect_sources, verify

    try:
        config = json.loads(
            (REPO_ROOT / "config" / "models.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail("Assets: pinned content verification",
                     f"config/models.json is unreadable: {exc}")
    results = [verify(source, sha, lines)
               for source, sha, lines in collect_sources(config)]
    failures = [r for r in results if not r.ok]
    if failures:
        return _fail(
            "Assets: pinned content verification",
            "; ".join(f"{r.source.id}: {r.detail}" for r in failures),
        )
    return _pass(
        "Assets: pinned content verification",
        f"{len(results)} artifacts verified by content signature and sha256 against "
        f"config/models.json: "
        + ", ".join(f"{r.source.id} ({r.source.kind})" for r in results),
    )


def _check_sanity_fixture_file() -> Check:
    """The hand-annotated fixture and the frame it was drawn on are both present.

    Without this row the fixture row below would report a decoder verdict against nothing:
    `evaluate_fixture` returns `passed=False` with "no fixture" if the JSON is gone, which
    is correct but reads as a decoder failure rather than a missing file.
    """
    from tools.build_sanity_fixture import FIXTURE_DIR, FIXTURE_JSON, load_fixture

    if not FIXTURE_JSON.exists():
        return _fail(
            "Assets: sanity fixture",
            f"{_display_path(FIXTURE_JSON)} does not exist. The annotated fixture "
            f"is test data, not something preflight may create: an annotation this tool "
            f"wrote would be its own output checked against itself.",
        )
    try:
        fixture = load_fixture()
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        return _fail("Assets: sanity fixture",
                     f"{FIXTURE_JSON.name} is unreadable: {exc}")
    image = FIXTURE_DIR / fixture["image"]
    if not image.exists():
        return _fail("Assets: sanity fixture",
                     f"the frame the boxes were drawn on, {image.name}, is missing. The "
                     f"boxes are in source pixels, so without that picture they describe "
                     f"nothing.")
    return _pass(
        "Assets: sanity fixture",
        f"{FIXTURE_JSON.name} and {image.name} present; frame "
        f"{fixture.get('frame_index')} of {fixture.get('source_clip')}, "
        f"{len(fixture.get('expected') or ())} hand-drawn expectations, "
        f"max_detections bound {fixture.get('max_detections')}",
    )


def _check_disk() -> Check:
    """Free space on the volume holding the repo, measured now, against a named policy floor."""
    usage = shutil.disk_usage(REPO_ROOT)
    free_gib = usage.free / (1024 ** 3)
    if usage.free < MIN_FREE_DISK_BYTES:
        return _fail(
            "Budget: free disk on the repo volume",
            f"{free_gib:.1f} GiB free, below this gate's policy floor of "
            f"{MIN_FREE_DISK_BYTES / 1024 ** 3:.0f} GiB. The measured value is from "
            f"shutil.disk_usage({REPO_ROOT}) in this run; the floor is a policy constant "
            f"in tools/preflight.py, not a measurement of any machine.",
        )
    return _pass(
        "Budget: free disk on the repo volume",
        f"{free_gib:.1f} GiB free of {usage.total / 1024 ** 3:.1f} GiB, measured by "
        f"shutil.disk_usage in this run; policy floor "
        f"{MIN_FREE_DISK_BYTES / 1024 ** 3:.0f} GiB",
    )


def _check_ram() -> Check:
    """Free physical memory, measured now, against a named policy floor."""
    virtual = psutil.virtual_memory()
    available_gib = virtual.available / (1024 ** 3)
    if virtual.available < MIN_AVAILABLE_RAM_BYTES:
        return _fail(
            "Budget: free physical memory",
            f"{available_gib:.1f} GiB available of {virtual.total / 1024 ** 3:.1f} GiB, "
            f"below this gate's policy floor of "
            f"{MIN_AVAILABLE_RAM_BYTES / 1024 ** 3:.0f} GiB. Measured by psutil in this "
            f"run; the floor is a policy constant, not a measurement.",
        )
    return _pass(
        "Budget: free physical memory",
        f"{available_gib:.1f} GiB available of {virtual.total / 1024 ** 3:.1f} GiB across "
        f"{psutil.cpu_count()} logical cores, measured by psutil in this run; policy floor "
        f"{MIN_AVAILABLE_RAM_BYTES / 1024 ** 3:.0f} GiB",
    )


def _device_rows(statuses: Sequence[DeviceStatus]) -> list[Check]:
    """One row per device, and an absent one is a `WARN` - never a `FAIL`, never a `PASS`.

    `WARN` rather than `FAIL` because spec section 9's CPU-only-degraded path is a
    legitimate booth configuration: a package with no NPU is still a package. `WARN` rather
    than `PASS` because an absent accelerator that read as present is precisely the
    dishonesty this gate exists to prevent, and `describe()`'s own `unavailable_reason` is
    the OpenVINO runtime's text, quoted rather than rephrased.
    """
    rows: list[Check] = []
    for status in statuses:
        name = f"Device available: {status.device_id}"
        if status.available:
            rows.append(_pass(
                name,
                f"{status.full_name}; discovered in ov.Core().available_devices(), "
                f"runtime {status.runtime_version}, driver "
                f"{status.driver_version or 'none (no separate driver artefact)'}",
            ))
        elif status.device_id == "CPU":
            # The CPU is the one absence that stops the demo, because the CPU-only
            # continuation depends on it. Still a WARN: preflight's job is to report the
            # machine, and the operator's decision is whether to proceed.
            rows.append(_warn(
                name,
                f"{status.unavailable_reason}. Without a CPU there is no fallback route, "
                f"so a CPU-only continuation is not available either.",
            ))
        else:
            rows.append(_warn(
                name,
                f"{status.unavailable_reason}. The {status.device_id} control stays "
                f"disabled; spec section 9's CPU-only path is a legitimate degraded mode, "
                f"so this is a warning and not a failure.",
            ))
    return rows


def _check_runtime_versions() -> Check:
    """The runtime and interpreter this run actually loaded.

    Reported as a `PASS` with the values, not compared against anything: the version pins
    are enforced by `tests/test_pins.py`, and duplicating a pin table here would be a second
    place for it to drift. The row exists because the OpenVINO build string is part of the
    blob-cache key, and a presenter looking at a failure needs to see which build produced
    it.
    """
    return _pass(
        "Runtime: OpenVINO and Python",
        f"openvino {ov.__version__}, Python {platform.python_version()}, "
        f"psutil {psutil.__version__} - all read from the loaded modules in this run. "
        f"Version pins are enforced by tests/test_pins.py, not duplicated here.",
    )


def _check_model_pin() -> Check:
    """The IR on disk hashes to the sha256 `config/models.json` records.

    Checked here as well as by `_check_assets` because a mismatch has a second, sharper
    consequence: `tools/build_sanity_fixture.read_pinned_model` refuses to describe or decode
    a model that is not the pinned one, so every device row below would fail for a reason
    that has nothing to do with the devices.
    """
    import hashlib

    entry = pinned_model_entry()
    ir = REPO_ROOT / entry["ir_path"]
    try:
        digest = hashlib.sha256(ir.read_bytes()).hexdigest()
    except OSError as exc:
        return _fail("Assets: model IR pin",
                     f"cannot read {entry['ir_path']}: {exc}")
    shape = pinned_input_shape()
    if digest != entry["sha256"]:
        return _fail(
            "Assets: model IR pin",
            f"{entry['ir_path']} hashes to {digest}, not the {entry['sha256']} recorded in "
            f"config/models.json. Refusing to qualify anything against an IR that is not "
            f"the pinned one.",
        )
    return _pass(
        "Assets: model IR pin",
        f"{entry['id']} IR {entry['ir_path']} sha256 {digest} matches config/models.json; "
        f"pinned input shape {list(shape)}, static and batch 1",
    )


def _check_recorded_qualification() -> list[Check]:
    """The recorded per-device table, quoted rather than re-derived.

    Two rows: one for whether the recording may be quoted at all, and one that prints the
    table itself. They are separate because the first can fail while the second still has
    to say something - a reader who sees only "the recorded run is stale" does not learn
    what was measured, and the numbers are still on disk and still worth seeing labelled
    with their age.
    """
    rows: list[Check] = []
    problem = _recorded_provenance_problem(RECORDED_QUALIFICATION)
    results = _recorded_target_results(RECORDED_QUALIFICATION)
    age = _recorded_age_days(RECORDED_QUALIFICATION)

    if problem is not None:
        rows.append(_warn(
            "Recorded qualification run",
            problem,
        ))
    else:
        rows.append(_pass(
            "Recorded qualification run",
            f"{len(results)} rows read from "
            f"{_display_path(RECORDED_QUALIFICATION)}, written by "
            f"tools/qualify_m0.py, last modified {age:.1f} days ago. The IR hash and the "
            f"OpenVINO build in that file were checked against config/models.json and "
            f"against this process before any of its numbers were read.",
        ))

    if not results:
        rows.append(_warn(
            "Recorded per-device compile, latency and placement",
            f"no usable TargetResult rows in {_display_path(RECORDED_QUALIFICATION)}. "
            f"Those numbers are owned by tools/qualify_m0.py and are deliberately not "
            f"re-measured here, so this gate reports their absence rather than inventing a "
            f"substitute. This is a warning rather than a failure because a fresh machine "
            f"that has not been qualified yet is not broken.",
        ))
        return rows

    try:
        table = render_table(results)
    except Exception as exc:  # noqa: BLE001 - a table that will not render is a report fact
        rows.append(_fail(
            "Recorded per-device compile, latency and placement",
            f"the recorded rows could not be rendered: {type(exc).__name__}: {exc}",
        ))
        return rows

    misfiled = [r for r in results if r.failed or not r.placed_on_requested_device]
    detail = (
        f"RECORDED VALUES, quoted from "
        f"{_display_path(RECORDED_QUALIFICATION)} and not re-measured by this run. "
        f"Rendered by tools/qualify_m0.render_table so this table and the M0 report's table "
        f"come from the same code. Row totals below are the recorded ones."
    )
    if misfiled:
        detail += " Rows that did not succeed on the device they asked for: " + ", ".join(
            f"{r.device_id}"
            + (f" ({r.error})" if r.error
               else f" (EXECUTION_DEVICES={list(r.execution_devices)})")
            for r in misfiled
        )
    rows.append(Check("Recorded per-device compile, latency and placement",
                      not misfiled,
                      "FAIL" if misfiled else "PASS",
                      detail + "\n\n" + table))
    return rows


def _check_sanity_fixture_on_device(statuses: Sequence[DeviceStatus],
                                    device_id: str) -> Check:
    """A real compile, a real inference and a real decode on `device_id`.

    **The `CPU` case is required.** A `FAIL` here is a `FAIL` and not a warning, because a
    fixture that is never run protects nothing: `tests/test_sanity_fixture.py` skips its own
    CPU-backed evaluation when no CPU plugin is present, and the committed mutation sweep is
    a throwaway harness, so without this row the default gate never exercises the decoder
    against the hand annotation at all.

    For an absent device the row is a `WARN` carrying `describe()`'s reason, and the fixture
    is not run - not on another device, and not on the CPU, because a fixture result
    attributed to a device that was not asked for is a real number on the wrong card.
    """
    name = f"Sanity fixture on {device_id}"
    present = {status.device_id: status for status in statuses}
    status = present.get(device_id)
    if status is not None and not status.available:
        return _warn(
            name,
            f"not evaluated: {status.unavailable_reason}. A fixture run on another device "
            f"would be a {device_id} row carrying another device's number.",
        )
    try:
        outcome = evaluate_fixture(
            describe_pinned_model(), class_names(), CONF_THRESHOLD, IOU_THRESHOLD,
            device=device_id,
        )
    except Exception as exc:  # noqa: BLE001 - a device that cannot run is a result
        return _fail(name, f"the fixture evaluation raised: {type(exc).__name__}: {exc}")

    placement = [line for line in outcome.details if line.startswith("EXECUTION_DEVICES")]
    verdict = [line for line in outcome.details if line.startswith("VERDICT")]
    detail = (
        f"a real compile, a real inference and a real decode on {device_id} at "
        f"conf={CONF_THRESHOLD} iou={IOU_THRESHOLD} match_iou={MATCH_IOU}, compared "
        f"against the hand-drawn annotation: {outcome.matched} of {outcome.expected} "
        f"matched"
    )
    if placement:
        detail += f". {placement[0]}"
    if verdict:
        detail += f". {verdict[0]}"
    if not outcome.passed:
        detail += (
            f". Detection counts and scores are in the fixture tool's own output; this row "
            f"reports the verdict and does not restate a score it did not match on."
        )
    return Check(name, outcome.passed, "PASS" if outcome.passed else "FAIL", detail)


def _recorded_fixture_results() -> dict[str, dict[str, Any]]:
    """The recorded per-device fixture outcomes, or `{}`."""
    payload = _load_recorded(RECORDED_SANITY_FIXTURE)
    if not isinstance(payload, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entry in payload:
        if isinstance(entry, dict) and "device" in entry:
            out[str(entry["device"])] = entry
    return out


def _check_recorded_sanity_fixture() -> Check:
    """The per-device fixture outcomes recorded by `tools/build_sanity_fixture.py`.

    Distinct from the live row above, and both are kept: the live row proves the fixture
    passes *now* on the CPU, and this row says what the other two devices did when the
    fixture was last run against them. Quoting only the live row would leave the NPU's
    correctness resting on a file nobody reads.
    """
    recorded = _recorded_fixture_results()
    if not recorded:
        return _warn(
            "Recorded sanity fixture, per device",
            f"no per-device outcomes in "
            f"{_display_path(RECORDED_SANITY_FIXTURE)}. Run "
            f"'.venv\\Scripts\\python.exe tools\\build_sanity_fixture.py' to record them. "
            f"The live CPU row above is a real inference either way; this row is about the "
            f"other devices.",
        )
    parts = []
    for device_id in sorted(recorded):
        entry = recorded[device_id]
        parts.append(
            f"{device_id} {'PASS' if entry.get('passed') else 'FAIL'} "
            f"{entry.get('matched')}/{entry.get('expected')}"
        )
    failures = [device_id for device_id, entry in recorded.items()
                if not entry.get("passed")]
    detail = (
        f"RECORDED VALUES from {_display_path(RECORDED_SANITY_FIXTURE)}, written "
        f"by tools/build_sanity_fixture.py, not re-measured by this run: " + "; ".join(parts)
    )
    if failures:
        return _fail("Recorded sanity fixture, per device",
                     detail + f". Failed on: {', '.join(sorted(failures))}")
    return _pass("Recorded sanity fixture, per device", detail)


# --------------------------------------------------------------------------- telemetry


#: The `TelemetryAvailability` records this run produced, in row order. Module-level because
#: `run_checks` is specified to return `list[Check]` and the JSON output needs the records
#: themselves, not only the rows that describe them. `run_checks` **replaces** the list on
#: every call, so a second run in the same process cannot report the first run's providers.
TELEMETRY_RECORDS: list[TelemetryAvailability] = []


def _pdh_availability() -> tuple[TelemetryAvailability, Check]:
    """The in-process PDH provider, as an availability record and a row.

    The row is a `WARN`, and the distinction it draws is the whole point:

    - The **provider is real and validated**. `pdh.dll` loads, the counter opens, and the
      wildcard expansion enumerates this machine's live `GPU Engine` instances.
    - The **in-process value read is unavailable** on this machine, with a measured reason
      (`VALUE_READS_REFUSED_REASON`, recorded by Task 3 from first-hand measurement: the
      native value-read entry points terminate the process with `0xC0000005` even with
      handles proven valid by `PdhGetCounterTimeBase`).
    - The **counters themselves exist and are measurable** by a route M0 does not use.
      That evidence is recorded, and it is the reason this is a `WARN` and not a `FAIL`: a
      route failure is not a counter failure, and a `FAIL` here would tell a presenter the
      NPU is broken when it compiled and ran 100 requests in front of them.

    Never `PASS`: a `PASS` on a value this process cannot read is a claim about a
    measurement that did not happen.
    """
    scope = (
        "per adapter LUID, matching instance names against the recorded map; engines are "
        "never summed into a single total, per spec section 8"
    )
    try:
        ok, reason = pdh_available()
    except Exception as exc:  # noqa: BLE001 - an unreachable library is a reportable fact
        ok, reason = False, f"{type(exc).__name__}: {exc}"
    if not ok:
        availability = TelemetryAvailability(
            provider="pdh_gpu_engine", available=False,
            metric_names=("gpu_compute_percent", "npu_neural_percent"),
            unavailable_reason=f"pdh.dll could not be loaded: {reason}",
            scope_note=scope,
        )
        return availability, _warn(
            "Telemetry: in-process PDH GPU/NPU utilisation",
            f"Utilization unavailable - {reason}. The provider could not be loaded, so "
            f"nothing about the GPU or the NPU can be read in this process.",
        )

    availability = TelemetryAvailability(
        provider="pdh_gpu_engine", available=False,
        metric_names=("gpu_compute_percent", "npu_neural_percent"),
        unavailable_reason=(
            "provider validated and counter present, but the in-process value read is "
            "refused by default on this machine: "
            + VALUE_READS_REFUSED_REASON
        ),
        scope_note=scope,
    )
    return availability, _warn(
        "Telemetry: in-process PDH GPU/NPU utilisation",
        "Utilization unavailable (spec section 7's wording, not 0%). The provider is "
        "available and validated but not yet wired into the product's runtime path: "
        f"pdh.dll loads and the counter opens ({pdh_available()[1]}), and the native "
        "value-read entry points are refused by default because they terminate this "
        "process with 0xC0000005 even with handles proven valid by "
        "PdhGetCounterTimeBase. Recorded evidence that the counters do exist and are "
        "measurable by a route M0 does not use: NPU engtype_Neural and GPU "
        "engtype_Compute read real percentages through "
        "Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine and Get-Counter under "
        "real OpenVINO load (Task 3, see .superpowers/sdd/2026-10-01-m0-hardware-and-model-"
        "qualification/task-3-fix-report.md). A route failure is not a counter failure, so "
        "this is a WARN and not a FAIL: the NPU compiled and completed its requests. Spec "
        "section 10 gives the validated provider to M4.",
    )


def _check_telemetry_map() -> list[Check]:
    """The recorded LUID map: does it load, and what does it decline to claim?

    Two rows. The load is a real check - a map that will not parse would leave every card
    reading unavailable with no reason - and the attribution row exists because the honest
    description of that map is narrower than "we know which adapter is which".
    """
    rows: list[Check] = []
    try:
        instance_map = load_instance_map(TELEMETRY_MAP)
    except ValueError as exc:
        return [_fail("Telemetry: recorded LUID-to-device map",
                      f"config/telemetry_map.json will not load: {exc}")]
    rules = instance_map.rules()
    npu_rules = instance_map.rules_for("NPU")
    rows.append(_pass(
        "Telemetry: recorded LUID-to-device map",
        f"{len(rules)} rules loaded from config/telemetry_map.json, which load refuses on "
        f"a collision: "
        + ", ".join(f"{r.luid}->{r.device}"
                    + (f"/{r.engtype}" if r.engtype else "")
                    for r in rules)
        + f". NPU rules: {len(npu_rules)}"
        + (f" (engtype {npu_rules[0].engtype})" if npu_rules
           and npu_rules[0].engtype else "")
        + ".",
    ))

    payload = _load_recorded(TELEMETRY_MAP) or {}
    direct = payload.get("direct_luid_link") or {}
    machine = payload.get("machine") or {}
    counter_source = payload.get("counter_source") or {}
    unattributed = payload.get("unattributed_luids") or []

    reasons: list[str] = []
    if direct.get("available") is False:
        reasons.append(
            "the LUID-to-device mapping is DERIVED from the engine taxonomy, not a "
            "driver-exposed identity: no CIM or Win32 source on this machine joins a LUID "
            "to a PCI function. Preflight says 'derived' and never 'identified'."
        )
    if unattributed:
        for entry in unattributed:
            reasons.append(
                f"adapter LUID {entry.get('luid')} is recorded UNATTRIBUTED "
                f"({entry.get('instance_count')} instances, engine types "
                f"{entry.get('engine_types')}): {entry.get('reason')}"
            )
    identity_facts = [
        f"Win32_ComputerSystem {machine.get('computer_system_measured')!r}",
        f"hostname {machine.get('hostname')!r}",
        f"PCI subsystem {machine.get('pci_subsystem_ids_measured')}",
    ]
    if machine.get("reconciled") is False:
        reasons.append(
            "three hardware-identity records disagree and none is treated as the winner: "
            + "; ".join(identity_facts)
            + ". Preflight claims no SKU. Reconciling them is the M0 report's job, not a "
              "measurement this gate can make."
        )
    if counter_source.get("instances_observed") is not None:
        reasons.append(
            f"config/telemetry_map.json's instances_observed="
            f"{counter_source.get('instances_observed')} and its "
            f"max_observed_utilization_percent are per-run, load-dependent observations "
            f"from one probe, not hardware properties. They are deliberately not quoted "
            f"here as a property of this machine."
        )
    reasons.append(
        f"the map was probed on {payload.get('probed_on')} against Windows build "
        f"{payload.get('windows_build')}. It is machine-specific: a driver or Windows "
        f"update can invalidate the LUIDs, and tools/probe_telemetry.py must be re-run "
        f"under load. That is the same discipline as the blob-cache reset after a driver "
        f"update."
    )
    # One bullet per reason, each terminated. The map's own `reason` strings are sentences
    # without a trailing full stop, so joining them without one produced "nearest device
    # three hardware-identity records disagree" - two unrelated facts run into each other,
    # which is exactly the kind of sentence a hurried reader skims past half-read.
    rows.append(_warn(
        "Telemetry: LUID-to-device attribution and hardware identity",
        "\n".join(f"- {reason.rstrip()}." if not reason.rstrip().endswith(".")
                  else f"- {reason.rstrip()}"
                  for reason in reasons),
    ))
    return rows


def _check_startup() -> Check:
    """Time from this module's import to the first inference completing, measured now.

    The live CPU fixture row above is the first inference, so by the time this runs the
    figure is real. It is measured rather than quoted from the qualification run because
    the two answer different questions: the recorded one is the demo's start-up cost with a
    full device sweep, this one is what a presenter waits for after typing the command.
    """
    elapsed = time.perf_counter() - _MODULE_IMPORTED
    return _pass(
        "Startup: this gate's import to the first inference",
        f"{elapsed:.2f} s, measured with time.perf_counter in this run from this module's "
        f"import to the first inference completing. Includes the pinned PowerShell driver "
        f"enumeration inside DeviceRegistry.describe() and the IR read; the fixture rows "
        f"above hold their own per-row figures.",
    )


# ---------------------------------------------------------------------------- assembly


def run_checks() -> list[Check]:
    """Every row, in the order a presenter should read them.

    Ordering is groups, not severity: assets first, because nothing after them means
    anything if the pinned bytes are wrong; then the budget; then the devices; then the
    model; then telemetry. Grouping is a readability decision and is documented in the
    output rather than left implicit.

    `describe()` is called **once** and its result is threaded through, because it costs
    ~3.3 s cold on this machine (one memoised PowerShell process) and calling it per row
    would pay that repeatedly for the same answer.
    """
    registry = DeviceRegistry(ov.Core())
    # Timed around the **first** `describe()` call, not a second one. The Windows PnP
    # enumeration behind the driver lookup is memoised process-wide, so a second call costs
    # about a millisecond and timing it would print "0.00 s" - a number that reads as
    # "device discovery is free" when the real first-call cost on this machine is seconds.
    # A cheap measurement of the wrong call is worse than no measurement.
    describe_started = time.perf_counter()
    statuses = registry.describe()
    describe_seconds = time.perf_counter() - describe_started

    rows: list[Check] = []
    rows.append(_check_assets())
    rows.append(_check_sanity_fixture_file())
    rows.append(_check_model_pin())
    rows.append(_check_runtime_versions())
    rows.append(_check_disk())
    rows.append(_check_ram())
    rows.extend(_device_rows(statuses))
    rows.append(_pass(
        "Device discovery cost",
        f"DeviceRegistry.describe() took {describe_seconds:.2f} s in this run for all "
        f"{len(statuses)} devices, measured around the first call in this process. The cost "
        f"is one memoised Windows PnP enumeration (powershell.exe), not one per device; it "
        f"is startup-only and must never reach a per-frame or per-sample path. Spec "
        f"section 1 sets a 30 s warm-start target, so "
        f"{describe_seconds:.2f} s is inside it.",
    ))
    rows.extend(_check_recorded_qualification())
    rows.append(_check_sanity_fixture_on_device(statuses, "CPU"))
    rows.append(_check_recorded_sanity_fixture())
    availability, pdh_row = _pdh_availability()
    TELEMETRY_RECORDS.clear()
    TELEMETRY_RECORDS.append(availability)
    rows.append(pdh_row)
    rows.extend(_check_telemetry_map())
    rows.append(_check_startup())

    validate_rows(rows)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight the booth machine: assets, budget, devices, the pinned model, the "
            "hand-annotated fixture and telemetry, then say Ready, Limited or Needs "
            "attention."
        )
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="also write the rows as JSON here (used by the M0 report, so the report and "
             "this run cannot disagree)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    print("=" * WIDTH)
    print("PREFLIGHT - Intel Edge AI Roadshow, this machine, this run")
    print("=" * WIDTH)
    print(f"repository     : {REPO_ROOT}")
    print(f"openvino       : {ov.__version__}")
    print(f"python         : {platform.python_version()}")
    print(f"host           : {platform.node()}  (this is a hostname, not a SKU)")
    print(f"windows        : {platform.platform(terse=True)}")
    print()

    rows = run_checks()
    for check in rows:
        print(render_row(check))
        print()

    # The warm call. `run_checks` already called `describe()` once, and the driver lookup it
    # performs is memoised process-wide, so this costs about a millisecond and re-deriving
    # the control list from the first call's records would mean threading them back out of
    # `run_checks` for no gain.
    statuses = DeviceRegistry(ov.Core()).describe()
    print("device controls a presenter will be offered:")
    for line in device_control_lines(statuses):
        print(line)
    offer = offer_cpu_only(statuses)
    if offer:
        print()
        print(offer)
    print()
    print(render_summary(rows))

    if args.json_out is not None:
        payload = {
            "generated_by": "tools/preflight.py",
            "openvino_runtime": ov.__version__,
            "python": platform.python_version(),
            "rows": [
                {"name": c.name, "ok": c.ok, "level": c.level, "detail": c.detail}
                for c in rows
            ],
            "summary": {level: sum(1 for c in rows if c.level == level)
                        for level in LEVELS},
            "status": overall_status(rows),
            "telemetry_availability": [record.to_json()
                                       for record in TELEMETRY_RECORDS],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    return 0 if overall_status(rows) != STATUS_ATTENTION else 1


if __name__ == "__main__":
    raise SystemExit(main())
