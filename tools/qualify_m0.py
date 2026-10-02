"""The M0 qualification runner: compile the pinned model, run it, and report what happened.

Run from the repo root:

    .venv\\Scripts\\python.exe tools\\qualify_m0.py --requests 100

Writes `logs/qualification/m0.json` and prints the table. Exit code is 0 only when every
row succeeded **and** ran on the device it asked for.

**This is the module that produces M0's evidence.** Everything before it is a component.
The design is therefore shaped by one question: *what would make this table lie?*

Five ways, and what closes each:

1. **A device that failed, rendered as a device that was fast.** `TargetResult` carries
   `error` with every measurement left `None`, and `render_table` prints `unavailable`
   in every numeric column of an errored row *even if that row somehow carries numbers*.
   The brief's own test pins this: a failed target must render nothing that looks like a
   result, not even the request count it never reached. There is no path in this module
   that turns a failure into a number.
2. **A device that silently fell back to another.** `EXECUTION_DEVICES`, re-read from the
   request's own compiled model **after** the warmup inference, is the only proof of
   placement (AGENTS.md rule 5). It is compared to the requested device through
   `is_device`, never by substring, because the readings are inconsistent by device -
   measured on this machine: CPU `['CPU']`, GPU `['GPU.0']`, NPU the **bare string**
   `'NPU'`. A row whose reading does not name what it asked for is flagged `[fallback]`
   in the error column, next to the error text, so it cannot be missed in a glance.
3. **Compile time counted as inference time.** Separate columns, separately timed.
   `compile_seconds` wraps `compile_model` and nothing else; `warmup_seconds` wraps the
   single untimed-count warmup; the latency percentiles and `requests_per_second` come
   only from the timed loop that follows. `time.perf_counter` throughout - a monotonic
   clock, so a clock adjustment mid-run cannot produce a negative latency.
4. **A percentile that flatters the device.** `percentile` is a plain nearest-rank order
   statistic: no interpolation, no smoothing, no trimming, so p95 of 100 samples is the
   95th of the 100 samples that were taken. Nothing is rounded to a nicer number; the
   values in the JSON are the ones `perf_counter` produced.
5. **Memory attributed to the wrong thing.** `rss_bytes` is `psutil.Process()
   .memory_info().rss` - the **host process's** resident set, which contains the
   OpenVINO runtime, the model and the blob, and is not a measurement of accelerator
   memory. The NPU's own buffer is not visible through any property this milestone
   reads. The column is labelled `rss` for that reason and must not be quoted as "NPU
   memory" in the M0 report.

**`describe_outputs` is never called here.** It needs the `ov.Model` and cannot walk an
`ov.CompiledModel`; the spec is derived once from the IR by
`tools/build_sanity_fixture.py.describe_pinned_model`, which checks the IR's SHA-256
against `config/models.json` first. A qualification run that decoded against a guessed
layout would be measuring the wrong thing quickly.

**Startup is measured, not estimated**, two ways and both reported: from this module's
import to the first inference completing, and the process uptime at that moment. The
first is what a booth pays before it can draw a box; the second includes the interpreter
and the imports, which is what an operator watching a cold start actually sees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openvino as ov  # noqa: E402

from app.inference.devices import (  # noqa: E402
    DEVICE_IDS,
    TargetSpec,
    is_device,
    normalise_execution_devices,
    require_static_shape,
)
from app.inference.preprocess import preprocess_bgr  # noqa: E402
from tools.build_sanity_fixture import (  # noqa: E402
    FIXTURE_JSON,
    class_names,
    describe_pinned_model,
    pinned_input_shape,
    pinned_model_entry,
)

#: Stamped at import, before OpenVINO has done any work. This is the zero for the
#: "process start to first inference" figure the brief asks for: everything after this
#: line is the booth's real start-up cost, including the imports below.
_MODULE_IMPORTED = time.perf_counter()

OUTPUT_JSON = REPO_ROOT / "logs" / "qualification" / "m0.json"

#: The row label for the explicit CPU-only baseline. It is a *re-measurement of the
#: same pinned model pinned to the CPU*, run after the three device rows, and it exists
#: so the report can state that the fallback route works **with evidence** rather than by
#: assertion. It is a fourth run, not a copy of the CPU row: a fallback that reuses the
#: primary row's numbers has proved nothing about the fallback path.
FALLBACK_CAPTION = "CPU-only fallback baseline (same model, same shape, explicitly CPU)"

#: The row label for the primary per-device rows.
DEVICE_CAPTION = "explicit per-device compile, one warmup, then the timed loop"

UNAVAILABLE = "unavailable"

#: The placement marker. A token rather than a word so the tests can assert on its
#: presence exactly, without a legend that mentions it making every row look like a
#: fallback. The legend below does mention it, and says what it means.
FALLBACK_MARKER = "[fallback]"

#: The table's legend. It explains how to read the placement column and, in particular,
#: that a row can succeed and still be flagged - which is the state a latency table
#: cannot catch. It deliberately does **not** quote `FALLBACK_MARKER`: the marker is a
#: per-row flag, and a legend containing the token would make every row in the report
#: carry it, which is the opposite of a signal.
LEGEND = (
    "legend: placement is the EXECUTION_DEVICES reading, re-read from the request's own "
    "compiled model after the first inference - never the device that was asked for. "
    "It is compared through is_device, so CPU, GPU and NPU all match despite the "
    "inconsistent shapes the plugins return ('CPU', 'GPU.0', and the bare string 'NPU'). "
    "A row whose reading does not name its requested device is flagged in the note "
    "column: it measured a real latency, on the wrong hardware, and that is worth more "
    "than a clean-looking row. 'rss' is the HOST process resident set, not accelerator "
    "memory; the NPU's own buffer is not visible through any property this milestone "
    "reads. compile and warmup are timed separately from inference and are never added "
    "to it. An errored row prints 'unavailable' in every numeric column, whatever that "
    "row carries, and prints its error in full."
)


def percentile(samples: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of `samples`. No interpolation, no smoothing.

    Nearest-rank, not a linear-interpolated quantile, for a reason that is about honesty
    rather than taste: with 100 samples, p95 is the 95th sample that was actually
    observed. An interpolated p95 would be a number no request ever produced, and it
    would be a number chosen by a formula rather than measured.

        percentile(range(1, 101), 50) -> 50
        percentile(range(1, 101), 95) -> 95
        percentile([1, 2, 3], 50)     -> 2      (the middle sample, not 2.0 of 1.5)

    Empty input and a `q` outside `[0, 100]` raise rather than returning a number: a
    percentile of nothing is not zero.
    """
    if not isinstance(q, (int, float)) or isinstance(q, bool):
        raise TypeError(f"q must be a number; got {q!r}")
    if not 0.0 <= float(q) <= 100.0:
        raise ValueError(f"q must be within [0, 100]; got {q!r}")
    values = sorted(float(v) for v in samples)
    if not values:
        raise ValueError(
            "cannot take a percentile of no samples. An empty measurement set has no "
            "median; returning 0.0 would put a fabricated number in a latency column."
        )
    rank = max(1, math.ceil(float(q) / 100.0 * len(values)))
    return values[min(rank, len(values)) - 1]


@dataclass(frozen=True)
class TargetResult:
    """One device's qualification result, measured or explicitly not measured.

    Every field is either a real measurement or `None`. There is no `0.0` standing in
    for "did not happen": a device that could not compile carries `error` and `None`
    everywhere else, and `render_table` prints `unavailable` for each of those columns.

    `execution_devices` is the **normalised** `EXECUTION_DEVICES` reading - a tuple, via
    `normalise_execution_devices`, because the plugins return inconsistent shapes and a
    bare `'NPU'` string run through `tuple()` becomes `('N', 'P', 'U')`. Empty on a
    failure: a device that never ran did not run anywhere, and `()` says that, while
    `('CPU',)` on a failed NPU row would be a claim about placement that was never made.

    `rss_bytes` is the host process's resident set. See the module docstring for why it
    is not accelerator memory.
    """

    device_id: str
    full_name: str
    compile_seconds: float | None
    warmup_seconds: float | None
    requests: int
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    latency_ms_min: float | None
    latency_ms_max: float | None
    requests_per_second: float | None
    execution_devices: tuple[str, ...]
    rss_bytes: int | None
    error: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_devices",
                           tuple(str(d) for d in self.execution_devices))
        if not isinstance(self.requests, int) or isinstance(self.requests, bool):
            raise TypeError(f"requests must be an int; got {self.requests!r}")
        if self.requests < 0:
            raise ValueError(f"requests must not be negative; got {self.requests}")
        if self.error is not None and not str(self.error).strip():
            raise ValueError(
                f"{self.device_id}: error is set but empty. Unavailable is a stated "
                f"reason, never a blank - a row with an empty error renders as a "
                f"failure with no explanation, which is the one thing worse than a "
                f"zero."
            )
        for name in ("compile_seconds", "warmup_seconds", "latency_ms_p50",
                     "latency_ms_p95", "latency_ms_min", "latency_ms_max",
                     "requests_per_second"):
            value = getattr(self, name)
            if value is None:
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name} is {value!r}, which is not a measurement")
            # 0.0 is refused on purpose. A 640x640 YOLO11n inference on this hardware
            # takes single-digit milliseconds, so a zero here is not a fast device - it
            # is a clock that was not read. "Too fast to measure" belongs in `error`.
            if number <= 0.0:
                raise ValueError(
                    f"{self.device_id}: {name} is {number!r}. A zero here is not a fast "
                    f"measurement, it is an unmeasured one - and 0.00 in a latency "
                    f"column is exactly the number this project must never show. Use "
                    f"None, and say why in `error`."
                )
            object.__setattr__(self, name, number)
        if self.rss_bytes is not None:
            if not isinstance(self.rss_bytes, int) or isinstance(self.rss_bytes, bool):
                raise TypeError(f"rss_bytes must be an int; got {self.rss_bytes!r}")
            if self.rss_bytes <= 0:
                raise ValueError(f"rss_bytes must be positive; got {self.rss_bytes}")

    @property
    def placed_on_requested_device(self) -> bool:
        """Whether `EXECUTION_DEVICES` names the device this row asked for."""
        return is_device(self.execution_devices, self.device_id)

    @property
    def failed(self) -> bool:
        return self.error is not None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["execution_devices"] = list(self.execution_devices)
        payload["placed_on_requested_device"] = self.placed_on_requested_device
        payload["failed"] = self.failed
        return payload


def _fmt(value: Any, suffix: str, places: int = 2) -> str:
    """A measurement, or `unavailable`. Never a zero standing in for a missing one."""
    if value is None:
        return UNAVAILABLE
    return f"{float(value):.{places}f}{suffix}"


def _fmt_bytes(value: int | None) -> str:
    if value is None:
        return UNAVAILABLE
    mib = int(value) / (1024 * 1024)
    return f"{mib:.1f} MiB ({int(value)} B)"


#: Width of the full-name column, and the room kept for the ellipsis that says the
#: string was cut. The first version truncated at 34 characters with **no** marker, so
#: `Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)` printed as
#: `Intel(R) Arc(TM) 140V GPU (16GB) (` and the reader could not tell a cut string from
#: the whole name. That qualifier is the difference between an integrated and a discrete
#: GPU - precisely what a presenter is asked about at a booth - so the column is wide
#: enough to hold this machine's longest name and a cut is always marked.
FULL_NAME_WIDTH = 40
_ELLIPSIS = "..."


def _column(text: str, width: int = FULL_NAME_WIDTH) -> str:
    """`text` fitted to `width`, with an explicit ellipsis when it does not fit.

    A silent truncation is a small lie: the reader cannot tell that the string they are
    looking at is not the string the runtime reported. Three dots cost three columns and
    say "there is more, and it is in the JSON".
    """
    if len(text) <= width:
        return text
    return text[:max(0, width - len(_ELLIPSIS))] + _ELLIPSIS


def render_table(results: Sequence[TargetResult],
                 captions: Sequence[str] | None = None) -> str:
    """One row per device, and nothing on a failed row that looks like a result.

    Columns: device, full name, compile s, warmup s, requests, p50, p95, min, max,
    rps, `EXECUTION_DEVICES`, rss, note.

    **An errored row prints `unavailable` in every numeric column, whatever that row
    carries.** That is stronger than "a failed row has no numbers" and it is deliberate:
    a row is either a measurement or a failure, and a half-populated failure is the state
    where a reader is most likely to quote a number that was never measured. The error
    text is still printed in full - a failure that hides its reason is not honest, it is
    just quiet - and so is the `EXECUTION_DEVICES` reading, because that is how a
    fallback route is named.

    A row that succeeded but ran somewhere other than where it was asked to run is
    marked `FALLBACK_MARKER` next to its reading. That is a *success that misreports
    itself*, which is the failure mode a latency table cannot catch and the reason
    `EXECUTION_DEVICES` is the column that exists.
    """
    rows = list(results)
    notes: list[str] = []
    header = (f"{'device':<10} {'full name':<40} {'compile s':>10} {'warmup s':>9} "
              f"{'requests':>9} {'p50 ms':>9} {'p95 ms':>9} {'min ms':>9} {'max ms':>9} "
              f"{'rps':>8}  {'EXECUTION_DEVICES':<20} {'rss':<26} note")
    lines = [header, "-" * len(header)]
    for index, result in enumerate(rows):
        if result.failed:
            # Every measurement is suppressed, including any the caller populated.
            compile_s = warmup_s = p50 = p95 = lo = hi = rps = UNAVAILABLE
            requests = UNAVAILABLE
            rss = UNAVAILABLE
        else:
            compile_s = _fmt(result.compile_seconds, " s")
            warmup_s = _fmt(result.warmup_seconds, " s")
            p50 = _fmt(result.latency_ms_p50, " ms")
            p95 = _fmt(result.latency_ms_p95, " ms")
            lo = _fmt(result.latency_ms_min, " ms")
            hi = _fmt(result.latency_ms_max, " ms")
            rps = _fmt(result.requests_per_second, "")
            requests = str(result.requests)
            rss = _fmt_bytes(result.rss_bytes)
        placement = ",".join(result.execution_devices) if result.execution_devices \
            else UNAVAILABLE
        note_parts: list[str] = []
        if result.failed:
            note_parts.append(str(result.error))
        elif not result.placed_on_requested_device:
            note_parts.append(
                f"{FALLBACK_MARKER} asked for {result.device_id}, "
                f"EXECUTION_DEVICES says {list(result.execution_devices)}")
        if captions is not None and index < len(captions) and captions[index]:
            notes.append(f"  {result.device_id:<10} {captions[index]}")
        lines.append(
            f"{result.device_id:<10} {_column(result.full_name or UNAVAILABLE):<40} "
            f"{compile_s:>10} "
            f"{warmup_s:>9} {requests:>9} {p50:>9} {p95:>9} {lo:>9} {hi:>9} "
            f"{rps:>8}  {placement:<20} {rss:<26} {'; '.join(note_parts)}")
    out = "\n".join(lines)
    if notes:
        out += "\n\nrow notes:\n" + "\n".join(notes)
    return out + "\n\n" + LEGEND


def _failed(device_id: str, error: str, full_name: str = "") -> TargetResult:
    """A failure, as a record. Every measurement `None`, placement empty, requests 0.

    The one function in this module that is allowed to build a result without measuring
    anything, and it builds nothing else.
    """
    return TargetResult(
        device_id=device_id, full_name=full_name, compile_seconds=None,
        warmup_seconds=None, requests=0, latency_ms_p50=None, latency_ms_p95=None,
        latency_ms_min=None, latency_ms_max=None, requests_per_second=None,
        execution_devices=(), rss_bytes=None, error=error)


def qualify_target(core: ov.Core, spec: TargetSpec, image: np.ndarray,
                   num_requests: int) -> TargetResult:
    """Compile `spec` for its device and time `num_requests` real inferences.

    In order, and the order is the point:

    1. Read `FULL_DEVICE_NAME`. A device that cannot even be named has not been
       qualified, and the failure says which property was unreadable.
    2. `read_model`, then time `compile_model` **on its own**. Reading the IR is not
       compiling it and the two are reported apart from each other and from inference.
    3. Create the request and run **one** warmup inference, timed separately. The warmup
       exists so the timed loop measures steady state, and it is the first opportunity to
       read `EXECUTION_DEVICES` - before it, the value would be a claim about a placement
       that had not happened.
    4. Read `EXECUTION_DEVICES` from the request's own compiled model, after that warmup,
       and normalise it.
    5. Time `num_requests` inferences with `time.perf_counter` and take nearest-rank
       p50/p95 plus min, max and the achieved requests per second over the timed span
       only. The warmup is outside the span.

    **Any exception is caught and returned as `error`**, with every measurement `None` and
    placement empty. Nothing partial is reported: a compile that succeeded and a warmup
    that threw does not get to show its compile time next to no latency, because a reader
    scanning a table reads the number that is there. The exception's own type and text
    are carried verbatim, so the row says what the runtime said.

    The letterboxed blob is built once, outside every timer: preprocessing is not
    inference, and timing it would inflate every latency in the report.
    """
    device_id = spec.device_id
    full_name = ""
    try:
        if not isinstance(num_requests, int) or isinstance(num_requests, bool) \
                or num_requests <= 0:
            raise ValueError(f"num_requests must be a positive int; got {num_requests!r}")
        require_static_shape(tuple(spec.input_shape))
        full_name = str(core.get_property(device_id, "FULL_DEVICE_NAME"))
        model = core.read_model(str(spec.model_path))

        _, _, dst_h, dst_w = pinned_input_shape()
        if (dst_h, dst_w) != (spec.input_shape[2], spec.input_shape[3]):
            raise ValueError(
                f"TargetSpec says the input is {tuple(spec.input_shape)} but "
                f"config/models.json pins {pinned_input_shape()}. Compiling one shape "
                f"and feeding another is a silent mismatch, not a rounding."
            )
        blob, _letterbox = preprocess_bgr(image, dst_w, dst_h)

        compile_started = time.perf_counter()
        compiled = core.compile_model(model, device_id)
        compile_seconds = time.perf_counter() - compile_started

        request = compiled.create_infer_request()
        warmup_started = time.perf_counter()
        request.infer({0: blob})
        warmup_seconds = time.perf_counter() - warmup_started

        # After the first inference, and from the request's own compiled model.
        placement = normalise_execution_devices(
            request.get_compiled_model().get_property("EXECUTION_DEVICES"))

        latencies_ms: list[float] = []
        loop_started = time.perf_counter()
        for _ in range(num_requests):
            one = time.perf_counter()
            request.infer({0: blob})
            latencies_ms.append((time.perf_counter() - one) * 1000.0)
        loop_seconds = time.perf_counter() - loop_started

        rss = int(psutil.Process().memory_info().rss)
        return TargetResult(
            device_id=device_id, full_name=full_name, compile_seconds=compile_seconds,
            warmup_seconds=warmup_seconds, requests=num_requests,
            latency_ms_p50=percentile(latencies_ms, 50.0),
            latency_ms_p95=percentile(latencies_ms, 95.0),
            latency_ms_min=min(latencies_ms), latency_ms_max=max(latencies_ms),
            requests_per_second=num_requests / loop_seconds,
            execution_devices=placement, rss_bytes=rss, error=None)
    except Exception as exc:  # noqa: BLE001 - a device that cannot run is a result
        detail = f"{type(exc).__name__}: {exc}".strip()
        return _failed(device_id, f"qualification failed - {detail}", full_name)


def _row_seconds(result: TargetResult, elapsed: float) -> float | None:
    """Seconds from entering a row to its first inference - or `None` if it never ran.

    A device absent from `available_devices` skips `qualify_target` entirely, and the
    elapsed time for that skip is a few microseconds of Python. Recorded unconditionally
    it became a row reading `to_first_inference_s: 2.7e-06`, which is a *time to first
    inference* for a row where no inference happened. `None` says "this row never got
    there", which is the only true thing, and it keeps the field consistent with the
    other unmeasured fields on a failed row.
    """
    if result.failed or result.warmup_seconds is None:
        return None
    return float(elapsed)


def _process_uptime_seconds() -> float:
    """Seconds since this process started, from the OS rather than from a stopwatch."""
    created = psutil.Process().create_time()
    return max(0.0, time.time() - float(created))


def _fixture_image() -> np.ndarray:
    """The pinned sanity-fixture frame, so the qualification runs on a real frame."""
    if not FIXTURE_JSON.exists():
        raise FileNotFoundError(
            f"{FIXTURE_JSON} does not exist, and the qualification refuses to "
            f"synthesise a frame. Task 6's fixture is the pinned input; a random tensor "
            f"would measure the same arithmetic, but it would not be measuring the "
            f"product."
        )
    payload = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
    image_path = FIXTURE_JSON.parent / payload["image"]
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"cv2.imread returned None for {image_path}")
    return image


def run_all(num_requests: int) -> dict[str, Any]:
    """CPU, GPU, NPU, then the explicit CPU-only fallback baseline. Four real runs.

    The fallback baseline is a **separate** `qualify_target` call on its own `TargetSpec`,
    after the three device rows, not a copy of the CPU row. Reusing the CPU row's numbers
    would let the report say "the fallback works" on the strength of a measurement that
    never took the fallback path.
    """
    core = ov.Core()
    entry = pinned_model_entry()
    ir = REPO_ROOT / entry["ir_path"]
    shape = pinned_input_shape()
    image = _fixture_image()

    first_inference_at: float | None = None
    results: list[TargetResult] = []
    captions: list[str] = []
    rows: list[dict[str, Any]] = []

    for device_id in DEVICE_IDS:
        available = tuple(core.available_devices)
        spec = TargetSpec(device_id=device_id, model_path=ir, input_shape=shape)
        started = time.perf_counter()
        if device_id not in available:
            result = _failed(
                device_id,
                f"not in ov.Core().available_devices(), which reported "
                f"{list(available)}; the model was never compiled for this device")
        else:
            result = qualify_target(core, spec, image, num_requests)
        if not result.failed and first_inference_at is None:
            first_inference_at = time.perf_counter()
        results.append(result)
        captions.append(DEVICE_CAPTION)
        rows.append({"row": device_id, "caption": DEVICE_CAPTION,
                     "to_first_inference_s": _row_seconds(
                         result, time.perf_counter() - started),
                     **result.as_dict()})

    fallback_spec = TargetSpec(device_id="CPU", model_path=ir, input_shape=shape)
    fallback_started = time.perf_counter()
    fallback = qualify_target(core, fallback_spec, image, num_requests)
    results.append(fallback)
    captions.append(FALLBACK_CAPTION)
    rows.append({"row": "CPU-fallback-baseline", "caption": FALLBACK_CAPTION,
                 "to_first_inference_s": _row_seconds(
                     fallback, time.perf_counter() - fallback_started),
                 **fallback.as_dict()})

    return {
        "generated_by": "tools/qualify_m0.py",
        "openvino_runtime": str(ov.__version__),
        "opencv_version": str(cv2.__version__),
        "model_id": entry["id"],
        "model_ir_path": entry["ir_path"],
        "model_ir_sha256": hashlib.sha256(ir.read_bytes()).hexdigest(),
        "model_ir_sha256_matches_config_models_json":
            hashlib.sha256(ir.read_bytes()).hexdigest() == entry["sha256"],
        "input_shape": list(shape),
        "num_requests_per_row": num_requests,
        "available_devices": list(core.available_devices),
        "fixture_image": json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))["image"],
        "fixture_frame_index": json.loads(
            FIXTURE_JSON.read_text(encoding="utf-8"))["frame_index"],
        "startup": {
            "definition": "seconds from this module's import to the first inference of "
                          "the first row completing, and the process uptime at the same "
                          "moment. The first excludes the interpreter; the second "
                          "includes it, which is what a cold booth start looks like.",
            "module_import_to_first_inference_s":
                None if first_inference_at is None
                else first_inference_at - _MODULE_IMPORTED,
            "process_uptime_at_first_inference_s":
                None if first_inference_at is None else _process_uptime_seconds(),
            "note": "compile and warmup happen inside this window on the first row; they "
                    "are not added to it, and they are also reported per row.",
        },
        "notes": [
            "rss_bytes is the HOST process resident set (psutil), not accelerator "
            "memory. The NPU's own buffer is not visible through any property this "
            "milestone reads.",
            "latency_ms_* are nearest-rank percentiles of the timed loop only; the "
            "single warmup inference is outside the timed span and is reported as "
            "warmup_seconds.",
            "requests_per_second is num_requests divided by the timed span only.",
            "EXECUTION_DEVICES is read from the request's own compiled model after the "
            "warmup inference, and normalised: the plugins return ['CPU'], ['GPU.0'] "
            "and the bare string 'NPU' respectively.",
        ],
        "rows": rows,
    }


def _row_to_result(row: dict[str, Any]) -> TargetResult:
    """Rebuild a `TargetResult` from a JSON row, for rendering the report read-back.

    The report is rendered from the same records that were written, not from a second
    measurement: a table assembled from a re-run would be a different table, and the
    JSON is what a reader checks the table against. Only the declared fields are taken,
    so the derived keys in the row (`caption`, `placed_on_requested_device`, `failed`)
    cannot be fed back in and shadow a computed property.
    """
    known = {f.name for f in fields(TargetResult)}
    kwargs = {name: row[name] for name in known}
    kwargs["execution_devices"] = tuple(row["execution_devices"])
    return TargetResult(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Qualify the pinned detector on every device and report what ran.")
    parser.add_argument("--requests", type=int, default=100,
                        help="timed inferences per row (default 100)")
    parser.add_argument("--json-out", type=Path, default=OUTPUT_JSON)
    args = parser.parse_args(argv)

    print(f"openvino {ov.__version__}")
    print(f"qualifying with {args.requests} timed requests per row\n")
    report = run_all(args.requests)
    results = [_row_to_result(row) for row in report["rows"]]
    captions = [r["caption"] for r in report["rows"]]
    print(render_table(results, captions))

    startup = report["startup"]
    print(f"startup, module import -> first inference : "
          f"{startup['module_import_to_first_inference_s']} s")
    print(f"startup, process uptime at that moment    : "
          f"{startup['process_uptime_at_first_inference_s']} s")
    print(f"model IR sha256 matches config/models.json: "
          f"{report['model_ir_sha256_matches_config_models_json']}")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json_out}")

    ok = all(not r.failed and r.placed_on_requested_device for r in results)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'} - "
          f"{sum(1 for r in results if not r.failed and r.placed_on_requested_device)}"
          f" of {len(results)} rows succeeded on the device they requested")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
