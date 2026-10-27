"""Measure UI-thread input-acknowledgement lateness **and** inference backlog, together.

## What question this exists to settle

M1's input-acknowledgement gate measures **UI-thread tick lateness**: a `QTimer` on the Qt main
thread fires every 20 ms and records the gap between when the tick was due and when its slot ran.
It passes at a 200 ms p95 gate and a 2 s worst-tick gate.

M2 **asserted** that this gate is blind to inference backlog, on the grounds that a backlog does
not block the UI thread — it makes results arrive late. **Nothing demonstrated it.** Meanwhile the
same machine, running three CPU streams (spec section 5 row 4 — a state the booth will be in),
measures per-stream frame age **p50 11695 / 10672 / 11104 ms**, **p95 12315 / 11424 / 11902 ms**, a
backlog of **173-187 frames** per stream, each stream analysing **13.44-13.56 of the 15 fps**
offered, and **zero drops with all three streams `active`**. Every counter the old screen shows
reads healthy and the picture is eleven seconds old.

*(Every figure in that sentence is read from the committed
`docs/qualification/m2-headless/allcpu-steady-120s.json`, which is the record it is cited from.
The task brief quotes **10767 / 11085 / 11319 ms** and a backlog of **170-187** — a slightly
earlier run of the same measurement. The committed file is the authority and is what is quoted;
the difference is recorded rather than reconciled, because only one of the two can be opened and
checked.)*

So this is the experiment that decides whether "insensitive" is the right word or whether the
gate is **a green light on a system that is failing**. The claim is not sacred. It is M2's
assertion, and a result that contradicts it is as valuable as one that confirms it.

## The design rule that makes the result worth anything

**Both quantities, in one table, under the same conditions.** A tick figure measured with one
saturator count beside a frame-age figure measured with another is two unrelated numbers printed
on adjacent lines, and the comparison a reader will make anyway is not supported. Every row here
is one window: one application process, one device layout, one saturator count, one measured
window — and the tick percentiles and the frame-age percentiles come from *that window*, read on
the same thread at the same moment.

## It uses the application's own threading path, not a parallel harness

M1's first gate was rejected for measuring the *harness* rather than the application, and M2
recorded the same gap a second time. So this tool drives the app exactly the way the existing gate
does, through the **same three seams**:

- it writes a child program and runs it as `__main__`, because `multiprocessing` cannot spawn the
  inference worker from `python -c`;
- it replaces `app.main._start_feeders` with a wrapper that arms the tick timer **and then calls
  the application's own feeder start**, so the feeder, the telemetry timer, the reader threads and
  the result fold are `app/main.py`'s;
- it replaces `app.main._gui_application` to install the window timers, and then calls
  `app.main.main(argv)` for everything else.

A regression that moved inference onto the UI thread inside `app/` — inside `_on_result`, say, or
by calling `_start_feeders` with an inline `submit` — would move these numbers, because this probe
goes through the same functions the existing gate goes through. That is the property worth having,
and it is why this is not a new instrument.

## What it deliberately does NOT do

- **It does not tune a threshold.** The gate's 200 ms p95 and 2 s worst-tick are quoted from the
  existing gate and are *not* parameters here. This measures the gate; it does not fix it, and a
  threshold adjusted until the answer came out the way somebody wanted would not be a
  measurement.
- **It does not merge frame age with inference cost.** They merged once already and had to be
  unpicked. The NPU's measured p95 frame age is seconds while the inference itself costs about
  5.9 ms; the gap is queue wait. They are reported in their own columns with their own units and
  are never added, averaged, or presented as one number.
- **It does not reach into `app/` for the numbers.** Frame age and backlog come from
  `Supervisor.stream_frame_ages_ms` and `Supervisor.per_stream` — the application's own public
  accessors, the same ones `app/main.py::_publish_stream_metrics` reads — and the percentile is
  `app.main._percentile`, the app's own, so this tool cannot disagree with the screen about what
  p95 means. `tools/` importing `app/` is the allowed direction of the dependency.

## The honest limit of what a short hold can show

**Frame age is a consequence of queue depth, and queue depth takes time to build.** At 15 fps
offered and about 13.5 fps analysed (13.44-13.56 measured), a CPU stream gains 1.50 frames of
backlog per second, so the measured 173-187 frames is **115-125 s** of accumulation. A 6-second window cannot
show it. Every condition that is meant to reach the booth's real state therefore carries a
**long warm-up**, and `AckResult.warmup_s` is published next to the figures so a reader can see
how long the state was held rather than being asked to trust it.

See `AckResult` for the fields, `measure` for one condition, and `run_experiment` for the matrix
and the JSON it writes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The gate this measures, quoted from `tests/test_main.py::InputAcknowledgement`. **Not a
#: parameter of anything here**, on purpose - see the module docstring.
GATE_P95_MS = 200.0
GATE_MAX_TICK_MS = 2000.0

#: The timer period, the same 20 ms the existing gate uses. A shorter period would measure Qt's
#: timer coalescing as well as the application, and a longer one would hide a 15 ms stall.
TICK_MS = 20

#: The calibrated saturator count, and the same expression the existing gate uses. **A floor, not
#: a bound, and not comparable across machines** - at `cpu_count()` of 8 the same probe measured
#: p95 195.6 ms and max 1105.5 ms while still reporting green.
def calibrated_saturators() -> int:
    return max(2, (os.cpu_count() or 4) - 2)


@dataclasses.dataclass(frozen=True)
class AckResult:
    """One measured window: what the gate would say, and what the screen would say.

    Frozen because a result that can be edited after the fact is a result that can be reported
    favourably. Every field is either a measurement, a condition, or `None` meaning **not
    measured** - never `0` standing in for an absence.

    ## The two groups of fields, and why they are not one number

    `tick_p50_ms`, `tick_p95_ms`, `tick_max_ms` are **UI-thread tick lateness**: how late a 20 ms
    timer was. That is what a person at the booth experiences as "the screen did not respond".

    `frame_age_p50_ms`, `frame_age_p95_ms`, `backlog_per_stream` are **inference backlog**: how old
    the picture on the tile is, and how deep the queue behind it is. That is what "the booth is
    showing an eleven-second-old picture" is made of.

    **They are not added, averaged, or reconciled.** Frame age is not inference cost; on the NPU
    the measured p95 frame age is seconds while the inference itself costs about 5.9 ms, and the
    gap is queue wait. They stayed merged once already and had to be unpicked.
    """

    #: Window label, e.g. `allcpu-3s-0sat`. The key the JSON and the report table use.
    condition: str

    # -- what the gate measures ---------------------------------------------------------
    tick_p50_ms: float | None
    tick_p95_ms: float | None
    tick_max_ms: float | None
    #: How many ticks the window actually produced. A percentile over a handful of samples is a
    #: statement about almost nothing, so this is published and the tests check it.
    ticks: int

    # -- what the gate does not measure -------------------------------------------------
    #: Per-stream frame age p50, in milliseconds. **`None` for every stream merged is not
    #: offered**: M2's `stream_frame_ages_ms` records that the merged figure over three streams
    #: is "a number about no camera at all", so this field is the **worst** per-stream p50 - the
    #: camera a presenter would be looking at. `frame_age_p50_by_stream` has all three.
    frame_age_p50_ms: float | None
    frame_age_p95_ms: float | None
    frame_age_p50_by_stream: Mapping[str, float | None]
    frame_age_p95_by_stream: Mapping[str, float | None]
    backlog_per_stream: Mapping[str, int | None]

    # -- the conditions -----------------------------------------------------------------
    saturators: int
    streams: int
    #: `allcpu` (every stream on the CPU), `own-devices` (CPU / GPU / NPU, one each), or `cpu`.
    device_layout: str
    #: `stream_id -> requested device`, as passed on the command line.
    requested: Mapping[str, str]
    #: `stream_id -> confirmed devices`, read from the application's own `RoutePlan`. A screen
    #: whose placement was never proved is not the booth's screen.
    confirmed: Mapping[str, tuple[str, ...]]
    #: How long the load was held **before** this window opened, in seconds, as configured. This
    #: is the **budget**, not the elapsed time.
    warmup_s: float
    #: **How long the condition's load had actually been applied** when this window closed, in
    #: seconds, from the moment the child process started.
    #:
    #: Separate from `warmup_s` because the application runs **continuously** across repeats: the
    #: backlog does not reset between windows, so the second repeat's window opens about 200 ms
    #: after the first closed while the load has in fact been running for minutes. Reporting
    #: `warmup_s` alone would say "130 s of warm-up" for a window that opened after 0.2 s, which
    #: is true of the *first* window and false of every later one. A reader comparing repeats
    #: needs the elapsed figure, and the budget is still worth keeping so the configured intent
    #: is on the record too.
    load_applied_s: float
    #: The measured window, in seconds.
    window_s: float
    #: This repeat's index, and how many there are in total.
    repeat: int
    repeats: int
    #: `supervisor.frames_analysed` at the moment the window closed.
    frames_analysed: int
    #: Frames refused for want of room, summed. **`drops` is not backlog**: a drop is the queue
    #: shedding, and a queue that sheds and stays healthy is a different failure from one that
    #: grows. Both are reported so neither can be mistaken for the other.
    drops: int
    #: `True` only if the probe went through `app.main.run_windowed`'s own seams. A number from
    #: a substitute harness is not this gate's number, and the tests assert this.
    application_path: bool
    #: One line naming every condition, so a figure cannot be quoted without it.
    conditions_documented: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    # -- the two verdicts, kept apart ---------------------------------------------------

    def gate_verdict(self) -> str:
        """What the **existing gate** would say about this window. `'unmeasured'` if no ticks.

        **The gate's own thresholds, quoted.** It compares p95 against 200 ms and the worst tick
        against 2 s. Nothing here can change them, and this method is the only place the
        comparison is made, so there is one verdict per window rather than two that can disagree.
        """
        if self.tick_p95_ms is None or self.tick_max_ms is None:
            return "unmeasured"
        if self.tick_p95_ms >= GATE_P95_MS or self.tick_max_ms >= GATE_MAX_TICK_MS:
            return "fail"
        return "pass"

    def staleness_verdict(self) -> str:
        """What the **frame age** says about the same window, on the app's own 2 s rule.

        `UiBridge.STALE_AFTER_SECONDS` is 2.0 s and spec section 7 is its source. This is a
        **separate claim with separate words**: a window can pass one verdict and fail the other,
        and saying so is the whole point of putting them in one table.
        """
        from app.ui.bridge import STALE_AFTER_SECONDS

        if self.frame_age_p50_ms is None:
            return "unmeasured"
        return "stale" if self.frame_age_p50_ms > STALE_AFTER_SECONDS * 1000.0 else "fresh"

    def agrees_with_itself(self) -> bool:
        """Whether the two verdicts agree. **A `False` here is the finding, not a defect.**

        A window where the gate passes and the frame age is stale is the gate reporting a healthy
        system while the booth shows an old picture. `True` means the gate and the staleness rule
        happened to land on the same side of their own thresholds, which says nothing about
        whether the gate can see the backlog.
        """
        return self.gate_verdict() == "pass" and self.staleness_verdict() == "fresh"


# --------------------------------------------------------------------------------------
# The child program: one window, both quantities
# --------------------------------------------------------------------------------------
#
# **Same three seams as `tests/test_main.py::InputAcknowledgement`,** for the reason in the module
# docstring. The differences from that gate are all additive: the saturators are optional, the
# streams and device layout are parameters, the warm-up can be long enough to build a real
# backlog, and every window reports the frame age and backlog it ran with.
#
# It is a separate program rather than a second entry point in the test file because it is a
# **tool**, used by a test and by a person at a terminal, and `tests/test_main.py` is not
# importable without running that module's collection.

_CHILD = r'''
"""One window, measured through `app.main`'s own threading path. Both quantities, together.

Run as a **file** and as `__main__`. Two reasons, both measured on this project already:
`multiprocessing` on Windows re-imports the module in a spawned worker and trips its
`_check_not_importing_main` guard if it is run from `python -c`; and the original ack probe did
exactly that, exited 1, and printed p95 **20.9 ms** — which *passes* a 200 ms gate. **A probe that
cannot spawn and still reports a passing number is worse than no probe**, so this raises rather
than defaulting if it cannot reach a measured window.

Prints one `ACKLD {json}` line per window on stdout.
"""
import json
import math
import multiprocessing
import os
import sys
import time

sys.path.insert(0, @REPO@)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer

import app.main as main_module
# The application's OWN percentile. Imported, not reimplemented, so this tool cannot disagree
# with the screen about what p95 means - `app/main.py::_publish_stream_metrics` reads the ages
# and calls this same function.
from app.main import _percentile

SECONDS = float(os.environ["ACKLD_SECONDS"])
WARMUP = float(os.environ["ACKLD_WARMUP"])
SATURATORS = int(os.environ["ACKLD_SATURATORS"])
REPEATS = int(os.environ["ACKLD_REPEATS"])
TICK_MS = int(os.environ.get("ACKLD_TICK_MS", "20"))
CONDITION = os.environ.get("ACKLD_CONDITION", "unnamed")

#: The clock origin for "how long has this condition's load actually been applied". **Not the
#: process start**, which would include interpreter and device-discovery time and so overstate it
#: by several seconds.
STARTED = time.perf_counter()

lateness = []
state = {"due": 0.0, "tick": None, "supervisor": None, "bridge": None, "procs": [],
         "gate": None, "phase": "warmup", "warm_started": 0.0, "repeat": 0,
         "window": None, "warmed": 0}
_real_feeders = main_module._start_feeders


def burn(_gate_holder=None, _index=0, _gate=None):
    """One saturator core: CPU-only arithmetic, no allocation, no syscalls.

    **Waits on an event before burning**, and the saturators are spawned during warm-up rather
    than inside the window callback. Spawning them in the window callback cost the first version
    of the existing gate **one tick in a six-second window** — `Process.start()` on Windows
    re-imports the module in each child, that cost seconds, and it landed on the Qt main thread
    because the warm-up poller is a `QTimer`. The probe was timing process creation.
    """
    value = 0
    budget = (WARMUP + SECONDS * (REPEATS + 2)) * 4.0
    deadline = time.perf_counter() + budget
    while time.perf_counter() < deadline:
        if _gate is None or not _gate.is_set():
            time.sleep(0.02)
            deadline = time.perf_counter() + budget
            continue
        for i in range(20000):
            value = (value * 31 + i) % 1000003
    return value


def spawn_saturators():
    """Create the saturator processes once, gated. Returns `(processes, gate)`.

    With `SATURATORS == 0` this returns `([], None)` and nothing is spawned, which is not a
    special case bolted on: "the application's own load, no synthetic saturators" is
    **condition 1 of the experiment** and it is the one the verdict turns on.
    """
    if SATURATORS <= 0:
        return [], None
    gate = multiprocessing.Event()
    procs = [multiprocessing.Process(target=burn, args=(None, i, gate))
             for i in range(SATURATORS)]
    for proc in procs:
        proc.start()
    return procs, gate


def tick():
    """One UI-thread tick: the gap between when it was due and when this slot ran.

    Lateness can read very slightly negative on the first tick, because `due` is armed just
    before the timer starts and Qt may fire a touch early. **A negative lateness is not a
    measurement**, so it is clamped at zero - and the clamp is published as
    `negative_ticks_clamped` rather than left for a reader to wonder about.
    """
    now = time.perf_counter()
    raw = (now - state["due"]) * 1000.0
    if raw < 0.0:
        state["negative"] = state.get("negative", 0) + 1
    lateness.append(max(0.0, raw))
    state["due"] = now + TICK_MS / 1000.0


def maybe_open_window():
    """Warm up, then open the measured window. Re-entered every `POLL_MS` from a `QTimer`.

    **The `phase` guard is load-bearing.** The first version of the existing gate re-entered this
    on every poll; because `frames_analysed > 0` stays true forever once the route is up, each
    re-entry cleared `lateness` and restarted both timers, so the window was reset every poll,
    never elapsed, and only the backstop ever ended it. Measured: **one tick per window**, p50
    0.0 ms, which looks flawless and measures nothing.
    """
    if state["phase"] == "measuring":
        return
    supervisor = state["supervisor"]
    elapsed = time.perf_counter() - state["warm_started"]
    if state["phase"] == "warmup":
        # Warm-up ends when the route is confirmed and frames are being analysed, **or** when
        # the warm-up budget is spent - whichever comes first. Warming up unsaturated until the
        # pipeline is real, then measuring, measures what the plan asks about. With saturators
        # the first version measured a *starved worker*, which measures the CPU compile.
        ready = (supervisor.frames_analysed > state["warmed"]
                 and state["warmed"] > 0)
        if not (ready or elapsed >= WARMUP):
            return
        state["warmed"] = supervisor.frames_analysed
        state["phase"] = "measuring"
    lateness.clear()
    state["negative"] = 0
    state["due"] = time.perf_counter() + TICK_MS / 1000.0
    state["tick"].start(TICK_MS)
    if state["gate"] is not None:
        state["gate"].set()
    state["window"].start(int(SECONDS * 1000))


def close_window():
    """Close one window, report it, and either open the next or quit.

    **`report_window()` cannot be allowed to raise out of here.** Measured, and it is a real
    defect rather than a hypothetical: this function's first version called `report_window()`
    and then `APP["q"].quit()`, so when the report raised - it did, on a `stream_ids()` that is
    a property and not a method - the exception unwound out of the `QTimer` slot, PySide printed
    it to stderr and carried on, and **`quit()` was never called**. The application sat in
    `exec()` until the parent killed it at its 1800 s timeout, with no `ACKLD` line anywhere.
    The parent would have reported "no ACKLD line" rather than a wrong number, so nothing false
    got through - but the diagnosis cost 30 minutes and the fix is one `finally`.

    So: report inside `try`, and **always** stop the timer, terminate the saturators and quit.
    The last window's `quit()` happening after a raise is the difference between a 40 s run and
    a 30 minute one.
    """
    if state["gate"] is not None:
        state["gate"].clear()
    state["tick"].stop()
    try:
        report_window()
    except BaseException as exc:  # noqa: BLE001 - a probe must report, never die silently
        emit_error(f"reporting the window raised {type(exc).__name__}: {exc}")
    state["repeat"] += 1
    if state["repeat"] >= REPEATS:
        terminate_saturators()
        APP["q"].quit()
        return
    # Warm up again between repeats, so every repeat reports the state it was measured in and
    # not the state the previous window left behind.
    state["warm_started"] = time.perf_counter()
    state["phase"] = "warmup"


def terminate_saturators():
    for proc in state["procs"]:
        proc.terminate()
        proc.join(timeout=2.0)
    state["procs"] = []


def backlog_table():
    """`stream_id -> queue depth`, computed the way the application computes it.

    **`frames_accepted - frames_analysed - frames_invalidated`**, which is exactly
    `app/main.py::_publish_stream_metrics`'s arithmetic: submitted, taken off the queue, and
    thrown away by a route change are all excluded, so what is left is the queue's own depth.
    Reimplemented here rather than read from the bridge because the bridge's copy is private and
    it is refreshed on the telemetry interval; this is read at the moment the window closes.

    `None` before the stream has accepted anything, so "no work offered yet" never reads as a
    healthy empty queue.
    """
    supervisor = state["supervisor"]
    out = {}
    for stats in supervisor.per_stream():
        depth = (stats.frames_accepted - stats.frames_analysed
                 - stats.frames_invalidated)
        out[stats.stream_id] = depth if stats.frames_accepted > 0 else None
    return out


def emit_error(message):
    """One `ACKLD` line carrying an error and **`None` figures, not zeros.**

    A zero lateness and an absent measurement are different facts. A probe that failed must say
    so on stdout, where the parent reads it, and the parent must refuse the run - which it does,
    because `"error" in payload` is what it checks for.
    """
    print("ACKLD " + json.dumps({
        "condition": CONDITION, "repeat": -1, "error": message,
        "tick_p50_ms": None, "tick_p95_ms": None, "tick_max_ms": None, "ticks": 0,
        "saturators": SATURATORS, "warmup_s": WARMUP, "window_s": SECONDS,
        "application_path": True,
    }), flush=True)


def report_window():
    """One `ACKLD` line: the gate's numbers and the screen's numbers, from this window."""
    supervisor = state["supervisor"]
    ordered = sorted(lateness)

    def pct(fraction):
        # The app's own nearest-rank percentile, so p95 here and p95 on the card are the same
        # function applied to the same samples.
        return _percentile(ordered, fraction)

    # **`stream_ids` is a property, not a method.** Measured: calling it raised `TypeError:
    # 'tuple' object is not callable` from inside the `QTimer` slot, which is the hang in
    # `close_window`'s docstring.
    stream_ids = supervisor.stream_ids
    ages50, ages95 = {}, {}
    for stream_id in stream_ids:
        ages = supervisor.stream_frame_ages_ms(stream_id)
        ages50[stream_id] = _percentile(ages, 0.50)
        ages95[stream_id] = _percentile(ages, 0.95)

    def worst(table):
        values = [v for v in table.values() if v is not None]
        return max(values) if values else None

    plan = supervisor.route_plan()
    requested, confirmed = {}, {}
    for stream_id in stream_ids:
        requested[stream_id] = plan.requested_device(stream_id)
        confirmed[stream_id] = list(plan.confirmed_devices(stream_id))

    p50, p95, pmax = pct(0.50), pct(0.95), (ordered[-1] if ordered else None)
    print("ACKLD " + json.dumps({
        "condition": CONDITION,
        "repeat": state["repeat"],
        "repeats": REPEATS,
        "tick_p50_ms": None if p50 is None else round(p50, 3),
        "tick_p95_ms": None if p95 is None else round(p95, 3),
        "tick_max_ms": None if pmax is None else round(pmax, 3),
        "ticks": len(ordered),
        "negative_ticks_clamped": state.get("negative", 0),
        "tick_ms": TICK_MS,
        "frame_age_p50_by_stream": ages50,
        "frame_age_p95_by_stream": ages95,
        "frame_age_p50_ms": worst(ages50),
        "frame_age_p95_ms": worst(ages95),
        "backlog_per_stream": backlog_table(),
        "saturators": SATURATORS,
        "streams": len(stream_ids),
        "requested": requested,
        "confirmed": confirmed,
        "warmup_s": WARMUP,
        "load_applied_s": round(time.perf_counter() - STARTED, 1),
        "window_s": SECONDS,
        "frames_analysed": supervisor.frames_analysed,
        "drops": supervisor.drops,
        "application_path": True,
    }), flush=True)


def spy_feeders(bridge, supervisor, clip):
    """The application's own feeder start, plus this probe's timers.

    **The tick timer is armed here**, inside `run_windowed`'s own startup and before
    `application.exec()`, so the thread doing the measuring is the thread that renders the booth.
    The wrapper then calls the real thing, so the feeder, the reader threads and the result fold
    are `app/main.py`'s and not this probe's.
    """
    state["supervisor"] = supervisor
    state["bridge"] = bridge
    state["warm_started"] = time.perf_counter()
    # Spawned before the tick timer starts and from the warm-up path, so no process-creation
    # cost ever lands inside a measured window.
    state["procs"], state["gate"] = spawn_saturators()
    state["tick"] = QTimer()
    state["tick"].timeout.connect(tick)
    state["window"] = QTimer()
    state["window"].setSingleShot(True)
    state["window"].timeout.connect(close_window)
    # The warm-up poller runs on the UI thread too, so it is part of what is being measured -
    # and it is idle work, not analysis.
    state["warmup"] = QTimer()
    state["warmup"].timeout.connect(maybe_open_window)
    state["warmup"].start(200)
    return _real_feeders(bridge, supervisor, clip)


main_module._start_feeders = spy_feeders

_real_gui = main_module._gui_application
APP = {}
POLL_BACKSTOP = WARMUP + SECONDS * (REPEATS + 2)


def spy_gui():
    """The application's own GUI application, plus the backstop that cannot hang."""
    qobject = _real_gui()
    APP["q"] = qobject

    def stop_and_report():
        # Quits **first**, so a raising backstop cannot leave the event loop spinning until the
        # harness kills it - which is exactly what happened once already.
        qobject.quit()
        try:
            if state["phase"] == "measuring":
                report_window()
            else:
                # **`None` figures, loudly.** A zero that is really "never ran" would pass a gate.
                emit_error(f"the measured window never started: no frames were analysed "
                           f"within {WARMUP} s of warm-up")
        except BaseException as exc:  # noqa: BLE001
            emit_error(f"the backstop raised {type(exc).__name__}: {exc}")
        terminate_saturators()

    QTimer.singleShot(int(POLL_BACKSTOP * 1000), stop_and_report)
    return qobject


main_module._gui_application = spy_gui

if __name__ == "__main__":
    sys.exit(main_module.main(sys.argv[1:]))
'''


# --------------------------------------------------------------------------------------
# One condition
# --------------------------------------------------------------------------------------

#: The conditions the experiment runs. `label` is what the report table prints.
#:
#: **`warmup_s` is a measured number, not a preference.** At 15 fps offered and about 13.5 fps
#: analysed, a CPU stream gains 1.50 frames of backlog per second, so the 173-187 frames
#: M2 measured is **115-125 s** of accumulation. 187 / 1.50 = 124.7 and 173 / 1.50 = 115.3. The three-stream conditions therefore warm up for
#: `BACKLOG_WARMUP_S`; a 6-second window cannot show the booth's real state and saying otherwise
#: would be the whole error this task exists to catch.
BACKLOG_WARMUP_S = 130.0

CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "label": "three-cpu-no-saturators",
        "condition": "three streams, spec section 5 row 4, all on the CPU, no synthetic load",
        "argv": ["--device", "CPU", "--streams", "3",
                 "--select", "A=CPU", "--select", "B=CPU", "--select", "C=CPU"],
        "saturators": 0,
        "warmup_s": BACKLOG_WARMUP_S,
        "device_layout": "allcpu",
    },
    {
        "label": "three-own-devices-no-saturators",
        "condition": "three streams, one per device (CPU / GPU / NPU), no synthetic load",
        "argv": ["--device", "CPU", "--streams", "3",
                 "--select", "A=CPU", "--select", "B=GPU", "--select", "C=NPU"],
        "saturators": 0,
        "warmup_s": BACKLOG_WARMUP_S,
        "device_layout": "own-devices",
    },
    {
        "label": "one-cpu-saturated",
        "condition": "the existing gate's own conditions: one stream on the CPU, saturated",
        "argv": ["--device", "CPU", "--streams", "1"],
        "saturators": None,          # `None` means "the calibrated count"
        "warmup_s": 12.0,            # the existing gate's warm-up, kept so the number is comparable
        "device_layout": "cpu",
    },
    {
        "label": "three-cpu-saturated",
        "condition": ("the application's own load AND the existing gate's saturation - the "
                      "decisive pair with three-cpu-no-saturators"),
        "argv": ["--device", "CPU", "--streams", "3",
                 "--select", "A=CPU", "--select", "B=CPU", "--select", "C=CPU"],
        "saturators": None,
        "warmup_s": BACKLOG_WARMUP_S,
        "device_layout": "allcpu",
    },
)


# --------------------------------------------------------------------------------------
# Is this machine fit to measure on at all?
# --------------------------------------------------------------------------------------
#
# **The standing rule, made mechanical.** A tick, leak or latency figure taken while other
# Python processes are running is not a measurement of this application; it is a measurement of
# whatever else was on the machine. This was not hypothetical: M3 Task 3's first live run had
# `frames_analysed=0` and empty confirmed placements on a machine carrying **66 orphaned
# `python.exe` processes** - stale inference workers and two duplicate full-suite runs from
# interrupted experiments - and the failures looked like code defects until the census was taken.
#
# So `measure()` refuses to start a window on a machine it can see other Python processes on,
# unless the operator says so explicitly with `ACKLD_ALLOW_DIRTY=1`, and a dirty run is recorded
# as such in the JSON rather than being indistinguishable from a clean one.

#: Set to `1` to let a measurement proceed with competing Python processes present. **Deliberate
#: and recorded**, not a default: the point is that a run does not silently measure a machine
#: somebody else is already loading. See `assert_machine_clean`.
CLEAN_OVERRIDE_ENV = "ACKLD_ALLOW_DIRTY"


class MachineNotClean(RuntimeError):
    """**The refusal, as a type.** A measurement was declined because another Python process was
    on the machine - either before the window opened or during it.

    It is a `RuntimeError`, deliberately: every caller written before this class existed already
    refuses a dirty measurement by catching `RuntimeError` / letting one propagate, and those
    callers keep exactly the behaviour they had. Nothing about the refusal is relaxed by naming
    it - `measure()` still raises and still returns **no row**, so no figure measured on a dirty
    machine can reach a record.

    What the type buys is the one thing a *caller* could not do before: tell this refusal apart
    from a real failure ("the probe produced nothing", "the window never opened"). A test suite
    that hits it on a busy developer machine should **skip with the reason** rather than error -
    an error says "the code is broken", and this says "the machine is busy". A test suite that
    hit any other `RuntimeError` must still error, because that one *is* the code.

    Raised by `assert_machine_clean` and by `measure`'s mid-window census. `ACKLD_ALLOW_DIRTY=1`
    still turns both into a recorded, deliberately dirty run.
    """


def _process_table() -> dict[int, tuple[int, str]]:
    """`pid -> (parent_pid, image_name)` for every process on the machine.

    A `CreateToolhelp32Snapshot` walk: **no dependency, no elevation**, and - unlike `tasklist` -
    it returns the parent PID, which is what lets the census tell this run's measurement child
    apart from somebody else's process. Windows-only; `{}` elsewhere or if the snapshot fails.
    """
    if os.name != "nt":
        return {}
    import ctypes
    from ctypes import wintypes

    TH32CS_SNAPPROCESS = 0x00000002
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD),
                    ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_void_p),
                    ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_char * 260)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    snapshot = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == INVALID_HANDLE_VALUE:
        return {}
    table: dict[int, tuple[int, str]] = {}
    try:
        entry = PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
        more = k32.Process32First(snapshot, ctypes.byref(entry))
        while more:
            name = entry.szExeFile.decode("mbcs", errors="ignore")
            table[int(entry.th32ProcessID)] = (int(entry.th32ParentProcessID), name)
            more = k32.Process32Next(snapshot, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snapshot)
    return table


def _our_process_tree(table: Mapping[int, tuple[int, str]] | None = None) -> set[int]:
    """This process, every **ancestor**, and every **descendant of this process**.

    - **Ancestors**, so a wrapper (`python run_tests.py` running `python -m unittest`) is not
      mistaken for a competitor.
    - **Descendants of *this* process only** - the measurement child and its saturators - so the
      census can run *during* a child and still not count the child as competition. A competitor
      from another session is a child of a shared **ancestor** (the harness), not a child of us,
      so it stays outside this set. That is the discrimination that makes a mid-window competitor
      detectable at all.
    """
    table = table if table is not None else _process_table()
    ours: set[int] = set()
    pid = os.getpid()
    while pid and pid not in ours:
        ours.add(pid)
        pid = table.get(pid, (0, ""))[0]
    children: dict[int, list[int]] = {}
    for child, (parent, _name) in table.items():
        children.setdefault(parent, []).append(child)
    stack = [os.getpid()]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in ours:
                ours.add(child)
                stack.append(child)
    return ours


def python_processes(table: Mapping[int, tuple[int, str]] | None = None) -> list[int]:
    """Every `python.exe` PID on this machine.

    `[]` on a non-Windows host or if the census itself fails. **An empty census is read as "cannot
    see a competitor", not as "clean"** - `machine_state` says which it was.

    `table` is passed by `machine_state` so that **one snapshot** answers both "which PIDs are
    Python" and "which of them are ours". See `machine_state`.
    """
    table = table if table is not None else _process_table()
    return sorted(pid for pid, (_parent, name) in table.items()
                  if name.lower() == "python.exe")


def machine_state() -> dict[str, Any]:
    """A census of the machine: our process tree, every `python.exe`, and the difference.

    **`competing_pids` is the number that matters.** Anything running Python that is not this
    process or one of its ancestors is either a leaked worker from an interrupted run or a
    second suite, and either one invalidates a timing figure measured now.

    ## One snapshot, not two

    **The two questions must be answered from the same process-table snapshot.** The first
    version called `_our_process_tree()` (snapshot A) and then `python_processes()` (snapshot
    B). A child of ours that started between the two reads was in B's Python list but not in
    A's descendant set, so **our own measurement worker was reported as a competing process**.
    Measured: a mid-window poll refused a clean run with exactly one extra PID, twice, on a
    machine that idles with zero `python.exe` processes; the probe spawns three workers a
    moment after it is started, which is precisely the interval between the two reads. A guard
    that cries wolf is a guard somebody turns off, so the snapshot is taken once.
    """
    table = _process_table()
    ours = _our_process_tree(table)
    pythons = python_processes(table)
    competing = sorted(p for p in pythons if p not in ours)
    clean = not competing
    return {
        "clean": clean,
        "our_pids": sorted(ours),
        "python_pids": sorted(pythons),
        "competing_pids": competing,
        "detail": ("no Python process outside this run's process tree" if clean else
                   f"{len(competing)} Python process(es) outside this run's process tree: "
                   f"{competing}"),
    }


def assert_machine_clean(*, settle_s: float = 5.0,
                         allow_dirty: bool | None = None) -> dict[str, Any]:
    """Raise unless nothing but this run's own process tree is running Python.

    `settle_s` gives a just-finished child a moment to be reaped before the census is believed -
    a leaked worker from the previous condition would otherwise refuse the next one. The override
    is `ACKLD_ALLOW_DIRTY=1`, and it is **recorded** in the returned census so a dirty run is
    visible in the JSON instead of looking like a clean one.
    """
    if allow_dirty is None:
        allow_dirty = os.environ.get(CLEAN_OVERRIDE_ENV) == "1"
    state = machine_state()
    if not state["clean"] and settle_s > 0:
        deadline = time.perf_counter() + settle_s
        while not state["clean"] and time.perf_counter() < deadline:
            time.sleep(0.25)
            state = machine_state()
    if state["clean"]:
        return state
    message = (
        f"the machine is not clean enough to measure on: {state['detail']}. A tick, leak or "
        f"latency figure taken while other Python processes run is not a measurement of this "
        f"application. Stop them (on Windows: `Stop-Process -Id "
        f"{', '.join(str(p) for p in state['competing_pids'])}`) and run again, or set "
        f"{CLEAN_OVERRIDE_ENV}=1 to record a deliberately dirty run.")
    if not allow_dirty:
        raise MachineNotClean(message)
    state["overridden"] = True
    state["warning"] = message
    return state


def _run_monitored(argv: Sequence[str], *, env: Mapping[str, str], timeout: float,
                   stdout: Any, stderr: Any) -> tuple[int, list[dict[str, Any]]]:
    """Run the measurement child, **polling the census while it runs**.

    Returns `(returncode, censuses_that_saw_a_competitor)`. The census excludes this process's own
    descendants (see `_our_process_tree`), so the child being measured - and the saturators it
    spawns - are not mistaken for competition, while a process from another session is.

    Polling every 2 s for the length of a condition is cheap (a `CreateToolhelp32Snapshot` walk),
    and it is the difference between "the machine was clean before we started" and "the machine
    was clean for the whole window". A competitor that appears at minute two of a three-minute
    warm-up is caught here and the figure is refused rather than believed.
    """
    proc = subprocess.Popen(list(argv), stdout=stdout, stderr=stderr, text=True,
                            cwd=str(REPO_ROOT), env=dict(env))
    contaminated: list[dict[str, Any]] = []
    deadline = time.perf_counter() + timeout
    try:
        while proc.poll() is None:
            if time.perf_counter() > deadline:
                proc.kill()
                proc.wait()
                raise subprocess.TimeoutExpired(list(argv), timeout)
            state = machine_state()
            if not state["clean"]:
                contaminated.append(state)
            time.sleep(2.0)
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise
    return int(proc.returncode or 0), contaminated


def measure(condition: Mapping[str, Any], *, seconds: float, repeats: int,
            saturators: int | None = None, warmup_s: float | None = None,
            tick_ms: int = TICK_MS, timeout: float = 1800.0,
            python: str | None = None) -> list[AckResult]:
    """Run one condition, `repeats` times, and return one `AckResult` per window.

    **One child process for all the repeats of a condition**, because the fixed cost is startup:
    `DeviceRegistry.describe()` is 3.3-3.8 s cold on this machine, the CPU compile is seconds,
    and `Supervisor.stop()` joins with a grace timeout. Measured on the existing gate: one repeat
    per process cost **63 s** wall for 16 s of measurement.

    `warmup_s` **overrides the condition's own**, and that override is what lets the test suite
    run a cheap version of this without pretending it is the experiment. The `AckResult` carries
    the warm-up it actually used, so a short one is visible in the record instead of hidden in a
    constant.

    Raises `RuntimeError` if no window opened or the probe did not reach the application's
    threading path. It does not return a zeroed result: a probe that cannot measure and reports
    "no lateness, no backlog" is indistinguishable from a healthy system.

    Raises `MachineNotClean` - a `RuntimeError` subclass, so callers that already catch
    `RuntimeError` are unchanged - when another Python process is running before the window or
    appears during it. **Both refusals return no row**, so a figure measured on a dirty machine
    cannot be recorded; the type only lets a test suite skip with the reason instead of erroring.
    """
    label = str(condition["label"])
    # **Refuse to measure on a machine somebody else is loading.** A tick/latency figure taken
    # while a leaked worker or a second suite is running is not this application's figure. See
    # `assert_machine_clean` for the rule and the `ACKLD_ALLOW_DIRTY` override.
    assert_machine_clean()
    count = int(saturators if saturators is not None else
                (condition["saturators"] if condition["saturators"] is not None
                 else calibrated_saturators()))
    warm = float(warmup_s if warmup_s is not None else condition["warmup_s"])
    # **`--seconds` is deliberately not passed.** It is documented as "how long a *headless* run
    # lasts" and `run_windowed` ignores it — the windowed path runs until `quit()`. Passing it
    # would imply a cutoff that does not exist, and a reader checking the command line would have
    # to go and find that out.
    argv = ["--windowed"] + list(condition["argv"])

    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["ACKLD_SECONDS"] = repr(seconds)
    env["ACKLD_WARMUP"] = repr(warm)
    env["ACKLD_SATURATORS"] = str(count)
    env["ACKLD_REPEATS"] = str(repeats)
    env["ACKLD_TICK_MS"] = str(tick_ms)
    env["ACKLD_CONDITION"] = label

    with tempfile.TemporaryDirectory(prefix="ackld-") as tmp:
        script = pathlib.Path(tmp) / "ackld_probe.py"
        script.write_text(_CHILD.replace("@REPO@", repr(str(REPO_ROOT))),
                          encoding="utf-8", newline="\n")
        out_path = pathlib.Path(tmp) / "ackld_probe.stdout"
        err_path = pathlib.Path(tmp) / "ackld_probe.stderr"
        # **stdout/stderr to files, not pipes.** A pipe that is not drained while this loop polls
        # the census can fill and block the child, which would look exactly like a hang. A file
        # cannot.
        with open(out_path, "w", encoding="utf-8") as out_file, \
                open(err_path, "w", encoding="utf-8") as err_file:
            returncode, contaminated = _run_monitored(
                [python or sys.executable, str(script), *argv],
                env=env, timeout=timeout, stdout=out_file, stderr=err_file)
        stdout_text = out_path.read_text(encoding="utf-8", errors="replace")
        stderr_text = err_path.read_text(encoding="utf-8", errors="replace")
    completed = subprocess.CompletedProcess(argv, returncode, stdout_text, stderr_text)

    if contaminated and os.environ.get(CLEAN_OVERRIDE_ENV) != "1":
        # **A competitor that started mid-window is refused, not averaged in.** The census is
        # polled while the child runs; a second Python process appearing during the window means
        # this figure is about the application *and* about whatever else was running, and the
        # standing rule says such a figure is not a measurement. `MachineNotClean` rather than a
        # bare `RuntimeError`, so a caller can skip with this reason instead of reporting a
        # broken suite - the row is still never returned.
        raise MachineNotClean(
            f"{label}: another Python process started during the measured window, so this window "
            f"is not a clean measurement. Competing censuses seen: "
            f"{json.dumps(contaminated)[:2000]}. Stop the other process and re-run, or set "
            f"{CLEAN_OVERRIDE_ENV}=1 to keep a deliberately dirty figure.")

    lines = [ln for ln in completed.stdout.splitlines() if ln.startswith("ACKLD ")]
    if not lines:
        raise RuntimeError(
            f"{label}: the probe produced no ACKLD line. stdout:\n{completed.stdout[-3000:]}"
            f"\nstderr:\n{completed.stderr[-3000:]}")
    payloads = [json.loads(ln[len("ACKLD "):]) for ln in lines]
    return results_from_payloads(payloads, label, warmup_s=warm, window_s=seconds,
                                 saturators=count, repeats=repeats,
                                 stdout=completed.stdout, stderr=completed.stderr)


def results_from_payloads(payloads: Sequence[Mapping[str, Any]], label: str, *,
                          warmup_s: float, window_s: float, saturators: int, repeats: int,
                          stdout: str = "", stderr: str = "") -> list[AckResult]:
    """The probe's stdout lines into `AckResult`s, **refusing anything that did not measure.**

    Split out of `measure` so the refusal is testable without spawning a process and waiting six
    minutes for it. Two things are refused, and both would otherwise pass a gate:

    - a payload carrying `"error"` — the child's own "the measured window never started";
    - a payload with `application_path` false — a probe that did not go through
      `app.main.run_windowed`, whose numbers are not this gate's numbers however plausible they
      look;
    - **no payloads at all.** An empty list means the probe printed nothing, and returning `[]`
      would let "the probe produced nothing" read as "no lateness, no backlog, healthy" — the
      exact failure shape this project exists to prevent. `measure` refuses an empty stdout one
      level up; this refuses it one level down too, so the refusal is reachable without spawning
      a process and waiting six minutes to test it.
    """
    if not payloads:
        raise RuntimeError(
            f"{label}: the probe produced no ACKLD payloads at all, so nothing was measured. "
            f"Refusing to return an empty result that would read as a healthy system.\n"
            f"stdout tail:\n{stdout[-2000:]}\nstderr tail:\n{stderr[-2000:]}")
    broken = [dict(p) for p in payloads
              if "error" in p or not p.get("application_path")]
    if broken:
        raise RuntimeError(
            f"{label}: the probe did not reach a measured window through the application's "
            f"threading path:\n{json.dumps(broken, indent=2)}\n"
            f"stdout tail:\n{stdout[-2000:]}\nstderr tail:\n{stderr[-2000:]}")
    return [_from_payload(payload, label, warmup_s, window_s, saturators, repeats)
            for payload in payloads]


def _from_payload(payload: Mapping[str, Any], label: str, warmup_s: float,
                  window_s: float, saturators: int, repeats: int) -> AckResult:
    """One `AckLD` payload into an `AckResult`, with every condition field filled in.

    **The two verdict quantities are taken from the payload, never recomputed here** - the probe
    measured them on the thread that rendered the booth, and a second opinion formed from a
    different copy of the samples would be a different measurement.
    """
    backlog = dict(payload.get("backlog_per_stream") or {})
    confirmed = {k: tuple(v) for k, v in (payload.get("confirmed") or {}).items()}
    return AckResult(
        condition=label,
        tick_p50_ms=payload.get("tick_p50_ms"),
        tick_p95_ms=payload.get("tick_p95_ms"),
        tick_max_ms=payload.get("tick_max_ms"),
        ticks=int(payload.get("ticks") or 0),
        frame_age_p50_ms=payload.get("frame_age_p50_ms"),
        frame_age_p95_ms=payload.get("frame_age_p95_ms"),
        frame_age_p50_by_stream=dict(payload.get("frame_age_p50_by_stream") or {}),
        frame_age_p95_by_stream=dict(payload.get("frame_age_p95_by_stream") or {}),
        backlog_per_stream=backlog,
        saturators=int(payload.get("saturators", saturators)),
        streams=int(payload.get("streams") or 0),
        device_layout=_layout_for(label),
        requested=dict(payload.get("requested") or {}),
        confirmed=confirmed,
        warmup_s=warmup_s,
        load_applied_s=float(payload.get("load_applied_s") or warmup_s),
        window_s=window_s,
        repeat=int(payload.get("repeat", -1)),
        repeats=int(payload.get("repeats", repeats)),
        frames_analysed=int(payload.get("frames_analysed") or 0),
        drops=int(payload.get("drops") or 0),
        application_path=bool(payload.get("application_path")),
        conditions_documented=(
            f"{payload.get('streams')} stream(s), layout "
            f"{_layout_for(label)}, {payload.get('saturators')} saturator process(es), "
            f"warm-up budget {warmup_s:.0f} s, load applied for "
            f"{payload.get('load_applied_s')} s, {window_s:.0f} s window, "
            f"{payload.get('frames_analysed')} frames analysed, {payload.get('drops')} drops, "
            f"live through app.main.run_windowed"),
    )


def _layout_for(label: str) -> str:
    for condition in CONDITIONS:
        if condition["label"] == label:
            return str(condition["device_layout"])
    return "unrecorded"


# --------------------------------------------------------------------------------------
# The experiment: the matrix, and the JSON
# --------------------------------------------------------------------------------------

#: Where the CLI writes. **Under `logs/`, which is gitignored**, because every run overwrites it
#: and a re-run on a different day is a different measurement: this is the log of what was run.
#:
#: **A reviewed copy belongs in `docs/qualification/`, and that is a deliberate second step.**
#: `docs/qualification/` in this project holds records a later reader will treat as cited
#: evidence, and a figure nobody can open is not evidence. So the sequence is: the CLI writes to
#: `logs/`, a human reads the conditions in it, and only then does a copy go beside
#: `m3-capture/` as `docs/qualification/m3-input-ack.json`. The brief asks for the `logs/` path and
#: this is that path; the committed copy is the reviewed one and it carries the same bytes.
DEFAULT_JSON = REPO_ROOT / "logs" / "qualification" / "m3-input-ack.json"

#: The reviewed copy, written only on request with `--also-write-docs`. **Not the default**, and
#: deliberately so: putting a figure into the evidence directory without anybody having read its
#: conditions is how a stale number becomes a cited one.
DOCS_JSON = REPO_ROOT / "docs" / "qualification" / "m3-input-ack.json"


def run_experiment(*, seconds: float = 30.0, repeats: int = 2,
                   warmup_s: float | None = None, labels: Sequence[str] | None = None,
                   json_path: pathlib.Path | None = None,
                   python: str | None = None) -> dict[str, Any]:
    """Run the condition matrix and return the record, ready to be written as JSON.

    **Repeats are not optional and not one.** M1's gate already established that one run is an
    anecdote; this machine has also measurably slowed during M2-M3 for reasons not established,
    so a single figure from a single run is not a figure at all. Every row in the returned table
    carries its own repeat index and the number of repeats, and the table reports the **range**
    across repeats rather than a chosen member.
    """
    wanted = set(labels) if labels else None
    # **Assert the machine is clean before any timing.** A dirty machine is refused here, before
    # a single window opens, rather than producing four holes that each look like a probe fault.
    machine = assert_machine_clean()
    rows: list[AckResult] = []
    failures: list[dict[str, str]] = []
    for condition in CONDITIONS:
        if wanted and condition["label"] not in wanted:
            continue
        try:
            rows.extend(measure(condition, seconds=seconds, repeats=repeats,
                                warmup_s=warmup_s, python=python))
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            # **Recorded, not swallowed.** A condition that could not run is a hole in the table,
            # and a hole that reads as a zero would be worse than a hole that reads as a hole.
            failures.append({"condition": str(condition["label"]),
                             "error": str(exc)[:4000]})
    return build_record(rows, failures, seconds=seconds, repeats=repeats,
                        warmup_s=warmup_s, machine=machine)


def build_record(rows: Sequence[AckResult], failures: Sequence[Mapping[str, str]],
                 *, seconds: float, repeats: int, warmup_s: float | None,
                 machine: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The JSON payload: every row, the per-condition ranges, and the co-reporting table.

    **`the_table` is the deliverable.** Tick lateness and frame age side by side, one row per
    measured window, with every condition on the row. A reader must be able to see the two
    columns disagree without reading a sentence first.

    `machine` is the cleanliness census taken before the run (`machine_state`), so a figure
    cannot be quoted without the fact that no competing Python process was running when it was
    measured. `None` when the caller did not take one — a record built by a unit test.
    """
    ranges: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = ranges.setdefault(row.condition, {
            "condition": row.condition,
            "device_layout": row.device_layout,
            "streams": row.streams,
            "saturators": row.saturators,
            "warmup_s": row.warmup_s,
            "load_applied_s": row.load_applied_s,
            "window_s": row.window_s,
            "repeats_seen": 0,
            "tick_p50_ms": [], "tick_p95_ms": [], "tick_max_ms": [],
            "frame_age_p50_ms": [], "frame_age_p95_ms": [], "backlog": [],
        })
        entry["repeats_seen"] += 1
        for field in ("tick_p50_ms", "tick_p95_ms", "tick_max_ms",
                      "frame_age_p50_ms", "frame_age_p95_ms"):
            value = getattr(row, field)
            if value is not None:
                entry[field].append(round(float(value), 1))
        depths = [v for v in row.backlog_per_stream.values() if v is not None]
        if depths:
            entry["backlog"].append(max(depths))
    for entry in ranges.values():
        for field in ("tick_p50_ms", "tick_p95_ms", "tick_max_ms",
                      "frame_age_p50_ms", "frame_age_p95_ms", "backlog"):
            values = entry[field]
            entry[field] = {"min": min(values), "max": max(values),
                            "median": sorted(values)[len(values) // 2],
                            "samples": values} if values else None

    passing = [r for r in rows if r.gate_verdict() == "pass"]
    stale = [r for r in rows if r.staleness_verdict() == "stale"]
    disagreeing = [r for r in rows if r.gate_verdict() == "pass"
                   and r.staleness_verdict() == "stale"]
    return {
        "tool": "tools/input_ack_under_load.py",
        "what_this_measures": (
            "UI-thread tick lateness (what the input-acknowledgement gate measures) and "
            "per-stream inference backlog (what it does not), measured in the SAME window of "
            "the SAME application process."),
        "the_two_numbers_are_separate_claims": (
            "Frame age is not inference cost. On the NPU the measured p95 frame age is seconds "
            "while the inference itself costs about 5.9 ms; the gap is queue wait. They are "
            "never added, averaged, or presented as one figure."),
        "gate_thresholds_quoted_not_tuned": {
            "p95_ms": GATE_P95_MS,
            "worst_tick_ms": GATE_MAX_TICK_MS,
            "source": "tests/test_main.py::InputAcknowledgement",
            "note": ("These are the existing gate's own numbers, quoted from the gate and "
                     "deliberately not tuned. Nothing in this tool can change them; a threshold "
                     "adjusted until the answer came out the way somebody wanted would not be a "
                     "measurement."),
        },
        "tick_period_ms": TICK_MS,
        "window_s": seconds,
        "repeats_per_condition": repeats,
        "warmup_override_s": warmup_s,
        "machine_state": dict(machine) if machine is not None else None,
        "machine_clean_note": (
            "The census taken before the run. `competing_pids` must be empty for the timing "
            "figures to be a measurement of this application and nothing else; a run taken with "
            f"{CLEAN_OVERRIDE_ENV}=1 carries `overridden: true` and is deliberately dirty."),
        "calibrated_saturators": calibrated_saturators(),
        "saturator_note": (
            f"cpu_count()-2 = {calibrated_saturators()} here, the existing gate's calibration. "
            "It is a FLOOR, not a bound, and not comparable across machines: at cpu_count() of 8 "
            "the same probe measured p95 195.6 ms and max 1105.5 ms while still reporting "
            "green."),
        "backlog_warmup_note": (
            f"The three-stream conditions warm up for {BACKLOG_WARMUP_S:.0f} s because frame age "
            "is a consequence of accumulated queue depth: at 15 fps offered and 13.44-13.56 "
            "fps analysed a CPU stream gains 1.50 frames of backlog per second, so the "
            "173-187 frames M2 measured is 115-125 s of accumulation. A six-second window "
            "cannot show the booth's real state."),
        "the_table": [dict(row.as_dict(), gate_verdict=row.gate_verdict(),
                           staleness_verdict=row.staleness_verdict(),
                           verdicts_agree=row.agrees_with_itself()) for row in rows],
        "per_condition_ranges": list(ranges.values()),
        "summary": {
            "windows_measured": len(rows),
            "gate_pass": len(passing),
            "frame_age_stale": len(stale),
            "gate_passes_while_frame_age_stale": len(disagreeing),
            "disagreeing_conditions": sorted({r.condition for r in disagreeing}),
        },
        "conditions_failed": list(failures),
    }


def format_table(record: Mapping[str, Any]) -> str:
    """The combined table, as text. **Tick lateness and frame age, adjacent columns.**

    Written as a function rather than done by hand so the table in the report and the table in
    the JSON are the same code path, and a number cannot be transcribed wrongly into one of
    them.
    """
    header = (f"{'condition':<30}{'rep':>4}{'streams':>9}{'sat':>5}{'load_s':>8}"
              f"{'tick p50':>10}{'tick p95':>10}{'tick max':>10}"
              f"{'age p50':>10}{'age p95':>10}{'backlog':>9}{'gate':>7}{'age':>7}")
    lines = [header, "-" * len(header)]
    for row in record.get("the_table", ()):
        backlog = [v for v in (row.get("backlog_per_stream") or {}).values()
                   if v is not None]
        lines.append(
            f"{row['condition']:<30}{row['repeat']:>4}{row['streams']:>9}"
            f"{row['saturators']:>5}{row['load_applied_s']:>8.0f}"
            f"{_fmt(row['tick_p50_ms']):>10}{_fmt(row['tick_p95_ms']):>10}"
            f"{_fmt(row['tick_max_ms']):>10}"
            f"{_fmt(row['frame_age_p50_ms']):>10}{_fmt(row['frame_age_p95_ms']):>10}"
            f"{(max(backlog) if backlog else 0):>9}"
            f"{row.get('gate_verdict', '?'):>7}{row.get('staleness_verdict', '?'):>7}")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    """`None` prints as `n/a`, **never as `0.0`**. A zero lateness and an absent measurement are
    different facts and a table that conflates them is worse than no table."""
    if value is None:
        return "n/a"
    return f"{float(value):.1f}"


def write_record(record: Mapping[str, Any], path: pathlib.Path | None = None) -> pathlib.Path:
    target = pathlib.Path(path or DEFAULT_JSON)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.input_ack_under_load",
        description="Input-acknowledgement lateness and inference backlog, in one table.")
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="measured window per repeat, in seconds (default 30)")
    parser.add_argument("--repeats", type=int, default=2,
                        help="windows per condition (default 2). One is an anecdote.")
    parser.add_argument("--warmup", type=float, default=None,
                        help="override the condition's warm-up, in seconds. Recorded in the "
                             "output, so a short warm-up is visible rather than hidden.")
    parser.add_argument("--condition", dest="labels", action="append", default=None,
                        help="run only these conditions; repeatable. One of: "
                             + ", ".join(str(c["label"]) for c in CONDITIONS))
    parser.add_argument("--saturators", type=int, default=None,
                        help="override the saturator count for every condition, including 0")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help=f"where to write the JSON (default {DEFAULT_JSON})")
    parser.add_argument("--table-only", action="store_true",
                        help="print the table and do not write the JSON")
    parser.add_argument("--also-write-docs", action="store_true",
                        help=f"also write the reviewed copy to {DOCS_JSON}. Only after a human "
                             f"has read the conditions in the record: putting a figure into the "
                             f"evidence directory unreviewed is how a stale number becomes a "
                             f"cited one.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    kwargs: dict[str, Any] = {"seconds": args.seconds, "repeats": args.repeats,
                              "warmup_s": args.warmup, "labels": args.labels}
    if args.saturators is not None:
        # Applied per condition below, because `measure` takes it per call and the experiment
        # loops over conditions itself.
        rows: list[AckResult] = []
        failures: list[dict[str, str]] = []
        wanted = set(args.labels) if args.labels else None
        machine = assert_machine_clean()
        for condition in CONDITIONS:
            if wanted and condition["label"] not in wanted:
                continue
            try:
                rows.extend(measure(condition, seconds=args.seconds,
                                    repeats=args.repeats, saturators=args.saturators,
                                    warmup_s=args.warmup))
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                failures.append({"condition": str(condition["label"]),
                                 "error": str(exc)[:4000]})
        record = build_record(rows, failures, seconds=args.seconds, repeats=args.repeats,
                              warmup_s=args.warmup, machine=machine)
    else:
        record = run_experiment(**kwargs)

    print(format_table(record))
    summary = record["summary"]
    print()
    print(f"windows measured              : {summary['windows_measured']}")
    print(f"gate says pass                : {summary['gate_pass']}")
    print(f"frame age says stale          : {summary['frame_age_stale']}")
    print(f"gate passes while age is stale: {summary['gate_passes_while_frame_age_stale']}"
          f"  {summary['disagreeing_conditions'] or ''}")
    for failure in record["conditions_failed"]:
        print(f"CONDITION FAILED {failure['condition']}: {failure['error'][:300]}")
    if not args.table_only:
        target = write_record(record, args.out)
        print(f"wrote {target}")
        if args.also_write_docs and args.out is None:
            reviewed = write_record(record, DOCS_JSON)
            print(f"wrote the reviewed copy {reviewed}")
    # **Exit code from the account, not from the gate.** 0 when every requested condition ran.
    # A condition that could not run is a failure of this tool, not a pass.
    return 0 if not record["conditions_failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
