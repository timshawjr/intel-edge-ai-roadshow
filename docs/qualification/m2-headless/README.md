# M2 qualification records — route-change stress, steady-state frame age, and per-device latency

Runs of `tools/route_stress.py` and `tools/qualify_m0.py` on the demo machine. Written here as well
as to `logs/qualification/` because **`logs/` is git-ignored** — a record nobody can re-read is a
record nobody checked. The reports that read them are
[docs/qualification/M2-qualification.md](../M2-qualification.md) and the M2 section of
[`bench_report.md`](../../bench_report.md).

Reproduce any of them with the command in its row. `logs/qualification/` is the tools' default output
path; `--out` / `--json-out` chooses anywhere else.

## Route-change stress

| file | command | what it is |
|---|---|---|
| `route-stress-50-measured.json` | `python -m tools.route_stress --transitions 50 --timeout-s 30` | **Before M2's teardown fix.** 50 alternating warm toggles, deadline widened *on purpose* so the tail is measured instead of censored. 50/50 acknowledged. min 2.080 / p50 3.151 / p95 5.521 / max 5.544 s, **50/50 over 2 s**. Retained for comparison; superseded for the "after" column by the row below. |
| `route-stress-50-after-teardown-fix.json` | `python -m tools.route_stress --transitions 50 --timeout-s 30 --out <this file>` | **The headline run, after the fix.** Same 50 toggles, same four rows of the spec §5 table in the table's own order. 50/50 acknowledged, nothing censored. min 1.286 / p50 2.261 / p95 4.971 / max 5.185 s, **25/50 over 2 s**. `placements_confirmed` 50/50 and `placement_pids_matched` 50/50. |
| `route-stress-50-gate-2s.json` | `python -m tools.route_stress --transitions 50` | The same 50 toggles at `await_transition`'s own 2.0 s default, unchanged, **before** the fix. 23/50 acknowledged, 48/50 over 2 s. **This is what the booth does.** |
| `route-stress-50-gate-2s-after-teardown-fix.json` | `python -m tools.route_stress --transitions 50 --out <this file>` | The same, **after** the fix. 26/50 acknowledged, 37/50 over 2 s. **This run also leaks** — 182.4 MiB total growth with 175.3 MiB unexplained by retention, in the run with 24 unacknowledged transitions. Not characterised; see the report. |
| `route-stress-50-two-camera.json` | `python -m tools.route_stress --transitions 50 --timeout-s 30 --pattern both` | Row 1 against row 4: two cameras move per toggle, which the spec table never asks for. |
| `route-stress-fault-visible-fallback.json` | `python -m tools.route_stress --transitions 6 --timeout-s 30 --fault-stream C` | One camera's worker really dies. The **visible** half of M0's unmet gate 9: `confirmed_devices []`, `frames_analysed 0`, every refusal carrying a reason, `restart_count` exactly 1, `fell_back false`. |

## Steady state

| file | command | what it is |
|---|---|---|
| `allcpu-steady-120s.json` | `python -m tools.route_stress --steady-seconds 120` | **Spec §5 row 4**, three streams on the CPU, 120 s, no route change and no churn. p50 11 319 ms / p95 12 286 ms, backlog 173–187 per stream. **A different claim from the stress's frame ages** — see the file's own `claim` field. |
| `allcpu-steady-120s-hold2.json` | `python -m tools.route_stress --steady-seconds 120 --out <this file>` | The same hold, repeated. p50 14 309 / p95 15 562 ms, backlog 228–235. |
| `allcpu-steady-120s-hold3.json` | `python -m tools.route_stress --steady-seconds 120 --out <this file>` | The same hold, repeated again. p50 14 697 / p95 16 019 ms, backlog 226–242. |

**The headline is the range: p50 10.7–14.7 s, p95 12.3–16.1 s, backlog 173–242 frames per stream**, with
`errors: []`, **drops 0** and all three streams `active` on every row. Two earlier holds quoted in the
gitignored milestone log (10 767 and 11 085 ms) are *lower* than all three committed holds; the
picture gets older across repeated holds on this machine and **the cause is not established**.

## Per-device latency

| file | command | what it is |
|---|---|---|
| `single-frame-latency-run1.json` … `-run4.json` | `python tools/qualify_m0.py --requests 30 --json-out <this file>` | Four full runs, each in its own process. Single frame, letterboxed 640×640, batch 1, static, warm-up outside the timed span. `OVERALL: PASS - 4 of 4 rows succeeded on the device they requested` on every run. CPU p50 22.61 / 22.81 / 22.40 / 22.72 ms; GPU 3.45 / 3.54 / 3.42 / 3.40; NPU 5.25 / 5.21 / 5.25 / 5.74. |

**The NPU is slower than the GPU on this machine**, under the conditions named above, on an integrated
Arc 140V sharing system memory with the NPU. **This is not a throughput claim**; paired trials are M4.
The milestone working log quotes CPU 20.99–21.16 ms and NPU 5.92–5.99 ms — better than anything measured
in the M2 session, recorded only in a gitignored file, taken earlier the same day on a measurably
faster machine state. **A per-device latency figure is a property of a session, not of a machine.**

## Reading them

Three columns per toggle, and they do not agree:

- **`setup_s`** — seconds inside `begin_transition`, which is synchronous (terminate/join the old
  worker, retire its queues, drain, spawn the replacement).
- **`transition_elapsed_s`** — `TransitionOutcome.elapsed_s`. `_Transition.started` is stamped **after**
  `setup_s`, so this **excludes** it.
- **`elapsed_s`** — `setup_s + transition_elapsed_s`. **The toggle**, and the number the spec's 2 s gate
  is judged on.

**The 2 s gate is NOT met**: 25 of 50 toggles in the measured run exceeded it, and 37 of 50 exceeded it
at the real 2.0 s default. See §M2.3 of `bench_report.md`.

`frame_age_p50_ms` / `frame_age_p95_ms` in the **stress** files are measured **under churn** and must
never be quoted as the app's latency. The app's steady-state figures are the three
`allcpu-steady-120s*` files.

> `tools/route_stress.py` **must** be run as `python -m tools.route_stress` or under `unittest`. As a
> plain script every worker child fails to spawn, because `spawn` re-imports `__main__`.