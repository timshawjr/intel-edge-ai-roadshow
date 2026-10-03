# M2 Task 3 — route-change stress records

Five runs of `tools/route_stress.py`, all on 2026-10-03 on the demo machine. Written here
as well as to `logs/qualification/` because **`logs/` is git-ignored** — a record nobody
can re-read is a record nobody checked. The report that reads them is
`.superpowers/sdd/2026-10-01-m2-three-streams-and-device-toggles/task-3-report.md`.

Reproduce any of them with the command in its row. `logs/qualification/` is the tool's
default output path; `--out` chooses anywhere else.

| file | command | what it is |
|---|---|---|
| `route-stress-50-measured.json` | `-m tools.route_stress --transitions 50 --timeout-s 30` | **The headline run.** 50 alternating warm toggles, deadline widened *on purpose* so the tail is measured instead of censored. 50/50 acknowledged. |
| `route-stress-50-gate-2s.json` | `-m tools.route_stress --transitions 50` | The same 50 toggles at `await_transition`'s own 2.0 s default, unchanged. 23/50 acknowledged. **This is what the booth does.** |
| `route-stress-50-two-camera.json` | `… --timeout-s 30 --pattern both` | Row 1 against row 4: two cameras move per toggle, which the spec table never asks for. |
| `route-stress-fault-visible-fallback.json` | `… --transitions 6 --timeout-s 30 --fault-stream C` | One camera's worker really dies. The **visible** half of M0's unmet gate 9. |
| `allcpu-steady-120s.json` | `-m tools.route_stress --steady-seconds 120` | Spec §5 row 4 held for 120 s with no churn. **A different claim from the stress's frame ages** — see the file's own `claim` field. |

## Reading them

Three columns per toggle, and they do not agree:

- **`setup_s`** — seconds inside `begin_transition`, which is synchronous (terminate/join
  the old worker, retire its queues, drain, spawn the replacement).
- **`transition_elapsed_s`** — `TransitionOutcome.elapsed_s`. `_Transition.started` is
  stamped **after** `setup_s`, so this **excludes** it.
- **`elapsed_s`** — `setup_s + transition_elapsed_s`. **The toggle**, and the number the
  spec's 2 s gate is judged on.

The 2 s gate is **not met**: 50 of 50 toggles in the measured run exceeded it, and only
23 of 50 were acknowledged inside the real 2.0 s deadline. See §1 of the report.

`frame_age_p50_ms` / `frame_age_p95_ms` in the four stress files are measured **under churn**
and must never be quoted as the app's latency. The app's steady-state figure is
`allcpu-steady-120s.json`, and it was repeated three times: **p50 10.8–11.3 s, p95
11.6–12.3 s**, backlog 170–187 frames per stream.