# Working rules for the coding agent

These rules are binding for any agent doing work in this repository. They exist because the
product's central claim is that every number on screen was measured, and that claim is worthless if
the code is allowed to invent one.

Source of truth for product behaviour: `Intel-Edge-AI-Roadshow-Specification.md` v1.0. Design:
`docs/superpowers/specs/2026-10-01-intel-edge-ai-roadshow-design.md`.

## Binding rules

1. **Never claim a result you did not observe.** If you did not run it, write "not run" and paste
   real command output for every verification claim. A fabricated result is the worst possible
   failure mode for this project.
2. **Never put a number on screen that was not measured or sourced.** A missing counter means the
   UI says so. Hardware-spec TOPS and measured application FPS stay separate, and TOPS is never
   inferred from utilization.
3. **Static shapes only.** The NPU rejects dynamic shapes.
4. **Explicit `compile_model` targets for the primary routing demo.** Not `AUTO`, not `HETERO`, not
   `MULTI`. The presenter must always know where each job ran.
5. **`EXECUTION_DEVICES` is the only proof of placement.** Assert on it after compiling. A requested
   device preference is never presented as proof of execution.
6. **`config/models.json`, `config/platform_profiles.json` and `config/telemetry_map.json` are
   verified data, not suggestions.** Re-run `tools/verify_sources.py` and record the result if a
   source must change.
7. **No network at runtime.** Models, media, labels and fonts are provisioned once, up front, and the
   app is tested with networking disabled.
8. **Ask before adding a dependency.** The environment is pinned exactly. No ranges, no "latest".
9. **One milestone at a time** (spec §11, M0 → M6). Each milestone ends runnable. After each
   milestone run `tools/preflight.py`, paste the table into `docs/qualification/`, and commit
   `M<N>: <what works now>`.
10. **Keep layout tokens in `app/theme.py`.** No metric values or magic numbers in UI code.
11. **Telemetry sampling runs on a background thread** and reaches the UI only through Qt signals.
    Never touch widgets from a non-UI thread.
12. **Never crash in front of a customer.** Catch per-stream exceptions, mark that stream failed,
    keep the rest running, log a traceback, and restart a failed worker once automatically.
13. **The application runs as a standard user.** Never require elevation at runtime and never weaken
    a security setting (Defender exclusions, ASR rules, execution policy, AppLocker) to make
    something run. Anything needing elevation is an install step — say so and let a human run it.
14. **No simulated metrics in release mode.** A development banner is mandatory for any simulated
    value, and `app.main` refuses to start in release mode with a simulated provider registered.
15. **Unavailable is `null`, never zero.** Values older than 2 s are stale. An application-derived
    duty cycle may appear only in diagnostics, explicitly named as an estimate — never under a
    utilization label.

## Verification before declaring a milestone done

```powershell
.venv\Scripts\python.exe tools\verify_sources.py     # URLs verified by content, not status code
.venv\Scripts\python.exe tools\preflight.py          # every row PASS
.venv\Scripts\python.exe -m app.main --selftest      # runs with no network, exits 0
.venv\Scripts\python.exe -m unittest discover -s tests
```

Then run the actual demo for the duration the milestone requires and report the numbers you saw.

Hardware gates require this machine. A CPU run is never evidence of NPU readiness. Never declare
accelerator support from simulation, and never mark a gate passed on the strength of a mock screen.
