# M1 headless run summaries (committed evidence)

These are the exact JSON summaries `python -m app.main --headless` wrote on
**2026-10-02**, committed so the milestone's per-device table can be checked against
something in the tree rather than against a scrollback buffer.

The previous report quoted a table whose artefacts lived only in a gitignored `artifacts/`
directory, so under the project's rule 1 - never publish a number that nothing on disk
backs - the table was unverifiable from `HEAD`.

## How to reproduce

```
.venv\Scripts\python.exe -m app.main --headless --device CPU --seconds 8 --json out.json
```

Conditions, so the figures can be compared against another run rather than admired:

| | |
|---|---|
| Machine | Intel Core Ultra 9 288V (Lunar Lake), Windows 11 Pro build 26200.26200 |
| Accelerator | integrated Arc 140V iGPU, sharing memory with the NPU |
| Model | the pinned `yolo11n-fp16` IR, `config/models.json` |
| Clip | the pinned `retail_checkout` clip, looped |
| Feed | 15 fps target (`ClipSource`), 8 s feed window, 2 s drain |
| Platform | `QT_QPA_PLATFORM=offscreen` |

## The confirmed placement, which is the point of these files

| File | Requested | **Confirmed** (`EXECUTION_DEVICES`) | State | `fell_back` |
|---|---|---|---|---|
| `cpu.json` | CPU | **`CPU`** | active | `false` |
| `gpu.json` | GPU | **`GPU.0`** | active | `false` |
| `npu.json` | NPU | **`NPU`** | active | `false` |

`confirmed` is read from the `InferenceResult`, never from the requested device.
`GPU.0` is printed verbatim rather than normalised to `GPU`, because the confirmed reading
is the evidence and a friendlier name would be a second claim.

## Reading `ai_fps` correctly

`ai_fps` divides by **`feed_elapsed_s`**, the window in which frames were actually fed -
not by `elapsed_s`, which additionally contains the drain and the teardown. The previous
report quoted `121 frames ... over an 8 s window` and `ai_fps 11.91` in the same row; those
were two different denominators presented as one window, which is finding I4.

For `cpu.json`: `frames_analysed 121 / feed_elapsed_s 8.006 = 15.11`, and
`elapsed_s 10.139 = feed 8.006 + drain 2.013`. `drain_elapsed_s` is its own field so the
difference is visible rather than absorbed.

## What `frame_age_ms` is, and is not

Ingress-to-result age, so it includes queue wait behind the frame transport, not just
inference. The settled per-frame inference latencies from M0 are unchanged by any of this:
**CPU 20.99-21.16 ms, GPU 3.30-3.47 ms, NPU 5.92-5.99 ms** - single frame, 640x640, batch
1, 30 inferences, same integrated Arc 140V sharing memory with the NPU. **The NPU is
slower than the GPU on this machine**, and the screen says so rather than implying
otherwise. The two figures are never merged.

## The event counts

Each run emitted **one** `zone_occupancy` event with a `dedup_key`. That is the retail
scenario's whole claim: **people present in the checkout zone**. It is not a product
transfer, not a count, and carries no dwell figure - the clip has none, and the scope note
says so on the screen.