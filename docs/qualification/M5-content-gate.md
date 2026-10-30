# M5 content gate

Date: 2026-10-30
Milestone: M5 — records, export and the recorded-demonstration mode
Verdict: **NOT MET.** The provisioned footage falls short of the specification on **clip
count, pack count and duration band**, and no clip is rights-cleared. This document
measures what is on disk and states the gap. It does not acquire, generate, trim, pad or
substitute footage.

> **What "measured" means here.** Every duration, resolution, frame count, fps, SHA-256
> and byte count below comes from `cv2.VideoCapture` and the file bytes, not from
> `assets/scenarios/*/manifest.json`. The re-runnable gate is `tools/content_gate.py`;
> `tests/test_content_gate.py` re-measures the files on every run and fails if the
> inventory and the disk disagree. **Licence and rights are not properties of a video
> file**, and are recorded as unverified rather than inferred — see §3.

---

## 1. The requirement

Spec §4 / design §5: **three 45–90 s clips per pack, four packs, twelve clips in total.**

| requirement | required |
|---|---|
| packs | **4** |
| clips per pack | **3** |
| clips, total | **12** |
| each clip's duration | **45–90 s** |
| rights | redistribution position recorded and cleared |

## 2. What is provisioned — measured from the files

```
$ .venv\Scripts\python.exe -m tools.content_gate
M5 content gate - scenario retail_sanity
required: 12 clips (4 packs x 3), each 45-90 s
provisioned: 3 clip(s), 1 pack(s)

clip                                         resolution      fps frames duration s  band  app
---------------------------------------------------------------------------------------------
assets/media/one-by-one-person-detection.mp4 768x432     10.000   1394  139.40000    NO   no
assets/media/retail-checkout.mp4             1920x1080    29.970   3387  113.01290    NO  yes
assets/media/store-aisle-detection.mp4       720x404     59.940   3921   65.41535   yes   no
```

| clip | resolution | fps | frames | measured duration | in 45–90 s band? | used by the application |
|---|---|---:|---:|---:|---|---|
| `assets/media/retail-checkout.mp4` | 1920×1080 | 29.97002997002997 | 3387 | **113.0129 s** | **no** — 23.0129 s over | **yes** — `app/engine/retail_config.py` |
| `assets/media/store-aisle-detection.mp4` | 720×404 | 59.94005994005994 | 3921 | **65.41535 s** | **yes** | no (alternate clip) |
| `assets/media/one-by-one-person-detection.mp4` | 768×432 | 10.0 | 1394 | **139.4 s** | **no** — 49.4 s over | no (alternate clip) |

`assets/scenarios/retail_sanity/` contains **no video files** — only `manifest.json`. The
gate inventories it anyway; there is nothing there to measure.

**Usage is read from the application's own source**, not from a manifest: a clip counts as
used when its path or file name appears under `app/`. Only `retail-checkout.mp4` does. The
other two are provisioned alternates that no code path opens; the scenario manifest lists
them as `role: alternate`, and this gate does not upgrade a listing to a use.

## 3. Licence and rights — what could and could not be measured

**A video file does not carry a redistribution grant.** `tools/content_gate.py` looks for a
licence file beside each clip (`LICENSE`, `LICENCE`, `LICENSE.txt`, …) and reads the file
bytes for provenance. **None exists in this tree.** So every clip is recorded as:

- `licence`: `unverified - no licence text is embedded in the file`
- `rights_status`: `not cleared - no licence file sits beside the clip`

This is deliberate, and it is the honest result. The M0 record carries the licence strings
that were transcribed when the clips were provisioned:

| clip | licence the M0 record carries | rights status now |
|---|---|---|
| `retail-checkout.mp4` | Pexels License — free for commercial use, no attribution required | **not cleared** (text transcribed, provider terms not re-read, no licence file committed) |
| `store-aisle-detection.mp4` | Open Model Zoo sample media (provenance, not a permission) | **not cleared** |
| `one-by-one-person-detection.mp4` | Open Model Zoo sample media (provenance, not a permission) | **not cleared** |

The gate does not re-read `docs/qualification/M0-content-gate.md` and call that clearance.
It records **0 of 3 clips cleared**. Closing this is an operator/legal dependency, not an
engineering step.

## 4. The gap

| requirement | required | provisioned | gap |
|---|---:|---:|---:|
| scenario packs | 4 | **1** | **3 packs missing** |
| clips, total | 12 | **3** | **9 clips missing** |
| clips inside the 45–90 s band | 12 of 12 | **1 of 3** | **2 provisioned clips are over-length** |
| a clip measured at 45–90 s in every pack | 4 packs | **1 pack** (`retail_sanity`, 1 clip) | **3 packs have no footage at all** |
| rights-cleared clips | 3 of 3 in the provisioned pack, 12 of 12 required | **0 of 3** | **no redistribution position is recorded for any clip** |

The provisioned pack's **count** is met (3 clips). Its **duration band** is not: only
`store-aisle-detection.mp4` (65.41535 s) lies inside 45–90 s; `retail-checkout.mp4` is
23.0129 s over and `one-by-one-person-detection.mp4` is 49.4 s over. Nothing was trimmed to
fit: a trimmed clip would be a different clip, and the durations above are what a decoder
plays.

`tools/content_gate.py` returns exit code **1** while the gate is unmet, so a script cannot
mistake "the numbers are printed" for "the gate passed".

## 5. What this gate does not claim

- It does **not** claim the footage is sufficient for the M5 acceptance set (≥20 positive
  events and ≥20 relevant negative opportunities per pack). That is a separate requirement
  and no clip has been annotated to that level here.
- It does **not** close the shelf-to-basket gap. The delivered retail event remains
  **people-in-zone presence, 2 per clip** — not a people count, not a dwell measurement.
- It does **not** claim rights. The licence strings above are transcribed provenance, not a
  clearance, and the inventory says `not cleared`.
- It does **not** acquire, generate or substitute footage. The only synthetic video in the
  suite is a one-frame file written into a temporary directory to prove that an added clip
  changes the inventory, and it is deleted with that directory.

## 6. Reproduction

```powershell
.venv\Scripts\python.exe -m tools.content_gate
.venv\Scripts\python.exe -m unittest tests.test_content_gate -v
```

`tests/test_content_gate.py` pins each measured field in a committed `DECLARED` table and
re-measures the files; it also builds a temporary directory, adds and removes a decodable
clip, and asserts the inventory changes both ways. A clip added to `assets/media/` without
the table being updated fails that test.
