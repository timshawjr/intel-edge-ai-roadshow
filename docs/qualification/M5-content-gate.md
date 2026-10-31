# M5 content gate

Date: 2026-10-31
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
> file**, and are recorded as provenance rather than inferred — see §3.

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

Nine clips are on disk in `assets/media/`: the three original retail clips, plus six clips
added on 2026-10-31 from `intel-iot-devkit/sample-videos` (§3). All nine were measured in
this session.

```
$ .venv\Scripts\python.exe -m tools.content_gate
M5 content gate - scenario retail_sanity
required: 12 clips (4 packs x 3), each 45-90 s
provisioned: 9 clip(s), 1 pack(s)

clip                                                 resolution      fps frames duration s  band  app
-----------------------------------------------------------------------------------------------------
assets/media/bottle-detection.mp4                    640x360     29.833   1189   39.85475    NO   no
assets/media/car-detection.mp4                       768x432     12.500    377   30.16000    NO   no
assets/media/classroom.mp4                           1920x1080    30.000    984   32.80000    NO   no
assets/media/one-by-one-person-detection.mp4         768x432     10.000   1394  139.40000    NO   no
assets/media/people-detection.mp4                    768x432     12.000    596   49.66667   yes   no
assets/media/person-bicycle-car-detection.mp4        768x432     12.000    647   53.91667   yes   no
assets/media/retail-checkout.mp4                     1920x1080    29.970   3387  113.01290    NO  yes
assets/media/store-aisle-detection.mp4               720x404     59.940   3921   65.41535   yes   no
assets/media/worker-zone-detection.mp4               1920x1080    59.940   4548   75.87580   yes   no
```

| clip | resolution | fps | frames | measured duration | in 45–90 s band? | used by the application |
|---|---|---:|---:|---:|---|---|
| `assets/media/bottle-detection.mp4` | 640×360 | 29.833333333333332 | 1189 | **39.85474860335196 s** | **no** — 5.14525 s short | no |
| `assets/media/car-detection.mp4` | 768×432 | 12.5 | 377 | **30.16 s** | **no** — 14.84 s short | no |
| `assets/media/classroom.mp4` | 1920×1080 | 30.0 | 984 | **32.8 s** | **no** — 12.2 s short | no |
| `assets/media/one-by-one-person-detection.mp4` | 768×432 | 10.0 | 1394 | **139.4 s** | **no** — 49.4 s over | no (alternate clip) |
| `assets/media/people-detection.mp4` | 768×432 | 12.0 | 596 | **49.666666666666664 s** | **yes** | no |
| `assets/media/person-bicycle-car-detection.mp4` | 768×432 | 12.0 | 647 | **53.916666666666664 s** | **yes** | no |
| `assets/media/retail-checkout.mp4` | 1920×1080 | 29.97002997002997 | 3387 | **113.0129 s** | **no** — 23.0129 s over | **yes** — `app/engine/retail_config.py` |
| `assets/media/store-aisle-detection.mp4` | 720×404 | 59.94005994005994 | 3921 | **65.41535 s** | **yes** | no (alternate clip) |
| `assets/media/worker-zone-detection.mp4` | 1920×1080 | 59.94005994005994 | 4548 | **75.8758 s** | **yes** | no |

SHA-256 of each of the six clips added on 2026-10-31, as provisioned (each matches the
download it was copied from, checked by hashing both):

| clip | bytes | SHA-256 |
|---|---:|---|
| `people-detection.mp4` | 5,482,579 | `18ffe8672d741e3e29c9d891d22c59d453720b086c25b35c88b393d55f92f693` |
| `person-bicycle-car-detection.mp4` | 6,031,199 | `452b11b7e0efbd019f1d9570d0c790e90416ad4ad29eec6003872d08443140ef` |
| `worker-zone-detection.mp4` | 12,596,893 | `b8b58b8100a81600bc42ee7ee082ced777b12a32a3619139ef289d27cbdcd284` |
| `bottle-detection.mp4` | 504,961 | `d52ba94aedf8a923c342fe9ea1d2bd85f712c4cc0f49a6de1bac43eebe3a48ff` |
| `car-detection.mp4` | 2,811,553 | `d31e0ebf194cc16e50eea784e7c3b0b1bb7976a4fb9b3fec26cb92a6704cca34` |
| `classroom.mp4` | 13,548,133 | `a69bd5e39ff0e74286d13bcbfdadddd307b85decd2d9853db7518e0028785e31` |

`assets/scenarios/retail_sanity/` contains **no video files** — only `manifest.json`. The
gate inventories it anyway; there is nothing there to measure. The six new clips are not
listed in that manifest and **no scenario manifest was created for them**: pack fit is a
separate judgement, and the gate counts packs from `assets/scenarios/*/manifest.json`.

**Usage is read from the application's own source**, not from a manifest: a clip counts as
used when its path or file name appears under `app/`. Only `retail-checkout.mp4` does. The
other eight are provisioned alternates that no code path opens; the scenario manifest lists
two of the originals as `role: alternate`, and this gate does not upgrade a listing to a
use.

## 3. The six clips added on 2026-10-31, and the licence recorded

Six clips were copied into `assets/media/` under their original filenames, each with a
per-clip licence sidecar (`<clip>.LICENSE`). Source repository:
**`https://github.com/intel-iot-devkit/sample-videos`** (branch `master`, archived by its
owner on 2024-09-30 and read-only). The repository's `LICENSE` was read, not assumed: it
is the **Creative Commons Attribution 4.0 International (CC BY 4.0)** licence text.

Each sidecar records: the licence name and version, the source repository URL, the exact
file path within the repository, the retrieval date (2026-10-31), the SHA-256 and byte
count of the file as provisioned, the measured geometry, the attribution string to display,
the personality-rights caveat below, and the clearance status.

### 3.1 Two limits that are recorded, not glossed over

- **CC BY 4.0 §2(b)(1) does not license publicity, privacy or personality rights.** The
  licence grants copyright permissions only. Footage showing identifiable people carries a
  rights question the licence does not answer. `people-detection.mp4`,
  `person-bicycle-car-detection.mp4`, `worker-zone-detection.mp4`, `bottle-detection.mp4`,
  `car-detection.mp4` and `classroom.mp4` depict people; whether they are identifiable, and
  whether their appearance may be used in a public demo, is an operator/legal judgement.
- **A licence file is not a clearance.** The gate's contract says licence status is not a
  property of the bytes. A sidecar makes the status **"licence file present; clearance not
  established"** — never `cleared`. The sidecar records provenance; it does not upgrade the
  rights verdict.

### 3.2 What the gate now records

| clip | licence recorded beside it | rights status |
|---|---|---|
| `bottle-detection.mp4` | CC BY 4.0 | **licence file present; clearance not established** |
| `car-detection.mp4` | CC BY 4.0 | **licence file present; clearance not established** |
| `classroom.mp4` | CC BY 4.0 | **licence file present; clearance not established** |
| `people-detection.mp4` | CC BY 4.0 | **licence file present; clearance not established** |
| `person-bicycle-car-detection.mp4` | CC BY 4.0 | **licence file present; clearance not established** |
| `worker-zone-detection.mp4` | CC BY 4.0 | **licence file present; clearance not established** |
| `retail-checkout.mp4` | none beside it (Pexels License transcribed in the M0 record) | **not cleared** |
| `store-aisle-detection.mp4` | none beside it (Open Model Zoo provenance in the M0 record) | **not cleared** |
| `one-by-one-person-detection.mp4` | none beside it (Open Model Zoo provenance in the M0 record) | **not cleared** |

**0 of 9 clips are cleared.** The three original clips are unchanged: the M0 licence
strings are provenance, not a clearance, and no sidecar was invented for them.

## 4. The deviation: three provisioned clips are under 45 s

The operator decided to provision all six downloaded clips, **including the three that
measure under 45 s**. That is a product decision the operator is entitled to make. It is
recorded here as a **deviation**, and it does **not** satisfy the duration requirement:

| clip | measured duration | band floor | shortfall |
|---|---:|---:|---:|
| `assets/media/bottle-detection.mp4` | 39.85474860335196 s | 45 s | **5.14525139664804 s** |
| `assets/media/car-detection.mp4` | 30.16 s | 45 s | **14.84 s** |
| `assets/media/classroom.mp4` | 32.8 s | 45 s | **12.2 s** |

The gate reports **4 of 9** clips inside the band, not 7 of 9, and the duration verdict is
**NOT MET**. Letting these three count as in-band would relax a spec requirement; the
numbers above are what a decoder plays, and the deviation is visible rather than absorbed.
The three are used as provisioned — not padded, looped or re-timed to reach 45 s.

`classroom.mp4` is a classroom, not a medical facility. Its name and sidecar describe what
the file is; no pack-fit claim is made for it.

## 5. The gap

| requirement | required | provisioned | gap |
|---|---:|---:|---:|
| scenario packs | 4 | **1** | **3 packs missing** |
| clips, total | 12 | **9** | **3 clips missing** |
| clips inside the 45–90 s band | 9 of 9 (12 of 12 against the spec) | **4 of 9** | **5 provisioned clips outside the band** — three under 45 s (§4), two over 90 s |
| a clip measured at 45–90 s in every pack | 4 packs | **1 pack** (`retail_sanity`, 1 clip) | **3 packs have no footage at all** |
| rights-cleared clips | 9 of 9 provisioned, 12 of 12 required | **0 of 9** | **no clearance is established for any clip**; six have a provenance sidecar |

The provisioned pack's **count** is met (3 clips). Its **duration band** is not: only
`store-aisle-detection.mp4` (65.41535 s) lies inside 45–90 s; `retail-checkout.mp4` is
23.0129 s over and `one-by-one-person-detection.mp4` is 49.4 s over. Nothing was trimmed to
fit: a trimmed clip would be a different clip, and the durations above are what a decoder
plays. The six added clips are not in any pack manifest, so the pack count stays at one.

`tools/content_gate.py` returns exit code **1** while the gate is unmet, so a script cannot
mistake "the numbers are printed" for "the gate passed".

## 6. What this gate does not claim

- It does **not** claim the footage is sufficient for the M5 acceptance set (≥20 positive
  events and ≥20 relevant negative opportunities per pack). That is a separate requirement
  and no clip has been annotated to that level here.
- It does **not** close the shelf-to-basket gap. The delivered retail event remains
  **people-in-zone presence, 2 per clip** — not a people count, not a dwell measurement.
- It does **not** claim rights. The CC BY 4.0 sidecars record provenance; they are not a
  clearance, and the inventory says `clearance not established`. Personality and privacy
  rights remain outside the licence, as §2(b)(1) of CC BY 4.0 states.
- It does **not** claim the sub-45 s clips meet the band. They are a recorded deviation.
- It does **not** assert that any added clip fits a scenario pack. No pack manifest was
  created or edited.
- It does **not** acquire, generate or substitute footage. The only synthetic video in the
  suite is a one-frame file written into a temporary directory to prove that an added clip
  changes the inventory, and it is deleted with that directory.

## 7. Reproduction

```powershell
.venv\Scripts\python.exe -m tools.content_gate
.venv\Scripts\python.exe -m unittest tests.test_content_gate -v
```

`tests/test_content_gate.py` pins each measured field in a committed `DECLARED` table and
re-measures the files; it also builds a temporary directory, adds and removes a decodable
clip, and asserts the inventory changes both ways; proves that a clip-scoped sidecar changes
only the clip it is named after (and a bare `LICENSE` in the folder changes none); and
asserts that the three sub-45 s clips stay outside the band. A clip added to `assets/media/`
without the table being updated fails that test.
