# Notices and attribution

This repository contains third-party models and media. Their licences are **not** the same as this project's
own source, and several carry obligations that apply as soon as the repository or a built package is
distributed.

**Status: partial.** See "Outstanding" at the end. This file records what is known; it does not claim that
every obligation has been discharged.

---

## Models

### `yolo11n-fp16` — AGPL-3.0

The detector weights and the IR derived from them are licensed **GNU Affero General Public License v3.0**.

**This obligation has not been evaluated.** AGPL-3.0 is a strong copyleft licence, and distributing a model —
and code that loads it — can carry source-disclosure requirements. Anyone redistributing this repository or a
package built from it must make that assessment themselves before doing so.

- Model: YOLO11n, exported to OpenVINO IR at FP16, static 640×640, batch 1, no embedded NMS.
- Recorded in `config/models.json` (`license: "AGPL-3.0"`).

---

## Media

### CC BY 4.0 — Intel IoT DevKit sample videos

Source: **https://github.com/intel-iot-devkit/sample-videos**
Licence: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
Licence text: https://github.com/intel-iot-devkit/sample-videos/blob/master/LICENSE

The following files in `assets/media/` are from that repository and are redistributed under CC BY 4.0:

| file | notes |
|---|---|
| `bottle-detection.mp4` | licence sidecar present |
| `car-detection.mp4` | licence sidecar present |
| `classroom.mp4` | licence sidecar present |
| `people-detection.mp4` | licence sidecar present |
| `person-bicycle-car-detection.mp4` | licence sidecar present |
| `worker-zone-detection.mp4` | licence sidecar present |
| `store-aisle-detection.mp4` | **no sidecar on disk**; same repository, same licence |
| `one-by-one-person-detection.mp4` | **no sidecar on disk**; same repository, same licence |

**Attribution.** The clips are used unmodified apart from decoding. Intel Corporation is credited as the
source of the sample media, via the repository above. Per-clip sidecars (`<clip>.LICENSE`) record the source
URL, retrieval date and file SHA-256.

**Limit of the licence.** CC BY 4.0 grants copyright permissions only. Section 2(b)(1) expressly does **not**
license publicity, privacy, personality or trademark rights. Several clips show identifiable people, which
raises a likeness question the licence does not answer. Clearance for public display remains an operator
decision.

### Pexels — `retail-checkout.mp4`

Source: Pexels, asset **32658425** (recorded provenance in `config/models.json`).
Licence: **Pexels License** — free for commercial use, attribution not required.

The provider's terms were **not re-read** during this project. The status is recorded as-is and is **not**
claimed as booth redistribution clearance.

---

## Application source

No licence has been chosen for this project's own source. Absent a `LICENSE` file, default copyright applies:
the code is not open source even though it is currently published in a public repository.

---

## Outstanding

1. **AGPL-3.0 assessment** for `yolo11n-fp16` has not been done.
2. **No `LICENSE`** for the project's own source.
3. **Two clips lack sidecars** on disk (`store-aisle-detection`, `one-by-one-person-detection`); their licence
   is established here by provenance, not by a file beside them.
4. **Likeness / privacy rights** for footage showing people have not been assessed.
5. **No signature.** Built packages are unsigned; `docs/install/SIGNING.md` records the exact step and the
   consequence.
6. `config/models.json` records two clips as *"Open Model Zoo sample media"*, which is **inconsistent** with
   the CC BY 4.0 sidecars. The accurate position is CC BY 4.0 via the repository above.
