# Content request — the nine clips the retail demo is missing

**Purpose:** a sourcing brief precise enough to act on. Everything below was measured from the files on
disk by `tools/content_gate.py` (M5), not copied from a manifest.

**Current verdict: NOT MET.** 3 of 12 clips, 1 of 4 packs, 0 rights-cleared.

---

## What is required

| requirement | required | provisioned | missing |
|---|---:|---:|---:|
| scenario packs | 4 | 1 | **3** |
| clips per pack | 3 | 1 (retail) | **9** |
| each clip's duration | **45–90 s** | 1 of 3 in band | **2 over-length** |
| rights-cleared | 12 | **0** | **12** |

**9 new clips are needed**, 3 per pack across the three missing packs, **plus** rights clearance for all
twelve including the three already on disk.

---

## Hard technical constraints

These are not preferences; they are what the product can actually process.

- **Detector:** YOLO11n on the COCO 80-class label map. A scene must contain objects the detector really
  knows: **person, car, truck, bus, bicycle, motorcycle, bottle, cup, bowl, suitcase, handbag, chair, bench,
  dining table, laptop, cell phone**, etc.
  **Do not source footage whose story depends on a class outside the 80.** No product-brand recognition,
  no SKU labels, no faces-as-identity, no text reading.
- **Static shapes only** — the runtime rejects dynamic shapes. Clips are decoded and letterboxed to
  **640×640, batch 1**.
- **Resolution:** 720p or better, landscape. 1080p preferred for the hero camera.
- **Frame rate:** 25–30 fps preferred. **10 fps or below is visible as stutter at the booth.**
  (One current clip is 10 fps and one is 59.94 fps — neither is ideal.)
- **Duration: 45–90 seconds each.** This is a spec band, not advice. Two of the three current clips are
  over it (113.01 s and 139.40 s).
- **Codec:** H.264 in MP4, no DRM, no network fetch at runtime — files are provisioned once, up front.
- **No audio dependency** — audio may be present but nothing may rely on it.

---

## The three missing packs, with what each shot must contain

### Pack 2 — Smart city: intersection
- **Scene:** a fixed traffic camera looking at an intersection or crossing. Vehicles **and** pedestrians
  both visible and moving.
- **The AI must be able to do:** detector → tracker → **directional line crossing** →
  **stopped-object dwell**.
- **Primary outcome:** vehicles per minute and crossing count.
- **Limitation to display:** recorded-scene analytics; **no identity, no traffic-control action**.
- **Three distinct clips**, e.g. different times of day, different approach directions, one with a
  stopped vehicle for the dwell rule.

### Pack 3 — Medical: facility operations
- **Scene:** a staged corridor or waiting area with people sitting, standing and moving through.
- **The AI must be able to do:** person detection → tracking → **waiting-zone occupancy and dwell**.
- **Primary outcome:** occupancy and long-wait events.
- **Limitation to display:** operational demonstration only — **no diagnosis, no patient identification,
  no fall-detection claim**.
- **Sourcing note:** must be **staged or licensed stock**, not real patient footage. Do not source
  anything that could identify a real person or imply a clinical claim.

### Pack 4 — Government/defense: depot logistics
- **Scene:** a staged depot or loading area with **people and vehicles**.
- **The AI must be able to do:** detector → tracking → **loading-zone entry and dwell**.
- **Primary outcome:** vehicle arrivals and loading-zone dwell.
- **Limitation to display:** logistics demonstration only — **no threat classification, no targeting**.
- **Sourcing note:** staged facility, not an operational site.

### Pack 1 — Retail (already partly provisioned, still short)
- **Required scene:** a staged shopper moving **readily detectable items between defined zones**
  (shelf-exit → basket-entry). Items must be COCO classes — **bottle, cup, bowl, suitcase, handbag**.
- **What exists:** `retail-checkout.mp4` — 1920×1080, 29.97 fps, 113.01 s, used by the app.
  It shows a checkout counter and a pacing shopper; it **does not support a shelf-to-basket transfer**
  (the only reliably tracked product is stationary in the basket for the whole clip).
- **What is needed:** two further retail clips, 45–90 s, that actually show an item moving from a shelf
  zone into a basket zone, with the item class detectable and the person unobstructed enough to track.

---

## Rights — currently 0 of 12 cleared

**No clip on disk carries a licence file or a recorded redistribution position.** The gate records them
`unverified` / `not cleared`, deliberately, because licence is not a property of a video file.

For each accepted clip, record:

- source URL or supplier, and the date obtained
- **licence name and version** (e.g. CC BY 4.0, CC0, a paid stock licence number)
- whether **redistribution** as part of a demo package is permitted, and any attribution required
- a SHA-256 of the file as delivered
- the file's measured duration, resolution and fps

Prefer **CC0 / public-domain or explicitly redistributable** footage. If a clip is only "free to view",
it is not usable in a shipped package.

---

## What is already on disk (measured)

| file | resolution | fps | frames | duration | in 45–90 s? | used |
|---|---|---:|---:|---:|---|---|
| `assets/media/retail-checkout.mp4` | 1920×1080 | 29.970 | 3387 | **113.0129 s** | no (+23.0129 s) | **yes** |
| `assets/media/store-aisle-detection.mp4` | 720×404 | 59.940 | 3921 | **65.41535 s** | **yes** | no |
| `assets/media/one-by-one-person-detection.mp4` | 768×432 | 10.000 | 1394 | **139.4 s** | no (+49.4 s) | no |

`store-aisle-detection.mp4` is the only clip inside the duration band. It is currently unused and is a
plausible retail alternate.

---

## How to verify a candidate before accepting it

```powershell
# duration, fps and frame count, measured from the file
.venv\Scripts\python.exe -c "import cv2; c=cv2.VideoCapture(r'PATH'); print(c.get(cv2.CAP_PROP_FRAME_WIDTH), c.get(cv2.CAP_PROP_FRAME_HEIGHT), c.get(cv2.CAP_PROP_FPS), c.get(cv2.CAP_PROP_FRAME_COUNT), c.get(cv2.CAP_PROP_FRAME_COUNT)/c.get(cv2.CAP_PROP_FPS))"

# then re-run the gate, which re-inventories the disk
python -m tools.content_gate
```

`tools/content_gate.py` exits **1** while the gate is unmet, and its inventory is asserted against the
files on disk, so adding a clip without updating the inventory fails the suite.

---

## What NOT to do

- **Do not trim an existing clip to fit the band.** A trimmed 113 s clip is still the same scene and does
  not deliver the pack; it only makes the number pass. The gate would pass and the demo would not improve.
- **Do not relabel a generic object as a branded SKU.**
- **Do not source footage with identifiable people** for the medical pack, or from operational
  government/defense sites.
- **Do not record a clip and call it rights-cleared** — that is a rights claim, not a technical one.
- **Do not add clips outside the COCO classes** and expect events; the detector cannot see them.
