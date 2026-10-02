# M0 content gate

Date: 2026-10-01
Milestone: M0 — hardware and model path qualification
Verdict: **partially met. Rights are recorded but not cleared. The requested
shelf-to-basket scenario does not exist in the material provisioned at M0, and is not
claimed.** A reduced-scope fallback is committed to instead, and the three remaining packs
have no footage at all.

> Scope of this document: the spec's M0 content gate is a question about **assets and
> rights**, not about hardware. It is answered here from `config/models.json`,
> `assets/scenarios/retail_sanity/manifest.json` and from looking at the decoded footage.
> Nothing in it is inferred.

---

## 1. Retail clip rights

Three clips are provisioned, all verified by content signature and SHA-256 against
`config/models.json` (`tools/verify_sources.py`: `6 PASS, 0 FAIL` on 2026-10-01).

| id | source | owner / provenance | licence as recorded | SHA-256 | `distribution_status` |
|---|---|---|---|---|---|
| `retail_checkout` | Pexels 32658425 | Pexels; the same clip Intel's own retail loss-prevention reference downloads | Pexels License — free for commercial use, no attribution required | `e107db69…f5906` | **not upgraded.** "licence text carried over from the source manifest as recorded; NOT re-read at M0 and not claimed as booth redistribution clearance" |
| `store_aisle_alt` | `storage.openvinotoolkit.org/data/test_data/videos/store-aisle-detection.mp4` | Open Model Zoo sample media | Open Model Zoo sample media | `3526fee3…6ed6e04` | "recorded as-is; not claimed as booth redistribution clearance" |
| `queue_alt` | `storage.openvinotoolkit.org/data/test_data/videos/one-by-one-person-detection.mp4` | Open Model Zoo sample media | Open Model Zoo sample media | `a5964aa2…9ac07c` | "recorded as-is; not claimed as booth redistribution clearance" |

Three things this gate does **not** establish, stated plainly:

1. **No clip is upgraded to "cleared".** All three licence strings were carried over from
   the recorded source manifest at provisioning. The provider's own terms were **not
   re-read at M0**. The Pexels licence is very likely sufficient for a commercial booth,
   but "very likely" is not a clearance and is not written down as one.
2. **The two Open Model Zoo clips have no licence text at all** in the record — the
   `license_source` field says so in as many words. "Open Model Zoo sample media" is a
   provenance note, not a permission. The design already flags this: a tutorial download is
   not automatically redistribution-cleared.
3. **No owner contact, licence file or screenshot of the terms is committed.** The
   `distribution_status` strings are the only rights record that exists.

**What would close this:** re-read and record the Pexels terms for 32658425 at their
current date, attach the licence text, and obtain a written position on the two Open Model
Zoo clips. That is operator work with a legal question in it, not a build step, and it is
**not** M0's to decide.

## 2. Detector weight rights

| id | licence | provenance | distribution status |
|---|---|---|---|
| `yolo11n-fp16` | **AGPL-3.0** | `OpenVINO/YOLO11n-fp16-ov` on Hugging Face; an Ultralytics YOLO11 conversion to OpenVINO IR with FP16 weights, model creator Ultralytics | recorded, **not resolved** |

`config/models.json` records the licence source precisely: the model card's "License:
agpl-3.0" and "Legal information" field, read 2026-10-01, stating the original model is
distributed under the GNU Affero General Public License v3.0.

**The source manifest recorded no licence at all** for this artifact; the AGPL-3.0 came
from reading the model card, and it is the only licence the weights carry. AGPL-3.0 is
strong copyleft. Redistribution of a booth package that incorporates these weights is a
distribution question this milestone **records and does not answer**. It is flagged as an
unmet M6 gate below, and it is an operator/legal decision, not an engineering one.

## 3. Is there a clear shelf-to-basket clip? — answered by looking

**No. Within the material actually provisioned and content-verified in this repository,
there is no shelf-to-basket clip with a visible transfer and no prolonged occlusion.**
There is no shelf-to-basket *event* in any of the three clips.

This was answered by decoding frames and looking at them, not by reading filenames. A
labelled 4x3 contact sheet was built for each clip (12 evenly spaced frames each, native
tile resolution, every tile stamped with its frame index and timestamp), plus a second
denser pass over the regions where an event would be visible — **72 frames in all**.

**The harness is committed and re-runnable: `tools/build_contact_sheets.py`.** It writes
the five sheets to `logs/contact_sheets/` and prints the exact frame indices it decoded.
Its `CLIP_PASSES` table is the frame list transcribed from the lists below, and
`tests/test_contact_sheets.py` parses those lists back out of this document and fails if
the two ever disagree — so the frames cited here are the frames the tool decodes, and the
sheets are the pictures these conclusions were read from.

An earlier revision of this document said "Method and the harness are reproduced in
`bench_report.md`". **No harness was in `bench_report.md`**, and §Reproduction there says
the scripts were in a gitignored task report — so the one place in M0 a reader could not
check anything was the one place this document promised they could. The tool now exists
in this repository; `bench_report.md` names it rather than claiming to reproduce it.

```
$ .venv\Scripts\python.exe tools\build_contact_sheets.py
WROTE logs\contact_sheets\retail_checkout_contact.jpg  12 frames [0, 308, 616, 923, 1231, 1539, 1847, 2155, 2463, 2770, 3078, 3386]  2560x1080
WROTE logs\contact_sheets\retail_checkout_dense.jpg  12 frames [1400, 1500, 1560, 1620, 1680, 1740, 1800, 2000, 2200, 2400, 2600, 2900]  2560x1080
WROTE logs\contact_sheets\store_aisle_alt_contact.jpg  12 frames [0, 356, 713, 1069, 1425, 1782, 2138, 2495, 2851, 3207, 3564, 3920]  2560x1077
WROTE logs\contact_sheets\store_aisle_alt_dense.jpg  12 frames [900, 1000, 1069, 1140, 1200, 2400, 2495, 2560, 3100, 3207, 3270, 3400]  2560x1077
WROTE logs\contact_sheets\queue_alt_contact.jpg  12 frames [0, 127, 253, 380, 507, 633, 760, 886, 1013, 1140, 1266, 1393]  2560x1080

5 sheet(s) written, 0 warning(s). The frame lists above are the ones
docs/qualification/M0-content-gate.md cites; this tool invents no frames of its own.
```

**What a contact sheet can and cannot establish.** It shows what appears in the frames
looked at — a shelf, a basket, a room. It does **not** show an event sequence or an
occlusion, which is why the dense passes exist and why every conclusion below is phrased
about what appears in those frames and never about what the clip contains. Nothing in the
tool runs a detector: a detection drawn on a twelve-frame grid would be a claim it has no
standing to make.

**One measured detail the tool depends on, recorded because the first version got it
wrong.** `CAP_PROP_POS_FRAMES` read *after* `read()` is the position of the **next** frame,
not the one delivered — on `retail_checkout.mp4`, set 1400 / read / get returns 1401. That
version stamped every tile one frame high and reported 60 spurious seek warnings. The tool
reads the position between the set and the read, and reports any genuine seek discrepancy
as a warning rather than quietly stamping the wrong number.

### `retail_checkout` — 1920x1080, 29.97 fps, 3387 frames, 113.0129 s

Contact sheet at frames `[0, 308, 616, 923, 1231, 1539, 1847, 2155, 2463, 2770, 3078, 3386]`,
dense pass at frames `[1400, 1500, 1560, 1620, 1680, 1740, 1800, 2000, 2200, 2400, 2600, 2900]`.

What is actually in the picture, consistently across all 24 frames looked at:

- A **fixed high-angle camera on a self-checkout / till station**. The frame contains a flat
  counter surface, a recessed scanner plate in the centre, and a black plastic shopping
  basket at the left.
- The basket is **already full at frame 0** (apples, a gold-lidded jar, a dark bottle,
  bananas, a blue-and-yellow package). The clip never shows how those items got there.
- A person in a dark NASA t-shirt stands behind the counter and repeatedly **lifts an item
  out of the basket and places it on the scanner plate** — visible in the dense pass at
  frames 1500–1560 (an apple onto the plate), 1740–1800 (an apple and a jar moved across),
  2000 (an apple lifted clear), 2200 (an apple carried away), 2400 (a bottle), 2600 (a
  bottle and a dark item).
- A passerby's legs cross the top of the frame; at frame 2600 a second shopper with a
  basket passes at the left edge.
- **There is no shelf, no shelving unit and no shelf stock anywhere in this clip.** The
  camera never sees one.

So the clip shows *basket → counter*, not *shelf → basket*. It is exactly the reduced-scope
scenario and nothing more.

### `store_aisle_alt` — 720x404, 59.94 fps, 3921 frames, 65.41535 s

Contact sheet at frames `[0, 356, 713, 1069, 1425, 1782, 2138, 2495, 2851, 3207, 3564, 3920]`,
dense pass at frames `[900, 1000, 1069, 1140, 1200, 2400, 2495, 2560, 3100, 3207, 3270, 3400]`.

This is the only clip with **shelving in it**: a high-mounted wide shot down a housewares
aisle, shelving on both sides and a central island stacked with white tableware. Shoppers
walk the aisle and reach toward the right-hand shelves — clearly visible at frames 1069,
1140, 2495, 2560, 3207 and 3270, where a shopper in a teal top has an arm out to a shelf.

But there is **no basket, cart or trolley visible in any of the 24 frames looked at**, and
no item is seen leaving a shelf. At the clip's native 720x404, the shoppers are roughly
100–200 px tall and the shelf stock is a field of indistinguishable white blobs a few pixels
across. There is no visible transfer, and nothing in this clip could be annotated as one.
It is usable as a **people-detection aisle clip**, which is what
`assets/scenarios/retail_sanity/manifest.json` already calls it ("an aisle of people
walking; few packaged goods to name").

### `queue_alt` — 768x432, 10.0 fps, 1394 frames, 139.4 s

Contact sheet at frames `[0, 127, 253, 380, 507, 633, 760, 886, 1013, 1140, 1266, 1393]`.

An **office or lobby interior**, not a retail space: a white wall, a wooden floor, a small
round standing table with a sheet of paper on it, and one person at a time walking up to
the table and reading. No shelves, no basket, no counter, no goods. Not retail material at
all; it is a one-person-at-a-time person-detection clip.

### Verdict

| question | answer |
|---|---|
| A shelf is visible in the material? | Yes — `store_aisle_alt` only. |
| A basket or trolley is visible in the material? | Yes — `retail_checkout` only, and it is **already full** at frame 0. |
| Both in the same shot? | **No.** The two never co-occur. |
| A visible shelf-to-basket transfer? | **No.** No item is seen leaving a shelf in any clip. |
| A visible transfer of any kind, unoccluded? | Yes, in `retail_checkout`: basket → scanner plate, several times, per the frames above. |

## 4. The committed reduced-scope fallback

The requested scenario is therefore **left incomplete rather than claimed**. M0 commits to
the documented reduced scope, as an **explicitly labelled intermediate milestone**:

> ~~**"Items entering the checkout zone"** — for `retail_sanity`, the business event is an
> item leaving the shopper's basket and entering the checkout / scanner zone.~~
>
> **SUPERSEDED at M1, 2026-10-02, by measurement.** M0 committed to this reduced scope on the
> reasonable reading that the shopper's basket *is* the origin zone. Running the pinned detector
> over the whole clip showed that reading was wrong: **no product-class zone transition fires on
> `retail_checkout` at the pinned operating point.** The only reliably tracked product is the apple
> already sitting in the basket at `(0.00, 0.85, 0.09, 0.99)`, which is stationary for the entire
> clip (max corner deviation 0.0006 over 483 frames). The full-clip run under the M0 zone
> rectangles produced **0 events**.
>
> The delivered event is therefore **people-in-zone presence**, not an item transition:
>
> > **"No product-transfer event is detectable on this footage; this pack is people-in-zone
> > presence."** Shelf-to-basket is **not delivered**. Not delivered either is the item-transition
> > reading of "checkout zone", because the footage does not support it. The person is the only
> > object that moves, and it does not traverse — 17.441 frame-widths of cumulative travel against
> > 0.015 of net displacement, pacing behind the counter — so a traversal event fires once per
> > 113-second clip and would be a number that looks like a business event without being one.
>
> A guard test now runs the real pipeline over the real clip and **fails if a shipped
> configuration emits zero events**, so this class of misconfiguration cannot reappear. The
> counter surface is additionally excluded from event logic: its label is unstable (the dominant
> label changes in 228 of 1792 adjacent frame pairs, 12.7%), so a count over it would measure the
> model's mood rather than the scene.

Conditions attached, so the label cannot quietly become a claim:

- The pack's title in the UI carries the scope, e.g. "Retail — checkout zone (reduced
  scope: shelf-to-basket not filmed)".
- The presenter guide states the same, in the same words.
- The M5 acceptance set is annotated on the events that **actually occur** in
  `retail_checkout`. The `store_aisle_alt` and `queue_alt` clips carry no basket event at
  all and must not be scored as though they did.
- The shelf-to-basket scenario stays on the open-blockers list for M5 and is **not** closed
  by this document.

## 5. Footage requirements still outstanding

### 5.1 Clip count and duration against the spec

Spec §4 / design §5: **three clips of 45–90 s per pack, four packs, twelve clips total.**

Measured with `cv2.VideoCapture` on the copied files (re-measured in the Task 9 session,
2026-10-01, and matching `config/models.json` exactly):

| id | resolution | fps | frames | measured duration | inside the 45–90 s band? |
|---|---|---|---|---|---|
| `retail_checkout` | 1920x1080 | 29.97002997002997 | 3387 | **113.0129 s** | no — 23.0 s over |
| `store_aisle_alt` | 720x404 | 59.94005994005994 | 3921 | **65.41535 s** | **yes** |
| `queue_alt` | 768x432 | 10.0 | 1394 | **139.4 s** | no — 49.4 s over |

| pack | clips provisioned | clips required | clips in the 45–90 s band | status |
|---|---|---|---|---|
| `retail_sanity` | 3 | 3 | **1 of 3** | count met, **duration band not met** |
| pack 2 | **0** | 3 | 0 | **unmet** |
| pack 3 | **0** | 3 | 0 | **unmet** |
| pack 4 | **0** | 3 | 0 | **unmet** |

**This is an unmet M5 content gate.** No duration is rounded, padded, trimmed or re-encoded
to fit the band. `queue_alt` at 10 fps and 139.4 s and `retail_checkout` at 113.0129 s are
recorded at their measured lengths.

### 5.2 What still has to be sourced

1. **Nine clips minimum** — three per pack for the three packs that have none. Fifteen if
   the band is enforced across all four packs, which is what §5.1 says.
2. **A genuine shelf-to-basket clip** for `retail_sanity`, or a decision to keep the
   reduced scope permanently. This is the only item on the list that changes what the booth
   *shows* rather than how much of it there is.
3. **A written redistribution position on the two Open Model Zoo clips**, and a re-read of
   the Pexels terms for `retail_checkout`.
4. **A written position on AGPL-3.0** for the detector weights, before the M6 package is
   assembled.
5. Clips long enough to carry **≥20 positive events and ≥20 relevant negative
   opportunities** for the M5 acceptance set. Whether 45–90 s is enough for that on any
   pack is **not established** — M0 did not measure it, because the events do not exist yet
   in annotated form.

### 5.3 Where this leaves the gate

| gate item | status |
|---|---|
| Retail clip rights recorded | **met** (recorded, not cleared) |
| Detector weight rights recorded | **met** (AGPL-3.0 recorded, not resolved) |
| Shelf-to-basket clip available | **NOT met** — reduced-scope fallback committed, request left incomplete |
| Three 45–90 s clips per pack | **NOT met** — 1 of 12 clips in band; 9 clips absent |
| Redistribution clearance | **NOT established** for any clip |
