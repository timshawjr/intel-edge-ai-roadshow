"""The M1 retail configuration, and the measurements it rests on.

This module exists because a previous version of this work put two rectangles in a
test file, described them as "the honest configuration", and shipped them. They were
wrong. Everything below is a re-measurement, and where an earlier number in this file
was wrong the correction is recorded rather than quietly overwritten.

**How the numbers here were produced:** the pinned detector
(`yolo11n-fp16`, conf 0.25, iou 0.45, letterbox pad 114 - the pin's own operating
point) over **all 3387 frames** of `retail_checkout` at **stride 1**. 9833
detections. Clip facts from `config/models.json`, confirmed with `cv2`
(3387 frames, 113.0129 s, 29.97002997 fps, 1920x1080).

## Correction 1 - the old rectangles

| measurement | definition | value |
|---|---|---|
| detections inside old `BASKET_ZONE = (0.00,0.70,0.22,0.99)` | `Zone.contains` - the **whole box** inside the ROI | **1286, of which 1285 are `apple`** |
| detections inside old `CHECKOUT_ZONE = (0.35,0.30,0.75,0.62)` | `Zone.contains` - the **whole box** inside the ROI | **2, both `suitcase`** |

The definition is stated because the paragraph below uses the other one, and mixing them
is how a count of 1286 and a count of 3281 both end up described as "detections inside
the old `BASKET_ZONE`". On `contains_point` (the centroid) the same rectangle holds
**3281** detections - `apple` 1430, `suitcase` 819, `oven` 608, `laptop` 343 - because
a box straddling the ROI edge is half of what is inside it.

The apple at those coordinates is the only hand-annotated object in the repository
(`[0, 878, 150, 1062]`, frame 0) and it is **stationary for the entire clip**. An
earlier version of this file also claimed "0 person detections in the old
`BASKET_ZONE`, on every track". Re-measured, that is **3** (tracks 282, 339, 340, one
observation each). Immaterial to the conclusion - a presence rule on that rectangle
emits nothing either way - but it was a number stated as a measurement and it was
wrong, so it is corrected here rather than left standing.

## Correction 2 - the counter surface's labels ARE unstable

An earlier version of this file claimed the surface's label "changed **0 times** in
2063 frames". That was false, and the false number could not have been anything else.
The measurement was taken over a **track's** `class_id`, and a track's `class_id` is
**constant by construction**: the tracker only ever matches a detection to a track of
the same class. Measured over all 3387 frames, **0 of the 639 track ids** ever changed
`class_id`. A count taken that way is not a measurement of flicker - it is arithmetic
that returns zero for any footage at all. Which is the point worth keeping: a number
that cannot come out other than zero is not evidence, and it was quoted as evidence
that the surface was stable.

Re-measured instead at the **detection** level, stride 1, over **1792 adjacent frame
pairs in which both frames carry a detection larger than 8% of the frame**:

| measurement | value |
|---|---|
| label **set** changed | **298 of 1792 pairs (16.6%)** |
| **dominant** label (largest box) changed | **228 of 1792 pairs (12.7%)** |
| dominant label (highest score) changed | 175 of 1792 pairs (9.8%) |
| A->B->A alternation within 3 consecutive frames | 68 |
| distinct transitions observed | **28**, most frequent `suitcase`->`oven` 28, `oven`->`suitcase` 27, `suitcase`->`tv` 19, `tv`->`suitcase` 17 |

The clearest instance is four consecutive frames of the same object, essentially the
same box, two different labels: frame 1514 `[oven 0.152]`, 1515 `[oven 0.152]`,
1516 `[suitcase 0.154]`, 1517 `[suitcase 0.154]`. Nothing about the counter moved.

The counter/scanner surface is labelled **17 different ways** across the clip -
`suitcase` 837, `oven` 816, `laptop` 347, `tv` 285, `microwave` 192, plus one-offs
(`sandwich`, `skateboard`, `knife`, `tie`, `book`, `tennis_racket`, `pizza`, `bird`,
`cup`, `bottle`, `banana`, `cell_phone`). 2534 of them exceed 8% of frame area.

**The consequence is the same whichever definition you use**, and it is the reason
these classes are excluded from `tracked_classes`: the tracker matches **within a
class**, so every one of those label flips mints a new track id for the same physical
surface. A count over them counts label noise. Over the clip the tracker minted
**639 track ids**, 114 of them `person` and the large remainder these surface labels.

A third figure is recorded for the same reason. An earlier draft cited the surface's
centroid range as `cx 0.15..0.47`. That range is wrong for the surface as a whole:
measured over **all** surface detections (>8% of frame area, non-person) the range is
**`cx 0.125..0.873`, `cy 0.202..0.822`**, and `0.125..0.873` is right to three
decimals. The narrower figure is close to the **`oven` label alone**
(`cx 0.148..0.474`), and the `oven`/`suitcase`/`laptop` subset together measures
**`cx 0.138..0.493`** - the laptop detections alone sit at `cx 0.138..0.169`. All
three are recorded because a reader who takes any one of them as "where the counter
surface is" is being told less than the footage says.

## Correction 3 - what the band actually covers

`COUNTER_BAND = (0.00, 0.20, 1.00, 0.60)` was derived from **two** person tracks, and
it captures **about two thirds of person detections, not all of them**. Measured with
the shipped `Tracker` (`iou_threshold=0.45`, `max_misses=2`, `min_hits=2`):

| measurement | value |
|---|---|
| distinct person track ids minted over the clip | **114** |
| person tracks with any confirmed, current observation | **80** |
| ...of those, ever contributing an in-band observation | **18** |
| ...of those, in band for >= 1.0 s | **3** |
| person detections with centroid **inside** the band | **3512 of 5393 (65.1%)** |
| person detections with centroid **outside** the band | **1881 of 5393 (34.9%)**, `cy` 0.034..0.864 |
| confirmed, current person observations inside / outside | **3484 (66.0%) / 1795 (34.0%)** |

The outside observations are not marginal. They include long, coherent fragments of a
**passer-by at the top of the frame**: track 267 contributes 283 observations at
`cy` 0.036..0.042, track 214 contributes 182 at `cy` 0.036..0.041. A person whose head
is cut off above the frame has their centroid near `cy = 0.04`, which is outside the
band by design - the band is where someone *stands at the counter*, not where a head
is clipped.

The two dominant tracks are **100% inside**, which is why the configuration fires:

| track | in-band observations | out | centroid y |
|---|---|---|---|
| 1 | 470 | 0 | 0.298..0.497 |
| 99 | 2920 | 0 | 0.237..0.500 |

So this is a band that captures **the sustained presence at the counter**, not "people".
A reader who takes `COUNTER_BAND` as "where people are" will be wrong about a third of
the person detections in this clip, and that is stated here rather than left to be
discovered.

## Why the zone test is on the centre, not containment

Measured: of **3484** confirmed person observations whose centroid is in the band,
**0 (0.0%)** are fully contained by it. A person's box is routinely clipped by the
frame border (`x1 = -0.001`, `x2 = 0.999`), so `Zone.contains` is **never** true for a
person in this footage and a presence rule built on it would report an empty shop. See
`Zone.contains_point`.

## What the two events are, and are not

The guard measures **two `zone_occupancy` events**, one per person track (ids 1 and 41
at the guard's stride-5 sampling). Stated precisely:

- They are **two sustained presences, one per person-track** - **not a people count**.
  Three distinct person tracks sustain >= 1.0 s in the band; two of them survive to
  produce an event.
- They carry **no duration**. `BusinessEvent` has `at_time`, not a dwell figure, so
  nothing on screen can read "how long". If a dwell duration is ever displayed it has
  to be measured and carried separately; it is not in this record.

## The reduction, stated

> **No product-transfer event is detectable on this footage. This pack is
> people-in-zone presence.**

That is not "items entering the checkout zone" - the data does not support it - and it
is not shelf-to-basket. Both remain **unmet**. Nothing here may be presented as
evidence for either.

## The guard

`tests/test_retail_footage_guard.py` runs the real pipeline - `preprocess_bgr` ->
`infer` -> `decode_detections` -> `Tracker` -> `RetailEventMachine` - over this clip
and **fails if this configuration produces zero events**. Its absence is what let the
old rectangles sit in the suite looking like validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from app.contracts.scenario_manifest import ScenarioManifest
from app.inference.pinned import REPO_ROOT

__all__ = [
    "COUNTER_BAND",
    "MIN_PRESENCE_SECONDS",
    "RETAIL_CHECKOUT_CLIP",
    "clip_fps",
    "clip_measurements",
    "retail_sanity_manifest",
]

#: The clip this configuration was measured against, and the id it is filed under in
#: `config/models.json`.
RETAIL_CHECKOUT_CLIP: Final[str] = "assets/media/retail-checkout.mp4"
RETAIL_CHECKOUT_ID: Final[str] = "retail_checkout"

#: The presence zone, and where it came from. Two person tracks spanning the whole clip
#: measured **100% inside** it (470 and 2920 observations, centroid y 0.237..0.500); the
#: band takes the full width, because neither dominant track's centroid leaves it, and
#: brackets that y range with margin on both sides. See "Correction 3" above for what it
#: does *not* cover.
COUNTER_BAND: Final[tuple[float, float, float, float]] = (0.00, 0.20, 1.00, 0.60)

#: Seconds of continuous presence before an event is emitted: "someone stood at the
#: counter for at least two seconds". Not tuned to make a count look good - the guard
#: measures 3 / 2 / 2 events at 1.0 s, 2.0 s and 5.0 s respectively, so the count does
#: not depend on this number.
MIN_PRESENCE_SECONDS: Final[float] = 2.0

#: Only `person`. The counter-surface classes are excluded; see "Correction 2" for the
#: measurement and for why a count over them is label noise.
TRACKED_CLASSES: Final[tuple[str, ...]] = ("person",)


def clip_measurements(clip_id: str = RETAIL_CHECKOUT_ID) -> dict[str, Any]:
    """The clip's **measured** facts, read from `config/models.json`.

    Read rather than hardcoded, deliberately. This module previously carried
    `CLIP_FPS = 3387 / 113.0129`, which is the same number twice: if the clip were
    re-provisioned at a different length or rate, the guard's `now` would silently
    become wrong and dwell would be measured against a fiction.

    `app/inference/pinned.py` is the app-layer reader for **model** facts (IR path,
    input shape, class count, labels) and there is no media reader there; that file is
    also an M0 file outside this dispatch's write scope. So this reads the same
    verified config the pin reads, and adds no second source of truth for anything.

    Raises rather than defaulting. A missing or incomplete entry is a provisioning
    fault, and a defaulted fps would make every dwell figure in the guard wrong while
    looking like a measurement.
    """
    models_json = REPO_ROOT / "config" / "models.json"
    try:
        payload = json.loads(models_json.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(
            f"{models_json} could not be read: {exc}. The clip's measured frame count "
            f"and duration live there, and guessing them would make every dwell figure "
            f"in the footage guard wrong."
        ) from exc
    entry = next((m for m in payload.get("media", []) if m.get("id") == clip_id), None)
    if entry is None:
        raise KeyError(
            f"config/models.json has no media entry for {clip_id!r}; ids are "
            f"{sorted(m.get('id') for m in payload.get('media', []))}"
        )
    for key in ("measured_frames", "measured_duration_s", "measured_fps"):
        if entry.get(key) is None:
            raise KeyError(
                f"config/models.json media {clip_id!r} has no {key}; the footage guard "
                f"needs it to turn frame indices into elapsed seconds"
            )
    return entry


def clip_fps(clip_id: str = RETAIL_CHECKOUT_ID) -> float:
    """Frames per second of the clip, as recorded in `config/models.json`."""
    return float(clip_measurements(clip_id)["measured_fps"])


def retail_sanity_manifest() -> ScenarioManifest:
    """The M1 retail pack: **people in zone**, not products transferring.

    Presence-only. It declares no `origin_slot`/`destination_slot`, because declaring
    a transfer route it does not use would be a claim about the footage that the
    footage does not support - the thing this configuration exists to stop doing.

    `scope_note` travels with the data so the reduction is exported with the pack
    rather than living in a presenter script that can drift.
    """
    return ScenarioManifest(
        id="retail_sanity",
        version=1,
        title="Retail - checkout counter (people in zone)",
        business_line="People present at the checkout counter",
        class_allowlist=TRACKED_CLASSES,
        zones=(),
        event_rules={
            "presence_zone": "counter_band",
            "min_presence_seconds": MIN_PRESENCE_SECONDS,
            "tracked_classes": TRACKED_CLASSES,
            "counter_band": COUNTER_BAND,
        },
        source_ids=(RETAIL_CHECKOUT_ID,),
        # The pin's own operating point - the same pair the M0 fixture was annotated
        # at, fixed before the model was run. Not tuned here.
        conf_threshold=0.25,
        iou_threshold=0.45,
        scope_note=(
            "REDUCED SCOPE. No product-transfer event is detectable on "
            "retail_checkout: the clip contains no shelf, and the only sustained "
            "movement is a person at the counter. Shelf-to-basket is NOT delivered. "
            "The counter surface is labelled 17 different ways and its label changes "
            "on 12.7% of adjacent frame pairs; it is excluded from tracked_classes. "
            "COUNTER_BAND captures 65% of person detections, not all of them. See "
            "app/engine/retail_config.py."
        ),
    )