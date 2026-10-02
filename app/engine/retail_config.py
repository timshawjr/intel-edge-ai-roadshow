"""The M1 retail configuration, as **measured on the provisioned footage**.

This module exists because a previous version of this work put two rectangles in a
test file, described them as "the honest configuration", and shipped them. They were
wrong. This is the record of what the footage actually supports and the numbers
behind it.

## What the reviewer measured, reproduced here

`retail_checkout` re-decoded end to end with the pinned detector
(`yolo11n-fp16`, conf 0.25, iou 0.45, letterbox pad 114, `conf/iou` from the pin,
all frames of the clip):

| measurement | value |
|---|---|
| frames in the clip | **3387** (`cv2`, matches `config/models.json`) |
| duration | 113.0129 s at 29.97002997 fps |
| total detections | 9833 |
| distinct classes detected | 31 |
| detections inside the old `BASKET_ZONE = (0.00,0.70,0.22,0.99)` | **1286, of which 1285 are `apple`** |
| detections inside the old `CHECKOUT_ZONE = (0.35,0.30,0.75,0.62)` | **2, both `suitcase`** |
| person detections with their centroid in the old `BASKET_ZONE` | **0, on every track** |

The apple at those coordinates is the one hand-annotated object that exists in the
whole repository - `[0, 878, 150, 1062]` in frame 0 - and it is **stationary for the
entire clip**. So the old "basket" rectangle framed a stationary apple and the old
"checkout" rectangle framed two suitcase detections. A product-transfer event could
not fire, and the configuration's own test passed because the test used the same
rectangles.

## What the footage does support

One shopper, present for most of the clip. Tracked with the shipped `Tracker`
(`iou_threshold=0.45`, `max_misses=2`, `min_hits=2`) over **all** frames:

| track | frames observed | centroid x | centroid y |
|---|---|---|---|
| 1 | 0..471 | 0.24..0.78 | 0.31..0.50 |
| 99 | 465..3386 | 0.10..0.81 | 0.24..0.50 |

Both sit in a horizontal band with `y` between 0.24 and 0.50 across the full width.
That band is `COUNTER_BAND`, and it is where this configuration's zone comes from. It
is a **person-presence** zone, tested on the person's **centre** - see
`Zone.contains_point` for why full containment cannot be used here: a person's box is
routinely clipped by the frame border (`x1 = -0.001`, `x2 = 0.999`), so
`Zone.contains` is never true for them and a presence rule built on it would report an
empty shop.

## Classes that are NOT usable business events

The counter and scanner surface is labelled **17 different ways** across the clip -
`suitcase` (837), `oven` (816), `laptop` (347), `tv` (285), `microwave` (192), plus
one-offs (`sandwich`, `skateboard`, `knife`, `tie`, `book`, `tennis_racket`, `pizza`,
`bird`, `cup`, `bottle`, `banana`, `cell_phone`). 2534 of these are larger than 8% of
the frame. They are all the same physical surface.

Two honest notes about them:

- Across the clip the surface takes many different labels. Measured on **consecutive
  frames**, the set of labels attached to it changed **0 times** in 2063 frames - so
  the "flicker" claim needs care, and this module does not assert frame-to-frame
  flicker. What is certain, and what matters, is that a **class-partitioned** tracker
  mints a **separate track id per distinct label**: 639 track ids were minted over the
  clip, of which 114 are `person` and the large remainder are these surface labels.
  Any count taken over them is a count of label noise.
- They are therefore **excluded from `tracked_classes`** here. Not because they are
  wrong, but because a count built on them would not mean anything.

## The reduction, stated

> No product-transfer event is detectable on this footage. This pack is
> **people-in-zone presence**: how long a person is at the checkout counter.

That is not "items entering the checkout zone", and it is not shelf-to-basket. Both
remain **unmet**. Nothing in this module may be presented as evidence for either.

## The guard that keeps this honest

`tests/test_retail_footage_guard.py` runs the real pipeline - `preprocess_bgr` ->
`infer` -> `decode_detections` -> `Tracker` -> `RetailEventMachine` - over this clip
and **fails if this configuration produces zero events**. That test's absence is what
let the previous rectangles sit in the suite looking like validation.
"""

from __future__ import annotations

from typing import Final

from app.contracts.scenario_manifest import ScenarioManifest

__all__ = [
    "COUNTER_BAND",
    "MIN_PRESENCE_SECONDS",
    "RETAIL_CHECKOUT_CLIP",
    "retail_sanity_manifest",
]

#: The provisioned clip this configuration was measured against.
RETAIL_CHECKOUT_CLIP: Final[str] = "assets/media/retail-checkout.mp4"

#: The measured fps of that clip, from `config/models.json` and confirmed with `cv2`.
#: The guard divides frame indices by this to get seconds; it is a measurement, not a
#: default.
CLIP_FPS: Final[float] = 3387 / 113.0129

#: The zone, and where it came from. Two person tracks spanning the whole clip were
#: measured with their centroids at `y` 0.24..0.50 and `x` 0.10..0.81; the band takes
#: the full width (no person's centroid leaves it) and brackets the measured `y` range
#: with margin on both sides.
COUNTER_BAND: Final[tuple[float, float, float, float]] = (0.00, 0.20, 1.00, 0.60)

#: Seconds of continuous presence before an event is emitted. Chosen as a statement
#: about a booth visitor: "someone stood at the counter for at least two seconds".
#: It is **not** tuned to make a count look good - at 2.0 s the guard measures 2
#: events over the clip, and the same rule at 1.0 s or 5.0 s still fires.
MIN_PRESENCE_SECONDS: Final[float] = 2.0

#: Only `person`. The counter-surface classes are excluded; see the module note for the
#: measurement and for why a count over them is label noise.
TRACKED_CLASSES: Final[tuple[str, ...]] = ("person",)


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
        source_ids=("retail_checkout",),
        # The pin's own operating point - the same pair the M0 fixture was annotated
        # at, fixed before the model was run. Not tuned here.
        conf_threshold=0.25,
        iou_threshold=0.45,
        scope_note=(
            "REDUCED SCOPE. No product-transfer event is detectable on "
            "retail_checkout: the clip contains no shelf, and the only sustained "
            "movement is a person at the counter. Shelf-to-basket is NOT delivered. "
            "The counter surface is labelled 17 different ways by the detector and is "
            "excluded from tracked_classes. See app/engine/retail_config.py."
        ),
    )