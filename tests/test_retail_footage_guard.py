"""The general guard: **a shipped configuration must produce events on the real clip.**

This test file exists because of a review finding. The previous work put two
rectangles in `tests/test_events.py`, called them "the honest configuration", and
drove a transfer through them in a unit test. The rectangles were wrong. Measured over
the **whole** clip with the pinned detector they hold 1286 detections of which 1285 are
a stationary `apple`, and 2 `suitcase` detections respectively - **zero** product
transfers. The unit test passed because it used the same wrong rectangles, so nothing
in the suite could see the problem. A configuration that has never been run against
the footage it describes is not a validated configuration.

So: run the pipeline. `preprocess_bgr` -> `infer` -> `decode_detections` ->
`Tracker` -> `RetailEventMachine`, over the provisioned clip, and require at least one
event per loop. Zero events is a failure, not a pass, and not a skip.

## Sampling, and why it is sound

The guard samples every `SAMPLE_STRIDE`-th frame of the 3387-frame clip. It does that
for cost, and the number is in the test name so nobody reads the result as a
full-rate measurement:

- **Full clip on CPU is ~115 s** (measured: 3387 inferences). At stride 5 it is
  **~24 s** (measured: 678 inferences). The suite was 12.5 s before this file.
- Sampling is sound for a *dwell* measurement because dwell is elapsed wall-clock
  time between real observations (`now - entered_at`), not a frame count. Sampling
  reduces the resolution of the measurement, not the elapsed time it measures.
- It is **not** a claim about production frame rates, and does not reproduce the
  product's 15 fps analysis path. It answers one question: does this configuration
  detect anything at all in this footage?

Stride 5 rather than 1 because at stride 1 the tracker's `max_misses=2` tolerance is
two frames, and at stride 5 it is 25 frames. Coarser strides than about 6 start
inflating the event count through track fragmentation rather than genuine presence,
which would be the guard passing for the wrong reason. At stride 5 the two events
correspond to two real, sustained person presences (track ids 1 and 41, covering real
frames 0..485 and 485..3385).

## Skipping

If `openvino`, the CPU plugin, the IR or the clip is absent, this **skips**, following
M0's convention in `tests/test_sanity_fixture.py`. A skip is not a pass, and
`test_the_guard_covers_every_shipped_configuration` - which always runs - ensures the
guard cannot be quietly bypassed by a configuration nobody pointed it at.
"""

import unittest
from pathlib import Path

from app.engine.retail_config import (
    CLIP_FPS,
    RETAIL_CHECKOUT_CLIP,
    retail_sanity_manifest,
)

#: Every 5th frame of 3387 -> 678 inferences, ~24 s measured on CPU. Named in the test
#: below so the sample rate is never a silent detail.
SAMPLE_STRIDE = 5

#: The shipped configurations this guard is responsible for. A configuration added to
#: `app/engine/` and not listed here is a configuration nothing checks against the
#: footage; `test_the_guard_covers_every_shipped_configuration` fails if the list and
#: the guard disagree.
SHIPPED_CONFIGURATIONS = (retail_sanity_manifest,)


def _skip_reason() -> str | None:
    """Why this machine cannot run the guard, or `None` if it can."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        return "opencv is not importable"
    try:
        import openvino as ov
    except ImportError:
        return "openvino is not importable"
    if "CPU" not in ov.Core().available_devices:
        return "no CPU plugin on this machine"
    from app.inference.pinned import verified_ir_path
    try:
        if not verified_ir_path().exists():
            return "the pinned IR is not provisioned"
    except Exception as exc:  # noqa: BLE001 - a skip reason, never a pass
        return f"the pinned IR is unusable: {exc}"
    if not Path(RETAIL_CHECKOUT_CLIP).exists():
        return f"{RETAIL_CHECKOUT_CLIP} is not provisioned"
    return None


def run_clip(manifest) -> dict:
    """The whole pipeline over the provisioned clip. Returns the observed result.

    Deliberately imports the M0 inference path rather than re-implementing it, so the
    guard exercises the same decode the product does. Returns the machine's own
    numbers; it asserts nothing, so a test can state the expectation.
    """
    import cv2
    import numpy as np
    import openvino as ov

    from app.contracts.frame import FramePacket
    from app.engine.events import RetailEventMachine
    from app.engine.tracker import Tracker
    from app.inference.decode import decode_detections, describe_outputs, port_key
    from app.inference.pinned import (
        class_names,
        pinned_input_shape,
        pinned_model_entry,
        verified_ir_path,
    )
    from app.inference.preprocess import preprocess_bgr

    core = ov.Core()
    model = core.read_model(str(verified_ir_path()))
    spec = describe_outputs(model, num_classes=int(pinned_model_entry()["num_classes"]))
    request = core.compile_model(model, "CPU").create_infer_request()
    _, _, dst_h, dst_w = pinned_input_shape()

    tracker = Tracker(iou_threshold=manifest.iou_threshold)
    machine = RetailEventMachine(manifest, class_names=class_names())

    capture = cv2.VideoCapture(RETAIL_CHECKOUT_CLIP)
    sampled = detections_total = 0
    width = height = 0
    try:
        frame_index = 0
        while True:
            # `grab` advances without converting; `retrieve` converts. Skipping with
            # grab+retrieve measured 23.8 s against 35.7 s for read-on-every-frame,
            # and seeking per frame measured 139.8 s - this clip does not seek well.
            if not capture.grab():
                break
            if frame_index % SAMPLE_STRIDE:
                frame_index += 1
                continue
            ok, image = capture.retrieve()
            if not ok:
                break
            frame_index += 1
            sampled += 1
            width = width or image.shape[1]
            height = height or image.shape[0]

            blob, letterbox = preprocess_bgr(image, dst_w, dst_h)
            request.infer({0: blob})
            decoded = decode_detections(
                spec, {port_key(0): np.array(request.get_output_tensor(0).data)},
                letterbox, manifest.conf_threshold, manifest.iou_threshold,
            )
            detections_total += len(decoded)
            # `now` is the frame's true presentation time, so dwell measured from it is
            # real elapsed time even though frames are sampled.
            now = frame_index / CLIP_FPS
            frame = FramePacket(
                session_id="guard", scenario_id=manifest.id, stream_id="clip",
                frame_id=frame_index, source_pts=now, source_loop_id=0,
                captured_at=now, decoded_at=now, width=width, height=height,
                image=image,
            )
            for track in tracker.update(decoded, now=now):
                machine.observe(track, frame)
    finally:
        capture.release()

    return {
        "sampled_frames": sampled,
        "detections": detections_total,
        "events": machine.events,
        "abandoned": machine.abandoned_transfers,
        "skipped_unmapped_class": machine.skipped_unmapped_class,
        "tracker": tracker,
    }


class TheRealFootage(unittest.TestCase):
    def test_shipped_retail_config_emits_at_least_one_event_per_loop_stride5(self):
        reason = _skip_reason()
        if reason:
            self.skipTest(
                f"cannot run the real-clip guard: {reason}. A skip is not a pass - the "
                f"shipped retail configuration has NOT been verified against the footage "
                f"on this machine."
            )
        for build in SHIPPED_CONFIGURATIONS:
            with self.subTest(configuration=build.__name__):
                manifest = build()
                result = run_clip(manifest)
                events = result["events"]
                self.assertGreater(
                    result["sampled_frames"], 100,
                    "the guard must have analysed a real slice of the clip, not nothing",
                )
                self.assertGreater(
                    result["detections"], 0,
                    "the pinned detector returned nothing at all - the guard is not "
                    "measuring the configuration",
                )
                # The point of the test. Zero events on real footage is a failure.
                self.assertGreaterEqual(
                    len(events), 1,
                    f"{manifest.id} produced ZERO events over {result['sampled_frames']} "
                    f"sampled frames of {RETAIL_CHECKOUT_CLIP} "
                    f"({result['detections']} detections, "
                    f"abandoned={result['abandoned']}). A configuration that detects "
                    f"nothing is not a working configuration."
                )
                for event in events:
                    self.assertEqual(event.source_loop_id, 0)
                    self.assertFalse(event.uncertain)
                    self.assertNotEqual(event.class_name, "unknown")
                    self.assertIn(event.class_name, manifest.event_rules["tracked_classes"])

    def test_the_guard_covers_every_shipped_configuration(self):
        # Always runs, even where the guard above skipped. A configuration shipped
        # without the guard pointing at it is exactly how the previous rectangles
        # survived.
        import app.engine.retail_config as retail_config

        builders = [
            name for name in dir(retail_config)
            if name.endswith("_manifest") and callable(getattr(retail_config, name))
        ]
        self.assertTrue(builders, "no manifest builder found in app/engine/retail_config.py")
        covered = {b.__name__ for b in SHIPPED_CONFIGURATIONS}
        for name in builders:
            self.assertIn(name, covered,
                          f"{name} is shipped but no real-clip guard runs it")


if __name__ == "__main__":
    unittest.main()