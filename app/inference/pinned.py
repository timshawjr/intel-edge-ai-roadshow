"""The pinned model, as `config/models.json` states it. Pure `json` plus `pathlib`.

**This module exists because `app/` could not otherwise ship.** Until M0's final review
the only readers of `config/models.json` lived in `tools/build_sanity_fixture.py`, so the
IR path, the input shape, the class count and the label list were reachable *only* from a
tool. Nothing under `app/` imported `tools/`, and M6 cannot ship an `app/` that has to
import a tool to learn which model it runs. The readers moved here; the tools now import
from here, so there is still exactly one implementation of each.

**Pure stdlib, deliberately.** No OpenVINO, no `cv2`, no `numpy` - this module answers
"what does the pin say", not "load the model". That boundary is what lets a device-status
panel, a settings screen or a `--selftest` read the pin on a machine with no runtime
installed, and it is why the import here is `json` and `pathlib` and nothing else.

Three rules, each of which has already cost something:

1. **The IR's SHA-256 is checked before its bytes are used.** `ir_path()` returns where
   the file is; `verified_ir_path()` returns it only once the bytes hash to the pinned
   digest, and refuses otherwise. A swapped model is a wrong model at full speed, and the
   whole point of pinning is that the booth cannot be showing something else.
2. **The input shape comes from the pin, never from the model's output ports.** The
   output of this export is `[1, 84, 8400]`; `ModelOutputSpec.shapes` holds *output*
   ports, so a shape read from there letterboxes the frame to 8400x8400 and the runtime
   refuses the blob on all three plugins. See `pinned_input_shape`.
3. **Labels come from `labels.txt` in the pinned order**, never from a hardcoded tuple,
   so a re-export with a different label set cannot be decoded against the old names.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

#: The repository root, derived from this file's own location: `app/inference/pinned.py`
#: is two levels below it. Not a relative path, because this module is imported from
#: tools, from tests and from `app.main`, and only one of those three is ever "the repo
#: root" as a working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MODELS_JSON = REPO_ROOT / "config" / "models.json"

#: The one model this milestone qualifies. It is named here rather than taken as the
#: first detector in the file so that a re-export that changes which model is the
#: detector fails loudly instead of silently qualifying a different one.
MODEL_ID = "yolo11n-fp16"


def _models_config() -> dict[str, Any]:
    """`config/models.json` as parsed. Read on every call; it is a 4 KB file."""
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))


def pinned_model_entry() -> dict[str, Any]:
    """The pinned detector's entry, with its `role` checked.

    The `role` check is the reason this is not `next(...)` inline at each call site: a
    config that promoted an embedder to `detector` would otherwise qualify happily
    against a graph whose output layout is not `[1, 84, 8400]`.
    """
    entry = next(m for m in _models_config()["models"] if m["id"] == MODEL_ID)
    if entry["role"] != "detector":
        raise ValueError(f"{MODEL_ID} has role {entry['role']!r}, not 'detector'")
    return entry


def ir_path() -> Path:
    """Where the IR is recorded to be. Not a statement that those bytes are the pinned
    ones - that is `verified_ir_path()`, and the difference is the whole point."""
    return REPO_ROOT / pinned_model_entry()["ir_path"]


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file, streamed so a large artifact does not have to be resident."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_sha256() -> str:
    """The digest `config/models.json` pins for the IR."""
    return str(pinned_model_entry()["sha256"])


def verified_ir_path() -> Path:
    """`ir_path()`, but only after the bytes on disk match the pinned SHA-256.

    Raising is the right answer here and not a warning: a mismatched IR is not a degraded
    run, it is a different model, and a tool that carried on would decode a stranger's
    graph with this project's annotations and report the result as evidence.
    """
    entry = pinned_model_entry()
    path = REPO_ROOT / entry["ir_path"]
    try:
        actual = sha256_of(path)
    except OSError as exc:
        raise FileNotFoundError(
            f"the pinned IR {path} could not be read: {exc}. Provision the tree with "
            f"tools/provision_venv.py before running anything that needs the model."
        ) from exc
    if actual != entry["sha256"]:
        raise ValueError(
            f"{entry['id']} IR {entry['ir_path']} hashes to {actual} but "
            f"config/models.json pins {entry['sha256']}. Refusing to run against an IR "
            f"that is not the pinned one; re-provision rather than substitute."
        )
    return path


def pinned_input_shape() -> tuple[int, int, int, int]:
    """The pinned model's `[N, C, H, W]`, from `config/models.json`.

    **Not** from `ModelOutputSpec.shapes`, and the first version of this function got that
    wrong in a way worth recording. `ModelOutputSpec.shapes` holds the **output** ports,
    and this model's output is `[1, 84, 8400]` - so `spec.shapes[port_key(0)][-1]` is
    `8400`, and letterboxing the frame to 8400x8400 produced a blob the runtime refused on
    all three plugins with `The input tensor size is not equal to the model input type: got
    [1,3,8400,8400] expecting [1,3,640,640]`. The model input is `[1, 3, 640, 640]` in
    `config/models.json` and the spec does not carry it, so that is where it is read from.
    The error was legible and the device reported as failed rather than as zero, which is
    the behaviour that made the bug findable at all.

    The four-axis check is here, not at the call site, because the NPU rejects dynamic
    shapes and `-1` is how a dynamic dimension arrives in a config file.
    """
    shape = tuple(int(v) for v in pinned_model_entry()["input_shape"])
    if len(shape) != 4:
        raise ValueError(f"config/models.json input_shape must be [N,C,H,W]; got {shape!r}")
    if any(extent <= 0 for extent in shape):
        raise ValueError(
            f"config/models.json input_shape {shape!r} is not static. The NPU rejects "
            f"dynamic shapes, so a pinned input cannot carry -1 or 0."
        )
    return shape  # type: ignore[return-value]


def num_classes() -> int:
    """How many classes the pinned detector's class block holds, per the pin.

    Read from `input_shape`-adjacent config rather than from the labels file, so the
    cross-check in `labels_match_pin` has something to disagree with.
    """
    entry = pinned_model_entry()
    declared = entry.get("num_classes")
    if declared is not None:
        return int(declared)
    # The YOLO11 export's output width is `4 + num_classes`; with no explicit count in
    # the config this is the derivation, and it is recorded as one rather than as a read.
    raise KeyError(
        f"config/models.json records no num_classes for {entry['id']}, so the class count "
        f"is not something this module may guess. Add the field or read it from the IR."
    )


def labels_path() -> Path:
    """Where the label file is recorded to be."""
    return REPO_ROOT / pinned_model_entry()["labels_path"]


def class_names() -> tuple[str, ...]:
    """The pinned label list, in the order the model indexes its class block.

    Read from `labels.txt` and never from a hardcoded tuple, so a re-export with a
    different label set cannot be decoded against the old names.
    """
    lines = [line.strip() for line in labels_path().read_text(encoding="utf-8").splitlines()]
    return tuple(name for name in lines if name)


def labels_match_pin() -> bool:
    """Whether the label file's length equals the pin's declared class count.

    The cross-check that makes the class count load-bearing rather than decorative: an
    80-line label file decoded against a model expecting a different number of classes
    is a silent mislabel at best and an out-of-range read at worst.
    """
    return len(class_names()) == num_classes()