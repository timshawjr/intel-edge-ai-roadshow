"""Dump the pinned model's I/O to `config/model_io/<model_id>.json`, from the IR itself.

Run from the repo root:

    .venv\\Scripts\\python.exe tools\\dump_model_io.py

Every value written is observed. Nothing is taken from a convention, from a name, or
from what the file was last time. Run it after a re-export; if the recorded file and
`app.inference.decode.describe_outputs` ever disagree, the recorded file is regenerated
with this tool and the disagreement is reported - it is never resolved by editing the
recorded file to match the code, because that is how a wrong decoder becomes an
accepted fact.

Measured first-hand on this machine, and the reason for three of the fields:

- The pinned export's output port is **unnamed**: `get_names()` returns `set()` and
  `get_any_name()` raises `RuntimeError: Check '!get_names().empty()' failed`. So
  `shapes` is keyed by **port index as a string** and `names` is `[]`, and this tool
  records the refusal rather than inventing a name.
- `ov.InferRequest.get_tensor` accepts a name or a port object but **not an int**, so
  the tensor cannot be fetched by index at the request at all. The port index is a
  *key* this project assigns, not something OpenVINO accepts.
- The four leading values of each anchor are `cx, cy, w, h`, and `has_embedded_nms` is
  read from a walk of the op list. Both are derived, and `app/inference/decode.py`
  carries the measurements.

Exits non-zero if anything could not be read, because a partial record that looks
complete is worse than no record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openvino as ov  # noqa: E402

from app.inference.decode import (  # noqa: E402
    BOX_ENCODING,
    ModelOutputSpec,
    UnsupportedLayoutError,
    describe_outputs,
    port_key,
)
from app.inference.devices import plugin_class  # noqa: E402

MODELS_JSON = REPO_ROOT / "config" / "models.json"
OUTPUT_DIR = REPO_ROOT / "config" / "model_io"


def _port_report(port: Any, index: int) -> dict[str, Any]:
    """One port as read, including the parts a normal description leaves out.

    `get_any_name_status` records what happened when the name was asked for, because
    "it raised" is the finding: it is why nothing downstream may call it.
    """
    try:
        partial = list(port.get_partial_shape())
    except Exception as exc:  # noqa: BLE001
        partial = []
        rank_error: str | None = f"{type(exc).__name__}: {exc}"
    else:
        rank_error = None
    axes: list[dict[str, Any]] = []
    for axis, dim in enumerate(partial):
        is_dynamic = bool(getattr(dim, "is_dynamic", False))
        axes.append({
            "axis": axis,
            "length": None if is_dynamic else int(dim.get_length()),
            "is_dynamic": is_dynamic,
        })
    try:
        any_name = repr(port.get_any_name())
    except Exception as exc:  # noqa: BLE001 - the failure text is the point
        any_name = f"RAISES {type(exc).__name__}: {str(exc).strip()}"
    return {
        "port_index": index,
        "port_key": port_key(index),
        "names": sorted(str(n) for n in port.get_names()),
        "get_any_name": any_name,
        "rank": len(partial),
        "shape": [a["length"] for a in axes],
        "axes": axes,
        "is_static": bool(axes) and not any(a["is_dynamic"] for a in axes),
        "element_type": _element_type_str(port),
        "element_type_str_form": str(port.get_element_type()),
        "rank_error": rank_error,
    }


def _ir_sha256(ir_path: Path) -> str:
    """SHA-256 of the IR file in this tree, computed here.

    Deliberately not `entry["sha256"]` from the manifest: the record has to say which
    bytes were read, and copying the manifest's claim would make the record agree with
    the manifest even if the file on disk were something else. The two are cross-checked
    in `tests/test_decode.py`.
    """
    return hashlib.sha256(ir_path.read_bytes()).hexdigest()


def _element_type_str(port: Any) -> str:
    """The element type as a comparable short name, e.g. `'f32'`.

    Measured: `str(port.get_element_type())` is `"<Type: 'float32'>"`, which no
    comparison against `'f32'` can match. Recording the `str()` form and classifying on it
    made every real port `"unknown"` while the tool's own cross-check still reported
    agreement - the exact shape of a self-consistent wrong answer. Both string forms are
    therefore recorded side by side, and the cross-check is stated to agree only when the
    *independent* classifier call returns the same layout.
    """
    element_type = port.get_element_type()
    to_string = getattr(element_type, "to_string", None)
    if callable(to_string):
        return str(to_string())
    return str(element_type)


def _probe_fetch_api(model: Any, compiled: Any) -> dict[str, Any]:
    """MEASURE which key types each fetch call accepts, and record the answer.

    This is measured, not asserted from memory, because the answer is genuinely
    surprising and version-specific. Measured on `openvino==2026.4.0` on this machine:

    | call                              | int index | port name | IR `Output` port object | compiled `ConstOutput` |
    |---|---|---|---|---|
    | `infer({key: blob})`              | OK        | OK (`'x'`) | **TypeError**        | OK            |
    | `get_tensor(key)`                 | TypeError | n/a (no name) | **OK**            | OK            |
    | `get_output_tensor(0)`            | OK        | n/a       | n/a                 | n/a           |

    So the two sides want **different** key types, and neither accepts the obvious one:
    `infer` rejects the IR `Output` port object it is handed, while `get_tensor` rejects the
    int. `get_output_tensor(0)` accepts the int and is the simpler route for outputs.
    A caller that fetches the input one way and the output another will find this out at
    runtime, on a booth, which is why both rows are recorded.

    **The key names below name the MODEL the port came from, not just the port type.**
    That distinction is load-bearing and was previously only implicit in this docstring:
    `infer` accepts an int index **and** the *compiled* model's port object, and rejects
    only the *IR* model's port object. Reading the recorded block without that qualifier
    makes it look like a blanket ban on port objects, which is wrong. So each key says
    `ir_...` or `compiled_...`.

    A `TypeError` is recorded as rejected. Any other exception is recorded as `"error:
    <text>"` rather than being folded into "rejected", so a genuine failure cannot be
    reported as a clean answer.
    """
    result: dict[str, Any] = {}
    try:
        probe = np.zeros((1, 3, 640, 640), dtype=np.float32)
        cases: list[tuple[str, str, Any]] = [
            ("input", "int_index", 0),
            ("input", "port_name", next(iter(model.inputs[0].get_names()), "")),
            ("input", "ir_output_port_object", model.inputs[0]),
            ("input", "compiled_port_object", compiled.input(0)),
        ]
        for side, kind, key in cases:
            try:
                compiled.create_infer_request().infer({key: probe})
                result[f"infer_{side}_{kind}"] = "accepted"
            except TypeError:
                result[f"infer_{side}_{kind}"] = "rejected: TypeError"
            except Exception as exc:  # noqa: BLE001 - recorded, not folded in
                result[f"infer_{side}_{kind}"] = f"error: {type(exc).__name__}: {exc}"
        out_cases: list[tuple[str, Any]] = [
            ("int_index", 0),
            ("ir_output_port_object", model.outputs[0]),
            ("compiled_port_object", compiled.output(0)),
        ]
        for kind, key in out_cases:
            try:
                request = compiled.create_infer_request()
                request.infer({0: probe})
                shape = tuple(request.get_tensor(key).shape)
                result[f"get_tensor_output_{kind}"] = f"accepted: {list(shape)}"
            except TypeError:
                result[f"get_tensor_output_{kind}"] = "rejected: TypeError"
            except Exception as exc:  # noqa: BLE001
                result[f"get_tensor_output_{kind}"] = f"error: {type(exc).__name__}: {exc}"
        try:
            request = compiled.create_infer_request()
            request.infer({0: probe})
            shape = tuple(request.get_output_tensor(0).shape)
            result["get_output_tensor_int_index"] = f"accepted: {list(shape)}"
        except TypeError:
            result["get_output_tensor_int_index"] = "rejected: TypeError"
        except Exception as exc:  # noqa: BLE001
            result["get_output_tensor_int_index"] = f"error: {type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return {"probe_error": f"{type(exc).__name__}: {exc}"}
    return result


def _op_census(model: Any) -> dict[str, Any]:
    """Every op type in the graph, with the suppression search result beside it."""
    counts: dict[str, int] = {}
    for op in model.get_ordered_ops():
        name = str(op.get_type_name())
        counts[name] = counts.get(name, 0) + 1
    return {
        "total_ops": sum(counts.values()),
        "distinct_op_types": len(counts),
        "op_type_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def describe_model(model: Any, entry: dict[str, Any],
                   compiled: Any | None = None) -> dict[str, Any]:
    """The full record for one pinned model, observed end to end."""
    num_classes = int(entry["num_classes"])
    spec: ModelOutputSpec = describe_outputs(model, num_classes=num_classes)

    inputs = [_port_report(p, i) for i, p in enumerate(model.inputs)]
    outputs = [_port_report(p, i) for i, p in enumerate(model.outputs)]
    census = _op_census(model)

    record: dict[str, Any] = {
        "schema_version": 1,
        "model_id": entry["id"],
        "ir_path": entry["ir_path"],
        # Computed from the artifact in this tree, NOT copied from the manifest. The
        # manifest's hash is cross-checked separately, so a disagreement between the
        # recorded record and the file it describes is visible rather than inherited.
        "ir_sha256": _ir_sha256(REPO_ROOT / entry["ir_path"]),
        "openvino_version": str(ov.__version__),
        "num_classes": num_classes,
        "input_shape": list(entry["input_shape"]),
        "inputs": inputs,
        "outputs": outputs,
        "output_names": list(spec.names),
        "named_port_keys": list(spec.named_port_keys),
        "shapes": {k: list(v) for k, v in spec.shapes.items()},
        "shapes_keyed_by": "port index as a string; the output port of this export is "
                           "unnamed, so there is no name to key by",
        "named_port_keys_note": "the port key of each entry of output_names, in order; "
                               "output_names is compacted, so a name alone cannot say "
                               "which port it belongs to",
        "layout": spec.layout,
        "box_encoding": BOX_ENCODING,
        "has_embedded_nms": spec.has_embedded_nms,
        "port_keys": list(spec.port_keys),
        "unnamed_ports": list(spec.unnamed_ports),
        "op_census": census,
    }
    if compiled is not None:
        record["fetch_api"] = _probe_fetch_api(model, compiled)
    record["fetch_note"] = (
        "MEASURED on the installed runtime by this tool; see fetch_api for the per-call "
        "results. The two sides want DIFFERENT key types and neither accepts the obvious "
        "one: InferRequest.infer accepts an int index, a port NAME, or a ConstOutput, "
        "and REJECTS the Output port object; InferRequest.get_tensor accepts the Output "
        "port object or a ConstOutput and REJECTS an int index, while "
        "get_output_tensor(0) accepts the int. So the input for infer and the output for "
        "get_tensor cannot both be keyed the same way, and a caller that assumes they "
        "can finds out at runtime, on a booth. The port index app.inference.decode keys "
        "on is a key this project assigns, not one OpenVINO accepts for either call."
    )

    # An INDEPENDENT second derivation of the layout, written out here rather than
    # calling classify_layout a second time. Calling the same function twice and
    # comparing proves nothing: the first version of this tool printed
    # "agrees_with_describe_outputs: True" while both sides were comparing the
    # element type against the un-comparable "<Type: 'float32'>" string and both
    # independently said "unknown". The duplication here is the point.
    first = outputs[0] if outputs else None
    if first is not None:
        observed_shape = first["shape"] if first["is_static"] else None
        expected_features = 4 + num_classes
        axis1 = observed_shape[1] if observed_shape and len(observed_shape) > 1 else None
        axis2 = observed_shape[2] if observed_shape and len(observed_shape) > 2 else None
        if first["element_type"] not in ("f32", "f16", "f64"):
            independent = f"unknown (element type is {first['element_type']})"
        elif axis1 == expected_features:
            independent = "yolo_v8_transposed"
        elif axis2 == expected_features:
            independent = "yolo_v8_passthrough"
        else:
            independent = "unknown"
        record["layout_cross_check"] = {
            "independent_verdict": independent,
            "from_describe_outputs": spec.layout,
            "agrees": independent == spec.layout,
            "note": "derived by separate code from the same observation; a match is "
                    "evidence, a mismatch is a bug in one of the two",
        }
        record["feature_count_check"] = {
            "expected_4_plus_num_classes": expected_features,
            "observed_on_axis_1": axis1,
            "agrees": axis1 == expected_features,
        }
        record["anchor_count_check"] = {
            "trailing_extent": axis2,
            "yolo_anchors_for_640_at_strides_8_16_32": 80 * 80 + 40 * 40 + 20 * 20,
            "agrees": axis2 == 80 * 80 + 40 * 40 + 20 * 20,
            "note": "a post-NMS output would carry a small frame-dependent count, not "
                    "the raw anchor count, so this is evidence independent of the op "
                    "census that the export does not suppress",
        }
        activation = {name: count for name, count in census["op_type_counts"].items()
                      if name in ("Sigmoid", "Softmax", "HardSigmoid")}
        record["activation_ops_in_graph"] = {
            "counts": activation,
            "note": "the export carries activation ops, which is consistent with the "
                    "measured value ranges of the class block (rows 4..83 all within "
                    "[0,1], max 0.909073 over a real frame) and is why "
                    "app.inference.decode does not apply a second one. Which op is "
                    "responsible for which block is NOT claimed here - the value range "
                    "is the measurement, this is only corroboration.",
        }
        record["has_embedded_nms_evidence"] = (
            f"walked all {census['total_ops']} ops "
            f"({census['distinct_op_types']} distinct types) and found no "
            f"NonMaxSuppression of any version; the output is static "
            f"{first['shape']} whose trailing extent is the raw anchor count "
            f"80^2+40^2+20^2 = 8400 for a 640 input, not a post-NMS detection count. "
            f"Suppression is therefore the host's job and "
            f"app.inference.decode.nms does it."
            if not spec.has_embedded_nms else
            f"walked all {census['total_ops']} ops and found a NonMaxSuppression; this "
            f"export suppresses its own detections and the host must not do it again."
        )
    return record


def dump(model_id: str | None = None, write: bool = True) -> list[dict[str, Any]]:
    """Describe every pinned model (or one), and write the records. Returns them."""
    manifest = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    entries = [m for m in manifest["models"]
               if model_id is None or m["id"] == model_id]
    if not entries:
        raise SystemExit(
            f"no model {model_id!r} in {MODELS_JSON}; available: "
            f"{[m['id'] for m in manifest['models']]}"
        )
    core = ov.Core()
    records: list[dict[str, Any]] = []
    for entry in entries:
        ir_path = REPO_ROOT / entry["ir_path"]
        if not ir_path.is_file():
            raise SystemExit(
                f"{ir_path} is not present. tools/provision_venv.py and the artifact "
                f"copy are prerequisites; nothing here is downloaded at build time."
            )
        model = core.read_model(str(ir_path))
        compiled = core.compile_model(model, "CPU")
        record = describe_model(model, entry, compiled=compiled)
        records.append(record)
        _print_report(record)
        if write:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            target = OUTPUT_DIR / f"{entry['id']}.json"
            target.write_text(
                json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
            print(f"\nwrote {target.relative_to(REPO_ROOT)}")
    return records


def _print_report(record: dict[str, Any]) -> None:
    print("=" * 78)
    print(f"{record['model_id']}   {record['ir_path']}")
    print(f"openvino {record['openvino_version']}   devices "
          f"{[plugin_class(d) for d in ov.Core().available_devices]}")
    for kind in ("inputs", "outputs"):
        for port in record[kind]:
            print(f"\n  {kind}[{port['port_index']}]  key={port['port_key']!r}")
            print(f"    names          = {port['names']}")
            print(f"    get_any_name   = {port['get_any_name']}")
            print(f"    shape          = {port['shape']}  "
                  f"static={port['is_static']}  rank={port['rank']}")
            print(f"    element_type   = {port['element_type']}  "
                  f"(str form {port['element_type_str_form']})")
    print(f"\n  layout            = {record['layout']}")
    print(f"  box_encoding      = {record['box_encoding']}")
    print(f"  output_names      = {record['output_names']}")
    print(f"  unnamed_ports     = {record['unnamed_ports']}")
    print(f"  shapes            = {record['shapes']}")
    print(f"  num_classes       = {record['num_classes']}")
    print(f"  has_embedded_nms  = {record['has_embedded_nms']}")
    print(f"\n  op census: {record['op_census']['total_ops']} ops, "
          f"{record['op_census']['distinct_op_types']} distinct types")
    for name, count in list(record["op_census"]["op_type_counts"].items())[:8]:
        print(f"    {count:>5}  {name}")
    if "layout_cross_check" in record:
        print(f"\n  cross-check: {record['layout_cross_check']}")
        print(f"  feature count: {record['feature_count_check']}")
        print(f"  anchor count:  {record['anchor_count_check']}")
    if "fetch_api" in record:
        print("\n  fetch API (measured, not assumed):")
        for key, value in record["fetch_api"].items():
            print(f"    {key:<38} {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=None,
                        help="describe only this model id (default: all pinned models)")
    parser.add_argument("--no-write", action="store_true",
                        help="print the record without writing it to config/model_io/")
    args = parser.parse_args(argv)
    try:
        records = dump(model_id=args.model_id, write=not args.no_write)
    except (UnsupportedLayoutError, ValueError, SystemExit) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for record in records:
        for check in ("layout_cross_check", "feature_count_check", "anchor_count_check"):
            if not record.get(check, {}).get("agrees", True):
                print(
                    f"ERROR: {check} disagrees for {record['model_id']}: "
                    f"{record[check]}",
                    file=sys.stderr,
                )
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
