# M0 — Hardware and Model Path Qualification

Target machine: Intel Core Ultra 9 288V (Lunar Lake, Core Ultra Series 2), ASUS NUC14LNK, Windows 11 Pro build 26200.

> Nothing in this document is a target, a vendor claim, or a number copied from a
> datasheet. Every figure below was produced by a command run in this repository and its
> raw output is pasted verbatim. A quantity that could not be measured is written
> `unknown` and stays that way.

## Assets

Provisioning: `python tools/provision_venv.py` (the only step needing network). Artifacts are
copied from the already-downloaded, content-verified local cache and re-verified in this
tree; nothing is re-downloaded. Every SHA-256 below was computed with `hashlib.sha256` over
the file copied into this tree.

### Pinned environment

`pip freeze` from the provisioned `.venv`, 2026-10-01 (pip is not in the freeze; the pip
that ships with the venv is not recorded because no pip upgrade is performed — see
`tools/provision_venv.py`):

```
numpy==2.5.3
opencv-python==5.0.0.93
openvino==2026.4.0
openvino-telemetry==2025.2.0
psutil==7.2.2
PySide6==6.11.2
PySide6_Addons==6.11.2
PySide6_Essentials==6.11.2
PyYAML==6.0.3
shiboken6==6.11.2
```

`openvino-telemetry` is a transitive dependency of `openvino==2026.4.0`, not a direct pin.
The six direct pins in `requirements.txt` are enforced by `tests/test_pins.py`.

### Source verification

Command: `.venv\Scripts\python.exe tools\verify_sources.py` — content signatures plus the
recorded SHA-256. HTTP status is never accepted as proof; a 200 can return an error page.

```
STATUS  ID                    KIND      DETAIL
----------------------------------------------
PASS    yolo11n-fp16          xml       xml+OpenVINO-IR, 412503 bytes, sha256 4845b46cc9b6ac379a44484258b7ed98f836ed117d91f127c7728ea068250a95
PASS    yolo11n-fp16:weights  binary    binary, 5283218 bytes, sha256 f14632518ebb4e312a89c26faa4cd2f95d4264b6f5f22aa6e0d5707b4667863e
PASS    yolo11n-fp16:labels   text      text+utf-8, 80 lines, 701 bytes, sha256 82bf6a434f132c4d78534c0f2e0f991fc22c73cd911f6d5fc74a418c39c339ec
PASS    retail_checkout       mp4       mp4+ftyp, 64996307 bytes, sha256 e107db69155af4774f907abe8ddbf24493dfcf9d81d51f3e71ff25300e4f5906
PASS    store_aisle_alt       mp4       mp4+ftyp, 9214573 bytes, sha256 3526fee39cac70d6366471e4324a6d63b337c8f8ab99c015521bee0d60ed6e04
PASS    queue_alt             mp4       mp4+ftyp, 3291918 bytes, sha256 a5964aa259099a482a8b360ffc2c57b5a30f84d5919236a4dad01f8e929ac07c

Source verification: 6 PASS, 0 FAIL
```

Six rows: the IR, the weights, the 80-line label file, and the three clips. The
`text` row is a later addition — a recorded hash with no verification row is a hash
nobody checks. The SHA-256 is the gate; the shape checks above it are necessary but
cannot tell one valid IR from another.

### Recorded artifacts

| id | path | sha256 | measured |
|---|---|---|---|
| yolo11n-fp16 (IR) | `assets/models/yolo11n-fp16/yolo11n.xml` | `4845b46c…68a95` | input `[1,3,640,640]`, static |
| yolo11n-fp16 (weights) | `assets/models/yolo11n-fp16/yolo11n.bin` | `f1463251…8663e` | FP16, 5283218 bytes |
| yolo11n-fp16 (labels) | `assets/models/yolo11n-fp16/labels.txt` | `82bf6a43…339ec` | 80 classes |
| retail_checkout | `assets/media/retail-checkout.mp4` | `e107db69…f5906` | 1920x1080, 29.97002997002997 fps, 3387 frames, 113.0129 s |
| store_aisle_alt | `assets/media/store-aisle-detection.mp4` | `3526fee3…6ed6e04` | 720x404, 59.94005994005994 fps, 3921 frames, 65.41535 s |
| queue_alt | `assets/media/one-by-one-person-detection.mp4` | `a5964aa2…9ac07c` | 768x432, 10.0 fps, 1394 frames, 139.4 s |

Media figures are `cv2.VideoCapture` measurements on the copied files. `duration_s` is
`frames / fps`; the values are not rounded to a target.

### Licences and distribution status

| id | licence | distribution_status |
|---|---|---|
| yolo11n-fp16 | AGPL-3.0 (model card, read 2026-10-01) | recorded as-is; AGPL-3.0 is copyleft and is an open distribution question for the booth package |
| retail_checkout | Pexels License - free for commercial use, no attribution required | licence text carried over from the source manifest as recorded; **NOT re-read at M0** and not claimed as booth redistribution clearance |
| store_aisle_alt | Open Model Zoo sample media | recorded as-is; not claimed as booth redistribution clearance |
| queue_alt | Open Model Zoo sample media | recorded as-is; not claimed as booth redistribution clearance |

No clip is upgraded to "cleared". All three licence strings were carried over from the
source manifest rather than re-read from the provider at M0, so all three distribution
statuses are recorded as-is.

### Unmet content gate (carried to Task 9)

Spec §4 wants three 45–90 s clips per pack. The `retail_sanity` pack has three clips, but
only `store_aisle_alt` (65.41535 s) falls inside the 45–90 s band; `retail_checkout`
(113.0129 s) and `queue_alt` (139.4 s) are longer. The three remaining packs have no
provisioned clips at M0. No duration is rounded, trimmed, or padded to fit the band. This
is a declared unmet M5 content gate, not a solved problem.

## Device discovery

Not run. Task 1 provisions the environment and the assets only; device enumeration,
`EXECUTION_DEVICES`, latency and telemetry land in Tasks 2, 3 and 7.
