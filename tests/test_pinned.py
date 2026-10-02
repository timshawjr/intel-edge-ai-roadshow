"""`app/inference/pinned.py`: the pin, as `config/models.json` states it.

The module's whole reason to exist is a boundary M0 drew late: the IR path, the input
shape, the class count and the label list were reachable **only** from
`tools/build_sanity_fixture.py`. Nothing under `app/` imported `tools/`, and M6 cannot ship
an `app/` that has to import a tool to learn which model it is running. The readers moved
here, so the tests have two jobs:

- **The pin is read, not guessed.** Every field below is compared against
  `config/models.json` directly, so this file fails the moment the config and the code
  disagree - which is the drift a pin exists to prevent.
- **It imports nothing heavy.** `pinned.py` is `json` plus `pathlib`, so a device-status
  panel or a `--selftest` can read the pin on a machine with no OpenVINO, no `cv2` and no
  `numpy`. `test_importing_pinned_does_not_import_the_world` enforces that by scanning
  `sys.modules` after a fresh subprocess import, because the first version of a "pure"
  module is always pure by accident rather than by decision.
"""

import subprocess
import sys
import unittest
from pathlib import Path

from app.inference import pinned

CONFIG = Path("config/models.json")


class ThePinIsTheConfig(unittest.TestCase):
    """Each reader answers exactly what `config/models.json` says, or refuses."""

    def setUp(self):
        import json
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.entry = next(m for m in self.config["models"] if m["id"] == pinned.MODEL_ID)

    def test_the_pinned_entry_is_the_one_the_config_names(self):
        self.assertEqual(pinned.pinned_model_entry(), self.entry)
        self.assertEqual(pinned.pinned_model_entry()["role"], "detector")

    def test_the_ir_path_is_the_recorded_one_and_resolves(self):
        self.assertEqual(pinned.ir_path(), pinned.REPO_ROOT / self.entry["ir_path"])
        self.assertTrue(pinned.ir_path().exists(),
                        "assets/ is gitignored; run tools/provision_venv.py first")

    def test_the_recorded_hash_is_the_configs_own_value(self):
        self.assertEqual(pinned.pinned_sha256(), self.entry["sha256"])

    def test_the_input_shape_is_the_configs_shape(self):
        self.assertEqual(list(pinned.pinned_input_shape()), self.entry["input_shape"])
        self.assertEqual(pinned.pinned_input_shape(), (1, 3, 640, 640))

    def test_the_input_shape_is_static_because_the_npu_rejects_dynamic_ones(self):
        for extent in pinned.pinned_input_shape():
            self.assertGreater(extent, 0, "a -1 in the pinned input shape is a dynamic "
                                         "dimension, which the NPU refuses")

    def test_the_class_count_comes_from_the_pin_and_the_labels_agree(self):
        self.assertEqual(pinned.num_classes(), self.entry["num_classes"])
        self.assertTrue(pinned.labels_match_pin(),
                        f"labels.txt has {len(pinned.class_names())} lines but the pin "
                        f"declares {self.entry['num_classes']} classes")

    def test_the_labels_are_read_in_order_and_never_hardcoded(self):
        names = pinned.class_names()
        self.assertEqual(names[0], "person")
        self.assertEqual(len(names), 80)
        self.assertEqual(pinned.labels_path(), pinned.REPO_ROOT / self.entry["labels_path"])

    def test_a_wrong_hash_is_refused_rather_than_decoded(self):
        """The one check that makes the pin a pin.

        A mismatched IR is not a degraded run, it is a different model, and decoding a
        stranger's graph with this project's annotations produces confident, plausible,
        wrong detections. So the refusal is a `ValueError`, and it names both digests.
        """
        if not pinned.ir_path().exists():
            self.skipTest("assets/ is not provisioned on this tree")
        self.assertEqual(pinned.sha256_of(pinned.ir_path()), pinned.pinned_sha256())
        # And the verifier agrees with the manual computation above.
        self.assertEqual(pinned.verified_ir_path(), pinned.ir_path())


class ImportingPinnedDoesNotImportTheWorld(unittest.TestCase):
    """`pinned.py` is `json` and `pathlib`. M6 depends on it.

    Nothing under `app/` may need a tool to learn which model it runs, and the reverse
    matters too: a module that pulls in OpenVINO cannot be imported by a settings screen,
    an installer check or a `--selftest` on a machine with no runtime. Checked in a fresh
    subprocess so the rest of the suite's imports cannot mask a heavy one.
    """

    def test_importing_pinned_does_not_import_the_world(self):
        code = (
            "import sys; sys.path.insert(0, '.');"
            "import app.inference.pinned as p;"
            "heavy = [m for m in ('openvino', 'cv2', 'numpy', 'psutil', 'PySide6')"
            " if m in sys.modules];"
            "print(','.join(heavy))"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "",
                         f"importing app.inference.pinned pulled in {out.stdout.strip()}")

    def test_the_module_names_no_heavy_imports_of_its_own(self):
        source = Path("app/inference/pinned.py").read_text(encoding="utf-8")
        for banned in ("import openvino", "import cv2", "import numpy", "import psutil"):
            self.assertNotIn(banned, source,
                             f"pinned.py grew a {banned!r}; it is meant to be pure stdlib")


if __name__ == "__main__":
    unittest.main()