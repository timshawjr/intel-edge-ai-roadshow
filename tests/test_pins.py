import re, unittest
from pathlib import Path

PINNED = {"openvino", "PySide6", "numpy", "opencv-python", "psutil", "PyYAML"}

class PinsAreExact(unittest.TestCase):
    def test_every_requirement_is_exactly_pinned(self):
        text = Path("requirements.txt").read_text(encoding="utf-8")
        seen = set()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            self.assertRegex(line, r"^[A-Za-z0-9_.-]+==[^=<>!]+$",
                             f"not an exact pin: {line!r}")
            seen.add(line.split("==", 1)[0])
        self.assertEqual(seen, PINNED)
