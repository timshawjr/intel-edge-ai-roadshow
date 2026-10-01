import unittest
from pathlib import Path

PINNED = {"openvino", "PySide6", "numpy", "opencv-python", "psutil", "PyYAML"}

# The Global Constraint pins exact versions, not just names. A name-only check
# would accept openvino==2025.1.0, which is a different runtime with a different
# NPU behaviour, and the pin test would still pass.
EXPECTED = {
    "openvino": "2026.4.0",
    "PySide6": "6.11.2",
    "numpy": "2.5.3",
    "opencv-python": "5.0.0.93",
    "psutil": "7.2.2",
    "PyYAML": "6.0.3",
}


def parse_requirements():
    """Return {name: version} for every non-comment line of requirements.txt."""

    parsed = {}
    for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        parsed[name] = version
    return parsed


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

    def test_every_requirement_pins_the_pinned_version(self):
        self.assertEqual(parse_requirements(), EXPECTED)
