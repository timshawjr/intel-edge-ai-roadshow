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


class TheVenvIsThePinnedOne(unittest.TestCase):
    """`requirements.txt` is the *intent*; the installed environment is the *machine*.

    Every other test in this file read `requirements.txt`, which is a text file that
    describes what somebody meant to install. Nothing checked what was actually
    installed, so a venv carrying a different transitive resolution - one extra patch of
    numpy, a stale PySide6 - passed the pin test on a machine whose behaviour had
    already changed. `importlib.metadata.version` reads the installed distribution, so
    this asks the environment rather than the file.

    One assertion per **direct** pin, and only direct pins: `openvino-telemetry` is a
    transitive dependency of `openvino==2026.4.0` (recorded in the M0 report's pinned
    environment) and pinning it here would turn a resolver's choice into a constraint.
    """

    def test_each_direct_pin_is_the_installed_version(self):
        import importlib.metadata as metadata
        for name, version in sorted(EXPECTED.items()):
            try:
                installed = metadata.version(name)
            except metadata.PackageNotFoundError:
                self.fail(
                    f"{name}=={version} is pinned in requirements.txt but is not "
                    f"installed in the interpreter running this suite. The tests are "
                    f"running against something other than the pinned environment."
                )
            self.assertEqual(installed, version,
                             f"{name}: requirements.txt pins {version} and this "
                             f"interpreter has {installed}")

    def test_the_interpreter_running_the_suite_is_the_provisioned_one(self):
        """A pin test is only meaningful inside the venv it was written for.

        `tools/provision_venv.py` pins CPython 3.12, and this suite is written against
        that interpreter's stdlib. Run against a system Python, `EXPECTED` still matches
        `requirements.txt` and every pin above still passes - which is the state where
        the pin file is green and the machine is not. Asserting the major/minor is cheap
        and makes that state loud.
        """
        import sys
        self.assertEqual((sys.version_info.major, sys.version_info.minor), (3, 12),
                         f"running on Python {sys.version.split()[0]}; the pins were "
                         f"provisioned against 3.12")
