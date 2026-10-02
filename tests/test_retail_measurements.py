"""The measured claims in `app/engine/retail_config.py`.

That module's whole purpose is to be the record of what the footage supports, and its
first version shipped three false numbers. This module pins the numbers that replaced
them, so a future edit that quietly changes one fails here instead of at a trade show.

**These are pinned measurements, not live ones.** They are asserted against constants in
the module rather than re-decoded, because re-decoding 3387 frames on CPU takes ~115 s
and the *live* check - does the shipped configuration still emit events on the real
clip - is `tests/test_retail_footage_guard.py`'s job. What this module protects is the
prose: that the docstring's figures and its constants still agree with each other, and
that the claims it makes are the ones it can support.

If the detector or the clip is re-provisioned, these constants are expected to be
**re-measured and deliberately updated here**, with the reasoning recorded - which is
the point. A silent drift is the failure; a recorded one is a maintenance task.
"""

import json
import pkgutil
import re
import unittest
from pathlib import Path

from app.engine import retail_config
from app.engine.retail_config import (
    COUNTER_BAND,
    MIN_PRESENCE_SECONDS,
    RETAIL_CHECKOUT_CLIP,
    RETAIL_CHECKOUT_ID,
    TRACKED_CLASSES,
    clip_fps,
    clip_measurements,
    retail_sanity_manifest,
)

MODULES_JSON = Path("config/models.json")


def _module_doc() -> str:
    from app.engine import retail_config as module
    return (Path(module.__file__).read_text(encoding="utf-8"))


def _flat(text: str) -> str:
    """Whitespace-collapsed text.

    The prose assertions below must not be defeatable by re-wrapping a paragraph.
    An earlier version of this file asserted on raw source, so a claim could have
    survived a reflow that split it across a line break - the check passed against
    a docstring that still said the thing it was supposed to forbid.
    """
    return re.sub(r"\s+", " ", text)


class ClipFacts(unittest.TestCase):
    def test_the_clip_facts_are_read_from_verified_config(self):
        entry = clip_measurements()
        self.assertEqual(entry["id"], RETAIL_CHECKOUT_ID)
        self.assertEqual(entry["path"], RETAIL_CHECKOUT_CLIP)
        self.assertAlmostEqual(entry["measured_frames"], 3387)
        self.assertAlmostEqual(entry["measured_duration_s"], 113.0129, places=4)

    def test_fps_and_duration_agree_with_the_frame_count(self):
        entry = clip_measurements()
        self.assertAlmostEqual(
            entry["measured_frames"] / entry["measured_duration_s"], clip_fps(), places=4
        )

    def test_no_hardcoded_fps_constant_remains(self):
        # The name survives in the prose that records the correction, so what is pinned
        # is that it is not a live constant: not an attribute, and never assigned.
        self.assertFalse(hasattr(retail_config, "CLIP_FPS"))
        source = _module_doc()
        self.assertNotRegex(source, r"^\s*CLIP_FPS\s*=", )

    def test_an_unknown_clip_id_raises_rather_than_defaulting(self):
        with self.assertRaises(KeyError):
            clip_measurements("no_such_clip")


class CorrectedMeasurements(unittest.TestCase):
    """Each constant below replaced a number that was wrong in a shipped file."""

    def test_the_band_is_what_was_measured(self):
        self.assertEqual(COUNTER_BAND, (0.00, 0.20, 1.00, 0.60))
        self.assertEqual(MIN_PRESENCE_SECONDS, 2.0)
        self.assertEqual(TRACKED_CLASSES, ("person",))

    def test_the_docstring_records_the_corrected_flicker_numbers(self):
        # The false claim was "changed 0 times in 2063 frames". The true measurement,
        # stride 1 over 1792 adjacent pairs where both frames carry a >8% non-person
        # detection, is 298 label-SET changes (16.6%) and 228 dominant-label changes by
        # largest box (12.7%). Both are asserted so the docstring cannot drift back.
        doc = _module_doc()
        self.assertIn("298 of 1792 pairs (16.6%)", doc)
        self.assertIn("228 of 1792 pairs (12.7%)", doc)
        # 2063 is a real frame count - it is how many frames carry a >8% non-person
        # detection - and it is quoted here only as the figure the false claim was
        # built on. What is pinned is that it is presented as withdrawn and not as a
        # measurement: "changed 0 times" must never appear as a table row.
        self.assertIn("2063 frames", _flat(doc))
        self.assertIn("That was false", _flat(doc))
        for line in doc.splitlines():
            if "2063 frames" in line:
                self.assertNotIn("changed **0 times**", line,
                                 "the false claim must never appear as a table row")

    def test_the_docstring_records_both_corrections_to_the_centroid_range(self):
        # Three ranges, all measured, none of them the one a reader would guess:
        # the reviewer's `cx 0.15..0.47` is the wrong figure for the surface, the
        # previous round's `cx 0.125..0.873` is right to three decimals, and the
        # `oven`/`suitcase`/`laptop` subset is `cx 0.138..0.493`. The subset figure
        # was itself wrong once (`0.148..0.472`, which is close to `oven` alone).
        flat = _flat(_module_doc())
        self.assertIn("cx 0.125..0.873", flat)
        self.assertIn("cx 0.138..0.493", flat)
        self.assertNotIn("cx 0.148..0.472", flat,
                         "an unreproducible figure; it is oven alone at 0.148..0.474")
        self.assertIn("cx 0.148..0.474", flat)
        self.assertIn("cx 0.15..0.47", flat)  # quoted only as the withdrawn figure

    def test_the_old_zone_counts_state_which_zone_predicate_they_used(self):
        # 1286 and 3281 are both "detections inside the old BASKET_ZONE": one is
        # `Zone.contains` (whole box), the other `contains_point` (centroid). Which
        # is which is stated in the table, because a reader who assumes one and gets
        # the other is off by a factor of two and a half.
        flat = _flat(_module_doc())
        self.assertIn("Zone.contains", flat)
        self.assertIn("3281", flat)
        self.assertIn("1430", flat)

    def test_the_frame_by_frame_flip_is_recorded_as_an_instance(self):
        # The aggregate counts can be argued with; four consecutive frames of one
        # unmoved object carrying two labels cannot.
        self.assertIn("1514 `[oven 0.152]`", _flat(_module_doc()))

    def test_the_docstring_names_the_mechanism_of_the_false_zero(self):
        # The old figure was not a small measurement error; it was arithmetic that
        # returns zero for any footage. `class_id` is constant within a track by
        # construction, so a count taken over a track's label measures nothing - and
        # the docstring has to say *why* the old number was zero, or the next reader
        # is left to assume the surface genuinely was stable.
        flat = _flat(_module_doc())
        self.assertIn("constant by construction", flat)
        self.assertIn("0 of the 639 track ids", flat)
        self.assertIn("returns zero for any footage at all", flat)

    def test_the_docstring_does_not_claim_zero_person_detections_in_the_old_zone(self):
        # Was "0, on every track". Re-measured: 3 (tracks 282, 339, 340). The phrase
        # survives only inside the sentence that quotes and corrects it, so what is
        # pinned is the corrected number and the absence of the old table row.
        doc = _module_doc()
        self.assertNotIn("| **0, on every track** |", doc)
        self.assertIn("that is **3** (tracks 282, 339, 340", _flat(doc))

    def test_the_docstring_states_the_bands_limited_coverage(self):
        # The band captures ~65% of person detections, not all of them. A module titled
        # "as measured on the provisioned footage" must not imply it captures people.
        # The counts, not just the percentages, are pinned - a percentage with no
        # denominator is the shape of claim that started this.
        doc = _flat(_module_doc())
        self.assertIn("65.1%", doc)
        self.assertIn("34.9%", doc)
        self.assertIn("two thirds of person detections, not all of them", doc)
        self.assertIn("3512 of 5393", doc)
        self.assertIn("1881 of 5393", doc)

    def test_the_docstring_says_what_the_two_events_are_not(self):
        doc = _flat(_module_doc())
        self.assertIn("not a people count", doc)
        self.assertIn("carry **no duration**", doc)
        self.assertIn("two sustained presences, one per person-track", doc)

    def test_the_scope_note_states_the_reduction(self):
        note = retail_sanity_manifest().scope_note
        self.assertIn("No product-transfer event is detectable", note)
        self.assertIn("Shelf-to-basket is NOT delivered", note)
        self.assertIn("12.7% of adjacent frame pairs", note)
        self.assertIn("65% of person detections", note)

    def test_nothing_shipped_claims_items_entering_the_checkout_zone(self):
        # The capitalised form is a delivered claim; the lowercase form appears only
        # inside sentences that deny it, so only the capitalised one is pinned.
        for source in _shipped_sources():
            self.assertNotIn("Items entering the checkout zone", source)

    def test_the_withdrawn_wording_is_gone_from_the_test_suite_too(self):
        # `app/` is where a claim reaches a customer. The test suite is the next
        # tier: a fixture string is quoted in review, copied into a deck, and is a
        # delivered claim by the time anyone downstream reads it. Two fixture
        # `business_line`s and one scope note still carried the withdrawn wording
        # after the first fix round, so this scans `tests/` as well.
        #
        # `docs/` is deliberately **not** scanned: the M0 qualification docs and the
        # plan file are records of what was claimed at the time and are outside this
        # dispatch's write scope. They are reported, not rewritten.
        for path in sorted(Path("tests").glob("*.py")):
            # This module necessarily names the withdrawn strings - it is the thing
            # looking for them - so it excludes itself.
            if path.name == Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8")
            for withdrawn in ("Items entering the checkout zone",
                              "basket -> scanner plate",
                              "basket -> checkout"):
                # Line numbers only. The earlier version of this assertion put the
                # whole file in the failure message, which buries the one line that
                # matters under 700 lines of unrelated source.
                offenders = [n for n, line in enumerate(text.splitlines(), 1)
                             if withdrawn in line]
                self.assertEqual(offenders, [],
                                 f"{path} still carries the withdrawn claim "
                                 f"{withdrawn!r} on line(s) {offenders}")

    def test_no_shipped_source_still_describes_the_clip_as_a_basket_transfer(self):
        # Withdrawn claims, still present in two module docstrings after the first fix
        # round. A stale claim in shipped source is the failure this project treats as
        # disqualifying, so it is pinned rather than left to a reader's judgement.
        for source in _shipped_sources():
            for withdrawn in ("basket -> scanner plate", "basket -> checkout"):
                self.assertNotIn(withdrawn, source)

    def test_the_shipped_business_line_is_the_one_the_config_declares(self):
        self.assertEqual(retail_sanity_manifest().business_line,
                         "People present at the checkout counter")

    def test_no_shipped_docstring_cites_a_test_that_does_not_exist(self):
        # An earlier revision of `events.py` cited
        # `test_a_basket_to_checkout_move_is_a_candidate_transfer`, which had been
        # deleted. A citation to a missing test is a claim with nothing behind it.
        #
        # Scans **shipped source only** (`app/`). Scanning the test files themselves
        # would match every test's own `def` line and report the whole suite as a wall
        # of dangling citations.
        cited: set[str] = set()
        for source in _shipped_sources():
            # Strip file paths first: `tests/test_events.py` contains the *module*
            # name `test_events`, which is not a test method and would be reported as
            # a dangling citation.
            scrubbed = re.sub(r"[\w./\\-]*test_[\w]+\.py", " ", source)
            cited |= set(re.findall(r"\btest_[a-z0-9_]+\b", scrubbed))
        self.assertTrue(cited, "no test is cited by shipped source; the scan is broken")
        existing = _existing_test_names()
        for name in sorted(cited):
            self.assertIn(name, existing,
                          f"{name} is cited in a shipped docstring but no such test exists")

    def test_business_line_does_not_claim_a_product_transfer(self):
        text = retail_sanity_manifest().business_line.lower()
        for forbidden in ("shelf", "basket", "transfer", "item"):
            self.assertNotIn(forbidden, text)


def _events_doc() -> str:
    from app.engine import events
    return Path(events.__file__).read_text(encoding="utf-8")


def _shipped_sources() -> list[str]:
    """The text of every module under `app/` that a docstring can cite a test from."""
    import app as app_package

    out: list[str] = []
    for info in pkgutil.walk_packages(app_package.__path__, prefix="app."):
        try:
            module = __import__(info.name, fromlist=["*"])
        except Exception:  # noqa: BLE001 - a module that will not import cites nothing
            continue
        file = getattr(module, "__file__", None)
        if file:
            out.append(Path(file).read_text(encoding="utf-8"))
    return out


def _existing_test_names() -> set[str]:
    """Every `test_*` method in the M1 test modules.

    Walks the classes: `hasattr(module, name)` is always `False` for a test method,
    because the methods live on `unittest.TestCase` subclasses, not on the module.
    """
    names: set[str] = set()
    for stem in ("test_frame_contract", "test_zones", "test_scenario_manifest",
                 "test_tracker", "test_events", "test_retail_footage_guard",
                 "test_retail_measurements"):
        module = __import__(f"tests.{stem}", fromlist=["*"])
        for attr in vars(module).values():
            if isinstance(attr, type):
                names |= {n for n in vars(attr) if n.startswith("test_")}
    return names


class ConfigurationShape(unittest.TestCase):
    def test_the_pack_is_presence_only(self):
        rules = retail_sanity_manifest().event_rules
        self.assertEqual(rules["presence_zone"], "counter_band")
        self.assertNotIn("origin_slot", rules)
        self.assertNotIn("destination_slot", rules)

    def test_the_manifest_module_docstring_titles_itself_as_measured(self):
        # The title the finding objected to is fine; what was wrong was the numbers
        # under it. Assert the heading is still there so a rewrite notices.
        self.assertIn("measurements it rests on", _module_doc())


class ConfigFileIsReadOnly(unittest.TestCase):
    def test_this_module_never_writes_config(self):
        # `config/models.json` is verified data. The reader above must be read-only,
        # so the mtime is unchanged by importing and reading.
        before = MODULES_JSON.stat().st_mtime
        clip_fps()
        clip_measurements()
        self.assertEqual(MODULES_JSON.stat().st_mtime, before)

    def test_the_reader_returns_the_verified_values(self):
        payload = json.loads(MODULES_JSON.read_text(encoding="utf-8"))
        ids = {m["id"] for m in payload["media"]}
        self.assertIn(RETAIL_CHECKOUT_ID, ids)


if __name__ == "__main__":
    unittest.main()