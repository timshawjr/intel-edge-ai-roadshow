"""`CanvasLayoutAtFourScales` - the M3 gate, measured against an independent recomputation.

## Why "independent recomputation" and not bounds

A bounds check answers "is this inside the canvas", and a box can be inside the canvas and in the
wrong place. M1 proved it: bounds passed while **every** overlay box was ~400 rows wrong, dividing
`y` by `frame.width` instead of `frame.height`. M2 proved the other half - a `Repeater` delegate is
invisible to `findChild`, and `Math.max` passes `NaN` through, and both produced **passing tests over
empty results**.

So every position here is checked against a number computed **from `app/theme.py` by a different route
than the QML uses**, and the tests carry **negative controls**: a swapped-divisor recomputation that
must produce a *different* number, so an assertion cannot pass for the wrong reason.

## What is asserted, and where the numbers come from

| assertion | recomputed from | the QML's own route |
|---|---|---|
| each card's height | `(band - GAP*(n-1)) / n` | the same arithmetic in a delegate |
| the last card's bottom | `MARGIN + bandTotal*GRID.header + GAP + bandTotal*GRID.main` | anchored `top`/`bottom` on a band |
| the card column's `x` | `MARGIN + round(avail * VIDEO_WIDTH_FRACTION) + GAP` | anchored to `heroColumn.right` |
| the hero's aspect | tile `w/h` vs frame `w/h` | `anchors.fill` |

The last one is the M1 trap restated as a test: a tile's extent divided by the frame's extent must
equal the frame's aspect ratio, and the **swapped** division must not.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app import theme  # noqa: E402
from tools import capture_ui  # noqa: E402


def _code_only(path: pathlib.Path) -> str:
    """The file with its `//` comments removed, so a guard cannot be disabled by prose.

    Every QML file in `app/ui/` uses `//` for comments and none of them uses a `/* */` block.
    The first version of `test_the_layout_tokens_are_in_theme_and_not_in_the_qml` searched the
    whole file, found `0.42`, and failed **on the comment that explains where `0.42` went** -
    a guard that has to be switched off by the explanation of the thing it enforces.
    """
    kept = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        cut = line.find("//")
        kept.append(line if cut < 0 else line[:cut])
    return "\n".join(kept)


def band_total() -> float:
    """The height the three bands share: the canvas less the outer margin and the two gaps."""
    return theme.CANVAS_HEIGHT - theme.MARGIN * 2 - theme.GAP * 2


def main_band() -> tuple[float, float]:
    """`(top, bottom)` of the main band, in scene coordinates.

    **Recomputed from the tokens, not read off the screen.** `Main.qml` derives the bands as
    fractions of `bandTotal` inside a `Column` with `anchors.margins: Theme.MARGIN`, so the top
    of the header is `MARGIN` and each subsequent band starts a `GAP` lower.
    """
    total = band_total()
    top = theme.MARGIN + total * theme.GRID["header"] + theme.GAP
    return top, top + total * theme.GRID["main"]


def card_height(count: int) -> float:
    """One card's height when `count` cards share the main band."""
    return ((main_band()[1] - main_band()[0])
            - theme.GAP * max(0, count - 1)) / max(1, count)


def hero_column_width() -> float:
    """The video column's width: 68% of the canvas less the outer margins, rounded."""
    return round((theme.CANVAS_WIDTH - theme.MARGIN * 2) * theme.VIDEO_WIDTH_FRACTION)


class CanvasLayoutAtFourScales(unittest.TestCase):
    """The booth screen at 100 / 125 / 150 / 200 %, at the full 1920x1080 canvas."""

    STREAMS = ("A", "B", "C")

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="capture-canvas-")
        cls.results = {
            scale: capture_ui.capture(scale_percent=scale, seconds=10.0,
                                      out_dir=pathlib.Path(cls._tmp.name))
            for scale in capture_ui.SCALES
        }

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def result(self, scale):
        return self.results[scale]

    # -- the things must be there before anything is said about where they are ----------

    def test_every_scale_reached_the_state_it_was_asked_to_capture(self):
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            self.assertTrue(result.settled, f"{scale}%: missing {result.settle_missed}")
            self.assertEqual(result.absent, (), f"{scale}%: {list(result.absent)}")
            self.assertEqual(result.unreachable, (), f"{scale}%: {list(result.unreachable)}")
            self.assertEqual(
                (result.pixel_width, result.pixel_height),
                {100: (1920, 1080), 125: (2400, 1350), 150: (2880, 1620),
                 200: (3840, 2160)}[scale],
                f"{scale}%: the PNG must be the requested pixel size, not the logical size")

    def test_the_three_cards_and_three_tiles_are_present_and_non_degenerate(self):
        """**Assert the items exist, then assert their geometry.** In that order, always.

        M2's empty metrics column produced passing assertions because `""` was read for an item
        that was not there, and `Math.max` had swallowed a `NaN` height into a zero-height
        delegate. A degenerate item - zero width or height, or a name that is present but has no
        extent - is the shape that failure took, so it is what is checked.
        """
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            for sid in self.STREAMS:
                for name in (f"card_{sid}", f"streamAgeText_{sid}",
                             f"streamStaleText_{sid}", f"streamWindowText_{sid}",
                             f"streamBacklogText_{sid}", f"streamRateText_{sid}"):
                    self.assertIn(name, result.extents,
                                  f"{scale}%: {name} must be reached **and measured**. The "
                                  f"cards are `Repeater` delegates, invisible to `findChild`, "
                                  f"so this is the assertion a `findChild`-based reader could "
                                  f"not make.")
                    x, y, w, h = result.extents[name]
                    self.assertGreater(w, 1.0,
                                       f"{scale}%: {name} is {w} px wide - present but "
                                       f"degenerate, which is the shape M2's empty column took")
                    self.assertGreater(h, 1.0,
                                       f"{scale}%: {name} is {h} px tall - present but "
                                       f"degenerate")
                    self.assertTrue(all(value == value for value in (x, y, w, h)),
                                    f"{scale}%: {name} has a NaN coordinate {result.extents[name]}. "
                                    f"`Math.max` passes NaN through and nothing warns.")
            # The two thumbnails, and the hero.
            for name in ("heroTile", "thumb_B", "thumb_C"):
                self.assertIn(name, result.extents, f"{scale}%: {name}")
                self.assertGreater(result.extents[name][2], 1.0, f"{scale}%: {name} is degenerate")

    # -- positions, recomputed independently -------------------------------------------

    def test_each_card_gets_the_band_less_the_gaps(self):
        """**The fix this task made.** `parent.height / n` ignored the `Column`'s `spacing`, so
        three cards claimed 3 x 220 = 660 px of a 660 px band **plus** two 16 px gaps: 32 px of
        card drawn into the footer, with nothing clipping and nothing warning.
        """
        want = card_height(len(self.STREAMS))
        for scale in capture_ui.SCALES:
            for sid in self.STREAMS:
                got = result_h = self.result(scale).extents[f"card_{sid}"][3]
                self.assertAlmostEqual(got, want, places=1,
                                       msg=f"{scale}%: card {sid} is {got:.2f} px tall, "
                                           f"recomputed {want:.2f} = "
                                           f"(main band {main_band()[1] - main_band()[0]:.0f} - "
                                           f"{theme.GAP} * {len(self.STREAMS) - 1}) / "
                                           f"{len(self.STREAMS)}")

    def test_the_last_card_ends_exactly_on_the_main_band(self):
        """**An independent recomputation of a position, not a bounds check.**

        The **last** card's bottom must equal the main band's bottom, recomputed from `MARGIN`,
        `GAP` and `GRID`. A layout that pushed the column 32 px lower would still be inside the
        canvas and would pass any bounds assertion; this catches it. **Before this task, the
        last card ended at 852 against a band that ends at 820** - 32 px into the footer, drawn
        anyway because a `Column` does not clip.
        """
        _, bottom = main_band()
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            x, y, w, h = result.extents[f"card_C"]
            self.assertAlmostEqual(y + h, bottom, places=1,
                                   msg=f"{scale}%: the last card ends at {y + h:.2f}, the "
                                       f"main band ends at {bottom:.2f} "
                                       f"(MARGIN {theme.MARGIN} + bandTotal "
                                       f"{band_total():.0f} x GRID.header "
                                       f"{theme.GRID['header']} + GAP {theme.GAP} + "
                                       f"bandTotal x GRID.main {theme.GRID['main']})")

    def test_each_card_starts_one_gap_below_the_one_before(self):
        """**The pitch, which is the other half of the division.**

        Three cards of 220 px plus two 16 px gaps is 692 px of content in a 660 px band -
        32 px over, drawn into the footer because a `Column` does not clip. Three cards of
        `height + GAP` pitch, where `height` is the band less the gaps over the count, is the
        shape of the fix. Recomputed from the card height and `GAP`, not read off a neighbour.
        """
        want = card_height(len(self.STREAMS)) + theme.GAP
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            for first, second in (("A", "B"), ("B", "C")):
                got = result.extents[f"card_{second}"][1] - result.extents[f"card_{first}"][1]
                self.assertAlmostEqual(got, want, places=1,
                                       msg=f"{scale}%: card {second} starts {got:.2f} px "
                                           f"below card {first}; recomputed "
                                           f"{card_height(len(self.STREAMS)):.2f} + GAP "
                                           f"{theme.GAP} = {want:.2f}")

    def test_the_card_column_starts_where_the_video_column_ends(self):
        top, _ = main_band()
        want_x = theme.MARGIN + hero_column_width() + theme.GAP
        for scale in capture_ui.SCALES:
            for sid in self.STREAMS:
                got_x = self.result(scale).extents[f"card_{sid}"][0]
                self.assertAlmostEqual(got_x, want_x, places=1,
                                       msg=f"{scale}%: card {sid} starts at {got_x:.1f}; "
                                           f"MARGIN {theme.MARGIN} + round((CANVAS_WIDTH "
                                           f"{theme.CANVAS_WIDTH} - 2 x MARGIN) x "
                                           f"VIDEO_WIDTH_FRACTION "
                                           f"{theme.VIDEO_WIDTH_FRACTION}) = "
                                           f"{hero_column_width()} + GAP {theme.GAP} = "
                                           f"{want_x}")

    def test_a_swapped_divisor_would_not_satisfy_the_geometry_assertions(self):
        """**The negative control, and it is the M1 bug restated.**

        M1's overlay test passed its bounds check while every box was ~400 rows wrong, because
        the geometry divided `y` by `frame.width`. So each of the three assertions above has to
        be shown to **discriminate**: the same measurement, recomputed with the divisors swapped,
        must come out different. An assertion that a swapped recomputation also satisfies is not
        checking anything.
        """
        result = self.result(100)
        card = result.extents["card_C"]
        x, y, w, h = card
        # A swapped divisor on the column: the band's **width** where its height belongs.
        swapped_height = ((theme.CANVAS_WIDTH - theme.MARGIN * 2) - theme.GAP) / 2
        self.assertNotAlmostEqual(h, swapped_height, places=1,
                                  msg="the height assertion would survive a swapped divisor, so "
                                      "it is not testing the division")
        # A swapped divisor on the x position: the width fraction applied to the **height**.
        swapped_x = theme.MARGIN + round(
            (theme.CANVAS_HEIGHT - theme.MARGIN * 2) * theme.VIDEO_WIDTH_FRACTION) + theme.GAP
        self.assertNotAlmostEqual(x, swapped_x, places=1,
                                  msg="the x assertion would survive applying the width "
                                      "fraction to the height")
        # The pitch, with the card **width** where the height belongs - the M1 bug's exact shape.
        swapped_pitch = w + theme.GAP
        real_pitch = result.extents["card_B"][1] - result.extents["card_A"][1]
        self.assertNotAlmostEqual(real_pitch, swapped_pitch, places=1,
                                  msg="the pitch assertion would survive swapping the card's "
                                      "height for its width")
        # And a **width** assertion with a discriminator, on the two thumbnails: they split the
        # video column less one gap. The swapped version divides by the main band's height.
        thumb_w = result.extents["thumb_B"][2]
        want_w = (hero_column_width() - theme.GAP) / 2
        self.assertAlmostEqual(thumb_w, want_w, places=1,
                               msg=f"thumb_B is {thumb_w:.1f} px; recomputed "
                                   f"({hero_column_width()} - {theme.GAP}) / 2 = {want_w:.1f}")
        swapped_w = ((main_band()[1] - main_band()[0]) - theme.GAP) / 2
        self.assertNotAlmostEqual(thumb_w, swapped_w, places=1,
                                  msg="the thumbnail width assertion would survive dividing "
                                      "the band height instead of the video column width")

    # -- the staleness line ------------------------------------------------------------

    def test_the_staleness_line_is_present_and_visible_at_every_scale(self):
        """**It is the number a passer-by has to be able to read.**

        Present, visible, non-degenerate, and in the warning colour's weight - at every scale.
        Its font size is asserted separately, below, because *presence* and *size* are different
        claims and a task that shrank the font to fit would pass this one.
        """
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            for sid in self.STREAMS:
                name = f"streamStaleText_{sid}"
                self.assertIn(name, result.strings, f"{scale}%: {name}")
                self.assertTrue(result.visible.get(name),
                                f"{scale}%: {name} exists but is hidden, which is the same "
                                f"defect as it not existing")
                sentence = result.strings[name]
                self.assertTrue(sentence.startswith("STALE - "),
                                f"{scale}%: camera {sid} must say it in words, not only as a "
                                f"number: {sentence!r}")
                # **The sentence names the same age the headline shows.** If they could
                # disagree, one of them is lying and a reader cannot tell which.
                self.assertIn(result.strings[f"streamAgeText_{sid}"], sentence,
                              f"{scale}%: camera {sid} headline says "
                              f"{result.strings[f'streamAgeText_{sid}']!r} and the sentence says "
                              f"{sentence!r}")
                x, y, w, h = result.extents[name]
                self.assertGreater(h, theme.TYPE_DETAIL,
                                   f"{scale}%: {name} is {h} px tall, which is one line at "
                                   f"{theme.TYPE_DETAIL} px - the staleness sentence is meant "
                                   f"to have {theme.CARD_STALENESS_LINES} lines' worth of room")

    def test_no_font_size_on_the_screen_changed(self):
        """**The claim Task 6 needs, checked rather than asserted.**

        The layout gained room by changing **margins** and by fixing the card height's division.
        No font size was touched, and this reads every named item's `font.pixelSize` out of the
        live scene and compares it with `app/theme.py`'s ramp. If a future change shrinks a
        font to make a layout fit, this fails.
        """
        from tools.capture_ui import load_measured_seed

        seed = load_measured_seed()
        by_id = {entry["stream_id"]: entry for entry in seed["streams"]}
        result = self.result(100)
        # The staleness sentence: unchanged, and the ramp value it must be.
        for sid in self.STREAMS:
            self.assertEqual(result.font_pixels[f"streamStaleText_{sid}"], theme.TYPE_DETAIL,
                             f"the staleness sentence is {result.font_pixels[f'streamStaleText_{sid}']}"
                             f" px; TYPE_DETAIL is {theme.TYPE_DETAIL} and this task changed no "
                             f"font size")
        # The age headline: unchanged.
        for sid in self.STREAMS:
            self.assertEqual(result.font_pixels[f"streamAgeText_{sid}"],
                             theme.TYPE_NUMERAL_SMALL,
                             f"the age headline must stay {theme.TYPE_NUMERAL_SMALL} px")
        # **The whole set of sizes present, and which item holds each.** Not "everything is the
        # detail size" - that was the first version of this test and it was **wrong**: the
        # device badge is `TYPE_LABEL` and the age headlines are `TYPE_NUMERAL_SMALL`, both
        # legitimately. Asserting the *set* and the *holders* is the claim that is actually
        # true, and it still fails if a size or a role changes.
        sizes = {}
        for name, size in result.font_pixels.items():
            sizes.setdefault(size, []).append(name)
        self.assertEqual(sorted(sizes), sorted({theme.TYPE_DETAIL, theme.TYPE_LABEL,
                                                theme.TYPE_NUMERAL_SMALL}),
                         f"the sizes on screen are {sorted(sizes)}; the ramp has "
                         f"{sorted({theme.TYPE_DETAIL, theme.TYPE_LABEL, theme.TYPE_NUMERAL_SMALL})}"
                         f" and this task changed no font size")
        self.assertEqual(sorted(sizes[theme.TYPE_NUMERAL_SMALL]),
                         ["streamAgeText_A", "streamAgeText_B", "streamAgeText_C"],
                         f"the only items at {theme.TYPE_NUMERAL_SMALL} px are the three age "
                         f"headlines: {sorted(sizes[theme.TYPE_NUMERAL_SMALL])}")
        self.assertEqual(sorted(sizes[theme.TYPE_LABEL]), ["deviceBadgeText"],
                         f"the only item at {theme.TYPE_LABEL} px is the device badge: "
                         f"{sorted(sizes[theme.TYPE_LABEL])}")
        # The reason line, which Task 2 re-bound: unchanged at the detail size.
        self.assertEqual(result.font_pixels["streamReasonText_C"], theme.TYPE_DETAIL)
        # And the items Task 2 did not touch but which sit in the same columns.
        for name in ("streamWindowText_A", "streamBacklogText_A", "streamRateText_A",
                     "streamBadgeText_A", "cardText_system_cpu_percent",
                     "cardWindowNote_system_cpu_percent", "tileNoFrameText_A"):
            self.assertEqual(result.font_pixels.get(name), theme.TYPE_DETAIL,
                             f"{name} is {result.font_pixels.get(name)} px, expected "
                             f"{theme.TYPE_DETAIL}")

    # -- the failure reason ------------------------------------------------------------

    def test_the_failed_cameras_reason_says_how_much_it_is_not_showing(self):
        from tools.capture_ui import load_measured_seed

        seed = load_measured_seed()
        reason = seed["failed_reason"]
        hidden = len(reason) - theme.CARD_REASON_CHARS
        marker = theme.CARD_REASON_MARKER.format(hidden=hidden)
        for scale in capture_ui.SCALES:
            shown = self.result(scale).strings["streamReasonText_C"]
            self.assertTrue(shown.endswith(marker),
                            f"{scale}%: the visible reason must end with {marker!r}; it ends "
                            f"{shown[-40:]!r}")
            self.assertIn(str(hidden), shown,
                          f"{scale}%: the number of hidden characters must be on screen: "
                          f"{shown[-40:]!r}")

    def test_a_healthy_camera_has_no_reason_and_is_not_given_one(self):
        # **The cut is only for a camera that has something to say.** Cameras A and B render an
        # empty reason line, and an empty line that reads as a message is its own defect.
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            for sid in ("A", "B"):
                self.assertEqual(result.strings[f"streamReasonText_{sid}"], "",
                                 f"{scale}%: camera {sid} is healthy and must have no reason: "
                                 f"{result.strings[f'streamReasonText_{sid}']!r}")

    def test_a_camera_with_a_device_and_a_long_reason_still_fits_its_card(self):
        """**The worst case, which the four committed captures do not contain.**

        In the captures, the failed camera has **no confirmed device**, so its right column
        shows **four** rows and has **67.3 px** of slack (measured: the reason ends at 744.67 and
        the column's inner bottom is 812.0). The case that actually stresses the layout is a
        camera that **fell back successfully**: it has a confirmed device, so `cardText_` and
        `cardWindowNote_` appear, *and* it has a reason from the transition - **six** rows in the
        column where the reason is the last of them.

        This seeds that shape explicitly rather than hoping the booth produces it. It is the
        ruling's "three streams with 11-14 s frame ages and a 2331-character failure reason at
        once", with the extra device rows the ruling does not mention and that the booth will
        still show whenever a route settles on a fallback.
        """
        seed = capture_ui.load_measured_seed()
        streams = [dict(entry) for entry in seed["streams"]]
        streams[1]["confirmed_devices"] = ["GPU.0"]
        streams[1]["state"] = "active"
        streams[1]["fell_back"] = True
        streams[1]["errors"] = [seed["failed_reason"]]
        seeded = {"streams": streams, "failed_stream": "C",
                  "failed_reason": seed["failed_reason"],
                  "provenance": seed["provenance"]}
        for scale in capture_ui.SCALES:
            with tempfile.TemporaryDirectory(prefix="capture-worst-") as tmp:
                result = capture_ui.capture(scale_percent=scale, seconds=10.0,
                                            seed_frames=seeded, out_dir=pathlib.Path(tmp))
                # **The six rows are what this test is for.** Asserted first, because a test
                # that passes over four rows is a test that never reached the hard case - and
                # that is the M2 failure mode exactly: a passing assertion over a result that
                # was not the one intended.
                #
                # The metric key is the bridge's, not the device's: a GPU has
                # `gpu.utilisation.percent` and no `gpu_0_percent`, so these are the two rows
                # a confirmed non-CPU device adds. **Measured with all six present**: the
                # reason ends at 543.33 and the column's inner bottom is 586.67, so the six-row
                # case has **43.3 px** of slack - against **67.3 px** for the four-row case the
                # committed captures show. Fits, but by half.
                self.assertEqual(result.strings.get("cardText_gpu_utilisation_percent"),
                                 "Utilization unavailable",
                                 f"{scale}%: the device's own rows must be on the card for "
                                 f"this to be the six-row case. Got "
                                 f"{result.strings.get('cardText_gpu_utilisation_percent')!r}")
                self.assertIn("cardWindowNote_gpu_utilisation_percent", result.extents,
                              f"{scale}%: and the window note too: "
                              f"{sorted(n for n in result.extents if n.startswith('cardWindow'))}")
                self.assertTrue(result.strings["streamReasonText_B"].endswith("more"),
                                f"{scale}%: camera B must be carrying a cut reason: "
                                f"{result.strings['streamReasonText_B']!r}")
                # **The reason is the LAST row, so if it fits, the column fits.** Recomputed
                # from the measured extents rather than trusted from the overflow list alone.
                reason_bottom = (result.extents["streamReasonText_B"][1]
                                 + result.extents["streamReasonText_B"][3])
                column_bottom = (result.extents["card_B"][1] + result.extents["card_B"][3]
                                 - theme.SPACING["sm"])
                self.assertLessEqual(reason_bottom, column_bottom + 0.5,
                                     f"{scale}%: the six-row reason ends at {reason_bottom:.2f} "
                                     f"and the card's inner bottom is {column_bottom:.2f}")
                self.assertEqual(result.overflow, (),
                                 f"{scale}%: a camera with a device AND a long reason "
                                 f"overflows: {list(result.overflow)}")
                self.assertEqual(result.truncated, (),
                                 f"{scale}%: Qt still had to elide something: "
                                 f"{list(result.truncated)}")
                # **And the card is what is being measured against.** Without this,
                # `overflow == ()` also passes when the card has stopped clipping - which was
                # a surviving mutant, and a serious one: the measure would silently go back
                # to being window-only while the assertion kept passing.
                self.assertIn(
                    f"ancestor:card_B", result.clip_sources.values(),
                    f"{scale}%: camera B's card was not used as a clip, so the empty "
                    f"overflow list means nothing: {sorted(set(result.clip_sources.values()))}")
                card = result.extents["card_B"]
                right = result.extents["streamReasonText_B"]
                self.assertLessEqual(right[1] + right[3], card[1] + card[3] + 0.5,
                                     f"{scale}%: camera B's reason ends at "
                                     f"{right[1] + right[3]:.1f} and its card at "
                                     f"{card[1] + card[3]:.1f}")


class TheCutIsMadeInPythonNotByQt(unittest.TestCase):
    """The record the card shows, independent of any render."""

    def _bridge(self):
        from PySide6.QtCore import QCoreApplication

        app = QCoreApplication.instance() or QCoreApplication([])
        from app.ui.bridge import UiBridge

        return UiBridge()

    def test_a_short_reason_is_untouched(self):
        bridge = self._bridge()
        bridge.set_streams(["A"])
        bridge.apply_route({"stream": "A", "requested": "CPU", "confirmed": ["CPU"],
                            "state": "active", "fell_back": False,
                            "errors": ["a short stated reason"]})
        record = bridge.stream_cards[0]
        self.assertEqual(record["reason_display"], "a short stated reason")
        self.assertEqual(record["reason_display"], record["reason"])

    def test_a_long_reason_is_cut_with_the_number_that_is_missing(self):
        bridge = self._bridge()
        bridge.set_streams(["C"])
        reason = "x" * 500
        bridge.apply_route({"stream": "C", "requested": "NPU", "confirmed": [],
                            "state": "failed", "fell_back": False, "errors": [reason]})
        record = bridge.stream_cards[0]
        self.assertEqual(len(record["reason"]), 500, "the complete reason stays complete")
        self.assertEqual(
            record["reason_display"],
            reason[:theme.CARD_REASON_CHARS].rstrip()
            + theme.CARD_REASON_MARKER.format(hidden=500 - theme.CARD_REASON_CHARS))
        self.assertLess(len(record["reason_display"]), len(record["reason"]))

    def test_a_reason_exactly_at_the_budget_is_not_marked_as_cut(self):
        # **Off-by-one, and it is the difference between a marker that means something and one
        # that lies.** At exactly the budget nothing is hidden, so a marker would be a false
        # claim; one character over, one character is hidden and the marker must appear.
        #
        # **The boundary is written out rather than read from the token.** The first version
        # built the boundary from `theme.CARD_REASON_CHARS`, so raising the token moved the
        # boundary with it and the test passed at 41 as well as at 40 - a surviving mutant, and
        # one that would have let the budget drift upward unnoticed every time a string grew.
        self.assertEqual(theme.CARD_REASON_CHARS, 40,
                         "the declared character budget. It is a layout number other things "
                         "depend on - the reason's two-line capacity and the marker length - so "
                         "it is pinned here rather than read back from itself.")
        bridge = self._bridge()
        bridge.set_streams(["A"])
        exact = "y" * 40
        bridge.apply_route({"stream": "A", "requested": "CPU", "confirmed": ["CPU"],
                            "state": "active", "fell_back": False, "errors": [exact]})
        self.assertEqual(bridge.stream_cards[0]["reason_display"], exact,
                         "40 characters is the whole budget: nothing is hidden, so nothing "
                         "may be marked hidden")
        over = "y" * 41
        bridge.apply_route({"stream": "A", "requested": "CPU", "confirmed": ["CPU"],
                            "state": "active", "fell_back": False, "errors": [over]})
        self.assertTrue(bridge.stream_cards[0]["reason_display"].endswith(
            theme.CARD_REASON_MARKER.format(hidden=1)),
            bridge.stream_cards[0]["reason_display"])


class TheSeedToRouteRule(unittest.TestCase):
    """**The rule that built the busiest state, pinned.**

    It shipped a bug in this task's first version: the failed stream was rendered **`active`**
    with no confirmed device, because `entry.get("state", "failed")` always finds a `state` and
    the committed steady-state measurement records all three cameras as `active`. The capture
    still exited 0 and still reported plausible strings - a capture has no opinion about whether
    the state it was handed is a state the booth can be in.

    So the rule is a function, and these are its assertions.
    """

    def setUp(self):
        self.seed = capture_ui.load_measured_seed()
        self.routes = capture_ui.routes_from_seed(self.seed)

    def test_the_default_seed_fails_exactly_one_camera(self):
        failed = [r for r in self.routes if r["state"] == "failed"]
        self.assertEqual(len(failed), 1,
                         f"one camera, not {len(failed)}: "
                         f"{[(r['stream'], r['state']) for r in self.routes]}")
        self.assertEqual(failed[0]["stream"], self.seed["failed_stream"])

    def test_the_failed_camera_claims_no_device_and_carries_the_committed_reason(self):
        route = capture_ui.routes_from_seed(self.seed)[-1]
        self.assertEqual(route["confirmed"], [],
                         f"a failed route claims no device, and certainly not the requested "
                         f"one: {route['confirmed']}")
        self.assertEqual(route["errors"], [self.seed["failed_reason"]],
                         "the stated reason is the committed supervisor message, verbatim")

    def test_the_other_cameras_keep_their_measured_state_and_device(self):
        for route in self.routes:
            if route["stream"] == self.seed["failed_stream"]:
                continue
            entry = next(e for e in self.seed["streams"]
                         if e["stream_id"] == route["stream"])
            self.assertEqual(route["state"], entry["state"],
                             f"camera {route['stream']} keeps the measured state")
            self.assertEqual(route["confirmed"], entry["confirmed_devices"])
            self.assertEqual(route["errors"], [],
                             f"camera {route['stream']} is healthy and states no reason")

    def test_a_state_cannot_be_overridden_without_saying_so(self):
        """**`seed_state`, not `state`.** A caller that writes `state` on a seed entry is
        writing over a *measured* field, and this rule ignores it - deliberately, so the
        measured value and the chosen one can never be confused.
        """
        seed = json.loads(json.dumps(self.seed))
        victim = next(e for e in seed["streams"]
                      if e["stream_id"] == seed["failed_stream"])
        victim["state"] = "active"
        routes = capture_ui.routes_from_seed(seed)
        self.assertEqual(routes[-1]["state"], "failed",
                         "writing `state` on the failed stream must not resurrect it; the "
                         "field the rule reads for an override is `seed_state`")
        victim["seed_state"] = "pending"
        self.assertEqual(capture_ui.routes_from_seed(seed)[-1]["state"], "pending",
                         "`seed_state` is the override, and it is a different name on purpose")

    def test_the_worst_case_adds_a_reason_to_a_camera_that_has_a_device(self):
        seed = json.loads(json.dumps(self.seed))
        healthy = next(e for e in seed["streams"] if e["stream_id"] != seed["failed_stream"])
        healthy["errors"] = [self.seed["failed_reason"]]
        route = next(r for r in capture_ui.routes_from_seed(seed)
                     if r["stream"] == healthy["stream_id"])
        self.assertTrue(route["confirmed"], "and it keeps its device, which is the whole point")
        self.assertEqual(len(route["errors"]), 1)
        self.assertEqual(route["state"], "active",
                         "a camera with a device and a stated reason is not a failed route")

    def test_the_child_and_the_module_share_one_definition(self):
        self.assertEqual(capture_ui._CHILD.count("def routes_from_seed"), 1,
                         "the child must paste the rule in exactly once")
        self.assertIn("_ROUTES_FROM_SEED_SOURCE", capture_ui._CHILD,
                      "and it must be the shared source, not a second copy that can drift")
        compile(capture_ui._CHILD.replace("@@REPO@@", repr(str(REPO_ROOT))),
                "<child>", "exec")


class ClippingIsTheDecision(unittest.TestCase):
    """The `clip: true` decision, pinned so it cannot be quietly undone."""

    def test_the_card_the_two_columns_and_the_tiles_all_clip(self):
        sources = {
            "app/ui/ComputeCard.qml": ["clip: true"],
            "app/ui/VideoTile.qml": ["clip: true"],
        }
        for relative, needles in sources.items():
            text = (REPO_ROOT / relative).read_text(encoding="utf-8")
            for needle in needles:
                self.assertIn(needle, text,
                              f"{relative} must keep {needle!r}: it is the decision that makes "
                              f"`capture_ui`'s overflow a per-card measure instead of a "
                              f"window-only one")
        # **And the columns are named**, because an unnamed clipping ancestor makes the overflow
        # report say `ancestor:` and a reader cannot tell which box overflowed.
        card = (REPO_ROOT / "app/ui/ComputeCard.qml").read_text(encoding="utf-8")
        self.assertIn('objectName: "cardLeftColumn_" + card.streamId', card)
        self.assertIn('objectName: "cardRightColumn_" + card.streamId', card)
        # **Counting the line, not the substring.** The first version counted `card.count(
        # "clip: true")` and got **6**, because three of those are inside the comment that
        # explains the decision. A count that includes prose is a count of prose.
        code_lines = [line.strip() for line in card.split("\n")]
        self.assertEqual(sum(1 for line in code_lines if line == "clip: true"), 3,
                         "the card, the left column and the right column: three lines of "
                         "`clip: true`, and no more")

    def test_the_card_height_divides_out_the_gaps_rather_than_the_plain_band(self):
        text = (REPO_ROOT / "app/ui/Main.qml").read_text(encoding="utf-8")
        self.assertIn("Theme.GAP * Math.max(0, bridge.stream_ids.length - 1)", text,
                      "the card height must take the Column's spacing out of the band. Without "
                      "it, N cards claim the band plus N-1 gaps and the last card draws into "
                      "the footer - 32 px, measured.")
        self.assertIn("Math.max(1, bridge.stream_ids.length)", text,
                      "and the divisor stays guarded against a zero-length model, because "
                      "`Math.max` passes NaN through and a zero-height delegate is invisible "
                      "rather than wrong-looking")

    def test_the_layout_tokens_are_in_theme_and_not_in_the_qml(self):
        """**Code, not comments.** See `_code_only`: the first version of this test searched the
        whole file and found `0.42` in the comment explaining where `0.42` went.
        """
        card = _code_only(REPO_ROOT / "app/ui/ComputeCard.qml")
        for literal in ("0.42", "0.58", "maximumLineCount: 2", "maximumLineCount: 3",
                        "TYPE_NUMERAL_SMALL * 4"):
            self.assertNotIn(literal, card,
                             f"{literal!r} is a layout number and must live in "
                             f"app/theme.py, not in a QML binding")
        for token in ("CARD_COLUMN_SPLIT", "CARD_STALENESS_LINES", "CARD_REASON_LINES",
                      "CARD_ACTIVITY_LINES", "CARD_COLUMN_MIN_EM"):
            self.assertIn(f"Theme.{token}", card,
                          f"Theme.{token} must be what the card binds to")
            self.assertTrue(hasattr(theme, token),
                            f"app/theme.py must define {token}, or the binding reads undefined")
        tile = (REPO_ROOT / "app/ui/VideoTile.qml").read_text(encoding="utf-8")
        self.assertNotIn("0.16", tile,
                         "the thumbnail row's height fraction is a layout number and belongs in "
                         "app/theme.py; it is still inline and that is recorded, not fixed here")


if __name__ == "__main__":
    unittest.main()