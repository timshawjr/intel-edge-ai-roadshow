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
import math
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
                # **M4: the card's three rows and the footer's verdict line.** The four
                # rows that moved into the presenter drawer (`streamWindowText_*`,
                # `streamBacklogText_*`, `streamRateText_*`, `streamActivityText_*`) are
                # asserted on the drawer surface, in
                # `tests/test_ui_contract.py::PresenterDrawerOnTheScreen`, because they do
                # not exist on the always-visible screen any more.
                for name in (f"card_{sid}", f"cameraName_{sid}", f"cameraState_{sid}",
                             f"streamAgeText_{sid}", f"streamBadgeBox_{sid}",
                             f"streamBadgeText_{sid}", f"verdictRow_{sid}",
                             f"streamStaleText_{sid}", f"streamReasonText_{sid}"):
                    self.assertIn(name, result.extents,
                                  f"{scale}%: {name} must be reached **and measured**. The "
                                  f"cards are `Repeater` delegates, invisible to `findChild`, "
                                  f"so this is the assertion a `findChild`-based reader could "
                                  f"not make.")
                    x, y, w, h = result.extents[name]
                    # **The verdict pair are alternatives**: a camera states its staleness
                    # sentence or its failure reason, and the other item is present and
                    # zero-width. A zero width is degenerate *for a visible item*, so the
                    # width check applies to the one that is visible.
                    if name.startswith(("streamStaleText_", "streamReasonText_")):
                        if not result.visible.get(name):
                            self.assertEqual(w, 0.0,
                                             f"{scale}%: {name} is hidden and still occupies "
                                             f"{w} px, so the two verdicts are not exclusive")
                            continue
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
                # **One name for one value.** The first version wrote `got = result_h = ...`,
                # which bound two names to the same number and left `result_h` dead: a reader
                # could not tell which of the two the assertion was about, and renaming one
                # later would have silently changed nothing.
                got = self.result(scale).extents[f"card_{sid}"][3]
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

    def test_every_camera_states_its_verdict_at_every_scale(self):
        """**It is the number a passer-by has to be able to read.**

        Present, visible, non-degenerate - at every scale. **M4 moved the sentence out of the
        card and into the footer**, because at `TYPE_BOOTH_DETAIL` it is 918 px wide and the
        card's inner width is 567: on the card it wrapped to two lines and pushed the card past
        its band. So the sentence now names its camera and carries the same age the numeral
        shows, and a camera with a **stated failure reason** shows that reason instead - one
        verdict line per camera is the budget, and the reason is the more specific statement.
        """
        for scale in capture_ui.SCALES:
            result = self.result(scale)
            for sid in self.STREAMS:
                stale = f"streamStaleText_{sid}"
                reason = f"streamReasonText_{sid}"
                self.assertIn(stale, result.strings, f"{scale}%: {stale}")
                self.assertIn(reason, result.strings, f"{scale}%: {reason}")
                # **Exactly one of the two is visible**, so a camera cannot show two verdicts
                # in a one-line budget, and cannot show none.
                self.assertNotEqual(result.visible.get(stale), result.visible.get(reason),
                                    f"{scale}%: camera {sid} has both verdicts "
                                    f"{'visible' if result.visible.get(stale) else 'hidden'} - "
                                    f"the footer budget is one line per camera")
                if result.visible.get(stale):
                    sentence = result.strings[stale]
                    self.assertTrue(sentence.startswith(f"Camera {sid}: STALE - "),
                                    f"{scale}%: camera {sid} must say it in words, and say "
                                    f"which camera: {sentence!r}")
                    # **The sentence names the same age the headline shows.** If they could
                    # disagree, one of them is lying and a reader cannot tell which.
                    self.assertIn(result.strings[f"streamAgeText_{sid}"], sentence,
                                  f"{scale}%: camera {sid} headline says "
                                  f"{result.strings[f'streamAgeText_{sid}']!r} and the "
                                  f"sentence says {sentence!r}")
                else:
                    self.assertTrue(result.strings[reason].startswith(f"Camera {sid}: "),
                                    f"{scale}%: camera {sid}'s reason line must say which "
                                    f"camera it is about: {result.strings[reason]!r}")
                x, y, w, h = result.extents[stale if result.visible.get(stale) else reason]
                self.assertGreater(w, 1.0, f"{scale}%: the verdict line has no width")
                self.assertGreaterEqual(h, theme.TYPE_BOOTH_DETAIL,
                                        f"{scale}%: the verdict line is {h} px tall, less than "
                                        f"one line at {theme.TYPE_BOOTH_DETAIL} px")

    def test_no_font_size_on_the_screen_is_below_the_legibility_ladder(self):
        """**The claim M4 makes, checked rather than asserted.**

        M3's version of this test asserted that no size had changed, because that task changed
        nothing but margins. This task changes the type on purpose, so the claim is the one that
        matters instead: **every visible text on the always-visible screen is at or above the
        ladder**, the age is the numeral size, the camera names are the label size, and nothing
        outside those three sizes is drawn. A future change that shrinks a font to make a layout
        fit fails here rather than passing quietly - which is exactly how M3's screen ended up
        unreadable while every geometry assertion was green.
        """
        result = self.result(100)
        sizes = {}
        for name, size in result.font_pixels.items():
            # **Visible only.** A hidden item draws nothing, and counting it would put the
            # reply to "what is on the screen" in the hands of an item that is not.
            if result.visible.get(name) and result.strings.get(name, ""):
                sizes.setdefault(int(size), []).append(name)
        self.assertEqual(sorted(sizes),
                         sorted({theme.TYPE_BOOTH_DETAIL, theme.TYPE_BOOTH_LABEL,
                                 theme.TYPE_BOOTH_NUMERAL}),
                         f"the sizes drawn on the screen are {sorted(sizes)}; the booth ladder "
                         f"is {sorted({theme.TYPE_BOOTH_DETAIL, theme.TYPE_BOOTH_LABEL, theme.TYPE_BOOTH_NUMERAL})}")
        self.assertEqual(sorted(sizes[theme.TYPE_BOOTH_NUMERAL]),
                         ["streamAgeText_A", "streamAgeText_B", "streamAgeText_C"],
                         f"the only items at {theme.TYPE_BOOTH_NUMERAL} px are the three age "
                         f"headlines: {sorted(sizes[theme.TYPE_BOOTH_NUMERAL])}")
        self.assertEqual(sorted(sizes[theme.TYPE_BOOTH_LABEL]),
                         ["cameraName_A", "cameraName_B", "cameraName_C"],
                         f"the only items at {theme.TYPE_BOOTH_LABEL} px are the three camera "
                         f"names: {sorted(sizes[theme.TYPE_BOOTH_LABEL])}")
        # **The floor, stated as a floor**: no visible text anywhere on the always-visible
        # screen is smaller than `TYPE_BOOTH_DETAIL`.
        below = {name: size for name, size in result.font_pixels.items()
                 if result.visible.get(name) and result.strings.get(name, "")
                 and int(size) < theme.TYPE_BOOTH_DETAIL}
        self.assertEqual(below, {},
                         f"these visible items are below the ladder: {below}")

    # -- the failure reason ------------------------------------------------------------

    def test_the_font_census_is_the_one_on_record(self):
        """**The number the M4 report quotes, recomputed from the committed capture.**

        `{44: 18, 48: 3, 56: 3}` at the time of writing - the size of every `Text` **whose
        text is not empty and which is visible**, so a font that draws nothing is not counted
        and the number does not depend on which optional lines happened to be blank or hidden.
        Two claims, both measured: the committed capture is that census, and a live render of
        the same seed still is. A change that makes something fit by shrinking a font fails
        here rather than passing quietly.
        """
        committed_path = (REPO_ROOT / "docs" / "qualification" / "m3-capture" / "100pct.json")
        self.assertTrue(committed_path.exists(), f"no committed capture at {committed_path}")
        committed = json.loads(committed_path.read_text(encoding="utf-8"))

        def census(strings, fonts, visible):
            counts = {}
            for name, size in fonts.items():
                if strings.get(name) and visible.get(name):
                    counts[int(size)] = counts.get(int(size), 0) + 1
            return counts

        want = census(committed["strings"], committed["font_pixels"], committed["visible"])
        self.assertEqual(want, {theme.TYPE_BOOTH_DETAIL: 18,
                                theme.TYPE_BOOTH_LABEL: 3,
                                theme.TYPE_BOOTH_NUMERAL: 3},
                         f"the committed capture's census is {want}, not the one the report "
                         f"states; re-derive the report's number or fix the layout")
        got = census(self.result(100).strings, self.result(100).font_pixels,
                     self.result(100).visible)
        self.assertEqual(got, want,
                         f"a live render of the same seed now censuses {got}, and the "
                         f"committed capture says {want}: something changed a font size or "
                         f"made a line draw text it did not draw before")

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

    def test_a_camera_with_a_device_and_a_long_reason_still_fits_the_screen(self):
        """**The worst case, which the four committed captures do not contain.**

        A camera that **fell back successfully** has a confirmed device *and* a reason from the
        transition, and its metrics make it stale as well. M3 stressed the card with six rows
        for this shape. M4's card has three rows and the sentences are in the footer, so what
        this shape can now break is different, and it is what this test asserts:

        - the card's three rows still fit a 209.33 px card (the fallback badge is the longest
          badge string there is: `GPU.0 (fell back from CPU)`);
        - the footer's one-line budget still holds, because the camera shows its **reason**
          rather than its staleness sentence;
        - nothing overflows and nothing is elided, at every scale.

        It seeds the shape explicitly rather than hoping the booth produces it.
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
                # **The shape is asserted first**, because a test that passes over the
                # ordinary state is a test that never reached the hard case.
                self.assertEqual(result.strings.get("cameraName_B"), "Camera B",
                                 f"{scale}%: camera B must be on screen for this to be the "
                                 f"case at all: {result.strings.get('cameraName_B')!r}")
                self.assertIn("fell back from",
                              result.strings.get("streamBadgeText_B", ""),
                              f"{scale}%: the fallback badge is the longest badge and it must "
                              f"be the one under test: "
                              f"{result.strings.get('streamBadgeText_B')!r}")
                # **The reason wins the one verdict line**, and it says which camera it is.
                self.assertTrue(result.visible.get("streamReasonText_B"),
                                f"{scale}%: camera B has a stated reason, so its verdict line "
                                f"is the reason - the staleness sentence is the other branch")
                self.assertFalse(result.visible.get("streamStaleText_B"),
                                 f"{scale}%: both verdicts are visible, which the one-line "
                                 f"budget cannot hold")
                self.assertTrue(result.strings["streamReasonText_B"].startswith("Camera B: "),
                                f"{scale}%: and it names the camera: "
                                f"{result.strings['streamReasonText_B']!r}")
                self.assertTrue(result.strings["streamReasonText_B"].endswith("more"),
                                f"{scale}%: camera B must be carrying a cut reason: "
                                f"{result.strings['streamReasonText_B']!r}")
                # **The card's own inner box is the bound**, measured by Qt rather than
                # recomputed from the token the margin binds to - an expectation derived from
                # the thing under test moves with it.
                column = result.extents["cardColumn_B"]
                self.assertEqual(result.clip_sources["streamBadgeBox_B"],
                                 "ancestor:cardColumn_B",
                                 f"{scale}%: the badge's clip must be the card's column for its "
                                 f"extent to be the bound; it is "
                                 f"{result.clip_sources.get('streamBadgeBox_B')!r}")
                column_bottom = column[1] + column[3]
                badge_bottom = (result.extents["streamBadgeBox_B"][1]
                                + result.extents["streamBadgeBox_B"][3])
                self.assertLessEqual(badge_bottom, column_bottom + 0.5,
                                     f"{scale}%: the fallback badge ends at {badge_bottom:.2f} "
                                     f"and the measured card column ends at {column_bottom:.2f}")
                self.assertGreater(column_bottom - badge_bottom, 0.0,
                                   f"{scale}%: a card whose last row ends exactly on its "
                                   f"column's bottom has no margin left for a taller badge")
                self.assertEqual(result.overflow, (),
                                 f"{scale}%: a camera with a device AND a long reason "
                                 f"overflows: {list(result.overflow)}")
                self.assertEqual(result.truncated, (),
                                 f"{scale}%: Qt still had to elide something: "
                                 f"{list(result.truncated)}")
                # **And the card is what is being measured against.** Without this,
                # `overflow == ()` also passes when the card has stopped clipping - which was
                # a surviving mutant: the measure would silently go back to window-only while
                # the assertion kept passing.
                self.assertIn(
                    "ancestor:card_B", result.clip_sources.values(),
                    f"{scale}%: camera B's card was not used as a clip, so the empty "
                    f"overflow list means nothing: {sorted(set(result.clip_sources.values()))}")


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
        # **Asserting the whole source string, not the identifier.** `assertIn(
        # "_ROUTES_FROM_SEED_SOURCE", _CHILD)` was satisfied by the **comment** above the paste -
        # which names the constant - and would have passed with the paste deleted. This asserts
        # the rule's own text is in the child, verbatim, which is the claim being made.
        self.assertIn(capture_ui._ROUTES_FROM_SEED_SOURCE, capture_ui._CHILD,
                      "the child must contain the shared rule verbatim, not a comment about it")
        compile(capture_ui._CHILD.replace("@@REPO@@", repr(str(REPO_ROOT)))
                .replace("@@AUDIT_FLAG@@", repr(capture_ui.READ_AUDIT)),
                "<child>", "exec")


class ClippingIsTheDecision(unittest.TestCase):
    """The `clip: true` decision, pinned so it cannot be quietly undone."""

    def test_the_card_its_column_and_the_tiles_all_clip(self):
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
        # **And the column is named**, because an unnamed clipping ancestor makes the overflow
        # report say `ancestor:` and a reader cannot tell which box overflowed. **M4: one card
        # column, not two** - at the legibility ladder the card is name/state, age, device, and
        # a 42/58 split put `no device is running` in a 245 px box where it elided.
        card = (REPO_ROOT / "app/ui/ComputeCard.qml").read_text(encoding="utf-8")
        self.assertIn('objectName: "cardColumn_" + card.streamId', card)
        self.assertNotIn("cardLeftColumn_", card,
                         "the two-column card is gone: at 44 px the right column could not hold "
                         "`no device is running` (387 px) in 245 px")
        # **Counting the line, not the substring.** The first version counted `card.count(
        # "clip: true")` and got **6**, because three of those are inside the comment that
        # explains the decision. A count that includes prose is a count of prose.
        code_lines = [line.strip() for line in card.split("\n")]
        self.assertEqual(sum(1 for line in code_lines if line == "clip: true"), 2,
                         "the card and its column: two lines of `clip: true`, and no more")

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

    def test_a_type_size_as_the_height_floor_reintroduces_the_overrun_at_ten_cameras(self):
        """**The same 32 px defect, coming back through the floor.**

        Task 2 wrote `Math.max(Theme.TYPE_NUMERAL, (band - GAP*(n-1)) / n)`. The floor exists to
        stop a zero or negative height, and at three cameras it never engages. **At ten cameras
        the band cannot give each card 56 px**: `(660 - 144) / 10 = 51.6`, the floor raises all
        ten to 56, and the column claims **704 px in a 660 px band - 44 px into the footer**,
        drawn anyway, because a `Column` does not clip. That is the defect this task fixed,
        returning through the guard that was meant to prevent it.

        So the floor is `0`, and this asserts both halves: the code no longer floors with a type
        size, **and** the arithmetic it does use cannot overrun the band at any camera count from
        one to sixteen. The second half is computed here from `app/theme.py`, independently of
        the QML, and it is a negative control in the same test: the old expression is evaluated
        alongside it and shown to overrun.
        """
        code = _code_only(REPO_ROOT / "app/ui/Main.qml")
        self.assertNotIn("Math.max(Theme.TYPE_NUMERAL", code,
                         "a type size must not be the card height's floor: it raises every card "
                         "to 56 px and the column then overruns its band at any camera count the "
                         "band cannot supply that for (ten cameras: 704 px in 660).")
        self.assertIn("height: Math.max(0,", code,
                      "the floor's only job is to stop a negative height, and 0 does that")

        band = main_band()[1] - main_band()[0]

        def gaps(count):
            return theme.GAP * max(0, count - 1)

        for count in range(1, 17):
            height = max(0, (band - gaps(count)) / max(1, count))
            self.assertLessEqual(
                height * count + gaps(count), band + 0.5,
                f"{count} cards: {height * count + gaps(count):.2f} px of content in a "
                f"{band:.2f} px band. The expression under test cannot overrun the band at any "
                f"camera count.")
        # **The negative control, so the assertion above is not vacuous:** the floor Task 2 wrote
        # does overrun, and by more than the 32 px this task originally fixed.
        for count in (10, 12, 16):
            floored = max(theme.TYPE_NUMERAL, (band - gaps(count)) / count)
            self.assertGreater(
                floored * count + gaps(count), band,
                f"{count} cards with a {theme.TYPE_NUMERAL} px floor claim "
                f"{floored * count + gaps(count):.2f} px in a {band:.2f} px band - the "
                f"negative control for the assertion above")

    def test_the_layout_tokens_are_in_theme_and_not_in_the_qml(self):
        """**Code, not comments.** See `_code_only`: the first version of this test searched the
        whole file and found `0.42` in the comment explaining where `0.42` went.

        **M4: the card's tokens changed with the card.** It no longer splits into two columns,
        no longer wraps a sentence into two lines on the card, and its rows are the legibility
        ladder - so the tokens it must bind to are the ladder and the badge padding, and the
        four M3 card tokens are gone from `app/theme.py` because nothing binds to them.
        """
        card = _code_only(REPO_ROOT / "app/ui/ComputeCard.qml")
        for literal in ("0.42", "0.58", "maximumLineCount: 2", "maximumLineCount: 3",
                        "TYPE_NUMERAL_SMALL * 4"):
            self.assertNotIn(literal, card,
                             f"{literal!r} is a layout number and must live in "
                             f"app/theme.py, not in a QML binding")
        for token in ("TYPE_BOOTH_DETAIL", "TYPE_BOOTH_LABEL", "TYPE_BOOTH_NUMERAL",
                      "CARD_BADGE_INSET"):
            self.assertIn(f"Theme.{token}", card,
                          f"Theme.{token} must be what the card binds to")
            self.assertTrue(hasattr(theme, token),
                            f"app/theme.py must define {token}, or the binding reads undefined")
        # **And the tokens M3 pinned are gone, not orphaned.** A token nothing binds to is a
        # description of a layout that no longer exists.
        for gone in ("CARD_COLUMN_SPLIT", "CARD_COLUMN_MIN_EM", "CARD_STALENESS_LINES",
                     "CARD_REASON_LINES"):
            self.assertFalse(hasattr(theme, gone),
                             f"app/theme.py still defines {gone}, which nothing binds to "
                             f"since M4's card is one column of three rows")
        tile = _code_only(REPO_ROOT / "app/ui/VideoTile.qml")
        self.assertNotIn("0.16", tile,
                         "the thumbnail row's height fraction is a layout number and belongs in "
                         "app/theme.py; M4 moved it there as THUMB_ROW_FRACTION")
        self.assertIn("Theme.THUMB_ROW_FRACTION",
                      _code_only(REPO_ROOT / "app/ui/Main.qml"),
                      "and Main.qml must be what binds to it")


# ======================================================================================
# M4 Task 1: the booth screen at three metres
# ======================================================================================
#
# `CanvasLayoutAtFourScales` answers "is the layout inside the canvas and where the tokens
# say". It says nothing about whether a person standing three metres away can read the
# numbers, and M3's readability review showed why the distinction matters: **the layout was
# perfect and the screen was unreadable.** The four committed captures passed every geometry
# assertion while every element at `TYPE_DETAIL` subtended under 10 arcmin at 3 m.
#
# What "measured" means here, and what it deliberately does not mean:
#
# - **The font size is read out of the running scene graph** (`font_pixels`), not from
#   `app/theme.py`. A token check proves the token was applied; it does not prove the text
#   was drawn at that size, on that background, in that box.
# - **The cap height and x-height come from the font engine, at the size the scene
#   reports.** `QFontMetricsF.capHeight()` on this machine's registered Arial is 0.7164 em
#   and `xHeight()` is 0.5184 em; both are re-measured here and asserted against
#   `app/theme.py`, so the constants cannot drift away from the font that is rasterised.
# - **The ink is measured in the saved PNG**, inside each item's own reported extent. That
#   is the step that stops this being a property check: an item whose text is invisible,
#   blank, or drawn smaller than the property admits has no ink band of the right height.
# - The angle is `2 atan(h / 2d)` at the **measured** geometry, and there are two of them:
#   the monitor attached to this machine (`theme.ACTUAL_MONITOR_MM_PER_LOGICAL_PX`,
#   0.310 mm/logical px, at 2 m and 3 m) is the **gating** case; the 55-inch panel
#   (`theme.MM_PER_LOGICAL_PX`, 0.634 mm/logical px, at 3 m) is the **procurement
#   requirement** the booth must meet, and is asserted separately and labelled as such.
#   A run that only asserts the 55-inch case is not a statement about the delivered screen.
#
# The non-vacuity guards, because each of these has already happened in this repository:
# an absent item reads `""`; a `Repeater` delegate is invisible to `findChild`; `Math.max`
# passes `NaN` through; and a `1x1` window makes every click land on nothing while the tests
# pass. And **the element set is not a list this test keeps** - it is *every visible named
# `Text` in the scene*, with the exclusion list asserted empty. A legend that names only the
# elements it wants to check is how the 40 px title and every thumbnail caption stayed out
# of M3's font census.

#: The four things the booth screen exists to say, by item name. **This is a floor, not the
#: element set**: the set is every visible named text, and this list is asserted to be a
#: subset of it so the screen cannot pass by having removed them.
REQUIRED_ELEMENTS = (
    "recordedFootageText",          # the mandatory label
    "deviceBadgeText",              # the confirmed placement
    "streamAgeText_A", "streamAgeText_B", "streamAgeText_C",
    "streamBadgeText_A", "streamBadgeText_B", "streamBadgeText_C",
)

#: Per camera, at least one verdict sentence must be on screen and passing: the staleness
#: line for a stale camera, the stated reason for a failed one. "At least one" is the claim,
#: because a healthy camera is neither stale nor failed and correctly says neither.
REQUIRED_VERDICT_PREFIXES = ("streamStaleText_", "streamReasonText_")

#: **The exclusion list, and it must stay empty.** An empty list is the assertion that
#: nothing visible on the always-visible screen is exempt from the legibility threshold.
EXCLUDED_FROM_LEGIBILITY: tuple[str, ...] = ()


def _arcmin(height_mm: float, distance_m: float) -> float:
    """A height in millimetres, as an angle at a distance in metres."""
    return math.degrees(2.0 * math.atan(height_mm / (2.0 * distance_m * 1000.0))) * 60.0


def _verdict(angle: float) -> str:
    from app import theme

    if angle >= theme.READING_MIN_ARCMIN:
        return "PASS"
    if angle >= theme.READING_MARGINAL_ARCMIN:
        return "marginal"
    if angle >= theme.READING_ACUITY_ARCMIN:
        return "FAIL"
    return "FAIL (below acuity)"


class TheBoothScreenReadsAtThreeMetres(unittest.TestCase):
    """Every visible text on the live screen, at **two** geometries.

    The class keeps its original name because the requirement it was written for is still
    "read at three metres", but the delivered screen is judged against the monitor this
    machine is **actually attached to** (0.310 mm/logical px, measured), not the 55-inch
    panel the booth is required to provide (0.634 mm/logical px). On the attached monitor:

    - **2 m:** every element clears 16 arcmin of cap height (44 px = 16.8'); every
      element's **x-height is below** 16 arcmin (44 px = 12.2'), so mixed-case sentences
      do not clear the same band.
    - **3 m:** every element fails on both measures (44 px = 11.2' / 8.1').

    The 55-inch requirement is asserted separately: at 3 m it clears both measures, and
    that is a **booth procurement requirement**, never the achieved result.
    """

    #: The attached monitor's numbers, pinned per font size. Any change to a token, the
    #: geometry or the font moves these and fails, so the record cannot drift silently:
    #: `(2 m cap, 3 m cap, 2 m x-height, 3 m x-height)` in arcmin.
    ATTACHED_MONITOR_ANGLES = {
        44: (16.80, 11.20, 12.16, 8.11),
        48: (18.32, 12.22, 13.27, 8.85),
        56: (21.38, 14.25, 15.48, 10.32),
    }

    @classmethod
    def setUpClass(cls):
        from PySide6.QtGui import QFontDatabase, QGuiApplication

        cls._tmp = tempfile.TemporaryDirectory(prefix="legible-")
        cls.result = capture_ui.capture(scale_percent=100, seconds=10.0,
                                        out_dir=pathlib.Path(cls._tmp.name))
        # **The font the render used, in the parent process.** The child registers
        # `C:/Windows/Fonts/arial.ttf` before the scene loads; without the same registration
        # here, `QFontMetricsF` would measure a substituted font and the cap-height ratio
        # would be a fact about this process rather than about the screen.
        cls._app = QGuiApplication.instance() or QGuiApplication([])
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/arial.ttf")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def metrics(cls, name: str):
        """`QFontMetricsF` for the font **the scene reports** for `name`."""
        from PySide6.QtGui import QFont, QFontMetricsF

        font = QFont("Arial")
        font.setPixelSize(int(cls.result.font_pixels[name]))
        return QFontMetricsF(font)

    @classmethod
    def ink_span(cls, name: str) -> tuple[int, int]:
        """`(rows_with_ink, row_span)` inside `name`'s extent, in **device** pixels.

        The background test is "brightest channel above 70" because every text colour in
        the palette is above that and every surface is below it: `TEXT_ON_PANEL` 232,
        `COLOR_WARN` 240, `COLOR_GOOD` 208, `COLOR_NEUTRAL` 165, `COLOR_BAD` 255, against
        `BG` 20, `BG_PANEL` 35, `BG_PANEL_DEEP` 26 and `BG_VIDEO` 11. That is a property of
        the palette, so it is stated here rather than tuned per item.
        """
        from PySide6.QtGui import QImage

        box = cls.result.extents[name]
        scale_x, scale_y = cls.result.glyph_scale
        image = QImage(str(cls.result.png_path))
        rgb = image.convertToFormat(QImage.Format_RGB888)
        width, height, stride = rgb.width(), rgb.height(), rgb.bytesPerLine()
        raw = bytes(rgb.constBits())
        x0 = max(0, int(box[0] * scale_x))
        y0 = max(0, int(box[1] * scale_y))
        x1 = min(width - 1, int((box[0] + box[2]) * scale_x))
        y1 = min(height - 1, int((box[1] + box[3]) * scale_y))
        rows = 0
        first = last = -1
        for yy in range(y0, y1):
            base = yy * stride
            inked = False
            for xx in range(x0, x1):
                off = base + xx * 3
                pixel = raw[off:off + 3]
                if max(pixel[0], pixel[1], pixel[2]) > 70:
                    inked = True
                    break
            if inked:
                rows += 1
                if first < 0:
                    first = yy
                last = yy
        return rows, (0 if first < 0 else last - first + 1)

    @classmethod
    def angle(cls, name: str, *, metric: str, mm_per_px: float | None = None,
              distance_m: float | None = None) -> float:
        """Angle subtended by `name`'s cap height or x-height, in arcmin.

        Defaults to the **attached monitor's** measured geometry, because that is the
        delivered result. Pass `mm_per_px=theme.MM_PER_LOGICAL_PX` and
        `distance_m=theme.BOOTH_VIEWING_DISTANCE_M` for the 55-inch requirement.
        """
        from app import theme

        metrics = cls.metrics(name)
        height = metrics.capHeight() if metric == "cap" else metrics.xHeight()
        # Logical px -> device px -> mm. At 100 % `glyph_scale` is 1.0 and this is the
        # identity; it is applied anyway so the measurement is explicitly in device pixels.
        device = height * cls.result.glyph_scale[1]
        mm = (theme.ACTUAL_MONITOR_MM_PER_LOGICAL_PX if mm_per_px is None else mm_per_px)
        distance = 2.0 if distance_m is None else distance_m
        return _arcmin(device * mm, distance)

    def element_set(self) -> tuple[str, ...]:
        """Every **visible, non-empty, named** text item on the live screen."""
        scene = self.result
        return tuple(sorted(
            name for name in scene.font_pixels
            if scene.strings.get(name, "") and scene.visible.get(name)
            and name not in EXCLUDED_FROM_LEGIBILITY
        ))

    # -- 1. the measurement has inputs, and they are the screen's own ------------------

    def test_the_capture_rendered_the_logical_canvas_at_one_to_one(self):
        self.assertEqual((self.result.pixel_width, self.result.pixel_height), (1920, 1080),
                         "the capture must be the 1920x1080 render, so logical and device "
                         "pixels coincide and the mm conversion is unambiguous")
        self.assertEqual(tuple(self.result.glyph_scale), (1.0, 1.0),
                         f"glyph_scale is {self.result.glyph_scale}; the ink measurement "
                         f"scales extents by it and anything but 1.0 at 100 % means the "
                         f"extent and the image disagree")

    def test_the_font_ratios_this_test_divides_by_are_the_fonts_real_ones(self):
        from app import theme

        metrics = self.metrics("deviceBadgeText")
        size = self.result.font_pixels["deviceBadgeText"]
        self.assertAlmostEqual(metrics.capHeight() / size, theme.ARIAL_CAP_HEIGHT_EM,
                               places=3,
                               msg=f"capHeight()/pixelSize is {metrics.capHeight() / size}, "
                                   f"and app/theme.py says {theme.ARIAL_CAP_HEIGHT_EM}")
        self.assertAlmostEqual(metrics.xHeight() / size, theme.ARIAL_X_HEIGHT_EM, places=3,
                               msg=f"xHeight()/pixelSize is {metrics.xHeight() / size}, and "
                                   f"app/theme.py says {theme.ARIAL_X_HEIGHT_EM}")

    def test_the_required_elements_are_on_screen_and_not_degenerate(self):
        """Presence, visibility, text and size - **before** any angle is computed."""
        scene = self.result
        self.assertTrue(scene.walk_complete, f"the walk failed: {scene.walk_failures}")
        for name in REQUIRED_ELEMENTS:
            self.assertIn(name, scene.extents,
                          f"{name} was not reached. A `Repeater` delegate is invisible to "
                          f"findChild, so this asserts the walk - the reader that can see "
                          f"one - reached it. Items seen: {sorted(scene.extents)}")
            self.assertTrue(scene.visible.get(name), f"{name} is on screen but hidden")
            self.assertTrue(scene.strings.get(name, "").strip(),
                            f"{name} is on screen, visible and empty: "
                            f"{scene.strings.get(name)!r}")
            x, y, w, h = scene.extents[name]
            self.assertTrue(all(v == v for v in (x, y, w, h)),
                            f"{name} has a NaN extent {scene.extents[name]}; `Math.max` "
                            f"passes NaN through and nothing warns")
            self.assertGreater(w, 1.0, f"{name} is {w} px wide - degenerate")
            self.assertGreater(h, 1.0, f"{name} is {h} px tall - degenerate")

    def test_every_camera_states_a_verdict_on_the_always_visible_screen(self):
        scene = self.result
        for sid in ("A", "B", "C"):
            verdicts = [name for name in scene.strings
                        if name.endswith(sid) and name.startswith(REQUIRED_VERDICT_PREFIXES)
                        and scene.visible.get(name) and scene.strings.get(name, "").strip()]
            self.assertTrue(verdicts,
                            f"camera {sid} has neither a staleness sentence nor a stated "
                            f"reason on the always-visible screen, so its age has nothing "
                            f"beside it saying whether it is a problem")

    def test_nothing_visible_is_exempt_from_the_threshold(self):
        """A legend of exemptions is how the title and the captions escaped M3's census."""
        self.assertEqual(EXCLUDED_FROM_LEGIBILITY, (),
                         "every visible named text must be measured; an exclusion list is "
                         "a list of elements nobody is checking")
        scene = self.result
        unnamed = [name for name in scene.text_property
                   if scene.text_property[name] and name not in scene.font_pixels]
        self.assertEqual(unnamed, [],
                         f"these items have a `text` property but no readable font size, so "
                         f"they would be measured as nothing: {unnamed}")

    # -- 2. the measurement ------------------------------------------------------------

    def test_every_visible_text_on_the_live_screen_clears_the_threshold(self):
        """**The whole screen, not a selection of it.** Cap height and x-height both.

        **Conditional on the 55-inch panel the booth is required to provide** - this is
        the requirement's geometry, not the attached monitor's. The attached monitor's
        verdict is asserted by the four `attached_monitor` tests below, and it is the
        result the report leads with.
        """
        from app import theme

        failures = []
        table = []
        for name in self.element_set():
            cap = self.angle(name, metric="cap", mm_per_px=theme.MM_PER_LOGICAL_PX,
                             distance_m=theme.BOOTH_VIEWING_DISTANCE_M)
            xheight = self.angle(name, metric="x", mm_per_px=theme.MM_PER_LOGICAL_PX,
                                 distance_m=theme.BOOTH_VIEWING_DISTANCE_M)
            table.append((name, self.result.font_pixels[name], cap, xheight,
                          _verdict(cap), _verdict(xheight)))
            if cap < theme.READING_MIN_ARCMIN or xheight < theme.READING_MIN_ARCMIN:
                failures.append(f"{name}: {self.result.font_pixels[name]} px, cap "
                                f"{cap:.1f}', x-height {xheight:.1f}'")
        rendered = "\n".join(
            f"  {n:30s} {px:3d} px  cap {cap:5.1f}' {v1:7s} x {x:5.1f}' {v2}"
            for n, px, cap, x, v1, v2 in table)
        self.assertEqual(failures, [],
                         f"{len(failures)} of {len(table)} visible elements do not clear "
                         f"{theme.READING_MIN_ARCMIN:.0f}' at "
                         f"{theme.BOOTH_VIEWING_DISTANCE_M:.0f} m on the 55-inch panel the "
                         f"booth is required to provide (NOT the attached monitor):\n"
                         + "\n".join(failures)
                         + "\n\nthe rendered table was:\n" + rendered)

    def test_the_staleness_line_and_the_per_camera_age_are_among_the_passing(self):
        """**They are the reason the screen exists.** Spec section 7 says the frame age must
        be legible at a glance; M3 measured 8.5' at 3 m, i.e. not legible at all.

        This is the **55-inch requirement's** geometry. The attached monitor's verdict for
        the same two elements is in `test_every_element_clears_the_band_on_cap_height_at_
        two_metres_on_the_attached_monitor`; at 3 m on that monitor they fail.
        """
        from app import theme

        booth = dict(mm_per_px=theme.MM_PER_LOGICAL_PX,
                     distance_m=theme.BOOTH_VIEWING_DISTANCE_M)
        measured = set(self.element_set())
        for name in ("streamStaleText_A", "streamStaleText_B",
                     "streamAgeText_A", "streamAgeText_B", "streamAgeText_C"):
            self.assertIn(name, measured, f"{name} is not a visible element on the screen")
        for name in ("streamStaleText_A", "streamStaleText_B",
                     "streamAgeText_A", "streamAgeText_B", "streamAgeText_C",
                     "deviceBadgeText", "recordedFootageText"):
            self.assertGreaterEqual(
                self.angle(name, metric="cap", **booth), theme.READING_MIN_ARCMIN,
                f"{name} subtends {self.angle(name, metric='cap', **booth):.1f}' of cap "
                f"height at {theme.BOOTH_VIEWING_DISTANCE_M} m on the 55-inch requirement; "
                f"the threshold is {theme.READING_MIN_ARCMIN}")
            self.assertGreaterEqual(
                self.angle(name, metric="x", **booth), theme.READING_MIN_ARCMIN,
                f"{name}'s lowercase subtends {self.angle(name, metric='x', **booth):.1f}' - "
                f"a mixed-case sentence measured by its capitals only is the M3 mistake")

    # -- 3. the ink, so this is not a property check -----------------------------------

    def test_every_measured_element_has_an_ink_band_of_at_least_its_cap_height(self):
        """**The image, not the property.** A `Text` whose colour matches its surface, or
        whose box is clipped away, has the right `font.pixelSize` and no readable glyph.
        """
        problems = []
        for name in self.element_set():
            metrics = self.metrics(name)
            _rows, span = self.ink_span(name)
            cap = metrics.capHeight() * self.result.glyph_scale[1]
            line = metrics.lineSpacing() * self.result.glyph_scale[1]
            if span < cap - 1.0:
                problems.append(f"{name}: ink spans {span} px, cap height is {cap:.1f} px "
                                f"- nothing that tall is drawn inside its own extent")
            if span > line + 2.0:
                problems.append(f"{name}: ink spans {span} px in a {line:.1f} px line - "
                                f"that is not one line of text, so the extent is measuring "
                                f"something else")
        self.assertEqual(problems, [], "\n".join(problems))

    def test_the_largest_thing_on_the_screen_is_a_frame_age(self):
        """Spec section 7: the age is the figure the screen is about. If a heading or a
        caption ever grows past it, the screen has stopped being about the age."""
        biggest = max(self.element_set(), key=lambda n: self.result.font_pixels[n])
        self.assertTrue(biggest.startswith("streamAgeText_"),
                        f"the largest type on the screen is {biggest} at "
                        f"{self.result.font_pixels[biggest]} px; it must be a frame age")

    # -- 4. controls, so a passing assertion cannot be a passing nothing ----------------

    def test_a_degenerate_element_cannot_pass_the_ink_check(self):
        """**The 1x1-window shape, as a control on this test's own measurement.**

        A window that never got its size gives every item an extent of `(0, 0, 0, 0)`; the
        ink loop then reads no rows and `span` is 0, which is `< cap - 1`, so the ink test
        fails rather than passing over an empty screen. This asserts that arithmetic on the
        zero box directly, because "it failed for the right reason" is not something a
        reader can check from a green tick.
        """
        metrics = self.metrics("streamAgeText_A")
        cap = metrics.capHeight() * self.result.glyph_scale[1]
        self.assertGreater(cap, 1.0, "a zero cap height would make the comparison vacuous")
        self.assertLess(0.0, cap - 1.0,
                        "a zero-height ink band must fail the ink assertion; if this were "
                        "false the check would pass on a screen with no text on it")

    def test_the_threshold_discriminates_between_the_m3_sizes_and_the_shipped_ones(self):
        """**The negative control, on the actual formula.**

        Recomputed here from the measured font ratios and the stated panel: the sizes M3
        shipped (18 px detail, 34 px age) are below the threshold at 3 m, and the sizes now
        on screen are above it. If both came out above, this test would be measuring the
        panel rather than the type.
        """
        from PySide6.QtGui import QFont, QFontMetricsF

        from app import theme

        def at(size: int) -> tuple[float, float]:
            font = QFont("Arial")
            font.setPixelSize(size)
            metrics = QFontMetricsF(font)
            mm = theme.MM_PER_LOGICAL_PX
            return (_arcmin(metrics.capHeight() * mm, theme.BOOTH_VIEWING_DISTANCE_M),
                    _arcmin(metrics.xHeight() * mm, theme.BOOTH_VIEWING_DISTANCE_M))

        m3_detail = at(theme.TYPE_DETAIL)
        m3_age = at(theme.TYPE_NUMERAL_SMALL)
        shipped = at(self.result.font_pixels["streamStaleText_A"])
        self.assertLess(m3_detail[0], theme.READING_MIN_ARCMIN,
                        f"M3's 18 px detail text is {m3_detail[0]:.1f}' of cap height on the "
                        f"stated panel - if that passed, the threshold would be a restatement "
                        f"of the panel size rather than of the type")
        self.assertLess(m3_detail[1], theme.READING_MARGINAL_ARCMIN,
                        f"and its lowercase is {m3_detail[1]:.1f}'")
        self.assertGreaterEqual(shipped[0], theme.READING_MIN_ARCMIN)
        self.assertGreaterEqual(shipped[1], theme.READING_MIN_ARCMIN)
        self.assertGreater(m3_age[0], m3_detail[0],
                           "the age really was the largest thing on the M3 screen; the "
                           f"ordering must be preserved for this control to mean anything "
                           f"(34 px {m3_age[0]:.1f}' vs 18 px {m3_detail[0]:.1f}')")

    # -- 5. the measured monitor: the gating case, and the requirement it is not --------

    def test_the_attached_monitor_geometry_is_the_measured_panel_not_the_booth_requirement(
            self):
        """**The correction this class exists to make.**

        The delivered screen is judged on the monitor actually attached (`WmiMonitorBasic
        DisplayParams` 80 x 33 cm, `WmiMonitorID` DELL S3422DWG, 3440x1440 native, Windows
        at 150 % so a 2293x960 desktop). It is **not** judged on the 55-inch panel the booth
        is required to provide, which is roughly twice its scale. The default geometry of
        `angle()` must be the attached monitor, or a legibility test would pass on a panel
        this machine does not have.
        """
        from app import theme

        self.assertEqual(theme.ACTUAL_MONITOR_NAME, "DELL S3422DWG")
        self.assertEqual((theme.ACTUAL_MONITOR_WIDTH_MM,
                          theme.ACTUAL_MONITOR_HEIGHT_MM), (800.0, 330.0),
                         "WmiMonitorBasicDisplayParams reports MaxHorizontalImageSize=80, "
                         "MaxVerticalImageSize=33 (whole centimetres)")
        self.assertEqual((theme.ACTUAL_MONITOR_NATIVE_WIDTH_PX,
                          theme.ACTUAL_MONITOR_NATIVE_HEIGHT_PX), (3440, 1440))
        self.assertEqual((theme.ACTUAL_MONITOR_DESKTOP_WIDTH_PX,
                          theme.ACTUAL_MONITOR_DESKTOP_HEIGHT_PX), (2293, 960),
                         "3440x1440 at 150 % desktop scaling is 2293x960 logical; the app "
                         "sees the logical desktop")
        self.assertEqual(theme.ACTUAL_MONITOR_VIEWING_DISTANCES_M, (2.0, 3.0),
                         "the specification's 2-3 m, both asserted")

        # The 1080-px canvas is height-limited in a 960-px-tall desktop.
        self.assertAlmostEqual(theme.ACTUAL_MONITOR_CANVAS_SCALE, 960.0 / 1080.0, places=9)
        self.assertLess(theme.ACTUAL_MONITOR_CANVAS_SCALE, 1.0,
                        "if this were 1.0 the canvas would fit the desktop and the "
                        "letterbox derivation below would be wrong")

        # The derivation, from the measured inputs, independently of the constant's value.
        derived = (
            theme.ACTUAL_MONITOR_WIDTH_MM
            * (theme.CANVAS_WIDTH * theme.ACTUAL_MONITOR_CANVAS_SCALE
               / theme.ACTUAL_MONITOR_DESKTOP_WIDTH_PX)
            / theme.CANVAS_WIDTH
        )
        self.assertAlmostEqual(theme.ACTUAL_MONITOR_MM_PER_LOGICAL_PX, derived, places=9)
        self.assertEqual(round(theme.ACTUAL_MONITOR_MM_PER_LOGICAL_PX, 3), 0.310,
                         f"the attached monitor is "
                         f"{theme.ACTUAL_MONITOR_MM_PER_LOGICAL_PX:.4f} mm/logical px; "
                         f"0.310 is the figure M3 measured and this review records")

        # And the same number from the native physical pixels, where the 150 % scale
        # cancels: (1440/1080) x 800/3440.
        native = (theme.ACTUAL_MONITOR_NATIVE_HEIGHT_PX / theme.CANVAS_HEIGHT
                  * theme.ACTUAL_MONITOR_WIDTH_MM / theme.ACTUAL_MONITOR_NATIVE_WIDTH_PX)
        self.assertAlmostEqual(theme.ACTUAL_MONITOR_MM_PER_LOGICAL_PX, native, places=3,
                               msg="the logical-desktop and native-pixel derivations "
                                   "disagree; one of the measured inputs is wrong")

        # It is about half the 55-inch requirement - the whole point of the correction.
        self.assertLess(theme.ACTUAL_MONITOR_MM_PER_LOGICAL_PX,
                        0.55 * theme.MM_PER_LOGICAL_PX,
                        "the attached monitor must be materially smaller than the booth "
                        "requirement, or this correction is measuring nothing")

        # **The default geometry is the attached monitor, not the requirement.** A test
        # that defaults to the 55-inch panel would pass at 3 m while the delivered screen
        # fails there.
        default = self.angle("streamStaleText_A", metric="cap")
        explicit = self.angle("streamStaleText_A", metric="cap",
                              mm_per_px=theme.ACTUAL_MONITOR_MM_PER_LOGICAL_PX,
                              distance_m=2.0)
        self.assertAlmostEqual(default, explicit, places=9,
                               msg="the default angle must use the attached monitor's "
                                   "geometry at 2 m")
        self.assertAlmostEqual(default, 16.80, delta=0.15,
                               msg=f"the default is {default:.2f}', and 16.8' is the "
                                   f"attached monitor's 44 px cap height at 2 m - 34.4' "
                                   f"would be the 55-inch requirement at the same distance")

    def test_every_element_clears_the_band_on_cap_height_at_two_metres_on_the_attached_monitor(
            self):
        """**The achieved pass: cap height at 2 m on the attached monitor.**

        Every visible element, not a selection. The frame age and the staleness verdict are
        named with their numbers because they are the reason the screen exists.
        """
        from app import theme

        failures = []
        table = []
        for name in self.element_set():
            cap = self.angle(name, metric="cap", distance_m=2.0)
            table.append((name, self.result.font_pixels[name], cap))
            if cap < theme.READING_MIN_ARCMIN:
                failures.append(f"{name}: {self.result.font_pixels[name]} px, cap {cap:.1f}'")
        rendered = "\n".join(f"  {n:30s} {px:3d} px  cap {cap:5.1f}'"
                             for n, px, cap in table)
        self.assertEqual(failures, [],
                         f"{len(failures)} of {len(table)} visible elements do not clear "
                         f"{theme.READING_MIN_ARCMIN:.0f}' of cap height at 2 m on the "
                         f"attached monitor ({theme.ACTUAL_MONITOR_NAME}, "
                         f"{theme.ACTUAL_MONITOR_MM_PER_LOGICAL_PX:.3f} mm/logical px):\n"
                         + "\n".join(failures) + "\n\n" + rendered)
        # Named, with the pinned numbers. 44 px is 16.80' of cap at 2 m; 56 px is 21.38'.
        self.assertAlmostEqual(self.angle("streamStaleText_A", metric="cap"),
                               16.80, delta=0.15,
                               msg="the staleness verdict is the sentence a passer-by must "
                                   "read; its 44 px cap must measure 16.8' at 2 m")
        self.assertAlmostEqual(self.angle("streamAgeText_A", metric="cap"),
                               21.38, delta=0.15,
                               msg="the frame age is the largest figure on the screen; its "
                                   "56 px cap must measure 21.4' at 2 m")

    def test_the_attached_monitor_fails_at_three_metres_on_both_measures_and_is_pinned(self):
        """**The other half of the achieved result, and the numbers are pinned.**

        On the attached monitor every visible element is below 16' of cap height **and**
        x-height at 3 m. This is asserted rather than omitted because a report that leads
        with "passes at 2 m" must say where it stops passing, and a test that silently
        asserted nothing at 3 m would let the failure disappear.

        The per-size angles are pinned, so **shrinking a token fails this test** - not
        because the shrunken text would pass, but because the recorded measurement would
        no longer be the measurement, and the review and this test would disagree.
        """
        from app import theme

        seen_sizes = set()
        for name in self.element_set():
            px = self.result.font_pixels[name]
            cap = self.angle(name, metric="cap", distance_m=3.0)
            xheight = self.angle(name, metric="x", distance_m=3.0)
            self.assertLess(cap, theme.READING_MIN_ARCMIN,
                            f"{name} is {px} px and subtends {cap:.1f}' of cap height at "
                            f"3 m on the attached monitor; the attached monitor was "
                            f"measured to FAIL at 3 m, so a pass here means the geometry, "
                            f"the font or the token moved without the record being updated")
            self.assertLess(xheight, theme.READING_MIN_ARCMIN,
                            f"{name}'s lowercase subtends {xheight:.1f}' at 3 m on the "
                            f"attached monitor; it must be below the 16' band")
            expected = self.ATTACHED_MONITOR_ANGLES[px]
            self.assertAlmostEqual(cap, expected[1], delta=0.15,
                                   msg=f"{name}: {px} px measures {cap:.2f}' of cap at 3 m "
                                       f"on the attached monitor; the record says "
                                       f"{expected[1]:.2f}'")
            self.assertAlmostEqual(xheight, expected[3], delta=0.15,
                                   msg=f"{name}: {px} px measures {xheight:.2f}' of "
                                       f"x-height at 3 m; the record says {expected[3]:.2f}'")
            seen_sizes.add(px)
        self.assertEqual(seen_sizes, {44, 48, 56},
                         f"the screen no longer carries the whole ladder: {seen_sizes}")

    def test_the_mixed_case_sentences_do_not_clear_the_band_at_two_metres_on_the_attached_monitor(
            self):
        """**"Passes at 2 m" is a cap-height statement, and this says so.**

        Arial's lowercase is 0.5184 em against a 0.7164 em cap. At 0.310 mm/logical px a
        44 px sentence has a 12.2' x-height at 2 m - above the 10' marginal band, below the
        16' comfort band. So the 2 m verdict for the mixed-case sentences is **marginal**,
        not a pass, and the frame age's 15.5' is just short as well. Asserting this keeps
        the report's two-part verdict honest: the screen was not measured to clear 16' on
        both measures at either distance on the attached monitor.
        """
        from app import theme

        clearing = []
        for name in self.element_set():
            xheight = self.angle(name, metric="x", distance_m=2.0)
            if xheight >= theme.READING_MIN_ARCMIN:
                clearing.append(f"{name}: {self.result.font_pixels[name]} px, "
                                f"x-height {xheight:.1f}'")
        self.assertEqual(clearing, [],
                         "these elements clear 16' of x-height at 2 m on the attached "
                         "monitor; the review says none do, so the record and the render "
                         "disagree:\n" + "\n".join(clearing))
        # Not invisible: the sentences are in the marginal band, not below the acuity floor.
        for name in ("recordedFootageText", "deviceBadgeText", "streamStaleText_A"):
            xheight = self.angle(name, metric="x", distance_m=2.0)
            self.assertGreaterEqual(
                xheight, theme.READING_MARGINAL_ARCMIN,
                f"{name}'s 44 px lowercase is {xheight:.1f}' at 2 m - that is below the "
                f"{theme.READING_MARGINAL_ARCMIN}' marginal band, so calling it 'marginal' "
                f"would be flattering it")
        self.assertAlmostEqual(self.angle("streamStaleText_A", metric="x", distance_m=2.0),
                               12.16, delta=0.15)
        self.assertAlmostEqual(self.angle("streamAgeText_A", metric="x", distance_m=2.0),
                               15.48, delta=0.15)

    def test_the_fifty_five_inch_panel_is_a_requirement_not_the_delivered_result(self):
        """**The requirement is kept, and it is never read as the achievement.**

        At 3 m on a 55-inch panel the same 44 px element is 22.9' of cap height; on the
        attached monitor it is 11.2'. The 55-inch panel is what the booth must procure for
        the screen to read at 3 m. It is not what this machine measured.
        """
        from app import theme

        self.assertGreaterEqual(theme.BOOTH_PANEL_DIAGONAL_IN, 55.0,
                                "the booth requirement must not be deleted by the "
                                "correction that separated it from the delivered panel")
        self.assertAlmostEqual(theme.BOOTH_PANEL_WIDTH_MM, 1217.59, delta=0.01)
        self.assertAlmostEqual(theme.MM_PER_LOGICAL_PX, 0.63416, places=4)

        booth = self.angle("streamStaleText_A", metric="cap",
                           mm_per_px=theme.MM_PER_LOGICAL_PX, distance_m=3.0)
        attached = self.angle("streamStaleText_A", metric="cap", distance_m=3.0)
        self.assertGreaterEqual(booth, theme.READING_MIN_ARCMIN,
                                f"on the required 55-inch panel 44 px is {booth:.1f}' at "
                                f"3 m; the requirement would be pointless if it failed")
        self.assertLess(attached, theme.READING_MIN_ARCMIN,
                        f"on the attached monitor 44 px is {attached:.1f}' at 3 m")
        self.assertGreater(booth - attached, 10.0,
                           "the two geometries must be far enough apart that confusing "
                           "them changes the verdict; that separation is the correction")


if __name__ == "__main__":
    unittest.main()
