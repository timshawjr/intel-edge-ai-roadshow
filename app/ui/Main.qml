// Main.qml - the booth screen, sized to be read from three metres. **M4 Task 1.**
//
// ## What changed, and why it is a removal rather than a font size
//
// M3's readability review measured the screen at 3 m and found that **nothing on it cleared
// 16 arcmin**, the ISO 9241-3 minimum for comfortable sustained reading: the frame-age
// headline was 8.5', and every 18 px line was below the acuity floor. The arithmetic is
// unforgiving: at 3 m a character needs ~14 mm of cap height, which on the stated booth
// panel (`app/theme.py`) is a ~44 px Arial - **2.4x** the 18 px this screen shipped. Three
// cameras' worth of 44 px text does not fit beside the pictures at the old density, so the
// always-visible screen **lost rows** instead of gaining lines:
//
//   moved into the presenter drawer (`PresenterDrawer.qml`), where a presenter looks
//   closely: **per-camera backlog, analysis rate, frame-age window, activity line, device
//   utilisation figure and telemetry averaging window; the event counts; the `status:` and
//   `telemetry fresh` lines; the six-stage pipeline strip; the reduction note (it is stated
//   verbatim there, and the header copy was elided mid-word); and the seven 18 px `?` keys,
//   whose surface is the drawer and whose touch entry point is now one 44 px button.**
//
//   removed outright: **`PipelineStrip.qml`.** Twelve 18 px labels cannot be legible at 3 m
//   and the stage list does not fit the drawer's width without eliding; the topology is
//   stated in words in the drawer instead.
//
// What is left on the always-visible screen is what a passer-by is entitled to read: the
// mandatory recorded-footage label, the confirmed-device badge, three live pictures each
// captioned with its own camera and device, each camera's frame age as the largest figure
// on the screen, its device badge, and one verdict line per camera carrying the staleness
// sentence or the camera's own stated failure reason.
//
// ## The geometry
//
// 1920 x 1080 logical canvas, unchanged. 13% header / 66% main / 21% footer, and 68% video
// / 32% metrics - the main band's 66% and the video column's 68% are the design's section 9
// proportions and are unchanged; the header gained 1% and the footer lost 1% because the
// footer now holds three verdict lines instead of the pipeline strip and the control bar.
//
// No pixel count is in this file: every size and spacing is a token from `app/theme.py`.
//
// ## What is deliberately still always visible
//
// The error line, whenever there is one. It is never cleared implicitly, and it is 44 px
// like everything else: an error that disappears, or that nobody can read, is an error
// nobody saw.

import QtQuick
import QtQuick.Window

Window {
    id: window

    width: Theme.CANVAS_WIDTH
    height: Theme.CANVAS_HEIGHT
    visible: true
    color: Theme.TEXT_ON_PANEL
    title: "Intel Edge AI Roadshow - " + bridgeUi.recordedFootageLabel

    // **The presenter drawer (M3 Task 5).** A display state, like attract mode: the drawer
    // is an overlay anchored to the window's right edge, so opening it cannot reflow the
    // hero, the tiles, the badges or the verdict lines. `bridgeUi.drawerWidth` is the
    // width, from `app/theme.py`, so no pixel count lives in this file.
    property bool drawerOpen: false

    // The id is lower-case on purpose: QML rejects an id beginning with an uppercase
    // letter, and `Bridge { id: Bridge }` does not load. Measured: the window came up
    // with zero root objects and the only diagnostic was `IDs cannot start with an
    // uppercase letter` at the Bridge line.
    Bridge { id: bridgeUi }

    Rectangle {
        id: screen
        anchors.fill: parent
        // **Named so a test can read whether the live screen is on screen at all.** M3
        // Task 4's attract mode hides this rectangle; the attract render asserts it is not
        // visible, which is what makes "no live number is shown" a checkable claim.
        objectName: "liveScreen"
        visible: !bridge.attract_active
        color: Theme.BG

        // The three bands, as fractions of the height that is actually available.
        //
        // **Not** `CANVAS_HEIGHT * fraction`: that is 1080 * (0.13 + 0.66 + 0.21) = 1080
        // of band, plus 2 * MARGIN and 2 * GAP of chrome, which is 1128 px in a 1080 px
        // canvas. The first offscreen render lost the bottom of the control bar off the
        // screen (measured: the "status:" line clipped at the canvas edge), which is the
        // failure the design's letterboxing note is about - except here nothing was
        // letterboxed, it was simply off the end.
        readonly property real bandTotal: screen.height - Theme.MARGIN * 2 - Theme.GAP * 2
        readonly property real headerBand: bandTotal * Theme.GRID.header
        readonly property real mainBand: bandTotal * Theme.GRID.main
        readonly property real footerBand: bandTotal - headerBand - mainBand

        Column {
            anchors.fill: parent
            anchors.margins: Theme.MARGIN
            spacing: Theme.GAP

            // =====================================================================
            // Header - 13% of the available height
            // =====================================================================
            Item {
                width: parent.width
                height: screen.headerBand

                Column {
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    spacing: Theme.SPACING.sm

                    Item {
                        width: parent.width
                        height: labelBox.height

                        Row {
                            anchors.left: parent.left
                            anchors.verticalCenter: parent.verticalCenter
                            spacing: Theme.SPACING.lg

                            // THE MANDATORY LABEL. Unconditional, never inside a conditional.
                            Rectangle {
                                id: labelBox
                                anchors.verticalCenter: parent.verticalCenter
                                width: recordedFootageText.implicitWidth + Theme.MARGIN
                                height: recordedFootageText.implicitHeight
                                        + Theme.SPACING.sm * 2
                                radius: Theme.RADIUS
                                color: Theme.BG_PANEL_DEEP
                                border.color: Theme.ACCENT
                                border.width: 2

                                Text {
                                    id: recordedFootageText
                                    // **Named (M4).** Until this task the mandatory label
                                    // had an `id` and no `objectName`, so the harness - which
                                    // reads `objectName`s - could not see the one string the
                                    // spec requires to be unconditional. A label nobody can
                                    // measure is a label nobody is checking.
                                    objectName: "recordedFootageText"
                                    anchors.centerIn: parent
                                    text: bridgeUi.recordedFootageLabel
                                    color: Theme.TEXT_ON_PANEL
                                    font.family: Theme.FONT_FAMILY
                                    font.pixelSize: Theme.TYPE_BOOTH_DETAIL
                                    font.bold: true
                                }
                            }

                            Text {
                                id: appTitleText
                                objectName: "appTitle"
                                anchors.verticalCenter: parent.verticalCenter
                                text: "Intel Edge AI Roadshow"
                                color: Theme.TEXT_ON_PANEL
                                font.family: Theme.FONT_FAMILY
                                font.pixelSize: Theme.TYPE_BOOTH_DETAIL
                                font.bold: true
                            }
                        }

                        Row {
                            anchors.right: parent.right
                            anchors.verticalCenter: parent.verticalCenter
                            spacing: Theme.SPACING.md

                            // **The touch entry point to the explanation surface.** M1's
                            // lesson was that a booth machine is often touch-only and a
                            // hover-only affordance is unreachable there, so the drawer needs
                            // a button as well as F1. One button, not seven 18 px keys: the
                            // keys are all inside the drawer, and at 3 m a 44 px entry point
                            // is reachable while a 45 px-wide `cpu` chip was not.
                            Rectangle {
                                objectName: "explainOpenButton"
                                anchors.verticalCenter: parent.verticalCenter
                                width: explainOpenText.implicitWidth + Theme.SPACING.md
                                height: explainOpenText.implicitHeight
                                        + Theme.SPACING.sm * 2
                                radius: Theme.RADIUS
                                color: window.drawerOpen ? Theme.ACCENT : Theme.BG_PANEL_DEEP
                                border.color: Theme.BORDER
                                border.width: 2

                                Text {
                                    id: explainOpenText
                                    objectName: "explainOpenButtonText"
                                    anchors.centerIn: parent
                                    text: "what the boxes mean"
                                    color: window.drawerOpen
                                        ? Theme.TEXT_ON_ACCENT : Theme.COLOR_NEUTRAL
                                    font.family: Theme.FONT_FAMILY
                                    font.pixelSize: Theme.TYPE_BOOTH_DETAIL
                                }

                                MouseArea {
                                    objectName: "explainOpenMouse"
                                    anchors.fill: parent
                                    onClicked: window.drawerOpen = true
                                }
                            }

                            Rectangle {
                                id: routeBadge
                                anchors.verticalCenter: parent.verticalCenter
                                width: routeText.implicitWidth + Theme.MARGIN
                                height: routeText.implicitHeight + Theme.SPACING.sm * 2
                                radius: Theme.RADIUS
                                // Green only when a placement is confirmed. "pending" and
                                // "failed" are the neutral and the bad colour respectively,
                                // and both are also words.
                                color: bridgeUi.deviceConfirmed
                                    ? Theme.COLOR_GOOD : Theme.BG_PANEL_DEEP
                                border.color: bridge.route_state === "failed"
                                    ? Theme.COLOR_BAD
                                    : (bridgeUi.deviceConfirmed
                                        ? Theme.COLOR_GOOD : Theme.BORDER)
                                border.width: 2

                                Text {
                                    id: routeText
                                    // The one nameable handle on the device badge. It exists
                                    // so a test can read the **QML-bound** string after
                                    // `set_route` rather than reading the bridge's Python
                                    // property - the badge was `constant=True` with no notify
                                    // signal, so the Python side updated and the screen never
                                    // did, and only a read of the bound text can tell the two
                                    // apart.
                                    objectName: "deviceBadgeText"
                                    anchors.centerIn: parent
                                    text: "device: " + bridgeUi.deviceBadge
                                    color: bridgeUi.deviceConfirmed
                                        ? Theme.TEXT_ON_ACCENT : Theme.COLOR_NEUTRAL
                                    font.family: Theme.FONT_FAMILY
                                    font.pixelSize: Theme.TYPE_BOOTH_DETAIL
                                    font.bold: true
                                }
                            }
                        }
                    }

                    // The error line, whenever there is one. Visible whenever there is an
                    // error, and never cleared implicitly: an error that disappears on the
                    // next good frame is an error nobody saw.
                    Rectangle {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        height: errorText.implicitHeight + Theme.SPACING.sm * 2
                        visible: bridge.error_message.length > 0
                        radius: Theme.RADIUS
                        color: Theme.BG_PANEL_DEEP
                        border.color: Theme.COLOR_BAD
                        border.width: 2

                        Text {
                            id: errorText
                            objectName: "errorText"
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.leftMargin: Theme.SPACING.sm
                            anchors.rightMargin: Theme.SPACING.sm
                            text: bridge.error_message
                            color: Theme.COLOR_BAD
                            font.family: Theme.FONT_FAMILY
                            font.pixelSize: Theme.TYPE_BOOTH_DETAIL
                            elide: Text.ElideRight
                        }
                    }
                }
            }

            // =====================================================================
            // Main - 66%:  hero + two thumbnails on the left, cameras on the right
            // =====================================================================
            Item {
                width: parent.width
                height: screen.mainBand

                // -- hero, 68% of the width -------------------------------------
                Item {
                    id: heroColumn
                    width: Math.round(parent.width * Theme.VIDEO_WIDTH_FRACTION)
                    height: parent.height

                    // **The streams other than the hero's, as a real list.** With three
                    // cameras that is two thumbnails, each showing **its own** stream's
                    // frames; with one camera it is empty and the hero takes the whole
                    // height. Never a placeholder tile: a surface captioned with a camera
                    // that does not exist, showing another camera's picture, is the
                    // mislabelling this project keeps finding.
                    readonly property var otherStreams: {
                        var out = [];
                        var ids = bridge.stream_ids;
                        for (var i = 0; i < ids.length; ++i) {
                            if (ids[i] !== hero.streamId)
                                out.push(ids[i]);
                        }
                        return out;
                    }

                    VideoTile {
                        id: hero
                        // **A name on the hero**, so a test can read *which camera it is
                        // showing* as a property rather than inferring it from which tile
                        // happens to be largest.
                        objectName: "heroTile"
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        height: parent.height
                              - (heroColumn.otherStreams.length > 0
                                 ? thumbRow.height + Theme.GAP : 0)
                        hero: true
                        streamId: bridge.route_stream
                        drawOverlay: true
                        explainKey: "yolo"
                    }

                    // -- one thumbnail per other camera ---------------------------
                    Item {
                        id: thumbRow
                        visible: heroColumn.otherStreams.length > 0
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.bottom: parent.bottom
                        // **A token, not `0.16` inline (M4).** The row has to hold a
                        // `TYPE_BOOTH_DETAIL` caption bar, and at 16 % a 660 px band gave a
                        // 105 px thumbnail where the 44 px caption alone was more than half
                        // of the picture.
                        height: visible ? parent.height * Theme.THUMB_ROW_FRACTION : 0

                        Row {
                            anchors.fill: parent
                            spacing: Theme.GAP

                            Repeater {
                                model: heroColumn.otherStreams
                                delegate: VideoTile {
                                    required property string modelData
                                    objectName: "thumb_" + modelData
                                    width: heroColumn.otherStreams.length > 1
                                           ? (parent.width - Theme.GAP) / 2 : parent.width
                                    height: parent.height
                                    hero: false
                                    streamId: modelData
                                    drawOverlay: false
                                    // The camera's own confirmed device, per stream - not
                                    // the hero's. A thumbnail captioned with the wrong
                                    // device is the same defect as the wrong picture.
                                    //
                                    // **A plain string from a notified map, never the
                                    // list.** `stream_routes[sid].confirmed` is a Python
                                    // tuple that arrives in QML as a `QVariantList` which
                                    // is truthy, has no usable `length`, and stringifies
                                    // to **`-1`**. Measured: this caption read
                                    // `Camera B · -1` on a render whose Python side held
                                    // `confirmed: ('GPU.0',)` and whose picture above the
                                    // caption was real. Nothing about that screenshot says
                                    // the caption is wrong.
                                    caption: "Camera " + modelData + " \u00b7 "
                                             + (bridge.stream_confirmed_labels[modelData] !== undefined
                                                && bridge.stream_confirmed_labels[modelData].length > 0
                                                ? bridge.stream_confirmed_labels[modelData]
                                                : "not confirmed")
                                    explainKey: "tracking"
                                }
                            }
                        }
                    }
                }

                // -- one block per CAMERA, 32% ------------------------------------
                //
                // **What a passer-by reads from the aisle**: which camera, whether it is
                // running, how old its picture is, and where it is running. Three rows per
                // camera at the legibility ladder, and nothing else - the backlog, the
                // rate, the window, the activity line and the utilisation figure are in
                // `PresenterDrawer.qml`, because six rows at 44 px is 280 px per camera and
                // three cameras of that do not fit a 660 px band.
                Column {
                    anchors.left: heroColumn.right
                    anchors.leftMargin: Theme.GAP
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.bottom: parent.bottom
                    spacing: Theme.GAP

                    // -- one card per CAMERA, not per device (M2 Task 5) -------------
                    //
                    // Three cameras therefore produce three cards, and the `Repeater`
                    // builds one per stream the supervisor actually owns rather than three
                    // fixed ones - a supervisor with one stream draws one card, and one with
                    // four draws four, so a card can never exist for a camera that is not
                    // there.
                    //
                    // **`modelData`, not an index.** `streamId` is the model's own value, so
                    // a card reads its own record and not `modelData` from a shared
                    // delegate scope.
                    Repeater {
                        id: streamCards
                        model: bridge.stream_ids

                        delegate: ComputeCard {
                            // Named for the camera, so a test can read this card's bound
                            // strings and cannot accidentally read another's.
                            objectName: "card_" + modelData
                            streamId: modelData
                            width: parent.width
                            // A column of `n` cards in a fixed height: each gets its share.
                            //
                            // **`bridge.stream_ids.length`, not `count`.** Inside a
                            // delegate `count` was not in scope, so the division produced
                            // `NaN`, `Math.max` passed the `NaN` straight through, and the
                            // cards were **all zero-height and invisible** - the offscreen
                            // render came back with a completely empty metrics column and
                            // no QML warning at all, because a `NaN` geometry is legal.
                            //
                            // **The gaps come out of the height, not out of nowhere. M3
                            // Task 2.** `parent.height / n` ignores the `Column`'s
                            // `spacing`, so the cards claimed the full band **plus** two
                            // gaps: 32 px into the footer, drawn anyway because a `Column`
                            // does not clip. Now `(660 - 16 * 2) / 3 = 209.33` per card and
                            // the column ends exactly on the band.
                            //
                            // **The floor is `0`, not `Theme.TYPE_NUMERAL`.** A type size as
                            // the floor raises every card to that height and the column then
                            // claims more than its band at any camera count the band cannot
                            // supply a numeral for.
                            height: Math.max(0, (parent.height
                                                 - Theme.GAP * Math.max(0, bridge.stream_ids.length - 1))
                                                / Math.max(1, bridge.stream_ids.length))
                            onExplainRequested: window.drawerOpen = true
                        }
                    }
                }
            }

            // =====================================================================
            // Footer - 21%: one verdict line per camera, in words
            // =====================================================================
            //
            // **Why the sentence is here and not on the card.** At `TYPE_BOOTH_DETAIL` the
            // staleness sentence is 918 px wide; the metrics column's inner width is 567,
            // so on the card it wrapped to two lines and pushed the card past its band.
            // Full width it is one line, at the largest size on the screen after the age,
            // and it still carries the camera's name and the same age the numeral shows.
            Item {
                width: parent.width
                height: screen.footerBand

                Column {
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    spacing: Theme.SPACING.sm

                    Repeater {
                        model: bridge.stream_ids

                        delegate: Row {
                            required property string modelData
                            objectName: "verdictRow_" + modelData
                            width: parent.width
                            spacing: Theme.SPACING.sm

                            // This camera's record, or a **cold record** with no verdict -
                            // never `undefined`, which would make every binding on it read
                            // `undefined` and render as nothing.
                            readonly property var record:
                                bridge.stream_cards_by_id[modelData] !== undefined
                                ? bridge.stream_cards_by_id[modelData]
                                : ({ "staleness": "", "reason_display": "" })

                            Text {
                                id: staleText
                                objectName: "streamStaleText_" + modelData
                                // **The word, not only the number**, and the camera's name
                                // in front of it: this line sits in a different band from
                                // the card, so it has to say which camera it is about.
                                //
                                // **A camera with a stated failure reason shows the reason
                                // instead.** One verdict line per camera fits at 44 px; two
                                // do not, and the reason is the more specific statement. The
                                // age and the card's warning border still carry the
                                // staleness, and the reason line names the camera too.
                                visible: record.staleness !== ""
                                         && record.reason_display === ""
                                width: visible ? parent.width : 0
                                text: visible ? "Camera " + modelData + ": " + record.staleness
                                              : ""
                                color: Theme.COLOR_WARN
                                font.family: Theme.FONT_FAMILY
                                font.pixelSize: Theme.TYPE_BOOTH_DETAIL
                                font.bold: true
                                elide: Text.ElideRight
                            }

                            // This camera's own stated reason, **already cut and already
                            // saying so** - `reason_display` carries `... +N more`. The
                            // complete message stays published as `record.reason` and is
                            // readable in the drawer, which prints it in full.
                            Text {
                                objectName: "streamReasonText_" + modelData
                                visible: !staleText.visible && record.reason_display !== ""
                                width: visible ? parent.width : 0
                                text: visible
                                      ? "Camera " + modelData + ": " + record.reason_display : ""
                                color: Theme.COLOR_BAD
                                font.family: Theme.FONT_FAMILY
                                font.pixelSize: Theme.TYPE_BOOTH_DETAIL
                                elide: Text.ElideRight
                            }
                        }
                    }
                }
            }
        }
    }

    // =================================================================================
    // Attract mode (M3 Task 4) - and its type, like the live screen's, from the ladder
    // =================================================================================
    //
    // **The one screen whose whole purpose is to show no number it cannot date.** The live
    // screen above is hidden while this is up (`screen.visible: !bridge.attract_active`),
    // so there is no live figure behind it to be misread. Everything here comes from
    // `bridge.attract_headlines`, a constant list of recorded measurements, each rendered
    // **with the date it was recorded and the committed file it came from**.
    //
    // **M4: the idle screen is read from the aisle too.** The overlay's type was
    // `TYPE_HEADLINE` / `TYPE_LABEL` / `TYPE_DETAIL` - 40 / 26 / 18 px - and 18 px is below
    // the acuity floor at 3 m. A booth with nobody at it is exactly when a passer-by reads
    // it, so these sizes come from the same ladder as the live screen.
    //
    // The overlay is a `Loader` with `active: bridge.attract_active` and an inline
    // `Component`, so when attract mode is off these items **do not exist in the scene
    // graph at all**. That keeps the live screen's font census (and every committed
    // capture) exactly what it is.
    Loader {
        id: attractLoader
        // Named so the render can assert the overlay **loaded**, not merely that it was
        // asked for: an `active` Loader whose component failed reads exactly like an empty
        // overlay, and "no strings" must never be read as a pass.
        objectName: "attractLoader"
        anchors.fill: parent
        active: bridge.attract_active
        sourceComponent: attractOverlay
    }

    Component {
        id: attractOverlay

        Rectangle {
            objectName: "attractOverlay"
            anchors.fill: parent
            color: Theme.BG

            Column {
                anchors.centerIn: parent
                width: parent.width - Theme.MARGIN * 2
                spacing: Theme.SPACING.lg

                Text {
                    objectName: "attractTitle"
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: bridge.attract_title
                    color: Theme.TEXT_ON_PANEL
                    font.family: Theme.FONT_FAMILY
                    font.pixelSize: Theme.TYPE_BOOTH_NUMERAL
                    font.bold: true
                }

                // The recorded-surface label, spec section 9's wording. It is the first
                // thing a visitor reads and it says what the figures below it are.
                Text {
                    objectName: "attractRecordedLabel"
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: bridge.attract_recorded_label
                    color: Theme.COLOR_WARN
                    font.family: Theme.FONT_FAMILY
                    font.pixelSize: Theme.TYPE_BOOTH_LABEL
                    font.bold: true
                }

                Repeater {
                    model: bridge.attract_headlines

                    delegate: Column {
                        required property var modelData
                        required property int index
                        width: parent.width
                        spacing: Theme.SPACING.xs

                        Text {
                            objectName: "attractHeadline_" + index
                            width: parent.width
                            text: modelData.text
                            color: Theme.TEXT_ON_PANEL
                            font.family: Theme.FONT_FAMILY
                            font.pixelSize: Theme.TYPE_BOOTH_DETAIL
                            wrapMode: Text.WordWrap
                            horizontalAlignment: Text.AlignHCenter
                        }

                        // **The date and the source, on the same slide as the number.**
                        // A figure here without this line would be a number nobody can
                        // date, which is the failure this mode exists to prevent.
                        Text {
                            objectName: "attractHeadlineRecorded_" + index
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "recorded " + modelData.date + "  \u00b7  " + modelData.source
                            color: Theme.COLOR_NEUTRAL
                            font.family: Theme.FONT_FAMILY
                            font.pixelSize: Theme.TYPE_BOOTH_DETAIL
                        }
                    }
                }
            }
        }
    }

    // =================================================================================
    // The presenter drawer (M3 Task 5)
    // =================================================================================
    //
    // **An overlay, never a member of the layout `Column`.** It is anchored to the window's
    // right edge and spans the full height, so opening it cannot move the hero, the tiles,
    // the badges or the verdict lines - `tests/test_ui_contract.py` asserts the positions
    // are identical before and after, item by item. Its width comes from
    // `bridgeUi.drawerWidth`, which is `app/theme.py`'s token, so no pixel count is here.
    //
    // `active: window.drawerOpen` means the drawer does not exist in the scene graph while
    // it is closed, so it cannot change the booth screen's font census or its captures.
    Loader {
        id: drawerLoader
        // Named so the render can read its **status**: a `Loader` whose component failed
        // to load leaves an empty rectangle that every "does the drawer exist" question
        // answers `false` to, and the failure has to be visible somewhere.
        objectName: "drawerLoader"
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        width: bridgeUi.drawerWidth
        active: window.drawerOpen
        source: "PresenterDrawer.qml"
    }

    // Escape closes the presenter drawer, and nothing else.
    //
    // A `Shortcut` rather than `Keys.onEscapePressed`: `Window` is not an `Item`, so the
    // attached `Keys` property cannot bind to it and Qt reports `Could not attach Keys
    // property to ... is not an Item` (measured). `Shortcut` works on a window, and it is
    // also the shape the design's keyboard section will grow into.
    Shortcut {
        objectName: "escapeShortcut"
        sequence: "Escape"
        onActivated: window.drawerOpen = false
    }

    // The drawer's keyboard entry point. Spec section 9's presenter mode: a presenter can
    // reach the explanations from the keyboard without a pointer.
    Shortcut {
        objectName: "drawerOpenShortcut"
        sequence: "F1"
        onActivated: window.drawerOpen = !window.drawerOpen
    }
}
