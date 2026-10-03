// Main.qml - the booth screen.
//
// 1920 x 1080 logical canvas. ~68% video / 32% metrics across, and 12% header /
// 66% main / 22% pipeline+controls down, per the design's section 9. Every one of those
// numbers comes from `Theme`, which reads `app/theme.py`; there is no pixel count in this
// file.
//
// What is deliberately always visible:
//
//   - "Recorded footage \u2022 live inference", top left, unconditional.
//   - The confirmed device badge, which reads "pending" until an inference has proved a
//     placement and "failed - no device is running" once the route has given up.
//   - The error line, whenever there is one. It is never cleared implicitly.
//
// What is deliberately NOT here: a GPU or NPU utilisation number. Utilisation is not
// measurable on this machine and a machine-wide peak is not attributable to this process,
// so those two cards read "Utilization unavailable" and are visibly disabled. See
// ComputeCard.qml.

import QtQuick
import QtQuick.Window

Window {
    id: window

    width: Theme.CANVAS_WIDTH
    height: Theme.CANVAS_HEIGHT
    visible: true
    color: Theme.TEXT_ON_PANEL
    title: "Intel Edge AI Roadshow - " + bridgeUi.recordedFootageLabel

    // The pinned explanation panel. Escape closes it; a click on a card's "?" pins it.
    // Long copy must never obscure the hero, so it sits over the metrics column.
    property string pinnedExplanationKey: ""
    readonly property string pinnedExplanationText: pinnedExplanationKey.length > 0
        ? bridgeUi.explain(pinnedExplanationKey) : ""

    // The id is lower-case on purpose: QML rejects an id beginning with an uppercase
    // letter, and `Bridge { id: Bridge }` does not load. Measured: the window came up
    // with zero root objects and the only diagnostic was `IDs cannot start with an
    // uppercase letter` at the Bridge line.
    Bridge { id: bridgeUi }

    Rectangle {
        id: screen
        anchors.fill: parent
        // The window's own background. Distinct from every text colour: this fill was
        // once `Theme.TEXT_ON_PANEL` by accident, and the first render came back with
        // 842178 near-white pixels - a light screen with dark text, which is legible and
        // completely wrong for a booth (measured on the offscreen render).
        color: Theme.BG

        // The three bands, as fractions of the height that is actually available.
        //
        // **Not** `CANVAS_HEIGHT * fraction`: that is 1080 * (0.12 + 0.66 + 0.22) = 1080
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
            // Header - 12% of the available height
            // =====================================================================
            Item {
                width: parent.width
                height: screen.headerBand

                Row {
                    anchors.left: parent.left
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: Theme.SPACING.lg

                    // THE MANDATORY LABEL. Unconditional, never inside a conditional.
                    Rectangle {
                        anchors.verticalCenter: parent.verticalCenter
                        width: recordedLabelText.implicitWidth + Theme.MARGIN
                        height: recordedLabelText.implicitHeight + Theme.SPACING.md
                        radius: Theme.RADIUS
                        color: Theme.BG_PANEL_DEEP
                        border.color: Theme.ACCENT
                        border.width: 2

                        Text {
                            id: recordedLabelText
                            anchors.centerIn: parent
                            text: bridgeUi.recordedFootageLabel
                            color: Theme.TEXT_ON_PANEL
                            font.family: Theme.FONT_FAMILY
                            font.pixelSize: Theme.TYPE_LABEL
                            font.bold: true
                        }
                    }

                    Text {
                        anchors.verticalCenter: parent.verticalCenter
                        text: "Intel Edge AI Roadshow"
                        color: Theme.TEXT_ON_PANEL
                        font.family: Theme.FONT_FAMILY
                        font.pixelSize: Theme.TYPE_HEADLINE
                        font.bold: true
                    }
                }

                // Right of the header: the route badge and the reduction note.
                Column {
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: Theme.SPACING.xs

                    Rectangle {
                        id: routeBadge
                        anchors.right: parent.right
                        width: routeText.implicitWidth + Theme.MARGIN
                        height: routeText.implicitHeight + Theme.SPACING.sm * 2
                        radius: Theme.RADIUS
                        // Green only when a placement is confirmed. "pending" and
                        // "failed" are the neutral and the bad colour respectively, and
                        // both are also words.
                        color: bridgeUi.deviceConfirmed ? Theme.COLOR_GOOD : Theme.BG_PANEL_DEEP
                        border.color: bridge.route_state === "failed"
                            ? Theme.COLOR_BAD
                            : (bridgeUi.deviceConfirmed ? Theme.COLOR_GOOD : Theme.BORDER)
                        border.width: 2

                        Text {
                            id: routeText
                            // The one nameable handle on the device badge. It exists so a
                            // test can read the **QML-bound** string after `set_route`
                            // rather than reading the bridge's Python property - the
                            // badge was `constant=True` with no notify signal, so the
                            // Python side updated and the screen never did, and only a
                            // read of the bound text can tell the two apart.
                            objectName: "deviceBadgeText"
                            anchors.centerIn: parent
                            text: "device: " + bridgeUi.deviceBadge
                            color: bridgeUi.deviceConfirmed
                                ? Theme.TEXT_ON_ACCENT : Theme.COLOR_NEUTRAL
                            font.family: Theme.FONT_FAMILY
                            font.pixelSize: Theme.TYPE_LABEL
                            font.bold: true
                        }
                    }

                    // The reduction note. **Elided to a bound, because it is a sentence and a half and
                    // the header is 12% of the canvas.** The first render left it
                    // unbounded and it ran across the whole screen, over the mandatory
                    // label and off both edges (measured: the note's text starts at x=0
                    // while its right-anchored column begins at x=1032, so the two
                    // overlap in the middle). The full text is one click away on the
                    // `zone` explanation button.
                    Text {
                        id: scopeNoteText
                        anchors.right: parent.right
                        width: Math.min(implicitWidth,
                                        screen.width * Theme.VIDEO_WIDTH_FRACTION * 0.42)
                        horizontalAlignment: Text.AlignRight
                        text: bridge.scope_note
                        color: Theme.COLOR_NEUTRAL
                        font.family: Theme.FONT_FAMILY
                        font.pixelSize: Theme.TYPE_DETAIL
                        elide: Text.ElideRight
                    }
                }
            }

            // =====================================================================
            // Main - 66%:  hero + two thumbnails on the left, cards on the right
            // =====================================================================
            Item {
                width: parent.width
                height: screen.mainBand

                // -- hero, 68% of the width -------------------------------------
                Item {
                    id: heroColumn
                    width: Math.round(parent.width * Theme.VIDEO_WIDTH_FRACTION)
                    height: parent.height

                    VideoTile {
                        id: hero
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        // The hero takes all but the thumbnail row's height.
                        height: parent.height - thumbRow.height - Theme.GAP
                        hero: true
                        sourceName: "hero"
                        drawOverlay: true
                        explainKey: "yolo"
                    }

                    // -- two thumbnails -------------------------------------------
                    Item {
                        id: thumbRow
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.bottom: parent.bottom
                        height: parent.height * 0.16

                        Row {
                            anchors.fill: parent
                            spacing: Theme.GAP

                            VideoTile {
                                width: (parent.width - Theme.GAP) / 2
                                height: parent.height
                                hero: false
                                sourceName: "thumb0"
                                drawOverlay: false
                                caption: "Camera A \u00b7 " + bridge.device_confirmed
                                explainKey: "tracking"
                            }

                            VideoTile {
                                width: (parent.width - Theme.GAP) / 2
                                height: parent.height
                                hero: false
                                sourceName: "thumb1"
                                drawOverlay: false
                                caption: "same clip, replayed copy \u00b7 not a second camera"
                                explainKey: "zone"
                            }
                        }
                    }
                }

                // -- metrics column, 32% -----------------------------------------
                Column {
                    anchors.left: heroColumn.right
                    anchors.leftMargin: Theme.GAP
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.bottom: parent.bottom
                    spacing: Theme.GAP

                    // -- one card per CAMERA, not per device (M2 Task 5) -------------
                    //
                    // The card is about a stream: its own confirmed device, its own frame
                    // age with the window that age is taken over, its own backlog and
                    // analysis rate, and its own stated reason. Three cameras therefore
                    // produce three cards, and the `Repeater` builds one per stream the
                    // supervisor actually owns rather than three fixed ones - a supervisor
                    // with one stream draws one card, and one with four draws four, so a
                    // card can never exist for a camera that is not there.
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
                            // no QML warning at all, because a `NaN` geometry is legal. The
                            // model's own length is always in scope.
                            height: Math.max(Theme.TYPE_NUMERAL,
                                             parent.height / Math.max(1, bridge.stream_ids.length))
                            explainKey: "cpu"
                            onExplainRequested: key => window.pinnedExplanationKey = key
                        }
                    }
                }
            }

            // =====================================================================
            // Footer - 22%: pipeline strip, event counts, error line, controls
            // =====================================================================
            Item {
                width: parent.width
                height: screen.footerBand

                Column {
                    anchors.fill: parent
                    spacing: Theme.GAP

                    PipelineStrip {
                        id: pipeline
                        width: parent.width
                        height: Theme.TYPE_LABEL + Theme.SPACING.lg
                    }

                    Item {
                        width: parent.width
                        height: parent.height - pipeline.height - controlBar.height
                               - Theme.GAP * 2

                        // -- business events --------------------------------------
                        Row {
                            anchors.left: parent.left
                            anchors.verticalCenter: parent.verticalCenter
                            spacing: Theme.SPACING.md

                            Text {
                                text: "events:"
                                color: Theme.COLOR_NEUTRAL
                                font.family: Theme.FONT_FAMILY
                                font.pixelSize: Theme.TYPE_LABEL
                            }

                            // One Text rather than a Repeater of three.
                            //
                            // The row is three small numbers that change together, so a
                            // single binding is simpler and is readable as a whole - and
                            // unlike the Repeater's delegates it carries an `objectName` a
                            // test can actually find, so a regression that makes the row say
                            // something wrong is visible rather than assumed.
                            Text {
                                objectName: "eventCountRow"
                                // Gated on **frames analysed**, not on the route. A
                                // confirmed placement is established from the worker's
                                // route, which arrives before any frame has been analysed -
                                // so gating on `device_confirmed` left a real window, route
                                // confirmed and zero frames analysed, in which this row
                                // printed three hard zeros for an event machine that had not
                                // run a single frame.
                                //
                                // A kind the bridge has never heard of reads 0 **once
                                // something could have counted one**, and "unavailable"
                                // before that. The earlier wording called a bare 0 "a real
                                // count: the machine decided there were none", which is only
                                // true once the machine has looked at something.
                                readonly property var kinds: ["zone_occupancy",
                                                             "candidate_transfer",
                                                             "candidate_return"]
                                readonly property bool countedSomethingYet:
                                    bridge.frames_analysed > 0
                                text: kinds.map(k => k + " " + (
                                          !countedSomethingYet ? "unavailable"
                                          : (bridge.event_counts[k] !== undefined
                                             ? bridge.event_counts[k] : 0)
                                      )).join("    ")
                                color: Theme.TEXT_ON_PANEL
                                font.family: Theme.FONT_FAMILY
                                font.pixelSize: Theme.TYPE_DETAIL
                            }
                        }

                        // -- the error line ---------------------------------------
                        // Visible whenever there is an error, and never cleared
                        // implicitly: an error that disappears on the next good frame is
                        // an error nobody saw.
                        Rectangle {
                            anchors.right: parent.right
                            anchors.verticalCenter: parent.verticalCenter
                            width: errorText.implicitWidth + Theme.MARGIN
                            height: errorText.implicitHeight + Theme.SPACING.sm * 2
                            visible: bridge.error_message.length > 0
                            radius: Theme.RADIUS
                            color: Theme.BG_PANEL_DEEP
                            border.color: Theme.COLOR_BAD
                            border.width: 2

                            Text {
                                id: errorText
                                anchors.centerIn: parent
                                width: Math.min(implicitWidth, Theme.CANVAS_WIDTH * 0.4)
                                text: bridge.error_message
                                color: Theme.COLOR_BAD
                                font.family: Theme.FONT_FAMILY
                                font.pixelSize: Theme.TYPE_DETAIL
                                elide: Text.ElideRight
                            }
                        }
                    }

                    // -- control bar ---------------------------------------------
                    Rectangle {
                        id: controlBar
                        width: parent.width
                        height: Theme.TYPE_LABEL + Theme.SPACING.lg
                        radius: Theme.RADIUS
                        color: Theme.BG_PANEL

                        Row {
                            anchors.left: parent.left
                            anchors.leftMargin: Theme.MARGIN
                            anchors.verticalCenter: parent.verticalCenter
                            spacing: Theme.SPACING.lg

                            Text {
                                text: "status: " + (bridge.status_message.length > 0
                                                      ? bridge.status_message
                                                      : "running")
                                color: Theme.COLOR_NEUTRAL
                                font.family: Theme.FONT_FAMILY
                                font.pixelSize: Theme.TYPE_DETAIL
                            }

                            Text {
                                // Named so a test can read the rendered string; a bridge
                                // property being correct does not mean the screen is.
                                objectName: "stalenessText"
                                text: bridge.staleness.length > 0
                                    ? "telemetry " + bridge.staleness
                                    : "telemetry unavailable"
                                color: Theme.COLOR_NEUTRAL
                                font.family: Theme.FONT_FAMILY
                                font.pixelSize: Theme.TYPE_DETAIL
                            }
                        }

                        Row {
                            anchors.right: parent.right
                            anchors.rightMargin: Theme.MARGIN
                            anchors.verticalCenter: parent.verticalCenter
                            spacing: Theme.SPACING.sm

                            // The "?" buttons pin an explanation. All six keys are here,
                            // so every required explanation is reachable from the screen
                            // itself rather than from a presenter's memory.
                            Repeater {
                                model: bridgeUi.explanationKeys

                                delegate: Rectangle {
                                    required property string modelData
                                    width: helpLabel.implicitWidth + Theme.SPACING.md
                                    height: Theme.TYPE_DETAIL + Theme.SPACING.sm * 2
                                    radius: Theme.RADIUS
                                    color: window.pinnedExplanationKey === modelData
                                           ? Theme.ACCENT : Theme.BG_PANEL_DEEP
                                    border.color: Theme.BORDER
                                    border.width: 1

                                    Text {
                                        id: helpLabel
                                        anchors.centerIn: parent
                                        text: modelData
                                        color: Theme.COLOR_NEUTRAL
                                        font.family: Theme.FONT_FAMILY
                                        font.pixelSize: Theme.TYPE_DETAIL
                                    }

                                    MouseArea {
                                        anchors.fill: parent
                                        onClicked: window.pinnedExplanationKey =
                                            (window.pinnedExplanationKey === modelData
                                             ? "" : modelData)
                                    }
                                }
                            }
                        }
                    }
                }

                // -- the pinned explanation panel -------------------------------
                // Over the metrics column, never over the hero: a long explanation must
                // not obscure the picture.
                Rectangle {
                    visible: window.pinnedExplanationText.length > 0
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    width: parent.width * Theme.VIDEO_WIDTH_FRACTION * 0.9
                    height: Math.min(pinnedText.implicitHeight + Theme.MARGIN * 2,
                                     parent.height)
                    color: Theme.BG_PANEL
                    radius: Theme.RADIUS
                    border.color: Theme.ACCENT
                    border.width: 2

                    Text {
                        id: pinnedText
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: Theme.MARGIN
                        text: window.pinnedExplanationText
                        color: Theme.TEXT_ON_PANEL
                        font.family: Theme.FONT_FAMILY
                        font.pixelSize: Theme.TYPE_DETAIL
                        wrapMode: Text.WordWrap
                    }
                }
            }
        }
    }

    // Escape closes the pinned panel and nothing else.
    //
    // A `Shortcut` rather than `Keys.onEscapePressed`: `Window` is not an `Item`, so the
    // attached `Keys` property cannot bind to it and Qt reports `Could not attach Keys
    // property to ... is not an Item` (measured). `Shortcut` works on a window, and it is
    // also the shape the design's keyboard section will grow into.
    Shortcut {
        sequence: "Escape"
        onActivated: window.pinnedExplanationKey = ""
    }
}