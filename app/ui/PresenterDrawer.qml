// PresenterDrawer.qml - the presenter's explanation and detail surface. **M3 Task 5,
// extended by M4 Task 1.**
//
// ## What this is for
//
// A booth visitor asks "what are those boxes?", and the presenter needs an answer that is
// on the screen rather than in their memory. M1 shipped a `?` explanation panel whose call
// **did not exist** - `Bridge.qml` had lost its `explain` function and the call sat behind
// a short-circuited ternary, so the six explanation buttons did nothing at all on the
// screen whose whole purpose is explaining what the boxes mean. So this drawer is a
// surface whose every affordance is reachable and named, and
// `tests/test_ui_contract.py` clicks each one and reads the text it produced.
//
// ## M4: what moved in here, and why
//
// The always-visible screen had to be rebuilt at ~44 px to read at 3 m (see
// `Main.qml`). Six rows per camera could not survive that, so they moved **here**, where a
// presenter can look closely and read at arm's length:
//
//   **per camera** - the backlog, the analysis rate, the frame-age window, the activity
//   line (rate and both frame ages), the device's utilisation figure with the device's own
//   wording, and the telemetry sampler's averaging window;
//
//   **once** - the event counts, the `status:` line, the telemetry-freshness line, and the
//   pipeline's topology in words (the six-stage strip itself is gone: twelve 18 px labels
//   cannot be read at 3 m, and at the drawer's 566 px inner width the stage names elide).
//
// **Every figure that left the always-visible screen is on this surface, with the same item
// name it had on the card**, so the tests that read it were retargeted rather than dropped.
// Two things were genuinely removed: the **standalone utilisation paragraph** that this
// drawer used to carry, whose claim is the `utilisation` explanation's first sentence (it
// was 105 px of the column the six per-camera rows needed), and the **horizontally-laid-out
// pipeline strip**, whose stage names do not fit 566 px without eliding - the topology is
// stated in `drawerPipelineText` instead.
//
// ## It moves nothing
//
// It is loaded as an **overlay** by `Main.qml`'s `Loader`, anchored to the window's right
// edge and spanning its full height. It is not part of the layout `Column`, so opening it
// cannot reflow the hero, the tiles, the badges or the verdict lines - and the test asserts
// every one of those positions is identical before and after, item by item, because
// "nothing threw" is not the same claim.
//
// ## The two honesty statements, and why they are on this surface
//
// - **The retail scope note, verbatim.** `bridge.scope_note` is
//   `app/engine/retail_config.py`'s `scope_note`, read from the shipped pack. The screen
//   must not paraphrase it: the delivered event is **people-in-zone presence, 2 events per
//   clip - not a people count, not a dwell measurement.** The header used to carry an
//   elided copy at 18 px, cut mid-word with nothing saying so; it is stated here in full
//   instead.
// - **GPU/NPU utilisation is unavailable, and why.** The per-device line on each camera
//   block carries the device's own wording, and the `utilisation` explanation carries the
//   measured reason: the platform's own counters fault this process when read, and a
//   machine-wide peak is **not attributable to this process**.
//
// ## No layout token is in this file
//
// Every size, spacing and colour comes from `Theme`, with one deliberate exception stated
// rather than hidden: `border.width: 2`, which `Main.qml` and `ComputeCard.qml` also use.
// The drawer's width is `bridgeUi.drawerWidth` (from `app/theme.py`, 0.32 of the canvas =
// 614 px measured) and is applied by the `Loader` in `Main.qml`; the component fills the
// box it is given with `anchors.fill: parent`.

import QtQuick

Rectangle {
    id: drawer

    objectName: "presenterDrawer"
    // **The Loader does not size its item.** Measured: without this the drawer rendered
    // **0 px wide** - every string was present and every position assertion would have
    // compared zeros - because the loaded item keeps its own geometry and only the Loader
    // is anchored. So the drawer fills the box the Loader gives it, which is
    // `bridgeUi.drawerWidth` from `app/theme.py`.
    anchors.fill: parent
    color: Theme.BG_PANEL
    radius: Theme.RADIUS
    border.color: Theme.ACCENT
    border.width: 2
    // The drawer is its own clip, so `tools/capture_ui.py`'s overflow measure is a
    // per-drawer one: content that outgrows the panel is named rather than painted over
    // the screen behind it.
    clip: true

    /// Which explanation is expanded below. Defaults to the detector, so the drawer always
    /// shows something rather than an empty panel that reads as a rendering failure.
    property string selectedKey: "yolo"

    /// The selected explanation's text, resolved through the bridge. A constant-safe Slot
    /// call (`SLOT_CALLS_SAFE_IN_A_QML_BINDING`): it reads `EXPLANATIONS`, and the argument
    /// changes on a click, so the binding re-evaluates.
    readonly property string selectedText: drawer.selectedKey.length > 0
        ? bridge.explanation(drawer.selectedKey) : ""

    Column {
        id: body
        anchors.fill: parent
        anchors.margins: Theme.MARGIN
        // **`sm`, not `GAP`.** The drawer now carries the six per-camera rows that left the
        // live screen plus everything it already had, and at 16 px of spacing the column is
        // taller than the 1032 px it is given. 8 px keeps every paragraph distinct and the
        // whole surface inside its own clip.
        spacing: Theme.SPACING.sm

        Text {
            objectName: "drawerTitle"
            width: parent.width
            text: "What the boxes mean"
            color: Theme.TEXT_ON_PANEL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_LABEL
            font.bold: true
        }

        // -- the reduction, in the shipped pack's own words --------------------------
        Text {
            objectName: "drawerScopeNote"
            width: parent.width
            text: bridge.scope_note
            color: Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            wrapMode: Text.WordWrap
        }

        // -- and the one-sentence statement of what the delivered event is -----------
        Text {
            objectName: "drawerScopeReduction"
            width: parent.width
            // **The running pack's own sentence.** The retail pack's text is unchanged
            // word for word, so the M4 render still reads the reduction it was pinned to;
            // another pack replaces it with what *that* pack actually delivers. A
            // `notified` property, not a Slot call.
            text: bridge.delivered_event
            color: Theme.TEXT_ON_PANEL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            font.bold: true
            wrapMode: Text.WordWrap
        }

        // -- every explanation, reachable and named ---------------------------------
        //
        // **`bridge.explanation_keys`, not `bridge.explanationKeys`.** The first version
        // asked the Python bridge for `explanationKeys`, which it does not have; the model
        // was `undefined` and this `Repeater` produced **zero delegates** - the drawer
        // opened with no explanation buttons, which is the M1 dead-affordance shape on the
        // one surface whose job is explaining the boxes. The keys come from
        // `app.ui.bridge`'s `EXPLANATIONS`, which the M1 `?` row also offered.
        //
        // `Flow`, not `Row`. **Measured:** the seven buttons are 49 + 79 + 55 + 45 + 46 +
        // 91 + 74 = 439 px plus six 8 px gaps = 487 px, inside the drawer's 566 px inner
        // width, so today they fit on one row and the two layouts render identically. `Flow`
        // is the shape that stays honest when the key set grows.
        Flow {
            width: parent.width
            spacing: Theme.SPACING.sm

            Repeater {
                model: bridge.explanation_keys

                delegate: Rectangle {
                    required property string modelData
                    objectName: "drawerExplainButton_" + modelData
                    width: drawerLabel.implicitWidth + Theme.SPACING.md
                    height: Theme.TYPE_DETAIL + Theme.SPACING.sm * 2
                    radius: Theme.RADIUS
                    color: drawer.selectedKey === modelData
                           ? Theme.ACCENT : Theme.BG_PANEL_DEEP
                    border.color: Theme.BORDER
                    border.width: 1

                    Text {
                        id: drawerLabel
                        anchors.centerIn: parent
                        text: modelData
                        color: Theme.COLOR_NEUTRAL
                        font.family: Theme.FONT_FAMILY
                        font.pixelSize: Theme.TYPE_DETAIL
                    }

                    MouseArea {
                        objectName: "drawerExplainMouse_" + modelData
                        anchors.fill: parent
                        onClicked: drawer.selectedKey = modelData
                    }
                }
            }
        }

        // -- the selected explanation, in full ---------------------------------------
        //
        // In the drawer rather than in a panel under it, because a panel under the drawer
        // is a button that sets text a presenter cannot see - the M1 dead affordance again.
        Text {
            objectName: "drawerExplanationText"
            width: parent.width
            text: drawer.selectedText
            color: Theme.TEXT_ON_PANEL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            wrapMode: Text.WordWrap
        }

        // -- M5 Task 4: the live device controls ----------------------------------------
        //
        // **These are the controls a playback must not have.** They write
        // `bridge.deviceControl`, a writable property rather than a Slot call in a binding
        // (the standing guard bans the latter, and an assignment is not a binding). In
        // recorded mode the button and its `MouseArea` are both `enabled: false`, and the
        // bridge's setter refuses the write as well - two independent reasons a click
        // cannot become a live action. The render test clicks one and reads the emitted
        // request list, so "disabled" is proved by behaviour and not by the flag alone.
        Row {
            width: parent.width
            spacing: Theme.SPACING.sm

            Text {
                objectName: "deviceControlLabel"
                anchors.verticalCenter: parent.verticalCenter
                text: "Live device control - Camera " + bridge.route_stream
                color: Theme.TEXT_ON_PANEL
                font.family: Theme.FONT_FAMILY
                font.pixelSize: Theme.TYPE_DETAIL
                font.bold: true
            }

            Repeater {
                model: ["CPU", "GPU", "NPU"]

                delegate: Rectangle {
                    id: deviceControlButton
                    required property string modelData
                    objectName: "deviceControl_" + modelData
                    enabled: !bridge.recorded_mode
                    opacity: enabled ? 1.0 : Theme.DISABLED_OPACITY
                    width: deviceControlText.implicitWidth + Theme.SPACING.md
                    height: Theme.TYPE_DETAIL + Theme.SPACING.sm * 2
                    radius: Theme.RADIUS
                    color: Theme.BG_PANEL_DEEP
                    border.color: bridge.device_control === modelData
                        ? Theme.ACCENT : Theme.BORDER
                    border.width: 1

                    Text {
                        id: deviceControlText
                        anchors.centerIn: parent
                        text: modelData
                        color: deviceControlButton.enabled ? Theme.TEXT_ON_PANEL
                                                           : Theme.COLOR_NEUTRAL
                        font.family: Theme.FONT_FAMILY
                        font.pixelSize: Theme.TYPE_DETAIL
                    }

                    MouseArea {
                        objectName: "deviceControlMouse_" + modelData
                        anchors.fill: parent
                        enabled: parent.enabled
                        onClicked: bridge.device_control = modelData
                    }
                }
            }
        }

        // Why the controls above will not move anything. Shown only while recorded, so it
        // cannot be read as a claim about a live run.
        Text {
            objectName: "deviceControlNotice"
            width: parent.width
            visible: bridge.recorded_mode
            text: bridge.recorded_control_note
            color: Theme.COLOR_WARN
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            wrapMode: Text.WordWrap
        }

        // -- M4: this camera, precisely -----------------------------------------------
        //
        // The six rows that left the always-visible screen. One block per declared camera,
        // reading **that camera's own record** and nothing else, so a figure cannot appear
        // under another camera's heading - the mislabelling this project has shipped twice.
        Repeater {
            model: bridge.stream_ids

            delegate: Column {
                required property string modelData
                objectName: "drawerCameraBlock_" + modelData
                // **M5 Task 4:** a playback shows only the recorded surface's own dated
                // figures. These blocks read the live bridge, so showing them behind a
                // Recorded demonstration notice would be a live number on a recorded screen.
                visible: !bridge.recorded_mode
                width: parent.width
                spacing: Theme.SPACING.xs

                // This camera's record, or a cold one. Never `undefined`.
                readonly property var record:
                    bridge.stream_cards_by_id[modelData] !== undefined
                    ? bridge.stream_cards_by_id[modelData]
                    : ({ "state": "pending", "device": "", "confirmed": "",
                         "badge": "no device is running",
                         "age_p50_text": bridge.unavailable_text,
                         "age_p95_text": bridge.unavailable_text,
                         "fps_text": bridge.unavailable_text,
                         "backlog_text": bridge.unavailable_text,
                         "window_text": bridge.unavailable_text,
                         "summary_text": bridge.unavailable_text,
                         "stale": false, "staleness": "", "reason": "" })

                readonly property string deviceId: String(record.device)

                // **The metric name mapping, and it lives here now** because this is the
                // only surface that renders a device utilisation figure. The keys are the
                // bridge's (`system_cpu_percent`, `gpu_utilisation_percent`,
                // `npu_utilisation_percent`), and a device with no counter gets the word,
                // never 0.
                readonly property string metricName: deviceId === "CPU"
                    ? "system_cpu_percent"
                    : (deviceId === "GPU" ? "gpu_utilisation_percent"
                        : (deviceId === "NPU" ? "npu_utilisation_percent" : ""))

                readonly property bool utilisationAvailable: metricName.length > 0
                    && bridge.metric_text[metricName] !== undefined
                    && bridge.metric_text[metricName] !== bridge.unavailable_text

                readonly property string metricText: metricName.length > 0
                    && bridge.metric_text[metricName] !== undefined
                    ? bridge.metric_text[metricName] : bridge.unavailable_text

                readonly property string unavailableNotice: deviceId.length === 0
                    ? "no device is running"
                    : (bridge.unavailable_notice[deviceId] !== undefined
                        ? bridge.unavailable_notice[deviceId] : bridge.unavailable_text)

                readonly property string windowNote: metricName.length > 0
                    && bridge.window_notes[metricName] !== undefined
                    ? bridge.window_notes[metricName] : bridge.unavailable_text

                // The camera, its confirmed device, its device figure and the sampler's
                // window - **one row, three items**. At 18 px these together are 538 px of
                // the drawer's 566 px inner width, and a line each is 48 px the drawer does
                // not have in the worst case (the longest explanation, three cameras, three
                // telemetry windows). Each item keeps its own name, so a test still reads
                // each claim separately.
                Row {
                    width: parent.width
                    spacing: Theme.SPACING.sm

                    Text {
                        objectName: "drawerCameraHeading_" + modelData
                        text: "Camera " + modelData + " \u00b7 "
                              + (record.confirmed.length > 0 ? record.confirmed
                                                            : record.badge)
                        color: Theme.TEXT_ON_PANEL
                        font.family: Theme.FONT_FAMILY
                        font.pixelSize: Theme.TYPE_DETAIL
                        font.bold: true
                    }

                    Text {
                        objectName: metricName.length > 0 ? "cardText_" + metricName : ""
                        text: utilisationAvailable
                            ? deviceId + " " + metricText : unavailableNotice
                        color: utilisationAvailable ? Theme.TEXT_ON_PANEL : Theme.COLOR_NEUTRAL
                        font.family: Theme.FONT_FAMILY
                        font.pixelSize: Theme.TYPE_DETAIL
                        elide: Text.ElideRight
                    }

                    Text {
                        objectName: "cardWindowNote_" + metricName
                        visible: metricName.length > 0
                                 && windowNote !== bridge.unavailable_text
                        text: "measured " + windowNote
                        color: Theme.COLOR_NEUTRAL
                        font.family: Theme.FONT_FAMILY
                        font.pixelSize: Theme.TYPE_DETAIL
                        elide: Text.ElideRight
                    }
                }

                // Backlog and rate, on one line and on two items: the two numbers are
                // different claims (a queue length and a rate), so each keeps its own name.
                Row {
                    width: parent.width
                    spacing: Theme.SPACING.md

                    Text {
                        objectName: "streamBacklogText_" + modelData
                        text: "backlog " + record.backlog_text
                        color: Theme.TEXT_ON_PANEL
                        font.family: Theme.FONT_FAMILY
                        font.pixelSize: Theme.TYPE_DETAIL
                        elide: Text.ElideRight
                    }

                    Text {
                        objectName: "streamRateText_" + modelData
                        text: "analysis " + record.fps_text
                        color: Theme.TEXT_ON_PANEL
                        font.family: Theme.FONT_FAMILY
                        font.pixelSize: Theme.TYPE_DETAIL
                        elide: Text.ElideRight
                    }
                }

                Text {
                    objectName: "streamWindowText_" + modelData
                    width: parent.width
                    text: "window: " + record.window_text
                    color: Theme.COLOR_NEUTRAL
                    font.family: Theme.FONT_FAMILY
                    font.pixelSize: Theme.TYPE_DETAIL
                    elide: Text.ElideRight
                }

                // The rate and both ages: the activity line, unchanged and still the only
                // place the p95 is stated.
                Text {
                    objectName: "streamActivityText_" + modelData
                    width: parent.width
                    text: record.summary_text
                    color: Theme.COLOR_NEUTRAL
                    font.family: Theme.FONT_FAMILY
                    font.pixelSize: Theme.TYPE_DETAIL
                    wrapMode: Text.WordWrap
                    maximumLineCount: Theme.CARD_ACTIVITY_LINES
                    elide: Text.ElideRight
                }
            }
        }

        // -- the counts and the pipeline, in words -----------------------------------
        //
        // **The event row (M2 Task 5, R3).** Gated on **frames analysed**, not on the
        // route: a confirmed placement arrives from the worker's route before any frame
        // has been analysed, and gating on it printed three hard zeros for an event
        // machine that had not run a single frame. A kind the bridge has never heard of
        // reads 0 **once something could have counted one**, and `unavailable` before that.
        Text {
            objectName: "eventCountRow"
            width: parent.width
            visible: !bridge.recorded_mode
            readonly property var kinds: ["zone_occupancy", "candidate_transfer",
                                          "candidate_return"]
            readonly property bool countedSomethingYet: bridge.frames_analysed > 0
            text: kinds.map(k => k + " " + (
                      !countedSomethingYet ? "unavailable"
                      : (bridge.event_counts[k] !== undefined ? bridge.event_counts[k] : 0)
                  )).join("    ")
            color: Theme.TEXT_ON_PANEL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            wrapMode: Text.WordWrap
        }

        // **The six-stage strip, in one sentence.** `PipelineStrip.qml` carried twelve 18 px
        // labels - six stage names and six device badges - which cannot be read at 3 m and
        // which elide at this drawer's 566 px inner width. The claim it made is kept, in the
        // words it made it in: only `infer` runs on the chosen device, and every other stage
        // runs on the CPU because that is where this build's shared decode and
        // postprocessing path is.
        Text {
            objectName: "drawerPipelineText"
            width: parent.width
            text: "Pipeline: decode, preprocess, inference, decode detections, track, "
                  + "event. Only inference runs on the chosen device; the other five "
                  + "stages run on the CPU."
            color: Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            wrapMode: Text.WordWrap
        }

        Text {
            objectName: "drawerStatusText"
            width: parent.width
            visible: !bridge.recorded_mode
            // **The running pack and clip, on the presenter's own surface.** The header
            // has 291 px between the title and the explanation button at 44 px; a clip id
            // does not fit there without shrinking the type, so the selection lives here,
            // where a presenter reads at arm's length. A notified property, not a Slot.
            text: "status: " + (bridge.status_message.length > 0
                                  ? bridge.status_message : "running")
                  + "  \u00b7  " + bridge.selection_label
            color: Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            elide: Text.ElideRight
        }

        // The telemetry sampler's freshness. `bridge.staleness` is a **notified property**,
        // not a Slot call: `staleness_text()` reads live state and a binding over it would
        // be evaluated once at `engine.load`.
        Text {
            objectName: "stalenessText"
            width: parent.width
            visible: !bridge.recorded_mode
            text: bridge.staleness.length > 0
                ? "telemetry " + bridge.staleness : "telemetry unavailable"
            color: Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
        }
    }
}
