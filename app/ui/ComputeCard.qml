// ComputeCard.qml - one **camera's** card.
//
// M2 Task 5 changed what this card is about. It used to be a card for a **device** -
// CPU, GPU or NPU - with the device's utilisation as the headline figure. It is now a card
// for a **stream**, and the device that stream is really running on is one line of it.
// Three cameras, three cards, each reading only its own stream's record from
// `bridge.stream_cards_by_id[streamId]`.
//
// ## Why the headline is the frame age, and not the utilisation
//
// Measured, unsaturated, three streams on the CPU - which is literally spec section 5
// row 4, a state the booth will be in on a quiet day - gives per-stream frame age
// **p50 10767 / 11085 / 11319 ms** and **p95 11626 / 11905 / 12286 ms**, a backlog of
// **170-187 frames per stream**, about **13.5 of the 15 fps** analysed, **zero drops**,
// and all three streams reporting `active`.
//
// So the picture on screen is **eleven seconds old** and every other counter says the
// system is healthy. That is the most dangerous shape this project has produced, and it is
// why the age is the largest thing on the card, why the window it is taken over is
// printed next to it, and why the card carries a **word** - `STALE` - once the age passes
// the threshold spec section 7 quotes.
//
// ## Frame age is not inference cost
//
// On the NPU the p95 frame age is seconds and the inference itself is about 5.9 ms; the
// difference is queue wait. Two claims, two words. Nothing here adds them, and no single
// figure on this card is a blend of the two.
//
// ## Per device, not per screen
//
// The device utilisation line keeps M1's per-device wording exactly: the CPU card says
// "no CPU sample yet" because `psutil.cpu_percent` is called every 250 ms and works, and
// the GPU and NPU cards say "Utilization unavailable" because there is no counter for
// them and a machine-wide peak is not attributable to this process. A camera on the NPU
// therefore carries the accelerator's wording on **its own card**, which is the point of a
// per-stream layout: with three cameras on three devices there is one of each.
//
// ## The card is never dimmed
//
// M1 dimmed a card whose utilisation was unmeasurable, because the card had nothing else
// to show. This card has the frame age, the backlog, the placement and the reason, all of
// which are real, so dimming the card that tells a presenter the NPU stream is eleven
// seconds stale - because NPU utilisation cannot be measured - would dim the one warning
// that matters. The wording is on the line it belongs to; the card stays legible.

import QtQuick

Rectangle {
    id: card

    /// Which camera this card is about. Set by the delegate; the card reads nothing else.
    property string streamId: ""

    /// This camera's complete record, or a **cold record** when there is not one yet.
    ///
    /// The cold record is why no card can read a property of `undefined`: the bridge builds
    /// a full record for every id it has heard of, with `pending` state, `unavailable`
    /// figures and no device.
    readonly property var record: bridge.stream_cards_by_id[streamId] !== undefined
        ? bridge.stream_cards_by_id[streamId]
        : ({
            "stream": streamId, "state": "pending", "device": "",
            "requested": "", "confirmed": "", "fell_back": false,
            "badge": "no device is running",
            "age_p50_text": bridge.unavailable_text,
            "age_p95_text": bridge.unavailable_text,
            "fps_text": bridge.unavailable_text,
            "backlog_text": bridge.unavailable_text,
            "window_text": bridge.unavailable_text,
            "stale": false, "staleness": "", "reason": "",
            "summary_text": bridge.unavailable_text
        })

    /// This camera's **confirmed** device class, or `""` when nothing has confirmed.
    ///
    /// `""` is a real state and is rendered as `no device is running`: a failed route
    /// claims no device, and a card that borrowed the requested one would be claiming a
    /// placement that was never proved.
    readonly property string deviceId: String(record.device)

    /// The optional metric that describes this camera's device's utilisation.
    ///
    /// A name mapping, not a figure: the number itself comes from the bridge's notified
    /// `metric_text`. `""` for a stream with no device, and the utilisation line then says
    /// so instead of printing a utilisation for a device that is not running.
    readonly property string metricName: deviceId === "CPU" ? "system_cpu_percent"
        : (deviceId === "GPU" ? "gpu_utilisation_percent"
            : (deviceId === "NPU" ? "npu_utilisation_percent" : ""))

    readonly property bool utilisationAvailable: metricName.length > 0
        && bridge.metric_text[metricName] !== undefined
        && bridge.metric_text[metricName] !== bridge.unavailable_text

    /// The rendered utilisation figure for this camera's device, or the word.
    readonly property string metricText: metricName.length > 0
        && bridge.metric_text[metricName] !== undefined
        ? bridge.metric_text[metricName] : bridge.unavailable_text

    /// The per-device "why there is no number here", kept verbatim from M1.
    readonly property string unavailableNotice: deviceId.length === 0
        ? "no device is running"
        : (bridge.unavailable_notice[deviceId] !== undefined
            ? bridge.unavailable_notice[deviceId] : bridge.unavailable_text)

    /// The window the **device figure** was averaged over, or the bridge's word.
    ///
    /// **Not the frame-age window.** The left column's `window: N analysed frames` says how
    /// many frames the age is a percentile of; this says how long `psutil` averaged over for
    /// the utilisation figure. Two different claims about two different measurements, kept
    /// on two different lines - merging them would produce a sentence that is true of
    /// neither.
    readonly property string windowNote: metricName.length > 0
        && bridge.window_notes[metricName] !== undefined
        ? bridge.window_notes[metricName] : bridge.unavailable_text

    /// The state word, from **this stream's own** state and never the aggregate.
    readonly property string stateText: String(record.state)

    readonly property bool isFailed: stateText === "failed"

    /// The headline: this camera's measured p50 frame age, already formatted.
    readonly property string ageText: String(record.age_p50_text)

    /// The activity line, composed by the bridge. Frame age and rate only.
    readonly property string activityText: String(record.summary_text)

    /// Which explanation this card pins when clicked. See `explanationKeys` in Bridge.qml.
    property string explainKey: "cpu"

    /// The explanation's text, resolved from the bridge rather than duplicated here.
    readonly property string explanationText: bridge.explanation(explainKey)

    /// The camera's colour: good while it is running, bad once it has failed, neutral
    /// while it is still proving where it runs.
    ///
    /// **Colour is never the only signal**: the state is also a word on the card, and
    /// `STALE` is a word, and both are readable without colour.
    readonly property color accentColor: isFailed ? Theme.COLOR_BAD
        : (stateText === "active" ? Theme.COLOR_GOOD : Theme.COLOR_NEUTRAL)

    color: Theme.BG_PANEL
    radius: Theme.RADIUS
    border.color: isFailed ? Theme.COLOR_BAD
        : (record.stale ? Theme.COLOR_WARN : Theme.BORDER)
    // **A stale camera's card is outlined in the warning colour**, so the thing that
    // matters is visible across the room, not only to a reader of small text.
    border.width: (isFailed || record.stale) ? 2 : 1

    signal explainRequested(string key)

    // -- left column: the camera, its state, and how old the picture is ---------------
    //
    // **Two columns, and the split is 58/42.** M1's card was two columns because a single
    // stack of six rows did not fit the ~220 px the three cards share. The same arithmetic
    // applies here, and the first render of this layout showed it: the age read
    // `unavaila...` because `TYPE_NUMERAL` at 56 px does not fit "unavailable" in a 46%
    // column of a 590 px card. The age is therefore `TYPE_NUMERAL_SMALL` - still the
    // largest text on the card, since everything else is `TYPE_DETAIL` or `TYPE_LABEL` -
    // and the left column takes 58% so the figure and the window sit on one line each.
    Column {
        id: identityColumn
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.margins: Theme.SPACING.md
        anchors.right: rightColumn.left
        anchors.rightMargin: Theme.SPACING.md
        spacing: Theme.SPACING.xs

        Row {
            spacing: Theme.SPACING.sm

            Rectangle {
                width: Theme.SPACING.md
                height: Theme.SPACING.md
                radius: width / 2
                color: card.accentColor
                anchors.verticalCenter: parent.verticalCenter
            }

            Text {
                anchors.verticalCenter: parent.verticalCenter
                text: "Camera " + card.streamId
                color: Theme.TEXT_ON_PANEL
                font.family: Theme.FONT_FAMILY
                font.pixelSize: Theme.TYPE_LABEL
                font.bold: true
            }

            // The state as a word, per this camera only.
            Text {
                anchors.verticalCenter: parent.verticalCenter
                text: card.stateText
                color: card.accentColor
                font.family: Theme.FONT_FAMILY
                font.pixelSize: Theme.TYPE_LABEL
                font.bold: true
            }
        }

        // The measured age. The biggest thing on the card, because in the condition this
        // card exists for it is the only number that tells the presenter anything.
        Text {
            width: parent.width
            objectName: "streamAgeText_" + card.streamId
            text: card.ageText
            color: record.stale ? Theme.COLOR_WARN : Theme.TEXT_ON_PANEL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_NUMERAL_SMALL
            font.bold: true
            elide: Text.ElideRight
        }

        // **The word, not only the number.**
        Text {
            width: parent.width
            objectName: "streamStaleText_" + card.streamId
            visible: card.record.staleness !== ""
            text: card.record.staleness
            color: Theme.COLOR_WARN
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            font.bold: true
            wrapMode: Text.WordWrap
            maximumLineCount: 2
            elide: Text.ElideRight
        }
        // The window, because a percentile of a bounded deque is not comparable with a
        // percentile of a different one, and the age above is a percentile.
        Text {
            width: parent.width
            objectName: "streamWindowText_" + card.streamId
            text: "window: " + card.record.window_text
            color: Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            elide: Text.ElideRight
        }

        // The rate and both ages, on this camera's card rather than in the header, because
        // they are about this camera.
        Text {
            width: parent.width
            objectName: "streamActivityText_" + card.streamId
            visible: card.activityText.length > 0
            text: card.activityText
            color: Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            wrapMode: Text.WordWrap
            // **Two lines, not three.** The first render put the window on this line too
            // and the third line of it was drawn *below* the card's own bottom edge - a
            // Repeater delegate has no implicit clipping, so the overflow was drawn over
            // the pipeline strip. The window has its own labelled line now, and this line
            // is the rate and the two ages.
            maximumLineCount: 2
            elide: Text.ElideRight
        }
    }

    // -- right column: placement, backlog, and this camera's reason -------------------
    Column {
        id: rightColumn
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.margins: Theme.SPACING.md
        width: Math.max(Theme.TYPE_NUMERAL_SMALL * 4, parent.width * 0.42)
        spacing: Theme.SPACING.xs

        // The placement, from the **confirmed** reading. The requested device rides along
        // inside `badge` only when it differs, so a fallback is visible as a fallback.
        Rectangle {
            width: parent.width
            height: Theme.TYPE_DETAIL + Theme.SPACING.sm * 2
            radius: Theme.RADIUS
            color: Theme.BG_PANEL_DEEP
            objectName: "streamBadgeBox_" + card.streamId

            Text {
                anchors.fill: parent
                anchors.leftMargin: Theme.SPACING.sm
                anchors.rightMargin: Theme.SPACING.sm
                verticalAlignment: Text.AlignVCenter
                objectName: "streamBadgeText_" + card.streamId
                text: card.record.badge
                color: card.record.confirmed.length > 0
                    ? Theme.COLOR_GOOD
                    : (card.isFailed ? Theme.COLOR_BAD : Theme.COLOR_NEUTRAL)
                font.family: Theme.FONT_FAMILY
                font.pixelSize: Theme.TYPE_DETAIL
                elide: Text.ElideRight
            }
        }

        // **Backlog and rate on their own lines, not one joined line.** The first render
        // elided the joined form to `analysis...`, which drops the half of the pair the
        // presenter needs least visible - and a backlog with no rate beside it looks like a
        // queue on its own.
        Text {
            width: parent.width
            objectName: "streamBacklogText_" + card.streamId
            text: "backlog " + card.record.backlog_text
            color: Theme.TEXT_ON_PANEL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            elide: Text.ElideRight
        }

        Text {
            width: parent.width
            objectName: "streamRateText_" + card.streamId
            text: "analysis " + card.record.fps_text
            color: Theme.TEXT_ON_PANEL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            elide: Text.ElideRight
        }

        // This camera's own utilisation figure, with this **device's** wording.
        Text {
            width: parent.width
            // **Empty when there is no device**, so a failed camera does not leave an item
            // named `cardText_` - with no metric in the name, which is exactly the kind of
            // half-named handle the M1 card-text test was rewritten to stop reaching for.
            objectName: card.metricName.length > 0
                ? "cardText_" + card.metricName : ""
            visible: card.metricName.length > 0
            text: card.utilisationAvailable
                ? card.deviceId + " " + card.metricText
                : card.unavailableNotice
            color: card.utilisationAvailable ? Theme.TEXT_ON_PANEL : Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            elide: Text.ElideRight
        }

        // The **telemetry sampler's** window for this device's figure - a different claim
        // from the frame-age window on the left, and deliberately not merged with it: one
        // says how long `psutil` averaged over, the other says how many analysed frames
        // the age is a percentile of.
        //
        // Read from the bridge's **notified** `window_notes` map, never by calling
        // `window_text(...)`. That was a Slot call and therefore evaluated once, at
        // `engine.load`, when no sample existed - so this line never appeared in a live
        // session and its `visible:` guard was false throughout. The standing guard in
        // `tests/test_ui_contract.py` still bans the Slot call here.
        Text {
            width: parent.width
            objectName: "cardWindowNote_" + card.metricName
            visible: card.metricName.length > 0
                     && card.windowNote !== bridge.unavailable_text
            text: "measured " + card.windowNote
            color: Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            elide: Text.ElideRight
        }
        // supervisor's `errors` is all three concatenated, and a card showing that tells a
        // presenter which sentences are about the camera they are looking at.
        Text {
            width: parent.width
            objectName: "streamReasonText_" + card.streamId
            visible: card.record.reason !== ""
            text: card.record.reason
            color: card.isFailed ? Theme.COLOR_BAD : Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            wrapMode: Text.WordWrap
            maximumLineCount: 3
            elide: Text.ElideRight
        }
    }

    // Hover opens the explanation; a click pins it. The design wants both, because a
    // booth machine is often touch-only and hover alone is then unreachable.
    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.LeftButton
        onClicked: card.explainRequested(card.explainKey)
        onEntered: hoverHint.visible = true
        onExited: hoverHint.visible = false
    }

    // The hover hint. Short by design: the long copy is the pinned panel, and a long
    // explanation floating over the hero would obscure the picture.
    Rectangle {
        id: hoverHint
        visible: false
        anchors.left: parent.left
        anchors.bottom: parent.top
        anchors.bottomMargin: Theme.SPACING.sm
        width: hintText.implicitWidth + Theme.MARGIN
        height: hintText.implicitHeight + Theme.SPACING.sm * 2
        radius: Theme.RADIUS
        color: Theme.BG_PANEL_DEEP
        border.color: Theme.ACCENT
        border.width: 1
        opacity: 0.96

        Text {
            id: hintText
            anchors.centerIn: parent
            width: Math.min(implicitWidth, Theme.CANVAS_WIDTH * 0.28)
            text: "Camera " + card.streamId
                  + (card.deviceId.length > 0 ? " on " + card.deviceId : "")
                  + " - " + card.explanationText
            color: Theme.TEXT_ON_PANEL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            wrapMode: Text.WordWrap
        }
    }
}
