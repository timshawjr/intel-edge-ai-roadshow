// ComputeCard.qml - one device card: CPU, GPU or NPU.
//
// The card has two jobs and they are not the same job. The CPU card shows a **measured**
// utilisation. The GPU and NPU cards show "Utilization unavailable", because utilisation
// is not measurable on this machine (the platform's own counters fault this process when
// read) and a machine-wide peak is not attributable to this process either. Those cards
// are **visibly disabled** and they show what is real instead: the confirmed placement.
//
// The temptation this file is shaped against is filling the GPU and NPU cards with the
// same kind of number the CPU card has, because an empty card reads as broken hardware
// and a booth visitor concludes the accelerator is faulty. It is not faulty; the
// measurement does not exist. So the card says exactly that, in words.
//
// Colour is never the only signal: every card carries its device name as text and its
// state as a word, per the design's requirement that CPU/GPU/NPU carry consistent colours
// *plus* text labels.
//
// Every optional number comes from `bridge.display_text(name)`. There is no arithmetic
// here and no `?:` that could turn an absent value into a 0.
//
// LAYOUT: two columns rather than one stack. A single column of six rows does not fit the
// card's real height - the three cards share the 66% main band, so each is about 220 px -
// and the first offscreen render overflowed the CPU card's badge into the GPU card
// (measured: "confirmed: running here" drawn over the GPU heading). Two columns need
// about 140 px, so there is room and nothing spills.

import QtQuick

Rectangle {
    id: card

    /// "CPU" | "GPU" | "NPU". Rendered as text beside the colour.
    property string deviceId: "CPU"
    /// Whether this card's utilisation figure is measurable at all.
    property bool utilisationAvailable: false
    /// The optional metric name this card renders, e.g. "system_cpu_percent".
    property string metricName: ""
    /// Optional second metric, e.g. "app_cpu_percent" on the CPU card.
    property string secondaryMetricName: ""
    /// Inference activity, which is real on every device.
    property string activityText: ""
    /// Which explanation this card pins when clicked. See `explanationKeys` in Bridge.qml.
    property string explainKey: "cpu"

    /// The explanation's text, resolved from the bridge rather than duplicated here.
    readonly property string explanationText: bridge.explanation(explainKey)

    /// The window the primary figure was averaged over, or "unavailable". Resolved once
    /// here so it is one named binding rather than an expression in a delegate. The
    /// bridge branches on `validity` before it reads `window_seconds`, so an absent
    /// reading never divides by a zero window here or anywhere else.
    readonly property string windowNote: metricName.length > 0
        ? bridge.window_text(metricName) : ""

    /// Whether the runtime confirmed the work ran on *this* device.
    ///
    /// Compared against `bridge.device_confirmed`, which is the confirmed
    /// `EXECUTION_DEVICES` reading and nothing else. The requested device is deliberately
    /// not consulted: a card that asked "did we get what we asked for" would read
    /// "running here" on a route that fell back, which is review focus #5 exactly.
    ///
    /// `indexOf` rather than equality because the confirmed reading for the GPU is
    /// `'GPU.0'`, not `'GPU'`.
    readonly property bool confirmedHere:
        bridge.device_confirmed.length > 0
        && bridge.device_confirmed.indexOf(deviceId) >= 0

    readonly property color accentColor: {
        if (!utilisationAvailable)
            return Theme.COLOR_NEUTRAL
        if (deviceId === "CPU")
            return Theme.COLOR_GOOD
        return Theme.ACCENT
    }

    color: Theme.BG_PANEL
    radius: Theme.RADIUS
    border.color: Theme.BORDER
    border.width: 1
    // A disabled card is visibly disabled: dimmed, but not invisible. "You cannot read
    // this" and "this is not here" are different statements.
    opacity: utilisationAvailable ? 1.0 : 0.78

    signal explainRequested(string key)

    // -- left column: the identity and the figure -------------------------------------
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
            id: deviceRow
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
                text: card.deviceId
                color: Theme.TEXT_ON_PANEL
                font.family: Theme.FONT_FAMILY
                font.pixelSize: Theme.TYPE_LABEL
                font.bold: true
            }

            // The word, not the colour. "Utilization unavailable" beside the value,
            // always, on every card whose telemetry is unavailable.
            Text {
                anchors.verticalCenter: parent.verticalCenter
                width: Math.max(0, card.width - identityColumn.width - Theme.SPACING.md
                                        - Theme.SPACING.sm * 2 - Theme.TYPE_LABEL * 2)
                visible: !card.utilisationAvailable
                text: bridge.utilisation_unavailable_text
                color: Theme.COLOR_NEUTRAL
                font.family: Theme.FONT_FAMILY
                font.pixelSize: Theme.TYPE_DETAIL
                elide: Text.ElideRight
            }
        }

        // The value, or the word that says there is none.
        Text {
            width: parent.width
            text: card.metricName.length > 0
                ? bridge.display_text(card.metricName)
                : bridge.unavailable_text
            color: card.utilisationAvailable ? Theme.TEXT_ON_PANEL : Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_NUMERAL
            font.bold: true
        }

        Text {
            width: parent.width
            visible: card.metricName.length > 0
                     && card.windowNote !== bridge.unavailable_text
            text: "measured " + card.windowNote
            color: Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            elide: Text.ElideRight
        }
    }

    // -- right column: what is real on this device, and where it ran -------------------
    Column {
        id: rightColumn
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.margins: Theme.SPACING.md
        width: Math.max(Theme.TYPE_NUMERAL_SMALL * 4, parent.width * 0.42)
        spacing: Theme.SPACING.xs

        Text {
            width: parent.width
            visible: card.secondaryMetricName.length > 0
            text: bridge.display_text(card.secondaryMetricName)
            color: Theme.TEXT_ON_PANEL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_NUMERAL_SMALL
            elide: Text.ElideRight
        }

        Text {
            width: parent.width
            visible: card.activityText.length > 0
            text: card.activityText
            color: Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            wrapMode: Text.WordWrap
            maximumLineCount: 3
            elide: Text.ElideRight
        }

        // The placement badge.
        Rectangle {
            width: parent.width
            height: Theme.TYPE_DETAIL + Theme.SPACING.sm * 2
            radius: Theme.RADIUS
            color: Theme.BG_PANEL_DEEP

            Text {
                anchors.fill: parent
                anchors.leftMargin: Theme.SPACING.sm
                anchors.rightMargin: Theme.SPACING.sm
                verticalAlignment: Text.AlignVCenter
                text: card.confirmedHere
                    ? "confirmed: running here"
                    : (bridge.device_confirmed.length > 0
                        ? "confirmed elsewhere: " + bridge.device_confirmed
                        : "no placement confirmed")
                color: card.confirmedHere
                    ? Theme.COLOR_GOOD : Theme.COLOR_NEUTRAL
                font.family: Theme.FONT_FAMILY
                font.pixelSize: Theme.TYPE_DETAIL
                elide: Text.ElideRight
            }
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
            text: card.deviceId + " - " + card.explanationText
            color: Theme.TEXT_ON_PANEL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            wrapMode: Text.WordWrap
        }
    }
}