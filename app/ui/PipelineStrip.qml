// PipelineStrip.qml - decode, preprocess, infer, decode-detections, track, event.
//
// One row of stages, each with a device badge. **The badge on the `infer` stage is the
// confirmed device, never the requested one**, and it reads "pending" until an inference
// has proved a placement. Every other stage runs on the CPU because that is where this
// build's shared decode and postprocessing path is - spec §7 requires the decode
// behaviour to stay constant across comparisons, so those badges are facts about the
// topology rather than routing decisions, and they say "CPU" unconditionally rather than
// tracking the selector.

import QtQuick

Item {
    id: strip

    /// Stage names, in order. Data, not markup, so the strip and a summary line agree.
    property var stages: ["decode", "preprocess", "infer", "decode det.", "track", "event"]

    /// Which stage actually runs the model.
    readonly property int inferStageIndex: 2

    // The confirmed device badge, read straight off the bridge. Both properties are
    // declared here rather than reached through `Bridge.qml` because a `QtObject`
    // component's *properties* are not reachable from a sibling file under the type name
    // `Bridge` - that name resolves to the type inside this file, and reading an instance
    // property off the type yields `undefined` (measured: `Unable to assign [undefined]
    // to QString` at this binding, with the badge simply absent from the screen).
    // Functions declared on `Bridge.qml` are reachable from `Main.qml`, where the
    // instance's id is in scope; properties are not, so the context property is used.
    readonly property string inferBadge: bridge.device_display
    readonly property bool inferBadgeConfirmed: bridge.device_confirmed.length > 0

    implicitHeight: Theme.TYPE_LABEL + Theme.SPACING.lg

    Row {
        anchors.fill: parent

        Repeater {
            model: strip.stages

            delegate: Item {
                id: stage
                required property int index
                required property string modelData

                width: (strip.width - Theme.SPACING.xs * (strip.stages.length - 1))
                       / strip.stages.length
                height: strip.height

                readonly property bool isInfer: index === strip.inferStageIndex

                Rectangle {
                    anchors.fill: parent
                    radius: Theme.RADIUS
                    color: Theme.BG_PANEL
                    border.color: stage.isInfer ? Theme.ACCENT : Theme.BORDER
                    border.width: stage.isInfer ? 2 : 1

                    Column {
                        anchors.centerIn: parent
                        spacing: Theme.SPACING.xs
                        width: parent.width - Theme.SPACING.sm

                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            width: parent.width
                            text: stage.modelData
                            color: Theme.COLOR_NEUTRAL
                            font.family: Theme.FONT_FAMILY
                            font.pixelSize: Theme.TYPE_DETAIL
                            horizontalAlignment: Text.AlignHCenter
                            elide: Text.ElideRight
                        }

                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            width: parent.width
                            // The device badge. On the infer stage this is the CONFIRMED
                            // device - "pending" until an inference has proved a
                            // placement - and everywhere else the CPU, which is where
                            // this build runs the decode and the postprocessing.
                            text: stage.isInfer ? strip.inferBadge : "CPU"
                            color: stage.isInfer
                                ? (strip.inferBadgeConfirmed ? Theme.COLOR_GOOD
                                                            : Theme.COLOR_NEUTRAL)
                                : Theme.COLOR_NEUTRAL
                            font.family: Theme.FONT_FAMILY
                            font.pixelSize: Theme.TYPE_DETAIL
                            font.bold: stage.isInfer
                            horizontalAlignment: Text.AlignHCenter
                            elide: Text.ElideRight
                        }
                    }
                }

                // The arrow between stages, drawn only where a following stage exists so
                // the last stage does not point at nothing.
                Text {
                    visible: stage.index < strip.stages.length - 1
                    anchors.left: parent.right
                    anchors.leftMargin: Theme.SPACING.xs / 2
                    anchors.verticalCenter: parent.verticalCenter
                    text: "\u2192"
                    color: Theme.COLOR_NEUTRAL
                    font.family: Theme.FONT_FAMILY
                    font.pixelSize: Theme.TYPE_DETAIL
                }
            }
        }
    }
}