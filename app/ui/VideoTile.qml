// VideoTile.qml - one video surface: hero or thumbnail.
//
// The hero draws the newest decoded frame plus the overlay that belongs to *that* frame.
// The boxes come from `bridge.overlay_boxes`, which is built by `UiBridge.set_overlay`,
// which filters on `Track.is_current`. There is no QML-side truth table about whether a
// track is coasting, because that is the one thing QML cannot be trusted to hold
// correctly: a coasted track's box is the last one *seen*, so drawing it puts a
// plausible, correctly-placed rectangle over an object that has since left, and nothing
// about that rectangle looks wrong.
//
// The image `source` carries `bridge.frame_revision` as a query string. Without it the
// `Image` is fetched once and the hero shows frame 0 for the whole session - measured
// behaviour of a constant `image://` URL, and the reason this binding is not a constant.

import QtQuick

Item {
    id: tile

    /// True for the hero. Thumbnails get the border and the caption only.
    property bool hero: false
    /// Caption shown under a thumbnail.
    property string caption: ""
    /// Which image-provider path this tile asks for.
    property string sourceName: hero ? "hero" : "thumb"
    /// Whether overlay boxes and zone rectangles are drawn. Hero only.
    property bool drawOverlay: hero
    /// Explainer key for the hover panel.
    property string explainKey: "yolo"

    signal tileActivated()

    Image {
        id: video
        anchors.fill: parent
        fillMode: Image.PreserveAspectFit
        asynchronous: false
        cache: false
        // Anti-aliasing off: on a downscaled 1920x1080 frame it costs real time per
        // frame and the booth reads motion, not edges.
        smooth: false
        source: "image://frames/" + tile.sourceName + "?v=" + bridge.frame_revision
    }

    // Nothing is drawn before the first real frame arrives: the provider serves a 1x1
    // transparent image, so the tile shows its own background rather than a placeholder
    // that could be mistaken for a frozen stream.
    Rectangle {
        anchors.fill: parent
        visible: !bridge.frame_ready
        color: Theme.BG_VIDEO
        border.color: Theme.BORDER
        border.width: 1
        Text {
            anchors.centerIn: parent
            text: "no frame decoded yet"
            color: Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
        }
    }

    // -- zone rectangles -------------------------------------------------------------
    // Normalised coordinates straight from the shipped configuration, so what is drawn is
    // the ROI the event machine is testing against rather than a drawing of it.
    Repeater {
        model: tile.drawOverlay ? bridge.zones : []
        delegate: Rectangle {
            required property var modelData
            x: modelData.x * tile.width
            y: modelData.y * tile.height
            width: modelData.w * tile.width
            height: modelData.h * tile.height
            color: "transparent"
            border.color: Theme.ACCENT
            border.width: Theme.OVERLAY_ZONE
            opacity: 0.7
        }
    }

    // -- detection / track boxes ----------------------------------------------------
    Repeater {
        model: tile.drawOverlay ? bridge.overlay_boxes : []
        delegate: Item {
            required property var modelData
            x: modelData.x * tile.width
            y: modelData.y * tile.height
            width: Math.max(1, modelData.w * tile.width)
            height: Math.max(1, modelData.h * tile.height)

            Rectangle {
                anchors.fill: parent
                color: "transparent"
                // A confirmed track gets the accent; an unconfirmed one the neutral
                // colour. The event machine counts only confirmed tracks, so the colour
                // difference is the same distinction the counting makes.
                border.color: modelData.confirmed ? Theme.ACCENT : Theme.COLOR_NEUTRAL
                border.width: Theme.OVERLAY_BOX
            }

            // The track id, drawn on the box. Present because a track id is the identity
            // half of every event key on screen; a box nobody can name cannot be matched
            // to an event by a visitor asking "what is that?".
            Text {
                x: 2
                y: -Theme.OVERLAY_TRACK - 2
                text: "#" + modelData.track_id
                color: modelData.confirmed ? Theme.ACCENT : Theme.COLOR_NEUTRAL
                font.family: Theme.FONT_FAMILY
                font.pixelSize: Theme.OVERLAY_TRACK
            }
        }
    }

    Rectangle {
        anchors.fill: parent
        color: "transparent"
        border.color: tile.hero ? Theme.ACCENT : Theme.BORDER
        border.width: tile.hero ? 2 : 1
        radius: Theme.RADIUS
    }

    Rectangle {
        visible: tile.caption.length > 0
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: Theme.TYPE_DETAIL + Theme.SPACING.sm * 2
        color: Theme.BG_PANEL_DEEP
        radius: Theme.RADIUS
        Text {
            anchors.left: parent.left
            anchors.leftMargin: Theme.SPACING.sm
            anchors.verticalCenter: parent.verticalCenter
            text: tile.caption
            color: Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
        }
    }

    MouseArea {
        anchors.fill: parent
        onClicked: tile.tileActivated()
        // Hover opens the explanation; a click pins it. The pinned panel is what makes
        // the copy reachable on a touch-only booth machine.
        hoverEnabled: true
        acceptedButtons: Qt.LeftButton
    }
}