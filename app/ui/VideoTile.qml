// VideoTile.qml - one camera's video surface: hero or thumbnail.
//
// ## One tile, one camera
//
// `streamId` names **this tile's** camera, and the `Image` asks the provider for exactly
// that stream: `"image://frames/" + streamId + "?v=" + <that stream's revision>`. The
// provider serves `UiBridge.frame_for(streamId)` and has nothing else to serve - there is
// no cross-stream fallback, so a tile **cannot** show another camera's picture even by
// accident. Before this, every tile asked for `"hero"` and all three surfaces showed
// whichever camera delivered last.
//
// The revision comes from the bridge's **notified** `stream_frame_revisions` map, never
// from a method call: a QML binding that calls a method declares no property dependency
// and is evaluated once, which is the defect this repository has now found four times.
//
// ## A camera with no frame says so
//
// `frames_ready[streamId] !== true` is the guard, and a stream that has never delivered a
// frame is **absent from the map** rather than present with a 0 - `0` would be
// indistinguishable from "frame 0 just arrived". The tile then shows its own background
// and this sentence, in words, naming the camera. It does **not** hold a previous frame and
// it does not borrow the hero's.
//
// ## The hero draws the overlay
//
// The hero's boxes come from `bridge.overlay_boxes`, which is the **bound stream's** boxes
// (`UiBridge.set_route_stream` re-derives them when the hero moves, so pointing the screen
// at Camera C does not leave Camera A's rectangles drawn over it). There is no QML-side
// truth table about whether a track is coasting, because that is the one thing QML cannot
// be trusted to hold correctly: a coasted track's box is the last one *seen*, so drawing it
// puts a plausible, correctly-placed rectangle over an object that has since left, and
// nothing about that rectangle looks wrong.

import QtQuick

Item {
    id: tile

    // **The tile clips. M3 Task 2**, for the same reason the card does: two tiles share a row
    // and an `Item` does not clip, so a tile whose contents outgrew it would paint over its
    // neighbour. `capture_ui`'s `overflow` measure walks to the nearest clipping ancestor, so
    // this line also gives that measure a per-tile bound instead of the window's.
    clip: true

    /// Which camera this tile shows. **Required**, and it is what the image is addressed by.
    property string streamId: ""
    /// True for the hero. Thumbnails get the border and the caption only.
    property bool hero: false
    /// Caption shown under a thumbnail.
    property string caption: ""
    /// Whether overlay boxes and zone rectangles are drawn. Hero only.
    property bool drawOverlay: hero
    /// Explainer key for the hover panel.
    property string explainKey: "yolo"

    /// Whether **this** camera has delivered a frame. Not `bridge.frame_ready`, which is the
    /// hero's: three cameras at 15 fps would otherwise put "no frame decoded yet" over two
    /// working thumbnails.
    readonly property bool frameReady: streamId.length > 0
        && bridge.stream_frames_ready[streamId] === true
    /// This camera's own revision, or 0 for a camera that has delivered nothing.
    readonly property int frameRevision:
        bridge.stream_frame_revisions[streamId] !== undefined
        ? bridge.stream_frame_revisions[streamId] : 0

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
        objectName: "tileImage_" + tile.streamId
        source: tile.streamId.length > 0
            ? "image://frames/" + tile.streamId + "?v=" + tile.frameRevision
            : ""
    }

    // Nothing is drawn before this camera's first real frame arrives: the provider serves
    // a 1x1 transparent image, so the tile shows its own background rather than a
    // placeholder that could be mistaken for a frozen stream - and rather than another
    // camera's picture.
    Rectangle {
        anchors.fill: parent
        objectName: "tileNoFrame_" + tile.streamId
        visible: !tile.frameReady
        color: Theme.BG_VIDEO
        border.color: Theme.BORDER
        border.width: 1
        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: parent.verticalCenter
            // **Above the caption bar, not behind it.** The caption is drawn after this
            // panel (it is a sibling later in the file), so a sentence centred in the whole
            // tile had its lower half painted over by the caption - measured: the ink band
            // was 29 px against a 31.5 px cap height, which is exactly the "is the text
            // really drawn" defect the M4 ink check exists to catch.
            anchors.verticalCenterOffset: tile.caption.length > 0
                ? -(Theme.TYPE_BOOTH_DETAIL + Theme.SPACING.sm * 2) / 2 : 0
            width: parent.width - Theme.SPACING.md
            objectName: "tileNoFrameText_" + tile.streamId
            text: tile.streamId.length > 0
                ? "no frame from Camera " + tile.streamId + " yet"
                : "no camera on this surface"
            color: Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            // **The legibility ladder, M4.** This sentence is how a camera that has
            // delivered nothing says so, and at 18 px it was below the acuity floor at
            // 3 m - the one thing a passer-by most needs to read on a dead surface.
            font.pixelSize: Theme.TYPE_BOOTH_DETAIL
            horizontalAlignment: Text.AlignHCenter
            wrapMode: Text.WordWrap
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
            objectName: "tileBox_" + tile.streamId + "_" + modelData.track_id
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
            //
            // **M4: named, and on the ladder.** It used to be 18 px with no `objectName`,
            // so it was text on the picture that the harness could not see and a
            // passer-by could not read. A track id a visitor is invited to ask about has
            // to be legible from where the visitor stands.
            Text {
                objectName: "tileTrackLabel_" + tile.streamId + "_" + modelData.track_id
                x: 2
                y: -height - Theme.SPACING.xs
                text: "#" + modelData.track_id
                color: modelData.confirmed ? Theme.ACCENT : Theme.COLOR_NEUTRAL
                font.family: Theme.FONT_FAMILY
                font.pixelSize: Theme.TYPE_BOOTH_DETAIL
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
        // **M4: the caption bar has to hold a `TYPE_BOOTH_DETAIL` line.** A caption is a
        // claim about the picture above it; at 18 px it was a claim nobody at 3 m could
        // read, on the surface whose job is to say which camera is on it.
        height: Theme.TYPE_BOOTH_DETAIL + Theme.SPACING.sm * 2
        color: Theme.BG_PANEL_DEEP
        radius: Theme.RADIUS
        Text {
            // **Named, and this is the third M2 defect.**
            //
            // M2 shipped `Camera B · -1` under a thumbnail: a `QVariantList` concatenated with
            // a string. The fix was in the bridge (`stream_confirmed_labels`). **But the caption
            // `Text` had no `objectName`,** so `capture_ui`'s walk never yielded it, it was
            // absent from `strings` in all four committed captures, and **the defect the harness
            // was built to catch was invisible to the harness** - a regression here would have
            // left all four committed JSONs byte-identical and the CLI exiting 0.
            //
            // Naming it makes it one of the strings the instrument reads, and
            // `BOOTH_EXPECTED`/`BOOTH_NONEMPTY` list it so a capture waits for it and a test
            // asserts its value per camera. **A label nobody can read is not a label.**
            objectName: "tileCaptionText_" + tile.streamId
            anchors.left: parent.left
            anchors.leftMargin: Theme.SPACING.sm
            anchors.verticalCenter: parent.verticalCenter
            text: tile.caption
            color: Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_BOOTH_DETAIL
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