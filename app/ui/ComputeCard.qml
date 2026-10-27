// ComputeCard.qml - one **camera's** verdict block, sized to be read from three metres.
//
// ## What this card is, after M4 Task 1
//
// M3's card carried eleven rows: name and state, the frame age, the staleness
// sentence, the frame-age window, the activity line, the confirmed device, the backlog, the
// analysis rate, the utilisation figure, its averaging window and the stated reason - ten of
// them at 18 px. M3's
// readability review then measured it at 3 m: **the age was 8.5 arcmin and every 18 px line
// was below the acuity floor.** At the 44 px the arithmetic requires, those eleven rows are
// **764 px per camera** (recomputed row by row at the ladder in section 4 of the
// legibility review), so three cameras are **2 292 px in the 660 px band they share** - 3.5x over.
//
// So the card keeps what a passer-by reads from the aisle - **which camera, whether it is
// running, how old its picture is, and where it is running** - and the other six numbers
// moved into `PresenterDrawer.qml`, where a presenter looks closely:
//
//   backlog, analysis rate, frame-age window, activity line, the device's utilisation
//   figure and the telemetry averaging window. The staleness sentence moved to
//   `Main.qml`'s footer, where it is one line at full width instead of two at 567 px.
//
// The claim the card is *about* is unchanged, and it is the reason the age is the largest
// figure on the screen: measured, unsaturated, three streams on the CPU - spec section 5
// row 4, a state the booth is in on a quiet day - gives per-stream frame age p50 10767 /
// 11085 / 11319 ms, p95 11626 / 11905 / 12286 ms, a backlog of 170-187 frames per stream,
// about 13.5 of the 15 fps analysed, zero drops, and all three streams reporting `active`.
// The picture on screen is **eleven seconds old** and every other counter says the system is
// healthy.
//
// ## Frame age is not inference cost
//
// On the NPU the p95 frame age is seconds and the inference itself is about 5.9 ms; the
// difference is queue wait. Two claims, two words. Nothing here adds them, and no single
// figure on this card is a blend of the two.
//
// ## Per device, not per screen
//
// The card shows the **confirmed** device class for its own camera, or `no device is
// running`. A failed route claims no device and never borrows the one it asked for. The
// utilisation wording ("no CPU sample yet" / "Utilization unavailable") is in the drawer,
// on the line named after the device's own metric - it is a claim about a counter, not
// about the picture, so it is a presenter's reading rather than an aisle reading.
//
// ## The card is never dimmed
//
// M1 dimmed a card whose utilisation was unmeasurable, because the card had nothing else to
// show. This card has the frame age, the placement and the state, all of which are real, so
// dimming the card that tells a presenter the NPU stream is eleven seconds stale - because
// NPU utilisation cannot be measured - would dim the one warning that matters.

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
            "stale": false, "staleness": "", "reason": "", "reason_display": "",
            "summary_text": bridge.unavailable_text
        })

    /// This camera's **confirmed** device class, or `""` when nothing has confirmed.
    ///
    /// `""` is a real state and is rendered as `no device is running`: a failed route
    /// claims no device, and a card that borrowed the requested one would be claiming a
    /// placement that was never proved. The device's own *utilisation* wording lives in
    /// `PresenterDrawer.qml`, which is the surface that renders a utilisation figure.
    readonly property string deviceId: String(record.device)

    /// The state word, from **this stream's own** state and never the aggregate.
    readonly property string stateText: String(record.state)

    readonly property bool isFailed: stateText === "failed"

    /// The headline: this camera's measured p50 frame age, already formatted.
    readonly property string ageText: String(record.age_p50_text)

    /// Which explanation the presenter's drawer opens with. See `explanationKeys` in
    /// `Bridge.qml`.
    property string explainKey: "cpu"

    /// The camera's colour: good while it is running, bad once it has failed, neutral
    /// while it is still proving where it runs.
    ///
    /// **Colour is never the only signal**: the state is also a word on the card, the
    /// device badge is a word, and `STALE` is a word in the footer.
    readonly property color accentColor: isFailed ? Theme.COLOR_BAD
        : (stateText === "active" ? Theme.COLOR_GOOD : Theme.COLOR_NEUTRAL)

    color: Theme.BG_PANEL
    radius: Theme.RADIUS
    // **The card clips. M3 Task 2's `clip: true` decision, kept.**
    //
    // A `Column` does not clip, so before this a text longer than the card was **drawn
    // anyway, over the card below it** - which is the defect this milestone has already
    // recorded twice. With this line the card is the nearest clipping ancestor of
    // everything in it, so `capture_ui`'s `overflow` becomes a **per-card** measure: a card
    // whose content exceeds its own box is named rather than painted over a neighbour.
    clip: true
    border.color: isFailed ? Theme.COLOR_BAD
        : (record.stale ? Theme.COLOR_WARN : Theme.BORDER)
    // **A stale camera's card is outlined in the warning colour**, so the thing that
    // matters is visible across the room, not only to a reader of small text.
    border.width: (isFailed || record.stale) ? 2 : 1

    signal explainRequested(string key)

    Column {
        id: cardColumn
        // **Named, so an `overflow` report says *which* box.** `clip: true` makes this
        // Column the nearest clipping ancestor of everything in it, and `capture_ui`
        // records the ancestor's label - so without a name the report says `ancestor:` and
        // a reader has to guess between the column and the card.
        objectName: "cardColumn_" + card.streamId
        anchors.fill: parent
        anchors.margins: Theme.SPACING.sm
        spacing: Theme.SPACING.xs
        clip: true

        // Which camera, and whether it is running. **A row, not two rows**: at 48 px the
        // name and the state word together are 337 px of a 567 px inner width, and the
        // vertical budget is what decides this layout.
        Row {
            width: parent.width
            spacing: Theme.SPACING.sm

            Text {
                objectName: "cameraName_" + card.streamId
                // **Named in M4.** It had an `id` and no `objectName`, so the harness could
                // not see the word that says which camera the number below it belongs to.
                text: "Camera " + card.streamId
                color: Theme.TEXT_ON_PANEL
                font.family: Theme.FONT_FAMILY
                font.pixelSize: Theme.TYPE_BOOTH_LABEL
                font.bold: true
            }

            // The state as a word, per this camera only.
            Text {
                objectName: "cameraState_" + card.streamId
                text: card.stateText
                color: card.accentColor
                font.family: Theme.FONT_FAMILY
                font.pixelSize: Theme.TYPE_BOOTH_DETAIL
                font.bold: true
                anchors.verticalCenter: parent.verticalCenter
            }
        }

        // The measured age. The biggest thing on the screen, because in the condition this
        // card exists for it is the only number that tells the presenter anything, and
        // spec section 7 requires it to be legible at a glance.
        Text {
            objectName: "streamAgeText_" + card.streamId
            width: parent.width
            text: card.ageText
            color: record.stale ? Theme.COLOR_WARN : Theme.TEXT_ON_PANEL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_BOOTH_NUMERAL
            font.bold: true
            elide: Text.ElideRight
        }

        // The placement, from the **confirmed** reading. The requested device rides along
        // inside `badge` only when it differs, so a fallback is visible as a fallback.
        //
        // **A word, not only a colour**, and the one string the specification names as the
        // placement proof: `device: CPU` in the header is the routed camera's, and this is
        // each camera's own.
        Rectangle {
            objectName: "streamBadgeBox_" + card.streamId
            width: parent.width
            height: badgeText.implicitHeight + Theme.CARD_BADGE_INSET * 2
            radius: Theme.RADIUS
            color: Theme.BG_PANEL_DEEP

            Text {
                id: badgeText
                objectName: "streamBadgeText_" + card.streamId
                anchors.fill: parent
                anchors.leftMargin: Theme.SPACING.sm
                anchors.rightMargin: Theme.SPACING.sm
                verticalAlignment: Text.AlignVCenter
                text: card.record.badge
                color: card.record.confirmed.length > 0
                    ? Theme.COLOR_GOOD
                    : (card.isFailed ? Theme.COLOR_BAD : Theme.COLOR_NEUTRAL)
                font.family: Theme.FONT_FAMILY
                font.pixelSize: Theme.TYPE_BOOTH_DETAIL
                elide: Text.ElideRight
            }
        }
    }

    // Hover opens the explanation; a click opens it too. The design wants both, because a
    // booth machine is often touch-only and hover alone is then unreachable.
    //
    // **Named, M3 Task 5.** A card's click is one of the explanation affordances the M1
    // `?`-panel defect showed can be silently dead; a test reaches this `MouseArea` by name
    // and clicks it.
    //
    // **M4 removed the hover *hint* that used to sit above the card.** It was anchored to
    // `parent.top` - outside the card - and M3's `clip: true` on the card clipped it away,
    // so it could never be seen; naming it made `capture_ui` report it as overflow, which is
    // how the dead affordance surfaced. It is removed rather than repaired: the drawer is
    // the explanation surface now, and a hover-only hint is unreachable on a touch booth.
    MouseArea {
        objectName: "cardExplainMouse_" + card.streamId
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.LeftButton
        onClicked: card.explainRequested(card.explainKey)
    }
}
