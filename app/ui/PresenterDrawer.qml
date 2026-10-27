// PresenterDrawer.qml - the presenter's explanation surface. **M3 Task 5.**
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
// ## It moves nothing
//
// It is loaded as an **overlay** by `Main.qml`'s `Loader`, anchored to the window's right
// edge and spanning its full height. It is not part of the layout `Column`, so opening it
// cannot reflow the hero, the tiles, the badges or the staleness line - and the test
// asserts every one of those positions is identical before and after, item by item,
// because "nothing threw" is not the same claim.
//
// ## The two honesty statements, and why they are on this surface
//
// - **The retail scope note, verbatim.** `bridge.scope_note` is
//   `app/engine/retail_config.py`'s `scope_note`, read from the shipped pack. The screen
//   must not paraphrase it: the delivered event is **people-in-zone presence, 2 events per
//   clip - not a people count, not a dwell measurement.**
// - **GPU/NPU utilisation is unavailable, and why.** `bridge.gpu_reason` is the measured
//   reason: the platform's own counters fault this process when read, and a machine-wide
//   peak is **not attributable to this process**, so no utilisation number is shown.
//
// ## No layout token is in this file
//
// Every size, spacing and colour comes from `Theme`, with two deliberate exceptions stated
// rather than hidden: `border.width: 2`, which `Main.qml` and `ComputeCard.qml` also use,
// and the fact that there is **no automated colour-literal sweep** over `app/ui` - the
// guard that reads every `.qml` file is `SlotBindingAudit`, and it checks call targets, not
// colours. The drawer's width is `bridgeUi.drawerWidth` (from `app/theme.py`, 0.32 of the
// canvas = 614 px measured) and is applied by the `Loader` in `Main.qml`; the component
// fills the box it is given with `anchors.fill: parent`.

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
        spacing: Theme.GAP

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
            text: "Delivered event: people-in-zone presence, 2 events per clip. "
                  + "NOT a people count. NOT a dwell measurement."
            color: Theme.TEXT_ON_PANEL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            font.bold: true
            wrapMode: Text.WordWrap
        }

        // -- GPU/NPU utilisation: unavailable, with the measured reason --------------
        Text {
            objectName: "drawerUtilisationNote"
            width: parent.width
            text: "GPU and NPU utilisation is unavailable. " + bridge.gpu_reason
            color: Theme.COLOR_NEUTRAL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            wrapMode: Text.WordWrap
        }

        // -- every explanation, reachable and named ---------------------------------
        //
        // **`bridge.explanation_keys`, not `bridge.explanationKeys`.** The first version
        // asked the Python bridge for `explanationKeys`, which it does not have; the model
        // was `undefined` and this `Repeater` produced **zero delegates** - the drawer
        // opened with no explanation buttons, which is the M1 dead-affordance shape on the
        // one surface whose job is explaining the boxes. The keys come from
        // `app.ui.bridge`'s `EXPLANATIONS`, which is also what the footer's `?` row offers.
        //
        // `Flow`, not `Row`. **Measured:** the seven buttons are 49 + 79 + 55 + 45 + 46 +
        // 91 + 74 = 439 px plus six 8 px gaps = 487 px, inside the drawer's 566 px inner
        // width, so today they fit on one row and the two layouts render identically. `Flow`
        // is the shape that stays honest when the key set grows: the list is data, a `Row`
        // would draw an eighth key past the panel's clipped edge, and a button half on screen
        // is a dead affordance with extra steps. `Flow` wraps and shrinks no font.
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
        // In the drawer rather than in `Main.qml`'s pinned panel, because the pinned panel
        // sits under this overlay - a button that sets text a presenter cannot see is the
        // same dead affordance the M1 `?` panel was.
        Text {
            objectName: "drawerExplanationText"
            width: parent.width
            text: drawer.selectedText
            color: Theme.TEXT_ON_PANEL
            font.family: Theme.FONT_FAMILY
            font.pixelSize: Theme.TYPE_DETAIL
            wrapMode: Text.WordWrap
        }
    }
}
