// Bridge.qml - layout-derived values for QML, read once from `app/theme.py`.
//
// Two context properties arrive from Python: `Theme` (every token in app/theme.py, one
// constant property each) and `bridge` (the `UiBridge`). This component turns the theme
// fractions into pixel counts once, so no QML file has to multiply `CANVAS_HEIGHT` by
// `GRID.header` itself and get the arithmetic subtly different from its neighbour.
//
// **This file holds constants and derived sizes. It holds no formatting functions**,
// because a function defined on a QML `QtObject` is a trap here: `Main.qml` instantiates
// this type, and inside its own scope the name `Bridge` then resolves to *that instance*
// rather than to the type, so `Bridge.metric(...)` in a child file raised
// `TypeError: Property 'metric' of object Bridge is not a function` (measured, with the
// window still loading and every metric silently blank). Metric formatting goes through
// `bridge.display_text(name)` and `bridge.window_text(name)` directly - which is where
// the obligation to route every optional number through one formatter belongs anyway.
//
// What is here, and why:
//
//   - `recordedFootageLabel` - the mandatory label, bound to the Python constant and
//     falling back to the same literal in source so the copy is pinned by a text search as
//     well as by the bridge's own constant. `tests/test_ui_contract.py` asserts the two
//     agree, so they cannot drift apart without a failure.
//   - `deviceBadge` - the confirmed device, or "pending", or "failed". Never the requested
//     device standing in for a confirmed one.
//   - `utilisationUnavailableText` - what the GPU and NPU cards say instead of a number.
//
// No QML file in this tree contains a pixel count, a fraction, a colour literal or a font
// size: `tests/test_ui_contract.py` reads every `.qml` file and fails on any of them.

import QtQuick

QtObject {
    id: tokens

    // ---- the mandatory label -------------------------------------------------------

    readonly property string recordedFootageLabel:
        bridge.recorded_footage_label.length > 0
            ? bridge.recorded_footage_label
            : "Recorded footage \u2022 live inference"

    // ---- device honesty ------------------------------------------------------------

    readonly property string deviceBadge: bridge.device_display
    readonly property bool deviceConfirmed: bridge.device_confirmed.length > 0
    readonly property bool fellBack: bridge.device_fell_back

    // The confirmed device contains a named device, compared as a substring of the
    // *confirmed* reading only. `'GPU' in ('GPU.0',)` is False in Python but
    // `indexOf("GPU") >= 0` in QML is True, which is the right way round for a label:
    // this asks which device is shown, never which was asked for.
    // The comparison each card makes about itself, stated once here so three cards do not
    // each invent it. It asks about the **confirmed** reading only - a card that asked
    // "did we get what we asked for" would report "running here" on a route that fell
    // back, which is review focus #5. `indexOf` rather than equality because the GPU
    // confirms as `'GPU.0'`.
    function confirms(deviceId) {
        return deviceConfirmed && bridge.device_confirmed.indexOf(deviceId) >= 0
    }

    // ---- utilisation ---------------------------------------------------------------

    readonly property bool gpuUtilisationAvailable: bridge.gpu_available
    readonly property bool npuUtilisationAvailable: bridge.npu_available
    // The CPU card is not special. Main.qml used to hardcode its availability to a literal
    // true, which would have rendered an unavailable psutil reading coloured and
    // undimmed as though it were live. All three cards now answer the same question of
    // the bridge. (The literal is spelled without its value here on purpose: a source
    // check forbids a card ever hardcoding its own availability, and a comment that
    // quoted the old line verbatim would trip it.)
    readonly property bool cpuUtilisationAvailable: bridge.cpu_available
    readonly property string utilisationUnavailableText: bridge.utilisation_unavailable_text

    // The measured reason, full text. Reachable from the pinned panel; the card-sized
    // short reason is `bridge.gpu_reason` / `bridge.npu_reason`.
    readonly property string utilisationMeasuredReason: bridge.utilisation_measured_reason

    // ---- layout --------------------------------------------------------------------

    readonly property int canvasWidth: Theme.CANVAS_WIDTH
    readonly property int canvasHeight: Theme.CANVAS_HEIGHT

    readonly property int videoWidth: Math.round(canvasWidth * Theme.VIDEO_WIDTH_FRACTION)
    readonly property int metricsWidth: canvasWidth - videoWidth

    readonly property int headerHeight: Math.round(canvasHeight * Theme.GRID.header)
    readonly property int mainHeight: Math.round(canvasHeight * Theme.GRID.main)
    readonly property int footerHeight: canvasHeight - headerHeight - mainHeight

    // ---- explanation content ------------------------------------------------------

    readonly property var explanationKeys:
        ["yolo", "tracking", "zone", "cpu", "gpu", "utilisation", "metrics"]

    // The pinned panel's text. `Main.qml` called `bridgeUi.explain(...)` and **this
    // function did not exist** - it had been dropped from this file in an earlier edit,
    // and because the call sits behind `pinnedExplanationKey.length > 0` in a ternary,
    // the short-circuit meant QML never evaluated it. The window loaded, every card drew,
    // and the "?" buttons were dead at the booth.
    //
    // It is back, and `tests/test_ui_contract.py::SlotBindingAudit` now fails on any
    // `bridgeUi.*` call that names something this object does not actually have - the
    // check that would have caught it, since a dangling call is invisible until someone
    // clicks.
    //
    // Constant-safe to call from a binding: it reads `EXPLANATIONS` through the bridge, a
    // module constant, and its argument is a clicked key, so a change to that key
    // re-evaluates the binding anyway.
    function explain(key) {
        return bridge.explanation(key)
    }
}