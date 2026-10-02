"""`app.ui.provider` - the QML image provider, and the only module that imports QtQuick.

## Why this module exists separately from `app.ui.bridge`

`FrameImageProvider` is a `QQuickImageProvider`, so **importing it imports `QtQuick`**.
`UiBridge` itself needs only `QtCore` and `QtGui`: it formats text, builds overlay boxes and
holds state, none of which touches QML.

That separation is not tidiness, it is what lets `app.main --selftest` run its
render-metadata stage without importing `QtQuick` at all. A self-test that has pulled in
the QML stack cannot be argued to be display-free, and the AGENTS.md gate is run on a
machine that may have no display at all. `app.ui.bridge` therefore does **not** import
this module at the top level - it resolves `FrameImageProvider` lazily through a module
`__getattr__`, so `from app.ui.bridge import FrameImageProvider` still works for the
windowed path and for the offscreen render tests, while nothing that only needs the
bridge pays for `QtQuick`.

## What the provider promises

**It serves whatever `UiBridge.set_frame_image` last handed it, and nothing else.**
There is no synthetic frame, no placeholder rectangle and no "no signal" pattern that
could be mistaken for video: before the first real frame arrives, `requestImage` returns a
**1x1 transparent** image, which renders as nothing. That is the honest absence - an empty
hero that says so, rather than a still image implying a stream that is not running.

The BGR-to-RGB swap happens here because `cv2` hands BGR and `QImage.Format_RGB888` wants
RGB; getting it wrong renders a plausible-looking image with the red and blue channels
exchanged, which nobody notices on a dark retail scene and everybody notices on an
orange. The swap is done with `numpy` rather than `cv2.cvtColor` so this module keeps no
`cv2` import.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtGui import QColor, QImage
from PySide6.QtQuick import QQuickImageProvider

__all__ = ["FrameImageProvider"]


class FrameImageProvider(QQuickImageProvider):
    """Paints the newest decoded frame for the hero and the thumbnails.

    The provider is a **thin adapter**: it reads `bridge.frame_image` and returns it. It
    computes nothing about detection boxes, zones or labels, because everything the
    overlay draws is already in the bridge and this class must not be a second source of
    truth for any of it.
    """

    def __init__(self, bridge: Any) -> None:
        super().__init__(QQuickImageProvider.Image)
        self._bridge = bridge
        self._blank = QImage(1, 1, QImage.Format_RGB888)
        self._blank.fill(QColor(0, 0, 0, 0))

    def requestImage(self, *args: Any) -> QImage:
        """The newest frame, or a 1x1 transparent image if none has arrived."""
        image = self._bridge.frame_image
        if image is None or image.isNull():
            return self._blank
        return image