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

## What the provider promises, per camera

**It serves `UiBridge.frame_for(stream)` and nothing else.** The image id is
`image://frames/<stream_id>`, so a tile asks for **its own** stream by name.

Two rules, both load-bearing, and both about not lying:

- **No cross-stream fallback.** An id naming a stream that has never delivered a frame
  returns a **1x1 transparent** image, *not* the hero's frame and not the most recent frame
  from any camera. A tile captioned "Camera B" showing Camera A's shopper is the worst
  mislabelling this project can produce, and it is exactly what a fallback here would
  produce.
- **The absence is visible.** 1x1 transparent renders as nothing, so the tile shows its own
  background, and `VideoTile.qml`'s `visible:` guard puts a sentence on it. Before the
  per-stream split every tile asked for `"hero"` and so all three surfaces showed whichever
  camera arrived last - which is why the M2 report had to say the per-camera surfaces were
  not built.

The BGR-to-RGB swap happens in `UiBridge._store_image`, on the UI thread, because a
`QImage` built here would be built on whatever thread the scene graph asked from. Getting
the swap wrong renders a plausible-looking image with the red and blue channels exchanged,
which nobody notices on a dark retail scene and everybody notices on an orange.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtGui import QColor, QImage
from PySide6.QtQuick import QQuickImageProvider

__all__ = ["FrameImageProvider"]


class FrameImageProvider(QQuickImageProvider):
    """Paints one camera's newest decoded frame, addressed by stream id.

    The provider is a **thin adapter**: it reads `bridge.frame_for(stream_id)` and returns
    it. It computes nothing about detection boxes, zones or labels, because everything the
    overlay draws is already in the bridge and this class must not be a second source of
    truth for any of it.
    """

    def __init__(self, bridge: Any) -> None:
        super().__init__(QQuickImageProvider.Image)
        self._bridge = bridge
        self._blank = QImage(1, 1, QImage.Format_RGB888)
        self._blank.fill(QColor(0, 0, 0, 0))
        #: Every id this provider has been asked for, and what it answered. **A test seam
        #: that is also the audit trail**: a tile asking for `"hero"` or for another
        #: camera's id is visible here rather than only in the pixels.
        self.requests: list[tuple[str, bool]] = []

    def requestImage(self, *args: Any) -> QImage:
        """One camera's newest frame, or a 1x1 transparent image if it has none.

        ## Reading the stream id out of what Qt actually hands over

        Two things were measured here, and both produced a **silently blank hero** with the
        metrics column drawn and no warning anywhere:

        1. **The id is the FIRST argument, and Qt also passes a `QSize` as the last one.**
           `requestImage` is reached as `requestImage(id, requestedSize)`, so `args[-1]` is
           a `QSize(-1, -1)` and not the id at all. Measured: `('QSize(-1, -1)', False)` in
           the audit trail.
        2. **The id is the WHOLE URL, not the trailing segment.** For a `QString` `source`
           of `"image://frames/A?v=1"`, the argument is a **`QUrl`**, whose `str()` is that
           whole URL. Stripping the query and the slashes leaves `"image:/frames/A"`, which
           names no stream, so `frame_for` returns `None` and every tile got the 1x1 blank -
           with `source` reading correctly and `paintedWidth` reporting the scaled 1x1.

        So the id is taken from whichever argument has a `path()` (a `QUrl`) or is a
        non-empty string, and the stream is that path's **last** segment. The `?v=` query is
        dropped by the same split, and a plain `"A?v=2"` string is still accepted because
        that is what the Python-side tests pass.
        """
        raw_arg = ""
        path = None
        for candidate in args:
            to_path = getattr(candidate, "path", None)
            if callable(to_path):
                try:
                    path = to_path()
                except Exception:  # noqa: BLE001 - fall through to the string form
                    path = None
                raw_arg = str(candidate)
                break
            text = str(candidate).strip()
            if text and not text.startswith("PySide6"):
                raw_arg = text
                break
        stream = (path if path is not None else raw_arg).split("?", 1)[0]
        stream = stream.strip().lstrip("/").split("/")[-1].strip()
        image = self._bridge.frame_for(stream) if stream else None
        served = image is not None and not image.isNull()
        self.requests.append((raw_arg, served))
        return image if served else self._blank
