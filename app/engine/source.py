"""`ClipSource` - one clip, decoded at the analysis rate, with loops it admits to.

Two failure modes this module exists to prevent, both invisible on a booth screen until
someone freezes the footage:

**1. A loop that pretends not to be a loop.** The demo clip is 113 s long, so the booth
loops it. Frame 37 of loop 0 and frame 37 of loop 1 are the same pixels with the same
`frame_id`, and an overlay that binds to `frame_id` alone paints loop 0's boxes onto
loop 1's picture the first time the clip rewinds - confidently, and for a whole loop.
So `restart()` **increments `source_loop_id`** and every `FramePacket` carries it, and
the identity tuple that downstream code compares is
`(session_id, stream_id, source_loop_id, frame_id)`. The source never rewinds on its
own: reaching the end returns `None` and raises `finished`, and rewinding is a decision
somebody has to make, because it invalidates every result in flight.

**2. An "analysis rate" that is really the source rate.** The model wants 640x640 blobs
at a steady cadence, and the demo budget does not want a 29.97 fps inference. The
analysis therefore runs at `target_fps` - **at or below the source rate** - and the
source skips the frames in between with `VideoCapture.grab()`, which does not decode
them. The pacing deadline is computed from `target_fps` alone and never from the
source's frame count, so a source that could not keep up would show up as a growing
`frame.age` rather than as a silently reduced rate.

**The rate is reported as the rate, not as the request.** The source frame grid is
discrete, so 15 fps out of 29.97002997 is quantised: `step = round(29.97002997 / 15) = 2`
source frames per analysis frame, and the achieved rate is `14.985`. `target_fps` is
what the caller asked for; `analysis_fps` is what the frame grid delivers, and it is the
second number that belongs in a report. Claiming 15.000 would be reporting the request.

**Measured facts are read, not hardcoded.** The frame count and the fps come from
`config/models.json` through `app.engine.retail_config.clip_measurements` - the same
verified record the footage guard uses, so there is one source of truth - and the
container is asked as well; both are exposed so a caller can see they agree
(`total_frames` against `container_total_frames`). A clip that is not a listed asset
falls back to the container's own `CAP_PROP_FPS`/`CAP_PROP_FRAME_COUNT`, and says so by
exposing `media_id is None`.

**`source_pts` is `frame_id / source_fps`,** derived rather than read from
`CAP_PROP_POS_MSEC`, because that property is measured to report `0.0` for the first
frame of this clip and only becomes non-zero from the second frame on. A presentation
timestamp that starts at 0 and then jumps is a worse record than one derived from the
frame index and the container's own rate, which is exact.

**The clock is a parameter.** `clock` and `sleeper` are injected, defaulting to
`time.perf_counter` and `time.sleep`, so the pacing *schedule* can be asserted exactly in
a test without the test waiting for it, and so the module reads no clock that a caller
cannot see.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import cv2

from app.contracts.frame import FramePacket
from app.engine.retail_config import clip_measurements
from app.inference.pinned import MODELS_JSON, REPO_ROOT

__all__ = ["ClipSource", "recorded_media_for"]

#: Frames per analysis frame is clamped to at least 1. A `target_fps` above the source's
#: rate cannot make the decoder produce frames that do not exist, so the honest answer is
#: "every source frame, once", not a step of zero and an infinite loop.
MIN_STEP: Final[int] = 1


def _repo_relative(path: Path) -> str:
    """`path` as a repo-relative, forward-slash string, or the absolute form."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def recorded_media_for(path: Path) -> dict[str, Any] | None:
    """The `config/models.json` `media` entry whose `path` is `path`, or `None`.

    `None` means "this file is not a listed asset", which is a legitimate state - a
    clip dropped into the tree for a one-off test is not in the verified config - and is
    not the same as "no such file". It is also not an error: `ClipSource` falls back to
    the container's own rate and exposes `media_id is None` so a caller can tell.
    """
    wanted = _repo_relative(Path(path))
    try:
        payload = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    except OSError:
        # A missing or unreadable config means "no record", not "zero". The container is
        # still a source of truth about the file's own rate.
        return None
    for entry in payload.get("media", []) or ():
        recorded = str(entry.get("path") or "").replace("\\", "/")
        if recorded and recorded == wanted:
            return dict(entry)
    return None


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str; got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(
            f"{name} must be a non-empty string; got {value!r}. An empty identifier "
            f"cannot be told apart from a missing one, and it is part of the frame "
            f"identity every downstream result is matched against."
        )
    return value


def _require_rate(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number of frames per second; got {value!r}")
    rate = float(value)
    if rate != rate or rate in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if rate <= 0.0:
        raise ValueError(
            f"{name} must be > 0; got {rate}. A zero or negative frame rate has no "
            f"frame grid to step over and would make every timing on the path "
            f"meaningless."
        )
    return rate


class ClipSource:
    """One clip, read at the analysis rate, looping only when asked.

    The read pattern is `read()` - which returns the frame at `position`, paces to
    `target_fps`, and then moves `position` on by `step` - so the common caller is a
    loop of `read()` and the caller that wants to *discard* a frame calls `advance()`
    instead, which moves the position without decoding and without waiting.
    """

    def __init__(
        self,
        path: Path,
        stream_id: str,
        session_id: str,
        scenario_id: str,
        target_fps: float = 15.0,
        source_fps: float | None = None,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._path = Path(path)
        self._stream_id = _require_text("stream_id", stream_id)
        self._session_id = _require_text("session_id", session_id)
        self._scenario_id = _require_text("scenario_id", scenario_id)
        self._target_fps = _require_rate("target_fps", target_fps)
        if source_fps is not None:
            self._explicit_source_fps = _require_rate("source_fps", source_fps)
        else:
            self._explicit_source_fps = None
        if clock is None:
            import time as _time
            clock = _time.perf_counter
        if sleeper is None:
            import time as _time
            sleeper = _time.sleep
        self._clock = clock
        self._sleeper = sleeper

        if not self._path.exists():
            raise FileNotFoundError(
                f"the clip {self._path} does not exist. A source that cannot open its "
                f"file would present an empty stream, and an empty stream is "
                f"indistinguishable from a booth where nothing is happening."
            )
        if not self._path.is_file():
            raise FileNotFoundError(
                f"{self._path} is not a file. A source needs one video to read."
            )
        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            raise OSError(
                f"cv2.VideoCapture could not open {self._path}. The file exists and is "
                f"not empty, so the failure is the container or the codec rather than "
                f"the path, and decoding has to be refused rather than skipped."
            )

        self._media = recorded_media_for(self._path)
        self._media_id = str(self._media["id"]) if self._media else None
        container_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        container_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        recorded_fps, recorded_frames = self._recorded_rates()
        self._source_fps = (
            self._explicit_source_fps
            if self._explicit_source_fps is not None
            else (recorded_fps if recorded_fps is not None else _require_rate(
                "the clip's CAP_PROP_FPS", container_fps))
        )
        self._container_source_fps = container_fps
        self._total_frames = recorded_frames if recorded_frames is not None else container_frames
        if self._total_frames <= 0:
            raise ValueError(
                f"{self._path} reports {self._total_frames} frames, from the container "
                f"({container_frames}) and from config/models.json "
                f"({recorded_frames if recorded_frames is not None else 'no entry'}). "
                f"A clip of no frames has no frame grid and cannot be paced or stepped."
            )
        self._container_total_frames = container_frames

        self._step = max(
            MIN_STEP,
            int(round(self._source_fps / self._target_fps)),
        )
        self._interval = 1.0 / self._target_fps

        self._position = 0
        self._loop_id = 0
        self._finished = False
        self._closed = False
        self._decode_position = 0
        self._deadline: float | None = None

    # -- recorded facts -------------------------------------------------------------

    def _recorded_rates(self) -> tuple[float | None, int | None]:
        """`(fps, frame_count)` from `config/models.json` for this path, or `(None, None)`.

        Read through `app.engine.retail_config.clip_measurements`, which is the app-layer
        reader for the clip's measured facts and which **raises** rather than defaulting
        when the entry is missing or incomplete. That raising is wanted here too: a
        defaulted fps would silently repace the whole pipeline, and a defaulted frame
        count would make the source finish at the wrong frame.
        """
        if self._media is None:
            return None, None
        try:
            entry = clip_measurements(str(self._media["id"]))
        except (KeyError, FileNotFoundError):
            # The media list named the file but the validated reader cannot find it.
            # That is a config fault, and falling through to the container here would
            # hide it - the container is a measurement, the config is the pin.
            raise
        return float(entry["measured_fps"]), int(entry["measured_frames"])

    # -- the recorded identity of the clip ------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def scenario_id(self) -> str:
        return self._scenario_id

    @property
    def media_id(self) -> str | None:
        """The `config/models.json` media id, or `None` for an unlisted clip."""
        return self._media_id

    @property
    def source_fps(self) -> float:
        """The clip's own frame rate: the explicit argument, else the recorded value,
        else the container's `CAP_PROP_FPS`."""
        return self._source_fps

    @property
    def container_source_fps(self) -> float:
        """What `cv2.VideoCapture` reports for this file, whatever else was preferred.

        Exposed so the two can be compared. They agree for the pinned clip
        (29.97002997002997 both ways); a disagreement means the config describes a
        different file, which is worth seeing rather than hiding behind a preference.
        """
        return self._container_source_fps

    @property
    def total_frames(self) -> int:
        """How many frames one loop of this clip has."""
        return self._total_frames

    @property
    def container_total_frames(self) -> int:
        """What `cv2.VideoCapture` reports, for comparison with `total_frames`."""
        return self._container_total_frames

    @property
    def target_fps(self) -> float:
        """The **requested** analysis rate. See `analysis_fps` for the achieved one."""
        return self._target_fps

    @property
    def step(self) -> int:
        """Source frames consumed per analysis frame. At least 1."""
        return self._step

    @property
    def analysis_fps(self) -> float:
        """`source_fps / step` - the rate the discrete frame grid actually delivers.

        For the pinned clip at a 15 fps target this is **14.985**, not 15.000. The
        difference is not noise to be rounded away; it is the quantisation of a
        requested rate onto whole frames, and a report that quoted 15 would be quoting
        the request.
        """
        return self._source_fps / float(self._step)

    # -- position -------------------------------------------------------------------

    @property
    def position(self) -> int:
        """The source frame index of the next frame `read()` will return."""
        return self._position

    @property
    def loop_id(self) -> int:
        """How many times this clip has been rewound. Also `source_loop_id` on packets."""
        return self._loop_id

    @property
    def finished(self) -> bool:
        """Whether the clip has been read past its last frame. It has **not** wrapped."""
        return self._finished

    # -- reading --------------------------------------------------------------------

    def read(self) -> FramePacket | None:
        """The next frame to analyse, or `None` once the clip is exhausted.

        Blocks until the next `target_fps` slot, **except** for the first frame, which is
        returned immediately so the booth does not show a black screen for a pacing
        interval. Returns `None` - rather than wrapping - at the end, because a silent
        rewind would invalidate every result in flight without anybody deciding to.

        `captured_at` is taken before the decode and `decoded_at` after it, so the
        packet's own decode latency is the cost of decoding this frame and nothing else.
        """
        self._check_open()
        if self._finished:
            return None
        self._pace()
        captured_at = float(self._clock())
        ok, image = self._retrieve(self._position)
        decoded_at = float(self._clock())
        if not ok or image is None:
            # The decoder ran out before the recorded frame count said it would. That is
            # a fact about this file, so the source says so; it does not read past the
            # end and it does not report a frame it did not decode.
            self._finished = True
            return None
        packet = FramePacket(
            session_id=self._session_id,
            scenario_id=self._scenario_id,
            stream_id=self._stream_id,
            frame_id=self._position,
            source_pts=self._position / self._source_fps,
            source_loop_id=self._loop_id,
            captured_at=captured_at,
            decoded_at=decoded_at,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            image=image,
        )
        self._position += self._step
        if self._position >= self._total_frames:
            self._finished = True
        return packet

    def advance(self) -> int:
        """Skip the next analysis frame without decoding it, and return the position.

        The honest way to drop a frame the worker will never see. It neither paces nor
        touches the decoder - `grab()`-ing the skipped frames lazily is the decoder's
        business, and it happens on the next `read()`.
        """
        self._check_open()
        self._position += self._step
        if self._position >= self._total_frames:
            self._finished = True
        return self._position

    def restart(self) -> int:
        """Rewind to frame 0 of the next loop, and return the **new** `source_loop_id`.

        Every result already in flight refers to the old loop and is now obsolete; that
        is the point. The decoder is really seeked, not just the index moved, because a
        source that reports `position == 0` while the decoder sits at frame 3386 would
        return the last frame labelled as the first.
        """
        self._check_open()
        self._position = 0
        self._finished = False
        self._loop_id += 1
        self._decode_position = 0
        self._deadline = None
        if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
            # Seeking failed. Refusing to continue is the only answer that does not put
            # a mislabelled frame on the screen: the next read would otherwise decode
            # wherever the decoder actually is and label it frame 0 of loop N+1.
            raise OSError(
                f"cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, 0) failed on "
                f"{self._path}. Continuing would decode from wherever the decoder is and "
                f"label it frame 0 of loop {self._loop_id}, which is the wrong frame "
                f"wearing the right name."
            )
        return self._loop_id

    def close(self) -> None:
        """Release the decoder. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._capture.release()

    # -- internals ------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                f"this ClipSource over {self._path} is closed. It returns nothing rather "
                f"than `None`: `None` means the clip is exhausted, and reporting a "
                f"closed source as an exhausted one is a false claim about the footage."
            )

    def _pace(self) -> None:
        """Wait until the next analysis slot, then move the slot on by one interval.

        The deadline never accumulates unbounded debt. A reader that falls behind falls
        behind, and the next deadline is re-based on the current time rather than a burst
        of catch-up reads firing back to back. That is the honest behaviour: the way to
        see that the pipeline cannot keep up is to measure the achieved rate and the
        frame age, not to have the source silently change its own rate.
        """
        now = float(self._clock())
        if self._deadline is None:
            self._deadline = now
            return
        self._deadline += self._interval
        remaining = self._deadline - now
        if remaining <= 0.0:
            self._deadline = now
            return
        self._sleeper(remaining)

    def _retrieve(self, position: int) -> tuple[bool, Any]:
        """Decode the frame at `position`, grabbing the frames in between.

        `grab()` is always called before `retrieve()`, including for the very first
        frame: `cv2`'s `retrieve()` returns `False` on a capture that has not yet
        grabbed anything, which would have made this method report an empty clip
        starting at frame 0.
        """
        if position < self._decode_position:
            # Only reachable if a caller reached in and moved the decoder. Re-seek
            # rather than decode forward past the end and label the wrong picture.
            if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, position):
                raise OSError(
                    f"cv2.VideoCapture could not seek back to frame {position} of "
                    f"{self._path}"
                )
            self._decode_position = position
        while self._decode_position <= position:
            if not self._capture.grab():
                self._decode_position = self._total_frames
                return False, None
            self._decode_position += 1
        ok, image = self._capture.retrieve()
        self._decode_position = position + 1
        return bool(ok), image

    # -- context manager and diagnostics --------------------------------------------

    def __enter__(self) -> "ClipSource":
        self._check_open()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"ClipSource({self._path.name!r}, stream={self._stream_id!r}, "
                f"loop={self._loop_id}, position={self._position}/"
                f"{self._total_frames}, finished={self._finished}, "
                f"analysis_fps={self.analysis_fps:.4f} of {self._source_fps:.4f})")
