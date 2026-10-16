from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from .contracts import ContractError


class EndpointState(StrEnum):
    IDLE = "idle"
    IN_SPEECH = "in_speech"
    CLOSED = "closed"


class ClosureReason(StrEnum):
    SILENCE_TIMEOUT = "silence_timeout"
    MAX_DURATION = "max_duration"
    CANCELLED = "cancelled"
    QUEUE_OVERFLOW = "queue_overflow"


@dataclass(frozen=True)
class EndpointConfig:
    frame_ms: int = 20
    speech_threshold: float = 0.60
    onset_frames: int = 3
    offset_frames: int = 10
    pre_roll_frames: int = 5
    min_speech_frames: int = 3
    max_utterance_frames: int = 1_500
    max_buffered_frames: int = 1_600

    def __post_init__(self) -> None:
        positive = {
            "frame_ms": self.frame_ms,
            "onset_frames": self.onset_frames,
            "offset_frames": self.offset_frames,
            "min_speech_frames": self.min_speech_frames,
            "max_utterance_frames": self.max_utterance_frames,
            "max_buffered_frames": self.max_buffered_frames,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError("endpointing durations and limits must be positive")
        if self.pre_roll_frames < 0:
            raise ValueError("pre-roll must not be negative")
        if not 0.0 <= self.speech_threshold <= 1.0:
            raise ValueError("speech threshold must be between zero and one")
        if self.onset_frames > self.max_buffered_frames:
            raise ValueError("onset evidence must fit in the buffer")
        if self.max_utterance_frames > self.max_buffered_frames:
            raise ValueError("maximum utterance must fit in the buffer")


@dataclass(frozen=True)
class ScoredFrame:
    index: int
    probability: float
    payload: bytes = b""


@dataclass(frozen=True)
class Utterance:
    frames: tuple[ScoredFrame, ...]
    start_ms: int
    end_ms: int
    speech_frames: int
    reason: ClosureReason


class EndpointDetector:
    """Assemble one bounded utterance from deterministic speech scores."""

    def __init__(self, config: EndpointConfig | None = None) -> None:
        self.config = config or EndpointConfig()
        self.reset()

    def reset(self) -> None:
        self.state = EndpointState.IDLE
        self._ring: deque[ScoredFrame] = deque(maxlen=self.config.pre_roll_frames + self.config.onset_frames)
        self._utterance: list[ScoredFrame] = []
        self._onset_run = 0
        self._silence_run = 0
        self._speech_frames = 0
        self._last_index = -1

    def feed(self, probability: float, payload: bytes = b"") -> Utterance | None:
        if self.state is EndpointState.CLOSED:
            raise ContractError("ENDPOINT_STREAM_CLOSED", "Reset the endpoint detector before reusing it.")
        if not 0.0 <= probability <= 1.0:
            raise ContractError("VAD_SCORE_INVALID", "Speech probability must be between zero and one.")

        self._last_index += 1
        frame = ScoredFrame(self._last_index, probability, payload)
        is_speech = probability >= self.config.speech_threshold

        if self.state is EndpointState.IDLE:
            self._ring.append(frame)
            self._onset_run = self._onset_run + 1 if is_speech else 0
            if self._onset_run < self.config.onset_frames:
                return None
            self.state = EndpointState.IN_SPEECH
            self._utterance = list(self._ring)
            self._speech_frames = self.config.onset_frames
            self._silence_run = 0
            self._ring.clear()
            return self._check_bounds()

        self._utterance.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        bounded = self._check_bounds()
        if bounded is not None:
            return bounded
        if self._silence_run >= self.config.offset_frames:
            return self._close(ClosureReason.SILENCE_TIMEOUT)
        return None

    def cancel(self) -> Utterance | None:
        if self.state is EndpointState.CLOSED:
            return None
        if self.state is EndpointState.IDLE:
            self.reset()
            return None
        return self._close(ClosureReason.CANCELLED)

    def queue_overflow(self) -> Utterance | None:
        if self.state is EndpointState.CLOSED:
            return None
        if self.state is EndpointState.IDLE:
            self.reset()
            return None
        return self._close(ClosureReason.QUEUE_OVERFLOW)

    def feed_many(self, probabilities: Iterable[float]) -> Utterance | None:
        for probability in probabilities:
            utterance = self.feed(probability)
            if utterance is not None:
                return utterance
        return None

    def _check_bounds(self) -> Utterance | None:
        if len(self._utterance) > self.config.max_buffered_frames:
            return self._close(ClosureReason.QUEUE_OVERFLOW)
        if len(self._utterance) >= self.config.max_utterance_frames:
            return self._close(ClosureReason.MAX_DURATION)
        return None

    def _close(self, reason: ClosureReason) -> Utterance:
        frames = tuple(self._utterance)
        if not frames:
            raise ContractError("ENDPOINT_EMPTY", "An empty utterance cannot be closed.")
        if self._speech_frames < self.config.min_speech_frames and reason is ClosureReason.SILENCE_TIMEOUT:
            frames = ()
        self.state = EndpointState.CLOSED
        if not frames:
            start_ms = end_ms = 0
        else:
            start_ms = frames[0].index * self.config.frame_ms
            end_ms = (frames[-1].index + 1) * self.config.frame_ms
        return Utterance(frames, start_ms, end_ms, self._speech_frames, reason)
