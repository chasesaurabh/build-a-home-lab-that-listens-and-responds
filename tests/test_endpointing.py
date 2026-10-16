from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from home_lab_voice.contracts import ContractError  # noqa: E402
from home_lab_voice.endpointing import (  # noqa: E402
    ClosureReason,
    EndpointConfig,
    EndpointDetector,
    EndpointState,
)


def compact_config(**overrides: object) -> EndpointConfig:
    values = {
        "frame_ms": 20,
        "speech_threshold": 0.6,
        "onset_frames": 3,
        "offset_frames": 4,
        "pre_roll_frames": 2,
        "min_speech_frames": 3,
        "max_utterance_frames": 20,
        "max_buffered_frames": 24,
    }
    values.update(overrides)
    return EndpointConfig(**values)


class EndpointingTests(unittest.TestCase):
    def test_no_speech_stays_idle_and_bounded(self) -> None:
        detector = EndpointDetector(compact_config())
        self.assertIsNone(detector.feed_many([0.1] * 100))
        self.assertEqual(detector.state, EndpointState.IDLE)
        self.assertLessEqual(len(detector._ring), 5)

    def test_normal_utterance_keeps_preroll_and_hangover(self) -> None:
        detector = EndpointDetector(compact_config())
        result = detector.feed_many([0.1, 0.2, 0.7, 0.8, 0.9, 0.8, 0.1, 0.2, 0.1, 0.2])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.reason, ClosureReason.SILENCE_TIMEOUT)
        self.assertEqual(result.start_ms, 0)
        self.assertEqual(result.end_ms, 200)
        self.assertEqual(len(result.frames), 10)
        self.assertEqual(result.speech_frames, 4)

    def test_hesitation_does_not_split_utterance(self) -> None:
        detector = EndpointDetector(compact_config())
        result = detector.feed_many([0.8, 0.8, 0.8, 0.1, 0.1, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.reason, ClosureReason.SILENCE_TIMEOUT)
        self.assertEqual(result.speech_frames, 5)

    def test_short_noise_burst_never_opens(self) -> None:
        detector = EndpointDetector(compact_config())
        self.assertIsNone(detector.feed_many([0.1, 0.8, 0.9, 0.1, 0.1]))
        self.assertEqual(detector.state, EndpointState.IDLE)

    def test_max_duration_closes_stuck_stream(self) -> None:
        detector = EndpointDetector(compact_config(max_utterance_frames=8))
        result = detector.feed_many([0.9] * 20)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.reason, ClosureReason.MAX_DURATION)
        self.assertEqual(len(result.frames), 8)
        self.assertEqual(result.end_ms - result.start_ms, 160)

    def test_cancellation_and_queue_overflow_have_explicit_reasons(self) -> None:
        cancelled = EndpointDetector(compact_config())
        cancelled.feed_many([0.9, 0.9, 0.9, 0.9])
        cancelled_result = cancelled.cancel()
        self.assertIsNotNone(cancelled_result)
        assert cancelled_result is not None
        self.assertEqual(cancelled_result.reason, ClosureReason.CANCELLED)

        overflowed = EndpointDetector(compact_config())
        overflowed.feed_many([0.9, 0.9, 0.9, 0.9])
        overflow_result = overflowed.queue_overflow()
        self.assertIsNotNone(overflow_result)
        assert overflow_result is not None
        self.assertEqual(overflow_result.reason, ClosureReason.QUEUE_OVERFLOW)

    def test_reset_clears_state_and_allows_reuse(self) -> None:
        detector = EndpointDetector(compact_config())
        detector.feed_many([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
        self.assertEqual(detector.state, EndpointState.CLOSED)
        detector.reset()
        self.assertEqual(detector.state, EndpointState.IDLE)
        self.assertEqual(detector._last_index, -1)
        self.assertIsNone(detector.feed(0.1))

    def test_invalid_scores_and_closed_reuse_are_rejected(self) -> None:
        detector = EndpointDetector(compact_config())
        with self.assertRaises(ContractError) as caught:
            detector.feed(1.1)
        self.assertEqual(caught.exception.code, "VAD_SCORE_INVALID")

        detector.feed_many([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
        with self.assertRaises(ContractError) as closed:
            detector.feed(0.1)
        self.assertEqual(closed.exception.code, "ENDPOINT_STREAM_CLOSED")


if __name__ == "__main__":
    unittest.main()
