from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from home_lab_voice.audio_fixtures import (  # noqa: E402
    inspect_pcm16_wav,
    write_mono_sine,
    write_stereo_tone_and_silence,
)
from home_lab_voice.contracts import ContractError  # noqa: E402


class AudioFixtureTests(unittest.TestCase):
    def test_mono_reference_properties(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_mono_sine(Path(directory) / "mono.wav")
            info = inspect_pcm16_wav(path)
            self.assertEqual(info.channels, 1)
            self.assertEqual(info.sample_width_bytes, 2)
            self.assertEqual(info.sample_rate_hz, 16_000)
            self.assertEqual(info.frame_count, 16_000)
            self.assertEqual(info.data_bytes, 32_000)
            self.assertEqual(info.duration_seconds, 1.0)
            self.assertEqual(len(info.file_sha256), 64)
            self.assertGreater(info.peak_per_channel[0], 8_000)

    def test_stereo_channel_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_stereo_tone_and_silence(Path(directory) / "stereo.wav")
            info = inspect_pcm16_wav(path)
            self.assertEqual(info.channels, 2)
            self.assertEqual(info.data_bytes, 64_000)
            self.assertGreater(info.peak_per_channel[0], 8_000)
            self.assertEqual(info.peak_per_channel[1], 0)

    def test_dc_offset_is_observable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_mono_sine(Path(directory) / "offset.wav", amplitude=0.0, dc_offset=0.1)
            info = inspect_pcm16_wav(path)
            self.assertAlmostEqual(info.mean_per_channel[0], 3277, delta=1)

    def test_truncated_data_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_mono_sine(Path(directory) / "truncated.wav")
            content = path.read_bytes()
            path.write_bytes(content[:-100])
            with self.assertRaises(ContractError) as caught:
                inspect_pcm16_wav(path)
            self.assertEqual(caught.exception.code, "WAV_TRUNCATED")

    def test_random_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-a-wave.wav"
            path.write_bytes(b"not a wave")
            with self.assertRaises(ContractError) as caught:
                inspect_pcm16_wav(path)
            self.assertEqual(caught.exception.code, "WAV_MALFORMED")

    def test_decoded_size_limit_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_mono_sine(Path(directory) / "bounded.wav")
            with self.assertRaises(ContractError) as caught:
                inspect_pcm16_wav(path, max_decoded_bytes=1_000)
            self.assertEqual(caught.exception.code, "AUDIO_DECODE_LIMIT")


if __name__ == "__main__":
    unittest.main()
