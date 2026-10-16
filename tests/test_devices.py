from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from home_lab_voice.contracts import ContractError
from home_lab_voice.devices import capture_pcm16_wav, list_audio_devices, play_pcm16_wav, select_audio_device


class FakeAudioBackend:
    def __init__(self, *, truncate: bool = False):
        self.truncate = truncate
        self.checked: list[dict[str, object]] = []
        self.output_checked: list[dict[str, object]] = []
        self.captures: list[dict[str, object]] = []
        self.playbacks: list[dict[str, object]] = []

    def query_devices(self):
        return (
            {
                "name": "Fixture Microphone",
                "hostapi": 0,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48_000,
            },
            {
                "name": "Fixture Speakers",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48_000,
            },
            {
                "name": "Second Fixture Microphone",
                "hostapi": 1,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 16_000,
            },
        )

    def query_hostapis(self):
        return ({"name": "Fixture API"}, {"name": "Alternate API"})

    def default_devices(self):
        return 0, 1

    def check_input_settings(self, **settings):
        self.checked.append(settings)

    def capture_pcm16(self, **settings):
        self.captures.append(settings)
        size = settings["frame_count"] * settings["channels"] * 2
        if self.truncate:
            size -= 2
        return b"\x00" * size, 0

    def check_output_settings(self, **settings):
        self.output_checked.append(settings)

    def play_pcm16(self, content, **settings):
        self.playbacks.append({"content": content, **settings})
        return 0


class DeviceTests(unittest.TestCase):
    def test_device_inventory_marks_defaults_and_host_api(self):
        devices = list_audio_devices(FakeAudioBackend())
        self.assertEqual(len(devices), 3)
        self.assertTrue(devices[0].default_input)
        self.assertTrue(devices[1].default_output)
        self.assertEqual(devices[2].host_api, "Alternate API")

    def test_selects_default_and_unique_substring(self):
        devices = list_audio_devices(FakeAudioBackend())
        self.assertEqual(select_audio_device(devices, direction="input", selector=None).index, 0)
        self.assertEqual(select_audio_device(devices, direction="input", selector="Second").index, 2)
        self.assertEqual(select_audio_device(devices, direction="output", selector="1").index, 1)

    def test_selects_duplicate_name_by_host_api_qualification(self):
        devices = list(list_audio_devices(FakeAudioBackend()))
        devices.append(
            devices[0].__class__(
                index=3,
                name="Fixture Microphone",
                host_api="Alternate API",
                input_channels=2,
                output_channels=0,
                default_sample_rate_hz=44_100,
                default_input=False,
                default_output=False,
            )
        )
        selected = select_audio_device(
            devices,
            direction="input",
            selector="Alternate API::Fixture Microphone",
        )
        self.assertEqual(selected.index, 3)

    def test_rejects_malformed_host_api_qualification(self):
        devices = list_audio_devices(FakeAudioBackend())
        with self.assertRaisesRegex(ContractError, "HOST_API::DEVICE_NAME"):
            select_audio_device(devices, direction="input", selector="Fixture API::")

    def test_ambiguous_and_wrong_direction_are_rejected(self):
        devices = list_audio_devices(FakeAudioBackend())
        with self.assertRaisesRegex(ContractError, "ambiguous"):
            select_audio_device(devices, direction="input", selector="Fixture")
        with self.assertRaisesRegex(ContractError, "No matching output"):
            select_audio_device(devices, direction="output", selector="0")

    def test_bounded_capture_writes_valid_pcm(self):
        backend = FakeAudioBackend()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "capture.wav"
            result = capture_pcm16_wav(
                output,
                device_selector="0",
                duration_seconds=0.1,
                backend=backend,
            )
            self.assertEqual(result.wave.frame_count, 1_600)
            self.assertEqual(result.wave.sample_rate_hz, 16_000)
            self.assertEqual(result.wave.peak_per_channel, (0,))
            self.assertEqual(backend.checked[0]["device"], 0)
            self.assertEqual(backend.checked[0]["samplerate"], 48_000)
            self.assertEqual(backend.captures[0]["frame_count"], 4_800)
            self.assertEqual(backend.captures[0]["sample_rate_hz"], 48_000)

    def test_capture_requires_explicit_overwrite(self):
        backend = FakeAudioBackend()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "capture.wav"
            output.write_bytes(b"existing")
            with self.assertRaisesRegex(ContractError, "already exists"):
                capture_pcm16_wav(output, duration_seconds=0.1, backend=backend)

    def test_capture_bounds_and_truncation_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "capture.wav"
            with self.assertRaisesRegex(ContractError, "between"):
                capture_pcm16_wav(output, duration_seconds=31, backend=FakeAudioBackend())
            with self.assertRaisesRegex(ContractError, "shorter"):
                capture_pcm16_wav(
                    output,
                    duration_seconds=0.1,
                    backend=FakeAudioBackend(truncate=True),
                )
            self.assertFalse(output.exists())

    def test_bounded_playback_uses_selected_output_format(self):
        backend = FakeAudioBackend()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "speech.wav"
            from home_lab_voice.audio_fixtures import write_mono_sine

            write_mono_sine(source, sample_rate_hz=16_000, duration_seconds=0.1)
            result = play_pcm16_wav(source, device_selector="1", backend=backend)
        self.assertEqual(result.device.name, "Fixture Speakers")
        self.assertEqual(backend.output_checked[0]["samplerate"], 48_000)
        self.assertEqual(backend.playbacks[0]["channels"], 1)
        self.assertEqual(len(backend.playbacks[0]["content"]), 9_600)
        self.assertEqual(result.playback_sample_rate_hz, 48_000)


if __name__ == "__main__":
    unittest.main()
