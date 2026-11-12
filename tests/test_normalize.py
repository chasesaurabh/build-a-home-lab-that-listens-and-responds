import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import wave
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from home_lab_voice.normalize import normalize_audio
from home_lab_voice.contracts import ContractError

@unittest.skipUnless(importlib.util.find_spec('av'),'PyAV optional integration dependency')
class NormalizeTests(unittest.TestCase):
    def test_real_resampler_and_original_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.wav'
            with wave.open(str(source),'wb') as writer:
                writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(24000);writer.writeframes(struct.pack('<h',1000)*24000)
            original=source.read_bytes();result=normalize_audio(source,root/'normalized.wav')
            self.assertEqual(result['samples'],16000);self.assertEqual(result['input_rate_hz'],24000)
            self.assertEqual(source.read_bytes(),original)
            with self.assertRaises(ContractError):normalize_audio(source,root/'normalized.wav')
    def test_stereo_needs_deliberate_downmix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.wav'
            with wave.open(str(source),'wb') as writer:
                writer.setnchannels(2);writer.setsampwidth(2);writer.setframerate(16000);writer.writeframes(struct.pack('<hh',1000,-1000)*16000)
            with self.assertRaises(ContractError):normalize_audio(source,root/'normalized.wav')
            result=normalize_audio(source,root/'normalized.wav',allow_downmix=True)
            self.assertEqual(result['input_channels'],2);self.assertEqual(result['output_channels'],1)
    def test_invalid_file_has_no_output_or_temporary_derivative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.wav';source.write_bytes(b'not an audio file')
            with self.assertRaises(ContractError):normalize_audio(source,root/'normalized.wav')
            self.assertEqual([p.name for p in root.iterdir()],['source.wav'])
if __name__=='__main__':unittest.main()
