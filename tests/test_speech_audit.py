import sys,tempfile,unittest,wave,struct
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from audit_speech_file import audit
class SpeechAuditTests(unittest.TestCase):
    def test_silence_and_endpoint_counts_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'test.wav'
            with wave.open(str(p),'wb') as out:
                out.setnchannels(1);out.setsampwidth(2);out.setframerate(16000);out.writeframes(struct.pack('<hhhh',0,32767,-32768,0))
            r=audit(p);s=r['channel_statistics'][0]
            self.assertEqual(s['exact_zero_fraction'],.5);self.assertEqual(s['near_full_scale_samples'],2)
            self.assertEqual(s['peak_fraction_full_scale'],1)
