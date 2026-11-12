import json,tempfile,unittest
from pathlib import Path
from home_lab_voice.transcript_layers import build_layers,preserve_layers
class TranscriptLayerTests(unittest.TestCase):
    def raw(self):return {'duration_seconds':2.2,'segments':[{'start_seconds':0,'end_seconds':2.22,'text':'  hello   lab '}]}
    def test_preserves_raw_bytes_and_marks_display_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.json';data=json.dumps(self.raw()).encode();source.write_bytes(data)
            result=preserve_layers(source,root/'review',{'0':'Hello, lab.'})
            self.assertEqual((root/'review/raw.json').read_bytes(),data);self.assertEqual(source.read_bytes(),data)
            row=result['segments'][0];self.assertTrue(row['display_time_clamped']);self.assertEqual(row['raw_end_seconds'],2.22)
            self.assertEqual(row['display_end_ms'],2200);self.assertEqual(row['normalized_text'],'hello lab')
            self.assertEqual(row['corrected_text'],'Hello, lab.');self.assertTrue(row['reviewed_edit'])
    def test_invalid_time_and_unknown_correction_rejected(self):
        for value in [-1,float('nan'),3]:
            raw=self.raw();raw['segments'][0]['end_seconds']=value
            with self.assertRaises(ValueError):build_layers(raw)
        with self.assertRaises(ValueError):build_layers(self.raw(),{'1':'Invented segment'})
        with self.assertRaises(ValueError):build_layers(self.raw(),{'00':'Ambiguous index'})
    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.json';source.write_text(json.dumps(self.raw()))
            out=root/'review';preserve_layers(source,out)
            with self.assertRaises(FileExistsError):preserve_layers(source,out)
