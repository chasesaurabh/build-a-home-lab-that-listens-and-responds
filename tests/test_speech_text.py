import unittest
from home_lab_voice.speech_text import prepare_text
class SpeechTextTests(unittest.TestCase):
    def test_longest_match_and_offsets(self):
        rules=[{'id':'short','source':'16','spoken':'sixteen'},{'id':'long','source':'16 GB','spoken':'sixteen gigabytes'}]
        source='Use 16 GB and 16.';r=prepare_text(source,rules)
        self.assertEqual(r['spoken_text'],'Use sixteen gigabytes and sixteen.')
        for change in r['changes']:
            self.assertEqual(source[change['source_start']:change['source_end']],change['source_text'])
            self.assertEqual(r['spoken_text'][change['spoken_start']:change['spoken_end']],change['spoken_text'])
    def test_nonrecursive_and_word_boundaries(self):
        rules=[{'id':'a','source':'ASR','spoken':'LLM'},{'id':'b','source':'LLM','spoken':'ASR'}]
        self.assertEqual(prepare_text('ASR LLM XASR ASR2 asr',rules)['spoken_text'],'LLM ASR XASR ASR2 asr')
    def test_ambiguous_date_remains_literal(self):
        self.assertEqual(prepare_text('03/04/2026',[])['spoken_text'],'03/04/2026')
    def test_duplicate_and_expansion_limit(self):
        r={'id':'a','source':'X','spoken':'z'*200}
        with self.assertRaises(ValueError):prepare_text('X',[r,r])
        with self.assertRaises(ValueError):prepare_text('X '*300,[r])
