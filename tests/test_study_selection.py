from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from librispeech_selection import select

class SelectionTests(unittest.TestCase):
    def test_order_independent_and_speaker_balanced(self):
        refs={f'{speaker}-1-{clip}':'Permitted reference' for speaker in range(1,5) for clip in range(1,5)}
        result=select(refs,8)
        self.assertEqual(result,select(dict(reversed(list(refs.items()))),8))
        self.assertEqual(len(set(result)),8)
        self.assertEqual({speaker:sum(i.startswith(speaker+'-') for i in result) for speaker in ('1','2','3','4')},{'1':2,'2':2,'3':2,'4':2})
    def test_paths_empty_references_and_insufficient_inputs_rejected(self):
        for refs,count in [({'../../outside':'text'},1),({'1-2-3/../../x':'text'},1),({'1-2-3':''},1),({'1-2-3':'text'},2)]:
            with self.subTest(refs=refs),self.assertRaises(ValueError):select(refs,count)
if __name__=='__main__':unittest.main()
