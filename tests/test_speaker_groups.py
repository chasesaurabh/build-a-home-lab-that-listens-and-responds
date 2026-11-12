import importlib.util
import unittest
from home_lab_voice.speakers import average_linkage

@unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy optional dependency required')
class SpeakerGroupTests(unittest.TestCase):
    def test_geometry_and_deterministic_labels(self):
        labels=average_linkage([[1,0],[0,1],[.99,.01],[.01,.99]],2)
        self.assertEqual(labels,[0,1,0,1])
    def test_wrong_count_forces_merge(self):
        self.assertEqual(average_linkage([[1,0],[0,1]],1),[0,0])
    def test_invalid_vectors_are_rejected(self):
        for vectors,count in [([],1),([[0,0]],1),([[float('nan'),1]],1),([[1,0]],2)]:
            with self.subTest(vectors=vectors,count=count),self.assertRaises(ValueError):average_linkage(vectors,count)
