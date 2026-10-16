from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from home_lab_voice.evaluation import normalize_words, word_error_counts  # noqa: E402


class NormalizationTests(unittest.TestCase):
    def test_case_punctuation_and_whitespace_are_normalized(self) -> None:
        self.assertEqual(normalize_words("  Lab, STATUS!\nReady? "), ("lab", "status", "ready"))

    def test_unicode_compatibility_is_normalized(self) -> None:
        self.assertEqual(normalize_words("ＦＵＬＬ width"), ("full", "width"))

    def test_numbers_are_not_rewritten(self) -> None:
        self.assertNotEqual(normalize_words("sixteen"), normalize_words("16"))


class WordErrorRateTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        result = word_error_counts("service ready", "service ready")
        self.assertEqual(result.errors, 0)
        self.assertEqual(result.wer, 0.0)

    def test_substitution_deletion_and_insertion_counts(self) -> None:
        substitution = word_error_counts("service is ready", "service was ready")
        deletion = word_error_counts("service is ready", "service ready")
        insertion = word_error_counts("service ready", "the service ready")
        self.assertEqual((substitution.substitutions, substitution.deletions, substitution.insertions), (1, 0, 0))
        self.assertEqual((deletion.substitutions, deletion.deletions, deletion.insertions), (0, 1, 0))
        self.assertEqual((insertion.substitutions, insertion.deletions, insertion.insertions), (0, 0, 1))

    def test_rate_uses_reference_word_denominator(self) -> None:
        result = word_error_counts("one two three four", "one too three")
        self.assertEqual(result.errors, 2)
        self.assertEqual(result.reference_words, 4)
        self.assertEqual(result.wer, 0.5)

    def test_empty_pair_has_zero_rate(self) -> None:
        result = word_error_counts("", "")
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.errors, 0)

    def test_nonempty_hypothesis_with_empty_reference_has_undefined_rate(self) -> None:
        result = word_error_counts("", "unexpected words")
        self.assertIsNone(result.wer)
        self.assertEqual(result.insertions, 2)


if __name__ == "__main__":
    unittest.main()
