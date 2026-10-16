from __future__ import annotations

from dataclasses import dataclass
import unicodedata


@dataclass(frozen=True)
class WerResult:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    hypothesis_words: int
    wer: float | None

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def normalize_words(text: str) -> tuple[str, ...]:
    """Apply the book's explicit baseline WER normalization.

    The baseline uses Unicode NFKC, case folding, punctuation-to-space, and
    whitespace tokenization. Number words and digits are deliberately not
    rewritten because that requires an application-specific policy.
    """

    normalized = unicodedata.normalize("NFKC", text).casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return tuple(without_punctuation.split())


def word_error_counts(reference: str, hypothesis: str) -> WerResult:
    reference_words = normalize_words(reference)
    hypothesis_words = normalize_words(hypothesis)
    rows = len(reference_words) + 1
    columns = len(hypothesis_words) + 1

    # Each cell stores (total edits, substitutions, deletions, insertions).
    distance: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(columns)] for _ in range(rows)
    ]
    for row in range(1, rows):
        distance[row][0] = (row, 0, row, 0)
    for column in range(1, columns):
        distance[0][column] = (column, 0, 0, column)

    for row in range(1, rows):
        for column in range(1, columns):
            if reference_words[row - 1] == hypothesis_words[column - 1]:
                distance[row][column] = distance[row - 1][column - 1]
                continue
            diagonal = distance[row - 1][column - 1]
            deletion = distance[row - 1][column]
            insertion = distance[row][column - 1]
            candidates = (
                (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            )
            distance[row][column] = min(candidates)

    _, substitutions, deletions, insertions = distance[-1][-1]
    reference_count = len(reference_words)
    if reference_count:
        rate: float | None = (substitutions + deletions + insertions) / reference_count
    elif not hypothesis_words:
        rate = 0.0
    else:
        rate = None
    return WerResult(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=reference_count,
        hypothesis_words=len(hypothesis_words),
        wer=rate,
    )
