from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re


@dataclass(frozen=True)
class NarrationChunk:
    chunk_id: str
    sequence: int
    source_start: int
    source_end: int
    text: str
    text_sha256: str


def _bounded_pieces(text: str, max_chars: int) -> list[str]:
    sentences = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text.strip()) if piece.strip()]
    pieces: list[str] = []
    for sentence in sentences:
        remainder = sentence
        while len(remainder) > max_chars:
            split_at = remainder.rfind(" ", 0, max_chars + 1)
            if split_at <= 0:
                split_at = max_chars
            pieces.append(remainder[:split_at].strip())
            remainder = remainder[split_at:].strip()
        if remainder:
            pieces.append(remainder)
    return pieces


def plan_narration(source_text: str, *, max_chars: int = 500) -> tuple[NarrationChunk, ...]:
    if max_chars < 20:
        raise ValueError("max_chars must be at least 20")
    if not source_text.strip():
        return ()

    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", source_text):
        if paragraph.strip():
            pieces.extend(_bounded_pieces(paragraph, max_chars))

    chunks: list[NarrationChunk] = []
    cursor = 0
    for sequence, piece in enumerate(pieces, start=1):
        start = source_text.find(piece, cursor)
        if start < 0:
            # Whitespace inside a paragraph may have been folded by a caller. The
            # reference planner fails rather than inventing a source location.
            raise ValueError("chunk text cannot be mapped to its source")
        end = start + len(piece)
        digest = sha256(piece.encode("utf-8")).hexdigest()
        chunk_id = f"chunk_{sequence:04d}_{digest[:12]}"
        chunks.append(NarrationChunk(chunk_id, sequence, start, end, piece, digest))
        cursor = end
    return tuple(chunks)
