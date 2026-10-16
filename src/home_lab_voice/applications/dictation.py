from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

from ..contracts import ContractError


IDEMPOTENCY_KEY = re.compile(r"^idem_[A-Za-z0-9]{8,64}$")


def _digest(content: bytes) -> str:
    return sha256(content).hexdigest()


@dataclass(frozen=True)
class UndoToken:
    target: Path
    before: bytes
    before_digest: str
    after_digest: str
    idempotency_key: str


class ManagedTextDestination:
    """Commit reviewed UTF-8 text to a bounded directory with conflict-aware undo."""

    def __init__(self, root: Path, *, max_chars: int = 20_000) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_chars = max_chars
        self._completed: dict[str, UndoToken] = {}

    def _target(self, relative_name: str) -> Path:
        if not relative_name or Path(relative_name).is_absolute():
            raise ContractError("DESTINATION_INVALID", "Destination must be a relative managed text file.")
        target = (self.root / relative_name).resolve()
        if not target.is_relative_to(self.root) or target.suffix.lower() != ".txt":
            raise ContractError("DESTINATION_INVALID", "Destination must be a relative managed text file.")
        return target

    def commit(self, relative_name: str, text: str, idempotency_key: str) -> UndoToken:
        if not IDEMPOTENCY_KEY.fullmatch(idempotency_key):
            raise ContractError("IDEMPOTENCY_KEY_INVALID", "Idempotency key is invalid.")
        if not text.strip() or len(text) > self.max_chars:
            raise ContractError("DICTATION_TEXT_INVALID", "Reviewed dictation is empty or exceeds its limit.")
        target = self._target(relative_name)
        new_content = text.encode("utf-8")
        if idempotency_key in self._completed:
            earlier = self._completed[idempotency_key]
            if earlier.target != target or earlier.after_digest != _digest(new_content):
                raise ContractError("IDEMPOTENCY_CONFLICT", "Idempotency key was reused for different text or target.")
            return earlier

        before = target.read_bytes() if target.exists() else b""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(new_content)
        token = UndoToken(
            target=target,
            before=before,
            before_digest=_digest(before),
            after_digest=_digest(new_content),
            idempotency_key=idempotency_key,
        )
        self._completed[idempotency_key] = token
        return token

    def undo(self, token: UndoToken) -> None:
        target = token.target.resolve()
        if not target.is_relative_to(self.root):
            raise ContractError("DESTINATION_INVALID", "Undo target is outside the managed destination.")
        current = target.read_bytes() if target.exists() else b""
        if _digest(current) != token.after_digest:
            raise ContractError("UNDO_CONFLICT", "Destination changed after the dictation commit.")
        if token.before:
            target.write_bytes(token.before)
        elif target.exists():
            target.unlink()
