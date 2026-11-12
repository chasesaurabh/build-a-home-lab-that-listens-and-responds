from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

from .contracts import ContractError


class ConversationState(StrEnum):
    IDLE = "idle"
    ARMED = "armed"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    INTERPRETING = "interpreting"
    CONFIRMING = "confirming"
    ACTING = "acting"
    SPEAKING = "speaking"
    RECOVERING = "recovering"


class ConversationEvent(StrEnum):
    ARM = "arm"
    START_LISTENING = "start_listening"
    AUDIO_READY = "audio_ready"
    TRANSCRIPT_READY = "transcript_ready"
    READ_ONLY_INTENT = "read_only_intent"
    CONFIRMATION_REQUIRED = "confirmation_required"
    CONFIRMED = "confirmed"
    DENIED = "denied"
    ACTION_COMPLETE = "action_complete"
    RESPONSE_READY = "response_ready"
    PLAYBACK_COMPLETE = "playback_complete"
    CANCEL = "cancel"
    TIMEOUT = "timeout"
    FAIL = "fail"
    RECOVER = "recover"


@dataclass(frozen=True)
class Transition:
    sequence: int
    prior: ConversationState
    event: ConversationEvent
    current: ConversationState


@dataclass
class ConversationMachine:
    state: ConversationState = ConversationState.IDLE
    transitions: list[Transition] = field(default_factory=list)
    sequence: int = 0

    _NORMAL = {
        (ConversationState.IDLE, ConversationEvent.ARM): ConversationState.ARMED,
        (ConversationState.ARMED, ConversationEvent.START_LISTENING): ConversationState.LISTENING,
        (ConversationState.LISTENING, ConversationEvent.AUDIO_READY): ConversationState.TRANSCRIBING,
        (ConversationState.TRANSCRIBING, ConversationEvent.TRANSCRIPT_READY): ConversationState.INTERPRETING,
        (ConversationState.INTERPRETING, ConversationEvent.READ_ONLY_INTENT): ConversationState.ACTING,
        (ConversationState.INTERPRETING, ConversationEvent.CONFIRMATION_REQUIRED): ConversationState.CONFIRMING,
        (ConversationState.CONFIRMING, ConversationEvent.CONFIRMED): ConversationState.ACTING,
        (ConversationState.CONFIRMING, ConversationEvent.DENIED): ConversationState.SPEAKING,
        (ConversationState.ACTING, ConversationEvent.ACTION_COMPLETE): ConversationState.SPEAKING,
        (ConversationState.INTERPRETING, ConversationEvent.RESPONSE_READY): ConversationState.SPEAKING,
        (ConversationState.SPEAKING, ConversationEvent.PLAYBACK_COMPLETE): ConversationState.IDLE,
        (ConversationState.RECOVERING, ConversationEvent.RECOVER): ConversationState.IDLE,
    }

    _ACTIVE = {
        ConversationState.ARMED,
        ConversationState.LISTENING,
        ConversationState.TRANSCRIBING,
        ConversationState.INTERPRETING,
        ConversationState.CONFIRMING,
        ConversationState.ACTING,
        ConversationState.SPEAKING,
    }

    def dispatch(self, event: ConversationEvent) -> ConversationState:
        prior = self.state
        if event is ConversationEvent.CANCEL and prior in self._ACTIVE:
            current = ConversationState.IDLE
        elif event in {ConversationEvent.TIMEOUT, ConversationEvent.FAIL} and prior in self._ACTIVE:
            current = ConversationState.RECOVERING
        else:
            current = self._NORMAL.get((prior, event))
            if current is None:
                raise ContractError(
                    "CONVERSATION_TRANSITION_INVALID",
                    f"Event {event.value} is invalid while the conversation is {prior.value}.",
                )
        self.sequence += 1
        self.state = current
        self.transitions.append(Transition(self.sequence, prior, event, current))
        return current

    def reset_history(self) -> None:
        if self.state is not ConversationState.IDLE:
            raise ContractError(
                "CONVERSATION_ACTIVE",
                "Conversation history cannot be cleared while a turn is active.",
            )
        self.transitions.clear()
        self.sequence = 0


class TurnController:
    """Request-scoped control-plane exercise; does not run or cancel devices/models."""

    def __init__(self) -> None:
        self.machine = ConversationMachine()
        self.active_turn: str | None = None

    def begin(self) -> str:
        if self.active_turn is not None or self.machine.state is not ConversationState.IDLE:
            raise ContractError('TURN_ALREADY_ACTIVE', 'Finish or cancel the active turn first.')
        self.machine.reset_history()
        self.active_turn = uuid4().hex
        self.machine.dispatch(ConversationEvent.ARM)
        return self.active_turn

    def dispatch(self, turn_id: str, event: ConversationEvent) -> ConversationState:
        if self.active_turn is None or turn_id != self.active_turn:
            raise ContractError('TURN_STALE', 'The event does not belong to the active turn.')
        if not isinstance(event, ConversationEvent):
            raise ContractError('TURN_EVENT_INVALID', 'Expected a declared conversation event.')
        if len(self.machine.transitions) >= 64:
            raise ContractError('TURN_EVENT_LIMIT', 'Turn event limit reached.')
        state = self.machine.dispatch(event)
        if state is ConversationState.IDLE:
            self.active_turn = None
        return state
