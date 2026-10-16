from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from home_lab_voice.contracts import ContractError  # noqa: E402
from home_lab_voice.conversation import (  # noqa: E402
    ConversationEvent as Event,
    ConversationMachine,
    ConversationState as State,
)


def advance_to_interpreting(machine: ConversationMachine) -> None:
    for event in (Event.ARM, Event.START_LISTENING, Event.AUDIO_READY, Event.TRANSCRIPT_READY):
        machine.dispatch(event)


class ConversationMachineTests(unittest.TestCase):
    def test_read_only_turn_follows_explicit_path(self) -> None:
        machine = ConversationMachine()
        advance_to_interpreting(machine)
        for event in (Event.READ_ONLY_INTENT, Event.ACTION_COMPLETE, Event.PLAYBACK_COMPLETE):
            machine.dispatch(event)
        self.assertEqual(machine.state, State.IDLE)
        self.assertEqual([item.current for item in machine.transitions], [
            State.ARMED, State.LISTENING, State.TRANSCRIBING, State.INTERPRETING,
            State.ACTING, State.SPEAKING, State.IDLE,
        ])

    def test_confirmation_path_cannot_skip_confirmation(self) -> None:
        machine = ConversationMachine()
        advance_to_interpreting(machine)
        machine.dispatch(Event.CONFIRMATION_REQUIRED)
        self.assertEqual(machine.state, State.CONFIRMING)
        with self.assertRaises(ContractError):
            machine.dispatch(Event.ACTION_COMPLETE)
        machine.dispatch(Event.CONFIRMED)
        self.assertEqual(machine.state, State.ACTING)

    def test_denied_confirmation_returns_a_spoken_result(self) -> None:
        machine = ConversationMachine()
        advance_to_interpreting(machine)
        machine.dispatch(Event.CONFIRMATION_REQUIRED)
        machine.dispatch(Event.DENIED)
        self.assertEqual(machine.state, State.SPEAKING)

    def test_non_action_response_bypasses_acting(self) -> None:
        machine = ConversationMachine()
        advance_to_interpreting(machine)
        machine.dispatch(Event.RESPONSE_READY)
        self.assertEqual(machine.state, State.SPEAKING)

    def test_cancel_from_each_active_state_returns_idle(self) -> None:
        sequences = {
            State.ARMED: [Event.ARM],
            State.LISTENING: [Event.ARM, Event.START_LISTENING],
            State.TRANSCRIBING: [Event.ARM, Event.START_LISTENING, Event.AUDIO_READY],
            State.INTERPRETING: [Event.ARM, Event.START_LISTENING, Event.AUDIO_READY, Event.TRANSCRIPT_READY],
        }
        for expected, events in sequences.items():
            with self.subTest(state=expected):
                machine = ConversationMachine()
                for event in events:
                    machine.dispatch(event)
                self.assertEqual(machine.state, expected)
                machine.dispatch(Event.CANCEL)
                self.assertEqual(machine.state, State.IDLE)

    def test_timeout_and_failure_require_recovery(self) -> None:
        for event in (Event.TIMEOUT, Event.FAIL):
            with self.subTest(event=event):
                machine = ConversationMachine()
                machine.dispatch(Event.ARM)
                machine.dispatch(event)
                self.assertEqual(machine.state, State.RECOVERING)
                machine.dispatch(Event.RECOVER)
                self.assertEqual(machine.state, State.IDLE)

    def test_invalid_transition_is_rejected_without_mutation(self) -> None:
        machine = ConversationMachine()
        with self.assertRaises(ContractError) as caught:
            machine.dispatch(Event.ACTION_COMPLETE)
        self.assertEqual(caught.exception.code, "CONVERSATION_TRANSITION_INVALID")
        self.assertEqual(machine.state, State.IDLE)
        self.assertEqual(machine.transitions, [])

    def test_history_clear_is_idle_only(self) -> None:
        machine = ConversationMachine()
        machine.dispatch(Event.ARM)
        with self.assertRaises(ContractError) as caught:
            machine.reset_history()
        self.assertEqual(caught.exception.code, "CONVERSATION_ACTIVE")
        machine.dispatch(Event.CANCEL)
        machine.reset_history()
        self.assertEqual(machine.sequence, 0)
        self.assertEqual(machine.transitions, [])


if __name__ == "__main__":
    unittest.main()
