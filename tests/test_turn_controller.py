import unittest
from home_lab_voice.conversation import TurnController,ConversationEvent as Event,ConversationState as State
from home_lab_voice.contracts import ContractError

class TurnControllerTests(unittest.TestCase):
    def test_old_result_rejected_even_when_new_turn_is_in_matching_state(self):
        c=TurnController();first=c.begin();c.dispatch(first,Event.START_LISTENING);c.dispatch(first,Event.CANCEL)
        second=c.begin();c.dispatch(second,Event.START_LISTENING)
        with self.assertRaises(ContractError) as error:c.dispatch(first,Event.AUDIO_READY)
        self.assertEqual(error.exception.code,'TURN_STALE')
        self.assertEqual(c.machine.state,State.LISTENING)
        c.dispatch(second,Event.AUDIO_READY)
        self.assertEqual(c.machine.state,State.TRANSCRIBING)
    def test_timeout_requires_recovery_and_cannot_start_another_turn(self):
        c=TurnController();turn=c.begin();c.dispatch(turn,Event.TIMEOUT)
        with self.assertRaises(ContractError):c.begin()
        c.dispatch(turn,Event.RECOVER);self.assertIsNone(c.active_turn)
        self.assertNotEqual(turn,c.begin())
    def test_completed_or_cancelled_turn_cannot_be_reused(self):
        c=TurnController();turn=c.begin();c.dispatch(turn,Event.CANCEL)
        for event in [Event.CONFIRMED,Event.ACTION_COMPLETE,Event.START_LISTENING]:
            with self.assertRaises(ContractError):c.dispatch(turn,event)
    def test_invalid_event_does_not_mutate_state(self):
        c=TurnController();turn=c.begin()
        with self.assertRaises(ContractError):c.dispatch(turn,'cancel')
        self.assertEqual(c.machine.state,State.ARMED)
