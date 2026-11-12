from datetime import timedelta
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from home_lab_voice.applications.assistant import ControlledAssistant
from home_lab_voice.contracts import ContractError
from test_assistant_application import intent, session, NOW

class ConfirmationRegressions(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        def diagnostic(service,kind):
            self.calls.append((service,kind))
            return {'state':'completed','attempt':len(self.calls)}
        self.app=ControlledAssistant(lambda s:{'state':'ready'},diagnostic)

    def prepare(self,key='idem_regression01',service='voice-service'):
        return self.app.prepare_diagnostic(intent('lab.diagnostic.run',
            {'service':service,'diagnostic':'healthcheck'}),session(),NOW,key)

    def test_role_revoked_after_prepare_prevents_provider(self):
        pending=self.prepare()
        with self.assertRaises(ContractError):
            self.app.confirm_and_run(pending,session('viewer'),NOW,pending.operation_digest)
        self.assertEqual(self.calls,[])

    def test_changed_pending_arguments_cannot_reuse_approval(self):
        pending=self.prepare()
        pending.arguments['service']='media-indexer'
        with self.assertRaises(ContractError):
            self.app.confirm_and_run(pending,session(),NOW,pending.operation_digest)
        self.assertEqual(self.calls,[])

    def test_new_pending_same_idempotency_key_calls_provider_once(self):
        first=self.prepare()
        one=self.app.confirm_and_run(first,session(),NOW,first.operation_digest)
        second=self.prepare()
        two=self.app.confirm_and_run(second,session(),NOW,second.operation_digest)
        self.assertEqual(one.payload,two.payload)
        self.assertEqual(len(self.calls),1)

    def test_conflicting_key_is_rejected_before_second_effect(self):
        first=self.prepare()
        self.app.confirm_and_run(first,session(),NOW,first.operation_digest)
        with self.assertRaises(ContractError):
            second=self.prepare(service='media-indexer')
            self.app.confirm_and_run(second,session(),NOW,second.operation_digest)
        self.assertEqual(len(self.calls),1)

    def test_invalid_key_rejected_before_effect(self):
        with self.assertRaises(ContractError):
            pending=self.prepare(key='bad')
            self.app.confirm_and_run(pending,session(),NOW,pending.operation_digest)
        self.assertEqual(self.calls,[])

    def test_expiration_tampering_rejected(self):
        pending=self.prepare()
        pending.expires_at=NOW+timedelta(days=5)
        with self.assertRaises(ContractError):
            self.app.confirm_and_run(pending,session(),NOW,pending.operation_digest)
        self.assertEqual(self.calls,[])

if __name__=='__main__': unittest.main()
