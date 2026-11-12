from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from home_lab_voice.lab_control import LocalLab
from home_lab_voice.contracts import ContractError
from home_lab_voice.cli import main

class DurableLabTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)/'lab';self.lab=LocalLab(self.root);self.lab.initialize()
    def prepare(self,key='diagnostic-00001'):
        return self.lab.prepare('voice-service','queue_snapshot',key,now=100)
    def test_real_queue_changes_report_and_restart_reuses_original_result(self):
        self.lab.enqueue('voice-service','job-original-0001');pending=self.prepare()
        result=LocalLab(self.root).confirm(pending['id'],pending['digest'],now=101)
        self.assertEqual(result['result']['queue_counts'],{'queued':1})
        self.lab.enqueue('voice-service','job-original-0002')
        repeated=LocalLab(self.root).confirm(pending['id'],pending['digest'],now=1000)
        self.assertTrue(repeated['replayed']);self.assertEqual(repeated['result'],result['result'])
        self.assertEqual(self.lab.status('voice-service')['queue_counts'],{'queued':2})
    def test_expired_request_stays_expired_after_prepare_again(self):
        pending=self.prepare();again=self.lab.prepare('voice-service','queue_snapshot','diagnostic-00001',now=500)
        self.assertEqual(again['expires'],pending['expires'])
        with self.assertRaises(ContractError):LocalLab(self.root).confirm(again['id'],again['digest'],now=500)
    def test_revoked_request_cannot_run(self):
        pending=self.prepare();self.assertTrue(self.lab.revoke(pending['id'])['revoked'])
        with self.assertRaises(ContractError):self.lab.confirm(pending['id'],pending['digest'],now=101)
    def test_digest_and_workspace_binding(self):
        pending=self.prepare();other=LocalLab(Path(self.temporary.name)/'other');other.initialize()
        with self.assertRaises(ContractError):other.confirm(pending['id'],pending['digest'],now=101)
        with self.assertRaises(ContractError):self.lab.confirm(pending['id'],'0'*64,now=101)
    def test_conflicting_idempotency_key_rejected(self):
        self.prepare()
        with self.assertRaises(ContractError):self.lab.prepare('media-indexer','queue_snapshot','diagnostic-00001',now=101)
    def test_concurrent_confirmation_commits_one_result(self):
        pending=self.prepare()
        def confirm(_):return LocalLab(self.root).confirm(pending['id'],pending['digest'],now=101)
        with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(confirm,range(4)))
        self.assertEqual(sum(not r['replayed'] for r in results),1)
        self.assertEqual(len({json.dumps(r['result']) for r in results}),1)
    def test_existing_workspace_and_unknown_targets_rejected(self):
        with self.assertRaises(FileExistsError):self.lab.initialize()
        with self.assertRaises(ContractError):self.lab.status('../../outside')
        with self.assertRaises(ContractError):self.lab.prepare('voice-service','restart','diagnostic-00001')
    def test_cli_across_distinct_python_processes(self):
        repo=Path(__file__).resolve().parents[1]
        runner="import sys;sys.path.insert(0,'src');from home_lab_voice.cli import main;raise SystemExit(main())"
        def cli(*args):
            result=subprocess.run([sys.executable,'-c',runner,'lab','--root',str(self.root),*args],cwd=repo,capture_output=True,text=True,check=True)
            return json.loads(result.stdout)
        cli('enqueue','--service','media-indexer','--job-id','job-subprocess-001')
        pending=cli('prepare','--service','media-indexer','--diagnostic','healthcheck','--idempotency-key','diagnostic-subprocess-001')
        one=cli('confirm','--request-id',pending['id'],'--digest',pending['digest'])
        two=cli('confirm','--request-id',pending['id'],'--digest',pending['digest'])
        self.assertEqual(one['result'],{'database_check':'ok','jobs':1});self.assertTrue(two['replayed'])

    def turn(self,text,extra=()):
        output=io.StringIO();errors=io.StringIO()
        code=main(['lab-turn','--root',str(self.root),'--text',text,'--idempotency-key','turn-request-0001',*extra],stdout=output,stderr=errors)
        return code,json.loads(output.getvalue() or errors.getvalue())
    def test_typed_turn_prepares_but_never_confirms_diagnostic(self):
        code,result=self.turn('Run healthcheck on voice-service')
        self.assertEqual(code,3);self.assertEqual(result['action']['state'],'pending')
        with self.lab._connection() as db:self.assertEqual(db.execute("SELECT count(*) FROM requests WHERE state='done'").fetchone()[0],0)
    def test_negation_has_no_pending_operation(self):
        code,result=self.turn('Do not run healthcheck on voice-service')
        self.assertEqual(result['effect'],'none')
        with self.lab._connection() as db:self.assertEqual(db.execute('SELECT count(*) FROM requests').fetchone()[0],0)
    def test_invalid_model_result_never_reaches_local_lab(self):
        with patch('home_lab_voice.cli.interpret',return_value={'valid':False,'validation_error':'MODEL_ARGUMENTS_INVALID'}):
            code,result=self.turn('some command',['--model','fixture'])
        self.assertEqual(code,2);self.assertEqual(result['effect'],'none')
        with self.lab._connection() as db:self.assertEqual(db.execute('SELECT count(*) FROM requests').fetchone()[0],0)
    def test_speech_failure_keeps_reviewable_pending_record(self):
        code,result=self.turn('Run healthcheck on voice-service',['--speak-output',str(self.root/'response.wav')])
        self.assertEqual(code,3);self.assertEqual(result['speech_error'],'TTS_MODEL_REQUIRED')
        self.assertEqual(result['action']['state'],'pending')

    def test_unexpected_speech_failure_preserves_authoritative_result(self):
        with patch('home_lab_voice.cli.KokoroRuntime',side_effect=RuntimeError('simulated backend fault')):
            code,result=self.turn('Run healthcheck on voice-service',[
                '--speak-output',str(self.root/'response.wav'),'--tts-model-dir',str(self.root/'model')])
        self.assertEqual(code,3);self.assertEqual(result['speech_error'],'TTS_FAILED')
        pending=result['action']
        self.assertEqual(self.lab.confirm(pending['id'],pending['digest'])['result']['database_check'],'ok')

if __name__=='__main__':unittest.main()
