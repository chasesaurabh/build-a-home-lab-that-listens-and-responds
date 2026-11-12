"""Exercise actual CLI dependency failure, preserved approval, revocation and replay."""
import argparse,json,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);workspace=a.output/'workspace';rows=[]
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex(('127.0.0.1',18105))==0:raise ValueError('Stop your study model container before this dependency-failure exercise.')
    def cli(name,args,expected):
        runner="import sys;sys.path.insert(0,'src');from home_lab_voice.cli import main;raise SystemExit(main())"
        started=time.perf_counter();result=subprocess.run([sys.executable,'-c',runner,*map(str,args)],cwd=ROOT,capture_output=True,text=True,timeout=120)
        value=json.loads(result.stdout or result.stderr)
        # Store structured fields only. Diagnostic messages can contain local paths.
        safe={k:v for k,v in value.items() if k!='message'}
        row={'step':name,'exit_code':result.returncode,'expected_exit':expected,'wall_seconds':time.perf_counter()-started,'result':safe}
        rows.append(row);(a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        if result.returncode!=expected:raise AssertionError(name+' returned unexpected exit code')
        return value
    cli('initialize',['lab','--root',workspace,'init'],0)
    base=['lab-turn','--root',workspace,'--idempotency-key','recovery-command-001']
    cli('model_unavailable',[*base,'--text','Get status for voice-service','--model','voice-2b'],2)
    status=cli('typed_fallback',[*base,'--text','Get status for voice-service'],0)
    assert status['status']=='completed'
    pending=cli('speech_model_unavailable',[*base,'--text','Run healthcheck on voice-service',
        '--tts-model-dir',a.output/'missing-model','--speak-output',a.output/'response.wav'],3)
    assert pending.get('speech_error') and pending['action']['state']=='pending' and not (a.output/'response.wav').exists()
    action=pending['action']
    cli('wrong_approval',['lab','--root',workspace,'confirm','--request-id',action['id'],'--digest','0'*64],2)
    revoked=cli('revoke',['lab','--root',workspace,'revoke','--request-id',action['id']],0)
    assert revoked['revoked']
    cli('revoked_approval',['lab','--root',workspace,'confirm','--request-id',action['id'],'--digest',action['digest']],2)
    second=cli('new_review',['lab','--root',workspace,'prepare','--service','voice-service','--diagnostic','healthcheck','--idempotency-key','recovery-command-002'],0)
    one=cli('confirmed_once',['lab','--root',workspace,'confirm','--request-id',second['id'],'--digest',second['digest']],0)
    two=cli('replayed_after_process_restart',['lab','--root',workspace,'confirm','--request-id',second['id'],'--digest',second['digest']],0)
    assert not one['replayed'] and two['replayed'] and one['result']==two['result'] and one['observed_at']==two['observed_at']
    cli('conflicting_key',['lab','--root',workspace,'prepare','--service','media-indexer','--diagnostic','healthcheck','--idempotency-key','recovery-command-002'],2)
    print(json.dumps({'checks':len(rows),'passed':True,'scope':'Actual CLI processes and local SQLite effects. No microphone, playback or external-service changes.'}))
if __name__=='__main__':main()
