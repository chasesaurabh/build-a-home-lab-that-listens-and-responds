"""Generate fixed stock-voice examples and report objective output observations."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from home_lab_voice.runtimes import KokoroRuntime
from audit_speech_file import audit
def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cases=[('status-a','The local queue contains three jobs.'),('status-b','The local queue contains three jobs.'),
           ('confirmation','Review the diagnostic for voice service in the terminal before confirming.'),
           ('technical','Use A S R with an L L M. The host has sixteen gigabytes.')]
    config={'cases':[{'id':i,'text':t} for i,t in cases],'voice':'af_heart','language_code':'a','speed':1.0,
        'scope':'Four authored synthesis files, including one repeated sentence. Objective file checks, no subjective rating.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    model=KokoroRuntime(a.model_dir);rows=[]
    for ident,text in cases:
        start=time.perf_counter();model.synthesize(text,a.output/(ident+'.wav'));elapsed=time.perf_counter()-start
        rows.append({'id':ident,'text':text,'wall_seconds':elapsed,'audit':audit(a.output/(ident+'.wav'))})
        (a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(ident,rows[-1]['audit']['duration_seconds'],flush=True)
    print(json.dumps({'files':len(rows),'repeated_sentence_byte_equal':rows[0]['audit']['sha256']==rows[1]['audit']['sha256']}))
if __name__=='__main__':main()
