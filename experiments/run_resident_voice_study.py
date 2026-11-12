"""Keep actual GPU ASR and local LLM resident while completing speech turns.

An optional separately installed CPU speech interpreter generates response audio
while both GPU models remain loaded. No model-proposed tools are executed.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from home_lab_voice.cuda_libraries import register_nvidia_wheels,verify_loaded_nvidia_wheels
from home_lab_voice.interpretation import interpret
from run_asr_study import Sampler
from run_interpretation_study import grade

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:18105/v1');p.add_argument('--model',required=True)
    p.add_argument('--tts-python',type=Path);p.add_argument('--tts-model-dir',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if bool(args.tts_python)!=bool(args.tts_model_dir):p.error('Supply both TTS interpreter and model directory')
    args.output.mkdir(parents=True,exist_ok=False)
    fixture=json.loads((args.fixtures/'manifest.json').read_text());acquisition=json.loads((args.models/'acquisition.json').read_text())
    model=next(m for m in acquisition['models'] if m['name']=='large-v3');directory=args.models/model['relative_directory']
    for record in model['files']:
        if sha(directory/record['path'])!=record['sha256']:raise ValueError('ASR model hash mismatch')
    for case in fixture['cases']:
        if sha(args.fixtures/case['audio'])!=case['audio_sha256']:raise ValueError('Fixture hash mismatch')
    handles,versions=register_nvidia_wheels()
    from faster_whisper import WhisperModel
    write(args.output/'config.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'asr_model':model,'llm_alias':args.model,
        'fixture_manifest_sha256':sha(args.fixtures/'manifest.json'),'script_sha256':sha(Path(__file__)),
        'adapter_sha256':sha(ROOT/'src/home_lab_voice/interpretation.py'),'nvidia_wheels':versions,
        'scope':'Co-residency integration. Sequential turns with both GPU models resident; no throughput/concurrency scaling claim. No tools executed.'})
    sampler=Sampler(args.output/'resources.json');sampler.start();rows=[]
    try:
        t=time.perf_counter();asr=WhisperModel(str(directory),device='cuda',compute_type='float16',cpu_threads=4,num_workers=1,local_files_only=True)
        write(args.output/'load.json',{'asr_cached_load_seconds':time.perf_counter()-t})
        for case in fixture['cases']:
            t=time.perf_counter();segments,_=asr.transcribe(str(args.fixtures/case['audio']),language='en',beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=False)
            text=' '.join(s.text.strip() for s in segments).strip();asr_seconds=time.perf_counter()-t
            result=interpret(text,endpoint=args.endpoint,model=args.model)
            row={'id':case['id'],'source_text':case['source_text'],'transcript':text,'asr_seconds':asr_seconds,
                 'expected':case['expected'],'interpretation':result,'grade':grade(case['expected'],result)}
            if args.tts_python:
                response='Review the proposed operation in the terminal.' if (result.get('proposal') or {}).get('disposition')=='accept' else 'No operation was performed.'
                command=[str(args.tts_python),'-m','home_lab_voice','synthesize',str(args.output/(case['id']+'-response.wav')),
                         '--model-dir',str(args.tts_model_dir),'--text',response]
                started=time.perf_counter();spoken=subprocess.run(command,capture_output=True,text=True,timeout=120)
                (args.output/(case['id']+'-tts-stderr.txt')).write_text(spoken.stderr,encoding='utf-8')
                row['tts']={'exit_code':spoken.returncode,'process_wall_seconds':time.perf_counter()-started,
                    'output_sha256':sha(args.output/(case['id']+'-response.wav')) if spoken.returncode==0 else None}
                if spoken.returncode:row['tts']['error']='Inspect retained stderr; text interpretation remains authoritative.'
            rows.append(row);write(args.output/'predictions.json',rows)
            print(case['id'],row['grade'],flush=True)
        write(args.output/'loaded-cuda-libraries.json',verify_loaded_nvidia_wheels())
        write(args.output/'summary.json',{'attempted':len(rows),'exact_match':sum(r['grade']['exact_match'] for r in rows),
            'overaccepted':sum(r['grade']['overaccepted'] for r in rows),'tts_completed':sum(r.get('tts',{}).get('exit_code')==0 for r in rows),
            'completed_utc':datetime.now(timezone.utc).isoformat()})
    except Exception as exc:write(args.output/'failure.json',{'type':type(exc).__name__,'error':str(exc)});raise
    finally:sampler.finish()
if __name__=='__main__':main()
