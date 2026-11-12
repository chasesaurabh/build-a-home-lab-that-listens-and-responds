"""Record offline ASR predictions, word-error counts and sampled resources.

One process per configuration. Keep every output directory immutable after a run.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from home_lab_voice.evaluation import word_error_counts

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def write(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

class Sampler:
    def __init__(self,output,interval=0.25):
        self.output=output; self.interval=interval
        self.stop_event=threading.Event(); self.samples=[]; self.errors=[]
    def start(self):
        self.t0=time.perf_counter()
        self.worker=threading.Thread(target=self.loop,daemon=True); self.worker.start()
    def loop(self):
        import psutil
        process=psutil.Process()
        while not self.stop_event.is_set():
            record={'elapsed_seconds':time.perf_counter()-self.t0,'process_rss_bytes':process.memory_info().rss}
            try:
                result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,check=True)
                record['devices']=[{'used_mib':int(p[0]),'total_mib':int(p[1]),'utilization_percent':int(p[2])}
                    for p in (line.split(',') for line in result.stdout.strip().splitlines())]
            except Exception as exc:
                record['gpu_sample_error']=type(exc).__name__
            self.samples.append(record)
            self.stop_event.wait(self.interval)
    def finish(self):
        self.stop_event.set(); self.worker.join(timeout=6)
        write(self.output,{'interval_target_seconds':self.interval,'note':'Total device sampled use, not exact process peak or minimum required capacity.',
                          'samples':self.samples})

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--model',choices=['tiny.en','base.en','small.en','large-v3'],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--compute-type',choices=['int8','float16','float32','int8_float16'],required=True)
    p.add_argument('--split',choices=['development','evaluation'],required=True)
    p.add_argument('--limit',type=int)
    p.add_argument('--warm-repeats',type=int,default=0)
    p.add_argument('--dll-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.limit is not None and args.limit < 1: p.error('limit must be positive')
    if args.warm_repeats<0: p.error('warm-repeats must be nonnegative')
    args.output.mkdir(parents=True,exist_ok=False)
    if args.dll_dir:
        if os.name!='nt': p.error('--dll-dir is only for Windows')
        # Explicit user-supplied vendor library directory; only this process changes.
        dll_handle=os.add_dll_directory(str(args.dll_dir.resolve()))
        os.environ['PATH']=str(args.dll_dir.resolve())+os.pathsep+os.environ.get('PATH','')
    from faster_whisper import WhisperModel
    corpus=json.loads((args.corpus/'manifest.json').read_text(encoding='utf-8'))
    acquisition=json.loads((args.models/'acquisition.json').read_text(encoding='utf-8'))
    model=next(m for m in acquisition['models'] if m['name']==args.model)
    directory=args.models/model['relative_directory']
    for record in model['files']:
        if sha(directory/record['path'])!=record['sha256']: raise ValueError('Model digest mismatch')
    selected=[r for r in corpus['records'] if r['split']==args.split]
    if args.limit: selected=selected[:args.limit]
    if not selected: raise ValueError('No selected utterances')
    for r in selected:
        if sha(args.corpus/r['path'])!=r['sha256']: raise ValueError('Corpus digest mismatch')
    settings={'language':'en','task':'transcribe','beam_size':5,'best_of':1,'temperature':0,
       'condition_on_previous_text':False,'vad_filter':False,'word_timestamps':False,'initial_prompt':None}
    config={'schema_version':1,'started_utc':datetime.now(timezone.utc).isoformat(),
       'model':model,'device':args.device,'compute_type':args.compute_type,'cpu_threads':4,'num_workers':1,
       'split':args.split,'limit':args.limit,'selected_ids':[r['id'] for r in selected],
       'corpus_manifest_sha256':sha(args.corpus/'manifest.json'),'scorer_sha256':sha(ROOT/'src/home_lab_voice/evaluation.py'),
       'script_sha256':sha(Path(__file__)),'settings':settings,'warm_repeats':args.warm_repeats,
       'environment':{'os':platform.platform(),'python':platform.python_version(),
         'packages':{k:importlib.metadata.version(k) for k in ['faster-whisper','ctranslate2','numpy','av','psutil']}},
       'limits':'Read English audiobook subset. No household, accent-wide or training-decontamination claim. Timing subset repetitions are not independent quality cases.'}
    write(args.output/'config.json',config)
    sampler=Sampler(args.output/'resources.json'); sampler.start()
    predictions=[]
    try:
        t=time.perf_counter()
        runtime=WhisperModel(str(directory),device=args.device,compute_type=args.compute_type,cpu_threads=4,num_workers=1,local_files_only=True)
        write(args.output/'load.json',{'seconds':time.perf_counter()-t,'cached_model_files':True})
        cases=[(r,0) for r in selected]
        # Stable subset across models; choose at most 12 clips from each test slice.
        timing=[]
        for subset in sorted({r['subset'] for r in selected}):
            timing.extend([r for r in selected if r['subset']==subset][:12])
        cases.extend((r,repeat) for repeat in range(1,args.warm_repeats+1) for r in timing)
        with (args.output/'predictions.jsonl').open('w',encoding='utf-8') as out:
            for index,(r,repeat) in enumerate(cases):
                t=time.perf_counter()
                record={'id':r['id'],'subset':r['subset'],'reference':r['reference'],
                   'audio_sha256':r['sha256'],'duration_seconds':r['duration_seconds'],
                   'repeat':repeat,'first_request_in_process':index==0}
                try:
                    segments,info=runtime.transcribe(str(args.corpus/r['path']),**settings)
                    segments=list(segments) # Generator iteration performs inference.
                    elapsed=time.perf_counter()-t
                    text=' '.join(s.text.strip() for s in segments).strip()
                    record.update({'status':'ok','hypothesis':text,'wall_seconds':elapsed,
                       'rtf':elapsed/r['duration_seconds'],'segments':[asdict(s) for s in segments],
                       'word_errors':asdict(word_error_counts(r['reference'],text)),
                       'language':info.language,'language_probability':info.language_probability})
                except Exception as exc:
                    record.update({'status':'error','error_type':type(exc).__name__,'error':str(exc),
                                   'wall_seconds':time.perf_counter()-t})
                out.write(json.dumps(record,ensure_ascii=False)+'\n'); out.flush()
                predictions.append(record)
                if index%12==0: print(f'{args.model} {args.device} {index+1}/{len(cases)} {record["status"]}',flush=True)
        summary={}
        for subset in sorted({r['subset'] for r in selected}):
            initial=[r for r in predictions if r['repeat']==0 and r['subset']==subset]
            successful=[r for r in initial if r['status']=='ok']
            counts={k:sum(r['word_errors'][k] for r in successful) for k in ['substitutions','deletions','insertions','reference_words']}
            errors=sum(counts[k] for k in ['substitutions','deletions','insertions'])
            summary[subset]={'attempted':len(initial),'completed':len(successful),'failed':len(initial)-len(successful),
                **counts,'completed_cases_wer':errors/counts['reference_words'] if counts['reference_words'] else None}
        write(args.output/'summary.json',{'subsets':summary,'completed_utc':datetime.now(timezone.utc).isoformat()})
    except Exception as exc:
        write(args.output/'failure.json',{'type':type(exc).__name__,'error':str(exc)})
        raise
    finally:
        sampler.finish()

if __name__=='__main__': main()
