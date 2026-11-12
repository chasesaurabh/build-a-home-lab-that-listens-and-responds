"""Fixed-threshold YAMNet study: read speech and authored stress signals."""
import argparse,hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from home_lab_voice.events import EventTagger
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--librispeech',type=Path,required=True)
    p.add_argument('--nonspeech',type=Path,required=True);p.add_argument('--acquisition',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    speech=json.loads((a.librispeech/'manifest.json').read_text());stress=json.loads((a.nonspeech/'manifest.json').read_text())
    cases=[]
    for r in speech['records']:
        cases.append({'id':r['id'],'split':r['split'],'source':'librispeech','file':r['path'],'sha256':r['sha256'],
                      'expected_speech':True,'kind':r['subset']})
    for r in stress['cases']:
        cases.append({'id':r['id'],'split':r['split'],'source':'synthetic','file':r['file'],'sha256':r['sha256'],
                      'expected_speech':False,'kind':r['kind']})
    write(a.output/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'cases':cases,
        'thresholds':{'accept':.5,'review':.2},'aggregation':'whole-clip mean','tuning':'None; fixed before inference.',
        'script_sha256':sha(Path(__file__)),'adapter_sha256':sha(ROOT/'src/home_lab_voice/events.py'),
        'scope':'Read English speech vs original synthetic signals. No household or life-safety performance claim.'})
    tagger=EventTagger(a.acquisition);rows=[]
    with (a.output/'predictions.jsonl').open('x',encoding='utf-8') as output:
        for i,case in enumerate(cases):
            path=(a.librispeech if case['source']=='librispeech' else a.nonspeech)/case['file']
            if sha(path)!=case['sha256']:raise ValueError('Fixture identity mismatch')
            started=time.perf_counter();result=tagger.annotate(path)
            row={'case':case,'result':result,'wall_seconds':time.perf_counter()-started}
            rows.append(row);output.write(json.dumps(row)+'\n');output.flush()
            if i%32==0:print(i+1,len(cases),flush=True)
    summary={}
    for split in ('development','evaluation'):
        subset=[r for r in rows if r['case']['split']==split]
        counts={'n':len(subset),'true_positive':0,'false_positive':0,'false_negative':0,'true_negative':0,'speech_review':0}
        for row in subset:
            prediction=row['result']['annotations'][0]['disposition'];positive=row['case']['expected_speech']
            counts['speech_review']+=prediction=='review'
            counts[('true_positive' if positive else 'false_positive') if prediction=='accept' else ('false_negative' if positive else 'true_negative')]+=1
        summary[split]=counts
    write(a.output/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
