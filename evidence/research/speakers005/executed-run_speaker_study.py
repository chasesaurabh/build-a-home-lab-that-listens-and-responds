"""Score known-count grouping on stitched read-speech fixtures, not meeting DER."""
import argparse,hashlib,itertools,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from home_lab_voice.speakers import SpeakerGrouper
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def score(reference,turns):
    truth=sorted({r['speaker'] for r in reference});pred=sorted({r['speaker'] for r in turns if r['speaker'] is not None})
    overlap={(p,t):0. for p in pred for t in truth};unknown=0.;outside=0.;labelled=0.
    for turn in turns:
        covered=0.
        for region in reference:
            duration=max(0,min(turn['end'],region['end'])-max(turn['start'],region['start']));covered+=duration
            if turn['speaker'] is None:unknown+=duration
            else:overlap[turn['speaker'],region['speaker']]+=duration;labelled+=duration
        outside+=turn['end']-turn['start']-covered
    best=0.;mapping={}
    if pred:
        n=min(len(pred),len(truth))
        for chosen in itertools.combinations(pred,n):
            for assigned in itertools.permutations(truth,n):
                candidate=dict(zip(chosen,assigned));correct=sum(overlap[p,t] for p,t in candidate.items())
                if correct>best:best=correct;mapping=candidate
    clip_seconds=sum(r['end']-r['start'] for r in reference)
    return {'labelled_clip_seconds':labelled,'wrong_group_seconds_after_best_mapping':labelled-best,
        'group_error_fraction':(labelled-best)/labelled if labelled else None,'unknown_clip_seconds':unknown,
        'detected_seconds_in_inserted_gaps':outside,'labelled_fraction_of_whole_source_clips':labelled/clip_seconds,
        'best_anonymous_mapping':mapping,
        'scope':'Duration-weighted group errors within source clips. Clip spans include internal pauses. This is not diarization error rate, VAD recall, or meeting accuracy.'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--vad-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);fixture=json.loads((a.fixtures/'manifest.json').read_text())
    write(a.output/'config.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
        'fixtures_sha256':sha(a.fixtures/'manifest.json'),'script_sha256':sha(Path(__file__)),
        'adapter_sha256':sha(ROOT/'src/home_lab_voice/speakers.py'),'vad_sha256':sha(a.vad_model),
        'primary_count':4,'sensitivity_counts':[2,6],'policy':'Fixed before inference; no threshold tuning.'})
    model=SpeakerGrouper(a.weights,a.vad_model);rows=[]
    for case in fixture['cases']:
        audio=a.fixtures/case['audio']
        if sha(audio)!=case['sha256']:raise ValueError('Fixture mismatch')
        for count in [4,2,6]:
            started=time.perf_counter();result=model.group(audio,count)
            row={'case':case['id'],'split':case['split'],'requested_count':count,'result':result,
                'wall_seconds':time.perf_counter()-started,'score':score(case['segments'],result['turns'])}
            rows.append(row);write(a.output/'predictions.json',rows)
            print(case['id'],count,json.dumps(row['score']),flush=True)
    write(a.output/'summary.json',[{'case':r['case'],'count':r['requested_count'],**r['score']} for r in rows])
if __name__=='__main__':main()
