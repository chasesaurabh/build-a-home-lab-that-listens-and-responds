"""Recompute study aggregates from raw predictions, never from hand-entered tables."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from home_lab_voice.evaluation import word_error_counts
from dataclasses import asdict

def percentile(values,q):
    ordered=sorted(values);return ordered[max(0,math.ceil(q*len(ordered))-1)] if ordered else None

def summarize(run):
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (run/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    first=[r for r in rows if r['repeat']==0];repeated=[r for r in rows if r['repeat']>0 and r['status']=='ok']
    if len(first)!=len(config['selected_ids']) or {r['id'] for r in first}!=set(config['selected_ids']):raise ValueError('Case coverage mismatch')
    if len(first)!=len({r['id'] for r in first}):raise ValueError('Duplicate evaluation case')
    for row in rows:
        if row['status']=='ok' and asdict(word_error_counts(row['reference'],row['hypothesis']))!=row['word_errors']:
            raise ValueError('Raw text scoring mismatch')
    subsets={}
    for subset in sorted({r['subset'] for r in first}):
        selected=[r for r in first if r['subset']==subset];ok=[r for r in selected if r['status']=='ok']
        counts={key:sum(r['word_errors'][key] for r in ok) for key in ('substitutions','deletions','insertions','reference_words')}
        errors=counts['substitutions']+counts['deletions']+counts['insertions']
        subsets[subset]={'n':len(selected),'failed':len(selected)-len(ok),**counts,'wer':errors/counts['reference_words'] if counts['reference_words'] else None,
                         'audio_seconds':sum(r['duration_seconds'] for r in selected)}
    resources=json.loads((run/'resources.json').read_text(encoding='utf-8'))['samples']
    return {'run':run.name,'model':config['model']['name'],'device':config['device'],'compute_type':config['compute_type'],
       'corpus_manifest_sha256':config['corpus_manifest_sha256'],'subsets':subsets,
       'cached_model_load_seconds':json.loads((run/'load.json').read_text())['seconds'],
       'first_request_seconds':rows[0]['wall_seconds'],'warm_request_n':len(repeated),
       'warm_wall_median_seconds':statistics.median(r['wall_seconds'] for r in repeated) if repeated else None,
       'warm_wall_p95_seconds':percentile([r['wall_seconds'] for r in repeated],.95),
       'warm_rtf_median':statistics.median(r['rtf'] for r in repeated) if repeated else None,
       'warm_rtf_p95':percentile([r['rtf'] for r in repeated],.95),
       'sampled_peak_device_mib':max((d['used_mib'] for r in resources for d in r.get('devices',[])),default=None),
       'sampled_peak_process_rss_bytes':max((r['process_rss_bytes'] for r in resources),default=None),
       'raw_predictions_sha256':hashlib.sha256((run/'predictions.jsonl').read_bytes()).hexdigest()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result={'normalization':'NFKC, casefold, punctuation to spaces, whitespace tokenization; no digit expansion.',
            'latency_scope':'72 warm requests from 24 fixed clips, three repeats each. P95 nearest rank. Repeats are not independent accuracy cases.',
            'memory_scope':'Sampled total-device allocation and process RSS, not exact peaks or minimum GPU capacity.',
            'results':[summarize(run) for run in args.runs]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
