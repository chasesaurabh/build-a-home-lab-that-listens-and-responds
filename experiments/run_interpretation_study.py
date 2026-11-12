"""Compare command proposals against frozen text cases, without invoking tools."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from home_lab_voice.interpretation import interpret, SYSTEM_PROMPT
from home_lab_voice.applications.workflows import interpret_assistant_command
from home_lab_voice.contracts import ContractError
from run_asr_study import Sampler

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def grade(expected,result):
    proposal=result.get('proposal')
    valid=result.get('valid',False)
    match=bool(valid and proposal and all(proposal[k]==expected[k] for k in ('disposition','intent','arguments')))
    accepted=bool(valid and proposal and proposal['disposition']=='accept')
    # Overacceptance counts executable-schema proposals for cases that require no operation.
    return {'exact_match':match,'accepted':accepted,'overaccepted':accepted and expected['disposition']!='accept',
            'wrong_accepted_operation':accepted and expected['disposition']=='accept' and not match}

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--split',choices=['development','evaluation'],required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:18105/v1');p.add_argument('--model',required=True)
    p.add_argument('--baseline',action='store_true');p.add_argument('--warm-repeats',type=int,default=3)
    p.add_argument('--unconstrained',action='store_true')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not 0<=args.warm_repeats<=10:p.error('warm repeats must be 0 to 10')
    corpus=json.loads(args.cases.read_text(encoding='utf-8'))
    selected=[c for c in corpus['cases'] if c['split']==args.split]
    if not selected:raise ValueError('No selected cases')
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'config.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'model':args.model,
        'baseline':args.baseline,'structured':not args.unconstrained,'split':args.split,'cases_sha256':digest(args.cases),
        'script_sha256':digest(Path(__file__)),'adapter_sha256':digest(ROOT/'src/home_lab_voice/interpretation.py'),
        'system_prompt':SYSTEM_PROMPT,'warm_repeats':args.warm_repeats,
        'limits':corpus['scope'],'selection':corpus['selection']})
    timing=[]
    for disposition in ('accept','clarify','abstain'):
        timing.extend([c for c in selected if c['expected']['disposition']==disposition][:4])
    work=[(c,0) for c in selected]+[(c,r) for r in range(1,args.warm_repeats+1) for c in timing]
    rows=[];sampler=Sampler(args.output/'resources.json');sampler.start()
    try:
        with (args.output/'predictions.jsonl').open('w',encoding='utf-8') as out:
            for index,(case,repeat) in enumerate(work):
                row={'case':case,'repeat':repeat,'first_request_in_process':index==0}
                t=time.perf_counter()
                try:
                    if args.baseline:
                        try:
                            intent,arguments=interpret_assistant_command(case['transcript'])
                            proposal={'disposition':'accept','intent':'lab.status.read' if intent=='status' else 'lab.diagnostic.run',
                                      'arguments':arguments,'evidence_quotes':[case['transcript']]}
                        except ContractError:
                            proposal={'disposition':'abstain','intent':None,'arguments':{},'evidence_quotes':[]}
                        result={'valid':True,'proposal':proposal,'wall_seconds':time.perf_counter()-t}
                    else:result=interpret(case['transcript'],endpoint=args.endpoint,model=args.model,structured=not args.unconstrained)
                    row.update({'status':'ok','result':result,'grade':grade(case['expected'],result)})
                except Exception as exc:
                    row.update({'status':'error','error_type':type(exc).__name__,'error':str(exc),
                                'wall_seconds':time.perf_counter()-t,'grade':grade(case['expected'],{})})
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();rows.append(row)
                if index%8==0:print(f'{args.model} {index+1}/{len(work)} {row["status"]}',flush=True)
    finally:sampler.finish()
    first=[r for r in rows if r['repeat']==0]
    summary={'completed_utc':datetime.now(timezone.utc).isoformat(),'attempted':len(first),
             'transport_errors':sum(r['status']=='error' for r in first),
             'valid_proposals':sum(r.get('result',{}).get('valid',False) for r in first),
             **{k:sum(r['grade'][k] for r in first) for k in ('exact_match','accepted','overaccepted','wrong_accepted_operation')}}
    summary['by_category']={cat:{'n':sum(r['case']['category']==cat for r in first),
        'exact_match':sum(r['grade']['exact_match'] for r in first if r['case']['category']==cat)}
        for cat in sorted({r['case']['category'] for r in first})}
    write(args.output/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
