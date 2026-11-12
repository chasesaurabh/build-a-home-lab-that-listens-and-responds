"""Verify exported research bytes and independently recount headline results."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/research'
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
def main():
 manifest=load(E/'export-manifest.json');verified=0;asr=[];intent=[]
 for r in manifest['records']:
  path=(E/r['public_path']).resolve();path.relative_to(E.resolve())
  assert hashlib.sha256(path.read_bytes()).hexdigest()==r['public_sha256'],r['public_path'];verified+=1
 for run in sorted(E.glob('asr-*-evaluation002')):
  first=[r for r in rows(run/'predictions.jsonl') if r['repeat']==0];assert len(first)==192
  values={}
  summary=load(run/'summary.json')['subsets']
  for subset in ['test-clean','test-other']:
   sample=[r for r in first if r['subset']==subset];assert len(sample)==96
   errors=sum(sum(r['word_errors'][k] for k in ['substitutions','deletions','insertions']) for r in sample)
   denominator=sum(r['word_errors']['reference_words'] for r in sample)
   values[subset]={'word_errors':errors,'reference_words':denominator,'wer':errors/denominator}
   reported=summary[subset]
   assert reported['attempted']==reported['completed']==96 and reported['failed']==0
   assert denominator==reported['reference_words']
   assert errors==sum(reported[k] for k in ['substitutions','deletions','insertions'])
   assert abs(errors/denominator-reported['completed_cases_wer'])<1e-12
  asr.append({'run':run.name,'subsets':values})
 prompts=set()
 for run in sorted(E.glob('interpretation-*-evaluation*')):
  first=[r for r in rows(run/'predictions.jsonl') if r['repeat']==0];assert len(first)==64
  exact=over=0
  for r in first:
   result=r['result'];proposal=result.get('proposal');expected=r['case']['expected']
   exact+=bool(result.get('valid') and proposal and all(proposal.get(k)==expected[k] for k in ['disposition','intent','arguments']))
   over+=bool(result.get('valid') and proposal and proposal.get('disposition')=='accept' and expected['disposition']!='accept')
  summary=load(run/'summary.json');assert exact==summary['exact_match'] and over==summary['overaccepted'],run.name
  intent.append({'run':run.name,'exact':exact,'overaccepted':over})
  if not load(run/'config.json')['baseline']:
   expected_prompt=load(run/'config.json')['system_prompt']
   for r in rows(run/'predictions.jsonl'):
    prompt=r['result']['request']['messages'][0]['content'];assert prompt==expected_prompt;prompts.add(prompt)
 assert len(prompts)==1
 print(json.dumps({'verified_files':verified,'asr':asr,'interpretation':intent,'distinct_final_model_prompts':len(prompts)},indent=2))
if __name__=='__main__':main()
