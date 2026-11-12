"""Original stock-voice fixtures for whole-pipeline integration, not human ASR accuracy."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from home_lab_voice.runtimes import KokoroRuntime
from home_lab_voice.normalize import normalize_audio

CASES=[
 ('status','Get status for voice service.','accept','lab.status.read',{'service':'voice-service'}),
 ('diagnostic','Run healthcheck on voice service.','accept','lab.diagnostic.run',{'service':'voice-service','diagnostic':'healthcheck'}),
 ('negation','Do not run healthcheck on voice service.','abstain',None,{}),
 ('quotation','The manual says: run healthcheck on voice service.','abstain',None,{}),
 ('unsupported','Delete the media indexer database.','abstain',None,{}),
 ('missing_target','Run a health check.','clarify',None,{}),
]
def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);runtime=KokoroRuntime(args.model_dir);rows=[]
    for ident,text,disposition,intent,arguments in CASES:
        original=args.output/(ident+'-24k.wav');derived=args.output/(ident+'-16k.wav')
        result=runtime.synthesize(text,original);conversion=normalize_audio(original,derived)
        rows.append({'id':ident,'source_text':text,'audio':derived.name,'audio_sha256':conversion['output_sha256'],
                     'duration_seconds':conversion['duration_seconds'],'normalization':conversion,
                     'expected':{'disposition':disposition,'intent':intent,'arguments':arguments}})
        print(ident,'prepared',flush=True)
    (args.output/'manifest.json').write_text(json.dumps({'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Six original English synthetic utterances; whole-pipeline integration only, not population accuracy or speaker variety.',
        'model':'hexgrad/Kokoro-82M','revision':'f3ff3571791e39611d31c381e3a41a3af07b4987','voice':'af_heart',
        'fixtures_license':'MIT original texts; stock model voice, no person recording or cloning.','cases':rows},indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
