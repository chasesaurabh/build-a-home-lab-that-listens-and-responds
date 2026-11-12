"""Start an explicitly named, loopback-only text server and retain startup evidence."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time
from urllib.request import build_opener,ProxyHandler

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--alias',required=True);p.add_argument('--gpu-fraction',type=float,required=True)
    p.add_argument('--image',default='vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90')
    p.add_argument('--quantization',choices=['modelopt_fp4']);p.add_argument('--kv-cache-dtype',choices=['auto','fp8'],default='auto')
    p.add_argument('--startup-timeout',type=int,default=1800);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not .01<=args.gpu_fraction<=.95:p.error('GPU fraction must reserve some device headroom')
    if subprocess.run(['docker','container','inspect',args.name],capture_output=True).returncode==0:raise ValueError('Named container exists; inspect before retrying')
    model=args.model_path.resolve(strict=True);args.output.mkdir(parents=True,exist_ok=False)
    command=['docker','run','-d','--name',args.name,'--gpus','all','-p','127.0.0.1:18105:8000',
        '--mount',f'type=bind,source={model},target=/model,readonly','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
        args.image,'/model','--served-model-name',args.alias,'--max-model-len','4096','--max-num-seqs','1',
        '--gpu-memory-utilization',str(args.gpu_fraction),'--enforce-eager','--limit-mm-per-prompt','{"image":0,"video":0}',
        '--kv-cache-dtype',args.kv_cache_dtype]
    if args.quantization:command+=['--quantization',args.quantization]
    record={'started_utc':datetime.now(timezone.utc).isoformat(),'command':command,
            'image_id':subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip(),
            'note':'Text-only interpretation. Loopback listener; offline model lookup. Memory allocation is not minimum hardware.'}
    def save():
        (args.output/'startup.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        logs=subprocess.run(['docker','logs',args.name],capture_output=True)
        (args.output/'server.log').write_bytes(logs.stdout+b'\n'+logs.stderr)
    save();t=time.monotonic()
    try:
        record['container_id']=subprocess.check_output(command,text=True).strip()
        while time.monotonic()-t<args.startup_timeout:
            state=subprocess.check_output(['docker','inspect',args.name,'--format','{{.State.Status}}'],text=True).strip()
            if state!='running':record['status']='failed_'+state;break
            try:
                with build_opener(ProxyHandler({})).open('http://127.0.0.1:18105/v1/models',timeout=3) as response:models=json.load(response)
                if args.alias in [m['id'] for m in models.get('data',[])]:record['status']='ready';record['advertised_models']=models;break
            except (OSError,ValueError):pass
            time.sleep(5)
        else:record['status']='startup_timeout_inspect_container'
    except Exception as exc:record.update({'status':'error','error':str(exc)});raise
    finally:record['seconds_until_observation']=time.monotonic()-t;save()
    print(json.dumps({k:record.get(k) for k in ('status','seconds_until_observation','container_id')}))
    if record['status']!='ready':raise SystemExit(1)
if __name__=='__main__':main()
