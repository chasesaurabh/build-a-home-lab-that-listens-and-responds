"""Hash cached model snapshots and verify large-file identities against pinned Hub revisions."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi

MODELS=[('Qwen/Qwen3.5-2B','15852e8c16360a2fea060d615a32b45270f8a8fc'),
        ('Qwen/Qwen3.5-27B','fc05daec18b0a78c049392ed2e771dde82bdf654'),
        ('nvidia/Qwen3.5-122B-A10B-NVFP4','98915d837c4e7c87ac8296d02e89de19b3207e6d')]
def main():
    p=argparse.ArgumentParser();p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);api=HfApi()
    for repo,rev in MODELS:
        directory=args.cache_root/repo.replace('/','--')/rev
        info=api.model_info(repo,revision=rev,files_metadata=True)
        if info.sha!=rev:raise ValueError('Revision mismatch')
        remote={s.rfilename:s for s in info.siblings};files=[]
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or '.cache' in path.relative_to(directory).parts:continue
            name=path.relative_to(directory).as_posix()
            if name not in remote:raise ValueError('Unrecognized local model file: '+name)
            h=hashlib.sha256();gitsha=hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block);gitsha.update(block)
            sibling=remote[name];lfs=sibling.lfs
            expected=lfs.sha256 if lfs else sibling.blob_id
            observed=h.hexdigest() if lfs else gitsha.hexdigest()
            if not expected or observed!=expected:raise ValueError('Remote identity mismatch: '+name)
            files.append({'path':name,'bytes':path.stat().st_size,'sha256':h.hexdigest(),'remote_identity_verified':True})
        required=[s.rfilename for s in info.siblings if s.rfilename.endswith('.safetensors')]
        if not required or not set(required)<=set(f['path'] for f in files):raise ValueError('Missing model weights')
        record={'model':repo,'revision':rev,'license':info.card_data.to_dict().get('license'),
                'verified_utc':datetime.now(timezone.utc).isoformat(),'files':files}
        (args.output/(repo.replace('/','--')+'.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'model':repo,'verified_files':len(files),'bytes':sum(f['bytes'] for f in files)}),flush=True)
if __name__=='__main__':main()
