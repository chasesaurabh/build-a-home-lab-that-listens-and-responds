"""Acquire only pinned ECAPA embedding weights and their source documentation."""
import argparse,hashlib,json
from pathlib import Path
from urllib.request import urlopen
REVISION='0f99f2d0ebe89ac095bcc5903c4dd8f72b367286'
WEIGHT_SHA='0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2'
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for name in ['embedding_model.ckpt','hyperparams.yaml','README.md']:
        with urlopen(f'https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/{REVISION}/{name}',timeout=60) as source,(a.output/name).open('xb') as target:
            total=0
            while block:=source.read(1024*1024):
                total+=len(block)
                if total>100*1024*1024:raise ValueError('Download exceeds size limit')
                target.write(block)
        sha=hashlib.sha256((a.output/name).read_bytes()).hexdigest()
        if name=='embedding_model.ckpt' and sha!=WEIGHT_SHA:raise ValueError('Weight identity mismatch')
        rows.append({'path':name,'sha256':sha})
    (a.output/'acquisition.json').write_text(json.dumps({'model':'speechbrain/spkrec-ecapa-voxceleb',
        'revision':REVISION,'license':'Apache-2.0','files':rows},indent=2)+'\n')
    print('Acquired and verified',len(rows),'files')
if __name__=='__main__':main()
