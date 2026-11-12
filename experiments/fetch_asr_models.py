"""Download official Faster-Whisper conversions at resolved immutable commits."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

REPOS = {
 'tiny.en':'Systran/faster-whisper-tiny.en',
 'base.en':'Systran/faster-whisper-base.en',
 'small.en':'Systran/faster-whisper-small.en',
 'large-v3':'Systran/faster-whisper-large-v3',
}
FILES=['config.json','preprocessor_config.json','model.bin','tokenizer.json','vocabulary.*','README.md']
REVISIONS={
 'tiny.en':'0d3d19a32d3338f10357c0889762bd8d64bbdeba',
 'base.en':'3d3d5dee26484f91867d81cb899cfcf72b96be6c',
 'small.en':'d1d751a5f8271d482d14ca55d9e2deeebbae577f',
 'large-v3':'edaa852ec7e145841d8ffdb056a99866b5f0a478',
}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def acquire(root,name):
    from huggingface_hub import HfApi, snapshot_download
    repo=REPOS[name]
    info=HfApi().model_info(repo,revision=REVISIONS[name])
    revision=info.sha
    if revision!=REVISIONS[name]: raise ValueError('Pinned commit mismatch')
    card=info.card_data.to_dict() if info.card_data else {}
    if card.get('license')!='mit': raise ValueError(f'Inspect model license before acquisition: {repo}')
    directory=root/name/revision
    snapshot_download(repo,revision=revision,allow_patterns=FILES,local_dir=directory,max_workers=4)
    files=[]
    for path in sorted(directory.iterdir()):
        if path.is_file(): files.append({'path':path.name,'bytes':path.stat().st_size,'sha256':sha(path)})
    if not {'model.bin','config.json','tokenizer.json'}.issubset({f['path'] for f in files}):
        raise ValueError('Incomplete model')
    result={'name':name,'repository':repo,'revision':revision,'license':'MIT',
            'source':f'https://huggingface.co/{repo}/tree/{revision}',
            'relative_directory':directory.relative_to(root).as_posix(),'files':files}
    print(f'Acquired {name} at {revision}',flush=True)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve(); root.mkdir(parents=True,exist_ok=True)
    manifest=root/'acquisition.json'
    if manifest.exists(): raise SystemExit('Preserve existing acquisition; use its pinned revisions')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(acquire,root,name) for name in REPOS]
        models=[f.result() for f in futures]
    manifest.write_text(json.dumps({'schema_version':1,'acquired_utc':datetime.now(timezone.utc).isoformat(),
        'models':models},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()
