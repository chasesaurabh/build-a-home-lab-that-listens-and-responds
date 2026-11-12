"""Download one pinned model and verify every retained file against Hub identities."""
import argparse,hashlib,json,fnmatch
from pathlib import Path,PurePosixPath
from urllib.request import urlopen
from urllib.parse import quote
from verify_interpretation_models import MODELS
def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=['2b','27b','122b'],default='2b')
    p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    repo,revision=dict(zip(['2b','27b','122b'],MODELS))[a.size]
    directory=a.root/repo.replace('/','--')/revision
    if directory.exists():raise ValueError('Use a new destination to preserve prior acquisition')
    with urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true',timeout=60) as response:info=json.load(response)
    if info['sha']!=revision:raise ValueError('Revision mismatch')
    remote={s['rfilename']:s for s in info['siblings']};files=[];directory.mkdir(parents=True)
    patterns=['*.safetensors','*.json','*.model','*.txt','*.md','*.jinja','LICENSE*']
    for name,sibling in remote.items():
        if not any(fnmatch.fnmatchcase(name,pattern) for pattern in patterns):continue
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:raise ValueError('Unsafe model path')
        path=directory.joinpath(*relative.parts);path.parent.mkdir(parents=True,exist_ok=True)
        size=sibling['size'];total=0
        with urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{quote(name)}',timeout=120) as response,path.open('xb') as output:
            while block:=response.read(8*1024*1024):
                total+=len(block)
                if total>size:raise ValueError('Download exceeds declared size')
                output.write(block)
        if total!=size:raise ValueError('Incomplete download')
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or '.cache' in path.relative_to(directory).parts:continue
        name=path.relative_to(directory).as_posix();sibling=remote[name]
        sha=hashlib.sha256();blob=hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as source:
            for block in iter(lambda:source.read(8*1024*1024),b''):sha.update(block);blob.update(block)
        expected=sibling['lfs']['sha256'] if sibling.get('lfs') else sibling['blobId']
        observed=sha.hexdigest() if sibling.get('lfs') else blob.hexdigest()
        if not expected or observed!=expected:raise ValueError('Remote identity mismatch: '+name)
        files.append({'path':name,'sha256':sha.hexdigest(),'bytes':path.stat().st_size})
    required={s['rfilename'] for s in info['siblings'] if s['rfilename'].endswith('.safetensors')}
    if not required or not required<={r['path'] for r in files}:raise ValueError('Missing weights')
    result={'model':repo,'revision':revision,'license':info.get('cardData',{}).get('license'),'files':files}
    (a.root/(a.size+'-acquisition.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'model':repo,'files':len(files),'bytes':sum(r['bytes'] for r in files)}))
if __name__=='__main__':main()
