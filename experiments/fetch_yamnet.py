"""Acquire the official version-1 YAMNet bundle, with safe extraction and hashes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request

URL = 'https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-sha256',default='b80da2a1a56926fb0767205051a200dd7b3beaf3ea1ea126c42a53943996e5e0');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    archive=args.output/'archive.tar.gz'
    with urllib.request.urlopen(URL,timeout=60) as source, archive.open('xb') as target:
        total=0
        while block:=source.read(1024*1024):
            total+=len(block)
            if total>128*1024*1024:raise ValueError('Bundle exceeds 128 MiB')
            target.write(block)
    sha=digest(archive)
    if args.expected_sha256 and sha!=args.expected_sha256:raise ValueError('Archive identity mismatch')
    destination=args.output/'model';destination.mkdir()
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers()
        if sum(m.size for m in members)>256*1024*1024:raise ValueError('Expanded bundle exceeds limit')
        for member in members:
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or '\\' in member.name or ':' in member.name:
                raise ValueError('Unsafe member path')
            if not(member.isdir() or member.isfile()):raise ValueError('Links and special members rejected')
            target=destination.joinpath(*path.parts)
            if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with bundle.extractfile(member) as source,target.open('xb') as output:output.write(source.read())
    record={'acquired_utc':datetime.now(timezone.utc).isoformat(),'source':URL,'archive_sha256':sha,
            'files':[{'path':x.relative_to(destination).as_posix(),'sha256':digest(x)} for x in sorted(destination.rglob('*')) if x.is_file()]}
    (args.output/'acquisition.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'archive_sha256':sha,'files':len(record['files'])}))
if __name__=='__main__':main()
