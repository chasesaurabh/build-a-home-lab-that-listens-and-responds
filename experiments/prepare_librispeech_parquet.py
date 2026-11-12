"""Acquire the same seeded selection from OpenSLR's pinned Hugging Face mirror."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from librispeech_selection import SEED, SUBSETS, select, sha

REPO='openslr/librispeech_asr'
REVISION='71cacbfb7e2354c4226d01e70d77d5fca3d04ba1'
FILES={'dev-clean':'all/validation.clean/0000.parquet',
       'dev-other':'all/validation.other/0000.parquet',
       'test-clean':'all/test.clean/0000.parquet',
       'test-other':'all/test.other/0000.parquet'}

def prepare(root,subset):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import soundfile as sf
    filename=FILES[subset]
    path=Path(hf_hub_download(REPO,filename,repo_type='dataset',revision=REVISION,local_dir=root/'downloads'))
    table=pq.read_table(path,columns=['id','text','speaker_id'])
    rows=table.to_pylist()
    refs={r['id']:r['text'] for r in rows}
    if len(refs)!=len(rows): raise ValueError('Duplicate utterance ID')
    chosen=select(refs,SUBSETS[subset]); wanted=set(chosen)
    saved={}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64,columns=['id','audio','speaker_id']):
        for r in batch.to_pylist():
            ident=r['id']
            if ident not in wanted: continue
            rel=Path('audio')/subset/f'{ident}.flac'
            target=root/rel; target.parent.mkdir(parents=True,exist_ok=True)
            audio=r['audio']['bytes']
            if not audio or len(audio)>64*1024*1024: raise ValueError('Invalid embedded audio')
            if target.exists() and target.read_bytes()!=audio: raise ValueError('Existing fixture differs')
            if not target.exists(): target.write_bytes(audio)
            info=sf.info(target)
            if info.samplerate!=16000 or info.channels!=1 or info.frames<=0: raise ValueError('Unexpected format')
            saved[ident]={'id':ident,'speaker_id':str(r['speaker_id']),'subset':subset,
               'split':'development' if subset.startswith('dev') else 'evaluation',
               'path':rel.as_posix(),'sha256':sha(target),'reference':refs[ident],
               'sample_rate':info.samplerate,'channels':info.channels,'frames':info.frames,
               'duration_seconds':info.duration}
    if set(saved)!=wanted: raise ValueError('Missing selected audio')
    print(f'{subset}: {len(saved)} selected',flush=True)
    return {'subset':subset,'source_file':filename,'sha256':sha(path)},[saved[i] for i in chosen]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); args=p.parse_args()
    root=args.root.resolve(); root.mkdir(parents=True,exist_ok=True)
    output=root/'manifest.json'
    if output.exists(): raise SystemExit('Preserve frozen manifest; use a new root')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(prepare,root,s) for s in SUBSETS]
        results=[f.result() for f in futures]
    records=[r for _,items in results for r in items]
    data={'schema_version':1,'selection_seed':SEED,'source':'https://www.openslr.org/12',
      'mirror_repository':REPO,'mirror_revision':REVISION,'license':'CC-BY-4.0',
      'attribution':'LibriSpeech, Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, ICASSP 2015; based on LibriVox public-domain audiobooks.',
      'license_url':'https://creativecommons.org/licenses/by/4.0/',
      'selection':'Speaker-round-robin, seeded SHA-256 ordering, frozen before inference.',
      'limitations':'Read English audiobooks, not household conversations. Public test sets may have appeared in model training.',
      'source_files':[a for a,_ in results],'records':records}
    output.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'utterances':len(records),'duration_seconds':sum(r['duration_seconds'] for r in records),
                     'manifest_sha256':sha(output)}),flush=True)

if __name__=='__main__': main()
