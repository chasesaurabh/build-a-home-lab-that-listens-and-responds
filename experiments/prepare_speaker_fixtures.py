"""Make reproducible non-overlapping read-speech mixtures with known speakers."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from prepare_librispeech_parquet import REPO,REVISION,FILES
from huggingface_hub import hf_hub_download
SEED='voice-speaker-groups-005'
def key(value):return hashlib.sha256((SEED+'|'+str(value)).encode()).hexdigest()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);cases=[]
    for subset,split in [('dev-clean','development'),('test-clean','evaluation')]:
        path=Path(hf_hub_download(REPO,FILES[subset],repo_type='dataset',revision=REVISION))
        table=pq.read_table(path,columns=['id','speaker_id','text']);by_speaker={}
        for row in table.to_pylist():by_speaker.setdefault(str(row['speaker_id']),[]).append(row)
        speakers=sorted((s for s,rows in by_speaker.items() if len(rows)>=4),key=key)[:4]
        chosen={s:sorted(by_speaker[s],key=lambda r:key(r['id']))[:4] for s in speakers}
        ids={r['id'] for rows in chosen.values() for r in rows};saved={}
        directory=a.output/split;directory.mkdir()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64,columns=['id','audio']):
            for row in batch.to_pylist():
                if row['id'] in ids:
                    out=directory/(row['id']+'.flac');out.write_bytes(row['audio']['bytes']);saved[row['id']]=out
        pieces=[];segments=[];offset=0
        for turn in range(4):
            for speaker in speakers:
                row=chosen[speaker][turn];clip=saved[row['id']];samples,rate=sf.read(clip,dtype='float32')
                if rate!=16000 or samples.ndim!=1:raise ValueError('Unexpected fixture format')
                silence=np.zeros(8000,dtype=np.float32);pieces.append(silence);offset+=len(silence)
                segments.append({'start':offset/16000,'end':(offset+len(samples))/16000,'speaker':speaker,
                    'utterance':row['id'],'source_sha256':sha(clip),'reference_text':row['text']})
                pieces.append(samples);offset+=len(samples)
        output=directory/'mixture.wav';sf.write(output,np.concatenate(pieces),16000,subtype='PCM_16')
        cases.append({'id':split,'split':split,'audio':output.relative_to(a.output).as_posix(),'sha256':sha(output),
            'known_speaker_count':4,'duration_seconds':offset/16000,'segments':segments,'source_parquet_sha256':sha(path)})
    result={'seed':SEED,'source_repository':REPO,'revision':REVISION,'license':'CC-BY-4.0',
        'attribution':'LibriSpeech, Panayotov, Chen, Povey and Khudanpur, ICASSP 2015; LibriVox audiobooks.',
        'scope':'Two synthetic turn sequences, four distinct readers each, sixteen clips each, 0.5-second inserted gaps. No overlap or room mixing. Clip boundaries are not human speech activity annotations.',
        'cases':cases}
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'cases':len(cases),'seconds':[c['duration_seconds'] for c in cases]}))
if __name__=='__main__':main()
