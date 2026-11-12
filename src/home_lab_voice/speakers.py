"""Anonymous, known-count speaker grouping for bounded non-overlapping files."""
import hashlib
from pathlib import Path
from .runtimes import SileroVadRuntime

WEIGHT_SHA='0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2'

def average_linkage(vectors, count):
    """Deterministic average cosine linkage; no person identification."""
    import numpy as np
    matrix=np.asarray(vectors,dtype=float)
    if matrix.ndim!=2 or not np.isfinite(matrix).all() or not 1<=count<=len(matrix):
        raise ValueError('Invalid embeddings or cluster count')
    norms=np.linalg.norm(matrix,axis=1)
    if np.any(norms<1e-12):raise ValueError('Zero-length embedding')
    matrix=matrix/norms[:,None];similarity=matrix@matrix.T;groups=[[i] for i in range(len(matrix))]
    while len(groups)>count:
        options=[(float(similarity[np.ix_(groups[i],groups[j])].mean()),-i,-j)
                 for i in range(len(groups)) for j in range(i+1,len(groups))]
        _,negative_i,negative_j=max(options);i,j=-negative_i,-negative_j
        groups[i]=sorted(groups[i]+groups[j]);del groups[j]
    labels=[0]*len(matrix)
    for label,group in enumerate(sorted(groups,key=lambda g:g[0])):
        for index in group:labels[index]=label
    return labels

class SpeakerGrouper:
    def __init__(self, weights:Path, vad_model:Path):
        if hashlib.sha256(Path(weights).read_bytes()).hexdigest()!=WEIGHT_SHA:raise ValueError('ECAPA identity mismatch')
        if hashlib.sha256(Path(vad_model).read_bytes()).hexdigest()!='1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3':
            raise ValueError('VAD identity mismatch')
        import torch
        from speechbrain.lobes.features import Fbank
        from speechbrain.processing.features import InputNormalization
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        torch.set_num_threads(4);self.torch=torch
        self.features=Fbank(n_mels=80);self.normalize=InputNormalization(norm_type='sentence',std_norm=False)
        self.model=ECAPA_TDNN(input_size=80,channels=[1024,1024,1024,1024,3072],kernel_sizes=[5,3,3,3,1],
            dilations=[1,2,3,4,1],attention_channels=128,lin_neurons=192)
        self.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));self.model.eval()
        self.vad=SileroVadRuntime(model_path=vad_model)

    def group(self, audio:Path, count:int):
        import soundfile as sf
        if type(count)!=int or not 1<=count<=8:raise ValueError('Speaker count must be an integer from 1 to 8')
        if audio.stat().st_size>32*1024*1024:raise ValueError('Audio exceeds 32 MiB')
        with sf.SoundFile(audio) as source:
            if source.samplerate!=16000 or source.channels!=1 or source.format!='WAV' or source.subtype!='PCM_16':
                raise ValueError('Expected mono 16 kHz PCM16 WAV')
            if not 1<=len(source)/16000<=600:raise ValueError('Expected 1 to 600 seconds')
            samples=source.read(dtype='float32')
        regions=self.vad.analyze(audio).regions;turns=[];vectors=[];indices=[]
        for region in regions:
            start=region.start_seconds
            while start<region.end_seconds:
                end=min(start+8,region.end_seconds)
                turns.append({'start':start,'end':end,'speaker':None,'reason':'short region'})
                if len(turns)>512:raise ValueError('Region limit exceeded')
                if end-start>=1:
                    if len(vectors)>=128:raise ValueError('Embedding window limit exceeded')
                    clip=samples[round(start*16000):round(end*16000)]
                    with self.torch.inference_mode():
                        features=self.features(self.torch.from_numpy(clip.copy()).unsqueeze(0))
                        vector=self.model(self.normalize(features,self.torch.ones(1))).reshape(-1).numpy()
                    indices.append(len(turns)-1);vectors.append(vector)
                start=end
        if not vectors or len(vectors)<count:
            return {'turns':turns,'status':'insufficient_long_regions','known_speaker_count':count}
        labels=average_linkage(vectors,count)
        for index,label in zip(indices,labels):turns[index].update({'speaker':f'SPEAKER_{label:02d}','reason':'known-count clustering'})
        return {'source_sha256':hashlib.sha256(audio.read_bytes()).hexdigest(),'turns':turns,'status':'ok',
            'known_speaker_count':count,'embedding_sha256':WEIGHT_SHA,'minimum_region_seconds':1,'maximum_region_seconds':8,
            'scope':'Anonymous per-file groups, known speaker count, no overlap handling or identity/authentication. Embeddings are not saved.'}
