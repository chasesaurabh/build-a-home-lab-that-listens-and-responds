"""Bounded, file-based YAMNet annotations; no alarms or external actions."""
import csv
import hashlib
import json
import math
from pathlib import Path

SELECTED = {0:'Speech',494:'Silence',495:'Sine wave',514:'White noise'}

class EventTagger:
    def __init__(self, acquisition: Path):
        acquisition=Path(acquisition)
        record=json.loads(acquisition.read_text(encoding='utf-8'))
        root=acquisition.parent/'model'
        for entry in record['files']:
            path=(root/entry['path']).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError('Invalid model manifest path')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError('Model artifact identity mismatch')
        self.bundle_sha256=record['archive_sha256']
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        self.tf=tf
        self.model=tf.saved_model.load(str(root))
        with (root/'assets/yamnet_class_map.csv').open(newline='',encoding='utf-8') as f:
            self.labels=[row['display_name'] for row in csv.DictReader(f)]
        if len(self.labels)!=521 or any(self.labels[i]!=name for i,name in SELECTED.items()):
            raise ValueError('Unexpected YAMNet class map')

    def annotate(self, path: Path):
        import numpy as np
        import soundfile as sf
        path=Path(path)
        if not path.is_file() or path.stat().st_size>64*1024*1024:
            raise ValueError('Input must be a file of at most 64 MiB')
        with sf.SoundFile(path) as audio:
            if audio.format not in ('WAV','FLAC') or audio.samplerate!=16000 or audio.channels!=1:
                raise ValueError('Expected mono 16 kHz WAV or FLAC')
            if not 8000<=len(audio)<=16000*60:raise ValueError('Expected 0.5 to 60 seconds')
            samples=audio.read(dtype='float32')
        if not np.isfinite(samples).all() or np.max(np.abs(samples))>1:
            raise ValueError('Expected finite normalized waveform')
        scores,_,_=self.model(self.tf.convert_to_tensor(samples))
        matrix=np.asarray(scores)
        if matrix.ndim!=2 or matrix.shape[1]!=521 or not np.isfinite(matrix).all():
            raise ValueError('Invalid score matrix')
        means=matrix.mean(axis=0)
        annotations=[{'class_index':i,'label':name,'mean_score':float(means[i]),
            'disposition':'accept' if means[i]>=.5 else 'review' if means[i]>=.2 else 'abstain'}
            for i,name in SELECTED.items()]
        return {'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'model_bundle_sha256':self.bundle_sha256,'duration_seconds':len(samples)/16000,
            'frame_count':len(matrix),'aggregation':'mean over model frames',
            'thresholds':{'accept':.5,'review':.2},'annotations':annotations,
            'top_five':[{'class_index':int(i),'label':self.labels[i],'mean_score':float(means[i])}
                        for i in np.argsort(means)[-5:][::-1]],
            'mean_scores':[float(v) for v in means],
            'scope':'Whole-clip review annotation. Scores are not calibrated probabilities. No actions.'}
