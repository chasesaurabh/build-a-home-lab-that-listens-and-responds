"""Measure the speech gate and raw ASR on authored non-speech signals.

These synthetic stress fixtures do not estimate household false-alarm rates.
Raw ASR is deliberately run even on rejected inputs for comparison; the application
path remains VAD-gated and never turns these records into actions.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys
import time
import wave
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from home_lab_voice.runtimes import SileroVadRuntime,FasterWhisperRuntime

KINDS=('tone','hum','clicks','white_noise','chirp','dc_offset')
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def make(path,kind,amplitude,seconds,seed):
    rate=16000;rng=random.Random(seed);samples=[]
    for i in range(round(seconds*rate)):
        t=i/rate
        if kind=='silence':value=0
        elif kind=='tone':value=amplitude*math.sin(2*math.pi*1000*t)
        elif kind=='hum':value=amplitude*(.7*math.sin(2*math.pi*60*t)+.3*math.sin(2*math.pi*120*t))
        elif kind=='clicks':value=amplitude if i%(rate//4)<8 else 0
        elif kind=='white_noise':value=amplitude*rng.uniform(-1,1)
        elif kind=='chirp':value=amplitude*math.sin(2*math.pi*(200*t+400*t*t))
        elif kind=='dc_offset':value=amplitude
        else:raise ValueError('Unknown fixture')
        samples.append(round(max(-1,min(1,value))*32767))
    with wave.open(str(path),'wb') as writer:
        writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(rate);writer.writeframes(struct.pack('<'+'h'*len(samples),*samples))
    return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'samples':len(samples),'duration_seconds':len(samples)/rate,
            'peak_fraction_full_scale':max(map(abs,samples))/32768,'rms_fraction_full_scale':math.sqrt(sum(v*v for v in samples)/len(samples))/32768,
            'mean_fraction_full_scale':sum(samples)/len(samples)/32768}

def main():
    p=argparse.ArgumentParser();p.add_argument('--models-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);audio=args.output/'audio';audio.mkdir();cases=[]
    for split,seconds in [('development',3),('evaluation',4.5)]:
        configurations=[('silence',0)]+[(kind,amplitude) for kind in KINDS for amplitude in (.03,.5)]
        for index,(kind,amplitude) in enumerate(configurations):
            ident=f'{split}-{index:02d}-{kind}';seed=20261111+len(cases)
            case={'id':ident,'split':split,'kind':kind,'amplitude_parameter':amplitude,'seed':seed,'expected_speech':False,'file':'audio/'+ident+'.wav'}
            case.update(make(args.output/case['file'],kind,amplitude,seconds,seed));cases.append(case)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'26 original synthetic non-speech signals. 13development/13evaluation. No threshold tuning; fixed VAD0.5 and default duration/padding.',
              'limitations':'No real rooms, microphones, music recordings or prevalence sample. No speech positives, so no miss-rate claim.',
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases}
    write(args.output/'manifest.json',manifest) # Freeze all inputs before loading models.
    vad=SileroVadRuntime(model_path=args.models_root/'vad/silero-vad-6.2.3/silero_vad.onnx')
    asr=FasterWhisperRuntime(args.models_root/'asr/faster-whisper-base.en',cpu_threads=4)
    rows=[]
    for case in cases:
        source=args.output/case['file'];t=time.perf_counter();regions=vad.analyze(source);vad_seconds=time.perf_counter()-t
        t=time.perf_counter();transcript=asr.transcribe(source);asr_seconds=time.perf_counter()-t
        row={'id':case['id'],'split':case['split'],'kind':case['kind'],'vad_speech':bool(regions.regions),
             'vad_speech_seconds':regions.speech_seconds,'raw_asr_text':transcript.text,'raw_asr_nonempty':bool(transcript.text.strip()),
             'gate_would_pass':bool(regions.regions) and bool(transcript.text.strip()),'vad_seconds':vad_seconds,'asr_seconds':asr_seconds}
        rows.append(row);write(args.output/'predictions.json',rows);print(case['id'],row['vad_speech'],repr(transcript.text),flush=True)
    summary={split:{'n':sum(r['split']==split for r in rows),'vad_false_activations':sum(r['vad_speech'] for r in rows if r['split']==split),
        'raw_asr_nonempty':sum(r['raw_asr_nonempty'] for r in rows if r['split']==split),'nonempty_after_gate':sum(r['gate_would_pass'] for r in rows if r['split']==split)} for split in ('development','evaluation')}
    write(args.output/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
