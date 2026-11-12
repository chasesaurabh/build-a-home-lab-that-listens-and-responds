"""Objective PCM checks, deliberately separate from listening quality."""
import argparse,hashlib,json,math,struct,wave
from pathlib import Path

def audit(path):
    path=Path(path)
    if path.stat().st_size>64*1024*1024:raise ValueError('File exceeds64MiB')
    with wave.open(str(path),'rb') as source:
        channels=source.getnchannels();rate=source.getframerate();frames=source.getnframes()
        if source.getsampwidth()!=2 or source.getcomptype()!='NONE' or channels not in (1,2):raise ValueError('Expected mono/stereo PCM16')
        if not 8000<=rate<=192000 or not 0<frames<=rate*600:raise ValueError('Invalid duration or rate')
        stats=[{'samples':0,'sum':0,'squared':0,'peak':0,'zero':0,'near_full_scale':0} for _ in range(channels)]
        seen=0
        while block:=source.readframes(65536):
            if len(block)%(channels*2):raise ValueError('Partial audio frame')
            values=struct.unpack('<'+'h'*(len(block)//2),block);seen+=len(values)//channels
            for index,value in enumerate(values):
                row=stats[index%channels];row['samples']+=1;row['sum']+=value;row['squared']+=value*value
                row['peak']=max(row['peak'],abs(value));row['zero']+=value==0;row['near_full_scale']+=abs(value)>=32760
        if seen!=frames:raise ValueError('Truncated audio')
    results=[]
    for row in stats:
        n=row['samples'];rms=math.sqrt(row['squared']/n)/32768
        results.append({'peak_fraction_full_scale':row['peak']/32768,'rms_fraction_full_scale':rms,
            'rms_dbfs':20*math.log10(rms) if rms else None,'dc_offset_fraction_full_scale':row['sum']/n/32768,
            'exact_zero_fraction':row['zero']/n,'near_full_scale_samples':row['near_full_scale']})
    return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sample_rate_hz':rate,'channels':channels,
        'frames':frames,'duration_seconds':frames/rate,'channel_statistics':results,
        'scope':'File integrity and sample statistics only. Near-full-scale counts are a diagnostic, not proof of clipping or a listening score.'}
def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.source)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
