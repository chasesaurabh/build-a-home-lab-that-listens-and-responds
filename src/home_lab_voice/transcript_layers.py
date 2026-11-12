"""Preserve raw transcripts and create traceable display-only corrections."""
import hashlib,json,math,unicodedata
from pathlib import Path

def build_layers(raw:dict,corrections:dict|None=None):
    if not isinstance(raw,dict) or not isinstance(raw.get('segments'),list) or len(raw['segments'])>4096:
        raise ValueError('Expected at most 4096 transcript segments')
    duration=raw.get('duration_seconds')
    if type(duration) not in (int,float) or not math.isfinite(duration) or not 0<duration<=600:
        raise ValueError('Invalid transcript duration')
    corrections={} if corrections is None else corrections
    if not isinstance(corrections,dict) or any(not isinstance(k,str) or not k.isdecimal() for k in corrections):
        raise ValueError('Corrections map segment-number strings to replacement text')
    if any(str(int(k))!=k or int(k)>=len(raw['segments']) for k in corrections):raise ValueError('Unknown correction segment')
    rows=[];previous_start=-1
    for index,segment in enumerate(raw['segments']):
        if not isinstance(segment,dict):raise ValueError('Invalid segment')
        start,end=segment.get('start_seconds'),segment.get('end_seconds');text=segment.get('text')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (start,end)):
            raise ValueError('Invalid segment time')
        if start<0 or end<start or end>duration+.05 or start<previous_start:raise ValueError('Segment outside ordered timeline')
        if not isinstance(text,str) or len(text)>20000:raise ValueError('Invalid segment text')
        previous_start=start
        normalized=' '.join(unicodedata.normalize('NFKC',text).split())
        corrected=corrections.get(str(index),normalized)
        if not isinstance(corrected,str) or len(corrected)>20000:raise ValueError('Invalid correction text')
        rows.append({'segment_id':index,'raw_start_seconds':start,'raw_end_seconds':end,
            'display_start_ms':round(min(start,duration)*1000),'display_end_ms':round(min(end,duration)*1000),
            'display_time_clamped':end>duration,'raw_text':text,'normalized_text':normalized,
            'corrected_text':corrected,'reviewed_edit':str(index) in corrections})
    return {'schema_version':'transcript-layers.v1','duration_seconds':duration,
        'normalization':'Unicode NFKC and collapsed whitespace only; no punctuation, spelling or number rewrite.',
        'time_policy':'Nearest integer millisecond for display; up to50ms end overshoot visibly clamped. Raw values preserved. Overlap is allowed.',
        'segments':rows,'scope':'Derived text and navigation coordinates, not forced alignment or proof that a correction matches audio.'}

def preserve_layers(source:Path,output:Path,corrections:dict|None=None):
    if source.stat().st_size>2*1024*1024:raise ValueError('Transcript exceeds2MiB')
    data=source.read_bytes();result=build_layers(json.loads(data),corrections)
    output.mkdir(parents=True,exist_ok=False)
    (output/'raw.json').write_bytes(data)
    result['raw_sha256']=hashlib.sha256(data).hexdigest()
    (output/'layers.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (output/'display.txt').write_text('\n'.join(r['corrected_text'] for r in result['segments'])+'\n',encoding='utf-8')
    return result
