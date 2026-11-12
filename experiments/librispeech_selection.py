"""Deterministic speaker-round-robin selection, independent of model outcomes."""
from collections import defaultdict
import hashlib
from pathlib import Path
import re

SEED='home-lab-voice-asr-v2-2026-11-10'
SUBSETS={'dev-clean':32,'dev-other':32,'test-clean':96,'test-other':96}
IDENTIFIER=re.compile(r'[0-9]+-[0-9]+-[0-9]+\Z')

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def ranked(value):return hashlib.sha256((SEED+'|'+value).encode()).hexdigest()

def select(refs:dict[str,str],count:int)->list[str]:
    if not isinstance(count,int) or count<1:raise ValueError('Selection count must be positive')
    groups=defaultdict(list)
    for ident,text in refs.items():
        if not isinstance(ident,str) or not IDENTIFIER.fullmatch(ident):raise ValueError('Unsafe or invalid LibriSpeech identifier')
        if not isinstance(text,str) or not text.strip():raise ValueError('Empty transcript reference')
        groups[ident.split('-')[0]].append(ident)
    speakers=sorted(groups,key=ranked)
    for speaker in speakers:groups[speaker].sort(key=ranked)
    result=[]
    while len(result)<count:
        changed=False
        for speaker in speakers:
            if groups[speaker] and len(result)<count:result.append(groups[speaker].pop(0));changed=True
        if not changed:raise ValueError('Insufficient utterances')
    return result
