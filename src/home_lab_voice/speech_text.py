"""Single-pass literal pronunciation preparation with preserved source offsets."""
import hashlib,json,re

def prepare_text(text:str,rules:list[dict]):
    if not isinstance(text,str) or not text.strip() or len(text)>20000:raise ValueError('Expected1..20000 text characters')
    if not isinstance(rules,list) or len(rules)>64:raise ValueError('Expected at most64 rules')
    identifiers=set();patterns=set()
    for rule in rules:
        if not isinstance(rule,dict) or set(rule)!={'id','source','spoken'}:raise ValueError('Rules require id,source,spoken only')
        if any(not isinstance(v,str) for v in rule.values()):raise ValueError('Rule values must be strings')
        if not re.fullmatch('[a-z][a-z0-9_-]{0,47}',rule['id']) or rule['id'] in identifiers:raise ValueError('Invalid or duplicate rule ID')
        if not 1<=len(rule['source'])<=80 or not 1<=len(rule['spoken'])<=200:raise ValueError('Invalid rule length')
        if rule['source'] in patterns:raise ValueError('Duplicate source rule')
        identifiers.add(rule['id']);patterns.add(rule['source'])
    ordered=sorted(rules,key=lambda r:(-len(r['source']),r['id']));pieces=[];changes=[];position=0;spoken_length=0
    def word(character):return character.isalnum() or character=='_'
    while position<len(text):
        match=None
        for rule in ordered:
            end=position+len(rule['source'])
            if not text.startswith(rule['source'],position):continue
            if word(rule['source'][0]) and position and word(text[position-1]):continue
            if word(rule['source'][-1]) and end<len(text) and word(text[end]):continue
            match=rule;break
        if match:
            replacement=match['spoken'];end=position+len(match['source'])
            changes.append({'rule_id':match['id'],'source_start':position,'source_end':end,
                'spoken_start':spoken_length,'spoken_end':spoken_length+len(replacement),
                'source_text':text[position:end],'spoken_text':replacement})
            position=end
        else:replacement=text[position];position+=1
        pieces.append(replacement);spoken_length+=len(replacement)
        if spoken_length>40000:raise ValueError('Prepared text exceeds40000 characters')
    spoken=''.join(pieces)
    return {'source_text':text,'spoken_text':spoken,'source_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'spoken_sha256':hashlib.sha256(spoken.encode()).hexdigest(),
        'rules_sha256':hashlib.sha256(json.dumps(rules,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'changes':changes,'policy':'Case-sensitive literal matches, longest first, word-edge checks, single pass; unmatched text unchanged.',
        'scope':'Text preparation, not a listening-quality or pronunciation-correctness score.'}
