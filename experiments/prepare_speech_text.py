import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from home_lab_voice.speech_text import prepare_text
def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--rules',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.source.stat().st_size>128*1024 or a.rules.stat().st_size>128*1024:raise ValueError('Input exceeds128KiB')
    raw=a.source.read_bytes();result=prepare_text(raw.decode('utf-8-sig'),json.loads(a.rules.read_text(encoding='utf-8-sig')))
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'source.txt').write_bytes(raw)
    (a.output/'spoken.txt').write_text(result['spoken_text'],encoding='utf-8')
    (a.output/'preparation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'changes':len(result['changes']),'source_sha256':result['source_sha256'],'spoken_sha256':result['spoken_sha256']}))
if __name__=='__main__':main()
