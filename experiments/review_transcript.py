import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from home_lab_voice.transcript_layers import preserve_layers
def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--corrections',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.corrections and a.corrections.stat().st_size>2*1024*1024:raise ValueError('Correction file exceeds2MiB')
    corrections=json.loads(a.corrections.read_text(encoding='utf-8-sig')) if a.corrections else None
    result=preserve_layers(a.source,a.output,corrections)
    print(json.dumps({'raw_sha256':result['raw_sha256'],'segments':len(result['segments']),
        'reviewed_edits':sum(r['reviewed_edit'] for r in result['segments'])}))
if __name__=='__main__':main()
