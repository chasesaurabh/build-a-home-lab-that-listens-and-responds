"""Annotate one permitted clip with the separately installed CPU event model."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from home_lab_voice.events import EventTagger
def main():
    p=argparse.ArgumentParser();p.add_argument('audio',type=Path);p.add_argument('--acquisition',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Output already exists')
    result=EventTagger(a.acquisition).annotate(a.audio)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['annotations']))
if __name__=='__main__':main()
