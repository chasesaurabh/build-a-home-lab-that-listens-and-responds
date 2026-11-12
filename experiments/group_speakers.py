import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from home_lab_voice.speakers import SpeakerGrouper
def main():
    p=argparse.ArgumentParser();p.add_argument('audio',type=Path);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--vad-model',type=Path,required=True);p.add_argument('--speakers',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    result=SpeakerGrouper(a.weights,a.vad_model).group(a.audio,a.speakers)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(result['status'],len(result['turns']),'regions')
if __name__=='__main__':main()
