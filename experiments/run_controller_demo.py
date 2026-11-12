"""Exercise request ownership without pretending to operate audio hardware."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from home_lab_voice.conversation import TurnController,ConversationEvent as Event
from home_lab_voice.contracts import ContractError
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    c=TurnController();old=c.begin();c.dispatch(old,Event.START_LISTENING);c.dispatch(old,Event.CANCEL)
    current=c.begin();c.dispatch(current,Event.START_LISTENING)
    try:c.dispatch(old,Event.AUDIO_READY)
    except ContractError as error:stale_error=error.code
    else:raise AssertionError('Stale result accepted')
    for event in [Event.AUDIO_READY,Event.TRANSCRIPT_READY,Event.CONFIRMATION_REQUIRED]:c.dispatch(current,event)
    try:c.dispatch(current,Event.ACTION_COMPLETE)
    except ContractError as error:early_error=error.code
    else:raise AssertionError('Confirmation bypass accepted')
    for event in [Event.CONFIRMED,Event.ACTION_COMPLETE,Event.PLAYBACK_COMPLETE]:c.dispatch(current,event)
    result={'scope':'Control-plane state exercise. No capture, model, permission or playback side effects.',
        'stale_result_rejected':stale_error,'confirmation_bypass_rejected':early_error,'final_state':c.machine.state,
        'transitions':[{'sequence':t.sequence,'prior':t.prior,'event':t.event,'current':t.current} for t in c.machine.transitions]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
