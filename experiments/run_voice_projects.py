"""Exercise real local CPU speech models and all four bounded projects.

The generated command and narration are original synthetic fixtures. No microphone
is opened. Model downloads must be completed and verified before this command.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from home_lab_voice.runtimes import FasterWhisperRuntime,SileroVadRuntime,KokoroRuntime
from home_lab_voice.normalize import normalize_audio
from home_lab_voice.applications.workflows import SearchableAudioArchive,ReviewedDictation,NarrationProject,interpret_assistant_command
from home_lab_voice.contracts import AudioEnvelope,ContractError
from home_lab_voice.lab_control import LocalLab
from home_lab_voice.audio_fixtures import write_mono_sine

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--models-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    def record(name,started,**fields):
        item={'name':name,'seconds':time.perf_counter()-started,**fields};results.append(item)
        (args.output/'results.json').write_text(json.dumps({'schema_version':1,
            'scope':'Actual CPU speech integration on original synthetic fixtures; not human-speech accuracy or subjective voice quality.',
            'started_utc':started_utc,'steps':results},indent=2)+'\n',encoding='utf-8')
        print(name,fields.get('status','passed'),flush=True)
    started_utc=datetime.now(timezone.utc).isoformat()
    tts=KokoroRuntime(args.models_root/'tts/Kokoro-82M');vad=SileroVadRuntime(model_path=args.models_root/'vad/silero-vad-6.2.3/silero_vad.onnx')
    asr=FasterWhisperRuntime(args.models_root/'asr/faster-whisper-base.en',cpu_threads=4)
    command='Get status for voice service.';t=time.perf_counter()
    speech=tts.synthesize(command,args.output/'command-24k.wav')
    record('synthesize_command',t,status='passed',text=command,audio_sha256=digest(speech.output),audio_seconds=speech.duration_seconds)
    t=time.perf_counter();normalization=normalize_audio(speech.output,args.output/'command-16k.wav')
    record('normalize_command',t,status='passed',**normalization)
    source=args.output/'command-16k.wav';t=time.perf_counter();regions=vad.analyze(source);transcript=asr.transcribe(source)
    if not regions.regions:raise AssertionError('Expected speech regions')
    mode,arguments=interpret_assistant_command(transcript.text)
    if mode!='status' or arguments!={'service':'voice-service'}:raise AssertionError('Recognized command did not match intended operation')
    record('vad_asr_command',t,status='passed',reference=command,transcript=transcript.text,speech_seconds=regions.speech_seconds)
    envelope=AudioEnvelope(request_id='req_voiceprojects001',source_kind='licensed_fixture',media_type='audio/wav',
        sample_rate_hz=16000,channels=1,captured_at=datetime.now(timezone.utc),consent_basis='licensed_fixture',
        retention_class='project_archive',content_sha256=digest(source))
    archive=SearchableAudioArchive(args.output/'archive');t=time.perf_counter()
    archived=archive.ingest(source,envelope,vad,asr);matches=archive.search('status')
    if len(matches)!=1:raise AssertionError('Archive search did not return the source')
    if not archive.delete(archived.source_digest) or archived.managed_audio.exists() or archived.transcript_record.exists():raise AssertionError('Paired archive deletion failed')
    record('archive_search_delete',t,status='passed',matches_before_deletion=len(matches),matches_after_deletion=len(archive.search('status')))
    workflow=ReviewedDictation(args.output/'dictation');t=time.perf_counter();preview=workflow.preview(source,vad,asr)
    try:workflow.commit(preview.preview_id,'0'*64,'notes.txt','idem_voiceprojects01')
    except ContractError as exc:
        if exc.code!='DICTATION_APPROVAL_MISMATCH':raise
    else:raise AssertionError('Wrong approval accepted')
    token=workflow.commit(preview.preview_id,preview.approval_digest,'notes.txt','idem_voiceprojects01')
    repeated=ReviewedDictation(args.output/'dictation').commit(preview.preview_id,preview.approval_digest,'notes.txt','idem_voiceprojects01')
    if token!=repeated or token.target.read_text()!=preview.text:raise AssertionError('Dictation repeat mismatch')
    ReviewedDictation(args.output/'dictation').undo('idem_voiceprojects01')
    if token.target.exists():raise AssertionError('Undo failed')
    record('dictation_review_commit_repeat_undo',t,status='passed',transcript=preview.text)
    t=time.perf_counter();narration=NarrationProject(args.output/'narration').build(
        'A local voice assistant starts with deliberate input. Review its interpretation before approving a diagnostic.',tts,max_chars=65)
    record('narration_assembly',t,status='passed',chunks=narration.chunk_count,master_sha256=digest(narration.master_audio))
    t=time.perf_counter();lab=LocalLab(args.output/'lab');lab.initialize();lab.enqueue('voice-service','job-voice-study-001')
    status=lab.status(arguments['service']);pending=lab.prepare('voice-service','healthcheck','diagnostic-study-001')
    result=LocalLab(args.output/'lab').confirm(pending['id'],pending['digest'])
    repeated=LocalLab(args.output/'lab').confirm(pending['id'],pending['digest'])
    if result['result']!={'database_check':'ok','jobs':1} or not repeated['replayed']:raise AssertionError('Durable diagnostic mismatch')
    response=tts.synthesize('The local voice service queue has one job.',args.output/'response.wav')
    record('controlled_assistant',t,status='passed',queue=status['queue_counts'],diagnostic=result['result'],replayed=repeated['replayed'],response_sha256=digest(response.output))
    t=time.perf_counter();silence=write_mono_sine(args.output/'silence.wav',duration_seconds=1,amplitude=0)
    try:archive.ingest(silence,AudioEnvelope(request_id='req_voiceprojects002',source_kind='licensed_fixture',media_type='audio/wav',sample_rate_hz=16000,channels=1,captured_at=datetime.now(timezone.utc),consent_basis='licensed_fixture',retention_class='project_archive',content_sha256=digest(silence)),vad,asr)
    except ContractError as exc:
        if exc.code!='WORKFLOW_NO_SPEECH':raise
        record('silence_rejected_before_archive',t,status='passed',code=exc.code)
    else:raise AssertionError('Silence was accepted')
    print('All real CPU project checks passed.',flush=True)
if __name__=='__main__':main()
