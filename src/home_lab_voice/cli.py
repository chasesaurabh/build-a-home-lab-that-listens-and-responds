from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence, TextIO

from .applications.archive import ManagedArchive
from .applications.assistant import ControlledAssistant, Session
from .applications.dictation import ManagedTextDestination
from .applications.narration import plan_narration
from .applications.workflows import (
    NarrationProject,
    ReviewedDictation,
    SearchableAudioArchive,
    interpret_assistant_command,
    transcribe_speech,
)
from .audio_fixtures import inspect_pcm16_wav, write_mono_sine
from .contracts import AudioEnvelope, ContractError, IntentEnvelope
from .devices import AudioBackend, capture_pcm16_wav, list_audio_devices, play_pcm16_wav, select_audio_device
from .runtimes import FasterWhisperRuntime, KokoroRuntime, SileroVadRuntime


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return value


def _write_json(stream: TextIO, value: Any) -> None:
    stream.write(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _intent(name: str, arguments: dict[str, str], request_id: str) -> IntentEnvelope:
    return IntentEnvelope.from_dict(
        {
            "schema_version": "intent.v1",
            "request_id": request_id,
            "disposition": "accept",
            "intent": name,
            "arguments": arguments,
            "evidence": [{"source": "typed_cli"}],
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="home-lab-voice",
        description="Run safe companion workflows; live commands require explicit optional local dependencies.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fixture = commands.add_parser("fixture", help="Create a deterministic PCM WAV fixture.")
    fixture.add_argument("output", type=Path)
    fixture.add_argument("--duration", type=float, default=1.0)

    devices = commands.add_parser("devices", help="List locally visible capture and playback devices.")
    devices.add_argument("--direction", choices=("all", "input", "output"), default="all")
    devices.add_argument("--select", dest="selector")

    capture = commands.add_parser("capture", help="Capture a bounded consented PCM WAV from one device.")
    capture.add_argument("output", type=Path)
    capture.add_argument("--device")
    capture.add_argument("--duration", type=float, default=3.0)
    capture.add_argument("--sample-rate", type=int, default=16_000)
    capture.add_argument("--channels", type=int, choices=(1, 2), default=1)
    capture.add_argument("--consent-basis", default="self_recorded")
    capture.add_argument("--retention-class", default="ephemeral")
    capture.add_argument("--overwrite", action="store_true")

    playback = commands.add_parser("play", help="Play one bounded local PCM WAV on an explicit output device.")
    playback.add_argument("source", type=Path)
    playback.add_argument("--device", required=True)

    vad = commands.add_parser("vad", help="Detect speech regions in a bounded local PCM WAV file.")
    vad.add_argument("source", type=Path)
    vad.add_argument("--model-file", type=Path)
    vad.add_argument("--threshold", type=float, default=0.5)
    vad.add_argument("--min-speech-ms", type=int, default=250)
    vad.add_argument("--min-silence-ms", type=int, default=100)
    vad.add_argument("--speech-pad-ms", type=int, default=30)

    transcribe = commands.add_parser("transcribe", help="Transcribe a bounded local WAV on CPU/int8.")
    transcribe.add_argument("source", type=Path)
    transcribe.add_argument("--model-dir", type=Path, required=True)
    transcribe.add_argument("--language", default="en")
    transcribe.add_argument("--beam-size", type=int, default=5)
    transcribe.add_argument("--initial-prompt")

    synthesize = commands.add_parser("synthesize", help="Synthesize bounded text with a local Kokoro snapshot.")
    synthesize.add_argument("output", type=Path)
    synthesize.add_argument("--model-dir", type=Path, required=True)
    text_source = synthesize.add_mutually_exclusive_group(required=True)
    text_source.add_argument("--text")
    text_source.add_argument("--text-file", type=Path)
    synthesize.add_argument("--voice", default="af_heart")
    synthesize.add_argument("--language-code", default="a")
    synthesize.add_argument("--speed", type=float, default=1.0)
    synthesize.add_argument("--overwrite", action="store_true")

    listen = commands.add_parser("listen", help="Capture, detect speech, and transcribe one bounded local turn.")
    listen.add_argument("recording", type=Path)
    listen.add_argument("--device", required=True)
    listen.add_argument("--duration", type=float, default=8.0)
    listen.add_argument("--model-dir", type=Path, required=True)
    listen.add_argument("--vad-model-file", type=Path)
    listen.add_argument("--language", default="en")
    listen.add_argument("--beam-size", type=int, default=5)
    listen.add_argument("--consent-basis", default="self_recorded")
    listen.add_argument("--retention-class", default="ephemeral")
    listen.add_argument("--overwrite", action="store_true")

    archive = commands.add_parser("archive", help="Validate and import a permitted PCM WAV file.")
    archive.add_argument("source", type=Path)
    archive.add_argument("--root", type=Path, required=True)
    archive.add_argument("--request-id", default="req_cliarchive01")
    archive.add_argument("--consent-basis", default="licensed_fixture")
    archive.add_argument("--retention-class", default="project_archive")

    dictation = commands.add_parser("dictation", help="Commit reviewed text to an approved destination.")
    dictation.add_argument("--root", type=Path, required=True)
    dictation.add_argument("--target", required=True)
    source = dictation.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    dictation.add_argument("--idempotency-key", default="idem_cliwrite01")

    narration = commands.add_parser("narration", help="Plan stable narration chunks from permitted text.")
    narration.add_argument("source", type=Path)
    narration.add_argument("--output", type=Path)
    narration.add_argument("--max-chars", type=int, default=500)

    assistant = commands.add_parser("assistant", help="Run a closed, typed assistant action.")
    assistant.add_argument("mode", choices=("status", "diagnostic"))
    assistant.add_argument("--service", choices=("media-indexer", "voice-service"), required=True)
    assistant.add_argument("--diagnostic", choices=("healthcheck", "queue_snapshot"))
    assistant.add_argument("--session-id", default="session-cli-01")
    assistant.add_argument("--request-id", default="req_cliassist01")
    assistant.add_argument("--idempotency-key", default="idem_cliassist01")
    assistant.add_argument("--confirm")

    archive_workflow = commands.add_parser(
        "archive-workflow", help="Transcribe and store permitted audio in a searchable managed archive."
    )
    archive_workflow.add_argument("source", type=Path)
    archive_workflow.add_argument("--root", type=Path, required=True)
    archive_workflow.add_argument("--model-dir", type=Path, required=True)
    archive_workflow.add_argument("--vad-model-file", type=Path)
    archive_workflow.add_argument("--request-id", default="req_cliarchive02")
    archive_workflow.add_argument("--consent-basis", default="licensed_fixture")
    archive_workflow.add_argument("--retention-class", default="project_archive")

    archive_search = commands.add_parser("archive-search", help="Search managed archive transcripts locally.")
    archive_search.add_argument("query")
    archive_search.add_argument("--root", type=Path, required=True)

    archive_delete = commands.add_parser("archive-delete", help="Delete managed audio and its transcript record.")
    archive_delete.add_argument("source_digest")
    archive_delete.add_argument("--root", type=Path, required=True)

    dictation_preview = commands.add_parser(
        "dictation-preview", help="Transcribe audio into a review record without changing the destination."
    )
    dictation_preview.add_argument("source", type=Path)
    dictation_preview.add_argument("--root", type=Path, required=True)
    dictation_preview.add_argument("--model-dir", type=Path, required=True)
    dictation_preview.add_argument("--vad-model-file", type=Path)

    dictation_commit = commands.add_parser(
        "dictation-commit", help="Commit one reviewed preview after exact digest approval."
    )
    dictation_commit.add_argument("--root", type=Path, required=True)
    dictation_commit.add_argument("--preview-id", required=True)
    dictation_commit.add_argument("--approve", required=True)
    dictation_commit.add_argument("--target", required=True)
    dictation_commit.add_argument("--idempotency-key", default="idem_cliwrite02")

    dictation_undo = commands.add_parser("dictation-undo", help="Undo an unchanged managed dictation commit.")
    dictation_undo.add_argument("--root", type=Path, required=True)
    dictation_undo.add_argument("--idempotency-key", required=True)

    narration_build = commands.add_parser(
        "narration-build", help="Synthesize and assemble a reviewed narration project with a manifest."
    )
    narration_build.add_argument("source", type=Path)
    narration_build.add_argument("--root", type=Path, required=True)
    narration_build.add_argument("--model-dir", type=Path, required=True)
    narration_build.add_argument("--voice", default="af_heart")
    narration_build.add_argument("--language-code", default="a")
    narration_build.add_argument("--speed", type=float, default=1.0)
    narration_build.add_argument("--max-chars", type=int, default=500)
    narration_build.add_argument("--overwrite", action="store_true")

    assistant_turn = commands.add_parser(
        "assistant-turn", help="Interpret typed or transcribed input through a closed assistant grammar."
    )
    assistant_input = assistant_turn.add_mutually_exclusive_group(required=True)
    assistant_input.add_argument("--text")
    assistant_input.add_argument("--audio", type=Path)
    assistant_turn.add_argument("--asr-model-dir", type=Path)
    assistant_turn.add_argument("--vad-model-file", type=Path)
    assistant_turn.add_argument("--session-id", default="session-cli-turn-01")
    assistant_turn.add_argument("--request-id", default="req_cliassist02")
    assistant_turn.add_argument("--idempotency-key", default="idem_cliassist02")
    assistant_turn.add_argument("--confirm")
    assistant_turn.add_argument("--tts-model-dir", type=Path)
    assistant_turn.add_argument("--speak-output", type=Path)
    assistant_turn.add_argument("--voice", default="af_heart")
    assistant_turn.add_argument("--language-code", default="a")
    assistant_turn.add_argument("--play-device")
    return parser


def run(
    args: argparse.Namespace,
    stdout: TextIO,
    *,
    audio_backend: AudioBackend | None = None,
    vad_runtime: Any | None = None,
    asr_runtime_factory: Callable[[Path], Any] | None = None,
    tts_runtime_factory: Callable[[Path, str, str], Any] | None = None,
) -> int:
    if args.command == "fixture":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        path = write_mono_sine(args.output, duration_seconds=args.duration)
        _write_json(stdout, inspect_pcm16_wav(path))
        return 0

    if args.command == "devices":
        devices = list_audio_devices(audio_backend)
        if args.direction != "all":
            devices = tuple(
                device
                for device in devices
                if (device.input_channels if args.direction == "input" else device.output_channels) > 0
            )
        if args.selector:
            if args.direction == "all":
                raise ContractError("AUDIO_DIRECTION_REQUIRED", "Device selection requires input or output direction.")
            devices = (
                select_audio_device(devices, direction=args.direction, selector=args.selector),
            )
        _write_json(stdout, {"devices": devices})
        return 0

    if args.command == "capture":
        result = capture_pcm16_wav(
            args.output,
            device_selector=args.device,
            duration_seconds=args.duration,
            sample_rate_hz=args.sample_rate,
            channels=args.channels,
            consent_basis=args.consent_basis,
            retention_class=args.retention_class,
            overwrite=args.overwrite,
            backend=audio_backend,
        )
        _write_json(stdout, result)
        return 0

    if args.command == "play":
        result = play_pcm16_wav(args.source, device_selector=args.device, backend=audio_backend)
        _write_json(stdout, result)
        return 0

    if args.command == "vad":
        runtime = vad_runtime or SileroVadRuntime(model_path=args.model_file)
        result = runtime.analyze(
            args.source,
            threshold=args.threshold,
            min_speech_duration_ms=args.min_speech_ms,
            min_silence_duration_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
        )
        _write_json(stdout, result)
        return 0

    if args.command == "transcribe":
        runtime = (
            asr_runtime_factory(args.model_dir)
            if asr_runtime_factory is not None
            else FasterWhisperRuntime(args.model_dir)
        )
        result = runtime.transcribe(
            args.source,
            language=args.language,
            beam_size=args.beam_size,
            initial_prompt=args.initial_prompt,
        )
        _write_json(stdout, result)
        return 0

    if args.command == "synthesize":
        text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
        runtime = (
            tts_runtime_factory(args.model_dir, args.voice, args.language_code)
            if tts_runtime_factory is not None
            else KokoroRuntime(args.model_dir, voice=args.voice, language_code=args.language_code)
        )
        result = runtime.synthesize(text, args.output, speed=args.speed, overwrite=args.overwrite)
        _write_json(stdout, result)
        return 0

    if args.command == "listen":
        capture = capture_pcm16_wav(
            args.recording,
            device_selector=args.device,
            duration_seconds=args.duration,
            sample_rate_hz=16_000,
            channels=1,
            consent_basis=args.consent_basis,
            retention_class=args.retention_class,
            overwrite=args.overwrite,
            backend=audio_backend,
        )
        detector = vad_runtime or SileroVadRuntime(model_path=args.vad_model_file)
        vad_result = detector.analyze(args.recording)
        if not vad_result.regions:
            _write_json(stdout, {"status": "no_speech", "capture": capture, "vad": vad_result})
            return 4
        recognizer = (
            asr_runtime_factory(args.model_dir)
            if asr_runtime_factory is not None
            else FasterWhisperRuntime(args.model_dir)
        )
        transcription = recognizer.transcribe(
            args.recording,
            language=args.language,
            beam_size=args.beam_size,
        )
        _write_json(
            stdout,
            {
                "status": "transcribed",
                "capture": capture,
                "vad": vad_result,
                "transcription": transcription,
            },
        )
        return 0

    if args.command == "archive":
        wave = inspect_pcm16_wav(args.source)
        envelope = AudioEnvelope(
            request_id=args.request_id,
            source_kind="licensed_fixture" if args.consent_basis == "licensed_fixture" else "permitted_file",
            media_type="audio/wav",
            sample_rate_hz=wave.sample_rate_hz,
            channels=wave.channels,
            captured_at=datetime.now(timezone.utc),
            consent_basis=args.consent_basis,
            retention_class=args.retention_class,
            content_sha256=wave.file_sha256,
        )
        record = ManagedArchive(args.root).import_wav(args.source, envelope)
        _write_json(stdout, record)
        return 0

    if args.command == "dictation":
        text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
        token = ManagedTextDestination(args.root).commit(args.target, text, args.idempotency_key)
        _write_json(
            stdout,
            {
                "status": "committed",
                "target": token.target,
                "before_sha256": token.before_digest,
                "after_sha256": token.after_digest,
                "idempotency_key": token.idempotency_key,
            },
        )
        return 0

    if args.command == "narration":
        text = args.source.read_text(encoding="utf-8")
        result = {
            "source": args.source.resolve(),
            "source_characters": len(text),
            "chunks": plan_narration(text, max_chars=args.max_chars),
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_json(stdout, result)
        return 0

    if args.command == "assistant":
        now = datetime.now(timezone.utc)
        session = Session(args.session_id, "operator", now + timedelta(minutes=5))

        def status_provider(service: str) -> dict[str, str]:
            return {"service": service, "state": "demo-ready", "source": "safe-reference-provider"}

        def diagnostic_provider(service: str, diagnostic: str) -> dict[str, str]:
            return {
                "service": service,
                "diagnostic": diagnostic,
                "state": "demo-completed",
                "source": "safe-reference-provider",
            }

        controlled = ControlledAssistant(status_provider, diagnostic_provider)
        if args.mode == "status":
            result = controlled.read_status(
                _intent("lab.status.read", {"service": args.service}, args.request_id), session, now
            )
            _write_json(stdout, result)
            return 0
        if not args.diagnostic:
            raise ContractError("TOOL_DIAGNOSTIC_REQUIRED", "Diagnostic mode requires --diagnostic.")
        pending = controlled.prepare_diagnostic(
            _intent(
                "lab.diagnostic.run",
                {"service": args.service, "diagnostic": args.diagnostic},
                args.request_id,
            ),
            session,
            now,
            args.idempotency_key,
        )
        if args.confirm != pending.operation_digest:
            _write_json(
                stdout,
                {
                    "status": "confirmation_required",
                    "operation_digest": pending.operation_digest,
                    "operation": pending.arguments,
                    "instruction": "Review the operation, then repeat the command with --confirm and this exact digest.",
                },
            )
            return 3
        result = controlled.confirm_and_run(pending, session, now, args.confirm)
        _write_json(stdout, result)
        return 0

    if args.command == "archive-workflow":
        wave = inspect_pcm16_wav(args.source)
        envelope = AudioEnvelope(
            request_id=args.request_id,
            source_kind="licensed_fixture" if args.consent_basis == "licensed_fixture" else "permitted_file",
            media_type="audio/wav",
            sample_rate_hz=wave.sample_rate_hz,
            channels=wave.channels,
            captured_at=datetime.now(timezone.utc),
            consent_basis=args.consent_basis,
            retention_class=args.retention_class,
            content_sha256=wave.file_sha256,
        )
        detector = vad_runtime or SileroVadRuntime(model_path=args.vad_model_file)
        recognizer = asr_runtime_factory(args.model_dir) if asr_runtime_factory else FasterWhisperRuntime(args.model_dir)
        result = SearchableAudioArchive(args.root).ingest(args.source, envelope, detector, recognizer)
        _write_json(stdout, {"status": "archived", "result": result})
        return 0

    if args.command == "archive-search":
        matches = SearchableAudioArchive(args.root).search(args.query)
        _write_json(stdout, {"status": "searched", "query": args.query, "match_count": len(matches), "matches": matches})
        return 0

    if args.command == "archive-delete":
        deleted = SearchableAudioArchive(args.root).delete(args.source_digest)
        _write_json(stdout, {"status": "deleted" if deleted else "not_found", "source_digest": args.source_digest})
        return 0

    if args.command == "dictation-preview":
        detector = vad_runtime or SileroVadRuntime(model_path=args.vad_model_file)
        recognizer = asr_runtime_factory(args.model_dir) if asr_runtime_factory else FasterWhisperRuntime(args.model_dir)
        preview = ReviewedDictation(args.root).preview(args.source, detector, recognizer)
        _write_json(
            stdout,
            {
                "status": "review_required",
                "preview": preview,
                "instruction": "Review the text, then use dictation-commit with this preview ID and exact approval digest.",
            },
        )
        return 3

    if args.command == "dictation-commit":
        token = ReviewedDictation(args.root).commit(
            args.preview_id, args.approve, args.target, args.idempotency_key
        )
        _write_json(
            stdout,
            {
                "status": "committed",
                "target": token.target,
                "before_sha256": token.before_digest,
                "after_sha256": token.after_digest,
                "idempotency_key": token.idempotency_key,
            },
        )
        return 0

    if args.command == "dictation-undo":
        target = ReviewedDictation(args.root).undo(args.idempotency_key)
        _write_json(stdout, {"status": "undone", "target": target})
        return 0

    if args.command == "narration-build":
        text = args.source.read_text(encoding="utf-8")
        runtime = (
            tts_runtime_factory(args.model_dir, args.voice, args.language_code)
            if tts_runtime_factory
            else KokoroRuntime(args.model_dir, voice=args.voice, language_code=args.language_code)
        )
        result = NarrationProject(args.root).build(
            text, runtime, max_chars=args.max_chars, speed=args.speed, overwrite=args.overwrite
        )
        _write_json(stdout, {"status": "built", "result": result})
        return 0

    if args.command == "assistant-turn":
        transcript = None
        if args.audio is not None:
            if args.asr_model_dir is None:
                raise ContractError("ASR_MODEL_REQUIRED", "Audio assistant input requires --asr-model-dir.")
            detector = vad_runtime or SileroVadRuntime(model_path=args.vad_model_file)
            recognizer = (
                asr_runtime_factory(args.asr_model_dir)
                if asr_runtime_factory
                else FasterWhisperRuntime(args.asr_model_dir)
            )
            _, transcript = transcribe_speech(args.audio, detector, recognizer)
            command_text = transcript.text
        else:
            command_text = args.text
        assert command_text is not None
        mode, arguments = interpret_assistant_command(command_text)
        now = datetime.now(timezone.utc)
        session = Session(args.session_id, "operator", now + timedelta(minutes=5))

        def status_provider(service: str) -> dict[str, str]:
            return {"service": service, "state": "demo-ready", "source": "safe-reference-provider"}

        def diagnostic_provider(service: str, diagnostic: str) -> dict[str, str]:
            return {"service": service, "diagnostic": diagnostic, "state": "demo-completed", "source": "safe-reference-provider"}

        controlled = ControlledAssistant(status_provider, diagnostic_provider)
        if mode == "status":
            action = controlled.read_status(_intent("lab.status.read", arguments, args.request_id), session, now)
            response_text = f"{arguments['service']} status is {action.payload['state']}."
        else:
            pending = controlled.prepare_diagnostic(
                _intent("lab.diagnostic.run", arguments, args.request_id), session, now, args.idempotency_key
            )
            if args.confirm != pending.operation_digest:
                _write_json(
                    stdout,
                    {
                        "status": "confirmation_required",
                        "input_text": command_text,
                        "transcription": transcript,
                        "operation_digest": pending.operation_digest,
                        "operation": pending.arguments,
                    },
                )
                return 3
            action = controlled.confirm_and_run(pending, session, now, args.confirm)
            response_text = f"{arguments['diagnostic']} for {arguments['service']} {action.payload['state']}."
        speech = None
        playback = None
        if args.speak_output is not None:
            if args.tts_model_dir is None:
                raise ContractError("TTS_MODEL_REQUIRED", "Spoken assistant output requires --tts-model-dir.")
            runtime = (
                tts_runtime_factory(args.tts_model_dir, args.voice, args.language_code)
                if tts_runtime_factory
                else KokoroRuntime(args.tts_model_dir, voice=args.voice, language_code=args.language_code)
            )
            speech = runtime.synthesize(response_text, args.speak_output)
            if args.play_device is not None:
                playback = play_pcm16_wav(args.speak_output, device_selector=args.play_device, backend=audio_backend)
        elif args.play_device is not None:
            raise ContractError("ASSISTANT_SPEECH_REQUIRED", "Playback requires --speak-output.")
        _write_json(
            stdout,
            {
                "status": "completed",
                "input_text": command_text,
                "transcription": transcript,
                "action": action,
                "response_text": response_text,
                "speech": speech,
                "playback": playback,
            },
        )
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    audio_backend: AudioBackend | None = None,
    vad_runtime: Any | None = None,
    asr_runtime_factory: Callable[[Path], Any] | None = None,
    tts_runtime_factory: Callable[[Path, str, str], Any] | None = None,
) -> int:
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    try:
        return run(
            build_parser().parse_args(argv),
            output,
            audio_backend=audio_backend,
            vad_runtime=vad_runtime,
            asr_runtime_factory=asr_runtime_factory,
            tts_runtime_factory=tts_runtime_factory,
        )
    except (ContractError, FileNotFoundError, PermissionError, UnicodeError, ValueError) as error:
        code = error.code if isinstance(error, ContractError) else type(error).__name__.upper()
        _write_json(errors, {"status": "error", "code": code, "message": str(error)})
        return 2


def entrypoint() -> None:
    raise SystemExit(main())
