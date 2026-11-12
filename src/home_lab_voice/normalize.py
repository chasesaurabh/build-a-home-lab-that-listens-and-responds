"""Explicit file normalization for permitted WAV/FLAC, with preserved input hashes."""
import hashlib
import os
from pathlib import Path
import tempfile
import wave
from .contracts import ContractError

def normalize_audio(source:Path,output:Path,*,allow_downmix=False):
    source=source.resolve(strict=True);output=output.absolute()
    if source==output.resolve() or output.exists():raise ContractError('NORMALIZE_OUTPUT_EXISTS','Choose a new output; original audio is preserved.')
    if source.suffix.casefold() not in ('.wav','.flac') or not 0<source.stat().st_size<=64*1024*1024:
        raise ContractError('NORMALIZE_INPUT_LIMIT','Use one permitted WAV/FLAC of at most 64 MiB.')
    if output.suffix.casefold()!='.wav':raise ContractError('NORMALIZE_OUTPUT_FORMAT','Output must be a WAV file.')
    try:import av
    except ImportError as exc:raise ContractError('NORMALIZE_DEPENDENCY','Install the ASR dependencies, including PyAV.') from exc
    # Caller chooses the parent explicitly; no recursive writes to an arbitrary hierarchy.
    if not output.parent.is_dir():raise ContractError('NORMALIZE_PARENT_MISSING','Create the intended output directory first.')
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    temporary=None;sample_count=0;input_rate=None;channels=None
    try:
        with av.open(str(source)) as container:
            if container.format.name not in ('wav','flac') or len(container.streams.audio)!=1:
                raise ContractError('NORMALIZE_CONTAINER','Expected exactly one WAV/FLAC audio stream.')
            stream=container.streams.audio[0]
            channels=stream.codec_context.channels;input_rate=stream.codec_context.sample_rate
            if channels not in (1,2) or (channels==2 and not allow_downmix):
                raise ContractError('NORMALIZE_CHANNELS','Stereo requires explicit --allow-downmix; other channel layouts are unsupported.')
            resampler=av.AudioResampler(format='s16',layout='mono',rate=16000)
            fd,name=tempfile.mkstemp(prefix='.normalize-',suffix='.wav',dir=output.parent);os.close(fd);temporary=Path(name)
            with wave.open(str(temporary),'wb') as writer:
                writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(16000)
                def emit(frames):
                    nonlocal sample_count
                    for frame in frames:
                        sample_count+=frame.samples
                        if sample_count>16000*600:raise ContractError('NORMALIZE_DURATION','Decoded audio exceeds ten minutes.')
                        writer.writeframes(frame.to_ndarray().astype('<i2',copy=False).tobytes())
                for frame in container.decode(stream):emit(resampler.resample(frame))
                emit(resampler.resample(None))
        if sample_count==0:raise ContractError('NORMALIZE_EMPTY','No decoded samples.')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=before:
            raise ContractError('NORMALIZE_SOURCE_CHANGED','Source changed during conversion.')
        # Exclusive final creation also rejects a destination that appeared during decoding.
        created=False
        try:
            with output.open('xb') as dest,temporary.open('rb') as origin:
                created=True
                for block in iter(lambda:origin.read(1024*1024),b''):dest.write(block)
        except OSError:
            if created:output.unlink(missing_ok=True)
            raise
        return {'source_sha256':before,'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'input_rate_hz':input_rate,'input_channels':channels,'output_rate_hz':16000,'output_channels':1,
            'output_pcm_bits':16,'samples':sample_count,'duration_seconds':sample_count/16000,
            'method':'PyAV/FFmpeg resampler, s16 mono; explicit stereo rematrix when requested.',
            'av_version':av.__version__}
    except av.error.FFmpegError as exc:
        raise ContractError('NORMALIZE_DECODE_FAILED','The permitted audio file could not be decoded.') from exc
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
