"""
Shared utilities for the Skillfy Feature 3 dubbing scripts: audio extraction,
transcription, translation, timing/timeline composition, and video remuxing.

Used by both the recommended local pipeline (timed_chatterbox_dub_poc.py,
kokoro_dub_poc.py, timed_kokoro_dub_poc.py) and the experiments/ scripts
(IndicF5, Indic Parler-TTS). See RESEARCH_LOCAL_PIPELINE.md for background on
why each model/tool was chosen.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

# Windows consoles default to a codepage (e.g. cp1252) that can't represent
# Devanagari/Tamil/etc. script - force UTF-8 so printing translated text doesn't crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

LANGUAGES = {
    "hindi": "hin_Deva",
    "tamil": "tam_Taml",
    "telugu": "tel_Telu",
    "bengali": "ben_Beng",
    "marathi": "mar_Deva",
    "gujarati": "guj_Gujr",
    "kannada": "kan_Knda",
    "malayalam": "mal_Mlym",
    "punjabi": "pan_Guru",
    "odia": "ory_Orya",
    "urdu": "urd_Arab",
}

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
WHISPER_MODEL = "small"
FINAL_SAMPLE_RATE = 24000
FADE_MS = 30


def resolve_ffmpeg():
    env_path = os.getenv("FFMPEG_PATH")
    if env_path:
        ffmpeg_path = Path(env_path)
        if ffmpeg_path.exists():
            return str(ffmpeg_path)
        print(f"FFMPEG_PATH is set but does not exist: {ffmpeg_path}")
        sys.exit(1)

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        print(
            "'ffmpeg' not found on PATH. Install ffmpeg, set FFMPEG_PATH to ffmpeg.exe, "
            "or install the local extra dependency 'imageio-ffmpeg'."
        )
        sys.exit(1)


def resolve_target_lang(user_value):
    value = user_value.strip().lower()
    if value in LANGUAGES.values():
        return value
    if value in LANGUAGES:
        return LANGUAGES[value]
    print(f"Invalid --target '{user_value}'. Valid options (name or code):")
    for name, code in LANGUAGES.items():
        print(f"  {name} ({code})")
    sys.exit(1)


def check_binary(name, purpose):
    if shutil.which(name) is None:
        print(f"'{name}' not found on PATH. It's required for {purpose}.")
        sys.exit(1)


def extract_audio(video_path, wav_path):
    print("Extracting audio from video...")
    ffmpeg = resolve_ffmpeg()
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(video_path), "-ar", "16000", "-ac", "1", str(wav_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed to extract audio:\n{e.stderr}")
        sys.exit(1)


def audio_duration_seconds(wav_path):
    info = sf.info(str(wav_path))
    return info.frames / float(info.samplerate)


def transcribe(wav_path):
    """Transcribe a whole clip in one shot, returning (full_text, segments)."""
    print(f"Transcribing audio (faster-whisper, model={WHISPER_MODEL})...")
    from faster_whisper import WhisperModel

    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    raw_segments, _ = model.transcribe(str(wav_path), word_timestamps=False)
    segments = [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in raw_segments
        if s.text.strip()
    ]
    if not segments:
        print("Transcription produced no text - check the input audio has clear speech.")
        sys.exit(1)
    text = " ".join(s["text"] for s in segments)
    print(f"  Transcript: {text}")
    return text, segments


def transcribe_segments(wav_path):
    """Transcribe a clip into indexed, timestamped segments (for segment-by-segment
    translation/synthesis + timeline placement)."""
    print(f"Transcribing audio with timestamps (faster-whisper, model={WHISPER_MODEL})...")
    from faster_whisper import WhisperModel

    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    whisper_segments, _ = model.transcribe(str(wav_path), word_timestamps=False)

    segments = []
    for index, segment in enumerate(whisper_segments):
        text = segment.text.strip()
        if not text:
            continue
        segments.append(
            {
                "index": index,
                "start": float(segment.start),
                "end": float(segment.end),
                "source_text": text,
            }
        )

    if not segments:
        print("Transcription produced no speech segments - check the input audio has clear speech.")
        sys.exit(1)

    print(f"  Found {len(segments)} timed speech segments.")
    return segments


def full_source_transcript(segments):
    return " ".join(segment["source_text"] for segment in segments).strip()


def pick_reference_segment(segments, min_duration=2.0, max_duration=8.0):
    """Pick a short, single-utterance segment to use as the voice-cloning reference,
    rather than the whole (likely multi-speaker/noisy) source track."""
    in_range = [s for s in segments if min_duration <= (s["end"] - s["start"]) <= max_duration]
    if in_range:
        return max(in_range, key=lambda s: s["end"] - s["start"])
    return max(segments, key=lambda s: s["end"] - s["start"])


def slice_reference_clip(source_wav, segment, out_path, max_duration=8.0):
    data, sample_rate = sf.read(str(source_wav), dtype="float32", always_2d=True)
    start_sample = max(0, int(segment["start"] * sample_rate))
    end_sample = min(len(data), int(min(segment["end"], segment["start"] + max_duration) * sample_rate))
    sf.write(str(out_path), data[start_sample:end_sample], sample_rate)
    return out_path


def translate(text, target_lang):
    """Translate a whole block of text in one NLLB-200 call."""
    print(f"Translating to {target_lang} (NLLB-200)...")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        max_length=256,
    )
    translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"  Translated: {translated}")
    return translated


def load_translation_model():
    print(f"Loading translation model ({NLLB_MODEL})...")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    return tokenizer, model


def translate_segments(segments, target_lang, tokenizer, model):
    """Translate each segment's source_text individually via NLLB-200, storing
    the result in segment['translated_text']."""
    print(f"Translating {len(segments)} segments to {target_lang}...")
    bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    for segment in segments:
        inputs = tokenizer(segment["source_text"], return_tensors="pt")
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=bos_token_id,
            max_length=160,
        )
        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        segment["translated_text"] = translated
        print(f"  [{segment['index']:04d}] {segment['start']:.2f}-{segment['end']:.2f}s: {translated}")

    return segments


def write_segments_json(path, segments):
    with path.open("w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def load_wav_mono(path):
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return mono, sample_rate


def write_wav_mono(path, audio, sample_rate=FINAL_SAMPLE_RATE):
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate)


def fade_edges(audio, sample_rate):
    if len(audio) == 0:
        return audio
    fade_samples = min(int(sample_rate * FADE_MS / 1000), len(audio) // 2)
    if fade_samples <= 0:
        return audio
    faded = audio.copy()
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    faded[:fade_samples] *= fade_in
    faded[-fade_samples:] *= fade_out
    return faded


def compose_timed_audio(segments, total_duration, out_wav_path, sample_rate=FINAL_SAMPLE_RATE):
    """Place each segment's synthesized audio (segment['segment_wav']) onto a
    silent timeline at its original start/end time, trimming/fading as needed."""
    print("Composing timed dubbed audio track...")
    total_samples = max(1, int(np.ceil(total_duration * sample_rate)))
    timeline = np.zeros(total_samples, dtype=np.float32)

    for segment in segments:
        audio, audio_sample_rate = load_wav_mono(segment["segment_wav"])
        if audio_sample_rate != sample_rate:
            print(f"Unexpected sample rate in {segment['segment_wav']}: {audio_sample_rate}")
            sys.exit(1)

        start_sample = max(0, int(round(segment["start"] * sample_rate)))
        segment_duration = max(0.0, segment["end"] - segment["start"])
        allowed_samples = max(1, int(round(segment_duration * sample_rate)))
        available_samples = max(0, total_samples - start_sample)
        placed_samples = min(len(audio), allowed_samples, available_samples)

        if placed_samples <= 0:
            segment["timing_action"] = "skipped"
            segment["placed_duration"] = 0.0
            continue

        action = "placed"
        if len(audio) > allowed_samples:
            action = "trimmed"
            audio = fade_edges(audio[:placed_samples], sample_rate)

        end_sample = start_sample + placed_samples
        timeline[start_sample:end_sample] += audio[:placed_samples]
        segment["timing_action"] = action
        segment["placed_duration"] = placed_samples / float(sample_rate)

    timeline = np.clip(timeline, -1.0, 1.0)
    write_wav_mono(out_wav_path, timeline, sample_rate)
    return segments


def remux(video_path, new_audio_path, out_path):
    print("Remuxing dubbed audio onto video...")
    ffmpeg = resolve_ffmpeg()
    try:
        subprocess.run(
            [
                ffmpeg, "-y",
                "-i", str(video_path),
                "-i", str(new_audio_path),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac",
                "-shortest",
                str(out_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed to remux video:\n{e.stderr}")
        sys.exit(1)
    print(f"Saved dubbed output to: {out_path}")
