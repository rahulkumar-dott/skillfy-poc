"""
Skillfy Feature 3 POC: timestamp-based local video dubbing using Kokoro-82M.

Same segment-timestamp approach as timed_chatterbox_dub_poc.py (transcribe
with timestamps, translate each segment, synthesize each one, place on a
silent timeline at its original start/end time) but uses Kokoro-82M for
synthesis instead of cloning.

Kokoro-82M does NOT clone the source speaker's voice - it uses a fixed preset
voice (hf_alpha). Traded off deliberately for speed: Kokoro runs in seconds per
segment on CPU, vs. Chatterbox's ~30-40s/segment or IndicF5's ~10 minutes per
sentence. Kokoro also only supports Hindi among our 11 target languages, so
this script is Hindi-only.
"""

import sys
from pathlib import Path

import numpy as np

from common import (
    audio_duration_seconds,
    compose_timed_audio,
    extract_audio,
    fade_edges,
    load_translation_model,
    remux,
    resolve_target_lang,
    transcribe_segments,
    translate_segments,
    write_segments_json,
    write_wav_mono,
)

KOKORO_LANG_CODE = "h"
KOKORO_VOICE = "hf_alpha"
FINAL_SAMPLE_RATE = 24000


def synthesize_segments(segments, segment_dir):
    print(f"Loading Kokoro-82M once for segment synthesis (voice={KOKORO_VOICE})...")
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=KOKORO_LANG_CODE)
    segment_dir.mkdir(parents=True, exist_ok=True)

    for segment in segments:
        out_path = segment_dir / f"segment_{segment['index']:04d}.wav"
        text = segment["translated_text"]
        print(f"Synthesizing segment {segment['index']:04d}...")

        chunks = [audio for _, _, audio in pipeline(text, voice=KOKORO_VOICE)]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        audio = fade_edges(audio, FINAL_SAMPLE_RATE)
        write_wav_mono(out_path, audio, FINAL_SAMPLE_RATE)

        segment["segment_wav"] = str(out_path)
        segment["generated_duration"] = len(audio) / float(FINAL_SAMPLE_RATE)

    return segments


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Skillfy Feature 3 POC - timestamp-based local dubbing with Kokoro-82M (Hindi only, no voice cloning)."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to local input video file")
    parser.add_argument("--target", "-t", required=True, help="Must be 'hindi' - Kokoro only supports Hindi here")
    parser.add_argument("--output", "-o", default=None, help="Output video path (default: derived)")
    parser.add_argument("--work-dir", default=None, help="Directory for dubbing artifacts")
    args = parser.parse_args()

    if args.target.strip().lower() not in ("hindi", "hin_deva"):
        print("Kokoro-82M only supports Hindi among our 11 target languages. Use --target hindi.")
        sys.exit(1)
    target_lang = resolve_target_lang(args.target)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    work_dir = Path(args.work_dir) if args.work_dir else input_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    source_wav = work_dir / f"{input_path.stem}_source.wav"
    segments_json = work_dir / f"{input_path.stem}_hindi_kokoro_segments.json"
    segment_dir = work_dir / f"{input_path.stem}_hindi_kokoro_segments"
    timed_dubbed_wav = work_dir / f"{input_path.stem}_hindi_kokoro_timed_dubbed.wav"
    out_path = Path(args.output) if args.output else work_dir / f"{input_path.stem}_hindi_kokoro_timed.mp4"

    extract_audio(input_path, source_wav)
    total_duration = audio_duration_seconds(source_wav)

    segments = transcribe_segments(source_wav)
    tokenizer, translation_model = load_translation_model()
    segments = translate_segments(segments, target_lang, tokenizer, translation_model)
    write_segments_json(segments_json, segments)

    segments = synthesize_segments(segments, segment_dir)
    segments = compose_timed_audio(segments, total_duration, timed_dubbed_wav)
    write_segments_json(segments_json, segments)

    remux(input_path, timed_dubbed_wav, out_path)

    print("Saved timestamped Kokoro artifacts:")
    print(f"  Segment JSON: {segments_json}")
    print(f"  Segment WAVs: {segment_dir}")
    print(f"  Timed dubbed video: {out_path}")


if __name__ == "__main__":
    main()
