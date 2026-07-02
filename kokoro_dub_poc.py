"""
Skillfy Feature 3 POC: local/offline dubbing using Kokoro-82M for speech synthesis.

Pipeline: extract audio (ffmpeg) -> transcribe (faster-whisper) -> translate
(NLLB-200) -> synthesize (Kokoro-82M) -> remux onto video (ffmpeg).

Kokoro-82M does NOT clone the source speaker's voice - it uses a fixed preset
voice. Traded off deliberately for speed: Kokoro runs in seconds on CPU, vs.
IndicF5's ~10 minutes per sentence, which made same-day iteration impossible.
Kokoro also only supports Hindi among our 11 target languages (not the other
10) - this script only supports --target hindi for that reason.

Use this when you need a fast, reliable, no-cloning fallback. For real voice
cloning, use timed_chatterbox_dub_poc.py instead.
"""

import sys

import numpy as np
import soundfile as sf

from common import extract_audio, remux, transcribe, translate

KOKORO_LANG_CODE = "h"
KOKORO_VOICE = "hf_alpha"
SAMPLE_RATE = 24000


def synthesize_with_kokoro(text):
    print(f"Synthesizing speech with Kokoro-82M (voice={KOKORO_VOICE})...")
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=KOKORO_LANG_CODE)
    chunks = []
    for _, _, audio in pipeline(text, voice=KOKORO_VOICE):
        chunks.append(audio)
    return np.concatenate(chunks)


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Skillfy Feature 3 POC - local dubbing with Kokoro-82M (Hindi only, no voice cloning)."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to local input video file")
    parser.add_argument("--target", "-t", required=True, help="Must be 'hindi' - Kokoro only supports Hindi here")
    parser.add_argument("--output", "-o", default=None, help="Output video path (default: derived)")
    args = parser.parse_args()

    if args.target.strip().lower() not in ("hindi", "hin_deva"):
        print("Kokoro-82M only supports Hindi among our 11 target languages. Use --target hindi.")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    out_path = Path(args.output) if args.output else Path(f"{input_path.stem}_hindi_kokoro.mp4")
    work_dir = input_path.parent
    source_wav = work_dir / f"{input_path.stem}_source.wav"
    dubbed_wav = work_dir / f"{input_path.stem}_hindi_kokoro_dubbed.wav"

    extract_audio(input_path, source_wav)
    source_text, _ = transcribe(source_wav)
    translated_text = translate(source_text, "hin_Deva")

    audio = synthesize_with_kokoro(translated_text)
    sf.write(str(dubbed_wav), audio, SAMPLE_RATE)

    remux(input_path, dubbed_wav, out_path)


if __name__ == "__main__":
    main()
