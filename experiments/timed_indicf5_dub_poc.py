"""
EXPERIMENT (did not produce usable output - see experiments/README.md):
timestamp-based local video dubbing using AI4Bharat IndicF5.

Same segment-timestamp approach as the working ../timed_chatterbox_dub_poc.py:
transcribe with timestamps, translate each segment, synthesize each one,
place on a silent timeline at its original start/end time - but using IndicF5
for synthesis, which never produced intelligible output in testing. Kept for
reference. See experiments/README.md and RESEARCH_LOCAL_PIPELINE.md.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    audio_duration_seconds,
    check_binary,
    compose_timed_audio,
    extract_audio,
    fade_edges,
    full_source_transcript,
    load_translation_model,
    load_wav_mono,
    pick_reference_segment,
    remux,
    resolve_target_lang,
    slice_reference_clip,
    transcribe,
    transcribe_segments,
    translate_segments,
    write_segments_json,
    write_wav_mono,
)
from indicf5_dub_poc import ESPEAK_CODES, load_indicf5_model

FINAL_SAMPLE_RATE = 24000


def synthesize_with_indicf5(model, text, reference_audio, reference_text):
    import soundfile as sf
    import torch
    import torchaudio

    original_torchaudio_load = torchaudio.load

    def soundfile_torchaudio_load(uri, *args, **kwargs):
        data, sample_rate = sf.read(str(uri), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T.copy())
        return waveform, sample_rate

    torchaudio.load = soundfile_torchaudio_load
    try:
        audio = model(text, ref_audio_path=str(reference_audio), ref_text=reference_text)
    finally:
        torchaudio.load = original_torchaudio_load

    import numpy as np

    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    return np.asarray(audio, dtype=np.float32).reshape(-1), FINAL_SAMPLE_RATE


def synthesize_with_espeak(text, target_lang, out_path):
    check_binary("espeak-ng", "the non-cloned fallback voice")
    espeak_code = ESPEAK_CODES[target_lang]
    try:
        subprocess.run(
            ["espeak-ng", "-v", espeak_code, "-w", str(out_path), text],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"espeak-ng failed to synthesize speech:\n{e.stderr}")
        sys.exit(1)
    return load_wav_mono(out_path)


def synthesize_segments(segments, target_lang, reference_audio, reference_text, segment_dir):
    segment_dir.mkdir(parents=True, exist_ok=True)
    model = None
    if target_lang not in ESPEAK_CODES:
        print("Loading IndicF5 once for segment synthesis...")
        model = load_indicf5_model()
    else:
        print(f"Using espeak-ng fallback for {target_lang}; voice cloning is not available.")

    for segment in segments:
        out_path = segment_dir / f"segment_{segment['index']:04d}.wav"
        text = segment["translated_text"]
        print(f"Synthesizing segment {segment['index']:04d}...")

        if model is not None:
            audio, sample_rate = synthesize_with_indicf5(model, text, reference_audio, reference_text)
            raw_audio = audio
        else:
            raw_audio, sample_rate = synthesize_with_espeak(text, target_lang, out_path)

        if sample_rate != FINAL_SAMPLE_RATE:
            print(f"Segment {segment['index']:04d} has unsupported sample rate {sample_rate}; expected {FINAL_SAMPLE_RATE}.")
            sys.exit(1)

        raw_audio = fade_edges(raw_audio, sample_rate)
        write_wav_mono(out_path, raw_audio, sample_rate)
        segment["segment_wav"] = str(out_path)
        segment["generated_duration"] = len(raw_audio) / float(sample_rate)

    return segments


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="EXPERIMENT - timestamp-based IndicF5 local dubbing (did not produce usable output)."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to local input video file")
    parser.add_argument(
        "--target", "-t", required=True,
        help="Target language name or code, e.g. 'hindi' or 'hin_Deva'",
    )
    parser.add_argument(
        "--reference-clip", default=None,
        help="Short reference audio clip for voice cloning (defaults to the input video's own audio)",
    )
    parser.add_argument("--output", "-o", default=None, help="Output video path (default: derived)")
    parser.add_argument("--work-dir", default=None, help="Directory for timed dubbing artifacts")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    target_lang = resolve_target_lang(args.target)
    target_label = args.target.lower()
    work_dir = Path(args.work_dir) if args.work_dir else input_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    source_wav = work_dir / f"{input_path.stem}_source.wav"
    segments_json = work_dir / f"{input_path.stem}_{target_label}_segments.json"
    segment_dir = work_dir / f"{input_path.stem}_{target_label}_segments"
    timed_dubbed_wav = work_dir / f"{input_path.stem}_{target_label}_timed_dubbed.wav"
    out_path = Path(args.output) if args.output else work_dir / f"{input_path.stem}_{target_label}_timed.mp4"

    extract_audio(input_path, source_wav)
    total_duration = audio_duration_seconds(source_wav)

    segments = transcribe_segments(source_wav)

    if args.reference_clip:
        reference_audio = Path(args.reference_clip)
        reference_text = full_source_transcript(transcribe_segments(reference_audio))
    else:
        reference_wav = work_dir / f"{input_path.stem}_reference.wav"
        ref_segment = pick_reference_segment(segments)
        reference_audio = slice_reference_clip(source_wav, ref_segment, reference_wav)
        reference_text = full_source_transcript(transcribe_segments(reference_audio))
        print(f"  Using reference clip {reference_audio}: {reference_text}")

    tokenizer, translation_model = load_translation_model()
    segments = translate_segments(segments, target_lang, tokenizer, translation_model)
    write_segments_json(segments_json, segments)

    segments = synthesize_segments(segments, target_lang, reference_audio, reference_text, segment_dir)
    segments = compose_timed_audio(segments, total_duration, timed_dubbed_wav)
    write_segments_json(segments_json, segments)

    remux(input_path, timed_dubbed_wav, out_path)

    print("Saved timestamped artifacts:")
    print(f"  Source audio: {source_wav}")
    print(f"  Segment JSON: {segments_json}")
    print(f"  Segment WAVs: {segment_dir}")
    print(f"  Timed dubbed WAV: {timed_dubbed_wav}")
    print(f"  Timed dubbed video: {out_path}")


if __name__ == "__main__":
    main()
