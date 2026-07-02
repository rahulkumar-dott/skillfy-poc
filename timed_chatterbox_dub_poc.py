"""
Skillfy Feature 3 POC: timestamp-based local video dubbing using Chatterbox
Multilingual (real zero-shot voice cloning, MIT license).

RECOMMENDED - this is the working local pipeline. Transcribes with timestamps,
translates each segment, synthesizes each one in the cloned voice, places on a
silent timeline at its original start/end time, and remuxes onto the video.

Chatterbox Multilingual supports 23+ languages but only Hindi among our 11
target languages (Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada,
Malayalam, Punjabi, Odia, Urdu are not supported) - this script is Hindi-only.
Note: "Chatterbox-Turbo" (the fast 350M variant) is English-only; this uses
the separate, heavier Chatterbox Multilingual model instead, since Turbo has
no Hindi support at all.

Reference-clip handling: uses a short (~2-8s), clean, single-utterance slice
of the source audio as the cloning reference, not the whole track
(pick_reference_segment/slice_reference_clip in common.py) - re-transcribes
the actual sliced clip for ref_text so it always matches what's really in the
reference audio. See RESEARCH_LOCAL_PIPELINE.md for why these matter.
"""

import sys
from pathlib import Path

from common import (
    audio_duration_seconds,
    extract_audio,
    fade_edges,
    load_translation_model,
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

CHATTERBOX_LANGUAGE_ID = "hi"


def load_chatterbox_model():
    print("Loading Chatterbox Multilingual (CPU)...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    return ChatterboxMultilingualTTS.from_pretrained(device="cpu")


def compose_timed_audio(segments, total_duration, out_wav_path, sample_rate):
    import numpy as np
    import soundfile as sf

    print("Composing timed dubbed audio track...")
    total_samples = max(1, int(np.ceil(total_duration * sample_rate)))
    timeline = np.zeros(total_samples, dtype=np.float32)

    for segment in segments:
        audio, audio_sample_rate = sf.read(segment["segment_wav"], dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
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


def synthesize_segments(segments, reference_audio, segment_dir):
    import numpy as np
    import soundfile as sf

    segment_dir.mkdir(parents=True, exist_ok=True)

    # Resumable: skip segments already synthesized to disk on a prior (possibly
    # interrupted) run, so re-running the same command picks up where it left off
    # instead of redoing expensive CPU synthesis from scratch.
    pending = []
    for segment in segments:
        out_path = segment_dir / f"segment_{segment['index']:04d}.wav"
        if out_path.exists():
            print(f"Segment {segment['index']:04d} already synthesized - reusing {out_path}")
            segment["segment_wav"] = str(out_path)
            info = sf.info(str(out_path))
            segment["generated_duration"] = info.frames / float(info.samplerate)
        else:
            pending.append(segment)

    if not pending:
        sample_rate = sf.info(segments[0]["segment_wav"]).samplerate
        return segments, sample_rate

    model = load_chatterbox_model()

    for segment in pending:
        out_path = segment_dir / f"segment_{segment['index']:04d}.wav"
        text = segment["translated_text"]
        print(f"Synthesizing segment {segment['index']:04d} (cloned voice)...")

        wav = model.generate(text, language_id=CHATTERBOX_LANGUAGE_ID, audio_prompt_path=str(reference_audio))
        audio = wav.squeeze().cpu().numpy()
        audio = fade_edges(np.asarray(audio, dtype="float32"), model.sr)
        write_wav_mono(out_path, audio, model.sr)

        segment["segment_wav"] = str(out_path)
        segment["generated_duration"] = len(audio) / float(model.sr)

    return segments, model.sr


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Skillfy Feature 3 POC - timestamp-based local dubbing with Chatterbox Multilingual (Hindi only, real voice cloning)."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to local input video file")
    parser.add_argument("--target", "-t", required=True, help="Must be 'hindi' - Chatterbox Multilingual only supports Hindi here")
    parser.add_argument(
        "--reference-clip", default=None,
        help="Short reference audio clip for voice cloning (defaults to an auto-picked clean slice of the input video's own audio)"
    )
    parser.add_argument("--output", "-o", default=None, help="Output video path (default: derived)")
    parser.add_argument("--work-dir", default=None, help="Directory for dubbing artifacts")
    args = parser.parse_args()

    if args.target.strip().lower() not in ("hindi", "hin_deva"):
        print("Chatterbox Multilingual only supports Hindi among our 11 target languages in this script. Use --target hindi.")
        sys.exit(1)
    target_lang = resolve_target_lang(args.target)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    work_dir = Path(args.work_dir) if args.work_dir else input_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    source_wav = work_dir / f"{input_path.stem}_source.wav"
    reference_wav = work_dir / f"{input_path.stem}_reference.wav"
    segments_json = work_dir / f"{input_path.stem}_hindi_chatterbox_segments.json"
    segment_dir = work_dir / f"{input_path.stem}_hindi_chatterbox_segments"
    timed_dubbed_wav = work_dir / f"{input_path.stem}_hindi_chatterbox_timed_dubbed.wav"
    out_path = Path(args.output) if args.output else work_dir / f"{input_path.stem}_hindi_chatterbox_timed.mp4"

    extract_audio(input_path, source_wav)
    total_duration = audio_duration_seconds(source_wav)

    # Resumable: if a prior (possibly interrupted) run already transcribed,
    # translated, and picked a reference clip, reuse that instead of redoing
    # transcription/translation/reference-selection on every retry - this saves
    # meaningful time before even reaching the slow model-load + synthesis step.
    if segments_json.exists() and reference_wav.exists():
        import json

        with segments_json.open("r", encoding="utf-8") as f:
            segments = json.load(f)
        if segments and all("translated_text" in s for s in segments):
            print(f"Reusing existing transcription/translation from {segments_json}")
            reference_audio = reference_wav
        else:
            segments = None
    else:
        segments = None

    if segments is None:
        segments = transcribe_segments(source_wav)

        if args.reference_clip:
            reference_audio = Path(args.reference_clip)
            reference_text, _ = transcribe(reference_audio)
        else:
            ref_segment = pick_reference_segment(segments)
            reference_audio = slice_reference_clip(source_wav, ref_segment, reference_wav)
            reference_text, _ = transcribe(reference_audio)
            print(f"  Using reference clip {reference_audio}: {reference_text}")

        tokenizer, translation_model = load_translation_model()
        segments = translate_segments(segments, target_lang, tokenizer, translation_model)
    write_segments_json(segments_json, segments)

    segments, sample_rate = synthesize_segments(segments, reference_audio, segment_dir)
    segments = compose_timed_audio(segments, total_duration, timed_dubbed_wav, sample_rate)
    write_segments_json(segments_json, segments)

    remux(input_path, timed_dubbed_wav, out_path)

    print("Saved timestamped Chatterbox artifacts:")
    print(f"  Reference clip: {reference_audio}")
    print(f"  Segment JSON: {segments_json}")
    print(f"  Segment WAVs: {segment_dir}")
    print(f"  Timed dubbed video: {out_path}")


if __name__ == "__main__":
    main()
