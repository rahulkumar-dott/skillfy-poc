"""
EXPERIMENT (never fully tested - see experiments/README.md):
timestamp-based local dubbing with Indic Parler-TTS.

This is an alternate TTS backend to indicf5_dub_poc.py. It keeps the same
Whisper timestamp -> NLLB translation -> timed audio merge flow, but replaces
IndicF5 voice cloning with ai4bharat/indic-parler-tts.

Indic Parler-TTS does NOT clone the source speaker - it generates speech from
text plus a voice/style description. Abandoned before real testing: the model
is gated on Hugging Face and access was never granted/pursued once Chatterbox
Multilingual (real cloning, MIT license, faster) turned out to work. Kept here
for reference. See experiments/README.md and RESEARCH_LOCAL_PIPELINE.md.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    audio_duration_seconds,
    compose_timed_audio,
    extract_audio,
    fade_edges,
    load_translation_model,
    load_wav_mono,
    remux,
    resolve_target_lang,
    write_segments_json,
    write_wav_mono,
)

INDIC_PARLER_MODEL = "ai4bharat/indic-parler-tts"
DEFAULT_DESCRIPTION = (
    "Rohit's voice is clear, natural, and close-sounding. He speaks Hindi at a "
    "moderate pace with a slightly expressive conversational tone. The recording "
    "is very clear audio with almost no background noise."
)
FINAL_SAMPLE_RATE = 24000


def transcribe_segments(wav_path, start_segment=0, max_segments=None):
    """Like common.transcribe_segments, but supports slicing a range of segments
    for quick tests (--start-segment/--max-segments) without redoing the whole clip."""
    print("Transcribing audio with timestamps (faster-whisper)...")
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8")
    whisper_segments, _ = model.transcribe(str(wav_path), word_timestamps=False)

    segments = []
    for index, segment in enumerate(whisper_segments):
        text = segment.text.strip()
        if not text:
            continue
        if index < start_segment:
            continue
        segments.append(
            {
                "index": index,
                "start": float(segment.start),
                "end": float(segment.end),
                "source_text": text,
            }
        )
        if max_segments is not None and len(segments) >= max_segments:
            break

    if not segments:
        print("Transcription produced no selected speech segments.")
        sys.exit(1)

    print(f"  Selected {len(segments)} timed speech segments.")
    return segments


def translate_segments(segments, target_lang, tokenizer, model):
    """Like common.translate_segments, but with anti-repetition generation params -
    Indic Parler-TTS's translated input was prone to repeated phrases without these."""
    print(f"Translating {len(segments)} segments to {target_lang}...")
    bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    for segment in segments:
        inputs = tokenizer(segment["source_text"], return_tensors="pt")
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=bos_token_id,
            max_length=120,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        segment["translated_text"] = translated
        print(f"  [{segment['index']:04d}] {segment['start']:.2f}-{segment['end']:.2f}s: {translated}")

    return segments


def load_indic_parler_tts():
    print(f"Loading Indic Parler-TTS ({INDIC_PARLER_MODEL})...")
    import torch
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    hf_token = os.getenv("HF_TOKEN")
    try:
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            INDIC_PARLER_MODEL,
            token=hf_token,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(INDIC_PARLER_MODEL, token=hf_token)
        description_tokenizer = AutoTokenizer.from_pretrained(
            model.config.text_encoder._name_or_path,
            token=hf_token,
        )
    except (GatedRepoError, HfHubHTTPError, OSError) as e:
        message = str(e)
        if "gated repo" in message.lower() or "403" in message or "401" in message:
            print(
                "\nCannot download ai4bharat/indic-parler-tts from Hugging Face.\n\n"
                "Fix:\n"
                "  1. Open https://huggingface.co/ai4bharat/indic-parler-tts\n"
                "  2. Request/accept access for this model.\n"
                "  3. Make sure .env has an HF_TOKEN for the same Hugging Face account.\n"
                "  4. If the token is fine-grained, grant it read access to this gated repo.\n"
            )
            sys.exit(1)
        raise

    sample_rate = int(getattr(model.config, "sampling_rate", FINAL_SAMPLE_RATE))
    if sample_rate != FINAL_SAMPLE_RATE:
        print(f"Indic Parler-TTS sampling rate is {sample_rate}; using it for all generated audio.")

    return {
        "device": device,
        "model": model,
        "tokenizer": tokenizer,
        "description_tokenizer": description_tokenizer,
        "sample_rate": sample_rate,
    }


def synthesize_with_indic_parler(tts, text, description):
    import torch

    description_inputs = tts["description_tokenizer"](
        description,
        return_tensors="pt",
    ).to(tts["device"])
    prompt_inputs = tts["tokenizer"](
        text,
        return_tensors="pt",
    ).to(tts["device"])

    with torch.inference_mode():
        generation = tts["model"].generate(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )

    audio = generation.cpu().numpy().squeeze().astype(np.float32)
    return audio.reshape(-1), tts["sample_rate"]


def synthesize_segments(segments, segment_dir, description):
    segment_dir.mkdir(parents=True, exist_ok=True)
    tts = load_indic_parler_tts()

    for segment in segments:
        out_path = segment_dir / f"segment_{segment['index']:04d}.wav"
        if out_path.exists():
            audio, sample_rate = load_wav_mono(out_path)
            print(f"Reusing existing segment {segment['index']:04d}: {out_path}")
        else:
            print(f"Synthesizing segment {segment['index']:04d} with Indic Parler-TTS...")
            audio, sample_rate = synthesize_with_indic_parler(
                tts,
                segment["translated_text"],
                description,
            )
            audio = fade_edges(audio, sample_rate)
            write_wav_mono(out_path, audio, sample_rate)

        segment["segment_wav"] = str(out_path)
        segment["generated_duration"] = len(audio) / float(sample_rate)
        segment["sample_rate"] = sample_rate

    return segments, tts["sample_rate"]


def main():
    parser = argparse.ArgumentParser(
        description="EXPERIMENT - Indic Parler-TTS local dubbing (never fully tested, no voice cloning)."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to local input video file")
    parser.add_argument(
        "--target",
        "-t",
        required=True,
        help="Target language name or code, e.g. 'hindi' or 'hin_Deva'",
    )
    parser.add_argument("--output", "-o", default=None, help="Output video path (default: derived)")
    parser.add_argument("--work-dir", default=None, help="Directory for dubbing artifacts")
    parser.add_argument("--start-segment", type=int, default=0, help="First Whisper segment index to process")
    parser.add_argument("--max-segments", type=int, default=None, help="Limit segment count for quick tests")
    parser.add_argument(
        "--description",
        default=DEFAULT_DESCRIPTION,
        help="Indic Parler-TTS voice/style description",
    )
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
    segments_json = work_dir / f"{input_path.stem}_{target_label}_indic_parler_segments.json"
    segment_dir = work_dir / f"{input_path.stem}_{target_label}_indic_parler_segments"
    timed_dubbed_wav = work_dir / f"{input_path.stem}_{target_label}_indic_parler_timed_dubbed.wav"
    out_path = Path(args.output) if args.output else work_dir / f"{input_path.stem}_{target_label}_indic_parler_timed.mp4"

    extract_audio(input_path, source_wav)
    total_duration = audio_duration_seconds(source_wav)

    segments = transcribe_segments(source_wav, args.start_segment, args.max_segments)
    tokenizer, translation_model = load_translation_model()
    segments = translate_segments(segments, target_lang, tokenizer, translation_model)
    write_segments_json(segments_json, segments)

    segments, sample_rate = synthesize_segments(segments, segment_dir, args.description)
    segments = compose_timed_audio(segments, total_duration, timed_dubbed_wav, sample_rate)
    write_segments_json(segments_json, segments)

    remux(input_path, timed_dubbed_wav, out_path)

    print("Saved Indic Parler-TTS timestamped artifacts:")
    print(f"  Source audio: {source_wav}")
    print(f"  Segment JSON: {segments_json}")
    print(f"  Segment WAVs: {segment_dir}")
    print(f"  Timed dubbed WAV: {timed_dubbed_wav}")
    print(f"  Timed dubbed video: {out_path}")


if __name__ == "__main__":
    main()
