"""
EXPERIMENT (did not produce usable output - see experiments/README.md):
fully local/offline dubbing pipeline using AI4Bharat IndicF5 for voice cloning.

Pipeline: extract audio (ffmpeg) -> transcribe (faster-whisper) -> translate
(NLLB-200-distilled-600M) -> clone + synthesize (AI4Bharat IndicF5) -> remux
onto video (ffmpeg).

STATUS: in every real test run, IndicF5's output was garbled/unintelligible,
even after fixing two real bugs (reference audio being the whole noisy source
track instead of a clean clip; a ref_text/ref_audio desync after trimming).
The remaining cause was never conclusively isolated - see
experiments/README.md and RESEARCH_LOCAL_PIPELINE.md for the full story. Kept
here for reference, not recommended for use. The working solution is
../timed_chatterbox_dub_poc.py.

KNOWN LIMITATIONS:
  - No segment-level timestamp resync: the full transcript is translated and
    synthesized as one block, then wholesale-replaces the video's audio track.
    Lip-sync/timing will not be precise. (See timed_indicf5_dub_poc.py for the
    segment-timed variant, which has the same underlying quality problem.)
  - Voice cloning is NOT available for Urdu (no verified open-source model found).
    Urdu falls back to a non-cloned espeak-ng voice.
  - Output quality was unintelligible in every real test - see status above.
  - Requires ffmpeg on PATH (audio extraction/remux) and, for Urdu only,
    espeak-ng on PATH (fallback voice). Neither is pip-installable.
  - CPU inference (Whisper + NLLB-200 + IndicF5) is slow - ~10 minutes per
    sentence, which made iteration impractical.
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    LANGUAGES,
    check_binary,
    extract_audio,
    pick_reference_segment,
    remux,
    resolve_target_lang,
    slice_reference_clip,
    transcribe,
    translate,
)

# espeak-ng language codes for the non-cloned fallback path (Urdu only, for now).
ESPEAK_CODES = {
    "urd_Arab": "ur",
}

INDICF5_MODEL = os.getenv("INDICF5_MODEL", "ai4bharat/IndicF5")


def load_indicf5_model():
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
    from f5_tts.infer import utils_infer as f5_utils
    import torch
    from transformers import AutoConfig, AutoModel, PreTrainedModel

    try:
        # get_init_context (meta-device init) only exists on transformers 5.x.
        # This environment can resolve to an older transformers (e.g. pinned down
        # by f5-tts/pydub deps), so guard the patch rather than assuming it's there.
        has_init_context = hasattr(PreTrainedModel, "get_init_context")
        original_get_init_context = PreTrainedModel.get_init_context if has_init_context else None
        original_torch_compile = getattr(torch, "compile", None)
        original_load_model = f5_utils.load_model
        original_load_checkpoint = f5_utils.load_checkpoint

        def cpu_init_context(cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels):
            return []

        def load_model_compat(model_cls, model_cfg, ckpt_path=None, *args, **kwargs):
            if ckpt_path is None:
                try:
                    ckpt_path = hf_hub_download(
                        INDICF5_MODEL,
                        filename="model.safetensors",
                        token=os.getenv("HF_TOKEN"),
                        local_files_only=True,
                    )
                except LocalEntryNotFoundError:
                    ckpt_path = hf_hub_download(
                        INDICF5_MODEL,
                        filename="model.safetensors",
                        token=os.getenv("HF_TOKEN"),
                    )
            if "device" in kwargs:
                kwargs["device"] = str(kwargs["device"])
            return original_load_model(model_cls, model_cfg, ckpt_path, *args, **kwargs)

        def load_checkpoint_compat(model, ckpt_path, device, dtype=None, use_ema=True):
            if str(ckpt_path).endswith(".safetensors"):
                from safetensors.torch import load_file

                if dtype is None:
                    dtype = torch.float32
                model = model.to(dtype)
                checkpoint = load_file(ckpt_path, device=str(device))

                state_dict = {}
                for key, value in checkpoint.items():
                    if key in ["initted", "step"] or not key.startswith("ema_model."):
                        continue
                    key = key.replace("ema_model.", "", 1)
                    key = key.replace("_orig_mod.", "", 1)
                    if key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                        continue
                    state_dict[key] = value

                model.load_state_dict(state_dict)
                del checkpoint
                torch.cuda.empty_cache()
                return model.to(str(device))

            return original_load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

        # IndicF5's remote model code builds a torchaudio vocoder inside __init__.
        # Transformers 5 initializes custom models on the meta device by default,
        # which breaks that vocoder setup on CPU/Windows. Build this model normally.
        # IndicF5 also calls an older f5-tts load_model API; f5-tts 1.1.x requires
        # an explicit ckpt_path, so provide the repo's model.safetensors here.
        if has_init_context:
            PreTrainedModel.get_init_context = classmethod(cpu_init_context)
        f5_utils.load_model = load_model_compat
        f5_utils.load_checkpoint = load_checkpoint_compat
        if original_torch_compile is not None:
            torch.compile = lambda module, *args, **kwargs: module

        try:
            config = AutoConfig.from_pretrained(
                INDICF5_MODEL,
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True,
            )
            return AutoModel.from_config(config, trust_remote_code=True)
        finally:
            if has_init_context:
                PreTrainedModel.get_init_context = original_get_init_context
            f5_utils.load_model = original_load_model
            f5_utils.load_checkpoint = original_load_checkpoint
            if original_torch_compile is not None:
                torch.compile = original_torch_compile
    except HfHubHTTPError as e:
        if getattr(e.response, "status_code", None) == 403:
            print(
                "\nCannot download AI4Bharat IndicF5 from Hugging Face: your token was "
                "rejected with 403 Forbidden.\n\n"
                "Fix:\n"
                "  1. Open your Hugging Face token settings.\n"
                "  2. Edit the fine-grained token used in .env as HF_TOKEN.\n"
                "  3. Enable access to public gated repositories, and make sure you have "
                "accepted the IndicF5 model terms if Hugging Face prompts for them.\n"
                "  4. Re-run this script.\n\n"
                "Alternative: download the model separately and set INDICF5_MODEL in .env "
                "to the local model directory."
            )
            sys.exit(1)
        raise
    except LocalEntryNotFoundError:
        print(
            f"\nCould not find IndicF5 model files for '{INDICF5_MODEL}' locally, and "
            "Hugging Face download failed. Check your network/token, or set "
            "INDICF5_MODEL in .env to a local model directory."
        )
        sys.exit(1)
    except OSError as e:
        if "huggingface.co" in str(e) or "cached files" in str(e):
            print(
                f"\nCould not load IndicF5 model '{INDICF5_MODEL}'. If you are using the "
                "Hugging Face repo, check HF_TOKEN permissions. If you are offline, set "
                "INDICF5_MODEL in .env to a local model directory.\n\n"
                f"Original error: {e}"
            )
            sys.exit(1)
        raise


def clone_and_synthesize(translated_text, target_lang, reference_audio, reference_transcript, out_wav_path):
    if target_lang not in ESPEAK_CODES:
        print("Synthesizing speech in cloned voice (IndicF5)...")
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        model = load_indicf5_model()

        original_torchaudio_load = torchaudio.load

        def soundfile_torchaudio_load(uri, *args, **kwargs):
            data, sample_rate = sf.read(str(uri), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T.copy())
            return waveform, sample_rate

        # torchaudio 2.11+ routes load() through torchcodec on Windows, which
        # needs matching native DLLs. Our reference is a simple WAV, so soundfile
        # avoids that brittle decode path.
        torchaudio.load = soundfile_torchaudio_load
        try:
            audio = model(
                translated_text,
                ref_audio_path=str(reference_audio),
                ref_text=reference_transcript,
            )
        finally:
            torchaudio.load = original_torchaudio_load

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        sf.write(str(out_wav_path), np.array(audio, dtype=np.float32), samplerate=24000)
        return

    print(
        f"WARNING: voice cloning is not available for {target_lang} (Urdu) - "
        f"see RESEARCH_LOCAL_PIPELINE.md. Falling back to a non-cloned espeak-ng voice."
    )
    check_binary("espeak-ng", "the Urdu fallback voice")
    espeak_code = ESPEAK_CODES[target_lang]
    try:
        subprocess.run(
            ["espeak-ng", "-v", espeak_code, "-w", str(out_wav_path), translated_text],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"espeak-ng failed to synthesize speech:\n{e.stderr}")
        sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="EXPERIMENT - IndicF5 local dubbing (did not produce usable output - see experiments/README.md)."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to local input video file")
    parser.add_argument(
        "--target", "-t", required=True,
        help="Target language name or code, e.g. 'hindi' or 'hin_Deva'"
    )
    parser.add_argument(
        "--reference-clip", default=None,
        help="Short reference audio clip for voice cloning (defaults to the input video's own audio)"
    )
    parser.add_argument("--output", "-o", default=None, help="Output video path (default: derived)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    target_lang = resolve_target_lang(args.target)

    out_path = Path(args.output) if args.output else Path(f"{input_path.stem}_{args.target.lower()}.mp4")

    work_dir = input_path.parent
    source_wav = work_dir / f"{input_path.stem}_source.wav"
    reference_wav = work_dir / f"{input_path.stem}_reference.wav"
    dubbed_wav = work_dir / f"{input_path.stem}_{args.target.lower()}_dubbed.wav"

    extract_audio(input_path, source_wav)
    source_text, segments = transcribe(source_wav)

    if args.reference_clip:
        # User-supplied clip: transcribe it separately so ref_text actually
        # matches what's said in that specific clip.
        reference_audio = Path(args.reference_clip)
        reference_text, _ = transcribe(reference_audio)
    else:
        # Default: a short single-utterance slice of the source, not the whole
        # (likely multi-speaker/noisy) track - see pick_reference_segment().
        ref_segment = pick_reference_segment(segments)
        reference_audio = slice_reference_clip(source_wav, ref_segment, reference_wav)
        # Re-transcribe the actual sliced clip rather than reusing the original
        # segment's text - slicing can trim audio shorter than the segment
        # (see slice_reference_clip's max_duration cap), which would otherwise
        # leave ref_text describing content that isn't in ref_audio anymore.
        reference_text, _ = transcribe(reference_audio)
        print(f"  Using reference clip {reference_audio}: {reference_text}")

    translated_text = translate(source_text, target_lang)
    clone_and_synthesize(translated_text, target_lang, reference_audio, reference_text, dubbed_wav)
    remux(input_path, dubbed_wav, out_path)


if __name__ == "__main__":
    main()
