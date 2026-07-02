# Research: Fully Local / CPU-Only Multilingual Dubbing Pipeline

This documents the options investigated for a version of the Feature 3 dubbing POC that
runs entirely offline on CPU (no cloud API, no GPU, zero ongoing cost), as a comparison
point against the cloud-based ElevenLabs POC (`dub_poc.py`).

Target languages (11): Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada,
Malayalam, Punjabi, Odia, Urdu.

## Headline finding

**No single open-source model does both "genuine voice cloning" and "all 11 target
languages."** The closest fit — AI4Bharat's IndicF5 — covers 10 of 11 (everything
except Urdu). This mirrors the same kind of gap the ElevenLabs free tier has (no
cloning at all); here the gap is narrower (one language) but still real and should be
disclosed the same way when demoing.

## 1. Speech-to-text (transcribe the source video's audio)

| Option | License | Setup | CPU speed | Timestamps | Verdict |
|---|---|---|---|---|---|
| **faster-whisper** (chosen) | MIT | `pip install faster-whisper` + ffmpeg on PATH, no compilation | Fast — int8 quantized, a few sec to ~30s for a short clip on `small`/`medium` | Native word + segment level | **Recommended** |
| whisper.cpp | MIT | Needs compilation (or prebuilt `pywhispercpp` wheel) | Fast, memory-efficient | Yes | Viable alt, faster-whisper edges it out on CPU speed |
| openai-whisper (original) | MIT | Trivial pip install | Noticeably slower (no quantization) | Yes | Not recommended for CPU-only |
| Vosk | Apache 2.0 | Trivial pip install, lightweight models | Very fast | Limited | Lower accuracy, best for clean/simple audio |
| Qwen3-ASR / Cohere-transcribe (2026) | Apache 2.0 | pip install | GPU-oriented | Yes | Not well suited to laptop CPU |

Known caveat: Whisper's word error rate is measurably higher on Indian-accented
English (~21.8% in one study) vs. native-accent speech — expect more transcription
errors than a clean-audio demo would suggest.

**Recommendation: faster-whisper**, `small` or `medium` model, int8 quantization,
`word_timestamps=True`.

## 2. Translation (English → 11 Indian languages)

| Option | License | All 11 languages? | Setup | Quality for Indic languages |
|---|---|---|---|---|
| **NLLB-200-distilled-600M** (chosen) | CC-BY-NC-4.0 (non-commercial) | Yes | Zero extra setup, pure HF download (~2.4GB) | Good, but generalist (200 languages) |
| AI4Bharat IndicTrans2 | MIT | Yes | `transformers` + `IndicTransToolkit` pip package | Purpose-built for Indic languages, better than general-purpose models on the IN22 benchmark |
| Argos Translate | MIT-ish (permissive) | **No — only 3 of 11** (Hindi, Bengali, Urdu) | Simplest `pip install` | Disqualified: fails the language-coverage requirement outright |

**Recommendation (updated after implementation): NLLB-200-distilled-600M.**
IndicTrans2 was the original pick (better Indic-language quality, MIT license), but
`IndicTransToolkit` ships a Cython extension (`processor.pyx`) that failed to build on
the target Windows machine with `error: Microsoft Visual C++ 14.0 or greater is
required` — no prebuilt Windows wheel is published, so it needs a full C++ build
toolchain installed. Rather than requiring a multi-GB Visual Studio Build Tools
install for a POC, we swapped to NLLB-200, which is pure `transformers` + PyTorch
with no compiled dependencies. Trade-off: CC-BY-NC-4.0 license (non-commercial only)
instead of MIT — acceptable for this POC, would need revisiting (e.g. building
IndicTrans2 on a Linux CI box, or finding a prebuilt wheel) before any commercial/
production use.

Language codes used: `hin_Deva`, `tam_Taml`, `tel_Telu`, `ben_Beng`, `mar_Deva`,
`guj_Gujr`, `kan_Knda`, `mal_Mlym`, `pan_Guru`, `ory_Orya`, `urd_Arab` (source: `eng_Latn`).

## 3. Voice cloning + text-to-speech

| Option | License | Cloning? | Indian language coverage (of our 11) | CPU feasibility |
|---|---|---|---|---|
| **AI4Bharat IndicF5** (chosen) | MIT | Yes — genuine zero-shot (reference clip + its transcript) | **10 of 11**: Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia. **Urdu not supported.** | Not benchmarked publicly; expect non-real-time but workable for short clips (shares F5-TTS architecture) |
| Coqui XTTS-v2 | CPML (non-commercial) | Yes | Only Hindi | ~30s+ per inference on CPU |
| Chatterbox Multilingual (Resemble AI) | MIT | Yes | Only Hindi | CPU-supported via community server |
| AI4Bharat indic-parler-tts | MIT | **No** — fixed/described voices only, explicitly will not support cloning | Broad | N/A |
| F5-TTS (base) | MIT | Yes | Not in native language set | 10-30x slower than realtime on CPU |

**Recommendation: AI4Bharat IndicF5** for 10 of 11 languages. It's the only model
combining real cloning with meaningful Indic-language coverage, and it's MIT-licensed
(unlike XTTS-v2's non-commercial CPML weights).

**Urdu fallback:** No open-source cloning-capable model with verified Urdu support was
found. The pipeline falls back to **espeak-ng** (`ur` language code confirmed
supported) for Urdu only — a non-cloned, noticeably more robotic stock voice. This is
a known, disclosed limitation, not a bug: if Urdu is selected, the script prints a
clear warning before falling back, rather than silently producing a lower-quality
result or crashing.

**Cross-lingual caveat:** IndicF5's documented usage pattern pairs a reference audio
clip with a transcript *in the same language/script as the reference audio*. Using an
English reference clip (the likely source-video voice) to clone into Hindi/Tamil/etc.
output is a cross-lingual use case not explicitly validated by AI4Bharat's own
examples — voice timbre should still transfer reasonably (this is standard for
F5-TTS-style flow-matching models), but pronunciation/prosody fidelity in the cloned
voice is unverified until tested. Flagging this rather than assuming it works
perfectly out of the box.

**Confirmed empirically (first real test run):** the dominant quality problem turned
out to be a simpler, more basic mistake than the cross-lingual caveat above — the
initial implementation used the *entire* source track (a multi-speaker, music-and-sound-effects-laden movie trailer clip) as the voice-cloning reference. IndicF5, like
other zero-shot cloning models, needs a short (~2-8s), clean, single-speaker reference
utterance; feeding it a whole noisy multi-speaker trailer produced unintelligible
output. Fixed by auto-selecting one short single-utterance segment from the
transcription (`pick_reference_segment()` in `local_dub_poc.py`) instead of the whole
track. **Practical implication for the real feature**: Skillfy's actual content
(a single instructor speaking continuously) is a much better-suited input than a
movie trailer was as a stress test — this failure mode is much less likely to recur
on real target content, but the auto-selection fix makes the pipeline robust either
way.

## 4. Pipeline architecture simplification

Unlike the ElevenLabs Dubbing API (which handles transcription, translation, cloning,
and video re-sync server-side in one call), a local pipeline has to do all of this
itself. For this POC, the approach is deliberately simplified:

- Transcribe the **full** source audio to text (not per-segment).
- Translate the **full** text as one block.
- Clone-generate the **full** translated speech as a single audio track.
- Replace the original video's audio track wholesale with the new track.

This sacrifices precise sentence-level timing/lip-sync in favor of simplicity — a
segment-by-segment timestamp-aligned resync (real lip-sync engineering) is a
legitimate follow-up, not part of this POC.

## 5. External dependency

**ffmpeg** is required (audio extraction from the source video, and remuxing the new
audio track back onto the video). It is not pip-installable — must be installed
separately and available on PATH. The script checks for it and fails with a clear
message if missing, rather than a confusing subprocess error.

## Summary recommendation

| Stage | Tool | License |
|---|---|---|
| Transcription | faster-whisper (small/medium, int8) | MIT |
| Translation | NLLB-200-distilled-600M | CC-BY-NC-4.0 (non-commercial) |
| Voice cloning + TTS | AI4Bharat IndicF5 (10/11 languages) | MIT |
| Urdu fallback | espeak-ng (no cloning) | GPL |
| Audio/video muxing | ffmpeg (external system dependency) | LGPL/GPL depending on build |

This pipeline is mostly-permissive (MIT for ASR and TTS) with one non-commercial
license in the mix (NLLB-200, swapped in for a Windows build-toolchain reason, not a
capability reason — see Translation section above), fully offline, and two clearly
disclosed gaps: **no voice cloning for Urdu**, and **non-commercial-only translation
licensing** until IndicTrans2 can be built (e.g. on Linux, or via a prebuilt wheel).
