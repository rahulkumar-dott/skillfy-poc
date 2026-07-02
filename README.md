# Skillfy Feature 3 POC — Multilingual AI Dubbing

Proof-of-concept for dubbing educational videos into Indian regional languages
with a humanized, cloned voice. Two independent approaches were built and
tested: a cloud API pipeline (ElevenLabs) and a fully local/offline pipeline
(open-source models, no per-request cost).

## TL;DR — what actually works

| Script | Voice cloning? | Languages | Cost | Status |
|---|---|---|---|---|
| **`timed_chatterbox_dub_poc.py`** | **Yes, real** (Chatterbox Multilingual) | Hindi only | Free (local CPU) | **Recommended** |
| `kokoro_dub_poc.py` / `timed_kokoro_dub_poc.py` | No (fixed voice) | Hindi only | Free (local CPU) | Fast, reliable fallback when cloning isn't needed |
| `dub_poc.py` | Yes, paid tier only | All 11 languages | ElevenLabs API cost | Works, but cloud-dependent and costs money for real cloning |
| `experiments/*.py` | Varies | Varies | Free (local CPU) | **Did not produce usable results** - kept for reference, see `experiments/README.md` |

**For a working local demo with real voice cloning, use `timed_chatterbox_dub_poc.py`.**

## Setup

1. Install [`uv`](https://docs.astral.sh/uv/) if you don't have it.
2. Install [ffmpeg](https://ffmpeg.org/) and make sure it's on your PATH.
3. Copy `.env.example` to `.env` and fill in the values you need (see below).
4. Install dependencies:
   ```
   uv sync --extra local
   ```
   This pulls in torch, transformers, faster-whisper, chatterbox-tts, kokoro, etc.
   Expect a multi-GB download on first install.

   On Windows, some dependencies (e.g. `curated-tokenizers`, a transitive
   dependency of `kokoro`) need a C++ compiler to build from source. If you
   hit a `Microsoft Visual C++ 14.0 or greater is required` error, install
   [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   with the "Desktop development with C++" workload.

## Running the recommended pipeline

```
uv run timed_chatterbox_dub_poc.py --input your_video.mp4 --target hindi
```

- Input: any local video file with clear English speech.
- Output: `<input>_hindi_chatterbox_timed.mp4` in the same directory, alongside
  intermediate artifacts (extracted audio, per-segment clips, segment timing JSON).
- **Resumable**: if interrupted (e.g. hit a time limit), rerun the exact same
  command - it skips already-transcribed/translated/synthesized segments and
  picks up where it left off.
- CPU-only inference: expect roughly 30-90 seconds per sentence. A short
  (~10-60s) clip is a reasonable first test.

For a fast, no-cloning fallback (e.g. if you just want to validate the
pipeline mechanics quickly), use `kokoro_dub_poc.py` instead — same CLI shape,
much faster, but the output uses a generic voice, not the source speaker's.

## Environment variables (`.env`)

| Variable | Needed for | Notes |
|---|---|---|
| `ELEVENLABS_API_KEY` | `dub_poc.py` (cloud pipeline) | Free tier works but has no real voice cloning - see script docstring |
| `HF_TOKEN` | `experiments/indicf5_dub_poc.py`, `experiments/indic_parler_tts_dub_poc.py` | Both use gated Hugging Face models requiring an accepted-terms account |

Chatterbox Multilingual and Kokoro (the working local pipeline) are **not**
gated - no `HF_TOKEN` needed for `timed_chatterbox_dub_poc.py` or
`kokoro_dub_poc.py`.

## Project layout

```
common.py                      # shared utilities: audio extraction, transcription,
                                # translation, timeline composition, remuxing
dub_poc.py                     # ElevenLabs cloud pipeline (works, costs money for real cloning)
timed_chatterbox_dub_poc.py    # RECOMMENDED - local pipeline, real voice cloning
kokoro_dub_poc.py               # local pipeline, no cloning, fast fallback
timed_kokoro_dub_poc.py         # ^ timestamp-aligned variant
RESEARCH_LOCAL_PIPELINE.md      # research notes on model/tool choices and trade-offs
experiments/                    # approaches that were tried and did not pan out - see experiments/README.md
```

## Known limitations

- **Only Hindi has real local voice cloning.** Chatterbox Multilingual supports
  23+ languages but only Hindi among Skillfy's 11 target regional languages.
  The other 10 (Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam,
  Punjabi, Odia, Urdu) have no verified open-source model that both clones
  voices and supports that language - see `RESEARCH_LOCAL_PIPELINE.md`.
- **No lip-sync.** Audio is timestamp-aligned to the original segments, but
  there's no video-side lip movement adjustment.
- **CPU-only.** No GPU acceleration was used or required, but this makes
  synthesis slow (tens of seconds per sentence) - not suitable for long-form
  content without significant time investment.
- **ElevenLabs cloud pipeline** (`dub_poc.py`) supports all 11 languages with
  real cloning, but only on a **paid** tier - the free tier has no cloning at all.
