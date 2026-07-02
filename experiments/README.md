# Experiments

Approaches that were tried for the local dubbing pipeline and did **not**
become the recommended solution. Kept for reference and research transparency
- see the parent `README.md` for the working solution
(`../timed_chatterbox_dub_poc.py`) and `../RESEARCH_LOCAL_PIPELINE.md` for the
full model comparison.

## `indicf5_dub_poc.py` / `timed_indicf5_dub_poc.py` — AI4Bharat IndicF5

**Status: did not produce usable output.** IndicF5 was the first voice-cloning
model tried, chosen because it's the only open-source model found that
combines real zero-shot cloning with broad Indic-language coverage (10 of 11
target languages).

In every real test run, the output was garbled/unintelligible, even after
fixing two real bugs along the way:

1. The reference audio was initially the *entire* source track (e.g. a
   multi-speaker, music-and-sound-effects-laden video), when the model needs a
   short (~2-8s), clean, single-speaker reference clip. Fixed by
   auto-selecting a short single-utterance segment (`pick_reference_segment`/
   `slice_reference_clip`, now in `../common.py`).
2. After that fix, the reference audio was correctly trimmed but the
   reference *text* still described the untrimmed segment's full content - a
   desync between what the reference audio actually contained and what its
   transcript claimed. Fixed by re-transcribing the actual trimmed clip.

Even with both fixes applied, and even when testing with a hand-written,
grammatically correct Hindi sentence (to rule out bad translation as the
cause), the output remained unintelligible. The leading hypothesis is a
cross-lingual limitation: IndicF5's own documented example pairs a reference
clip with a transcript *in the same Indic script* as the target output (e.g.
Punjabi reference → Hindi target); using an English reference clip/transcript
may be out-of-distribution for the model. **This was never conclusively
proven** - the test that would have isolated it (AI4Bharat's own official
example, zero English involved) was interrupted and its output deleted before
anyone listened to the result.

Getting IndicF5 to load at all on Windows/CPU also required non-trivial
compatibility monkey-patches (see `load_indicf5_model()` in
`indicf5_dub_poc.py`) to work around a `transformers` version mismatch and a
`torchaudio`/`torchcodec` DLL loading issue on Windows. It's possible one of
these patches introduced a subtle correctness bug that was never ruled out as
an alternative explanation.

CPU inference was also very slow (~10 minutes per sentence), which made
further iteration on this hypothesis impractical - Chatterbox Multilingual
was tried next and worked, so IndicF5 was abandoned rather than debugged
further.

## `indic_parler_tts_dub_poc.py` — AI4Bharat Indic Parler-TTS

**Status: never fully tested.** This model does **not** clone voices - it
generates speech from text plus a natural-language voice/style description
(e.g. "Rohit's voice is clear, natural..."), which seemed like it might be
more stable than IndicF5 on CPU. It's a gated model on Hugging Face; getting
access was never pursued because Chatterbox Multilingual (real cloning,
faster, MIT license) was already working by the time this would have been
prioritized.

## Why Chatterbox Multilingual won instead

See `../RESEARCH_LOCAL_PIPELINE.md` and `../README.md` for the full
comparison. In short: MIT license, genuine zero-shot cloning from a few
seconds of reference audio, and roughly 30-40 seconds per sentence on CPU
(vs. IndicF5's ~10 minutes) - the speed difference alone made it possible to
actually iterate and validate the pipeline in a single session. Its one real
limitation for this project is language coverage: only Hindi among Skillfy's
11 target languages, versus IndicF5's (theoretical, unproven) 10-of-11
coverage.
