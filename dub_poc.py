"""
Skillfy Feature 3 POC: Multilingual AI Dubbing via ElevenLabs Dubbing API (free tier).

FREE TIER NOTICE:
  - No voice cloning (that's a paid Creator-tier feature). Output uses ElevenLabs'
    default automatic dubbing voice, NOT the original speaker's cloned voice.
  - Output is watermarked and has no commercial license.
  - Free tier gives ~10,000 credits/month; automatic dubbing costs ~2,000+
    credits/minute of source audio, i.e. only ~5 minutes of dubbing per month total.
  - Odia ('or') dubbing support is unverified against ElevenLabs' actual capability
    until the first live test - the script will fail gracefully if it's rejected.
"""

import argparse
import mimetypes
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.elevenlabs.io/v1"

LANGUAGES = {
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "bengali": "bn",
    "marathi": "mr",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "punjabi": "pa",
    "odia": "or",
    "urdu": "ur",
}

FREE_TIER_BANNER = """\
=== FREE TIER NOTICE ===
This POC runs on the ElevenLabs FREE tier:
  - No voice cloning (paid Creator-tier feature) - output uses ElevenLabs'
    default automatic dubbing voice, NOT the original speaker's cloned voice.
  - Output is watermarked and has no commercial license.
  - Free tier gives ~10,000 credits/month; dubbing costs ~2,000+ credits/min.
    That's only ~5 minutes of dubbing possible per month - budget test runs
    accordingly and check your ElevenLabs dashboard credit balance first.
========================
"""


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


def get_api_key():
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print(
            "Missing ELEVENLABS_API_KEY environment variable.\n"
            "Set it first, e.g.:\n"
            "  PowerShell: $env:ELEVENLABS_API_KEY = \"your-key-here\"\n"
            "  bash:       export ELEVENLABS_API_KEY=your-key-here"
        )
        sys.exit(1)
    return api_key


def submit_dubbing_job(api_key, video_path, target_lang, source_lang):
    url = f"{API_BASE}/dubbing"
    headers = {"xi-api-key": api_key}
    # Free-tier accounts must request a watermarked dub (non-watermarked requires Starter+).
    data = {"target_lang": target_lang, "source_lang": source_lang, "watermark": "true"}

    content_type = mimetypes.guess_type(video_path)[0] or "application/octet-stream"

    with open(video_path, "rb") as f:
        files = {"file": (Path(video_path).name, f, content_type)}
        try:
            response = requests.post(url, headers=headers, data=data, files=files)
        except requests.exceptions.RequestException as e:
            print(f"Network error while submitting dubbing job: {e}")
            sys.exit(1)

    if response.status_code in (400, 422):
        detail = response.json().get("detail", {}) if response.headers.get("Content-Type", "").startswith("application/json") else {}
        reason = detail.get("message", response.text) if isinstance(detail, dict) else response.text
        print(
            f"Dubbing request rejected for target language '{target_lang}' - exiting.\n"
            f"Reason: {reason}"
        )
        sys.exit(1)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        print(f"Failed to submit dubbing job: {response.text}")
        sys.exit(1)

    dubbing_id = response.json().get("dubbing_id")
    if not dubbing_id:
        print(f"No dubbing_id in response: {response.text}")
        sys.exit(1)

    print(f"Submitted dubbing job: {dubbing_id}")
    return dubbing_id


def poll_until_dubbed(api_key, dubbing_id, poll_interval, timeout):
    url = f"{API_BASE}/dubbing/{dubbing_id}"
    headers = {"xi-api-key": api_key}
    start = time.time()

    while True:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error while polling job status: {e}")
            sys.exit(1)

        status = response.json().get("status")

        if status == "dubbed":
            print("Dubbing complete.")
            return
        if status == "failed":
            error = response.json().get("error", "unknown error")
            print(f"Dubbing job failed: {error}")
            sys.exit(1)

        # ElevenLabs doesn't publicly document the full status enum (observed values
        # include "preparing", "dubbing"), so treat anything besides "dubbed"/"failed"
        # as still in progress rather than guessing at every possible value.
        elapsed = int(time.time() - start)
        if elapsed > timeout:
            print(
                f"Timed out after {timeout}s waiting for dubbing job {dubbing_id}.\n"
                f"The job may still complete - check later with this dubbing_id."
            )
            sys.exit(1)
        print(f"  ...status={status} ({elapsed}s elapsed)")
        time.sleep(poll_interval)


def download_result(api_key, dubbing_id, target_lang, out_path_base):
    url = f"{API_BASE}/dubbing/{dubbing_id}/audio/{target_lang}"
    headers = {"xi-api-key": api_key}

    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to download dubbed result: {e}")
        sys.exit(1)

    content_type = response.headers.get("Content-Type", "")
    if "mp4" in content_type:
        ext = ".mp4"
    elif "mpeg" in content_type or "mp3" in content_type:
        ext = ".mp3"
    else:
        print(f"Warning: unexpected Content-Type '{content_type}', defaulting to .mp4")
        ext = ".mp4"

    out_path = Path(str(out_path_base) + ext)
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Saved dubbed output to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Skillfy Feature 3 POC - dub a video into a regional language via ElevenLabs."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to local input video file")
    parser.add_argument(
        "--target", "-t", required=True,
        help="Target language name or code, e.g. 'hindi' or 'hi'"
    )
    parser.add_argument("--source-lang", default="auto", help="Source language code (default: auto)")
    parser.add_argument("--output", "-o", default=None, help="Output file path base (extension auto-detected)")
    parser.add_argument("--poll-interval", type=int, default=10, help="Seconds between status polls (default: 10)")
    parser.add_argument("--timeout", type=int, default=1800, help="Max seconds to wait for dubbing (default: 1800)")
    args = parser.parse_args()

    print(FREE_TIER_BANNER)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    target_lang = resolve_target_lang(args.target)
    api_key = get_api_key()

    if args.output:
        out_base = args.output
    else:
        out_base = f"{input_path.stem}_{target_lang}"

    dubbing_id = submit_dubbing_job(api_key, str(input_path), target_lang, args.source_lang)
    poll_until_dubbed(api_key, dubbing_id, args.poll_interval, args.timeout)
    download_result(api_key, dubbing_id, target_lang, out_base)

    print(FREE_TIER_BANNER)


if __name__ == "__main__":
    main()
