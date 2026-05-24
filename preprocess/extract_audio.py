#!/usr/bin/env python3
"""Extract 16 kHz mono WAV from .mp4 clips in parallel with ffmpeg.

By default, run from the repo root for MSR-VTT:

    python preprocess/extract_audio.py [--workers 32] [--limit N] [--overwrite]

For other datasets:

    python preprocess/extract_audio.py --dataset vatex
    python preprocess/extract_audio.py --dataset charades

Or pass arbitrary paths:

    python preprocess/extract_audio.py \
        --video_dir /path/to/VideoData --out_dir /path/to/AudioData

Behaviour:
  * Skips videos with no audio stream silently (counted in the summary).
  * Skips already-extracted files unless ``--overwrite``.
  * Filters AppleDouble metadata files (``._video*.mp4``).
  * Progress via tqdm.

Requires ``ffmpeg`` and ``ffprobe`` on PATH.
"""
import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from tqdm import tqdm


def has_audio_stream(mp4_path: str) -> bool:
    """Return True if the mp4 has at least one audio stream."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", mp4_path],
            capture_output=True, text=True, timeout=10,
        )
        return "audio" in out.stdout
    except Exception:
        return False


def extract_one(args):
    mp4_path, out_path, overwrite = args
    if (not overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return ("skip_exists", os.path.basename(mp4_path))
    if not has_audio_stream(mp4_path):
        return ("no_audio", os.path.basename(mp4_path))
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", mp4_path,
            "-vn",                  # no video
            "-ac", "1",             # mono
            "-ar", "16000",         # 16 kHz
            "-acodec", "pcm_s16le", # standard wav
            out_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return ("ffmpeg_err", f"{os.path.basename(mp4_path)} :: {r.stderr[:120]}")
        return ("ok", os.path.basename(mp4_path))
    except Exception as e:
        return ("exc", f"{os.path.basename(mp4_path)} :: {e}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="msrvtt",
                    help="Dataset name; resolves --video_dir/--out_dir to "
                         "data/<dataset>/{VideoData,AudioData} when those "
                         "flags are not given. Default: msrvtt.")
    ap.add_argument("--video_dir", default=None,
                    help="Directory containing .mp4 files. "
                         "Defaults to data/<dataset>/VideoData.")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for .wav files. "
                         "Defaults to data/<dataset>/AudioData.")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process the first N videos (smoke test).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.video_dir is None:
        args.video_dir = os.path.join(repo_root, "data", args.dataset, "VideoData")
    if args.out_dir is None:
        args.out_dir = os.path.join(repo_root, "data", args.dataset, "AudioData")

    os.makedirs(args.out_dir, exist_ok=True)
    # Filter AppleDouble files (`._video*.mp4`) which are not real videos.
    mp4s = sorted(p for p in glob(os.path.join(args.video_dir, "*.mp4"))
                  if not os.path.basename(p).startswith("._"))
    if args.limit > 0:
        mp4s = mp4s[: args.limit]
    print(f"[extract_audio] {len(mp4s)} videos -> {args.out_dir} (workers={args.workers})")

    tasks = []
    for p in mp4s:
        vid = os.path.splitext(os.path.basename(p))[0]
        tasks.append((p, os.path.join(args.out_dir, f"{vid}.wav"), args.overwrite))

    counts = {"ok": 0, "skip_exists": 0, "no_audio": 0, "ffmpeg_err": 0, "exc": 0}
    err_examples = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(extract_one, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc="extract"):
            tag, msg = f.result()
            counts[tag] += 1
            if tag in ("ffmpeg_err", "exc") and len(err_examples) < 5:
                err_examples.append(msg)

    print("\n== summary ==")
    for k, v in counts.items():
        print(f"  {k:12s} {v}")
    if err_examples:
        print("\nfirst few errors:")
        for m in err_examples:
            print(" ", m)


if __name__ == "__main__":
    main()
