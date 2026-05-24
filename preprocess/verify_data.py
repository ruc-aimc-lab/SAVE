#!/usr/bin/env python3
"""End-to-end data sanity checker for SAVE.

Run from the repo root *after* every preparation step has been done:

    python preprocess/verify_data.py                  # default: msrvtt
    python preprocess/verify_data.py --dataset vatex

What it checks (on a per-file basis, with concrete error messages):

    1. Pretrained weights exist and load cleanly.
           pretrained/CLIP/ViT-B-32.pt
           pretrained/AST/audioset_10_10.pth
           pretrained/ImageBind/imagebind_huge.pth

    2. Annotations & ASR JSON for the chosen dataset are present and
       parseable (file names depend on the dataset, see DATASET_CONFIG).

    3. Every video_id referenced by the train / test split has a matching
       ``.mp4`` under ``data/<dataset>/VideoData/``.

    4. Every required video_id has a matching ``.wav`` under
       ``data/<dataset>/AudioData/`` *or* the global silent_file.wav exists
       as a fallback. We list how many wavs are missing.

    5. Every video_id has both ``AudioFeature/{vid}.pt`` and
       ``VideoFeature/{vid}.pt`` of shape ``(1024,)`` under
       ``data/<dataset>/FeatureData/ImageBind/``.

Returns exit code 0 on full success, 1 otherwise.
"""
import argparse
import os
import sys

import pandas as pd
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAINED = os.path.join(REPO_ROOT, "pretrained")


# Per-dataset annotation file names. Currently only MSR-VTT is fully wired
# up here (it is what the released checkpoints reproduce). For the other
# datasets, fill in the file names below to match what you placed under
# data/<dataset>/Annotations/, then re-run.
DATASET_CONFIG = {
    "msrvtt": {
        "caption_json": "Annotations/MSRVTT_data.json",
        "train_csv":    "Annotations/MSRVTT_train.9k.csv",
        "test_csv":     "Annotations/MSRVTT_JSFUSION_test.csv",
        "asr_json":     "msrvtt10k_asr_text.json",
    },
    # "vatex": {
    #     "caption_json": "Annotations/<...>.json",
    #     "train_csv":    "Annotations/<...>.csv",
    #     "test_csv":     "Annotations/<...>.csv",
    #     "asr_json":     "<...>.json",
    # },
    # "charades": { ... },
    # "lsmdc":    { ... },
}


def _ok(msg):
    print(f"  [ OK ] {msg}")


def _fail(msg, errors):
    print(f"  [FAIL] {msg}")
    errors.append(msg)


def _check_files_exist(items, errors, label):
    print(f"\n## {label}")
    for path, desc in items:
        if path is None:
            continue
        if os.path.exists(path):
            _ok(f"{desc}  ({path})")
        else:
            _fail(f"missing {desc}  ({path})", errors)


def _check_weights(errors):
    print("\n## Pretrained weights")
    weights = [
        (os.path.join(PRETRAINED, "CLIP", "ViT-B-32.pt"),         "CLIP ViT-B/32"),
        (os.path.join(PRETRAINED, "AST", "audioset_10_10.pth"),    "AST audioset_10_10"),
        (os.path.join(PRETRAINED, "ImageBind", "imagebind_huge.pth"), "ImageBind huge"),
    ]
    for path, desc in weights:
        if not os.path.exists(path):
            _fail(f"missing {desc}: {path}", errors)
            continue
        try:
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                _ok(f"{desc} loaded ({len(obj)} keys)")
            else:
                _ok(f"{desc} loaded ({type(obj).__name__})")
        except Exception as e:
            _fail(f"{desc} could not be loaded: {e}", errors)


def _collect_required_ids(data_root, cfg):
    """Read train + test csv and return the union of video_id strings.

    Different datasets use different column names; we look for the most
    common ones in order of preference.
    """
    ids = set()
    for csv_rel in [cfg["train_csv"], cfg["test_csv"]]:
        if csv_rel is None:
            continue
        path = os.path.join(data_root, csv_rel)
        df = pd.read_csv(path)
        for col in ("video_id", "vid_id", "videoid", "id"):
            if col in df.columns:
                ids.update(df[col].astype(str).tolist())
                break
        else:
            # Fall back to the first column if no recognised name.
            ids.update(df.iloc[:, 0].astype(str).tolist())
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="msrvtt",
                    choices=sorted(DATASET_CONFIG.keys()),
                    help="Which dataset under data/<dataset>/ to verify.")
    ap.add_argument("--skip_weights", action="store_true",
                    help="Skip the pretrained-weight torch.load checks.")
    args = ap.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    data_root = os.path.join(REPO_ROOT, "data", args.dataset)
    print(f"verifying dataset = {args.dataset}  (data_root = {data_root})")

    errors = []

    if not args.skip_weights:
        _check_weights(errors)

    # Annotations / ASR
    annot_items = [
        (os.path.join(data_root, rel) if rel else None,
         os.path.basename(rel) if rel else None)
        for rel in (cfg["caption_json"], cfg["train_csv"],
                    cfg["test_csv"], cfg["asr_json"])
    ]
    _check_files_exist(annot_items, errors, "Annotations & ASR")

    if any(rel and not os.path.exists(os.path.join(data_root, rel))
           for rel in (cfg["train_csv"], cfg["test_csv"])):
        print("\nFATAL: missing train/test split. Stop.")
        sys.exit(1)

    # Per-id presence.
    print(f"\n## Per-id presence (train + test of {args.dataset})")
    ids = _collect_required_ids(data_root, cfg)
    print(f"  required video_ids = {len(ids)}")

    miss_mp4, miss_wav, miss_af, miss_vf = [], [], [], []
    video_dir = os.path.join(data_root, "VideoData")
    audio_dir = os.path.join(data_root, "AudioData")
    af_dir = os.path.join(data_root, "FeatureData", "ImageBind", "AudioFeature")
    vf_dir = os.path.join(data_root, "FeatureData", "ImageBind", "VideoFeature")
    for vid in ids:
        if not os.path.exists(os.path.join(video_dir, f"{vid}.mp4")):
            miss_mp4.append(vid)
        if not os.path.exists(os.path.join(audio_dir, f"{vid}.wav")):
            miss_wav.append(vid)
        if not os.path.exists(os.path.join(af_dir, f"{vid}.pt")):
            miss_af.append(vid)
        if not os.path.exists(os.path.join(vf_dir, f"{vid}.pt")):
            miss_vf.append(vid)

    print(f"  missing mp4               : {len(miss_mp4)}  e.g. {miss_mp4[:3]}")
    print(f"  missing wav               : {len(miss_wav)}  e.g. {miss_wav[:3]}")
    print(f"    (wav-less videos use silent_file.wav fallback during training)")
    print(f"  missing AudioFeature .pt  : {len(miss_af)}  e.g. {miss_af[:3]}")
    print(f"  missing VideoFeature .pt  : {len(miss_vf)}  e.g. {miss_vf[:3]}")

    if miss_mp4:
        _fail(f"{len(miss_mp4)} required mp4 missing", errors)
    if miss_af or miss_vf:
        _fail(f"{len(miss_af) + len(miss_vf)} required teacher .pt missing "
              f"(run `bash fe.sh --dataset {args.dataset}` again, "
              f"optionally with --overwrite)", errors)

    # Spot-check a few teacher feature shapes.
    print("\n## Teacher feature shape spot-check (first 5 ids)")
    for vid in ids[:5]:
        try:
            a = torch.load(os.path.join(af_dir, f"{vid}.pt"), map_location="cpu")
            v = torch.load(os.path.join(vf_dir, f"{vid}.pt"), map_location="cpu")
            if a.shape != (1024,) or v.shape != (1024,):
                _fail(f"{vid} wrong shape audio={tuple(a.shape)} video={tuple(v.shape)}", errors)
            else:
                _ok(f"{vid}: audio {tuple(a.shape)} video {tuple(v.shape)}")
        except FileNotFoundError:
            pass  # already counted above

    # silent_file.wav (must exist regardless of wav coverage).
    silent = os.path.join(REPO_ROOT, "silent_file.wav")
    if not os.path.exists(silent):
        _fail(f"missing silent_file.wav at repo root ({silent})", errors)

    print("\n========================")
    if errors:
        print(f"FAIL: {len(errors)} issue(s).")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("OK: all checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
