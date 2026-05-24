"""ImageBind teacher feature extraction for SAVE.

Reads .mp4 files directly (no pre-extracted frames) and writes audio /
video teacher features as torch.Tensor ``.pt`` files.

Default layout (relative to the repo root, for ``--dataset msrvtt``):
    data/msrvtt/FeatureData/ImageBind/AudioFeature/{video_id}.pt
    data/msrvtt/FeatureData/ImageBind/VideoFeature/{video_id}.pt

For other datasets, pass ``--dataset vatex`` etc.; all paths are then
resolved under ``data/<dataset>/`` automatically. You can also override
each directory individually via ``--video_dir`` / ``--audio_dir`` /
``--out_audio_dir`` / ``--out_video_dir``.

Behaviour:
  * Frames are sampled (32 evenly-spaced) in-memory via OpenCV.
  * For videos without a corresponding ``.wav`` in --audio_dir, the script
    falls back to ``silent_file.wav`` so every video still gets a feature.
  * AppleDouble ``._video*.mp4`` metadata files are filtered out.

Environment variables (all optional):
  * ``IMAGEBIND_DIR``  -- path to your ImageBind clone (containing ``imagebind/``).
                          If unset and ``--imagebind_dir`` not given, we expect
                          ``imagebind`` to be importable (``pip install -e .``
                          inside the upstream ImageBind repo, for example).

CLI overrides everything; env vars are only fallback.
"""
import argparse
import math
import os
import sys
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Repository root (= directory containing this file).
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ImageBind requires `from timm.layers import ...`, but our pinned timm 0.4.5
# only provides `timm.models.layers`. Alias before the imagebind import.
try:
    import timm.models.layers as _timm_layers  # noqa: WPS433
    sys.modules.setdefault("timm.layers", _timm_layers)
except ImportError:
    pass


def _add_imagebind_to_path(cli_value):
    p = cli_value or os.environ.get("IMAGEBIND_DIR", "")
    if p:
        sys.path.append(p)


# We delay the imagebind import until after sys.path is patched (in main()).
# Forward declarations to make linters happy.
data = None
imagebind_model = None
ModalityType = None


def _import_imagebind():
    global data, imagebind_model, ModalityType
    from imagebind import data as _data  # noqa: WPS433
    from imagebind.models import imagebind_model as _ib  # noqa: WPS433
    from imagebind.models.imagebind_model import ModalityType as _MT  # noqa: WPS433
    data = _data
    imagebind_model = _ib
    ModalityType = _MT


warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Same vision transform that ImageBind itself uses internally.
_VISION_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


def sample_frames_from_mp4(mp4_path: str, num_samples: int = 32) -> torch.Tensor:
    """Sample ``num_samples`` frames evenly from an mp4.

    Returns a normalised tensor of shape ``(num_samples, 3, 224, 224)``.
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Fall back to sequential read then resample.
        frames_bgr = []
        ok, fr = cap.read()
        while ok:
            frames_bgr.append(fr)
            ok, fr = cap.read()
        cap.release()
        if not frames_bgr:
            raise RuntimeError(f"no frames in {mp4_path}")
        idxs = np.linspace(0, len(frames_bgr) - 1, num_samples).astype(int)
        chosen = [frames_bgr[i] for i in idxs]
    else:
        idxs = np.linspace(0, total - 1, num_samples).astype(int)
        chosen = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if not ok or fr is None:
                # Some codecs misreport seek capability; replay sequentially.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                fr = None
                for _ in range(int(idx) + 1):
                    ok, _fr = cap.read()
                    if not ok:
                        break
                    fr = _fr
                if fr is None:
                    fr = np.zeros((224, 224, 3), dtype=np.uint8)
            chosen.append(fr)
        cap.release()

    tensors = []
    for bgr in chosen:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensors.append(_VISION_TRANSFORM(pil))
    return torch.stack(tensors, dim=0)  # (N,3,224,224)


def main(args):
    _add_imagebind_to_path(args.imagebind_dir)
    _import_imagebind()

    device = args.device
    print(f"[Device {device}] loading ImageBind huge ...")
    model = imagebind_model.imagebind_huge(pretrained=True).to(device)
    model.eval()

    video_dir = args.video_dir
    audio_dir = args.audio_dir
    out_video_dir = args.out_video_dir
    out_audio_dir = args.out_audio_dir
    silent_wav = args.silent_wav
    if not os.path.isfile(silent_wav):
        raise FileNotFoundError(
            f"silent_wav not found: {silent_wav}. Default expects "
            "`silent_file.wav` at the repo root."
        )
    os.makedirs(out_video_dir, exist_ok=True)
    os.makedirs(out_audio_dir, exist_ok=True)

    # Discover videos by mp4 listing.
    # Exclude AppleDouble metadata files like `._videoX.mp4`.
    all_video_ids = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(video_dir)
        if f.endswith(".mp4") and not f.startswith("._")
    )
    if args.limit > 0:
        all_video_ids = all_video_ids[: args.limit]

    # Partition for multi-GPU sharding.
    chunks = np.array_split(all_video_ids, args.num_parts)
    video_ids = list(chunks[args.cur_part])
    print(
        f"[Device {device}] processing {len(video_ids)} videos "
        f"(part {args.cur_part + 1}/{args.num_parts})"
    )

    # Optional skip already-extracted.
    if not args.overwrite:
        kept = []
        for vid in video_ids:
            ap = os.path.join(out_audio_dir, f"{vid}.pt")
            vp = os.path.join(out_video_dir, f"{vid}.pt")
            if os.path.exists(ap) and os.path.exists(vp):
                continue
            kept.append(vid)
        skipped = len(video_ids) - len(kept)
        if skipped:
            print(
                f"[Device {device}] skip {skipped} already-extracted, "
                f"{len(kept)} to do"
            )
        video_ids = kept

    if not video_ids:
        print(f"[Device {device}] nothing to do")
        return

    bs = args.batch_size
    nbat = math.ceil(len(video_ids) / bs)
    pbar = tqdm(range(nbat), desc=f"{device}", dynamic_ncols=True)

    for i in pbar:
        batch_ids = video_ids[i * bs : (i + 1) * bs]

        # Audio paths: silent fallback for missing wav.
        batch_audio_paths = []
        for vid in batch_ids:
            ap = os.path.join(audio_dir, f"{vid}.wav")
            batch_audio_paths.append(ap if os.path.exists(ap) else silent_wav)

        # Vision tensor: (bs * 32, 3, 224, 224).
        vision_chunks = []
        for vid in batch_ids:
            mp4 = os.path.join(video_dir, f"{vid}.mp4")
            try:
                v = sample_frames_from_mp4(mp4, num_samples=32)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {vid} mp4 read fail: {e}; using zeros")
                v = torch.zeros(32, 3, 224, 224)
            vision_chunks.append(v)
        vision = torch.cat(vision_chunks, dim=0).to(device)

        audio = data.load_and_transform_audio_data(batch_audio_paths, device)

        with torch.no_grad():
            embeddings = model({
                ModalityType.AUDIO: audio,
                ModalityType.VISION: vision,
            })
            audio_emb = embeddings[ModalityType.AUDIO]                       # (bs, D_a)
            video_emb = embeddings[ModalityType.VISION]                      # (bs*32, D_v)
            video_emb = video_emb.view(-1, 32, video_emb.size(-1)).mean(1)   # (bs, D_v)

        for j, vid in enumerate(batch_ids):
            torch.save(
                audio_emb[j].detach().cpu(),
                os.path.join(out_audio_dir, f"{vid}.pt"),
            )
            torch.save(
                video_emb[j].detach().cpu(),
                os.path.join(out_video_dir, f"{vid}.pt"),
            )


def parse_args():
    p = argparse.ArgumentParser(description="Extract ImageBind teacher features.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--num_parts", type=int, default=1,
                   help="number of shards (e.g. 2 for two GPUs).")
    p.add_argument("--cur_part", type=int, default=0,
                   help="which shard this process owns (0-indexed).")
    p.add_argument("--limit", type=int, default=0,
                   help="If > 0, only process the first N video_ids (smoke test).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-compute even if .pt already exists.")

    p.add_argument("--imagebind_dir", default="",
                   help="Path to ImageBind clone (containing the `imagebind/` "
                        "package). Empty = rely on `IMAGEBIND_DIR` env var or "
                        "`pip install`-ed package.")

    p.add_argument("--dataset", default="msrvtt",
                   help="Dataset name; resolves all --*_dir defaults to "
                        "data/<dataset>/... when those flags are not given. "
                        "Default: msrvtt.")
    p.add_argument("--video_dir", default=None,
                   help="Directory containing .mp4 files. "
                        "Defaults to data/<dataset>/VideoData.")
    p.add_argument("--audio_dir", default=None,
                   help="Directory containing .wav files. "
                        "Defaults to data/<dataset>/AudioData.")
    p.add_argument("--out_audio_dir", default=None,
                   help="Output directory for audio teacher features. "
                        "Defaults to data/<dataset>/FeatureData/ImageBind/AudioFeature.")
    p.add_argument("--out_video_dir", default=None,
                   help="Output directory for video teacher features. "
                        "Defaults to data/<dataset>/FeatureData/ImageBind/VideoFeature.")
    p.add_argument("--silent_wav",
                   default=os.path.join(_REPO_ROOT, "silent_file.wav"),
                   help="Wav used as fallback for videos missing an audio track.")
    args = p.parse_args()

    # Resolve --dataset-based defaults.
    ds_root = os.path.join(_REPO_ROOT, "data", args.dataset)
    if args.video_dir is None:
        args.video_dir = os.path.join(ds_root, "VideoData")
    if args.audio_dir is None:
        args.audio_dir = os.path.join(ds_root, "AudioData")
    if args.out_audio_dir is None:
        args.out_audio_dir = os.path.join(ds_root, "FeatureData", "ImageBind", "AudioFeature")
    if args.out_video_dir is None:
        args.out_video_dir = os.path.join(ds_root, "FeatureData", "ImageBind", "VideoFeature")
    return args


if __name__ == "__main__":
    main(parse_args())
