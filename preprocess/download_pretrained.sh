#!/usr/bin/env bash
# Download all pre-trained weights into ./pretrained/
#
# Layout produced:
#   pretrained/CLIP/ViT-B-32.pt
#   pretrained/AST/audioset_10_10.pth
#   pretrained/ImageBind/imagebind_huge.pth
set -e

cd "$(dirname "$0")/.."   # repo root

mkdir -p pretrained/CLIP pretrained/AST pretrained/ImageBind
mkdir -p modules .checkpoints

echo "==> [1/3] CLIP ViT-B/32"
CLIP_PT=pretrained/CLIP/ViT-B-32.pt
if [ ! -f "${CLIP_PT}" ]; then
    curl -fL --retry 3 -o "${CLIP_PT}" \
        'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt'
fi

echo "==> [2/3] AST audioset_10_10"
AST_PT=pretrained/AST/audioset_10_10.pth
if [ ! -f "${AST_PT}" ]; then
    # https://github.com/YuanGongND/ast: "Full AudioSet, 10 tstride, 10 fstride".
    curl -fL --retry 3 -o "${AST_PT}" \
        'https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1'
fi

echo "==> [3/3] ImageBind huge"
IB_PT=pretrained/ImageBind/imagebind_huge.pth
if [ ! -f "${IB_PT}" ]; then
    # https://github.com/facebookresearch/ImageBind/releases/tag/initial-release
    curl -fL --retry 3 -o "${IB_PT}" \
        'https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth'
fi

echo "==> done"
ls -lh pretrained/CLIP/* pretrained/AST/* pretrained/ImageBind/*
