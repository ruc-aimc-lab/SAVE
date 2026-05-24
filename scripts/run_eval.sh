#!/usr/bin/env bash
# Evaluate a trained SAVE checkpoint on MSR-VTT 1k test split.
#
# Required:
#   INIT_MODEL=path/to/pytorch_model.bin.4    (trained checkpoint)
# Optional:
#   DATA_PATH=./data/msrvtt
#   OUTPUT_DIR=./outputs/eval_msrvtt9k
#   CUDA_VISIBLE_DEVICES=0
set -e

DATA_PATH="${DATA_PATH:-./data/msrvtt}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval_msrvtt9k}"
INIT_MODEL="${INIT_MODEL:-./outputs/save_msrvtt9k/pytorch_model.bin.4}"

if [ ! -f "${INIT_MODEL}" ]; then
    echo "ERROR: INIT_MODEL not found at: ${INIT_MODEL}" >&2
    echo "Set INIT_MODEL=<path-to-checkpoint> before running this script." >&2
    exit 1
fi

export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-24}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"

RPort=$(shuf -i 10000-19999 -n1)
Margin=0.1
beta=0.2
Tau=1.0

python -m torch.distributed.launch \
    --master_port "${RPort}" \
    --nproc_per_node=1 \
    main_task_retrieval.py \
    --do_eval --num_thread_reader=12 \
    --epochs=5 --batch_size=128 --n_display=10 \
    --train_csv "${DATA_PATH}/Annotations/MSRVTT_train.9k.csv" \
    --val_csv   "${DATA_PATH}/Annotations/MSRVTT_JSFUSION_test.csv" \
    --data_path "${DATA_PATH}/Annotations/MSRVTT_data.json" \
    --features_path "${DATA_PATH}/VideoData" \
    --audio_path    "${DATA_PATH}/AudioData" \
    --asr_path      "${DATA_PATH}/msrvtt10k_asr_text.json" \
    --lr 1e-4 \
    --max_words 32 --max_frames 12 --batch_size_val 32 \
    --datatype msrvtt --expand_msrvtt_sentences \
    --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 12 \
    --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 --eval_max_frame 12 \
    --temperature "${Tau}" --warmup_proportion 0.1 \
    --cross_num_hidden_layers 4 --audio_query_layers 4 \
    --beta "${beta}" --margin_BD "${Margin}" --gradient_accumulation_steps 1 \
    --init_model "${INIT_MODEL}" \
    --output_dir "${OUTPUT_DIR}"
