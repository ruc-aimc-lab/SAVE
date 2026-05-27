#!/usr/bin/env bash
# Train SAVE on VATEX.
#
# Override via env (all optional):
#   DATA_PATH=./data/vatex
#   OUTPUT_DIR=./outputs/save_vatex
#   NPROC=4                        # number of GPUs
#   CUDA_VISIBLE_DEVICES=0,1,2,3   # which GPUs
#   BATCH_SIZE=128                 # global batch size (across GPUs)
#
# Defaults match the settings in old vatex script.

set -e

DATA_PATH="${DATA_PATH:-./data/vatex}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/save_vatex}"
NPROC="${NPROC:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"

export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-24}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"

RPort=$(shuf -i 10000-19999 -n1)
Margin=0.05
beta=0.2
Tau=1.0

python -m torch.distributed.launch \
    --master_port "${RPort}" \
    --nproc_per_node="${NPROC}" \
    main_task_retrieval.py \
    --do_train --num_thread_reader=12 \
    --epochs=5 --batch_size="${BATCH_SIZE}" --n_display=10 \
    --train_csv "${DATA_PATH}/Annotations/VATEX_train.csv" \
    --val_csv   "${DATA_PATH}/Annotations/VATEX_val.csv" \
    --test_csv  "${DATA_PATH}/Annotations/VATEX_test.csv" \
    --data_path "${DATA_PATH}/Annotations" \
    --features_path "${DATA_PATH}/VideoData" \
    --audio_path    "${DATA_PATH}/AudioData" \
    --asr_path      "${DATA_PATH}/Annotations/vatex_asr_text.json" \
    --lr 1e-4 --coef_lr 1e-3 \
    --max_words 32 --max_frames 12 --batch_size_val 32 \
    --datatype vatex \
    --feature_framerate 1 --freeze_layer_num 12 \
    --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 --eval_max_frame 12 \
    --temperature "${Tau}" --warmup_proportion 0.1 \
    --cross_num_hidden_layers 4 --audio_query_layers 4 \
    --beta "${beta}" --margin_BD "${Margin}" --gradient_accumulation_steps 1 \
    --output_dir "${OUTPUT_DIR}"