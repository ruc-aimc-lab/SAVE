#!/usr/bin/env bash
# ImageBind teacher feature extraction.
#
# Default: 2 GPUs in parallel (cuda:0 + cuda:1), batch_size=20.
# Override via env:
#   NPROC=N           number of GPUs (default 2)
#   BATCH_SIZE=20     batch size per GPU
#   IMAGEBIND_DIR=... path to your ImageBind clone (or `pip install`-ed; optional)
#   any extra `imagebind_fe.py` flag can also be passed as positional args.
#
# Run from repo root:
#   bash fe.sh
set -e

NPROC="${NPROC:-2}"
BATCH_SIZE="${BATCH_SIZE:-20}"

pids=()
for ((i = 0; i < NPROC; i++)); do
    if [ "$i" -eq "$((NPROC - 1))" ]; then
        # Last shard runs in foreground so the script blocks until done.
        python imagebind_fe.py \
            --device "cuda:${i}" \
            --batch_size "${BATCH_SIZE}" \
            --num_parts "${NPROC}" \
            --cur_part "${i}" "$@"
    else
        python imagebind_fe.py \
            --device "cuda:${i}" \
            --batch_size "${BATCH_SIZE}" \
            --num_parts "${NPROC}" \
            --cur_part "${i}" "$@" &
        pids+=("$!")
    fi
done

# Wait for backgrounded shards.
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "[fe.sh] all ${NPROC} parts done"
