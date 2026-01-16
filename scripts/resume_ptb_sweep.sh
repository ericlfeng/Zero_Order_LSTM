#!/bin/bash

##############################################################################
# Resume interrupted PTB sweep runs from checkpoints
# Finds .pt checkpoint files and restarts runs that didn't complete
##############################################################################

# Configuration - should match your original sweep
RESULTS_DIR="${1:-./results_lstm3layer_ptb}"

echo "============================================================"
echo "RESUME INTERRUPTED RUNS"
echo "============================================================"
echo "[INFO] Scanning for interrupted runs in ${RESULTS_DIR}..."

# --- Detect Available GPUs ---
GPU_IDS=($(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null))
NUM_GPUS=${#GPU_IDS[@]}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[ERROR] No GPUs detected. Exiting."
    exit 1
fi
echo "[INFO] Detected ${NUM_GPUS} GPUs with IDs: ${GPU_IDS[@]}"

# Find all checkpoint files
CHECKPOINT_FILES=($(find "${RESULTS_DIR}" -name "checkpoint_*.pt" 2>/dev/null | sort))

if [ ${#CHECKPOINT_FILES[@]} -eq 0 ]; then
    echo "[INFO] No checkpoint files found in ${RESULTS_DIR}"
    exit 0
fi

echo "[INFO] Found ${#CHECKPOINT_FILES[@]} checkpoint files"
echo ""

resume_counter=0
skipped_complete=0
skipped_running=0

for CKPT_FILE in "${CHECKPOINT_FILES[@]}"; do
    # Extract run name from checkpoint filename
    BASENAME=$(basename "$CKPT_FILE" .pt)
    RUN_NAME=${BASENAME#checkpoint_}
    
    # Check if this run already completed
    if ls "${RESULTS_DIR}"/success_*"${RUN_NAME}"*.json 1>/dev/null 2>&1 || \
       ls "${RESULTS_DIR}"/converged_*"${RUN_NAME}"*.json 1>/dev/null 2>&1; then
        skipped_complete=$((skipped_complete + 1))
        continue
    fi
    
    # Check if a screen session for this run is already active
    SCREEN_NAME=$(echo "${RUN_NAME}" | cut -c1-60)
    if screen -ls 2>/dev/null | grep -q "${SCREEN_NAME}"; then
        skipped_running=$((skipped_running + 1))
        continue
    fi
    
    # Get checkpoint info
    CKPT_INFO=$(python3 -c "
import torch
ckpt = torch.load('${CKPT_FILE}', map_location='cpu', weights_only=False)
iteration = ckpt.get('iteration', 0)
max_iters = ckpt.get('args', {}).get('max_iterations', 500000)
pct = 100 * iteration / max_iters if max_iters > 0 else 0
print(f'{iteration}/{max_iters} ({pct:.1f}%)')
" 2>/dev/null)
    
    echo "[RESUME] ${RUN_NAME}"
    echo "         Progress: ${CKPT_INFO}"
    
    # Assign GPU round-robin
    gpu_index=$(( resume_counter % NUM_GPUS ))
    assigned_gpu_id=${GPU_IDS[$gpu_index]}
    
    # Launch resume in screen session
    screen -dmS "$SCREEN_NAME" bash -c "
        echo '[INFO] Resuming run: ${RUN_NAME}';
        echo '[INFO] Checkpoint: ${CKPT_FILE}';
        echo '[INFO] GPU: cuda:${assigned_gpu_id}';
        
        python rge_series_experiments.py \
            --resume_from '${CKPT_FILE}' \
            --device cuda:${assigned_gpu_id};
        
        echo '[INFO] Finished resumed run: ${RUN_NAME}';
        exec bash
    "
    
    resume_counter=$(( resume_counter + 1 ))
    sleep 1
done

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "  Resumed:          ${resume_counter}"
echo "  Skipped (done):   ${skipped_complete}"
echo "  Skipped (running): ${skipped_running}"
echo ""
if [ "$resume_counter" -gt 0 ]; then
    echo "Use 'screen -ls' to see active sessions"
    echo "Use 'screen -r <name>' to attach to a session"
fi
echo "============================================================"
