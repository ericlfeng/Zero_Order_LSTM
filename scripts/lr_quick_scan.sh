#!/bin/bash

##############################################################################
# Quick parallel LR scan - test multiple LRs simultaneously
# Faster than binary search for finding the divergence boundary
##############################################################################

# Configuration
MODEL_SCALES=(1 2)  # Test on small models only
TASK="copy"
ARCH="DNC"
SOLVER="1.5-SPSA"
SEQ_LENGTH=100
TEST_ITERATIONS=300  # Very short - just to detect divergence

# Test these learning rates in parallel
TEST_LRS=(0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005)

# Fixed hyperparameters
hidden_size=128
memory_size=128
head_size=128
num_heads=1
input_dim=32

# Kill any existing screen sessions
echo "[INFO] Killing existing screen sessions..."
screen -ls | grep '\.' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null

# Detect GPUs
GPU_IDS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
NUM_GPUS=${#GPU_IDS[@]}
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[ERROR] No GPUs detected."
    exit 1
fi

echo "=========================================="
echo "Quick LR Scan for 1.5-SPSA"
echo "=========================================="
echo "Testing LRs: ${TEST_LRS[@]}"
echo "Model scales: ${MODEL_SCALES[@]}"
echo "Test iterations: $TEST_ITERATIONS"
echo "Running on $NUM_GPUS GPUs"
echo ""

run_counter=0
mkdir -p ./lr_scan_results

# Launch all tests in parallel
for MODEL_SCALE in "${MODEL_SCALES[@]}"; do
    this_hidden_size=$((hidden_size * MODEL_SCALE))
    this_memory_size=$((memory_size * MODEL_SCALE))
    this_head_size=$((head_size * MODEL_SCALE))
    this_num_head=1
    this_input_size=32
    
    for LR in "${TEST_LRS[@]}"; do
        # Assign GPU
        gpu_index=$((run_counter % NUM_GPUS))
        assigned_gpu_id=${GPU_IDS[$gpu_index]}
        device_string="cuda:${assigned_gpu_id}"
        
        SESSION_NAME="lr_scan_scale${MODEL_SCALE}_lr${LR}"
        OUTPUT_DIR="./lr_scan_results/scale${MODEL_SCALE}_lr${LR}"
        
        echo "[INFO] Launching: Scale=$MODEL_SCALE, LR=$LR on GPU $assigned_gpu_id"
        
        screen -dmS "$SESSION_NAME" bash -c "
        python rge_series_experiments.py \
            --model_type ${ARCH} \
            --device ${device_string} \
            --task ${TASK} \
            --seq_length ${SEQ_LENGTH} \
            --hidden_size ${this_hidden_size} \
            --memory_size ${this_memory_size} \
            --head_size ${this_head_size} \
            --num_heads ${this_num_head} \
            --input_size ${this_input_size} \
            --micro_batch_size 1 \
            --macro_batch_size 1 \
            --max_iterations ${TEST_ITERATIONS} \
            --log_interval 50 \
            --learning_rate ${LR} \
            --epsilon ${LR} \
            --sanger_rank 1 \
            --weight_decay 0 \
            --max_num 120 \
            --grad_clip 0 \
            --num_perturbations 8 \
            --tokenizer char_level \
            --distribution rad \
            --beta1 0. \
            --beta2 0. \
            --solver ${SOLVER} \
            --sanger_qr_every 100 \
            --saturating_alpha 0.1 \
            --warmup_iters 1 \
            --seed 42 \
            --alpha_eye_scalar 1.0 \
            --beta_eigen_sanger 0 \
            --output_dir ${OUTPUT_DIR} \
            --overfit_to_one_batch_flag \
            2>&1 | tee ${OUTPUT_DIR}/log.txt
        echo 'Done'
        exec bash
        "
        
        run_counter=$((run_counter + 1))
        sleep 0.5
    done
done

total_runs=$run_counter
echo ""
echo "[INFO] Launched $total_runs tests"
echo "[INFO] Waiting for completion (this will take ~5-15 minutes)..."
echo ""

# Wait for all to complete
while [ $(screen -ls 2>/dev/null | grep "lr_scan_" | wc -l) -gt 0 ]; do
    running=$(screen -ls 2>/dev/null | grep "lr_scan_" | wc -l)
    echo "[INFO] $running/$total_runs tests still running..."
    sleep 10
done

echo ""
echo "=========================================="
echo "LR Scan Complete! Analyzing results..."
echo "=========================================="
echo ""

# Analyze results
for MODEL_SCALE in "${MODEL_SCALES[@]}"; do
    echo "Model Scale $MODEL_SCALE:"
    echo "  LR      | Status      | Final Loss"
    echo "  --------|-------------|------------"
    
    for LR in "${TEST_LRS[@]}"; do
        LOG_FILE="./lr_scan_results/scale${MODEL_SCALE}_lr${LR}/log.txt"
        
        if [ -f "$LOG_FILE" ]; then
            if grep -q "SUCCESS FINISHED" "$LOG_FILE"; then
                status="✅ SUCCESS"
                final_loss=$(grep "Train Loss:" "$LOG_FILE" | tail -1 | grep -oP 'Train Loss: \K[0-9.]+')
            elif grep -q "FAILED DIVERGING" "$LOG_FILE"; then
                status="❌ DIVERGED"
                final_loss="NaN/High"
            else
                final_loss=$(grep "Train Loss:" "$LOG_FILE" | tail -1 | grep -oP 'Train Loss: \K[0-9.]+' || echo "?")
                if (( $(echo "$final_loss > 7.0" | bc -l 2>/dev/null || echo 0) )); then
                    status="⚠️  UNSTABLE"
                else
                    status="⏱️  TRAINING"
                fi
            fi
        else
            status="❓ NO DATA"
            final_loss="?"
        fi
        
        printf "  %-7s | %-11s | %s\n" "$LR" "$status" "$final_loss"
    done
    echo ""
done

echo "=========================================="
echo "Recommendations:"
echo "=========================================="
echo ""
echo "Look for the HIGHEST LR that shows ✅ SUCCESS or ⏱️ TRAINING"
echo "Avoid LRs that show ❌ DIVERGED or ⚠️ UNSTABLE"
echo ""
echo "To run full sweep with a chosen LR (e.g., 0.005):"
echo "  sed -i 's/LEARNING_RATES=.*/LEARNING_RATES=(0.005)/' scripts/series_1.5SPSA_hpp_sweeps.sh"
echo "  sed -i 's/MODEL_SCALES=.*/MODEL_SCALES=(1 2 4 8 16)/' scripts/series_1.5SPSA_hpp_sweeps.sh"
echo "  ./scripts/series_1.5SPSA_hpp_sweeps.sh"
echo ""

# Optional: cleanup
read -p "Delete scan results? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf ./lr_scan_results
    echo "Cleaned up."
fi

