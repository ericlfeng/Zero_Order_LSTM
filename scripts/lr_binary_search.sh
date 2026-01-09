#!/bin/bash

##############################################################################
# Binary search to find optimal learning rate for 1.5-SPSA
# Tests each LR for a short run to detect divergence quickly
##############################################################################

# Configuration
MODEL_SCALE=2  # Test on a small model for speed
TASK="copy"
ARCH="DNC"
SOLVER="1.5-SPSA"
SEQ_LENGTH=100
TEST_ITERATIONS=500  # Short run to detect divergence
DIVERGENCE_THRESHOLD=7.0

# Binary search parameters
LR_LOW=0.0001
LR_HIGH=0.1
TOLERANCE=0.0001  # Stop when range is smaller than this
MAX_BINARY_SEARCH_ITERATIONS=10

# Fixed hyperparameters
hidden_size=128
memory_size=128
head_size=128
num_heads=1
input_dim=32

this_hidden_size=$((hidden_size * MODEL_SCALE))
this_memory_size=$((memory_size * MODEL_SCALE))
this_head_size=$((head_size * MODEL_SCALE))
this_num_head=1
this_input_size=32

# Detect GPU
GPU_IDS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
NUM_GPUS=${#GPU_IDS[@]}
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[ERROR] No GPUs detected."
    exit 1
fi
device_string="cuda:${GPU_IDS[0]}"

echo "=========================================="
echo "Learning Rate Binary Search for 1.5-SPSA"
echo "=========================================="
echo "Testing on: Model Scale=$MODEL_SCALE, Task=$TASK"
echo "Test iterations: $TEST_ITERATIONS"
echo "Initial range: [$LR_LOW, $LR_HIGH]"
echo ""

# Binary search loop
iteration=0
best_stable_lr=0
best_unstable_lr=$LR_HIGH

while (( $(echo "$LR_HIGH - $LR_LOW > $TOLERANCE" | bc -l) )) && [ $iteration -lt $MAX_BINARY_SEARCH_ITERATIONS ]; do
    iteration=$((iteration + 1))
    
    # Test middle point
    LR_TEST=$(echo "scale=6; ($LR_LOW + $LR_HIGH) / 2" | bc)
    
    echo "----------------------------------------"
    echo "Iteration $iteration: Testing LR=$LR_TEST"
    echo "Current range: [$LR_LOW, $LR_HIGH]"
    
    # Create unique output directory for this test
    TEST_OUTPUT_DIR="./lr_search_temp/test_lr_${LR_TEST}"
    mkdir -p "$TEST_OUTPUT_DIR"
    
    # Run short training
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
        --learning_rate ${LR_TEST} \
        --epsilon ${LR_TEST} \
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
        --output_dir ${TEST_OUTPUT_DIR} \
        --overfit_to_one_batch_flag \
        2>&1 | tee ${TEST_OUTPUT_DIR}/log.txt
    
    # Check if it diverged
    if grep -q "FAILED DIVERGING" ${TEST_OUTPUT_DIR}/log.txt; then
        echo "❌ DIVERGED - LR too high"
        LR_HIGH=$LR_TEST
        best_unstable_lr=$LR_TEST
    elif grep -q "SUCCESS FINISHED" ${TEST_OUTPUT_DIR}/log.txt; then
        echo "✅ CONVERGED - Could try higher LR"
        LR_LOW=$LR_TEST
        best_stable_lr=$LR_TEST
    else
        # Check final loss
        final_loss=$(grep "Train Loss:" ${TEST_OUTPUT_DIR}/log.txt | tail -1 | grep -oP 'Train Loss: \K[0-9.]+')
        
        if (( $(echo "$final_loss > $DIVERGENCE_THRESHOLD" | bc -l) )); then
            echo "❌ HIGH LOSS ($final_loss) - Likely unstable"
            LR_HIGH=$LR_TEST
            best_unstable_lr=$LR_TEST
        else
            echo "✅ STABLE ($final_loss) - Could try higher LR"
            LR_LOW=$LR_TEST
            best_stable_lr=$LR_TEST
        fi
    fi
    
    echo "Updated range: [$LR_LOW, $LR_HIGH]"
    echo ""
done

# Recommend a conservative value (80% of highest stable LR)
recommended_lr=$(echo "scale=6; $best_stable_lr * 0.8" | bc)

echo "=========================================="
echo "Binary Search Complete!"
echo "=========================================="
echo "Best stable LR found: $best_stable_lr"
echo "First unstable LR: $best_unstable_lr"
echo "RECOMMENDED LR: $recommended_lr (80% of best stable)"
echo ""
echo "To use this LR in your full sweep:"
echo "  sed -i 's/LEARNING_RATES=.*/LEARNING_RATES=($recommended_lr)/' scripts/series_1.5SPSA_hpp_sweeps.sh"
echo ""
echo "Cleaning up temporary files..."
rm -rf ./lr_search_temp

echo "Done!"

