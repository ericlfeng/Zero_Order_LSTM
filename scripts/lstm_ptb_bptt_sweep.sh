#!/bin/bash

##############################################################################
# Run N-layer LSTM Penn Treebank training with Backpropagation (BPTT)
# Sweeps over model scales and learning rates
##############################################################################


########## CONFIGURE N_LAYERS HERE #############
n_layers=3

########## RUN PREFIX #############
RUN_PREFIX="lstm${n_layers}L_ptb_bptt"

# Define hyperparameter arrays:
TASKS=("penn_tree_bank")
ARCHITECTURES=("LSTM")

MODEL_SCALES=(1 2 4 8 16 32 64)
# base hidden=111 so scale 64 -> hidden=7104
hidden_size=111
memory_size=111
head_size=0
num_heads=1
input_dim=128

# PTB typically uses longer sequences
INPUT_SAMPLE_LENGTHS=(10)
MICRO_BATCH_SIZES=(1024)
MACRO_BATCH_SIZES=(1)

LEARNING_RATES=(0.001 0.0001)
WEIGHT_DECAYS=(0)
GRAD_CLIPS=(0)

# Adam optimizer settings
USE_ADAM=true
BETA1s=(0.9)
BETA2s=(0.999)

# Train on full dataset (not overfitting)
OVERFITS=(false)

# Other configurations:
LOG_INTERVAL=100
MAX_ITERS=500000
CHECKPOINT_INTERVAL=500

WANDB=true
WANDB_PROJ="Zero_Order_Opt_LSTM_PTB_BPTT"

# Function to truncate a long session name (if needed)
truncate_name() {
  echo "$1" | cut -c1-65
}

# --- Detect Available GPUs ---
echo "[INFO] Detecting available GPUs..."
GPU_IDS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
NUM_GPUS=${#GPU_IDS[@]}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[ERROR] No GPUs detected by nvidia-smi. Exiting."
    exit 1
fi
echo "[INFO] Detected ${NUM_GPUS} GPUs with IDs: ${GPU_IDS[@]}"

run_counter=0

# --- Loop over the hyperparameters ---
for TASK in "${TASKS[@]}"; do
    for ARCH in "${ARCHITECTURES[@]}"; do
        for MODEL_SCALE in "${MODEL_SCALES[@]}"; do
            for INPUT_SAMPLE_LENGTH in "${INPUT_SAMPLE_LENGTHS[@]}"; do
                for MICRO_BS in "${MICRO_BATCH_SIZES[@]}"; do
                    for MACRO_BS in "${MACRO_BATCH_SIZES[@]}"; do
                        for LR in "${LEARNING_RATES[@]}"; do
                            for BETA1 in "${BETA1s[@]}"; do
                                for BETA2 in "${BETA2s[@]}"; do
                                    for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
                                        for GRAD_CLIP in "${GRAD_CLIPS[@]}"; do
                                            for OVERFIT in "${OVERFITS[@]}"; do
                                            
                                              this_hidden_size=$(( hidden_size * MODEL_SCALE ))
                                              this_memory_size=$(( memory_size * MODEL_SCALE ))
                                              this_head_size=${head_size}
                                              this_num_head=${num_heads}
                                              this_input_size=${input_dim}
                                              
                                              # Define RUN_NAME_BASE
                                              if [ "$USE_ADAM" = true ]; then
                                                OPT_NAME="adam"
                                              else
                                                OPT_NAME="sgd"
                                              fi
                                              RUN_NAME_BASE="${RUN_PREFIX}_${run_counter}_s${MODEL_SCALE}_lr${LR}_${OPT_NAME}"

                                              EXTRA_FLAGS=""
                                              
                                              if [ "$USE_ADAM" = true ]; then
                                                EXTRA_FLAGS+=" --use_adam"
                                              fi
                                              
                                              if [ "$OVERFIT" = true ]; then
                                                EXTRA_FLAGS+=" --overfit_to_one_batch_flag"
                                              fi

                                              if [ "$WANDB" = true ]; then
                                                EXTRA_FLAGS+=" --wandb"
                                                EXTRA_FLAGS+=" --wandb_proj ${WANDB_PROJ}"
                                                EXTRA_FLAGS+=" --wandb_run_name ${RUN_NAME_BASE}"
                                              fi

                                              gpu_index=$(( run_counter % NUM_GPUS ))
                                              assigned_gpu_id=${GPU_IDS[$gpu_index]}
                                              device_string="cuda:${assigned_gpu_id}"
                                              
                                              RUN_NAME=$(truncate_name "${RUN_NAME_BASE}")
                                              echo "[INFO] Launching screen session: $RUN_NAME_BASE"
                                              
                                              screen -dmS "$RUN_NAME" bash -c "
                                              echo '[INFO] Starting run: $RUN_NAME';
                                              export WANDB_RUN_NAME=$RUN_NAME;
                                              python rge_series_experiments.py \
                                                    --model_type ${ARCH} \
                                                    --device ${device_string} \
                                                    --task ${TASK} \
                                                    --seq_length ${INPUT_SAMPLE_LENGTH} \
                                                    --hidden_size ${this_hidden_size} \
                                                    --memory_size ${this_memory_size} \
                                                    --head_size ${this_head_size} \
                                                    --num_heads ${this_num_head} \
                                                    --input_size ${this_input_size} \
                                                    --n_layers ${n_layers} \
                                                    --micro_batch_size ${MICRO_BS} \
                                                    --macro_batch_size ${MACRO_BS} \
                                                    --max_iterations ${MAX_ITERS} \
                                                    --log_interval ${LOG_INTERVAL} \
                                                    --learning_rate ${LR} \
                                                    --weight_decay ${WEIGHT_DECAY} \
                                                    --grad_clip ${GRAD_CLIP} \
                                                    --tokenizer char_level \
                                                    --beta1 ${BETA1} \
                                                    --beta2 ${BETA2} \
                                                    --solver BPTT \
                                                    --warmup_iters 1 \
                                                    --seed 42 \
                                                    --output_dir ./results_lstm${n_layers}layer_ptb_bptt \
                                                    --checkpoint_interval ${CHECKPOINT_INTERVAL} \
                                                    ${EXTRA_FLAGS} \
                                                    ;
                                              echo '[INFO] Finished run: $RUN_NAME_BASE';
                                              exec bash
                                              "
                                              run_counter=$(( run_counter + 1 ))
                                              
                                              # Rate limiting: small delay between launches
                                              LAUNCH_DELAY=${LAUNCH_DELAY:-1}
                                              if [ "$LAUNCH_DELAY" -gt 0 ]; then
                                                  sleep $LAUNCH_DELAY
                                              fi
                                             
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "[INFO] Done launching all ${run_counter} screen sessions."
echo "[INFO] Results will be in WandB project: ${WANDB_PROJ}"
echo "[INFO] Results directory: ./results_lstm${n_layers}layer_ptb_bptt"