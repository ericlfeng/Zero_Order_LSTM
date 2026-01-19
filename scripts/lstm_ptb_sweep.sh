#!/bin/bash

##############################################################################
# Run N-layer LSTM Penn Treebank training with 1.5-SPSA
# Sweeps over learning rates, model scales, and perturbations
##############################################################################



########## CONFIGURE N_LAYERS HERE #############
n_layers=3

########## RUN PREFIX #############
RUN_PREFIX="lstm${n_layers}L_ptb"

########## EARLY STOPPING CONFIGURATION #############
ENABLE_EARLY_STOPPING=false
CONVERGENCE_TRACKER_FILE="./convergence_tracker_${RUN_PREFIX}.json"

# Reset convergence tracker at start (optional - comment out to resume)
if [ "$ENABLE_EARLY_STOPPING" = true ]; then
    echo "[INFO] Resetting convergence tracker: ${CONVERGENCE_TRACKER_FILE}"
    rm -f "${CONVERGENCE_TRACKER_FILE}"
fi


# Define hyperparameter arrays:
TASKS=("penn_tree_bank")
ARCHITECTURES=("LSTM")

MODEL_SCALES=(16)
# base hidden=111 so scale 64 -> ~1B params
hidden_size=111
memory_size=111
head_size=0
num_heads=1
input_dim=128

# PTB typically uses longer sequences
INPUT_SAMPLE_LENGTHS=(10)
MICRO_BATCH_SIZES=(1024)
MACRO_BATCH_SIZES=(1)

# Learning rate sweep - PTB may need different LRs than copy task
LEARNING_RATES=(0.01 0.001)
EPSILONS=(0.1)

MAX_NUMS=(120)
WEIGHT_DECAYS=(0)
GRAD_CLIPS=(0)
SOLVERS=("1SPSA" "1.5-SPSA")

BETA1s=(0.)
BETA2s=(0.)
PROBE_PROCONDITIONINGS=(false)

SANGER_RANKS=(1)
alpha_eye_scalars=(1.0)
beta_eigen_sangers=(0)

NUM_PERTURBATIONS=(8 96 512)
saturating_alphas=(0.1)

# For PTB, you can train on full dataset or overfit to one batch
OVERFITS=(false)

# Other configurations:
LOG_INTERVAL=100
MAX_ITERS=500000
CHECKPOINT_INTERVAL=500  # Save resumable .pt checkpoint every N iterations (0 to disable)

TIE_EPS_TO_LR=true
ADAM=false
WANDB=true

WANDB_PROJ="Zero_Order_Opt_LSTM_PTB"

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
        for SOLVER in "${SOLVERS[@]}"; do
            for MODEL_SCALE in "${MODEL_SCALES[@]}"; do
                for INPUT_SAMPLE_LENGTH in "${INPUT_SAMPLE_LENGTHS[@]}"; do
                    for MICRO_BS in "${MICRO_BATCH_SIZES[@]}"; do
                        for MACRO_BS in "${MACRO_BATCH_SIZES[@]}"; do
                            for LR in "${LEARNING_RATES[@]}"; do
                                for EPS in "${EPSILONS[@]}"; do
                                for PROBE_PROCONDITIONING in "${PROBE_PROCONDITIONINGS[@]}"; do
                                for BETA1 in "${BETA1s[@]}"; do
                                for BETA2 in "${BETA2s[@]}"; do
                                for SANGER_RANK in "${SANGER_RANKS[@]}"; do
                                for beta_eigen_sanger in  "${beta_eigen_sangers[@]}"; do
                                # Only sweep saturating_alpha for 1.5-SPSA (1SPSA doesn't use it)
                                if [ "$SOLVER" = "1SPSA" ]; then
                                    alphas_to_use=(0.0)  # 1SPSA ignores alpha, so just use default
                                else
                                    alphas_to_use=("${saturating_alphas[@]}")
                                fi
                                
                                for saturating_alpha in "${alphas_to_use[@]}"; do
                                for alpha_eye_scalar in "${alpha_eye_scalars[@]}"; do
                                    for MAX_NUM in "${MAX_NUMS[@]}"; do
                                        for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
                                            for GRAD_CLIP in "${GRAD_CLIPS[@]}"; do
                                                for OVERFIT in "${OVERFITS[@]}"; do
                                                
                                                  this_hidden_size=$(( hidden_size * MODEL_SCALE ))
                                                  this_memory_size=$(( memory_size * MODEL_SCALE ))
                                                  this_head_size=${head_size}
                                                  this_num_head=${num_heads}
                                                  this_input_size=${input_dim}
                                                  
                                                  for numPert in "${NUM_PERTURBATIONS[@]}"; do
                                                      
                                                    # Define RUN_NAME_BASE FIRST (before it's used in WANDB flags)
                                                    RUN_NAME_BASE="${RUN_PREFIX}_${run_counter}_pert${numPert}_s${MODEL_SCALE}_${SOLVER}_lr${LR}_sa${saturating_alpha}"

                                                    EXTRA_FLAGS=""
                                                    if [ "$PROBE_PROCONDITIONING" = true ]; then
                                                      EXTRA_FLAGS+=" --use_probe_preconditioning"
                                                    fi
                                                    
                                                    if [ "$ADAM" = true ]; then
                                                      EXTRA_FLAGS+=" --use_adam"
                                                    fi

                                                    if [ "$TIE_EPS_TO_LR" = true ]; then
                                                       EPS=$LR
                                                    fi
                                                    
                                                    if [ "$OVERFIT" = true ]; then
                                                      EXTRA_FLAGS+=" --overfit_to_one_batch_flag"
                                                    fi
    
                                                    if [ "$WANDB" = true ]; then
                                                      EXTRA_FLAGS+=" --wandb"
                                                      EXTRA_FLAGS+=" --wandb_proj ${WANDB_PROJ}"
                                                      EXTRA_FLAGS+=" --wandb_run_name ${RUN_NAME_BASE}"
                                                    fi
                                                    
                                                    if [ "$ENABLE_EARLY_STOPPING" = true ]; then
                                                      EXTRA_FLAGS+=" --enable_early_stopping"
                                                      EXTRA_FLAGS+=" --convergence_tracker_file ${CONVERGENCE_TRACKER_FILE}"
                                                      EXTRA_FLAGS+=" --model_scale ${MODEL_SCALE}"
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
                                                          --epsilon ${EPS} \
                                                          --sanger_rank ${SANGER_RANK} \
                                                          --weight_decay ${WEIGHT_DECAY} \
                                                          --max_num ${MAX_NUM} \
                                                          --grad_clip ${GRAD_CLIP} \
                                                          --num_perturbations ${numPert} \
                                                          --tokenizer char_level \
                                                          --distribution rad \
                                                          --beta1 ${BETA1} \
                                                          --beta2 ${BETA2} \
                                                          --solver ${SOLVER} \
                                                          --sanger_qr_every 100 \
                                                          --saturating_alpha ${saturating_alpha} \
                                                          --warmup_iters 1 \
                                                          --seed 42 \
                                                          --alpha_eye_scalar ${alpha_eye_scalar} \
                                                          --beta_eigen_sanger ${beta_eigen_sanger} \
                                                          --output_dir ./results_lstm${n_layers}layer_ptb \
                                                          --checkpoint_interval ${CHECKPOINT_INTERVAL} \
                                                          ${EXTRA_FLAGS} \
                                                          ;
                                                    echo '[INFO] Finished run: $RUN_NAME_BASE';
                                                    exec bash
                                                    "
                                                    run_counter=$(( run_counter + 1 ))
                                                    
                                                    # Rate limiting: small delay between launches to prevent GPU memory overload
                                                    LAUNCH_DELAY=${LAUNCH_DELAY:-1}  # Default 1 second, set to 0 to disable
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

if [ "$ENABLE_EARLY_STOPPING" = true ]; then
    echo ""
    echo "============================================================"
    echo "CROSS-RUN EARLY STOPPING ENABLED"
    echo "============================================================"
    echo "Convergence tracker file: ${CONVERGENCE_TRACKER_FILE}"
    echo ""
    echo "To monitor convergence in real-time, run:"
    echo "  python scripts/convergence_tracker.py --tracker_file ${CONVERGENCE_TRACKER_FILE} --action watch"
    echo ""
    echo "To view current convergence summary:"
    echo "  python scripts/convergence_tracker.py --tracker_file ${CONVERGENCE_TRACKER_FILE} --action summary"
    echo "============================================================"
fi

