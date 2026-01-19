#!/bin/bash
# Targeted PTB 64-scale runs across 8 GPU node (skip GPU 5)


TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Base shared flags
BASE_PARAMS="--task penn_tree_bank --model_type LSTM --n_layers 3 --micro_batch_size 1024 --macro_batch_size 1 --seq_length 10 --max_iterations 500000 --distribution rad --tokenizer char_level --warmup_iters 1 --seed 42 --beta1 0. --beta2 0."

# Wandb settings
WANDB_PARAMS="--wandb --wandb_proj LSTM_PTB_64_Targeted"

# Checkpointing (every 100 iters)
CHECKPOINT="--checkpoint_interval 100"

# Model scale 64 parameters
HIDDEN=$((111 * 64))
MEMORY=$((111 * 64))
HEAD_SIZE=0
NUM_HEADS=1
INPUT_DIM=128

ROOT_DIR="./results_ptb_64target"

mkdir -p $ROOT_DIR

####################################################################################
# NO LOOPS, skip GPU 5)
####################################################################################

# GPU0: lr=0.01, pert=96, solver=1SPSA
RUN="ptb64_lr01_p96_1_${TIMESTAMP}"
screen -dmS $RUN bash -c "
CUDA_VISIBLE_DEVICES=0 python rge_series_experiments.py \
  $BASE_PARAMS $WANDB_PARAMS $CHECKPOINT \
  --output_dir $ROOT_DIR/$RUN \
  --hidden_size $HIDDEN --memory_size $MEMORY --head_size $HEAD_SIZE --num_heads $NUM_HEADS --input_size $INPUT_DIM \
  --learning_rate 0.01 --epsilon 0.01 --solver 1SPSA --num_perturbations 96 --saturating_alpha 0.0 \
  --wandb_run_name $RUN; exec bash"
echo "Started $RUN on GPU 0"


# GPU1: lr=0.01, pert=512, solver=1SPSA
RUN="ptb64_lr01_p512_1_${TIMESTAMP}"
screen -dmS $RUN bash -c "
CUDA_VISIBLE_DEVICES=1 python rge_series_experiments.py \
  $BASE_PARAMS $WANDB_PARAMS $CHECKPOINT \
  --output_dir $ROOT_DIR/$RUN \
  --hidden_size $HIDDEN --memory_size $MEMORY --head_size $HEAD_SIZE --num_heads $NUM_HEADS --input_size $INPUT_DIM \
  --learning_rate 0.01 --epsilon 0.01 --solver 1SPSA --num_perturbations 512 --saturating_alpha 0.0 \
  --wandb_run_name $RUN; exec bash"
echo "Started $RUN on GPU 1"


# GPU2: lr=0.001, pert=96, solver=1SPSA
RUN="ptb64_lr001_p96_1_${TIMESTAMP}"
screen -dmS $RUN bash -c "
CUDA_VISIBLE_DEVICES=2 python rge_series_experiments.py \
  $BASE_PARAMS $WANDB_PARAMS $CHECKPOINT \
  --output_dir $ROOT_DIR/$RUN \
  --hidden_size $HIDDEN --memory_size $MEMORY --head_size $HEAD_SIZE --num_heads $NUM_HEADS --input_size $INPUT_DIM \
  --learning_rate 0.001 --epsilon 0.001 --solver 1SPSA --num_perturbations 96 --saturating_alpha 0.0 \
  --wandb_run_name $RUN; exec bash"
echo "Started $RUN on GPU 2"


# GPU3: lr=0.001, pert=512, solver=1SPSA
RUN="ptb64_lr001_p512_1_${TIMESTAMP}"
screen -dmS $RUN bash -c "
CUDA_VISIBLE_DEVICES=3 python rge_series_experiments.py \
  $BASE_PARAMS $WANDB_PARAMS $CHECKPOINT \
  --output_dir $ROOT_DIR/$RUN \
  --hidden_size $HIDDEN --memory_size $MEMORY --head_size $HEAD_SIZE --num_heads $NUM_HEADS --input_size $INPUT_DIM \
  --learning_rate 0.001 --epsilon 0.001 --solver 1SPSA --num_perturbations 512 --saturating_alpha 0.0 \
  --wandb_run_name $RUN; exec bash"
echo "Started $RUN on GPU 3"


#####################################
# 1.5-SPSA
#####################################

# GPU4: lr=0.01, pert=96, solver=15
RUN="ptb64_lr01_p96_15_${TIMESTAMP}"
screen -dmS $RUN bash -c "
CUDA_VISIBLE_DEVICES=4 python rge_series_experiments.py \
  $BASE_PARAMS $WANDB_PARAMS $CHECKPOINT \
  --output_dir $ROOT_DIR/$RUN \
  --hidden_size $HIDDEN --memory_size $MEMORY --head_size $HEAD_SIZE --num_heads $NUM_HEADS --input_size $INPUT_DIM \
  --learning_rate 0.01 --epsilon 0.01 --solver 1.5-SPSA --num_perturbations 96 --saturating_alpha 0.1 \
  --wandb_run_name $RUN; exec bash"
echo "Started $RUN on GPU 4"


# GPU6: lr=0.01, pert=512, solver=15
RUN="ptb64_lr01_p512_15_${TIMESTAMP}"
screen -dmS $RUN bash -c "
CUDA_VISIBLE_DEVICES=6 python rge_series_experiments.py \
  $BASE_PARAMS $WANDB_PARAMS $CHECKPOINT \
  --output_dir $ROOT_DIR/$RUN \
  --hidden_size $HIDDEN --memory_size $MEMORY --head_size $HEAD_SIZE --num_heads $NUM_HEADS --input_size $INPUT_DIM \
  --learning_rate 0.01 --epsilon 0.01 --solver 1.5-SPSA --num_perturbations 512 --saturating_alpha 0.1 \
  --wandb_run_name $RUN; exec bash"
echo "Started $RUN on GPU 6"


# GPU7: lr=0.001, pert=96, solver=15
RUN="ptb64_lr001_p96_15_${TIMESTAMP}"
screen -dmS $RUN bash -c "
CUDA_VISIBLE_DEVICES=7 python rge_series_experiments.py \
  $BASE_PARAMS $WANDB_PARAMS $CHECKPOINT \
  --output_dir $ROOT_DIR/$RUN \
  --hidden_size $HIDDEN --memory_size $MEMORY --head_size $HEAD_SIZE --num_heads $NUM_HEADS --input_size $INPUT_DIM \
  --learning_rate 0.001 --epsilon 0.001 --solver 1.5-SPSA --num_perturbations 96 --saturating_alpha 0.1 \
  --wandb_run_name $RUN; exec bash"
echo "Started $RUN on GPU 7"


# GPU7 again for final slot
RUN="ptb64_lr001_p512_15_${TIMESTAMP}"
screen -dmS $RUN bash -c "
CUDA_VISIBLE_DEVICES=7 python rge_series_experiments.py \
  $BASE_PARAMS $WANDB_PARAMS $CHECKPOINT \
  --output_dir $ROOT_DIR/$RUN \
  --hidden_size $HIDDEN --memory_size $MEMORY --head_size $HEAD_SIZE --num_heads $NUM_HEADS --input_size $INPUT_DIM \
  --learning_rate 0.001 --epsilon 0.001 --solver 1.5-SPSA --num_perturbations 512 --saturating_alpha 0.1 \
  --wandb_run_name $RUN; exec bash"
echo "Started $RUN on GPU 7"

echo "All PTB-64 targeted runs started."
echo "Results stored in: $ROOT_DIR"
echo "To view a job, use: screen -r SESSION_NAME"
echo "To detach: Ctrl+A then D"
echo "To list screens: screen -ls"
