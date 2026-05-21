#!/bin/bash
set -x

# ============================================================
# ALFWorld Evaluation Script (Validation Seen & Unseen)
# Model: Qwen2.5-1.5B-Instruct (RewardFlow checkpoint)
# Evaluates: global_step_100, global_step_150 x seeds 0,1,2
# ============================================================

ENGINE=vllm
VLLM_ATTN_BACKEND=FLASH_ATTN
export ALFWORLD_DATA=/workspace/verl-agent/alfworld_data

N_GPUS=2
VAL_DATA_SIZE=128
SEEDS=(123 456 789)

# Base HuggingFace model (architecture init; weights overridden by resume_from_path)
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct

CHECKPOINT_DIR=/workspace/RewardFlow/results/checkpoints/alfworld/rewardflow_25_step/Qwen2.5-1.5B-Instruct
LOG_DIR=/workspace/RewardFlow/results/eval_logs/alfworld/rewardflow_25_step/Qwen2.5-1.5B-Instruct

cd /workspace/RewardFlow
export VLLM_ATTENTION_BACKEND=$VLLM_ATTN_BACKEND

# Prepare dummy data (required by the dataloader, not used during val_only)
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size 16 \
    --val_data_size $VAL_DATA_SIZE

# ============================================================
# Function: evaluate one checkpoint x one split x one seed
# Usage: run_eval <step_num> <eval_dataset> <split_name> <seed>
# ============================================================
run_eval() {
    local STEP_NUM=$1           # e.g. 100
    local EVAL_DATASET=$2       # eval_in_distribution | eval_out_of_distribution
    local SPLIT_NAME=$3         # seen | unseen
    local SEED=$4

    local CKPT_PATH=${CHECKPOINT_DIR}/global_step_${STEP_NUM}
    local LOG_FILE=${LOG_DIR}/step_${STEP_NUM}/${SPLIT_NAME}_seed${SEED}.log

    if [ ! -d "$CKPT_PATH" ]; then
        echo "ERROR: Checkpoint not found at ${CKPT_PATH}, skipping."
        return 1
    fi

    mkdir -p $(dirname $LOG_FILE)

    echo ""
    echo "============================================================"
    echo "Step  : ${STEP_NUM}  |  Split: ${SPLIT_NAME}  |  Seed: ${SEED}"
    echo "Log   : ${LOG_FILE}"
    echo "============================================================"

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=rewardflow \
        data.train_files=$HOME/data/verl-agent/text/train.parquet \
        data.val_files=$HOME/data/verl-agent/text/test.parquet \
        data.train_batch_size=16 \
        data.val_batch_size=$VAL_DATA_SIZE \
        data.max_prompt_length=2048 \
        data.max_response_length=512 \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.use_invalid_action_penalty=True \
        actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
        algorithm.use_kl_in_reward=False \
        algorithm.gamma=0.9 \
        env.env_name=alfworld/AlfredTWEnv \
        env.seed=$SEED \
        env.max_steps=50 \
        env.rollout.n=1 \
        env.resources_per_worker.num_cpus=0.1 \
        env.alfworld.eval_dataset=$EVAL_DATASET \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=1 \
        trainer.total_epochs=1 \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path=$CKPT_PATH \
        trainer.experiment_name=eval_${SPLIT_NAME}_step${STEP_NUM}_seed${SEED} \
        trainer.default_local_dir=${LOG_DIR}/step_${STEP_NUM} \
        2>&1 | tee $LOG_FILE
}

# ============================================================
# global_step_100 / global_step_150  x  seen/unseen  x  3 seeds
# ============================================================
for STEP in 100 150; do
    for SEED in "${SEEDS[@]}"; do
        run_eval $STEP eval_in_distribution     seen   $SEED
        run_eval $STEP eval_out_of_distribution unseen $SEED
    done
done

echo ""
echo "============================================================"
echo "All evaluations complete."
echo "Logs saved to: ${LOG_DIR}"
echo "============================================================"
