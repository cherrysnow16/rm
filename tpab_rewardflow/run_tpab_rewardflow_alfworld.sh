#!/bin/bash
set -x

# ============================================================
# TPAB-RewardFlow ALFWorld Training Script
# Usage:
#   bash tpab_rewardflow/run_tpab_rewardflow_alfworld.sh [vllm|sglang]
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

ENGINE="vllm"
if [[ $# -gt 0 ]]; then
    case "$1" in
        vllm|sglang)
            ENGINE="$1"
            shift
            ;;
    esac
fi

# ---- Paths ----
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export ALFWORLD_DATA="${ALFWORLD_DATA:-/home/jbnu_hyun/.cache/alfworld}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

# Blackwell sm_120 stability:
#   - LD_PRELOAD system NCCL 2.30.4 over torch's bundled 2.26.2+cuda12.2.
#     The bundled NCCL crashes inside _sync_params_and_buffers (FSDP init
#     broadcast) on sm_120 with "illegal memory access". NCCL is 2.x ABI
#     forward-compatible so the newer .so works with torch linked at 2.26.
#   - TORCH_NCCL_BLOCKING_WAIT=1: surface async NCCL errors immediately so
#     we don't get the misleading watchdog crash chain after the real fault.
SYSTEM_NCCL=/usr/lib/x86_64-linux-gnu/libnccl.so.2
if [[ -f "$SYSTEM_NCCL" ]]; then
    export LD_PRELOAD="${SYSTEM_NCCL}${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

# ---- Experiment ----
SEED=0
N_GPUS=4
TOTAL_EPOCHS=100
TRAIN_DATA_SIZE=16
VAL_DATA_SIZE=128
GROUP_SIZE=8
EXPERIMENT_NAME="tpab_rewardflow_seed${SEED}"
WANDB_PROJECT="verl_agent_alfworld"

MODEL_NAME=$(basename $MODEL_PATH)
CHECKPOINT_DIR=/workspace/rewardflow_sciworld/results/checkpoints/alfworld/${EXPERIMENT_NAME}/${MODEL_NAME}

# ============================================================

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $TRAIN_DATA_SIZE \
    --val_data_size $VAL_DATA_SIZE

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=tpab_rewardflow \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$TRAIN_DATA_SIZE \
    data.val_batch_size=$VAL_DATA_SIZE \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.9 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=$SEED \
    env.max_steps=25 \
    env.rollout.n=$GROUP_SIZE \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=True "$@"
