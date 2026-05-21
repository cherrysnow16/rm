#!/bin/bash
set -x

MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
ENGINE=vllm
VLLM_ATTN_BACKEND=FLASH_ATTN   # XFORMERS / FLASH_ATTN / FLASHINFER
export ALFWORLD_DATA=/workspace/verl-agent/alfworld_data
SEED=0
N_GPUS=2
TOTAL_EPOCHS=150
TRAIN_DATA_SIZE=16
VAL_DATA_SIZE=128
GROUP_SIZE=8
EXPERIMENT_NAME=rewardflow_seed${SEED}

# wandb
WANDB_PROJECT=verl_agent_alfworld
# WANDB_API_KEY=your_key_here  # 필요시 주석 해제

# 저장 경로: /workspace/RewardFlow/results/checkpoints/alfworld/{experiment_name}/{model_name}/global_step_XXX
MODEL_NAME=$(basename $MODEL_PATH)
CHECKPOINT_DIR=/workspace/RewardFlow/results/checkpoints/alfworld/${EXPERIMENT_NAME}/${MODEL_NAME}

# ============================================================

cd /workspace/RewardFlow
export VLLM_ATTENTION_BACKEND=$VLLM_ATTN_BACKEND

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $TRAIN_DATA_SIZE \
    --val_data_size $VAL_DATA_SIZE

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rewardflow \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
    trainer.val_before_train=True $@
