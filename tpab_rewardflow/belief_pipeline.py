"""
Belief Construction Pipeline for TPAB-RewardFlow.

Stage 1: Decompose task into ordered subgoals using successful trajectories (1 LLM call per uid-group).
Stage 2: Identify when each subgoal is first achieved per trajectory, then assign belief indices (1 batched LLM call).
"""

import re
import numpy as np

from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

STAGE1_PROMPT_TEMPLATE = """You are analyzing successful trajectories of a goal-directed task.

Your task is: {task_description}

Below are successful trajectories that completed the task:
{success_trajectories}

Your goal is to identify the minimal set of high-level subgoals that are necessary and sufficient to complete the task.

To do this:
1. Carefully read the task description and understand what the final goal state must be.
2. Look for common patterns across all successful trajectories — which key actions appear in every success?
3. Abstract these common key actions into high-level subgoals that reflect meaningful progress toward the task goal.

Requirements:
- Ignore low-level navigation actions (e.g. "go to", "inventory", "open")
- Focus only on actions that directly advance the task goal
- Each subgoal should be a short phrase (at most 5 words)
- Subgoals must be mutually exclusive and collectively exhaustive
- There should be at most 5 subgoals
- Order them as they must occur to complete the task

First, reason about the task and trajectories step by step within <think> </think> tags.
Then, output the subgoals within <subgoals> </subgoals> tags.

Output format:
<think>
...
</think>
<subgoals>
1. <subgoal>
2. <subgoal>
...
</subgoals>"""

STAGE2_SUCCESS_PROMPT_TEMPLATE = """You are analyzing a trajectory of a goal-directed task.

Your task is: {task_description}

The task is decomposed into the following ordered subgoals:
{subgoals}

Below is the full trajectory:
{full_trajectory}

This trajectory SUCCESSFULLY completed the task. All subgoals were achieved.
Identify the step at which each subgoal is first achieved.

Step assignment rules:
- The step number must be the exact "Step N" number from the trajectory.
- A subgoal is achieved only when the action at that step directly and completely fulfills it — not a preparatory or partial action.
- If multiple steps could qualify, use the earliest one.
- Write the exact action text from that step alongside the step number.

In your reasoning, explicitly state for each subgoal:
- Which step and what action completed it.
- Why that action directly and completely fulfills the subgoal.

First, reason step-by-step about each subgoal within <think> </think> tags.
Then, output the results within <assignments> </assignments> tags.

Output format:
<think>
...
</think>
<assignments>
{assignments_format}
</assignments>"""

STAGE2_FAILURE_PROMPT_TEMPLATE = """You are analyzing a trajectory of a goal-directed task.

Your task is: {task_description}

The task is decomposed into the following ordered subgoals:
{subgoals}

Below is the full trajectory:
{full_trajectory}

This trajectory FAILED to complete the task. Determine which subgoals were achieved and which were not.

Step assignment rules:
- The step number must be the exact "Step N" number from the trajectory.
- A subgoal is achieved only when the action at that step directly and completely fulfills it — not a preparatory or partial action.
- If multiple steps could qualify, use the earliest one.
- Write the exact action text from that step alongside the step number.
- If the subgoal was never achieved, mark it as "none".

In your reasoning, explicitly state for each subgoal:
- Whether it was achieved or not, and why.
- If achieved, which step and what action completed it.
- If not achieved, what was attempted or why it fell short.

First, reason step-by-step about each subgoal within <think> </think> tags.
Then, output the results within <assignments> </assignments> tags.

Output format:
<think>
...
</think>
<assignments>
{assignments_format}
</assignments>"""


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def _format_success_trajectories(success_traj_actions: list[list[str]]) -> str:
    parts = []
    for traj_idx, actions in enumerate(success_traj_actions):
        lines = [f"Trajectory {traj_idx + 1}:"]
        for step_idx, action in enumerate(actions):
            lines.append(f"  Step {step_idx}: {action}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _format_trajectory_actions(actions: list[str]) -> str:
    lines = []
    for step_idx, action in enumerate(actions):
        lines.append(f"Step {step_idx}: {action}")
    return "\n".join(lines)


def _format_subgoals(subgoals: list[str]) -> str:
    return "\n".join(f"{i + 1}. {sg}" for i, sg in enumerate(subgoals))


def _format_assignments_template(subgoals: list[str]) -> str:
    return "\n".join(f"{sg}: step number action taken   (or \"none\" if not achieved)" for sg in subgoals)


def format_stage1_prompt(task_description: str, success_traj_actions: list[list[str]]) -> str:
    return STAGE1_PROMPT_TEMPLATE.format(
        task_description=task_description,
        success_trajectories=_format_success_trajectories(success_traj_actions),
    )


def format_stage2_prompt(task_description: str, subgoals: list[str],
                         actions: list[str], won: bool = True) -> str:
    template = STAGE2_SUCCESS_PROMPT_TEMPLATE if won else STAGE2_FAILURE_PROMPT_TEMPLATE
    return template.format(
        task_description=task_description,
        subgoals=_format_subgoals(subgoals),
        full_trajectory=_format_trajectory_actions(actions),
        assignments_format=_format_assignments_template(subgoals),
    )


# ---------------------------------------------------------------------------
# Response Parsers
# ---------------------------------------------------------------------------

def parse_stage1_response(response: str) -> list[str] | None:
    """Extract subgoals from <subgoals>...</subgoals> block. Returns None on failure."""
    match = re.search(r"<subgoals>(.*?)</subgoals>", response, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    block = match.group(1).strip()
    subgoals = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match "1. <subgoal text>" or "1. text"
        m = re.match(r"^\d+\.\s*<subgoal>\s*(.*?)\s*(?:</subgoal>)?$", line, re.IGNORECASE)
        if m:
            subgoals.append(m.group(1).strip())
        else:
            m = re.match(r"^\d+\.\s+(.+)$", line)
            if m:
                subgoals.append(m.group(1).strip())
    if not subgoals:
        return None
    return subgoals


def parse_stage2_response(response: str, subgoals: list[str], num_actions: int) -> dict[int, int] | None:
    """
    Extract subgoal achievement steps from <assignments>...</assignments> block.
    Returns dict {subgoal_idx_0based: step_number} for achieved subgoals, or None on failure.
    Subgoals marked "none" are excluded from the dict.
    """
    match = re.search(r"<assignments>(.*?)</assignments>", response, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    block = match.group(1).strip()
    subgoal_to_idx = {sg.strip().lower(): i for i, sg in enumerate(subgoals)}
    assignments = {}
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match "<subgoal text>: <step_number> <action...>" or "<subgoal text>: none"
        m = re.match(r"^(.+?):\s+(.+)$", line)
        if not m:
            continue
        sg_text = m.group(1).strip().lower()
        value = m.group(2).strip()
        sg_idx = subgoal_to_idx.get(sg_text)
        if sg_idx is None:
            continue
        if value.lower() == "none":
            continue  # skip unachieved subgoals
        step_match = re.match(r"(\d+)", value)
        if step_match:
            step = int(step_match.group(1))
            if 0 <= step < num_actions:
                assignments[sg_idx] = step
    if not assignments:
        return None
    return assignments


# ---------------------------------------------------------------------------
# Segment / Belief Derivation
# ---------------------------------------------------------------------------

def derive_segments(assignments: dict[int, int], num_states: int) -> list[frozenset]:
    """
    Convert subgoal achievement steps into per-state achieved-subgoal sets.

    assignments: {subgoal_idx_0based: achievement_action_step}
        e.g. {0: 4, 1: 21, 2: 23}

    Returns list[frozenset]: at state s, the frozenset contains all subgoal indices
    whose achievement_step <= s. Order-independent — {0} vs {1} are different beliefs
    even if both represent "1 subgoal achieved", so independent subgoals achieved in
    different orders never collapse to the same node.

    Example:
        state 0..3 → frozenset()      (nothing yet)
        state 4    → frozenset({0})   (subgoal 0 done)
        state 5..20 → frozenset({0})
        state 21   → frozenset({0,1}) (subgoal 1 done)
        state 22   → frozenset({0,1})
        state 23   → frozenset({0,1,2})
    """
    if not assignments:
        return [frozenset()] * num_states

    sorted_achievements = sorted(assignments.items(), key=lambda x: x[1])  # [(sg_idx, step)]

    beliefs = []
    achieved = set()
    ach_idx = 0
    for state_pos in range(num_states):
        while ach_idx < len(sorted_achievements) and sorted_achievements[ach_idx][1] <= state_pos:
            achieved.add(sorted_achievements[ach_idx][0])
            ach_idx += 1
        beliefs.append(frozenset(achieved))
    return beliefs


# ---------------------------------------------------------------------------
# LLM Inference Utilities
# ---------------------------------------------------------------------------

def build_llm_dataproto(prompts: list[str], tokenizer, config) -> DataProto:
    """
    Tokenize a batch of plain text prompts and build a DataProto for generate_sequences.
    Follows the same pattern as rollout_loop.py preprocess_batch (text-only path).
    """
    row_dicts = []
    for prompt_text in prompts:
        # Apply chat template
        chat = [{"role": "user", "content": prompt_text}]
        prompt_with_template = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize with left padding
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_template,
            tokenizer=tokenizer,
            max_length=config.data.max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation=config.data.truncation,
        )

        # Position IDs
        position_ids = compute_position_id_with_mask(attention_mask)

        # Raw prompt IDs (for responses decoding reference, truncated same way)
        raw_prompt_ids = tokenizer.encode(prompt_with_template, add_special_tokens=False)
        if len(raw_prompt_ids) > config.data.max_prompt_length:
            if config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-config.data.max_prompt_length:]
            else:
                raw_prompt_ids = raw_prompt_ids[:config.data.max_prompt_length]

        row_dicts.append({
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": np.array(raw_prompt_ids, dtype=object),
        })

    batch = collate_fn(row_dicts)
    meta_info = {
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    return DataProto.from_single_dict(data=batch, meta_info=meta_info)


def run_llm_batch(prompts: list[str], tokenizer, actor_rollout_wg, config) -> list[str]:
    """Batch LLM inference via actor_rollout_wg.generate_sequences."""
    batch_input = build_llm_dataproto(prompts, tokenizer, config)
    batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
    batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
    batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
    responses = tokenizer.batch_decode(batch_output.batch["responses"], skip_special_tokens=True)
    return responses


# ---------------------------------------------------------------------------
# Pipeline Entry Points
# ---------------------------------------------------------------------------

def run_belief_pipeline_batch(
    task_description: str,
    raw_state_list: list[list[dict]],
    raw_action_list: list[list[str]],
    episode_rewards: list[float],
    tokenizer,
    actor_rollout_wg,
    config,
) -> list[list[int]]:
    """
    Run Stage 1 + Stage 2 belief pipeline for all trajectories in one uid-group.

    Args:
        task_description: task instruction string
        raw_state_list: output of envs.state_preprocess; raw_state_list[i] is a list of
            {"state": obs, "reward": float, "active_masks": bool} dicts
        raw_action_list: output of envs.state_preprocess; raw_action_list[i] is a list of
            action strings (len = len(state_list[i]) - 1)
        episode_rewards: scalar total reward per trajectory (len == len(raw_state_list))
        tokenizer, actor_rollout_wg, config: inference dependencies

    Returns:
        beliefs: beliefs[i][j] = frozenset of achieved subgoal indices for trajectory i at state position j
    """
    num_trajs = len(raw_state_list)
    fallback_beliefs = [[frozenset()] * len(raw_state_list[i]) for i in range(num_trajs)]

    # Identify successful trajectories (total reward > 0)
    success_indices = [i for i, r in enumerate(episode_rewards) if r > 0]
    if not success_indices:
        print("[TPAB] No successful trajectories in this uid-group. Using fallback beliefs (all 0).")
        return fallback_beliefs

    # ---- Stage 1: Subgoal decomposition ----
    success_traj_actions = [raw_action_list[i] for i in success_indices]
    stage1_prompt = format_stage1_prompt(task_description, success_traj_actions)

    try:
        stage1_responses = run_llm_batch([stage1_prompt], tokenizer, actor_rollout_wg, config)
        subgoals = parse_stage1_response(stage1_responses[0])
    except Exception as e:
        print(f"[TPAB] Stage 1 LLM call failed: {e}. Using fallback subgoal.")
        subgoals = None

    if subgoals is None:
        print("[TPAB] Stage 1 parse failed. Using fallback subgoal ['complete task'].")
        subgoals = ["complete task"]

    print(f"[TPAB Stage1] subgoals ({len(subgoals)}): {subgoals}")

    # ---- Stage 2: Per-trajectory belief assignment (batched) ----
    # Use success-specific prompt for won trajectories, failure-specific for lost ones.
    stage2_prompts = []
    for i in range(num_trajs):
        won = episode_rewards[i] > 0
        stage2_prompts.append(
            format_stage2_prompt(task_description, subgoals, raw_action_list[i], won=won)
        )

    try:
        stage2_responses = run_llm_batch(stage2_prompts, tokenizer, actor_rollout_wg, config)
    except Exception as e:
        print(f"[TPAB Stage2] LLM call failed: {e}. Using fallback beliefs.")
        return fallback_beliefs

    # Parse responses and derive beliefs per trajectory
    beliefs_list = []
    num_subgoals = len(subgoals)
    for i in range(num_trajs):
        num_states = len(raw_state_list[i])
        num_actions = len(raw_action_list[i])
        assignments = parse_stage2_response(stage2_responses[i], subgoals, num_actions)

        if assignments is None:
            print(f"[TPAB] Stage 2 parse failed for traj {i}. Using fallback beliefs (all 0).")
            beliefs_list.append([frozenset()] * num_states)
        else:
            beliefs = derive_segments(assignments, num_states)
            print(f"[TPAB Stage2] traj {i} beliefs: {beliefs}")
            beliefs_list.append(beliefs)

    return beliefs_list
