"""
Core functions for TPAB-RewardFlow (Task-Progress-Aware Belief for RewardFlow).

Key difference from vanilla RewardFlow:
  node_key = hash(observed_state, belief_index)   ← TPAB
  node_key = hash(observed_state)                 ← vanilla RewardFlow

This resolves aliasing: the same observed state at different task-progress stages
maps to different graph nodes, so reward propagates along progress-respecting paths.
"""

import numpy as np
from collections import defaultdict

from rewardflow.core_rewardflow import to_hashable
from agent_system.multi_turn_rollout.utils import extract_unique_states, build_trajectory
from tpab_rewardflow.propagation import propagate_reward_decay
from tpab_rewardflow.belief_pipeline import run_belief_pipeline_batch


# ---------------------------------------------------------------------------
# Belief-augmented node key helpers
# ---------------------------------------------------------------------------

def to_hashable_belief(state, belief_set) -> tuple:
    """Belief-augmented node key: (hashable_state, frozenset_of_achieved_subgoals)."""
    if isinstance(belief_set, int):
        belief_set = frozenset()  # legacy int fallback
    return (to_hashable(state), frozenset(belief_set))


def extract_unique_belief_states(state_list_with_beliefs: list[list[dict]]):
    """
    Build index mappings over belief-augmented node keys.

    Args:
        state_list_with_beliefs: list of trajectories; each trajectory is a list of dicts
            {"state": obs, "reward": float, "belief": int}

    Returns:
        belief_state_to_idx: dict[(hashable_state, belief_int) -> int]
        idx_to_belief_state: dict[int -> (state, belief_int)]
    """
    belief_state_to_idx = {}
    idx_to_belief_state = {}
    counter = 0
    for traj in state_list_with_beliefs:
        for step in traj:
            key = to_hashable_belief(step["state"], step["belief"])
            if key not in belief_state_to_idx:
                belief_state_to_idx[key] = counter
                idx_to_belief_state[counter] = (step["state"], step["belief"])
                counter += 1
    return belief_state_to_idx, idx_to_belief_state


def build_belief_trajectory(
    state_list_with_beliefs: list[list[dict]],
    action_list: list[list[str]],
    belief_state_to_idx: dict,
) -> list[list[tuple]]:
    """
    Build trajectory tuples using belief-augmented node keys.
    Mirrors build_trajectory() from utils.py but uses (state, belief) as key.

    Returns:
        trajectory[i] = list of (src_idx, action, dst_idx, src_reward, dst_reward)
    """
    trajectory = []
    for i in range(len(state_list_with_beliefs)):
        traj = []
        state_seq = state_list_with_beliefs[i]
        for j in range(len(state_seq) - 1):
            src = state_seq[j]
            dst = state_seq[j + 1]
            src_key = to_hashable_belief(src["state"], src["belief"])
            dst_key = to_hashable_belief(dst["state"], dst["belief"])
            src_idx = belief_state_to_idx[src_key]
            dst_idx = belief_state_to_idx[dst_key]
            action_label = action_list[i][j] if j < len(action_list[i]) else ""
            traj.append((src_idx, action_label, dst_idx, src["reward"], dst["reward"]))
        trajectory.append(traj)
    return trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_uid_groups(uid_list: list) -> list[tuple[int, int]]:
    """
    Find contiguous groups of identical uid values.
    Returns list of (start_idx, end_idx) pairs (end_idx exclusive).
    Mirrors the logic in core_rewardflow.py lines 330–339.
    """
    if not uid_list:
        return []
    groups = []
    prev = uid_list[0]
    start = 0
    for idx in range(1, len(uid_list)):
        if uid_list[idx] != prev:
            groups.append((start, idx))
            start = idx
            prev = uid_list[idx]
    groups.append((start, len(uid_list)))
    return groups


def _get_task_description(envs, start_idx: int) -> str:
    """Retrieve the task description for the trajectory group starting at start_idx."""
    try:
        tasks = envs.tasks
        return tasks[start_idx % len(tasks)]
    except (AttributeError, IndexError, TypeError):
        return "complete the task"


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def apply_tpab_rewardflow_propagation(
    total_batch_list: list,
    config,
    tokenizer,
    envs,
    actor_rollout_wg,
) -> list:
    """
    TPAB-RewardFlow propagation: extends RewardFlow with belief-augmented node keys.

    Node key = hash(observed_state, subgoal_index) instead of hash(observed_state).
    A two-stage LLM pipeline assigns each step a belief (subgoal index):
      Stage 1 (once per uid-group): decompose task into subgoals from successful trajectories.
      Stage 2 (batched over all trajectories in group): assign subgoal indices per step.

    Falls back to vanilla RewardFlow behavior (all beliefs = 0) when:
      - No successful trajectories exist in a uid-group.
      - LLM calls or parsing fail.

    Args:
        total_batch_list: List[List[Dict]] — trajectories x steps, from rollout loop.
        config: training config.
        tokenizer: HF tokenizer.
        envs: environment manager (must implement state_preprocess, clean_trajectory).
        actor_rollout_wg: distributed actor worker group for LLM inference.

    Returns:
        total_batch_list with 'step_rewards' and 'step_rewards_abs' populated.
    """
    from agent_system.multi_turn_rollout.utils import unique_trajectory

    # Step 1: Extract state/action sequences
    raw_state_list, raw_action_list = envs.state_preprocess(total_batch_list, tokenizer)

    # Step 2: Group trajectories by uid
    uid_list = [total_batch_list[i][0]["uid"] for i in range(len(total_batch_list))]
    first_last_indices = _compute_uid_groups(uid_list)

    # Step 3: Per uid-group: belief pipeline + graph construction
    value_dict_list = []
    belief_state_to_idx_list = []
    all_beliefs: list[list[frozenset] | None] = [None] * len(total_batch_list)

    for (start_idx, end_idx) in first_last_indices:
        selected_state_list = raw_state_list[start_idx:end_idx]
        selected_action_list = raw_action_list[start_idx:end_idx]

        # Compute episode rewards for this group (sum of per-step rewards)
        episode_rewards = []
        for i in range(start_idx, end_idx):
            ep_r = sum(
                step["reward"] for step in raw_state_list[i]
            )
            episode_rewards.append(ep_r)

        # Get task description
        task_description = _get_task_description(envs, start_idx)

        n_success = sum(1 for r in episode_rewards if r > 0)
        print(f"[TPAB] uid-group [{start_idx}:{end_idx}]  "
              f"trajs={end_idx - start_idx}  success={n_success}  "
              f"task: {task_description}")

        # --- Step A: Filter actions via existing clean_trajectory (vanilla, no beliefs) ---
        # Build vanilla trajectory first to reuse the same clean_trajectory logic
        # so prompt actions and graph edges are guaranteed identical.
        _, vanilla_state_to_idx, vanilla_idx_to_state = extract_unique_states(selected_state_list)
        vanilla_traj = build_trajectory(selected_state_list, selected_action_list, vanilla_state_to_idx)
        cleaned_vanilla = envs.clean_trajectory(vanilla_traj, vanilla_idx_to_state)

        # Extract filtered action strings and their original indices per trajectory
        filtered_action_list = []
        orig_indices_list = []
        for local_i, ctraj in enumerate(cleaned_vanilla):
            raw_actions = selected_action_list[local_i]
            f_actions, f_orig = [], []
            ptr = 0
            for (_, action, _, _, _) in ctraj:
                while ptr < len(raw_actions) and raw_actions[ptr] != action:
                    ptr += 1
                if ptr < len(raw_actions):
                    f_actions.append(action)
                    f_orig.append(ptr)
                    ptr += 1
            filtered_action_list.append(f_actions)
            orig_indices_list.append(f_orig)

        # --- Belief pipeline (uses filtered actions identical to graph edges) ---
        beliefs_list = run_belief_pipeline_batch(
            task_description=task_description,
            raw_state_list=selected_state_list,
            raw_action_list=filtered_action_list,
            episode_rewards=episode_rewards,
            tokenizer=tokenizer,
            actor_rollout_wg=actor_rollout_wg,
            config=config,
        )

        # Remap per-filtered-state beliefs back to original state positions
        remapped_beliefs_list = []
        for local_i, filtered_beliefs in enumerate(beliefs_list):
            orig_indices = orig_indices_list[local_i]
            num_states_orig = len(selected_state_list[local_i])
            remapped = []
            for j in range(num_states_orig):
                p = sum(1 for oi in orig_indices if oi < j)
                b = filtered_beliefs[p] if p < len(filtered_beliefs) else (filtered_beliefs[-1] if filtered_beliefs else frozenset())
                remapped.append(b)
            remapped_beliefs_list.append(remapped)

        # Store beliefs indexed by global trajectory position
        for local_i, beliefs in enumerate(remapped_beliefs_list):
            all_beliefs[start_idx + local_i] = beliefs

        # Attach remapped beliefs to state dicts
        state_list_with_beliefs = []
        for local_i, state_seq in enumerate(selected_state_list):
            beliefs = remapped_beliefs_list[local_i]
            augmented = []
            for j, step in enumerate(state_seq):
                b = beliefs[j] if j < len(beliefs) else beliefs[-1]
                augmented.append({
                    "state": step["state"],
                    "reward": step["reward"],
                    "belief": b,
                })
            state_list_with_beliefs.append(augmented)

        # --- Build belief-augmented graph ---
        belief_state_to_idx, idx_to_belief_state = extract_unique_belief_states(state_list_with_beliefs)
        trajectory = build_belief_trajectory(state_list_with_beliefs, selected_action_list, belief_state_to_idx)
        # Pass state strings (not belief tuples) so "Nothing happens" check works correctly
        idx_to_state_str = {idx: str(state) for idx, (state, _) in idx_to_belief_state.items()}
        cleaned_trajectory = envs.clean_trajectory(trajectory, idx_to_state_str)
        flat_unique_traj = unique_trajectory(cleaned_trajectory)

        G, value_dict = propagate_reward_decay(
            flat_unique_traj,
            gamma=config.algorithm.gamma,
            max_iter=1000,
        )

        # Fill missing nodes
        for idx in range(len(idx_to_belief_state)):
            if idx not in value_dict:
                value_dict[idx] = 0.0


        value_dict_list.append(value_dict)
        belief_state_to_idx_list.append(belief_state_to_idx)

    # Build lookup: trajectory index -> uid-group index
    traj_to_group = {}
    for group_idx, (start_idx, end_idx) in enumerate(first_last_indices):
        for i in range(start_idx, end_idx):
            traj_to_group[i] = group_idx

    # Step 4: Assign step rewards using belief-augmented lookups
    new_reward_list = []
    new_reward_abs_list = []

    for i in range(len(total_batch_list)):
        batch_idx = traj_to_group[i]
        beliefs = all_beliefs[i] if all_beliefs[i] is not None else [frozenset()] * len(total_batch_list[i])

        # Initial state
        initial_obs = total_batch_list[i][0]["anchor_obs"]
        initial_belief = beliefs[0] if beliefs else frozenset()
        initial_key = to_hashable_belief(initial_obs, initial_belief)

        state_idx = belief_state_to_idx_list[batch_idx].get(initial_key)
        prev_reward = value_dict_list[batch_idx].get(state_idx, 0.0) if state_idx is not None else 0.0

        new_reward = []
        new_reward_abs = []

        for j in range(len(total_batch_list[i]) - 1):
            if not total_batch_list[i][j]["active_masks"]:
                break

            next_state_pos = j + 1
            cur_belief = beliefs[next_state_pos] if next_state_pos < len(beliefs) else beliefs[-1]
            cur_obs = total_batch_list[i][j + 1]["anchor_obs"]
            cur_key = to_hashable_belief(cur_obs, cur_belief)

            state_idx = belief_state_to_idx_list[batch_idx].get(cur_key)
            if state_idx is not None:
                cur_reward = value_dict_list[batch_idx].get(state_idx, prev_reward)
            else:
                cur_reward = prev_reward  # safe fallback: zero delta

            new_reward.append(cur_reward - prev_reward)
            new_reward_abs.append(cur_reward)
            prev_reward = cur_reward

        new_reward_list.append(new_reward)
        new_reward_abs_list.append(new_reward_abs)

    # Step 5: Write back step rewards
    for i in range(len(total_batch_list)):
        for j in range(len(total_batch_list[i])):
            if j < len(new_reward_list[i]):
                total_batch_list[i][j]["step_rewards"] = new_reward_list[i][j]
                total_batch_list[i][j]["step_rewards_abs"] = new_reward_abs_list[i][j]
            else:
                total_batch_list[i][j]["step_rewards"] = total_batch_list[i][j]["rewards"]
                total_batch_list[i][j]["step_rewards_abs"] = total_batch_list[i][j]["rewards"]

    print(f"[TPAB] length of total_batch_list: {len(total_batch_list)}")
    return total_batch_list
