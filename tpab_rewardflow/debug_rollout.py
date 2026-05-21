#!/usr/bin/env python3
"""
Real ALFWorld rollout debug script for TPAB-RewardFlow.

Runs 8 trajectories on the same ALFWorld task using the merged Qwen2.5-1.5B model,
then applies the 2-stage TPAB belief pipeline, and saves ALL raw LLM I/O.

Usage:
    cd /workspace/rewardflow_sciworld
    ALFWORLD_DATA=/root/.cache/alfworld python tpab_rewardflow/debug_rollout.py \
        --model_path /workspace/qwen2.5_1.5b_alfworld_merged \
        --n_rollouts 8 \
        --seed 0 \
        --max_steps 25 \
        --outdir /workspace/tpab_rollout_output
"""

import os
import sys
import json
import re
import argparse
import textwrap
from pathlib import Path
from collections import deque

# Set ALFWORLD_DATA before any alfworld import
if 'ALFWORLD_DATA' not in os.environ:
    os.environ['ALFWORLD_DATA'] = '/root/.cache/alfworld'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto

from tpab_rewardflow.belief_pipeline import (
    format_stage1_prompt,
    parse_stage1_response,
    format_stage2_prompt,
    parse_stage2_response,
    derive_segments,
    run_belief_pipeline_batch,
)
from tpab_rewardflow.tpab_rewardflow_core import (
    to_hashable_belief,
    extract_unique_belief_states,
    build_belief_trajectory,
)
from tpab_rewardflow.propagation import propagate_reward_decay
from rewardflow.core_rewardflow import to_hashable


# ---------------------------------------------------------------------------
# Prompt template (mirrors AlfWorldEnvironmentManager.build_text_obs)
# ---------------------------------------------------------------------------

ALFWORLD_TEMPLATE_NO_HIS = (
    "\nYou are an expert agent operating in the ALFRED Embodied Environment.\n"
    "Your current observation is: {current_observation}\n"
    "Your admissible actions of the current situation are: [{admissible_actions}].\n\n"
    "Now it's your turn to take an action.\n"
    "You should first reason step-by-step about the current situation. "
    "This reasoning process MUST be enclosed within <think> </think> tags. \n"
    "Once you've finished your reasoning, you should choose an admissible action "
    "for current step and present it within <action> </action> tags.\n"
)

ALFWORLD_TEMPLATE = (
    "\nYou are an expert agent operating in the ALFRED Embodied Environment. "
    "Your task is to: {task_description}\n"
    "Prior to this step, you have already taken {step_count} step(s). "
    "Below are the most recent {history_length} observations and the "
    "corresponding actions you took: {action_history}\n"
    "You are now at step {current_step} and your current observation is: {current_observation}\n"
    "Your admissible actions of the current situation are: [{admissible_actions}].\n\n"
    "Now it's your turn to take an action.\n"
    "You should first reason step-by-step about the current situation. "
    "This reasoning process MUST be enclosed within <think> </think> tags. \n"
    "Once you've finished your reasoning, you should choose an admissible action "
    "for current step and present it within <action> </action> tags.\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unique_trajectory(trajectory):
    result = []
    for traj in trajectory:
        seen = []
        for item in traj:
            if item not in seen:
                seen.append(item)
        result.append(seen)
    return [item for sublist in result for item in sublist]


def extract_task(obs: str) -> str:
    marker = 'Your task is to: '
    idx = obs.find(marker)
    if idx != -1:
        return obs[idx + len(marker):].strip().split('\n')[0]
    return "complete the task"


# Patterns stripped from obs before using as graph node key.
# "You are carrying: ..." and "Episode max steps reached." make otherwise-identical
# states look different in the graph; stripping them enables correct deduplication.
_STRIP_PATTERNS = [
    re.compile(r'\s*You are carrying:.*$', re.DOTALL),
    re.compile(r'\s*Episode max steps reached\.?', re.IGNORECASE),
    re.compile(r'\s*Nothing is added to your inventory\.?', re.IGNORECASE),
]

def clean_obs_for_key(obs: str) -> str:
    """Return observation string stripped of inventory/terminal suffixes for use as graph node key."""
    for pat in _STRIP_PATTERNS:
        obs = pat.sub('', obs)
    return obs.strip()


def alfworld_projection(response: str, admissible: list[str]) -> tuple[str, bool]:
    """Extract <action>...</action> and match to admissible commands."""
    lower = response.lower()
    start = lower.find('<action>')
    end = lower.find('</action>')
    has_think = '<think>' in lower and '</think>' in lower
    has_chinese = bool(re.search(r'[一-鿿]', response))

    if start == -1 or end == -1 or has_chinese or not has_think:
        return admissible[0], False

    extracted = lower[start + len('<action>'):end].strip()

    # Exact match
    for cmd in admissible:
        if cmd.lower() == extracted:
            return cmd, True

    # Partial match (extracted is substring of admissible or vice versa)
    for cmd in admissible:
        if extracted in cmd.lower() or cmd.lower() in extracted:
            return cmd, True

    # Fallback: first admissible
    return admissible[0], False


def build_prompt(obs: str, admissible: list[str], task: str,
                 history: list[tuple[str, str]], step: int) -> str:
    admissible_str = "\n ".join(f"'{s}'" for s in admissible if s != 'help')
    if not history:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=obs,
            admissible_actions=admissible_str,
        )
    history_str = ""
    for h_obs, h_act in history:
        history_str += f"\nObservation: {h_obs}\nAction: {h_act}\n"
    return ALFWORLD_TEMPLATE.format(
        task_description=task,
        step_count=step,
        history_length=len(history),
        action_history=history_str,
        current_step=step + 1,
        current_observation=obs,
        admissible_actions=admissible_str,
    )


# ---------------------------------------------------------------------------
# Local model wrapper (mirrors debug_tpab.py LocalModelWrapper)
# ---------------------------------------------------------------------------

class LocalModelWrapper:
    def __init__(self, model, tokenizer, max_new_tokens=512, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.world_size = 1

    def generate_sequences(self, batch_input: DataProto) -> DataProto:
        input_ids = batch_input.batch['input_ids'].to(self.device)
        attention_mask = batch_input.batch['attention_mask'].to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = input_ids.shape[1]
        responses = output[:, prompt_len:].cpu()
        return DataProto(batch={'responses': responses}, non_tensor_batch={})


def call_llm_single(model, tokenizer, prompt: str, max_new_tokens: int = 512,
                    device: str = 'cuda') -> str:
    """Single-prompt LLM call for agent actions (no DataProto overhead)."""
    chat = [{'role': 'user', 'content': prompt}]
    formatted = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False
    )
    ids = tokenizer(formatted, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Graph helpers (mirrors debug_tpab.py)
# ---------------------------------------------------------------------------

def tpab_graph_to_dot(G, value_dict, idx_to_belief_state) -> str:
    belief_colors = {0: '#AED6F1', 1: '#A9DFBF', 2: '#F9E79F', 3: '#F5CBA7', 4: '#D2B4DE'}
    lines = ['digraph TPAB {', '  rankdir=LR;',
             '  node [shape=box, style=filled, fontsize=9];']
    for node in G.nodes():
        val = value_dict.get(node, 0.0)
        if node in idx_to_belief_state:
            _, belief = idx_to_belief_state[node]
        else:
            belief = 0
        color = belief_colors.get(belief, '#FDFEFE')
        state_repr = ''
        if node in idx_to_belief_state:
            state, _ = idx_to_belief_state[node]
            state_repr = str(state)[:40].replace('"', "'").replace('\n', ' ')
        label = f'N{node}\\nb={belief}\\nV={val:.2f}\\n{state_repr}'
        lines.append(f'  {node} [label="{label}", fillcolor="{color}"];')
    for src, dst, data in G.edges(data=True):
        action = data.get('action', '')[:30].replace('"', "'")
        lines.append(f'  {src} -> {dst} [label="{action}"];')
    lines.append('}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# ALFWorld env setup
# ---------------------------------------------------------------------------

def make_env(config_path: str, seed: int, train: bool = True):
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import (
        get_environment,
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    env_class = get_environment('AlfredTWEnv')
    base_env = env_class(config, train_eval='train' if train else 'eval_in_distribution')
    env_inst = base_env.init_env(batch_size=1)
    env_inst.seed(seed)
    return env_inst


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def run_rollout(env_inst, model, tokenizer, seed: int, max_steps: int,
                device: str, history_length: int = 3) -> dict:
    """
    Run one trajectory. Returns dict with:
      obs_seq, action_seq, reward_seq, won, task,
      raw_steps: list of {step, prompt, raw_output, action, valid}
    """
    env_inst.seed(seed)
    obs_list, infos = env_inst.reset()
    obs = obs_list[0]
    task = extract_task(obs)

    history: list[tuple[str, str]] = []
    obs_seq = [obs]
    action_seq = []
    reward_seq = []
    raw_steps = []
    won = False

    for step in range(max_steps):
        admissible = infos['admissible_commands'][0]
        prompt = build_prompt(obs, admissible, task,
                              history[-history_length:] if history else [], step)

        raw_out = call_llm_single(model, tokenizer, prompt, max_new_tokens=512, device=device)
        action, valid = alfworld_projection(raw_out, admissible)

        raw_steps.append({
            'step': step,
            'prompt': prompt,
            'raw_output': raw_out,
            'action': action,
            'valid': valid,
            'admissible': admissible,
        })

        new_obs_list, scores, dones, infos = env_inst.step([action])
        new_obs = new_obs_list[0]
        reward = float(scores[0]) if scores else 0.0
        done = dones[0]

        # Update history
        history.append((obs, action))
        obs = new_obs
        obs_seq.append(obs)
        action_seq.append(action)
        reward_seq.append(reward)

        if done or infos.get('won', [False])[0] if isinstance(infos.get('won'), list) else infos.get('won', False):
            won = True
            break
        if done:
            break

    # Check won from infos
    if not won:
        won = bool(infos.get('won', [False])[0]) if isinstance(infos.get('won'), list) else bool(infos.get('won', False))

    return {
        'task': task,
        'obs_seq': obs_seq,
        'action_seq': action_seq,
        'reward_seq': reward_seq,
        'won': won,
        'episode_reward': sum(reward_seq),
        'raw_steps': raw_steps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/workspace/qwen2.5_1.5b_alfworld_merged')
    parser.add_argument('--n_rollouts', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0, help='ALFWorld game seed (same task repeated)')
    parser.add_argument('--max_steps', type=int, default=25)
    parser.add_argument('--history_length', type=int, default=3)
    parser.add_argument('--outdir', default='/workspace/tpab_rollout_output')
    parser.add_argument('--max_new_tokens_agent', type=int, default=512)
    parser.add_argument('--max_new_tokens_belief', type=int, default=1024)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # -----------------------------------------------------------------------
    # [1] Load model
    # -----------------------------------------------------------------------
    print('=' * 70)
    print('Loading model from:', args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    print(f'  Model loaded. dtype={model.dtype}')

    actor_rollout_wg = LocalModelWrapper(model, tokenizer,
                                         max_new_tokens=args.max_new_tokens_belief,
                                         device=device)

    # -----------------------------------------------------------------------
    # [2] Init ALFWorld env
    # -----------------------------------------------------------------------
    print('\nInitializing ALFWorld env ...')
    config_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'agent_system', 'environments', 'env_package', 'alfworld', 'configs', 'config_tw.yaml'
    )
    env_inst = make_env(config_path, seed=args.seed)
    print('  Env ready.')

    # -----------------------------------------------------------------------
    # [3] Run N rollouts (same game seed → same task repeated)
    # -----------------------------------------------------------------------
    print(f'\nRunning {args.n_rollouts} rollouts (seed={args.seed}, max_steps={args.max_steps})')
    print('=' * 70)

    all_rollouts = []
    for i in range(args.n_rollouts):
        print(f'\n--- Rollout {i} ---')
        rollout = run_rollout(
            env_inst, model, tokenizer,
            seed=args.seed,
            max_steps=args.max_steps,
            device=device,
            history_length=args.history_length,
        )
        all_rollouts.append(rollout)
        status = 'WON' if rollout['won'] else 'LOST'
        print(f'  Task   : {rollout["task"]}')
        print(f'  Steps  : {len(rollout["action_seq"])}')
        print(f'  Reward : {rollout["episode_reward"]:.1f}  [{status}]')
        print(f'  Actions: {rollout["action_seq"]}')

    task_description = all_rollouts[0]['task']

    # -----------------------------------------------------------------------
    # [4] Save raw rollout I/O
    # -----------------------------------------------------------------------
    rollout_io_path = outdir / 'rollout_raw_io.json'
    rollout_io = []
    for i, r in enumerate(all_rollouts):
        rollout_io.append({
            'rollout_idx': i,
            'task': r['task'],
            'won': r['won'],
            'episode_reward': r['episode_reward'],
            'steps': r['raw_steps'],  # each has prompt, raw_output, action, valid
        })
    with open(rollout_io_path, 'w') as f:
        json.dump(rollout_io, f, indent=2, ensure_ascii=False)
    print(f'\n[Saved] rollout raw I/O → {rollout_io_path}')

    # Human-readable text version
    rollout_txt_path = outdir / 'rollout_raw_io.txt'
    with open(rollout_txt_path, 'w') as f:
        for i, r in enumerate(all_rollouts):
            f.write(f'{"=" * 70}\n')
            f.write(f'ROLLOUT {i}  [{"WON" if r["won"] else "LOST"}]  reward={r["episode_reward"]:.1f}\n')
            f.write(f'Task: {r["task"]}\n')
            f.write(f'{"=" * 70}\n')
            for s in r['raw_steps']:
                f.write(f'\n--- Step {s["step"]} ---\n')
                f.write(f'[PROMPT]\n{s["prompt"]}\n')
                f.write(f'[RAW OUTPUT]\n{s["raw_output"]}\n')
                f.write(f'[CHOSEN ACTION] {s["action"]}  (valid={s["valid"]})\n')
            f.write('\n')
    print(f'[Saved] rollout raw I/O (text) → {rollout_txt_path}')

    # -----------------------------------------------------------------------
    # [5] Build raw_state_list / raw_action_list for belief pipeline
    # -----------------------------------------------------------------------
    raw_state_list = []
    raw_action_list = []
    episode_rewards = []
    for r in all_rollouts:
        # Apply clean_obs_for_key: strip "You are carrying:..." and "Episode max steps reached."
        # so that states differing only in inventory/terminal suffix map to the same graph node.
        state_seq = [{'state': clean_obs_for_key(obs), 'reward': 0} for obs in r['obs_seq']]
        # Assign rewards to states
        for j, rwd in enumerate(r['reward_seq']):
            if j + 1 < len(state_seq):
                state_seq[j + 1]['reward'] = rwd
        raw_state_list.append(state_seq)
        raw_action_list.append(r['action_seq'])
        episode_rewards.append(r['episode_reward'])

    print(f'\nEpisode rewards: {episode_rewards}')
    print(f'Success trajs  : {[r["won"] for r in all_rollouts]}')

    # -----------------------------------------------------------------------
    # [6] TPAB belief pipeline (Stage 1 + Stage 2) with raw I/O capture
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('[STAGE 1] Subgoal Decomposition')
    print('=' * 70)

    # -- Build a mini config for build_llm_dataproto --
    from types import SimpleNamespace
    cfg = SimpleNamespace()
    cfg.data = SimpleNamespace()
    cfg.data.max_prompt_length = 2048
    cfg.data.truncation = 'left'

    # Stage 1 manually (to capture raw I/O)
    success_indices = [i for i, r in enumerate(all_rollouts) if r['won']]
    if success_indices:
        success_actions = [raw_action_list[i] for i in success_indices]
        stage1_prompt = format_stage1_prompt(task_description, success_actions)
    else:
        stage1_prompt = format_stage1_prompt(task_description, [raw_action_list[0]])

    print('\n--- STAGE 1 PROMPT (truncated to 600 chars) ---')
    print(stage1_prompt[:600])

    from tpab_rewardflow.belief_pipeline import build_llm_dataproto, run_llm_batch
    stage1_responses = run_llm_batch([stage1_prompt], tokenizer, actor_rollout_wg, cfg)
    stage1_response = stage1_responses[0]

    print('\n--- STAGE 1 RESPONSE ---')
    print(stage1_response)

    subgoals = parse_stage1_response(stage1_response)
    if not subgoals:
        print('  [WARN] Parse failed, using fallback subgoal')
        subgoals = ['complete task']
    print(f'\n--- PARSED SUBGOALS ({len(subgoals)}) ---')
    for j, sg in enumerate(subgoals):
        print(f'  {j}: {sg}')

    # Save Stage 1 I/O
    with open(outdir / 'stage1_raw_io.txt', 'w') as f:
        f.write('=== STAGE 1 PROMPT ===\n')
        f.write(stage1_prompt + '\n\n')
        f.write('=== STAGE 1 RAW RESPONSE ===\n')
        f.write(stage1_response + '\n\n')
        f.write('=== PARSED SUBGOALS ===\n')
        for j, sg in enumerate(subgoals):
            f.write(f'{j}: {sg}\n')
    print(f'[Saved] stage1_raw_io.txt → {outdir / "stage1_raw_io.txt"}')

    # Stage 2 manually (to capture raw I/O)
    print('\n' + '=' * 70)
    print('[STAGE 2] Belief Assignment (all trajectories, batched)')
    print('=' * 70)

    stage2_prompts = []
    for i, r in enumerate(all_rollouts):
        p = format_stage2_prompt(task_description, subgoals, raw_action_list[i], won=r['won'])
        stage2_prompts.append(p)

    stage2_responses = run_llm_batch(stage2_prompts, tokenizer, actor_rollout_wg, cfg)

    beliefs_list = []
    stage2_io_records = []
    for i, (prompt, response) in enumerate(zip(stage2_prompts, stage2_responses)):
        num_actions = len(raw_action_list[i])
        num_states = len(raw_state_list[i])
        assignments = parse_stage2_response(response, len(subgoals), num_actions)

        print(f'\n--- Traj {i} (won={all_rollouts[i]["won"]}) ---')
        print(f'  Actions ({num_actions}): {raw_action_list[i]}')
        print(f'\n  STAGE 2 RESPONSE:')
        for line in response.split('\n'):
            print(f'    {line}')
        if assignments:
            print(f'  Assignments (0-based): {assignments}')
        else:
            print('  [WARN] Parse failed, using all-zero beliefs')

        beliefs = derive_segments(assignments or {}, num_states)
        beliefs_list.append(beliefs)
        print(f'  Beliefs per state: {beliefs}')

        stage2_io_records.append({
            'traj_idx': i,
            'won': all_rollouts[i]['won'],
            'num_actions': num_actions,
            'prompt': prompt,
            'raw_response': response,
            'assignments': assignments,
            'beliefs': beliefs,
        })

    # Save Stage 2 I/O
    with open(outdir / 'stage2_raw_io.json', 'w') as f:
        json.dump(stage2_io_records, f, indent=2, ensure_ascii=False)
    with open(outdir / 'stage2_raw_io.txt', 'w') as f:
        for rec in stage2_io_records:
            f.write(f'{"=" * 60}\n')
            f.write(f'TRAJ {rec["traj_idx"]}  won={rec["won"]}\n')
            f.write(f'{"=" * 60}\n')
            f.write('--- PROMPT ---\n')
            f.write(rec['prompt'] + '\n\n')
            f.write('--- RAW RESPONSE ---\n')
            f.write(rec['raw_response'] + '\n\n')
            f.write(f'assignments: {rec["assignments"]}\n')
            f.write(f'beliefs:     {rec["beliefs"]}\n\n')
    print(f'\n[Saved] stage2_raw_io.json → {outdir / "stage2_raw_io.json"}')
    print(f'[Saved] stage2_raw_io.txt  → {outdir / "stage2_raw_io.txt"}')

    # -----------------------------------------------------------------------
    # [7] Build belief-augmented graph
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('[GRAPH] Building Belief-Augmented State Graph')
    print('=' * 70)

    state_list_with_beliefs = []
    for i, state_seq in enumerate(raw_state_list):
        beliefs = beliefs_list[i]
        augmented = []
        for j, step in enumerate(state_seq):
            b = beliefs[j] if j < len(beliefs) else beliefs[-1]
            augmented.append({'state': step['state'], 'reward': step['reward'], 'belief': b})
        state_list_with_beliefs.append(augmented)

    belief_state_to_idx, idx_to_belief_state = extract_unique_belief_states(state_list_with_beliefs)
    trajectory = build_belief_trajectory(state_list_with_beliefs, raw_action_list, belief_state_to_idx)

    # Clean self-loops
    cleaned = [[t for t in traj if t[0] != t[2]] for traj in trajectory]
    flat_unique = unique_trajectory(cleaned)

    G, value_dict = propagate_reward_decay(flat_unique, gamma=0.95, max_iter=1000)

    for idx in range(len(idx_to_belief_state)):
        if idx not in value_dict:
            value_dict[idx] = 0.0

    print(f'  Unique belief-augmented nodes: {len(idx_to_belief_state)}')
    print(f'  Unique edges in graph: {G.number_of_edges()}')

    print('\n  Node values (belief-augmented):')
    for node in sorted(idx_to_belief_state.keys()):
        state, belief = idx_to_belief_state[node]
        val = value_dict.get(node, 0.0)
        state_repr = str(state)[:60].replace('\n', ' ')
        print(f'    Node {node:3d} [belief={belief}] V={val:.4f}  state: {state_repr}')

    print('\n  Graph edges (src → dst, action, ΔV):')
    for src, dst, data in G.edges(data=True):
        action = data.get('action', '')
        dv = value_dict.get(dst, 0.0) - value_dict.get(src, 0.0)
        print(f'    {src:3d} → {dst:3d}  [{action}]  ΔV={dv:+.4f}')

    # -----------------------------------------------------------------------
    # [8] Aliasing analysis
    # -----------------------------------------------------------------------
    from collections import defaultdict
    state_to_beliefs = defaultdict(set)
    for idx, (state, belief) in idx_to_belief_state.items():
        state_key = to_hashable(state)
        state_to_beliefs[state_key].add(belief)
    aliased = sum(1 for beliefs in state_to_beliefs.values() if len(beliefs) > 1)
    print(f'\n  --- ALIASING ANALYSIS ---')
    print(f'  Observations appearing at multiple beliefs (aliasing resolved): {aliased}')
    print(f'  Total unique nodes (TPAB): {len(idx_to_belief_state)}')
    vanilla_count = len(set(to_hashable(s) for s, _ in idx_to_belief_state.values()))
    print(f'  Total unique nodes (vanilla would have): {vanilla_count}')

    # -----------------------------------------------------------------------
    # [9] Save graph outputs
    # -----------------------------------------------------------------------
    graph_json = {
        'nodes': [
            {
                'id': idx,
                'belief': int(idx_to_belief_state[idx][1]),
                'value': float(value_dict.get(idx, 0.0)),
                'state': str(idx_to_belief_state[idx][0])[:200],
            }
            for idx in sorted(idx_to_belief_state.keys())
        ],
        'edges': [
            {
                'src': int(src),
                'dst': int(dst),
                'action': data.get('action', ''),
                'delta_v': float(value_dict.get(dst, 0.0) - value_dict.get(src, 0.0)),
            }
            for src, dst, data in G.edges(data=True)
        ],
    }
    graph_json_path = outdir / 'tpab_graph.json'
    with open(graph_json_path, 'w') as f:
        json.dump(graph_json, f, indent=2, ensure_ascii=False)
    print(f'\n[Saved] tpab_graph.json → {graph_json_path}')

    dot_str = tpab_graph_to_dot(G, value_dict, idx_to_belief_state)
    dot_path = outdir / 'tpab_graph.dot'
    with open(dot_path, 'w') as f:
        f.write(dot_str)
    print(f'[Saved] tpab_graph.dot → {dot_path}')

    png_path = outdir / 'tpab_graph.png'
    try:
        import subprocess
        subprocess.run(['dot', '-Tpng', str(dot_path), '-o', str(png_path)], check=True)
        print(f'[Saved] tpab_graph.png → {png_path}')
    except Exception as e:
        print(f'[WARN] Graphviz render failed: {e}')

    # -----------------------------------------------------------------------
    # [10] Summary
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('Done. Outputs:')
    for p in sorted(outdir.iterdir()):
        print(f'  {p}')
    print('=' * 70)


if __name__ == '__main__':
    main()
