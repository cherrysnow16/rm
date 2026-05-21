#!/usr/bin/env python3
"""
Analyze saved rollout data with TPAB belief pipeline.

1. Loads rollout_raw_io.json (action sequences + task)
2. Replays actions through ALFWorld env to recover obs sequences
3. Runs Stage 1 + Stage 2 belief pipeline (captures all raw LLM I/O)
4. Generates two graphs:
   - WITHOUT clean_obs_for_key (raw obs as node keys)
   - WITH    clean_obs_for_key (stripped obs as node keys)

Usage:
    cd /workspace/rewardflow_sciworld
    ALFWORLD_DATA=/root/.cache/alfworld python -u tpab_rewardflow/analyze_rollout.py \
        --rollout_json /workspace/tpab_rollout_output/rollout_raw_io.json \
        --model_path /workspace/qwen2.5_1.5b_alfworld_merged \
        --outdir /workspace/tpab_analysis_output
"""

import os, sys, re, json, argparse
from pathlib import Path

if 'ALFWORLD_DATA' not in os.environ:
    os.environ['ALFWORLD_DATA'] = '/root/.cache/alfworld'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM

from tpab_rewardflow.belief_pipeline import (
    format_stage1_prompt,
    format_stage2_prompt,
    parse_stage1_response,
    parse_stage2_response,
    derive_segments,
)
from tpab_rewardflow.tpab_rewardflow_core import (
    to_hashable_belief,
    extract_unique_belief_states,
    build_belief_trajectory,
)
from tpab_rewardflow.propagation import propagate_reward_decay
from rewardflow.core_rewardflow import to_hashable


# ---------------------------------------------------------------------------
# State cleaning (same as debug_rollout.py)
# ---------------------------------------------------------------------------

_STRIP_PATTERNS = [
    re.compile(r'\s*You are carrying:.*$', re.DOTALL),
    re.compile(r'\s*Episode max steps reached\.?', re.IGNORECASE),
    re.compile(r'\s*Nothing is added to your inventory\.?', re.IGNORECASE),
]

def clean_obs_for_key(obs: str) -> str:
    for pat in _STRIP_PATTERNS:
        obs = pat.sub('', obs)
    return obs.strip()


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


def call_llm_batch(model, tokenizer, prompts: list[str],
                   max_new_tokens: int = 1024, device: str = 'cuda') -> list[str]:
    """Direct batch LLM call — no DataProto/TensorDict."""
    chats = [[{'role': 'user', 'content': p}] for p in prompts]
    formatted = [
        tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
        for c in chats
    ]
    enc = tokenizer(formatted, return_tensors='pt', padding=True,
                    truncation=True, max_length=2048)
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = input_ids.shape[1]
    return tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# ALFWorld env replay
# ---------------------------------------------------------------------------

def make_env(seed: int):
    config_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'agent_system', 'environments', 'env_package', 'alfworld', 'configs', 'config_tw.yaml'
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
    env_class = get_environment('AlfredTWEnv')
    base_env = env_class(config, train_eval='train')
    env_inst = base_env.init_env(batch_size=1)
    env_inst.seed(seed)
    return env_inst


def replay_actions(env_inst, seed: int, actions: list[str]) -> dict:
    """Replay a fixed action sequence through the env, collect obs + rewards."""
    env_inst.seed(seed)
    obs_list, infos = env_inst.reset()
    obs_seq = [obs_list[0]]
    reward_seq = []
    won = False
    for action in actions:
        new_obs_list, scores, dones, infos = env_inst.step([action])
        obs_seq.append(new_obs_list[0])
        reward_seq.append(float(scores[0]) if scores else 0.0)
        won = bool(infos.get('won', [False])[0]) if isinstance(infos.get('won'), list) else bool(infos.get('won', False))
        if dones[0]:
            break
    return {'obs_seq': obs_seq, 'reward_seq': reward_seq, 'won': won,
            'episode_reward': sum(reward_seq)}


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def build_graph(state_list_with_beliefs, action_list):
    """Build TPAB graph and return (G, value_dict, idx_to_belief_state)."""
    belief_state_to_idx, idx_to_belief_state = extract_unique_belief_states(state_list_with_beliefs)
    trajectory = build_belief_trajectory(state_list_with_beliefs, action_list, belief_state_to_idx)
    cleaned = [[t for t in traj if t[0] != t[2]] for traj in trajectory]
    flat_unique = unique_trajectory(cleaned)
    G, value_dict = propagate_reward_decay(flat_unique, gamma=0.95, max_iter=1000)
    for idx in range(len(idx_to_belief_state)):
        if idx not in value_dict:
            value_dict[idx] = 0.0
    return G, value_dict, idx_to_belief_state


def graph_to_dot(G, value_dict, idx_to_belief_state, title: str) -> str:
    colors = {0: '#AED6F1', 1: '#A9DFBF', 2: '#F9E79F', 3: '#F5CBA7', 4: '#D2B4DE'}
    lines = [
        f'digraph G {{',
        f'  label="{title}";',
        f'  labelloc=t; fontsize=14;',
        f'  rankdir=LR;',
        f'  node [shape=box, style=filled, fontsize=8];',
    ]
    for node in sorted(idx_to_belief_state.keys()):
        state, belief = idx_to_belief_state[node]
        val = value_dict.get(node, 0.0)
        color = colors.get(belief, '#FDFEFE')
        state_repr = str(state)[:50].replace('"', "'").replace('\n', ' ')
        label = f'N{node}\\nb={belief}\\nV={val:.2f}\\n{state_repr}'
        lines.append(f'  {node} [label="{label}", fillcolor="{color}"];')
    for src, dst, data in G.edges(data=True):
        action = data.get('action', '')[:25].replace('"', "'")
        dv = value_dict.get(dst, 0.0) - value_dict.get(src, 0.0)
        lines.append(f'  {src} -> {dst} [label="{action} ({dv:+.2f})"];')
    lines.append('}')
    return '\n'.join(lines)


def aliasing_stats(idx_to_belief_state):
    from collections import defaultdict
    state_to_beliefs = defaultdict(set)
    for idx, (state, belief) in idx_to_belief_state.items():
        state_to_beliefs[to_hashable(state)].add(belief)
    aliased = [(k, v) for k, v in state_to_beliefs.items() if len(v) > 1]
    vanilla_count = len(state_to_beliefs)
    return aliased, vanilla_count


def render_png(dot_path: Path, png_path: Path):
    import subprocess
    try:
        subprocess.run(['dot', '-Tpng', str(dot_path), '-o', str(png_path)], check=True)
        print(f'[Saved] {png_path.name} → {png_path}')
    except Exception as e:
        print(f'[WARN] Graphviz render failed: {e}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollout_json', default='/workspace/tpab_rollout_output/rollout_raw_io.json')
    parser.add_argument('--model_path', default='/workspace/qwen2.5_1.5b_alfworld_merged')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--outdir', default='/workspace/tpab_analysis_output')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # -----------------------------------------------------------------------
    # [1] Load rollout data
    # -----------------------------------------------------------------------
    print('=' * 70)
    print(f'Loading rollout data from: {args.rollout_json}')
    with open(args.rollout_json) as f:
        rollout_data = json.load(f)

    task = rollout_data[0]['task']
    print(f'Task: {task}')
    print(f'Rollouts: {len(rollout_data)}')
    for r in rollout_data:
        actions = [s['action'] for s in r['steps']]
        print(f'  [{r["rollout_idx"]}] won={r["won"]} reward={r["episode_reward"]} steps={len(actions)}')

    # -----------------------------------------------------------------------
    # [2] Load model
    # -----------------------------------------------------------------------
    print(f'\nLoading model: {args.model_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    print(f'  dtype={model.dtype}  device={device}')

    # -----------------------------------------------------------------------
    # [3] Replay actions through env to recover obs sequences
    # -----------------------------------------------------------------------
    print('\nInitializing ALFWorld env for replay...')
    env_inst = make_env(seed=args.seed)
    print('  Env ready.')

    print('\nReplaying action sequences...')
    replays = []
    for r in rollout_data:
        actions = [s['action'] for s in r['steps']]
        result = replay_actions(env_inst, seed=args.seed, actions=actions)
        replays.append({
            'rollout_idx': r['rollout_idx'],
            'won': result['won'],
            'episode_reward': result['episode_reward'],
            'obs_seq': result['obs_seq'],
            'action_seq': actions,
            'reward_seq': result['reward_seq'],
        })
        print(f'  [{r["rollout_idx"]}] steps={len(actions)} won={result["won"]} reward={result["episode_reward"]:.1f}')

    # -----------------------------------------------------------------------
    # [4] Build raw_state_list and raw_action_list
    # -----------------------------------------------------------------------
    raw_action_list = [r['action_seq'] for r in replays]
    episode_rewards = [r['episode_reward'] for r in replays]
    won_flags = [r['won'] for r in replays]

    # raw: unprocessed obs
    raw_state_list_raw = []
    for r in replays:
        seq = [{'state': obs, 'reward': 0} for obs in r['obs_seq']]
        for j, rwd in enumerate(r['reward_seq']):
            if j + 1 < len(seq):
                seq[j + 1]['reward'] = rwd
        raw_state_list_raw.append(seq)

    # clean: clean_obs_for_key applied
    raw_state_list_clean = []
    for r in replays:
        seq = [{'state': clean_obs_for_key(obs), 'reward': 0} for obs in r['obs_seq']]
        for j, rwd in enumerate(r['reward_seq']):
            if j + 1 < len(seq):
                seq[j + 1]['reward'] = rwd
        raw_state_list_clean.append(seq)

    # -----------------------------------------------------------------------
    # [5] Belief pipeline — Stage 1
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('[STAGE 1] Subgoal Decomposition')
    print('=' * 70)

    success_indices = [i for i, r in enumerate(episode_rewards) if r > 0]
    if success_indices:
        stage1_action_seqs = [raw_action_list[i] for i in success_indices]
        print(f'  Using {len(success_indices)} successful trajectories.')
    else:
        # No successes: use trajectory with most steps as proxy
        longest = max(range(len(raw_action_list)), key=lambda i: len(raw_action_list[i]))
        stage1_action_seqs = [raw_action_list[longest]]
        print(f'  No successes. Using longest trajectory (idx={longest}) as proxy.')

    stage1_prompt = format_stage1_prompt(task, stage1_action_seqs)
    print(f'\n--- STAGE 1 PROMPT ({len(stage1_prompt)} chars) ---')
    print(stage1_prompt[:800] + ('...' if len(stage1_prompt) > 800 else ''))

    print('\n--- CALLING LLM ---')
    stage1_responses = call_llm_batch(model, tokenizer, [stage1_prompt],
                                      max_new_tokens=args.max_new_tokens, device=device)
    stage1_response = stage1_responses[0]

    print('\n--- STAGE 1 RAW RESPONSE ---')
    print(stage1_response)

    subgoals = parse_stage1_response(stage1_response)
    if not subgoals:
        print('[WARN] Parse failed. Using fallback: ["complete task"]')
        subgoals = ['complete task']
    print(f'\nParsed subgoals ({len(subgoals)}):')
    for j, sg in enumerate(subgoals):
        print(f'  {j}: {sg}')

    # Save Stage 1
    stage1_io = {'prompt': stage1_prompt, 'raw_response': stage1_response, 'subgoals': subgoals}
    with open(outdir / 'stage1_raw_io.json', 'w') as f:
        json.dump(stage1_io, f, indent=2, ensure_ascii=False)
    with open(outdir / 'stage1_raw_io.txt', 'w') as f:
        f.write('=== STAGE 1 PROMPT ===\n')
        f.write(stage1_prompt + '\n\n')
        f.write('=== STAGE 1 RAW RESPONSE ===\n')
        f.write(stage1_response + '\n\n')
        f.write('=== PARSED SUBGOALS ===\n')
        f.write('\n'.join(f'{j}: {sg}' for j, sg in enumerate(subgoals)))
    print(f'\n[Saved] stage1_raw_io.json / .txt → {outdir}')

    # -----------------------------------------------------------------------
    # [6] Belief pipeline — Stage 2
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('[STAGE 2] Belief Assignment (batched)')
    print('=' * 70)

    stage2_prompts = [
        format_stage2_prompt(task, subgoals, raw_action_list[i], won=won_flags[i])
        for i in range(len(replays))
    ]

    print(f'  Sending {len(stage2_prompts)} prompts to LLM...')
    stage2_responses = call_llm_batch(model, tokenizer, stage2_prompts,
                                      max_new_tokens=args.max_new_tokens, device=device)

    beliefs_list = []
    stage2_records = []
    for i, (prompt, response) in enumerate(zip(stage2_prompts, stage2_responses)):
        num_actions = len(raw_action_list[i])
        num_states = len(raw_state_list_clean[i])
        assignments = parse_stage2_response(response, len(subgoals), num_actions)

        print(f'\n--- Traj {i} (won={won_flags[i]}) ---')
        print(f'  Actions ({num_actions}): {raw_action_list[i]}')
        print(f'  RAW RESPONSE:')
        for line in response.split('\n'):
            print(f'    {line}')
        print(f'  Assignments (0-based): {assignments}')
        beliefs = derive_segments(assignments or {}, num_states)
        beliefs_list.append(beliefs)
        print(f'  Beliefs: {beliefs}')

        stage2_records.append({
            'traj_idx': i,
            'won': won_flags[i],
            'actions': raw_action_list[i],
            'prompt': prompt,
            'raw_response': response,
            'assignments': assignments,
            'beliefs': beliefs,
        })

    with open(outdir / 'stage2_raw_io.json', 'w') as f:
        json.dump(stage2_records, f, indent=2, ensure_ascii=False)
    with open(outdir / 'stage2_raw_io.txt', 'w') as f:
        for rec in stage2_records:
            f.write(f'{"=" * 60}\n')
            f.write(f'TRAJ {rec["traj_idx"]}  won={rec["won"]}\n')
            f.write(f'{"=" * 60}\n')
            f.write('--- PROMPT ---\n' + rec['prompt'] + '\n\n')
            f.write('--- RAW RESPONSE ---\n' + rec['raw_response'] + '\n\n')
            f.write(f'assignments: {rec["assignments"]}\n')
            f.write(f'beliefs:     {rec["beliefs"]}\n\n')
    print(f'\n[Saved] stage2_raw_io.json / .txt → {outdir}')

    # -----------------------------------------------------------------------
    # [7] Build state_list_with_beliefs for both raw and clean versions
    # -----------------------------------------------------------------------
    def attach_beliefs(raw_state_list, beliefs_list):
        result = []
        for i, state_seq in enumerate(raw_state_list):
            beliefs = beliefs_list[i]
            result.append([
                {'state': s['state'], 'reward': s['reward'],
                 'belief': beliefs[j] if j < len(beliefs) else beliefs[-1]}
                for j, s in enumerate(state_seq)
            ])
        return result

    swb_raw   = attach_beliefs(raw_state_list_raw,   beliefs_list)
    swb_clean = attach_beliefs(raw_state_list_clean, beliefs_list)

    # -----------------------------------------------------------------------
    # [8] Build graphs and generate PNGs
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('[GRAPH] Building both graphs')
    print('=' * 70)

    for label, swb, suffix in [
        ('WITHOUT preprocessing (raw obs)', swb_raw,   'raw'),
        ('WITH preprocessing (clean obs)',  swb_clean, 'clean'),
    ]:
        print(f'\n--- {label} ---')
        G, value_dict, idx_to_bs = build_graph(swb, raw_action_list)

        print(f'  Nodes: {len(idx_to_bs)}  Edges: {G.number_of_edges()}')

        aliased, vanilla_n = aliasing_stats(idx_to_bs)
        print(f'  Aliasing resolved: {len(aliased)}  (TPAB={len(idx_to_bs)}, vanilla={vanilla_n})')
        for state_key, beliefs in aliased:
            state_repr = str(state_key)[:60]
            print(f'    "{state_repr}..." → beliefs {sorted(beliefs)}')

        print('\n  Node values:')
        for node in sorted(idx_to_bs.keys()):
            state, belief = idx_to_bs[node]
            val = value_dict.get(node, 0.0)
            print(f'    N{node:3d} [b={belief}] V={val:.4f}  {str(state)[:60].replace(chr(10)," ")}')

        # Save graph JSON
        graph_data = {
            'nodes': [
                {'id': idx, 'belief': int(idx_to_bs[idx][1]),
                 'value': float(value_dict.get(idx, 0.0)),
                 'state': str(idx_to_bs[idx][0])[:200]}
                for idx in sorted(idx_to_bs.keys())
            ],
            'edges': [
                {'src': int(s), 'dst': int(d), 'action': data.get('action', ''),
                 'delta_v': float(value_dict.get(d, 0.0) - value_dict.get(s, 0.0))}
                for s, d, data in G.edges(data=True)
            ],
        }
        with open(outdir / f'graph_{suffix}.json', 'w') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        print(f'  [Saved] graph_{suffix}.json')

        dot_str = graph_to_dot(G, value_dict, idx_to_bs,
                               title=f'TPAB-RewardFlow ({suffix}) — {task}')
        dot_path = outdir / f'graph_{suffix}.dot'
        png_path = outdir / f'graph_{suffix}.png'
        with open(dot_path, 'w') as f:
            f.write(dot_str)
        render_png(dot_path, png_path)

    # -----------------------------------------------------------------------
    # [9] Summary
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('Done. Outputs:')
    for p in sorted(outdir.iterdir()):
        print(f'  {p}')
    print('=' * 70)


if __name__ == '__main__':
    main()
