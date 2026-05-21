#!/usr/bin/env python3
"""
Find a task where the model succeeds (at least one trajectory wins),
then run 8 mixed rollouts (greedy + sampled) and the full TPAB belief pipeline.

Phase 1: Try seeds 0..MAX_SEED_SEARCH with greedy decoding; stop at first win.
Phase 2: Run N_ROLLOUTS on the found seed:
           - 1 greedy rollout (the one that won)
           - N_ROLLOUTS-1 sampled (temperature=TEMPERATURE) for diversity
Phase 3: Stage 1 + Stage 2 belief pipeline; build both raw/clean graphs.

Usage:
    cd /workspace/rewardflow_sciworld
    ALFWORLD_DATA=/root/.cache/alfworld python -u tpab_rewardflow/find_success_and_analyze.py \
        --model_path /workspace/qwen2.5_1.5b_alfworld_merged \
        --outdir /workspace/tpab_success_output
"""

import os, sys, re, json, argparse, textwrap
from pathlib import Path
from collections import defaultdict

if 'ALFWORLD_DATA' not in os.environ:
    os.environ['ALFWORLD_DATA'] = '/root/.cache/alfworld'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
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
# Constants
# ---------------------------------------------------------------------------

MAX_SEED_SEARCH = 50     # maximum seeds to try before giving up
N_ROLLOUTS = 8           # number of rollouts per task
TEMPERATURE = 0.8        # temperature for sampled rollouts
MAX_STEPS = 30           # max steps per episode
HISTORY_LEN = 3

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
# Obs preprocessing
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
# LLM helpers (bypass DataProto entirely)
# ---------------------------------------------------------------------------

def call_llm_batch(model, tokenizer, prompts: list[str],
                   max_new_tokens: int = 1024,
                   do_sample: bool = False,
                   temperature: float = 1.0,
                   device: str = 'cuda') -> list[str]:
    chats = [[{'role': 'user', 'content': p}] for p in prompts]
    formatted = [
        tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
        for c in chats
    ]
    enc = tokenizer(formatted, return_tensors='pt', padding=True,
                    truncation=True, max_length=2048)
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs['do_sample'] = True
        gen_kwargs['temperature'] = temperature
    else:
        gen_kwargs['do_sample'] = False
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    prompt_len = input_ids.shape[1]
    return tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)


def call_llm_single(model, tokenizer, prompt: str,
                    max_new_tokens: int = 512,
                    do_sample: bool = False,
                    temperature: float = 1.0,
                    device: str = 'cuda') -> str:
    responses = call_llm_batch(
        model, tokenizer, [prompt],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        device=device,
    )
    return responses[0]


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def extract_task(obs: str) -> str:
    marker = 'Your task is to: '
    idx = obs.find(marker)
    if idx != -1:
        return obs[idx + len(marker):].strip().split('\n')[0]
    return "complete the task"


def alfworld_projection(response: str, admissible: list[str]) -> tuple[str, bool]:
    lower = response.lower()
    start = lower.find('<action>')
    end = lower.find('</action>')
    has_think = '<think>' in lower and '</think>' in lower
    has_chinese = bool(re.search(r'[一-鿿]', response))

    if start == -1 or end == -1 or has_chinese or not has_think:
        return admissible[0], False

    extracted = lower[start + len('<action>'):end].strip()

    for cmd in admissible:
        if cmd.lower() == extracted:
            return cmd, True
    for cmd in admissible:
        if extracted in cmd.lower() or cmd.lower() in extracted:
            return cmd, True
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
# ALFWorld env
# ---------------------------------------------------------------------------

def make_env(config_path: str, seed: int):
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    env_class = get_environment('AlfredTWEnv')
    base_env = env_class(cfg, train_eval='train')
    env_inst = base_env.init_env(batch_size=1)
    env_inst.seed(seed)
    return env_inst


def run_rollout(env_inst, model, tokenizer, seed: int,
                max_steps: int, do_sample: bool, temperature: float,
                device: str, history_length: int = HISTORY_LEN,
                verbose: bool = False) -> dict:
    env_inst.seed(seed)
    obs_list, infos = env_inst.reset()
    obs = obs_list[0]
    task = extract_task(obs)

    history = []
    obs_seq = [obs]
    action_seq = []
    reward_seq = []
    raw_steps = []
    won = False

    for step in range(max_steps):
        admissible = infos['admissible_commands'][0]
        prompt = build_prompt(obs, admissible, task,
                              history[-history_length:] if history else [], step)
        raw_out = call_llm_single(
            model, tokenizer, prompt,
            max_new_tokens=512,
            do_sample=do_sample,
            temperature=temperature,
            device=device,
        )
        action, valid = alfworld_projection(raw_out, admissible)

        if verbose:
            print(f'\n  ---- Step {step} ----')
            print(f'  [PROMPT]\n{textwrap.indent(prompt, "    ")}')
            print(f'  [RAW OUTPUT]\n{textwrap.indent(raw_out, "    ")}')
            print(f'  [ACTION] {action}  (valid={valid})')

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

        if verbose:
            print(f'  [NEW OBS] {new_obs[:120].replace(chr(10), " ")}')
            print(f'  [REWARD] {reward}')

        history.append((obs, action))
        obs = new_obs
        obs_seq.append(obs)
        action_seq.append(action)
        reward_seq.append(reward)

        if done:
            won_info = infos.get('won', [False])
            won = bool(won_info[0]) if isinstance(won_info, list) else bool(won_info)
            break

    if not won:
        won_info = infos.get('won', [False])
        won = bool(won_info[0]) if isinstance(won_info, list) else bool(won_info)

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
# Graph helpers
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


def _is_useless_action(action: str, to_state_str: str, from_idx: int, to_idx: int) -> bool:
    """Mirror of env_manager.py clean_trajectory filter conditions."""
    if action in ('inventory', 'look'):
        return True
    if 'examine' in action:
        return True
    if 'Nothing happens' in to_state_str:
        return True
    if from_idx == to_idx:
        return True
    return False


def clean_trajectory(trajectory, idx_to_belief_state):
    """
    Remove inventory/look/examine/Nothing-happens/self-loop transitions.
    When a useless action is encountered, the accumulated from_state is
    reconnected to the next valid to_state (same logic as env_manager.py).
    """
    cleaned = []
    for traj in trajectory:
        cleaned_traj = []
        cur_from_state = None
        j = 0
        while j < len(traj):
            from_idx, action, to_idx, from_reward, to_reward = traj[j]
            to_state_str = str(idx_to_belief_state[to_idx][0])

            if _is_useless_action(action, to_state_str, from_idx, to_idx):
                if cur_from_state is None:
                    cur_from_state = from_idx
                # scan forward for the next valid action
                while j < len(traj):
                    from_idx2, action2, to_idx2, fr2, tr2 = traj[j]
                    to_state_str2 = str(idx_to_belief_state[to_idx2][0])
                    if _is_useless_action(action2, to_state_str2, cur_from_state, to_idx2):
                        j += 1
                    else:
                        cleaned_traj.append((cur_from_state, action2, to_idx2, fr2, tr2))
                        cur_from_state = None
                        j += 1
                        break
            else:
                cleaned_traj.append((from_idx, action, to_idx, from_reward, to_reward))
                cur_from_state = None
                j += 1
        cleaned.append(cleaned_traj)
    return cleaned


def build_graph(state_list_with_beliefs, action_list):
    belief_state_to_idx, idx_to_belief_state = extract_unique_belief_states(state_list_with_beliefs)
    trajectory = build_belief_trajectory(state_list_with_beliefs, action_list, belief_state_to_idx)
    cleaned = clean_trajectory(trajectory, idx_to_belief_state)
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
        belief_size = len(belief) if isinstance(belief, frozenset) else belief
        color = colors.get(belief_size, '#FDFEFE')
        belief_repr = '{' + ','.join(str(x) for x in sorted(belief)) + '}' if isinstance(belief, frozenset) else str(belief)
        state_repr = str(state)[:50].replace('"', "'").replace('\n', ' ')
        label = f'N{node}\\nb={belief_repr}\\nV={val:.2f}\\n{state_repr}'
        lines.append(f'  {node} [label="{label}", fillcolor="{color}"];')
    for src, dst, data in G.edges(data=True):
        action = data.get('action', '')[:25].replace('"', "'")
        dv = value_dict.get(dst, 0.0) - value_dict.get(src, 0.0)
        lines.append(f'  {src} -> {dst} [label="{action} ({dv:+.2f})"];')
    lines.append('}')
    return '\n'.join(lines)


def render_png(dot_path: Path, png_path: Path):
    import subprocess
    try:
        subprocess.run(['dot', '-Tpng', str(dot_path), '-o', str(png_path)], check=True)
        print(f'[Saved] {png_path.name} → {png_path}')
    except Exception as e:
        print(f'[WARN] Graphviz render failed: {e}')


def aliasing_stats(idx_to_belief_state):
    state_to_beliefs = defaultdict(set)
    for idx, (state, belief) in idx_to_belief_state.items():
        # frozenset is not hashable inside a set directly — convert to tuple for set storage
        belief_key = tuple(sorted(belief)) if isinstance(belief, frozenset) else belief
        state_to_beliefs[to_hashable(state)].add(belief_key)
    aliased = [(k, v) for k, v in state_to_beliefs.items() if len(v) > 1]
    return aliased, len(state_to_beliefs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/workspace/qwen2.5_1.5b_alfworld_merged')
    parser.add_argument('--outdir', default='/workspace/tpab_success_output')
    parser.add_argument('--max_seed_search', type=int, default=MAX_SEED_SEARCH)
    parser.add_argument('--n_rollouts', type=int, default=N_ROLLOUTS)
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
    parser.add_argument('--temperature', type=float, default=TEMPERATURE)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'agent_system', 'environments', 'env_package', 'alfworld', 'configs', 'config_tw.yaml'
    )

    # -----------------------------------------------------------------------
    # [1] Load model
    # -----------------------------------------------------------------------
    print('=' * 70)
    print(f'Loading model: {args.model_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    print(f'  dtype={model.dtype}  device={device}')

    # -----------------------------------------------------------------------
    # [2] Phase 1: Find a seed where greedy rollout succeeds
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print(f'[Phase 1] Searching for a successful task (seeds 0..{args.max_seed_search - 1})')
    print('=' * 70)

    env_inst = make_env(config_path, seed=0)
    found_seed = None
    found_rollout = None

    for seed in range(args.max_seed_search):
        print(f'\n  Trying seed {seed}...', end=' ', flush=True)
        rollout = run_rollout(
            env_inst, model, tokenizer,
            seed=seed,
            max_steps=args.max_steps,
            do_sample=False,
            temperature=1.0,
            device=device,
        )
        status = 'WON ✓' if rollout['won'] else f'lost (steps={len(rollout["action_seq"])})'
        print(f'task="{rollout["task"]}"  {status}')

        if rollout['won']:
            found_seed = seed
            found_rollout = rollout
            break

    if found_seed is None:
        print(f'\n[ERROR] No successful rollout found in seeds 0-{args.max_seed_search - 1}.')
        print('  Falling back to seed 0 with all sampled rollouts.')
        found_seed = 0
        found_rollout = None

    print(f'\nSelected seed: {found_seed}')
    task_description = (found_rollout or rollout)['task']
    print(f'Task: {task_description}')

    # -----------------------------------------------------------------------
    # [3] Phase 2: Run N rollouts on the found seed (1 greedy + N-1 sampled)
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print(f'[Phase 2] Running {args.n_rollouts} rollouts on seed={found_seed}')
    print(f'  Rollout 0: greedy (already done)')
    print(f'  Rollouts 1-{args.n_rollouts - 1}: sampled (temperature={args.temperature})')
    print('=' * 70)

    all_rollouts = []
    # Rollout 0: the greedy one we already ran (or a fresh greedy if fallback)
    if found_rollout is not None:
        all_rollouts.append(found_rollout)
    else:
        r = run_rollout(env_inst, model, tokenizer, seed=found_seed,
                        max_steps=args.max_steps, do_sample=False,
                        temperature=1.0, device=device)
        all_rollouts.append(r)
    print(f'  [0] greedy  won={all_rollouts[0]["won"]}  steps={len(all_rollouts[0]["action_seq"])}')

    for i in range(1, args.n_rollouts):
        print(f'\n{"=" * 50}')
        print(f'  Rollout {i} (sampled, temperature={args.temperature})')
        print(f'{"=" * 50}')
        r = run_rollout(env_inst, model, tokenizer, seed=found_seed,
                        max_steps=args.max_steps, do_sample=True,
                        temperature=args.temperature, device=device,
                        verbose=True)
        all_rollouts.append(r)
        status = 'WON' if r['won'] else 'lost'
        print(f'\n  [{i}] sampled won={r["won"]}  steps={len(r["action_seq"])}  [{status}]')

    n_won = sum(r['won'] for r in all_rollouts)
    print(f'\nSummary: {n_won}/{args.n_rollouts} succeeded')
    for i, r in enumerate(all_rollouts):
        print(f'  [{i}] won={r["won"]} reward={r["episode_reward"]:.1f} actions={r["action_seq"]}')

    # -----------------------------------------------------------------------
    # [4] Save raw rollout I/O
    # -----------------------------------------------------------------------
    rollout_io = []
    for i, r in enumerate(all_rollouts):
        rollout_io.append({
            'rollout_idx': i,
            'task': r['task'],
            'won': r['won'],
            'episode_reward': r['episode_reward'],
            'steps': r['raw_steps'],
        })
    with open(outdir / 'rollout_raw_io.json', 'w') as f:
        json.dump(rollout_io, f, indent=2, ensure_ascii=False)
    with open(outdir / 'rollout_raw_io.txt', 'w') as f:
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
    print(f'\n[Saved] rollout_raw_io.json / .txt → {outdir}')

    # -----------------------------------------------------------------------
    # [5] Build raw_state_list / raw_action_list
    # -----------------------------------------------------------------------
    raw_action_list = [r['action_seq'] for r in all_rollouts]
    episode_rewards = [r['episode_reward'] for r in all_rollouts]
    won_flags = [r['won'] for r in all_rollouts]

    # raw: unprocessed obs (for graph_raw)
    raw_state_list_raw = []
    for r in all_rollouts:
        seq = [{'state': obs, 'reward': 0.0} for obs in r['obs_seq']]
        for j, rwd in enumerate(r['reward_seq']):
            if j + 1 < len(seq):
                seq[j + 1]['reward'] = rwd
        raw_state_list_raw.append(seq)

    # clean: clean_obs_for_key applied (for graph_clean)
    raw_state_list_clean = []
    for r in all_rollouts:
        seq = [{'state': clean_obs_for_key(obs), 'reward': 0.0} for obs in r['obs_seq']]
        for j, rwd in enumerate(r['reward_seq']):
            if j + 1 < len(seq):
                seq[j + 1]['reward'] = rwd
        raw_state_list_clean.append(seq)

    # -----------------------------------------------------------------------
    # [6] Stage 1 — Subgoal Decomposition
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('[Stage 1] Subgoal Decomposition')
    print('=' * 70)

    success_indices = [i for i, w in enumerate(won_flags) if w]
    if success_indices:
        stage1_action_seqs = [raw_action_list[i] for i in success_indices]
        print(f'  Using {len(success_indices)} successful trajectory action seqs.')
    else:
        longest = max(range(len(raw_action_list)), key=lambda i: len(raw_action_list[i]))
        stage1_action_seqs = [raw_action_list[longest]]
        print(f'  No successes — using longest traj (idx={longest}) as proxy.')

    stage1_prompt = format_stage1_prompt(task_description, stage1_action_seqs)
    print(f'\n--- STAGE 1 PROMPT ({len(stage1_prompt)} chars) ---')
    print(stage1_prompt[:800] + ('...' if len(stage1_prompt) > 800 else ''))

    print('\n--- CALLING LLM ---')
    [stage1_response] = call_llm_batch(model, tokenizer, [stage1_prompt],
                                       max_new_tokens=args.max_new_tokens, device=device)

    print('\n--- STAGE 1 RAW RESPONSE ---')
    print(stage1_response)

    subgoals = parse_stage1_response(stage1_response)
    if not subgoals:
        print('[WARN] Parse failed. Fallback: ["complete task"]')
        subgoals = ['complete task']
    print(f'\nParsed subgoals ({len(subgoals)}):')
    for j, sg in enumerate(subgoals):
        print(f'  {j}: {sg}')

    stage1_io = {'prompt': stage1_prompt, 'raw_response': stage1_response, 'subgoals': subgoals}
    with open(outdir / 'stage1_raw_io.json', 'w') as f:
        json.dump(stage1_io, f, indent=2, ensure_ascii=False)
    with open(outdir / 'stage1_raw_io.txt', 'w') as f:
        f.write('=== STAGE 1 PROMPT ===\n' + stage1_prompt + '\n\n')
        f.write('=== STAGE 1 RAW RESPONSE ===\n' + stage1_response + '\n\n')
        f.write('=== PARSED SUBGOALS ===\n')
        f.write('\n'.join(f'{j}: {sg}' for j, sg in enumerate(subgoals)))
    print(f'\n[Saved] stage1_raw_io.json / .txt → {outdir}')

    # -----------------------------------------------------------------------
    # [7] Stage 2 — Belief Assignment (batched)
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('[Stage 2] Belief Assignment (batched)')
    print('=' * 70)

    stage2_prompts = [
        format_stage2_prompt(task_description, subgoals, raw_action_list[i], won=won_flags[i])
        for i in range(len(all_rollouts))
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
    # [8] Build both graphs (raw obs and clean obs)
    # -----------------------------------------------------------------------
    def attach_beliefs(raw_state_list, beliefs_list):
        result = []
        for i, state_seq in enumerate(raw_state_list):
            bs = beliefs_list[i]
            result.append([
                {'state': s['state'], 'reward': s['reward'],
                 'belief': bs[j] if j < len(bs) else bs[-1]}
                for j, s in enumerate(state_seq)
            ])
        return result

    swb_raw   = attach_beliefs(raw_state_list_raw,   beliefs_list)
    swb_clean = attach_beliefs(raw_state_list_clean, beliefs_list)

    print('\n' + '=' * 70)
    print('[Graph] Building both graphs')
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
        for state_key, beliefs_set in aliased:
            print(f'    "{str(state_key)[:60]}..." → beliefs {sorted(beliefs_set)}')

        print('\n  Node values:')
        for node in sorted(idx_to_bs.keys()):
            state, belief = idx_to_bs[node]
            val = value_dict.get(node, 0.0)
            print(f'    N{node:3d} [b={belief}] V={val:.4f}  {str(state)[:60].replace(chr(10), " ")}')

        print('\n  Edges (src→dst, action, ΔV):')
        for src, dst, data in G.edges(data=True):
            action = data.get('action', '')
            dv = value_dict.get(dst, 0.0) - value_dict.get(src, 0.0)
            print(f'    {src:3d}→{dst:3d}  [{action}]  ΔV={dv:+.4f}')

        graph_data = {
            'nodes': [
                {'id': idx, 'belief': sorted(idx_to_bs[idx][1]) if isinstance(idx_to_bs[idx][1], frozenset) else idx_to_bs[idx][1],
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
                               title=f'TPAB ({suffix}) — {task_description}')
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
