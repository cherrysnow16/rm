#!/usr/bin/env python3
"""
Debug script for TPAB-RewardFlow belief pipeline and graph visualization.

Loads the merged ALFWorld-trained Qwen2.5-1.5B model, runs the belief pipeline
on realistic mock trajectories, then builds and saves the belief-augmented state graph.

Usage:
    cd /workspace/rewardflow_sciworld
    python tpab_rewardflow/debug_tpab.py --model_path /workspace/qwen2.5_1.5b_alfworld_merged
"""

import sys
import os
import json
import re
import argparse
import textwrap
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto

# Direct imports (bypasses __init__.py to avoid circular imports via rollout_loop)
from tpab_rewardflow.belief_pipeline import (
    format_stage1_prompt,
    parse_stage1_response,
    format_stage2_prompt,
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

def unique_trajectory(trajectory):
    """De-duplicate edges within each trajectory (local copy to avoid circular import)."""
    result = []
    for traj in trajectory:
        seen = []
        for item in traj:
            if item not in seen:
                seen.append(item)
        result.append(seen)
    flat = [item for sublist in result for item in sublist]
    return flat


# ---------------------------------------------------------------------------
# Local model wrapper  (mimics actor_rollout_wg.generate_sequences interface)
# ---------------------------------------------------------------------------

class LocalModelWrapper:
    """Wraps a HuggingFace model to look like actor_rollout_wg for the belief pipeline."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 1024, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.world_size = 1  # needed for pad_dataproto_to_divisor

    def generate_sequences(self, batch_input: DataProto) -> DataProto:
        input_ids = batch_input.batch["input_ids"].to(self.device)
        attention_mask = batch_input.batch["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only newly generated tokens (responses)
        prompt_len = input_ids.shape[1]
        responses = output[:, prompt_len:].cpu()

        return DataProto(
            batch={"responses": responses},
            non_tensor_batch={},
        )


# ---------------------------------------------------------------------------
# Mock ALFWorld trajectory data
# ---------------------------------------------------------------------------

TASK = "put a clean fork in countertop"

# Trajectory 1: SUCCESS — fork in drawer, clean, place on countertop
TRAJ1 = {
    "obs": [
        "You are in the middle of a room. Looking quickly around you, you see a countertop 1, a drawer 1, a sinkbasin 1. Your task is to: put a clean fork in countertop.",
        "You arrive at drawer 1. The drawer 1 is closed.",
        "You open the drawer 1. In it, you see a fork 1.",
        "You pick up the fork 1 from the drawer 1. You are carrying: a fork 1.",
        "You arrive at sinkbasin 1.",
        "You clean the fork 1 using the sinkbasin 1. You are carrying: a clean fork 1.",
        "You arrive at countertop 1.",
        "You put the fork 1 in/on the countertop 1. You Won!",
    ],
    "actions": [
        "go to drawer 1",
        "open drawer 1",
        "take fork 1 from drawer 1",
        "go to sinkbasin 1",
        "clean fork 1 with sinkbasin 1",
        "go to countertop 1",
        "put fork 1 in/on countertop 1",
    ],
    "rewards": [0, 0, 0, 0, 0, 1, 0, 10],
    "won": True,
}

# Trajectory 2: SUCCESS — fork in cabinet, clean, place on countertop
TRAJ2 = {
    "obs": [
        "You are in the middle of a room. Looking quickly around you, you see a countertop 1, a cabinet 1, a sinkbasin 1. Your task is to: put a clean fork in countertop.",
        "You arrive at cabinet 1.",
        "You open the cabinet 1. In it, you see a fork 1 and a plate 1.",
        "You pick up the fork 1 from the cabinet 1. You are carrying: a fork 1.",
        "You arrive at sinkbasin 1.",
        "You clean the fork 1 using the sinkbasin 1. You are carrying: a clean fork 1.",
        "You arrive at countertop 1.",
        "You put the fork 1 in/on the countertop 1. You Won!",
    ],
    "actions": [
        "go to cabinet 1",
        "open cabinet 1",
        "take fork 1 from cabinet 1",
        "go to sinkbasin 1",
        "clean fork 1 with sinkbasin 1",
        "go to countertop 1",
        "put fork 1 in/on countertop 1",
    ],
    "rewards": [0, 0, 0, 0, 0, 1, 0, 10],
    "won": True,
}

# Trajectory 3: FAILURE — found fork but placed without cleaning
TRAJ3 = {
    "obs": [
        "You are in the middle of a room. Looking quickly around you, you see a countertop 1, a drawer 1, a sinkbasin 1. Your task is to: put a clean fork in countertop.",
        "You arrive at drawer 1. The drawer 1 is closed.",
        "You open the drawer 1. In it, you see a fork 1.",
        "You pick up the fork 1 from the drawer 1. You are carrying: a fork 1.",
        "You arrive at countertop 1.",
        "Nothing happens. The fork 1 is not clean.",
    ],
    "actions": [
        "go to drawer 1",
        "open drawer 1",
        "take fork 1 from drawer 1",
        "go to countertop 1",
        "put fork 1 in/on countertop 1",
    ],
    "rewards": [0, 0, 0, 0, 0, 0],
    "won": False,
}

# Trajectory 4: FAILURE — searched wrong places, never reached goal
TRAJ4 = {
    "obs": [
        "You are in the middle of a room. Your task is to: put a clean fork in countertop.",
        "You arrive at cabinet 1.",
        "You open the cabinet 1. It is empty.",
        "You arrive at cabinet 2. The cabinet 2 is closed.",
        "You open the cabinet 2. In it, you see a plate 1.",
        "You arrive at drawer 1. The drawer 1 is closed.",
        "You open the drawer 1. In it, you see a fork 1.",
        "You pick up the fork 1 from the drawer 1. Episode max steps reached.",
    ],
    "actions": [
        "go to cabinet 1",
        "open cabinet 1",
        "go to cabinet 2",
        "open cabinet 2",
        "go to drawer 1",
        "open drawer 1",
        "take fork 1 from drawer 1",
    ],
    "rewards": [0, 0, 0, 0, 0, 0, 0, 0],
    "won": False,
}

ALL_TRAJS = [TRAJ1, TRAJ2, TRAJ3, TRAJ4]


_STRIP_PATTERNS = [
    re.compile(r'\s*You are carrying:.*$', re.DOTALL),
    re.compile(r'\s*Episode max steps reached\.?', re.IGNORECASE),
    re.compile(r'\s*Nothing is added to your inventory\.?', re.IGNORECASE),
]

def clean_obs_for_key(obs: str) -> str:
    """Strip inventory/terminal suffixes before using as graph node key."""
    for pat in _STRIP_PATTERNS:
        obs = pat.sub('', obs)
    return obs.strip()


def build_raw_lists(trajs):
    """Build raw_state_list and raw_action_list from mock trajectory data."""
    raw_state_list = []
    raw_action_list = []
    for traj in trajs:
        state_seq = []
        for i, obs in enumerate(traj["obs"]):
            reward = traj["rewards"][i] if i < len(traj["rewards"]) else 0
            state_seq.append({
                "state": clean_obs_for_key(obs),
                "reward": float(reward),
                "active_masks": True,
            })
        raw_state_list.append(state_seq)
        raw_action_list.append(traj["actions"])
    return raw_state_list, raw_action_list


# ---------------------------------------------------------------------------
# Simple config namespace
# ---------------------------------------------------------------------------

def make_config():
    return SimpleNamespace(
        data=SimpleNamespace(
            max_prompt_length=3072,
            truncation="left",
        )
    )


# ---------------------------------------------------------------------------
# Direct LLM call (bypasses DataProto infrastructure)
# ---------------------------------------------------------------------------

def call_llm_direct(prompt: str, model, tokenizer, device: str, max_new_tokens: int = 1024) -> str:
    """Call the model directly with a plain text prompt."""
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=3072).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Graph JSON builder (state_graph_to_dot.py format)
# ---------------------------------------------------------------------------

def build_graph_json(
    task: str,
    flat_unique_traj,
    value_dict: dict,
    idx_to_belief_state: dict,
    all_beliefs: list,
    raw_state_list: list,
    raw_action_list: list,
    all_trajs: list,
    belief_state_to_idx: dict,
    gamma: float,
):
    """Build a state_graph JSON compatible with state_graph_to_dot.py."""
    nodes = {}
    for idx, (state, belief) in idx_to_belief_state.items():
        nodes[str(idx)] = {
            "state": str(state),
            "V": round(value_dict.get(idx, 0.0), 6),
            "belief": belief,
        }

    edges = []
    for (src, action, dst, src_r, dst_r) in flat_unique_traj:
        edges.append({
            "src": src,
            "action": action,
            "dst": dst,
            "reward": dst_r,
            "V_src": round(value_dict.get(src, 0.0), 6),
            "V_dst": round(value_dict.get(dst, 0.0), 6),
        })

    full_trajs = []
    for i, traj in enumerate(all_trajs):
        beliefs = all_beliefs[i]
        steps = []
        for j, obs in enumerate(traj["obs"]):
            belief = beliefs[j] if j < len(beliefs) else beliefs[-1]
            key = to_hashable_belief(obs, belief)
            node_idx = belief_state_to_idx.get(key, -1)
            steps.append({
                "anchor_obs": obs,
                "belief": belief,
                "node_idx": node_idx,
                "V": round(value_dict.get(node_idx, 0.0), 6),
                "active": j < len(traj["actions"]),
            })
        full_trajs.append({
            "won": traj["won"],
            "steps_detail": steps,
        })

    return {
        "meta": {
            "task_name": task,
            "gamma": gamma,
            "num_trajs": len(all_trajs),
            "num_success": sum(1 for t in all_trajs if t["won"]),
        },
        "state_graph": {
            "nodes": nodes,
            "edges": edges,
        },
        "full_trajectories": full_trajs,
    }


# ---------------------------------------------------------------------------
# TPAB DOT generator (belief-aware coloring)
# ---------------------------------------------------------------------------

BELIEF_COLORS = ["#dbeafe", "#fef9c3", "#dcfce7", "#fce7f3", "#ede9fe"]

def tpab_graph_to_dot(graph_json: dict) -> str:
    """Convert TPAB graph JSON to Graphviz DOT with belief-colored nodes."""
    meta = graph_json["meta"]
    nodes = graph_json["state_graph"]["nodes"]
    edges = graph_json["state_graph"]["edges"]

    def esc(s): return s.replace("\\", "\\\\").replace('"', '\\"')
    def short(s, n=80): return (s[:n] + "...") if len(s) > n else s

    terminal_ids = {int(k) for k, v in nodes.items() if "Won" in v["state"]}

    lines = []
    lines.append("digraph TPAB_RewardFlow {")
    lines.append(f'  graph [label="TPAB-RewardFlow\\nTask: {esc(meta["task_name"])}", labelloc=t, fontsize=16];')
    lines.append("  rankdir=LR; splines=true; nodesep=0.5; ranksep=0.8;")
    lines.append('  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9];')
    lines.append('  edge [fontname="Helvetica", fontsize=8];')
    lines.append("")

    for nid_str in sorted(nodes.keys(), key=int):
        nid = int(nid_str)
        nd = nodes[nid_str]
        belief = nd.get("belief", 0)
        fill = BELIEF_COLORS[belief % len(BELIEF_COLORS)]
        V = nd["V"]
        state_summary = esc(short(nd["state"]))
        won_mark = " ★WON" if nid in terminal_ids else ""
        label = f"N{nid} [b={belief}]\\nV={V:.3f}{won_mark}\\n{state_summary}"
        if nid in terminal_ids:
            lines.append(f'  S{nid} [fillcolor="{fill}", color="#15803d", penwidth=3, label="{label}"];')
        else:
            lines.append(f'  S{nid} [fillcolor="{fill}", label="{label}"];')

    lines.append("")
    for edge in edges:
        src, dst = edge["src"], edge["dst"]
        action = esc(edge["action"])
        dv = edge["V_dst"] - edge["V_src"]
        dv_str = f"{dv:+.3f}"
        is_terminal = dst in terminal_ids
        color = "#15803d" if is_terminal else ("#b91c1c" if dv < -1e-9 else "#666666")
        pw = 3 if is_terminal else (2 if dv < -1e-9 else 1)
        lbl = f"{action}\\n{dv_str}"
        lines.append(f'  S{src} -> S{dst} [color="{color}", penwidth={pw}, label="{lbl}"];')

    lines.append("")
    # Legend
    lines.append("  subgraph cluster_legend {")
    lines.append('    label="Belief (subgoal phase)"; fontsize=10; style="dashed";')
    for i, color in enumerate(BELIEF_COLORS[:5]):
        lines.append(f'    L{i} [label="Belief {i}", fillcolor="{color}", style="rounded,filled", shape=box];')
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/workspace/qwen2.5_1.5b_alfworld_merged")
    parser.add_argument("--output_dir", default="/workspace/tpab_debug_output")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("TPAB-RewardFlow Debug Script")
    print("=" * 70)

    # 1. Load model & tokenizer
    print(f"\n[1] Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    print(f"    Model loaded. Device: {args.device}, dtype: {model.dtype}")

    # 2. Build mock trajectory data
    print(f"\n[2] Building mock trajectory data ({len(ALL_TRAJS)} trajectories)")
    raw_state_list, raw_action_list = build_raw_lists(ALL_TRAJS)
    episode_rewards = [sum(t["rewards"]) for t in ALL_TRAJS]
    print(f"    Episode rewards: {episode_rewards}")
    print(f"    Success trajs  : {[t['won'] for t in ALL_TRAJS]}")

    success_indices = [i for i, r in enumerate(episode_rewards) if r > 0]
    print(f"    Success indices: {success_indices}")

    # 3. Stage 1 — Subgoal Decomposition
    print("\n" + "=" * 70)
    print("[STAGE 1] Subgoal Decomposition")
    print("=" * 70)
    success_actions = [raw_action_list[i] for i in success_indices]
    stage1_prompt = format_stage1_prompt(TASK, success_actions)
    print("\n--- STAGE 1 PROMPT (truncated) ---")
    print(textwrap.indent(stage1_prompt[:800] + ("..." if len(stage1_prompt) > 800 else ""), "  "))

    print("\n--- CALLING LLM ---")
    stage1_response = call_llm_direct(stage1_prompt, model, tokenizer, args.device)
    print("\n--- STAGE 1 RESPONSE ---")
    print(textwrap.indent(stage1_response, "  "))

    subgoals = parse_stage1_response(stage1_response)
    if subgoals is None:
        print("\n[WARN] Stage 1 parse failed. Using fallback: ['complete task']")
        subgoals = ["complete task"]
    else:
        print(f"\n--- PARSED SUBGOALS ({len(subgoals)}) ---")
        for i, sg in enumerate(subgoals):
            print(f"  {i}: {sg}")

    # Save stage 1 output
    with open(os.path.join(args.output_dir, "stage1_output.txt"), "w") as f:
        f.write(f"TASK: {TASK}\n\nPROMPT:\n{stage1_prompt}\n\nRESPONSE:\n{stage1_response}\n\nPARSED SUBGOALS:\n")
        for i, sg in enumerate(subgoals):
            f.write(f"{i}: {sg}\n")

    # 4. Stage 2 — Belief Assignment (per trajectory)
    print("\n" + "=" * 70)
    print("[STAGE 2] Belief Assignment (per trajectory)")
    print("=" * 70)

    all_beliefs = []
    stage2_details = []

    for traj_idx in range(len(ALL_TRAJS)):
        print(f"\n--- Trajectory {traj_idx} (won={ALL_TRAJS[traj_idx]['won']}) ---")
        actions = raw_action_list[traj_idx]
        state_seq = raw_state_list[traj_idx]
        num_states = len(state_seq)
        num_actions = len(actions)

        stage2_prompt = format_stage2_prompt(TASK, subgoals, actions, won=ALL_TRAJS[traj_idx]["won"])
        print(f"  Actions ({num_actions}): {actions}")

        stage2_response = call_llm_direct(stage2_prompt, model, tokenizer, args.device, max_new_tokens=512)
        print(f"\n  STAGE 2 RESPONSE:\n{textwrap.indent(stage2_response, '    ')}")

        assignments = parse_stage2_response(stage2_response, len(subgoals), num_actions)
        if assignments is None:
            print(f"  [WARN] Parse failed → fallback beliefs all 0")
            beliefs = [0] * num_states
        else:
            print(f"  Assignments (0-based): {assignments}")
            beliefs = derive_segments(assignments, num_states)

        print(f"  Beliefs per state: {beliefs}")
        all_beliefs.append(beliefs)
        stage2_details.append({
            "traj_idx": traj_idx,
            "won": ALL_TRAJS[traj_idx]["won"],
            "actions": actions,
            "response": stage2_response,
            "assignments": assignments,
            "beliefs": beliefs,
        })

    # Save stage 2 output
    with open(os.path.join(args.output_dir, "stage2_output.json"), "w") as f:
        json.dump(stage2_details, f, indent=2, default=str)
    print(f"\n[Saved] stage2_output.json → {args.output_dir}")

    # 5. Build belief-augmented state graph
    print("\n" + "=" * 70)
    print("[GRAPH] Building Belief-Augmented State Graph")
    print("=" * 70)

    # Attach beliefs to state sequences
    state_list_with_beliefs = []
    for i, state_seq in enumerate(raw_state_list):
        beliefs = all_beliefs[i]
        augmented = []
        for j, step in enumerate(state_seq):
            b = beliefs[j] if j < len(beliefs) else beliefs[-1]
            augmented.append({"state": step["state"], "reward": step["reward"], "belief": b})
        state_list_with_beliefs.append(augmented)

    belief_state_to_idx, idx_to_belief_state = extract_unique_belief_states(state_list_with_beliefs)
    print(f"  Unique belief-augmented nodes: {len(belief_state_to_idx)}")

    trajectory = build_belief_trajectory(state_list_with_beliefs, raw_action_list, belief_state_to_idx)

    # Clean self-loops
    cleaned = []
    for traj in trajectory:
        cleaned.append([t for t in traj if t[0] != t[2]])

    flat_unique_traj = unique_trajectory(cleaned)
    print(f"  Unique edges in graph: {len(flat_unique_traj)}")

    G, value_dict = propagate_reward_decay(flat_unique_traj, gamma=args.gamma, max_iter=1000)

    # Fill missing
    for idx in range(len(idx_to_belief_state)):
        if idx not in value_dict:
            value_dict[idx] = 0.0

    print("\n  Node values (belief-augmented):")
    for idx in sorted(idx_to_belief_state.keys()):
        state, belief = idx_to_belief_state[idx]
        V = value_dict.get(idx, 0.0)
        state_short = str(state)[:60] + ("..." if len(str(state)) > 60 else "")
        print(f"    Node {idx:3d} [belief={belief}] V={V:.4f}  state: {state_short}")

    # Print graph edges
    print("\n  Graph edges (src → dst, action, ΔV):")
    for (src, action, dst, sr, dr) in flat_unique_traj:
        dv = value_dict.get(dst, 0) - value_dict.get(src, 0)
        print(f"    {src:3d} → {dst:3d}  [{action}]  ΔV={dv:+.4f}")

    # Compare: vanilla RewardFlow nodes vs TPAB nodes
    print("\n  --- ALIASING ANALYSIS ---")
    obs_to_beliefs = {}
    for idx, (state, belief) in idx_to_belief_state.items():
        obs_str = str(state)
        if obs_str not in obs_to_beliefs:
            obs_to_beliefs[obs_str] = []
        obs_to_beliefs[obs_str].append((belief, idx))
    aliased = {obs: pairs for obs, pairs in obs_to_beliefs.items() if len(pairs) > 1}
    print(f"  Observations appearing at multiple beliefs (aliasing resolved): {len(aliased)}")
    for obs, pairs in aliased.items():
        print(f"    '{obs[:60]}...' → beliefs {[b for b, _ in pairs]}")
    print(f"  Total unique nodes (TPAB): {len(belief_state_to_idx)}")
    print(f"  Total unique nodes (vanilla would have): {len(obs_to_beliefs)}")

    # 6. Save graph JSON
    graph_json = build_graph_json(
        task=TASK,
        flat_unique_traj=flat_unique_traj,
        value_dict=value_dict,
        idx_to_belief_state=idx_to_belief_state,
        all_beliefs=all_beliefs,
        raw_state_list=raw_state_list,
        raw_action_list=raw_action_list,
        all_trajs=ALL_TRAJS,
        belief_state_to_idx=belief_state_to_idx,
        gamma=args.gamma,
    )
    graph_json_path = os.path.join(args.output_dir, "tpab_graph.json")
    with open(graph_json_path, "w") as f:
        json.dump(graph_json, f, indent=2, default=str)
    print(f"\n[Saved] tpab_graph.json → {graph_json_path}")

    # 7. Generate DOT file
    dot_str = tpab_graph_to_dot(graph_json)
    dot_path = os.path.join(args.output_dir, "tpab_graph.dot")
    with open(dot_path, "w") as f:
        f.write(dot_str)
    print(f"[Saved] tpab_graph.dot → {dot_path}")

    # 8. Render to PNG if graphviz available
    png_path = os.path.join(args.output_dir, "tpab_graph.png")
    try:
        import subprocess
        result = subprocess.run(
            ["dot", "-Tpng", "-o", png_path, dot_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print(f"[Saved] tpab_graph.png → {png_path}")
        else:
            print(f"[WARN] Graphviz render failed: {result.stderr[:200]}")
    except (FileNotFoundError, Exception) as e:
        print(f"[INFO] Graphviz not available: {e}")
        print(f"       Install: apt-get install graphviz")
        print(f"       Then: dot -Tpng -o {png_path} {dot_path}")

    print("\n" + "=" * 70)
    print("Done. Outputs:")
    print(f"  {args.output_dir}/stage1_output.txt  — Stage 1 LLM prompt + response")
    print(f"  {args.output_dir}/stage2_output.json — Stage 2 per-traj assignments")
    print(f"  {args.output_dir}/tpab_graph.json    — Belief-augmented graph (JSON)")
    print(f"  {args.output_dir}/tpab_graph.dot     — Graphviz DOT for rendering")
    if os.path.exists(png_path):
        print(f"  {args.output_dir}/tpab_graph.png    — Rendered graph image")
    print("=" * 70)


if __name__ == "__main__":
    main()
