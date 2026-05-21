"""
Convert RewardFlow state_graph JSON to Graphviz DOT format.

Usage:
    python state_graph_to_dot.py <input.json> [output.dot]
    python state_graph_to_dot.py <input_dir/> [output_dir/]

If output path is omitted, prints DOT to stdout (single file) or writes
.dot files next to the JSON files (directory mode).
"""

import json
import re
import sys
from pathlib import Path

#python3 scripts/state_graph_to_dot.py /workspace/RewardFlow/outputs/aliasing/2026-04-14_14-10-35/aliasing /workspace/RewardFlow/outputs/aliasing/2026-04-14_14-10-35


# ---------------------------------------------------------------------------
# Observation text parsers
# ---------------------------------------------------------------------------

def _strip_article(s: str) -> str:
    """Remove leading 'a '/'an '/'the ' from a captured string."""
    return re.sub(r"^(a |an |the )", "", s, flags=re.IGNORECASE).strip()


def parse_location(state_text: str) -> str:
    """Extract agent location name from a TextWorld observation string."""
    if not state_text or "Won" in state_text:
        return None
    if "in the middle of a room" in state_text or "Welcome to TextWorld" in state_text:
        return "room_center"

    patterns = [
        r"You arrive at (.+?)\.",
        r"You open the (.+?)\.",
        r"You close the (.+?)\.",
        r"You pick up .+ from (?:the )?(.+?)\.",
        r"You move .+ to (?:the )?(.+?)\.",
        r"You clean .+ using (?:the )?(.+?)\.",
        r"You put .+ in (?:the )?(.+?)\.",
        r"You heat .+ with (?:the )?(.+?)\.",
        r"You cool .+ with (?:the )?(.+?)\.",
    ]
    for pat in patterns:
        m = re.search(pat, state_text)
        if m:
            return _strip_article(m.group(1))
    return None


def parse_inventory(state_text: str) -> str:
    """Extract held item(s) from a TextWorld observation string."""
    if "You are not carrying anything" in state_text:
        return None
    m = re.search(r"You are carrying: (.+?)\.", state_text)
    if m:
        item = m.group(1)
        # strip leading "a "
        item = re.sub(r"^a ", "", item)
        return item
    return None


def extract_obj_summary(oracle_state: list, max_items: int = 6) -> str:
    """Build a compact 'obj: X @ Y, ...' string from PDDL oracle facts."""
    inrec = []
    for fact in oracle_state:
        m = re.match(r"inreceptacle\((.+?): object, (.+?): receptacle\)", fact)
        if m:
            inrec.append(f"{m.group(1)} @ {m.group(2)}")
    if not inrec:
        return ""
    shown = inrec[:max_items]
    suffix = "..." if len(inrec) > max_items else ""
    return "obj: " + ", ".join(shown) + suffix


# ---------------------------------------------------------------------------
# DOT node / edge helpers
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    """Escape double-quotes and backslashes for a DOT label string."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _truncate_state_text(text: str, max_chars: int = 120) -> str:
    """Return the first sentence(s) of a TextWorld observation, up to max_chars."""
    # Strip the TextWorld welcome header if present (only on the very first obs)
    text = re.sub(r"-= Welcome to TextWorld, ALFRED! =-\s*\n\n", "", text).strip()
    # Use the first two sentences (split on ". ")
    sentences = re.split(r"(?<=\.)\s+", text)
    result = " ".join(sentences[:2])
    if len(result) > max_chars:
        result = result[:max_chars].rstrip() + "..."
    return result


def make_node_label(nid: int, node_data: dict, oracle_state: list | None) -> str:
    state_text = node_data["state"]
    V = node_data["V"]

    if "Won" in state_text or "SUCCESS" in state_text:
        return "\\n".join([f"State {nid}", f"V={V:.4f}", "SUCCESS"])

    summary = _truncate_state_text(state_text)
    parts = [f"State {nid}", f"V={V:.4f}", _esc(summary)]

    return "\\n".join(parts)


def node_attrs(nid: int, terminal_ids: set, winning_ids: set) -> dict:
    if nid in terminal_ids:
        return dict(
            shape="doubleoctagon",
            fillcolor="#bbf7d0",
            color="#15803d",
            penwidth=3,
        )
    if nid == 0:
        return dict(fillcolor="#dbeafe", penwidth=2)
    if nid in winning_ids:
        return dict(fillcolor="#fff7d6", penwidth=2)
    return dict(fillcolor="white", penwidth=1)


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def state_graph_to_dot(data: dict, graph_name: str = "RewardFlowStateGraph") -> str:
    """
    Convert a state_graph dict (from RewardFlow JSON) to a Graphviz DOT string.

    Parameters
    ----------
    data : dict
        Parsed JSON content of a RewardFlow output file.
    graph_name : str
        Identifier used as the DOT graph name.

    Returns
    -------
    str
        Complete DOT source string.
    """
    meta = data.get("meta", {})
    sg = data["state_graph"]
    nodes: dict = sg["nodes"]        # {str_id: {state, V}}
    edges: list = sg["edges"]        # [{src, action, dst, reward, V_src, V_dst}]
    trajectories: list = data.get("full_trajectories", [])

    # ---- task description ------------------------------------------------
    first_obs = nodes.get("0", {}).get("state", "")
    m = re.search(r"Your task is to: (.+?)\.", first_obs)
    task_desc = m.group(1) if m else meta.get("task_name", "unknown task")

    # ---- terminal nodes ---------------------------------------------------
    terminal_ids = {
        int(k)
        for k, v in nodes.items()
        if "Won" in v["state"] or "SUCCESS" in v["state"]
    }

    # ---- winning-path nodes + oracle state cache --------------------------
    obs_to_node: dict[str, int] = {v["state"]: int(k) for k, v in nodes.items()}
    winning_path_nodes: set[int] = set()
    node_oracle: dict[int, list] = {}

    for traj in trajectories:
        if not traj.get("won"):
            continue
        for step in traj.get("steps_detail", []):
            nid = obs_to_node.get(step.get("anchor_obs"))
            if nid is None:
                continue
            winning_path_nodes.add(nid)
            if nid not in node_oracle and step.get("oracle_state"):
                node_oracle[nid] = step["oracle_state"]

    winning_path_nodes |= terminal_ids  # terminal is always on winning path

    # ---- bad edges: 실패 trajectory → terminal 노드 연결 -----------------
    # 실패 traj의 마지막 행동이 성공 노드("You Won!")에 연결된 경우 DOT에서 제거
    bad_edges: set[tuple[int, int]] = set()
    for traj in trajectories:
        if traj.get("won"):
            continue
        steps = traj.get("steps_detail", [])
        for k in range(len(steps) - 1):
            if not steps[k].get("active"):
                break
            src_nid = obs_to_node.get(steps[k].get("anchor_obs"))
            dst_nid = obs_to_node.get(steps[k + 1].get("anchor_obs"))
            if src_nid is not None and dst_nid in terminal_ids:
                bad_edges.add((src_nid, dst_nid))

    # ---- re-propagate V values without bad edges --------------------------
    # bad edge가 포함된 채로 계산된 V값은 오염됐으므로 제거 후 재계산한다.
    import networkx as nx
    gamma = meta.get("gamma", 0.9)

    clean_G = nx.DiGraph()
    clean_reward: dict[int, float] = {}
    for edge in edges:
        src, dst = edge["src"], edge["dst"]
        if (src, dst) in bad_edges:
            continue
        clean_G.add_edge(src, dst)
        r = edge["reward"] if edge.get("reward") is not None else 0.0
        clean_reward.setdefault(src, r)
        clean_reward[dst] = 10.0 if dst in terminal_ids else clean_reward.get(dst, 0.0)

    # graph에 없는 isolated 노드도 초기화
    for nid_str in nodes:
        nid = int(nid_str)
        clean_reward.setdefault(nid, 10.0 if nid in terminal_ids else 0.0)

    # value iteration (propagate_reward_decay와 동일 로직)
    clean_V: dict[int, float] = dict(clean_reward)
    for _ in range(1000):
        new_V = clean_V.copy()
        for node in clean_G.nodes:
            max_val = clean_V.get(node, 0.0)
            for _, succ in clean_G.out_edges(node):
                prop = gamma * clean_V.get(succ, 0.0)
                if prop > max_val:
                    max_val = prop
            new_V[node] = max_val
        clean_V = new_V

    # node 데이터의 V를 재계산된 값으로 교체
    for nid_str in nodes:
        nid = int(nid_str)
        nodes[nid_str] = {**nodes[nid_str], "V": round(clean_V.get(nid, 0.0), 6)}

    # ---- build DOT --------------------------------------------------------
    lines = []
    lines.append(f"digraph {graph_name} {{")
    lines.append(
        f'    graph [label="REWARDFLOW STATE GRAPH\\nTask: {_esc(task_desc)}.",'
        f' labelloc=t, fontsize=20];'
    )
    lines.append("    rankdir=LR;")
    lines.append("    splines=true;")
    lines.append("    overlap=false;")
    lines.append("    nodesep=0.45;")
    lines.append("    ranksep=0.75;")
    lines.append(
        '    node [shape=box, style="rounded,filled", fillcolor=white,'
        ' color="#444444", fontname="Helvetica", fontsize=10];'
    )
    lines.append(
        '    edge [color="#666666", fontname="Helvetica", fontsize=9];'
    )
    lines.append("")

    # nodes
    for nid_str in sorted(nodes.keys(), key=int):
        nid = int(nid_str)
        label = make_node_label(nid, nodes[nid_str], node_oracle.get(nid))
        attrs = node_attrs(nid, terminal_ids, winning_path_nodes)

        attr_str = ", ".join(
            f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
            for k, v in attrs.items()
        )
        lines.append(f'    S{nid} [{attr_str}, label="{label}"];')

    lines.append("")

    # edges
    for step_i, edge in enumerate(edges):
        src: int = edge["src"]
        dst: int = edge["dst"]
        if (src, dst) in bad_edges:
            continue  # 실패 traj → 성공 노드 연결 제거
        action: str = edge["action"]
        dv: float = clean_V.get(dst, 0.0) - clean_V.get(src, 0.0)
        dv_str = f"{dv:+.4f}"

        is_terminal = dst in terminal_ids
        is_negative = dv < -1e-9

        if is_terminal:
            color = fontcolor = "#15803d"
            penwidth = 3
            lbl = f"step {step_i}\\n {_esc(action)}\\n {dv_str}\\n[WON!]"
        elif is_negative:
            color = fontcolor = "#b91c1c"
            penwidth = 2
            lbl = f"step {step_i}\\n {_esc(action)}\\n {dv_str}"
        else:
            color = fontcolor = "#666666"
            penwidth = 1
            lbl = f"step {step_i}\\n {_esc(action)}\\n {dv_str}"

        if is_terminal or is_negative:
            lines.append(
                f'    S{src} -> S{dst} [color="{color}", fontcolor="{fontcolor}",'
                f' penwidth={penwidth}, label="{lbl}"];'
            )
        else:
            lines.append(
                f'    S{src} -> S{dst} [color="{color}", label="{lbl}"];'
            )

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _process_file(json_path: Path, dot_path: Path | None) -> None:
    with open(json_path) as f:
        data = json.load(f)

    if "state_graph" not in data:
        print(f"[skip] No state_graph in {json_path}", file=sys.stderr)
        return

    # derive a safe graph name from the file stem
    graph_name = re.sub(r"[^A-Za-z0-9_]", "_", json_path.stem)

    dot_source = state_graph_to_dot(data, graph_name=graph_name)

    if dot_path is None:
        print(dot_source)
    else:
        dot_path.parent.mkdir(parents=True, exist_ok=True)
        dot_path.write_text(dot_source)
        print(f"[ok] {json_path.name} → {dot_path}", file=sys.stderr)


def main(argv: list[str]) -> None:
    if not argv:
        print(__doc__)
        sys.exit(0)

    src = Path(argv[0])
    dst = Path(argv[1]) if len(argv) > 1 else None

    if src.is_dir():
        json_files = sorted(src.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {src}", file=sys.stderr)
            sys.exit(1)
        out_dir = dst or src
        for jf in json_files:
            _process_file(jf, out_dir / jf.with_suffix(".dot").name)
    elif src.is_file():
        out_file = dst if dst and dst.suffix == ".dot" else (
            dst / src.with_suffix(".dot").name if dst and dst.is_dir()
            else None  # stdout if no dst
        )
        _process_file(src, out_file)
    else:
        print(f"Path not found: {src}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
