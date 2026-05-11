# viz/make_multitask_figs_v4.py
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import matplotlib.pyplot as plt

# --- make project imports work when running from repo root ---
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401

from env.wrappers import NoDropWrapper
import hrc_stage1 as hrc
from interventions.sampling import intervention_sampling


@dataclass
class TaskResult:
    env_id: str
    mission: Optional[str]
    active: List[str]
    parents_hat: Dict[str, List[str]]     # child -> sorted parents
    edges: List[Tuple[str, str]]          # list of (parent, child)
    levels: Dict[str, int]


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _order_within_level(nodes: List[str]) -> List[str]:
    """Heuristic ordering to reduce crossings and keep Success visually at the bottom."""
    ns = sorted(nodes)
    if "success" in ns:
        ns = [x for x in ns if x != "success"] + ["success"]
    return ns


def _layout_by_levels(nodes: List[str], levels: Dict[str, int]) -> Dict[str, Tuple[float, float]]:
    by: Dict[int, List[str]] = {}
    for n in nodes:
        by.setdefault(int(levels.get(n, 0)), []).append(n)

    pos: Dict[str, Tuple[float, float]] = {}
    for lv, ns in sorted(by.items()):
        ns = _order_within_level(ns)
        k = len(ns)
        for i, n in enumerate(ns):
            y = (i - (k - 1) / 2.0) * 1.35
            x = float(lv) * 2.2
            pos[n] = (x, y)
    return pos


def _arrow(ax, x1, y1, x2, y2, rad: float = 0.0, lw: float = 1.8):
    props = dict(arrowstyle="->", lw=lw)
    if abs(rad) > 1e-6:
        props["connectionstyle"] = f"arc3,rad={rad}"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=props)


def _draw_graph(title: str, nodes: List[str], edges: List[Tuple[str, str]], levels: Dict[str, int], out_png: Path):
    pos = _layout_by_levels(nodes, levels)
    plt.figure(figsize=(8.8, 4.2))
    ax = plt.gca()
    ax.set_title(title, fontsize=11)

    # edges first (behind nodes)
    for (u, v) in edges:
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        _arrow(ax, x1, y1, x2, y2, rad=0.0, lw=1.8)

    # nodes
    for n in nodes:
        x, y = pos[n]
        ax.scatter([x], [y], s=980, zorder=3)
        ax.text(
            x, y, n, ha="center", va="center", fontsize=10, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.25)
        )

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def _compute_group_means(DI: List[Dict], keys: List[str]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict]] = {}
    for r in DI:
        groups.setdefault(r["intervened"], []).append(r)

    out: Dict[str, Dict[str, float]] = {}
    for g, rows in groups.items():
        d: Dict[str, float] = {}
        for k in keys:
            d[k] = sum(int(rr["vars_max_window"].get(k, 0)) for rr in rows) / float(len(rows))
        d["_n"] = float(len(rows))
        out[g] = d
    return out


def _draw_bars(title: str, keys: List[str], means: Dict[str, Dict[str, float]], out_png: Path):
    groups = ["none"] + [g for g in sorted(means.keys()) if g != "none"]
    groups = [g for g in groups if g in means]

    m = len(keys)
    x = range(len(groups))
    width = 0.85 / max(1, m)

    plt.figure(figsize=(9.4, 3.8))
    ax = plt.gca()
    ax.set_title(title, fontsize=11)

    for j, k in enumerate(keys):
        ys = [means[g][k] for g in groups]
        xs = [i + (j - (m - 1) / 2.0) * width for i in x]
        ax.bar(xs, ys, width=width, label=k)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{g}\n(n={int(means[g]['_n'])})" for g in groups], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, ncol=min(3, m))
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def _edge_to_str(e: Tuple[str, str]) -> str:
    return f"{e[0]}->{e[1]}"


def _draw_edge_presence_matrix(tasks: List[TaskResult], out_png: Path, out_json: Path):
    all_edges: List[str] = sorted({_edge_to_str(e) for t in tasks for e in t.edges})
    task_names: List[str] = [t.env_id for t in tasks]

    mat: List[List[int]] = []
    for t in tasks:
        s = {_edge_to_str(e) for e in t.edges}
        mat.append([1 if e in s else 0 for e in all_edges])

    edge_support = {e: sum(mat[i][j] for i in range(len(tasks))) for j, e in enumerate(all_edges)}
    shared = sorted([e for e, c in edge_support.items() if c >= 2])
    unique = {task_names[i]: [all_edges[j] for j in range(len(all_edges)) if mat[i][j] == 1 and edge_support[all_edges[j]] == 1]
              for i in range(len(tasks))}

    payload = {
        "tasks": task_names,
        "edges": all_edges,
        "matrix": mat,
        "edge_support": edge_support,
        "shared_edges": shared,
        "unique_edges": unique,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    import numpy as np
    A = np.array(mat, dtype=float)

    plt.figure(figsize=(max(9, 0.35 * len(all_edges) + 3), max(3.7, 0.6 * len(tasks) + 1.8)))
    ax = plt.gca()
    ax.set_title("Cross-task causal slice — edge presence (1=present)", fontsize=11)
    ax.imshow(A, aspect="auto")

    ax.set_yticks(list(range(len(tasks))))
    ax.set_yticklabels(task_names, fontsize=9)
    ax.set_xticks(list(range(len(all_edges))))
    ax.set_xticklabels(all_edges, fontsize=8, rotation=45, ha="right")

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, str(int(A[i, j])), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def _module_of(node: str) -> str:
    if node in {"has_key", "has_ball", "has_box"}:
        return "Inventory"
    if node == "door_open":
        return "Door"
    if node == "at_goal":
        return "Navigation"
    if node == "success":
        return "Success"
    return "Other"


def _draw_module_graph(tasks: List[TaskResult], out_png: Path, out_json: Path):
    support: Dict[Tuple[str, str], int] = {}
    for t in tasks:
        seen: Set[Tuple[str, str]] = set()
        for (u, v) in t.edges:
            mu, mv = _module_of(u), _module_of(v)
            if mu == mv:
                continue
            seen.add((mu, mv))
        for e in seen:
            support[e] = support.get(e, 0) + 1

    modules = ["Inventory", "Door", "Navigation", "Success"]

    # Diamond layout to avoid collinear overlap:
    # Inventory (left), Door (top), Navigation (bottom), Success (right)
    pos = {
        "Inventory": (0.0, 0.0),
        "Door": (2.4, 1.2),
        "Navigation": (2.4, -1.2),
        "Success": (4.8, 0.0),
    }

    out_json.write_text(json.dumps({
        "modules": modules,
        "module_edges_weighted": {f"{a}->{b}": w for (a, b), w in sorted(support.items())},
        "note": "weight = number of tasks where at least one node-edge collapses into this module edge",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.figure(figsize=(8.2, 3.2))
    ax = plt.gca()
    ax.set_title("Module-level causal graph (collapsed across tasks)", fontsize=11)

    # Draw edges; add curvature for some to avoid overlap
    for (a, b), w in sorted(support.items()):
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        # slight curvature for long cross edges
        rad = 0.0
        if (a, b) == ("Inventory", "Success"):
            rad = 0.18
        if (a, b) == ("Door", "Success") or (a, b) == ("Navigation", "Success"):
            rad = -0.10
        _arrow(ax, x1, y1, x2, y2, rad=rad, lw=2.0)
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax.text(
            xm, ym + 0.12, f"w={w}", ha="center", va="bottom", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.18)
        )

    for m in modules:
        x, y = pos[m]
        ax.scatter([x], [y], s=1400, zorder=3)
        ax.text(
            x, y, m, ha="center", va="center", fontsize=10, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.25)
        )

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_ids", nargs="*", default=[
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-DoorKey-6x6-v0",
        "MiniGrid-DoorKeyBall-6x6-v0",
        "BabyAI-Pickup-v0",
        "MiniGrid-UnlockPickup-v0",
    ])
    ap.add_argument("--out_dir", type=str, default="viz_out_mt_v2")
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--H", type=int, default=300)
    ap.add_argument("--delta", type=int, default=60)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--min_support", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bars", action="store_true")
    ap.add_argument("--summary_dir", type=str, default="", help="Read *_summary.json from this directory and only draw")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    results: List[TaskResult] = []

    if not args.summary_dir:
        raise ValueError("make_multitask_figs_v4.py is draw-only; please pass --summary_dir <dir_with_summary_json>")
    summary_dir = Path(args.summary_dir)

    for env_id in args.env_ids:
        safe_id = env_id.replace(":", "_")
        summ_path = summary_dir / f"{safe_id}_summary.json"
        data = json.loads(summ_path.read_text(encoding="utf-8"))

        title = f"{data['env_id']}\n{data.get('mission','')}".strip()
        nodes = data["ACTIVE"]
        edges = [tuple(e) for e in data["H_edges"]]   # 注意：json 里是 [ ["a","b"], ... ]
        levels = {k: int(v) for k, v in data["levels"].items()}
        parents_hat = data.get("parents_hat", {})

        _draw_graph(title, nodes, edges, levels, out_dir / f"{safe_id}_graph.png")

        results.append(TaskResult(
            env_id=env_id,
            mission=data.get("mission", None),
            active=nodes,
            parents_hat=parents_hat,
            edges=edges,
            levels=levels,
        ))

    _draw_edge_presence_matrix(results,
        out_png=out_dir/"cross_task_edge_presence.png",
        out_json=out_dir/"cross_task_edge_presence.json")

    _draw_module_graph(
        results,
        out_png=out_dir / "module_level_graph.png",
        out_json=out_dir / "module_level_graph.json",
    )

    index = {
        "tasks": [{
            "env_id": r.env_id,
            "graph_png": f"{r.env_id.replace(':','_')}_graph.png",
            "bars_png": f"{r.env_id.replace(':','_')}_bars.png" if args.bars else None,
            "summary_json": f"{r.env_id.replace(':','_')}_summary.json",
        } for r in results],
        "cross_task_edge_presence_png": "cross_task_edge_presence.png",
        "cross_task_edge_presence_json": "cross_task_edge_presence.json",
        "module_level_graph_png": "module_level_graph.png",
        "module_level_graph_json": "module_level_graph.json",
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE. Outputs in:", out_dir)


if __name__ == "__main__":
    main()
