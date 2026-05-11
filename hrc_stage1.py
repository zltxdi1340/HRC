# hrc_stage1.py (granularity-ablation version: canonical agent-centric node names)
import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import itertools
import gymnasium as gym
import minigrid  # noqa: F401  # register MiniGrid envs
import env.doorkey_ball_env  # noqa: F401  # register custom env if present
import matplotlib.pyplot as plt

from interventions.sampling import intervention_sampling
from causal_discovery_simple import causal_discovery

FINAL = "task_success"

# Canonical node names used in this granularity branch.
OLD_TO_NEW = {
    "near_obj": "near_target",
    "has_obj": "has_target",
    "open_box": "opened_box",
    "door_open": "opened_door",
    "success": "task_success",
}


def _new(name: str) -> str:
    return OLD_TO_NEW.get(name, name)


def _new_set(xs) -> Set[str]:
    return {_new(x) for x in xs}


def make_env_quiet(env_id: str):
    if "BabyAI" not in env_id:
        return gym.make(env_id)
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return gym.make(env_id)


def active_subgoals(env) -> List[str]:
    """Return the Stage-1 active variables using canonical node names."""
    env_id = env.spec.id if env.spec else "unknown"

    if env_id == "MiniGrid-Empty-8x8-v0":
        return ["at_goal", FINAL]
    if env_id == "MiniGrid-FourRooms-v0":
        return ["at_goal", FINAL]
    if "GoToObject" in env_id:
        return ["near_target", FINAL]
    if "Fetch" in env_id:
        return ["near_target", "has_target", FINAL]
    if env_id == "BabyAI-Pickup-v0":
        return ["near_target", "has_target", FINAL]
    if env_id == "BabyAI-UnlockLocal-v0":
        return ["near_key", "has_key", "opened_door", FINAL]
    if env_id == "MiniGrid-DoorKey-6x6-v0":
        return ["near_key", "has_key", "opened_door", "at_goal", FINAL]
    if env_id == "MiniGrid-UnlockPickup-v0":
        return ["near_key", "has_key", "opened_door", "has_box", FINAL]
    if env_id == "BabyAI-KeyInBox-v0":
        return ["opened_box", "has_key", "opened_door", FINAL]
    if "DoorKeyBall" in env_id:
        return ["near_key", "has_key", "opened_door", "at_goal", "has_ball", FINAL]

    return ["at_goal", FINAL]


def detect_roots_manual(env, ACTIVE: List[str]) -> Set[str]:
    """
    Manual roots used to bootstrap iterative Stage-1 discovery.

    For granularity comparison we want the discovered chain to expose near_key -> has_key,
    so door-chain tasks start from near_key when it is part of ACTIVE.
    """
    env_id = env.spec.id if env.spec else "unknown"
    active = set(ACTIVE)

    if env_id in ("MiniGrid-Empty-8x8-v0", "MiniGrid-FourRooms-v0"):
        return {"at_goal"}
    if "GoToObject" in env_id:
        return {"near_target"}
    if "Fetch" in env_id or env_id == "BabyAI-Pickup-v0":
        return {"near_target"}
    if env_id in ("BabyAI-UnlockLocal-v0", "MiniGrid-DoorKey-6x6-v0", "MiniGrid-UnlockPickup-v0"):
        return {"near_key"} if "near_key" in active else {"has_key"}
    if env_id == "BabyAI-KeyInBox-v0":
        return {"opened_box"}
    if "DoorKeyBall" in env_id:
        return {"near_key"} if "near_key" in active else {"has_key"}
    return set()


def detect_roots_by_baseline(
    env,
    ACTIVE,
    *,
    T: int,
    H: int,
    delta: int,
    seed: int,
    rho: float = 0.4,
    exclude=None,
):
    if exclude is None:
        exclude = set()
    exclude = _new_set(exclude)
    exclude.add(FINAL)

    di0 = intervention_sampling(env, IS=set(), T=T, H=H, delta=delta, seed=seed, out_jsonl_path=None)
    rows = [r for r in di0 if r.get("intervened") == "none"]
    if not rows:
        return set()

    cand = [g for g in ACTIVE if g not in exclude]
    if not cand:
        return set()

    p_none = {g: sum(int(r["vars_max_window"].get(g, 0)) for r in rows) / float(len(rows)) for g in cand}
    roots = {g for g in cand if p_none.get(g, 0.0) >= rho}
    if roots:
        return roots

    best = max(cand, key=lambda x: p_none.get(x, 0.0))
    return {best} if p_none.get(best, 0.0) > 0.0 else set()


def _powerset_nonempty(xs):
    xs = sorted(list(xs))
    for r in range(1, len(xs) + 1):
        for comb in itertools.combinations(xs, r):
            yield set(comb)


def recheck_root_parents(DI, IS, roots, *, tau, min_support):
    out = {}
    for child in sorted(list(roots)):
        cand = set(IS) - {child}
        if not cand:
            continue

        rows = [r for r in DI if _new(r.get("intervened", "")) != child]
        if not rows:
            continue

        conds = []
        ys = []
        for r in rows:
            va = r.get("vars_after_int", {})
            cond = {g for g in cand if int(va.get(g, 0)) == 1}
            conds.append(cond)
            ys.append(int(r["vars_max_window"].get(child, 0)))

        best_P = None
        best_size = 10**9
        best_score = -1.0

        for P in _powerset_nonempty(cand):
            y1 = [ys[i] for i, c in enumerate(conds) if P.issubset(c)]
            y0 = [ys[i] for i, c in enumerate(conds) if not P.issubset(c)]
            if len(y1) < min_support or len(y0) < min_support:
                continue
            p1 = sum(y1) / float(len(y1))
            p0 = sum(y0) / float(len(y0))
            score = p1 - p0
            if score >= tau:
                if (len(P) < best_size) or (len(P) == best_size and score > best_score):
                    best_P = set(P)
                    best_size = len(P)
                    best_score = score

        if best_P is not None:
            out[child] = best_P
    return out


def _p(di: List[Dict], intervened: str, y: str, min_support: int) -> Optional[float]:
    intervened = _new(intervened)
    y = _new(y)
    rows = [r for r in di if _new(r.get("intervened", "")) == intervened]
    if len(rows) < min_support:
        return None
    return sum(int(r["vars_max_window"].get(y, 0)) for r in rows) / float(len(rows))


def _effect(di: List[Dict], gi: str, y: str, min_support: int) -> Optional[float]:
    p1 = _p(di, gi, y, min_support)
    p0 = _p(di, "none", y, min_support)
    if p1 is None or p0 is None:
        return None
    return p1 - p0


def _normalize_parent_dict(parents_hat: Dict[str, Set[str]], active: Set[str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {g: set() for g in active}
    for child, pa in parents_hat.items():
        child2 = _new(child)
        if child2 not in active:
            continue
        out.setdefault(child2, set()).update({_new(p) for p in pa if _new(p) in active and _new(p) != child2})
    return out


def _transitive_reachable(edges: Set[Tuple[str, str]], src: str, dst: str, banned: Tuple[str, str]) -> bool:
    children: Dict[str, Set[str]] = {}
    for u, v in edges:
        if (u, v) == banned:
            continue
        children.setdefault(u, set()).add(v)

    stack = list(children.get(src, set()))
    seen = set()
    while stack:
        x = stack.pop()
        if x == dst:
            return True
        if x in seen:
            continue
        seen.add(x)
        stack.extend(list(children.get(x, set())))
    return False


def build_edges(parents_hat: Dict[str, Set[str]]) -> Set[Tuple[str, str]]:
    edges = set()
    for child, pa in parents_hat.items():
        child2 = _new(child)
        for p in pa:
            p2 = _new(p)
            if p2 != child2:
                edges.add((p2, child2))
    return edges


def prune_transitive_edges(parents_hat: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    out = {k: set(v) for k, v in parents_hat.items()}
    edges = build_edges(out)
    to_remove = []
    for u, v in sorted(edges):
        if _transitive_reachable(edges, u, v, banned=(u, v)):
            to_remove.append((u, v))
    for u, v in to_remove:
        out.setdefault(v, set()).discard(u)
    return out


def _task_direct_priors(env_id: str, ACTIVE: List[str]) -> Dict[str, Set[str]]:
    active = set(ACTIVE)

    def keep(edges: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for p, c in edges:
            p2, c2 = _new(p), _new(c)
            if p2 in active and c2 in active:
                out.setdefault(c2, set()).add(p2)
        return out

    if env_id == "MiniGrid-Empty-8x8-v0":
        return keep([("at_goal", FINAL)])
    if env_id == "MiniGrid-FourRooms-v0":
        return keep([("at_goal", FINAL)])
    if "GoToObject" in env_id:
        return keep([("near_target", FINAL)])
    if ("Fetch" in env_id) or (env_id == "BabyAI-Pickup-v0"):
        return keep([("near_target", "has_target"), ("has_target", FINAL)])
    if env_id == "BabyAI-UnlockLocal-v0":
        return keep([("near_key", "has_key"), ("has_key", "opened_door"), ("opened_door", FINAL)])
    if env_id == "MiniGrid-DoorKey-6x6-v0":
        return keep([("near_key", "has_key"), ("has_key", "opened_door"), ("opened_door", "at_goal"), ("at_goal", FINAL)])
    if env_id == "MiniGrid-UnlockPickup-v0":
        return keep([("near_key", "has_key"), ("has_key", "opened_door"), ("opened_door", "has_box"), ("has_box", FINAL)])
    if env_id == "BabyAI-KeyInBox-v0":
        return keep([("opened_box", "has_key"), ("has_key", "opened_door"), ("opened_door", FINAL)])
    if "DoorKeyBall" in env_id:
        return keep([("near_key", "has_key"), ("has_key", "opened_door"), ("opened_door", "at_goal"), ("at_goal", FINAL), ("has_ball", FINAL)])
    return {}


def refine_direct_edges(
    di: List[Dict],
    env_id: str,
    IS: Set[str],
    ACTIVE: List[str],
    parents_hat: Dict[str, Set[str]],
    tau: float,
    min_support: int,
) -> Dict[str, Set[str]]:
    """
    Refine data-driven parent estimates and enforce task-typed direct priors.

    This branch uses canonical node names only. The priors are intentional: Stage-1
    should output the granularity graph we will use for G0/G1/G2 experiments, while
    data-driven discovery remains useful for sanity checks and for nodes not covered
    by the hard-coded task schema.
    """
    active = set(ACTIVE)
    IS = _new_set(IS)
    out = _normalize_parent_dict(parents_hat, active)

    relaxed_tau = max(0.02, 0.6 * float(tau))
    relaxed_support = max(12, int(min_support) // 2)

    is_keyinbox = env_id == "BabyAI-KeyInBox-v0"
    is_doorkey = env_id == "MiniGrid-DoorKey-6x6-v0"
    is_unlockpickup = env_id == "MiniGrid-UnlockPickup-v0"
    is_unlocklocal = env_id == "BabyAI-UnlockLocal-v0"
    is_door_task = is_doorkey or is_unlockpickup or is_unlocklocal or is_keyinbox
    is_fetch_like = ("Fetch" in env_id) or (env_id == "BabyAI-Pickup-v0")

    # Data-driven support, when available.
    if ("near_key" in active) and ("has_key" in active) and ("near_key" in IS):
        eff = _effect(di, "near_key", "has_key", relaxed_support)
        if eff is not None and eff >= relaxed_tau:
            out["has_key"] = {"near_key"}

    if is_keyinbox and ("opened_box" in active) and ("has_key" in active) and ("opened_box" in IS):
        eff = _effect(di, "opened_box", "has_key", relaxed_support)
        if eff is not None and eff >= relaxed_tau:
            out["has_key"] = {"opened_box"}

    if is_door_task and ("opened_door" in active) and ("has_key" in active) and ("has_key" in IS):
        eff = _effect(di, "has_key", "opened_door", relaxed_support)
        if eff is not None and eff >= relaxed_tau:
            out["opened_door"] = {"has_key"}

    if is_doorkey and ("at_goal" in active) and ("opened_door" in IS):
        eff = _effect(di, "opened_door", "at_goal", relaxed_support)
        if eff is not None and eff >= relaxed_tau:
            out["at_goal"] = {"opened_door"}

    if is_unlockpickup and ("has_box" in active) and ("opened_door" in IS):
        eff = _effect(di, "opened_door", "has_box", relaxed_support)
        if eff is not None and eff >= relaxed_tau:
            out["has_box"] = {"opened_door"}

    if is_fetch_like and ("near_target" in active) and ("has_target" in active) and ("near_target" in IS):
        eff = _effect(di, "near_target", "has_target", relaxed_support)
        if eff is not None and eff >= relaxed_tau:
            out["has_target"] = {"near_target"}

    # Enforce direct task priors to keep the final DAG connected and canonical.
    priors = _task_direct_priors(env_id, ACTIVE)
    for child, pa in priors.items():
        out[child] = set(pa)

    for g in ACTIVE:
        out.setdefault(g, set())

    return prune_transitive_edges(out)


def compute_levels(nodes: List[str], parents_hat: Dict[str, Set[str]]) -> Dict[str, int]:
    level = {n: 0 for n in nodes}
    for _ in range(len(nodes)):
        changed = False
        for n in nodes:
            pa = parents_hat.get(n, set())
            if not pa:
                continue
            lv = max(level.get(p, 0) for p in pa) + 1
            if lv > level[n]:
                level[n] = lv
                changed = True
        if not changed:
            break
    return level


def _would_create_cycle(parents_hat, new_parent, child):
    children = {n: set() for n in parents_hat}
    for ch, pas in parents_hat.items():
        for p in pas:
            children.setdefault(p, set()).add(ch)

    stack = [child]
    seen = set()
    while stack:
        x = stack.pop()
        if x == new_parent:
            return True
        if x in seen:
            continue
        seen.add(x)
        stack.extend(list(children.get(x, set())))
    return False


def _summary_to_graph_png_path(summary_path: Path) -> Path:
    stem = summary_path.stem
    if stem.endswith("_summary"):
        graph_stem = stem[:-8] + "_graph"
    else:
        graph_stem = stem + "_graph"
    return summary_path.with_name(graph_stem + ".png")


def _order_within_level(nodes: List[str]) -> List[str]:
    ns = sorted(nodes)
    if FINAL in ns:
        ns = [x for x in ns if x != FINAL] + [FINAL]
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
            x = float(lv) * 2.6
            y = (i - (k - 1) / 2.0) * 1.5
            pos[n] = (x, y)
    return pos


def _set_axes_padding(ax, pos: Dict[str, Tuple[float, float]], xpad: float = 1.0, ypad: float = 0.9):
    if not pos:
        return
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)


def _draw_arrow(ax, x1, y1, x2, y2, color="black", lw=2.0, alpha=1.0, linestyle="-"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=lw, color=color, alpha=alpha, linestyle=linestyle),
    )


def draw_current_graph(
    env_id: str,
    mission: Optional[str],
    active: List[str],
    edges: Set[Tuple[str, str]],
    levels: Dict[str, int],
    out_png: Path,
):
    pos = _layout_by_levels(active, levels)
    fig, ax = plt.subplots(
        figsize=(max(7.0, 2.5 * max(levels.values(), default=0) + 5.8), max(4.5, 1.6 * len(active)))
    )

    for u, v in sorted(edges):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        _draw_arrow(ax, x1, y1, x2, y2, color="tab:blue", lw=2.2, alpha=0.95)

    for n in active:
        x, y = pos[n]
        is_sink = n == FINAL
        face = "tab:green" if is_sink else "tab:blue"
        ax.scatter([x], [y], s=1200, color=face, alpha=0.92, zorder=3)
        ax.text(
            x,
            y,
            n,
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            zorder=4,
            fontweight="semibold",
        )

    _set_axes_padding(ax, pos)
    title = env_id
    if mission:
        title += "\n" + str(mission)
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()
    plt.tight_layout(pad=1.1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, required=True)
    ap.add_argument("--T", type=int, default=150)
    ap.add_argument("--H", type=int, default=450)
    ap.add_argument("--delta", type=int, default=120)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--min_support", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roots_mode", type=str, default="manual", choices=["manual", "baseline"])
    ap.add_argument("--roots_T", type=int, default=60)
    ap.add_argument("--roots_rho", type=float, default=0.4)
    ap.add_argument("--recheck_roots", action="store_true")
    ap.add_argument("--out_summary", type=str, default="", help="Write final summary.json to this path")
    args = ap.parse_args()

    env = make_env_quiet(args.env_id)
    env.reset(seed=args.seed)

    ACTIVE = active_subgoals(env)
    if args.roots_mode == "baseline":
        CS = set(
            detect_roots_by_baseline(
                env,
                ACTIVE,
                T=args.roots_T,
                H=args.H,
                delta=args.delta,
                seed=args.seed + 999,
                rho=args.roots_rho,
            )
        )
    else:
        CS = set(detect_roots_manual(env, ACTIVE))
    IS: Set[str] = set()

    parents_hat_global: Dict[str, Set[str]] = {g: set() for g in ACTIVE}
    H_edges: Set[Tuple[str, str]] = set()
    levels = {g: 0 for g in ACTIVE}

    print("env_id:", args.env_id)
    print("mission:", getattr(env.unwrapped, "mission", None))
    print("ACTIVE:", ACTIVE)
    print("init CS0:", CS)
    DI_last = None

    t = 1
    while (FINAL not in IS) and (len(CS) > 0):
        g_sel = sorted(list(CS))[0]
        CS.remove(g_sel)
        IS.add(g_sel)

        print(f"\n=== Iter {t} ===")
        print("IS:", sorted(IS), "CS:", sorted(CS))

        DI = intervention_sampling(
            env,
            IS,
            T=args.T,
            H=args.H,
            delta=args.delta,
            seed=args.seed + 1000 * t,
            out_jsonl_path=None,
        )
        DI_last = DI

        discovered = causal_discovery(DI, IS, ACTIVE, tau=args.tau, min_support=args.min_support)
        discovered = _normalize_parent_dict(discovered, set(ACTIVE))
        base = {g: set(parents_hat_global.get(g, set())) for g in ACTIVE}

        for child, pa in discovered.items():
            if child in IS:
                continue
            if pa:
                base[child] = set(pa)

        refined = refine_direct_edges(DI, args.env_id, IS, ACTIVE, base, tau=args.tau, min_support=args.min_support)
        for g in ACTIVE:
            refined.setdefault(g, set())

        parents_hat_global = refined
        H_edges = build_edges(parents_hat_global)
        levels = compute_levels(ACTIVE, parents_hat_global)

        CCS = set()
        for child, pa in refined.items():
            if child in IS or child in CS:
                continue
            if len(pa) == 0:
                continue
            if pa.issubset(IS):
                CCS.add(child)

        print("parents_hat(refined):", {k: sorted(list(v)) for k, v in parents_hat_global.items() if v})
        print("CCS:", sorted(CCS))
        print("H_edges:", sorted(list(H_edges)))
        print("levels:", {k: levels[k] for k in ACTIVE})

        CS |= CCS
        t += 1

    env.close()
    if DI_last is not None:
        print("[DEBUG] p_none(has_box) =", _p(DI_last, "none", "has_box", 1))
        print("[DEBUG] p_do_opened_door(has_box) =", _p(DI_last, "opened_door", "has_box", 1))
        print("[DEBUG] p_none(at_goal) =", _p(DI_last, "none", "at_goal", 1))
        print("[DEBUG] p_do_opened_door(at_goal) =", _p(DI_last, "opened_door", "at_goal", 1))

    if args.recheck_roots and DI_last is not None:
        roots_final = {g for g in ACTIVE if g != FINAL and len(parents_hat_global.get(g, set())) == 0}
        add = recheck_root_parents(DI_last, IS, roots_final, tau=args.tau, min_support=args.min_support)
        if add:
            for child, pa in add.items():
                pa2 = {p for p in pa if not _would_create_cycle(parents_hat_global, p, child)}
                if pa2:
                    parents_hat_global.setdefault(child, set()).update(pa2)
            parents_hat_global = prune_transitive_edges(parents_hat_global)
            H_edges = build_edges(parents_hat_global)
            levels = compute_levels(ACTIVE, parents_hat_global)
            print("\n[RECHECK] root parents added:", {k: sorted(list(v)) for k, v in add.items()})
            print("[RECHECK] H_edges:", sorted(list(H_edges)))
            print("[RECHECK] levels:", {k: levels[k] for k in ACTIVE})
    print("\nDONE. Final IS:", sorted(IS), "CS:", sorted(CS))

    if args.out_summary:
        parents_out = {k: sorted(list(v)) for k, v in parents_hat_global.items() if len(v) > 0}
        payload = {
            "env_id": args.env_id,
            "mission": getattr(env.unwrapped, "mission", None),
            "ACTIVE": ACTIVE,
            "parents_hat": parents_out,
            "H_edges": sorted(list(H_edges)),
            "levels": levels,
        }
        summary_path = Path(args.out_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        graph_png = _summary_to_graph_png_path(summary_path)
        draw_current_graph(
            env_id=args.env_id,
            mission=payload["mission"],
            active=ACTIVE,
            edges=H_edges,
            levels=levels,
            out_png=graph_png,
        )
        print(f"[WRITE] summary -> {summary_path}")
        print(f"[WRITE] graph   -> {graph_png}")


if __name__ == "__main__":
    main()
