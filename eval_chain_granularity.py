# eval_chain_granularity.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO

try:
    import minigrid  # noqa: F401
    import minigrid.envs  # noqa: F401
    import minigrid.envs.babyai  # noqa: F401
except Exception:
    pass

try:
    import env.doorkey_ball_env  # noqa: F401
except Exception:
    pass

from env.node_alias import to_new
from env.subgoals import extract_subgoals, parse_target_from_mission
from rl.gc_option_env import GCOptionEnv, Goal


@dataclass
class ModuleSpec:
    name: str
    model_path: str
    goal: str
    horizon: int = 256


@dataclass
class ChainSpec:
    name: str
    env_id: str
    granularity: str
    final_target: str
    total_train_steps: int
    modules: List[ModuleSpec]


def resolve_model_path(path_or_dir: str) -> Path:
    p = Path(path_or_dir)
    if p.is_file():
        return p
    if p.is_dir():
        cands = sorted(p.glob("*_final.zip"))
        if not cands:
            cands = sorted(p.rglob("*_final.zip"))
        if cands:
            return sorted(cands, key=lambda x: (len(x.parts), str(x)))[0]
    raise FileNotFoundError(f"Cannot resolve model path: {path_or_dir}")


def _goal_for_node(env: GCOptionEnv, node: str) -> Goal:
    node = to_new(node)
    target_obj = "none"
    target_color = "grey"

    if node in ("near_target", "has_target"):
        obj, color = parse_target_from_mission(env.env)
        if obj is not None:
            target_obj = obj
        if color is not None:
            target_color = color

    return Goal(node=node, target_obj=target_obj, target_color=target_color)


def _encoded_obs_for_current_state(env: GCOptionEnv):
    raw_obs = env.env.unwrapped.gen_obs() if hasattr(env.env.unwrapped, "gen_obs") else env.env.unwrapped.observation(env.env.unwrapped.grid)
    return env._encode_obs(raw_obs)


def run_one_module(env: GCOptionEnv, model: PPO, module: ModuleSpec, deterministic: bool) -> Dict:
    goal_node = to_new(module.goal)
    env._goal = _goal_for_node(env, goal_node)
    env._t = 0

    sg0 = extract_subgoals(env.env)
    env._prev_goal_val = int(sg0.get(goal_node, 0))

    if int(sg0.get(goal_node, 0)) == 1:
        return {
            "module_name": module.name,
            "goal": goal_node,
            "success": 1.0,
            "steps": 0,
            "already_satisfied": 1.0,
        }

    obs = _encoded_obs_for_current_state(env)
    final_info = {}
    success = 0.0
    steps = 0

    for _ in range(int(module.horizon)):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        final_info = info

        sg = info.get("subgoals", {})
        if int(sg.get(goal_node, 0)) == 1:
            success = 1.0
            break

        if terminated or truncated:
            break

    return {
        "module_name": module.name,
        "goal": goal_node,
        "success": success,
        "steps": steps,
        "already_satisfied": 0.0,
    }


def evaluate_chain(spec: ChainSpec, episodes: int, seed: int, deterministic: bool) -> Dict:
    model_cache: Dict[str, PPO] = {}
    for m in spec.modules:
        mp = str(resolve_model_path(m.model_path))
        if mp not in model_cache:
            model_cache[mp] = PPO.load(mp)
        m.model_path = mp

    all_goals = [to_new(m.goal) for m in spec.modules]
    if not all_goals:
        raise ValueError("Chain has no modules")

    env = GCOptionEnv(
        env_id=spec.env_id,
        goal_pool=all_goals,
        option_horizon=max(m.horizon for m in spec.modules),
        seed=seed,
        use_nodrop=False,
        use_precondition_do=False,
        parent_goal=None,
        child_goal=None,
        strict_parent_mode=False,
        strict_parent_fail_fast=False,
        reset_max_tries=128,
    )

    final_target = to_new(spec.final_target)

    final_successes = []
    chain_completed = []
    total_steps_list = []
    module_successes: Dict[str, List[float]] = {m.name: [] for m in spec.modules}
    module_steps: Dict[str, List[float]] = {m.name: [] for m in spec.modules}

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total_steps = 0
        completed = 1.0

        for m in spec.modules:
            model = model_cache[str(resolve_model_path(m.model_path))]
            r = run_one_module(env, model, m, deterministic=deterministic)
            module_successes[m.name].append(float(r["success"]))
            module_steps[m.name].append(float(r["steps"]))
            total_steps += int(r["steps"])

            if float(r["success"]) < 1.0:
                completed = 0.0
                break

        sg = extract_subgoals(env.env)
        final_success = float(int(sg.get(final_target, 0)) == 1)

        final_successes.append(final_success)
        chain_completed.append(completed)
        total_steps_list.append(float(total_steps))

    env.close()

    out = {
        "chain_name": spec.name,
        "env_id": spec.env_id,
        "granularity": spec.granularity,
        "final_target": final_target,
        "episodes": episodes,
        "seed": seed,
        "deterministic": deterministic,
        "total_train_steps": spec.total_train_steps,
        "final_success_rate": float(np.mean(final_successes)),
        "chain_completion_rate": float(np.mean(chain_completed)),
        "avg_total_steps": float(np.mean(total_steps_list)),
        "modules": " -> ".join([m.name for m in spec.modules]),
    }

    for m in spec.modules:
        out[f"module_success/{m.name}"] = float(np.mean(module_successes[m.name]))
        out[f"module_steps/{m.name}"] = float(np.mean(module_steps[m.name]))

    return out


def p(run_name: str) -> str:
    return str(Path("runs_granularity") / run_name)


PRESETS: Dict[str, ChainSpec] = {
    # Pickup chain, 200k fair budget.
    "pickup_fetch_g0": ChainSpec(
        name="pickup_fetch_g0",
        env_id="MiniGrid-Fetch-8x8-N3-v0",
        granularity="G0",
        final_target="task_success",
        total_train_steps=200_000,
        modules=[
            ModuleSpec("has_target", p("G0_pickup_fetch_has_target_s0_fair200k"), "has_target"),
        ],
    ),
    "pickup_fetch_g1": ChainSpec(
        name="pickup_fetch_g1",
        env_id="MiniGrid-Fetch-8x8-N3-v0",
        granularity="G1",
        final_target="task_success",
        total_train_steps=200_000,
        modules=[
            ModuleSpec("near_target", p("G1_pickup_fetch_near_target_s0"), "near_target"),
            ModuleSpec("near_target_to_has_target", p("G1_pickup_fetch_near_target_to_has_target_s0"), "has_target"),
        ],
    ),
    "pickup_babyai_g0": ChainSpec(
        name="pickup_babyai_g0",
        env_id="BabyAI-Pickup-v0",
        granularity="G0",
        final_target="task_success",
        total_train_steps=200_000,
        modules=[
            ModuleSpec("has_target", p("G0_pickup_babyai_has_target_s0_fair200k"), "has_target"),
        ],
    ),
    "pickup_babyai_g1": ChainSpec(
        name="pickup_babyai_g1",
        env_id="BabyAI-Pickup-v0",
        granularity="G1",
        final_target="task_success",
        total_train_steps=200_000,
        modules=[
            ModuleSpec("near_target", p("G1_pickup_babyai_near_target_s0"), "near_target"),
            ModuleSpec("near_target_to_has_target", p("G1_pickup_babyai_near_target_to_has_target_s0"), "has_target"),
        ],
    ),

    # DoorKey prefixes and full chain.
    "doorkey_has_key_g0": ChainSpec(
        name="doorkey_has_key_g0",
        env_id="MiniGrid-DoorKey-6x6-v0",
        granularity="G0",
        final_target="has_key",
        total_train_steps=200_000,
        modules=[
            ModuleSpec("has_key", p("G0_door_doorkey_has_key_s0_fair200k"), "has_key"),
        ],
    ),
    "doorkey_has_key_g1": ChainSpec(
        name="doorkey_has_key_g1",
        env_id="MiniGrid-DoorKey-6x6-v0",
        granularity="G1",
        final_target="has_key",
        total_train_steps=200_000,
        modules=[
            ModuleSpec("near_key", p("G1_door_doorkey_near_key_s0"), "near_key"),
            ModuleSpec("near_key_to_has_key", p("G1_door_doorkey_near_key_to_has_key_s0"), "has_key"),
        ],
    ),
    "doorkey_opened_door_g0": ChainSpec(
        name="doorkey_opened_door_g0",
        env_id="MiniGrid-DoorKey-6x6-v0",
        granularity="G0",
        final_target="opened_door",
        total_train_steps=350_000,
        modules=[
            ModuleSpec("has_key", p("G0_door_doorkey_has_key_s0_fair200k"), "has_key"),
            ModuleSpec("opened_door", p("G0_door_doorkey_opened_door_s0"), "opened_door"),
        ],
    ),
    "doorkey_opened_door_g1": ChainSpec(
        name="doorkey_opened_door_g1",
        env_id="MiniGrid-DoorKey-6x6-v0",
        granularity="G1",
        final_target="opened_door",
        total_train_steps=350_000,
        modules=[
            ModuleSpec("near_key", p("G1_door_doorkey_near_key_s0"), "near_key"),
            ModuleSpec("near_key_to_has_key", p("G1_door_doorkey_near_key_to_has_key_s0"), "has_key"),
            ModuleSpec("has_key_to_opened_door", p("G1_door_doorkey_has_key_to_opened_door_s0_fair150k"), "opened_door"),
        ],
    ),
    "doorkey_task_g0": ChainSpec(
        name="doorkey_task_g0",
        env_id="MiniGrid-DoorKey-6x6-v0",
        granularity="G0",
        final_target="task_success",
        total_train_steps=500_000,
        modules=[
            ModuleSpec("has_key", p("G0_door_doorkey_has_key_s0_fair200k"), "has_key"),
            ModuleSpec("opened_door", p("G0_door_doorkey_opened_door_s0"), "opened_door"),
            ModuleSpec("at_goal", p("G0_door_doorkey_at_goal_s0"), "at_goal"),
        ],
    ),
    "doorkey_task_g1": ChainSpec(
        name="doorkey_task_g1",
        env_id="MiniGrid-DoorKey-6x6-v0",
        granularity="G1",
        final_target="task_success",
        total_train_steps=500_000,
        modules=[
            ModuleSpec("near_key", p("G1_door_doorkey_near_key_s0"), "near_key"),
            ModuleSpec("near_key_to_has_key", p("G1_door_doorkey_near_key_to_has_key_s0"), "has_key"),
            ModuleSpec("has_key_to_opened_door", p("G1_door_doorkey_has_key_to_opened_door_s0_fair150k"), "opened_door"),
            ModuleSpec("opened_door_to_at_goal", p("G1_door_doorkey_opened_door_to_at_goal_s0_fair150k"), "at_goal"),
        ],
    ),

    # UnlockLocal prefixes.
    "unlocklocal_has_key_g0": ChainSpec(
        name="unlocklocal_has_key_g0",
        env_id="BabyAI-UnlockLocal-v0",
        granularity="G0",
        final_target="has_key",
        total_train_steps=200_000,
        modules=[
            ModuleSpec("has_key", p("G0_door_unlocklocal_has_key_s0_fair200k"), "has_key"),
        ],
    ),
    "unlocklocal_has_key_g1": ChainSpec(
        name="unlocklocal_has_key_g1",
        env_id="BabyAI-UnlockLocal-v0",
        granularity="G1",
        final_target="has_key",
        total_train_steps=200_000,
        modules=[
            ModuleSpec("near_key", p("G1_door_unlocklocal_near_key_s0"), "near_key"),
            ModuleSpec("near_key_to_has_key", p("G1_door_unlocklocal_near_key_to_has_key_s0"), "has_key"),
        ],
    ),
    "unlocklocal_task_g0": ChainSpec(
        name="unlocklocal_task_g0",
        env_id="BabyAI-UnlockLocal-v0",
        granularity="G0",
        final_target="task_success",
        total_train_steps=350_000,
        modules=[
            ModuleSpec("has_key", p("G0_door_unlocklocal_has_key_s0_fair200k"), "has_key"),
            ModuleSpec("opened_door", p("G0_door_unlocklocal_opened_door_s0"), "opened_door"),
        ],
    ),
    "unlocklocal_task_g1": ChainSpec(
        name="unlocklocal_task_g1",
        env_id="BabyAI-UnlockLocal-v0",
        granularity="G1",
        final_target="task_success",
        total_train_steps=350_000,
        modules=[
            ModuleSpec("near_key", p("G1_door_unlocklocal_near_key_s0"), "near_key"),
            ModuleSpec("near_key_to_has_key", p("G1_door_unlocklocal_near_key_to_has_key_s0"), "has_key"),
            ModuleSpec("has_key_to_opened_door", p("G1_door_unlocklocal_has_key_to_opened_door_s0_fair150k"), "opened_door"),
        ],
    ),

    # UnlockPickup prefixes and full chain.
    "unlockpickup_has_key_g0": ChainSpec(
        name="unlockpickup_has_key_g0",
        env_id="MiniGrid-UnlockPickup-v0",
        granularity="G0",
        final_target="has_key",
        total_train_steps=200_000,
        modules=[
            ModuleSpec("has_key", p("G0_door_unlockpickup_has_key_s0_fair200k"), "has_key"),
        ],
    ),
    "unlockpickup_has_key_g1": ChainSpec(
        name="unlockpickup_has_key_g1",
        env_id="MiniGrid-UnlockPickup-v0",
        granularity="G1",
        final_target="has_key",
        total_train_steps=200_000,
        modules=[
            ModuleSpec("near_key", p("G1_door_unlockpickup_near_key_s0"), "near_key"),
            ModuleSpec("near_key_to_has_key", p("G1_door_unlockpickup_near_key_to_has_key_s0"), "has_key"),
        ],
    ),
    "unlockpickup_opened_door_g0": ChainSpec(
        name="unlockpickup_opened_door_g0",
        env_id="MiniGrid-UnlockPickup-v0",
        granularity="G0",
        final_target="opened_door",
        total_train_steps=350_000,
        modules=[
            ModuleSpec("has_key", p("G0_door_unlockpickup_has_key_s0_fair200k"), "has_key"),
            ModuleSpec("opened_door", p("G0_door_unlockpickup_opened_door_s0"), "opened_door"),
        ],
    ),
    "unlockpickup_opened_door_g1": ChainSpec(
        name="unlockpickup_opened_door_g1",
        env_id="MiniGrid-UnlockPickup-v0",
        granularity="G1",
        final_target="opened_door",
        total_train_steps=350_000,
        modules=[
            ModuleSpec("near_key", p("G1_door_unlockpickup_near_key_s0"), "near_key"),
            ModuleSpec("near_key_to_has_key", p("G1_door_unlockpickup_near_key_to_has_key_s0"), "has_key"),
            ModuleSpec("has_key_to_opened_door", p("G1_door_unlockpickup_has_key_to_opened_door_s0_fair150k"), "opened_door"),
        ],
    ),
    "unlockpickup_task_g0": ChainSpec(
        name="unlockpickup_task_g0",
        env_id="MiniGrid-UnlockPickup-v0",
        granularity="G0",
        final_target="task_success",
        total_train_steps=500_000,
        modules=[
            ModuleSpec("has_key", p("G0_door_unlockpickup_has_key_s0_fair200k"), "has_key"),
            ModuleSpec("opened_door", p("G0_door_unlockpickup_opened_door_s0"), "opened_door"),
            ModuleSpec("has_box", p("G0_door_unlockpickup_has_box_s0"), "has_box"),
        ],
    ),
    "unlockpickup_task_g1": ChainSpec(
        name="unlockpickup_task_g1",
        env_id="MiniGrid-UnlockPickup-v0",
        granularity="G1",
        final_target="task_success",
        total_train_steps=500_000,
        modules=[
            ModuleSpec("near_key", p("G1_door_unlockpickup_near_key_s0"), "near_key"),
            ModuleSpec("near_key_to_has_key", p("G1_door_unlockpickup_near_key_to_has_key_s0"), "has_key"),
            ModuleSpec("has_key_to_opened_door", p("G1_door_unlockpickup_has_key_to_opened_door_s0_fair150k"), "opened_door"),
            ModuleSpec("opened_door_to_has_box", p("G1_door_unlockpickup_opened_door_to_has_box_s0_fair150k"), "has_box"),
        ],
    ),
}


def write_rows(rows: List[Dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_fields = []
    for r in rows:
        for k in r.keys():
            if k not in all_fields:
                all_fields.append(k)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, default="all", help="Preset name, comma-separated preset names, or all")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out_csv", type=str, default="results_granularity/chain_eval.csv")
    args = ap.parse_args()

    if args.preset == "all":
        names = list(PRESETS.keys())
    else:
        names = [x.strip() for x in args.preset.split(",") if x.strip()]

    rows = []
    for name in names:
        if name not in PRESETS:
            raise KeyError(f"Unknown preset {name}. Available: {sorted(PRESETS.keys())}")
        print(f"\n[EVAL CHAIN] {name}")
        row = evaluate_chain(PRESETS[name], episodes=args.episodes, seed=args.seed, deterministic=args.deterministic)
        rows.append(row)
        print(row)

    out_csv = Path(args.out_csv)
    write_rows(rows, out_csv)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
