# eval_door_transfer.py
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
from env.subgoals import extract_subgoals
from rl.gc_option_env import GCOptionEnv, Goal
from rl.g2_template_env import G2TemplateEnv


@dataclass
class ModuleSpec:
    name: str
    model_path: str
    goal_or_template: str
    external_check: Optional[str] = None
    horizon: int = 256
    parent_goal: Optional[str] = None


@dataclass
class ChainSpec:
    name: str
    method: str
    granularity: str
    env_id: str
    final_target: str
    modules: List[ModuleSpec]
    c_single: int
    c_agg: int
    c_adapt: int
    source_label: str = ""


def resolve_model_path(path_or_dir: str) -> Path:
    p = Path(path_or_dir)
    if p.is_file():
        return p
    if p.is_dir():
        cands = sorted(p.glob("*_final.zip")) or sorted(p.rglob("*_final.zip"))
        if cands:
            return sorted(cands, key=lambda x: (len(x.parts), str(x)))[0]
    raise FileNotFoundError(f"Cannot resolve model path: {path_or_dir}")


def _goal_for_node(node: str) -> Goal:
    return Goal(node=to_new(node), target_obj="none", target_color="grey")


def _gc_obs(env: GCOptionEnv):
    raw_obs = env.env.unwrapped.gen_obs() if hasattr(env.env.unwrapped, "gen_obs") else env.env.unwrapped.observation(env.env.unwrapped.grid)
    return env._encode_obs(raw_obs)


def run_gc_module(env: GCOptionEnv, model: PPO, module: ModuleSpec, deterministic: bool) -> Dict:
    goal_node = to_new(module.goal_or_template)
    env._goal = _goal_for_node(goal_node)
    env._t = 0

    sg0 = extract_subgoals(env.env)
    env._prev_goal_val = int(sg0.get(goal_node, 0))
    if int(sg0.get(goal_node, 0)) == 1:
        return {"success": 1.0, "steps": 0}

    obs = _gc_obs(env)
    success = 0.0
    steps = 0
    for _ in range(int(module.horizon)):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
        if int(info.get("subgoals", {}).get(goal_node, 0)) == 1:
            success = 1.0
            break
        if terminated or truncated:
            break
    return {"success": success, "steps": steps}


def run_g2_module(env: G2TemplateEnv, model: PPO, module: ModuleSpec, deterministic: bool) -> Dict:
    template = module.goal_or_template
    external_check = to_new(module.external_check or template)
    env.set_template(template, parent_goal=module.parent_goal)

    if external_check != "door_ready":
        sg0 = extract_subgoals(env.env)
        if int(sg0.get(external_check, 0)) == 1:
            return {"success": 1.0, "steps": 0}

    obs = env.encode_current_obs()
    success = 0.0
    steps = 0
    for _ in range(int(module.horizon)):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
        if external_check == "door_ready":
            if bool(info.get("is_success", False)):
                success = 1.0
                break
        else:
            if int(info.get("subgoals", {}).get(external_check, 0)) == 1:
                success = 1.0
                break
        if terminated or truncated:
            break
    return {"success": success, "steps": steps}


def evaluate_chain(spec: ChainSpec, episodes: int, seed: int, deterministic: bool) -> Dict:
    model_cache: Dict[str, PPO] = {}
    for m in spec.modules:
        mp = str(resolve_model_path(m.model_path))
        if mp not in model_cache:
            model_cache[mp] = PPO.load(mp)
        m.model_path = mp

    if spec.granularity in ("G0", "G1"):
        env = GCOptionEnv(
            env_id=spec.env_id,
            goal_pool=[to_new(m.goal_or_template) for m in spec.modules],
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
    else:
        env = G2TemplateEnv(
            env_id=spec.env_id,
            template=spec.modules[0].goal_or_template,
            option_horizon=max(m.horizon for m in spec.modules),
            seed=seed,
            use_nodrop=False,
            use_precondition_do=False,
            parent_goal=None,
            reset_max_tries=128,
        )

    final_target = to_new(spec.final_target)
    final_successes, chain_completed, total_steps_list = [], [], []
    module_successes: Dict[str, List[float]] = {m.name: [] for m in spec.modules}
    module_steps: Dict[str, List[float]] = {m.name: [] for m in spec.modules}

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        completed = 1.0
        total_steps = 0
        for m in spec.modules:
            model = model_cache[str(resolve_model_path(m.model_path))]
            r = run_gc_module(env, model, m, deterministic) if spec.granularity in ("G0", "G1") else run_g2_module(env, model, m, deterministic)
            module_successes[m.name].append(float(r["success"]))
            module_steps[m.name].append(float(r["steps"]))
            total_steps += int(r["steps"])
            if float(r["success"]) < 1.0:
                completed = 0.0
                break
        sg = extract_subgoals(env.env)
        final_successes.append(float(int(sg.get(final_target, 0)) == 1))
        chain_completed.append(completed)
        total_steps_list.append(float(total_steps))

    env.close()
    row = {
        "chain_name": spec.name,
        "method": spec.method,
        "source_label": spec.source_label,
        "env_id": spec.env_id,
        "granularity": spec.granularity,
        "final_target": final_target,
        "episodes": episodes,
        "seed": seed,
        "deterministic": deterministic,
        "c_single": spec.c_single,
        "c_agg": spec.c_agg,
        "c_adapt": spec.c_adapt,
        "c_build": spec.c_single + spec.c_agg,
        "c_total": spec.c_single + spec.c_agg + spec.c_adapt,
        "final_success_rate": float(np.mean(final_successes)),
        "chain_completion_rate": float(np.mean(chain_completed)),
        "avg_total_steps": float(np.mean(total_steps_list)),
        "modules": " -> ".join(m.name for m in spec.modules),
    }
    for m in spec.modules:
        row[f"module_success/{m.name}"] = float(np.mean(module_successes[m.name]))
        row[f"module_steps/{m.name}"] = float(np.mean(module_steps[m.name]))
    return row


TARGET = "BabyAI-UnlockPickup-v0"
RG = Path("runs_granularity")
RT = Path("runs_transfer_door")


def p(*parts) -> str:
    return str(Path(*parts))


def g0_chain(name, method, source_label, has_key_path, opened_door_path, c_single, c_agg, c_adapt):
    return ChainSpec(name, method, "G0", TARGET, "opened_door", [
        ModuleSpec("has_key", has_key_path, "has_key"),
        ModuleSpec("opened_door", opened_door_path, "opened_door"),
    ], c_single, c_agg, c_adapt, source_label)


def g1_chain(name, method, source_label, near_key_path, nk_hk_path, hk_od_path, c_single, c_agg, c_adapt):
    return ChainSpec(name, method, "G1", TARGET, "opened_door", [
        ModuleSpec("near_key", near_key_path, "near_key"),
        ModuleSpec("near_key_to_has_key", nk_hk_path, "has_key"),
        ModuleSpec("has_key_to_opened_door", hk_od_path, "opened_door"),
    ], c_single, c_agg, c_adapt, source_label)


def g2_chain(name, method, source_label, approach_key_path, pickup_key_path, approach_door_path, toggle_door_path, c_single, c_agg, c_adapt):
    return ChainSpec(name, method, "G2", TARGET, "opened_door", [
        ModuleSpec("approach_key", approach_key_path, "approach_key", external_check="near_key"),
        ModuleSpec("pickup_key", pickup_key_path, "pickup_key", external_check="has_key"),
        ModuleSpec("approach_door", approach_door_path, "approach_door", external_check="door_ready"),
        ModuleSpec("toggle_door", toggle_door_path, "toggle_door", external_check="opened_door"),
    ], c_single, c_agg, c_adapt, source_label)


def build_specs() -> List[ChainSpec]:
    specs: List[ChainSpec] = []
    A_SINGLE, A_AGG = 1_050_000, 0

    specs += [
        g0_chain("A_pool_G0_doorkey_zero", "A_expert_pool_source_chain", "DoorKey", p(RG,"G0_door_doorkey_has_key_s0_fair200k"), p(RG,"G0_door_doorkey_opened_door_s0"), A_SINGLE,A_AGG,0),
        g0_chain("A_pool_G0_unlocklocal_zero", "A_expert_pool_source_chain", "UnlockLocal", p(RG,"G0_door_unlocklocal_has_key_s0_fair200k"), p(RG,"G0_door_unlocklocal_opened_door_s0"), A_SINGLE,A_AGG,0),
        g0_chain("A_pool_G0_unlockpickup_zero", "A_expert_pool_source_chain", "UnlockPickup", p(RG,"G0_door_unlockpickup_has_key_s0_fair200k"), p(RG,"G0_door_unlockpickup_opened_door_s0"), A_SINGLE,A_AGG,0),
        g1_chain("A_pool_G1_doorkey_zero", "A_expert_pool_source_chain", "DoorKey", p(RG,"G1_door_doorkey_near_key_s0"), p(RG,"G1_door_doorkey_near_key_to_has_key_s0"), p(RG,"G1_door_doorkey_has_key_to_opened_door_s0_fair150k"), A_SINGLE,A_AGG,0),
        g1_chain("A_pool_G1_unlocklocal_zero", "A_expert_pool_source_chain", "UnlockLocal", p(RG,"G1_door_unlocklocal_near_key_s0"), p(RG,"G1_door_unlocklocal_near_key_to_has_key_s0"), p(RG,"G1_door_unlocklocal_has_key_to_opened_door_s0_fair150k"), A_SINGLE,A_AGG,0),
        g1_chain("A_pool_G1_unlockpickup_zero", "A_expert_pool_source_chain", "UnlockPickup", p(RG,"G1_door_unlockpickup_near_key_s0"), p(RG,"G1_door_unlockpickup_near_key_to_has_key_s0"), p(RG,"G1_door_unlockpickup_has_key_to_opened_door_s0_fair150k"), A_SINGLE,A_AGG,0),
        g2_chain("A_pool_G2_doorkey_zero", "A_expert_pool_source_chain", "DoorKey", p(RG,"G2_door_doorkey_approach_key_s0"), p(RG,"G2_door_doorkey_pickup_key_s0"), p(RG,"G2_door_doorkey_approach_door_s0"), p(RG,"G2_door_doorkey_toggle_door_s0"), A_SINGLE,A_AGG,0),
        g2_chain("A_pool_G2_unlocklocal_zero", "A_expert_pool_source_chain", "UnlockLocal", p(RG,"G2_door_unlocklocal_approach_key_s0"), p(RG,"G2_door_unlocklocal_pickup_key_s0"), p(RG,"G2_door_unlocklocal_approach_door_s0"), p(RG,"G2_door_unlocklocal_toggle_door_s0"), A_SINGLE,A_AGG,0),
        g2_chain("A_pool_G2_unlockpickup_zero", "A_expert_pool_source_chain", "UnlockPickup", p(RG,"G2_door_unlockpickup_approach_key_s0"), p(RG,"G2_door_unlockpickup_pickup_key_s0"), p(RG,"G2_door_unlockpickup_approach_door_s0"), p(RG,"G2_door_unlockpickup_toggle_door_s0"), A_SINGLE,A_AGG,0),
    ]

    B_SINGLE, B_AGG = 350_000, 140_000
    specs += [
        g0_chain("B_seq_G0_adapt0", "B_sequential", "DoorKey->UnlockLocal->UnlockPickup", p(RT,"B_SEQ_G0_has_key_step2_unlockpickup"), p(RT,"B_SEQ_G0_opened_door_step2_unlockpickup"), B_SINGLE,B_AGG,0),
        g1_chain("B_seq_G1_adapt0", "B_sequential", "DoorKey->UnlockLocal->UnlockPickup", p(RT,"B_SEQ_G1_near_key_step2_unlockpickup"), p(RT,"B_SEQ_G1_near_key_to_has_key_step2_unlockpickup"), p(RT,"B_SEQ_G1_has_key_to_opened_door_step2_unlockpickup"), B_SINGLE,B_AGG,0),
        g2_chain("B_seq_G2_adapt0", "B_sequential", "DoorKey->UnlockLocal->UnlockPickup", p(RT,"B_SEQ_G2_approach_key_step2_unlockpickup"), p(RT,"B_SEQ_G2_pickup_key_step2_unlockpickup"), p(RT,"B_SEQ_G2_approach_door_step2_unlockpickup"), p(RT,"B_SEQ_G2_toggle_door_step2_unlockpickup"), B_SINGLE,B_AGG,0),
    ]
    for budget in [35_000, 70_000, 140_000]:
        tag = budget // 1000
        specs += [
            g0_chain(f"B_seq_G0_adapt{tag}", "B_sequential", "DoorKey->UnlockLocal->UnlockPickup", p(RT,f"B_ADAPT{tag}_G0_has_key"), p(RT,f"B_ADAPT{tag}_G0_opened_door"), B_SINGLE,B_AGG,budget),
            g1_chain(f"B_seq_G1_adapt{tag}", "B_sequential", "DoorKey->UnlockLocal->UnlockPickup", p(RT,f"B_ADAPT{tag}_G1_near_key"), p(RT,f"B_ADAPT{tag}_G1_near_key_to_has_key"), p(RT,f"B_ADAPT{tag}_G1_has_key_to_opened_door"), B_SINGLE,B_AGG,budget),
            g2_chain(f"B_seq_G2_adapt{tag}", "B_sequential", "DoorKey->UnlockLocal->UnlockPickup", p(RT,f"B_ADAPT{tag}_G2_approach_key"), p(RT,f"B_ADAPT{tag}_G2_pickup_key"), p(RT,f"B_ADAPT{tag}_G2_approach_door"), p(RT,f"B_ADAPT{tag}_G2_toggle_door"), B_SINGLE,B_AGG,budget),
        ]
    return specs


def write_rows(rows: List[Dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out_csv", type=str, default="results_transfer_door/door_transfer_eval.csv")
    ap.add_argument("--only_method", type=str, default="", help="A_expert_pool_source_chain or B_sequential")
    ap.add_argument("--only_granularity", type=str, default="", help="G0, G1, or G2")
    args = ap.parse_args()

    specs = build_specs()
    if args.only_method:
        specs = [s for s in specs if s.method == args.only_method]
    if args.only_granularity:
        specs = [s for s in specs if s.granularity == args.only_granularity]

    rows = []
    for spec in specs:
        print(f"\n[EVAL] {spec.name} ({spec.method}, {spec.granularity}, c_adapt={spec.c_adapt})")
        row = evaluate_chain(spec, episodes=args.episodes, seed=args.seed, deterministic=args.deterministic)
        print({k: row[k] for k in ["chain_name", "method", "granularity", "c_total", "final_success_rate", "avg_total_steps"]})
        rows.append(row)

    write_rows(rows, Path(args.out_csv))
    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
