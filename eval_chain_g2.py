# eval_chain_g2.py
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
from rl.g2_template_env import G2TemplateEnv


@dataclass
class G2ModuleSpec:
    name: str
    model_path: str
    template: str
    external_check: str
    horizon: int = 256
    parent_goal: Optional[str] = None


@dataclass
class G2ChainSpec:
    name: str
    env_id: str
    final_target: str
    total_train_steps: int
    modules: List[G2ModuleSpec]


def p(run_name: str) -> str:
    return str(Path("runs_granularity") / run_name)


def resolve_model_path(path_or_dir: str) -> Path:
    pth = Path(path_or_dir)
    if pth.is_file():
        return pth
    if pth.is_dir():
        cands = sorted(pth.glob("*_final.zip"))
        if not cands:
            cands = sorted(pth.rglob("*_final.zip"))
        if cands:
            return sorted(cands, key=lambda x: (len(x.parts), str(x)))[0]
    raise FileNotFoundError(f"Cannot resolve model path: {path_or_dir}")


def run_g2_module(env: G2TemplateEnv, model: PPO, module: G2ModuleSpec, deterministic: bool) -> Dict:
    env.set_template(module.template, parent_goal=module.parent_goal)

    # If external target is already satisfied, skip.
    check = to_new(module.external_check)
    if check != "door_ready":
        sg0 = extract_subgoals(env.env)
        if int(sg0.get(check, 0)) == 1:
            return {
                "module_name": module.name,
                "template": module.template,
                "external_check": check,
                "success": 1.0,
                "steps": 0,
                "already_satisfied": 1.0,
                "pickup_attempts": 0,
                "drop_attempts": 0,
                "toggle_attempts": 0,
                "carrying_key": 0.0,
                "carrying_target": 0.0,
            }

    obs = env.encode_current_obs()
    success = 0.0
    steps = 0
    last_info = {}

    for _ in range(int(module.horizon)):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info
        steps += 1

        if check == "door_ready":
            # approach_door template success is internal; env reward detects it.
            if bool(info.get("is_success", False)):
                success = 1.0
                break
        else:
            sg = info.get("subgoals", {})
            if int(sg.get(check, 0)) == 1:
                success = 1.0
                break

        if terminated or truncated:
            break

    return {
        "module_name": module.name,
        "template": module.template,
        "external_check": check,
        "success": success,
        "steps": steps,
        "already_satisfied": 0.0,
        "pickup_attempts": int(last_info.get("debug_g2_pickup_attempt_count", 0)),
        "drop_attempts": int(last_info.get("debug_g2_drop_attempt_count", 0)),
        "toggle_attempts": int(last_info.get("debug_g2_toggle_attempt_count", 0)),
        "carrying_key": float(last_info.get("debug_g2_carrying_key", 0.0)),
        "carrying_target": float(last_info.get("debug_g2_carrying_target", 0.0)),
    }


def evaluate_g2_chain(spec: G2ChainSpec, episodes: int, seed: int, deterministic: bool) -> Dict:
    model_cache: Dict[str, PPO] = {}
    for m in spec.modules:
        mp = str(resolve_model_path(m.model_path))
        if mp not in model_cache:
            model_cache[mp] = PPO.load(mp)
        m.model_path = mp

    first = spec.modules[0]
    env = G2TemplateEnv(
        env_id=spec.env_id,
        template=first.template,
        option_horizon=max(m.horizon for m in spec.modules),
        seed=seed,
        use_nodrop=False,
        use_precondition_do=False,
        parent_goal=None,
        reset_max_tries=128,
    )

    final_target = to_new(spec.final_target)
    final_successes = []
    chain_completed = []
    total_steps_list = []
    module_successes: Dict[str, List[float]] = {m.name: [] for m in spec.modules}
    module_steps: Dict[str, List[float]] = {m.name: [] for m in spec.modules}
    module_pickup_attempts: Dict[str, List[float]] = {m.name: [] for m in spec.modules}
    module_drop_attempts: Dict[str, List[float]] = {m.name: [] for m in spec.modules}
    module_toggle_attempts: Dict[str, List[float]] = {m.name: [] for m in spec.modules}
    module_carrying_key = {m.name: [] for m in spec.modules}
    module_carrying_target = {m.name: [] for m in spec.modules}

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        completed = 1.0
        total_steps = 0

        for m in spec.modules:
            model = model_cache[str(resolve_model_path(m.model_path))]
            r = run_g2_module(env, model, m, deterministic=deterministic)
            module_successes[m.name].append(float(r["success"]))
            module_steps[m.name].append(float(r["steps"]))
            module_pickup_attempts[m.name].append(float(r.get("pickup_attempts", 0)))
            module_drop_attempts[m.name].append(float(r.get("drop_attempts", 0)))
            module_toggle_attempts[m.name].append(float(r.get("toggle_attempts", 0)))
            module_carrying_key[m.name].append(float(r.get("carrying_key", 0.0)))
            module_carrying_target[m.name].append(float(r.get("carrying_target", 0.0)))
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
        "granularity": "G2",
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
        out[f"module_pickup_attempts/{m.name}"] = float(np.mean(module_pickup_attempts[m.name]))
        out[f"module_drop_attempts/{m.name}"] = float(np.mean(module_drop_attempts[m.name]))
        out[f"module_toggle_attempts/{m.name}"] = float(np.mean(module_toggle_attempts[m.name]))
        out[f"module_carrying_key/{m.name}"] = float(np.mean(module_carrying_key[m.name]))
        out[f"module_carrying_target/{m.name}"] = float(np.mean(module_carrying_target[m.name]))

    return out


PRESETS: Dict[str, G2ChainSpec] = {
    "pickup_fetch_g2": G2ChainSpec(
        name="pickup_fetch_g2",
        env_id="MiniGrid-Fetch-8x8-N3-v0",
        final_target="task_success",
        total_train_steps=200_000,
        modules=[
            G2ModuleSpec("approach_target", p("G2_pickup_fetch_approach_target_s0"), "approach_target", "near_target"),
            G2ModuleSpec("pickup_target", p("G2_pickup_fetch_pickup_target_s0"), "pickup_target", "has_target"),
        ],
    ),
    "pickup_babyai_g2": G2ChainSpec(
        name="pickup_babyai_g2",
        env_id="BabyAI-Pickup-v0",
        final_target="task_success",
        total_train_steps=200_000,
        modules=[
            G2ModuleSpec("approach_target", p("G2_pickup_babyai_approach_target_s0"), "approach_target", "near_target"),
            G2ModuleSpec("pickup_target", p("G2_pickup_babyai_pickup_target_s0"), "pickup_target", "has_target"),
        ],
    ),

    "doorkey_has_key_g2": G2ChainSpec(
        name="doorkey_has_key_g2",
        env_id="MiniGrid-DoorKey-6x6-v0",
        final_target="has_key",
        total_train_steps=200_000,
        modules=[
            G2ModuleSpec("approach_key", p("G2_door_doorkey_approach_key_s0"), "approach_key", "near_key"),
            G2ModuleSpec("pickup_key", p("G2_door_doorkey_pickup_key_s0"), "pickup_key", "has_key"),
        ],
    ),
    "doorkey_opened_door_g2": G2ChainSpec(
        name="doorkey_opened_door_g2",
        env_id="MiniGrid-DoorKey-6x6-v0",
        final_target="opened_door",
        total_train_steps=350_000,
        modules=[
            G2ModuleSpec("approach_key", p("G2_door_doorkey_approach_key_s0"), "approach_key", "near_key"),
            G2ModuleSpec("pickup_key", p("G2_door_doorkey_pickup_key_s0"), "pickup_key", "has_key"),
            G2ModuleSpec("approach_door", p("G2_door_doorkey_approach_door_s0"), "approach_door", "door_ready"),
            G2ModuleSpec("toggle_door", p("G2_door_doorkey_toggle_door_s0"), "toggle_door", "opened_door"),
        ],
    ),
    "doorkey_task_g2": G2ChainSpec(
        name="doorkey_task_g2",
        env_id="MiniGrid-DoorKey-6x6-v0",
        final_target="task_success",
        total_train_steps=500_000,
        modules=[
            G2ModuleSpec("approach_key", p("G2_door_doorkey_approach_key_s0"), "approach_key", "near_key"),
            G2ModuleSpec("pickup_key", p("G2_door_doorkey_pickup_key_s0"), "pickup_key", "has_key"),
            G2ModuleSpec("approach_door", p("G2_door_doorkey_approach_door_s0"), "approach_door", "door_ready"),
            G2ModuleSpec("toggle_door", p("G2_door_doorkey_toggle_door_s0"), "toggle_door", "opened_door"),
            G2ModuleSpec("approach_goal", p("G2_door_doorkey_approach_goal_s0"), "approach_goal", "at_goal"),
        ],
    ),

    "unlocklocal_has_key_g2": G2ChainSpec(
        name="unlocklocal_has_key_g2",
        env_id="BabyAI-UnlockLocal-v0",
        final_target="has_key",
        total_train_steps=200_000,
        modules=[
            G2ModuleSpec("approach_key", p("G2_door_unlocklocal_approach_key_s0"), "approach_key", "near_key"),
            G2ModuleSpec("pickup_key", p("G2_door_unlocklocal_pickup_key_s0"), "pickup_key", "has_key"),
        ],
    ),
    "unlocklocal_task_g2": G2ChainSpec(
        name="unlocklocal_task_g2",
        env_id="BabyAI-UnlockLocal-v0",
        final_target="task_success",
        total_train_steps=350_000,
        modules=[
            G2ModuleSpec("approach_key", p("G2_door_unlocklocal_approach_key_s0"), "approach_key", "near_key"),
            G2ModuleSpec("pickup_key", p("G2_door_unlocklocal_pickup_key_s0"), "pickup_key", "has_key"),
            G2ModuleSpec("approach_door", p("G2_door_unlocklocal_approach_door_s0"), "approach_door", "door_ready"),
            G2ModuleSpec("toggle_door", p("G2_door_unlocklocal_toggle_door_s0"), "toggle_door", "opened_door"),
        ],
    ),

    "unlockpickup_has_key_g2": G2ChainSpec(
        name="unlockpickup_has_key_g2",
        env_id="MiniGrid-UnlockPickup-v0",
        final_target="has_key",
        total_train_steps=200_000,
        modules=[
            G2ModuleSpec("approach_key", p("G2_door_unlockpickup_approach_key_s0"), "approach_key", "near_key"),
            G2ModuleSpec("pickup_key", p("G2_door_unlockpickup_pickup_key_s0"), "pickup_key", "has_key"),
        ],
    ),
    "unlockpickup_opened_door_g2": G2ChainSpec(
        name="unlockpickup_opened_door_g2",
        env_id="MiniGrid-UnlockPickup-v0",
        final_target="opened_door",
        total_train_steps=350_000,
        modules=[
            G2ModuleSpec("approach_key", p("G2_door_unlockpickup_approach_key_s0"), "approach_key", "near_key"),
            G2ModuleSpec("pickup_key", p("G2_door_unlockpickup_pickup_key_s0"), "pickup_key", "has_key"),
            G2ModuleSpec("approach_door", p("G2_door_unlockpickup_approach_door_s0"), "approach_door", "door_ready"),
            G2ModuleSpec("toggle_door", p("G2_door_unlockpickup_toggle_door_s0"), "toggle_door", "opened_door"),
        ],
    ),
    "unlockpickup_task_g2": G2ChainSpec(
        name="unlockpickup_task_g2",
        env_id="MiniGrid-UnlockPickup-v0",
        final_target="task_success",
        total_train_steps=500_000,
        modules=[
            G2ModuleSpec("approach_key", p("G2_door_unlockpickup_approach_key_s0"), "approach_key", "near_key"),
            G2ModuleSpec("pickup_key", p("G2_door_unlockpickup_pickup_key_s0"), "pickup_key", "has_key"),
            G2ModuleSpec("approach_door", p("G2_door_unlockpickup_approach_door_s0"), "approach_door", "door_ready"),
            G2ModuleSpec("toggle_door", p("G2_door_unlockpickup_toggle_door_s0"), "toggle_door", "opened_door"),
            G2ModuleSpec("approach_target", p("G2_door_unlockpickup_approach_target_s0"), "approach_target", "near_target", parent_goal="opened_door"),
            G2ModuleSpec("pickup_target", p("G2_door_unlockpickup_pickup_target_s0"), "pickup_target", "has_target"),
        ],
    ),
}


def write_rows(rows: List[Dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, default="all", help="Preset name, comma-separated names, or all")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out_csv", type=str, default="results_granularity/chain_eval_g2.csv")
    args = ap.parse_args()

    if args.preset == "all":
        names = list(PRESETS.keys())
    else:
        names = [x.strip() for x in args.preset.split(",") if x.strip()]

    rows = []
    for name in names:
        if name not in PRESETS:
            raise KeyError(f"Unknown preset {name}. Available: {sorted(PRESETS.keys())}")
        print(f"\n[EVAL G2 CHAIN] {name}")
        row = evaluate_g2_chain(PRESETS[name], episodes=args.episodes, seed=args.seed, deterministic=args.deterministic)
        rows.append(row)
        print(row)

    out_csv = Path(args.out_csv)
    write_rows(rows, out_csv)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
