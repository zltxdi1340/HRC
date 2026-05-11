# eval_granularity_module.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

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

from env.node_alias import normalize_nodes, to_new
from rl.gc_option_env import GCOptionEnv


def resolve_model_path(path_or_dir: str):
    p = Path(path_or_dir)
    if p.is_file():
        return p
    if p.is_dir():
        cands = sorted(p.glob("*_final.zip"))
        if not cands:
            cands = sorted(p.rglob("*_final.zip"))
        if cands:
            return sorted(cands, key=lambda x: (len(x.parts), str(x)))[0]
    return None


def _split_csv(text: str):
    return [x.strip() for x in text.split(",") if x.strip()]


def eval_model(
    *,
    model_path: str,
    env_id: str,
    goals: list[str],
    parent_goal: str | None,
    child_goal: str | None,
    strict_parent_mode: bool,
    strict_parent_fail_fast: bool,
    use_precondition_do: bool,
    option_horizon: int,
    episodes: int,
    seed: int,
    deterministic: bool,
    reset_max_tries: int,
):
    env = GCOptionEnv(
        env_id=env_id,
        goal_pool=goals,
        option_horizon=option_horizon,
        seed=seed,
        use_nodrop=False,
        use_precondition_do=use_precondition_do,
        parent_goal=parent_goal,
        child_goal=child_goal,
        strict_parent_mode=strict_parent_mode,
        strict_parent_fail_fast=strict_parent_fail_fast,
        reset_max_tries=reset_max_tries,
    )
    model = PPO.load(model_path)

    successes = []
    steps = []

    for ep in range(episodes):
        obs, _info = env.reset(seed=seed + ep)
        done = False
        ep_steps = 0
        final_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_steps += 1
            final_info = info

        successes.append(float(final_info.get("is_success", False)))
        steps.append(float(ep_steps))

    env.close()
    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
    }


def append_csv(out_csv: Path, row: dict):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = out_csv.exists()
    fieldnames = [
        "model_name",
        "model_path",
        "env_id",
        "goals",
        "parent_goal",
        "child_goal",
        "strict_parent_mode",
        "use_precondition_do",
        "episodes",
        "seed",
        "success_rate",
        "avg_steps",
    ]
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="Path to .zip or run directory.")
    ap.add_argument("--model_name", type=str, default="")
    ap.add_argument("--env_id", type=str, required=True)
    ap.add_argument("--goals", type=str, required=True)
    ap.add_argument("--parent_goal", type=str, default="")
    ap.add_argument("--child_goal", type=str, default="")
    ap.add_argument("--strict_parent_mode", action="store_true")
    ap.add_argument("--no_strict_parent_fail_fast", action="store_true")
    ap.add_argument("--no_precondition_do", action="store_true")
    ap.add_argument("--option_horizon", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--reset_max_tries", type=int, default=128)
    ap.add_argument("--out_csv", type=str, default="results_granularity/eval_granularity_module.csv")
    args = ap.parse_args()

    model_path = resolve_model_path(args.model_path)
    if model_path is None:
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    goals = normalize_nodes(_split_csv(args.goals))
    parent_goal = to_new(args.parent_goal.strip()) if args.parent_goal.strip() else None
    child_goal = to_new(args.child_goal.strip()) if args.child_goal.strip() else None
    strict_parent_mode = bool(args.strict_parent_mode or (parent_goal and child_goal))
    strict_parent_fail_fast = False if args.no_strict_parent_fail_fast else strict_parent_mode
    use_precondition_do = not args.no_precondition_do

    result = eval_model(
        model_path=str(model_path),
        env_id=args.env_id,
        goals=goals,
        parent_goal=parent_goal,
        child_goal=child_goal,
        strict_parent_mode=strict_parent_mode,
        strict_parent_fail_fast=strict_parent_fail_fast,
        use_precondition_do=use_precondition_do,
        option_horizon=args.option_horizon,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=args.deterministic,
        reset_max_tries=args.reset_max_tries,
    )

    model_name = args.model_name.strip() or model_path.parent.name
    row = {
        "model_name": model_name,
        "model_path": str(model_path),
        "env_id": args.env_id,
        "goals": ",".join(goals),
        "parent_goal": parent_goal or "",
        "child_goal": child_goal or "",
        "strict_parent_mode": strict_parent_mode,
        "use_precondition_do": use_precondition_do,
        "episodes": args.episodes,
        "seed": args.seed,
        "success_rate": result["success_rate"],
        "avg_steps": result["avg_steps"],
    }
    print(row)
    append_csv(Path(args.out_csv), row)
    print(f"Saved/updated: {args.out_csv}")


if __name__ == "__main__":
    main()
