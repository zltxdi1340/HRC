import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rl.gc_option_env import GCOptionEnv


EPISODES = 200
SEED = 0
OPTION_HORIZON = 256
DETERMINISTIC = False


def resolve_model_path(path_or_dir: str):
    p = Path(path_or_dir)
    if p.is_file():
        return p
    if p.is_dir():
        cands = sorted(p.glob("*_final.zip"))
        if not cands:
            cands = sorted(p.rglob("*_final.zip"))
        if len(cands) >= 1:
            cands = sorted(cands, key=lambda x: (len(x.parts), str(x)))
            return cands[0]
    return None


def eval_model_on_env(model_path: str, make_env_fn, env_id: str, episodes: int = 200, seed: int = 0, deterministic: bool = False):
    env = make_env_fn(env_id, seed)
    model = PPO.load(model_path)

    success_list = []
    steps_list = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_steps = 0
        final_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_steps += 1
            final_info = info

        success_list.append(float(final_info.get("is_success", False)))
        steps_list.append(ep_steps)

    return {
        "success_rate": float(np.mean(success_list)),
        "avg_steps": float(np.mean(steps_list)),
    }


def save_long_and_pivot(rows, out_prefix: str, test_envs):
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    out_csv = out_dir / f"{out_prefix}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_name", "test_env", "success_rate", "avg_steps"],
        )
        writer.writeheader()
        writer.writerows(rows)

    pivot = {}
    for row in rows:
        model_name = row["model_name"]
        test_env = row["test_env"]
        success = row["success_rate"]
        if model_name not in pivot:
            pivot[model_name] = {}
        pivot[model_name][test_env] = success

    pivot_csv = out_dir / f"{out_prefix}_pivot.csv"
    with open(pivot_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model_name"] + list(test_envs)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name in pivot:
            row = {"model_name": model_name}
            for env_id in test_envs:
                row[env_id] = pivot[model_name].get(env_id, "")
            writer.writerow(row)

    print(f"Saved long CSV to: {out_csv}")
    print(f"Saved pivot CSV to: {pivot_csv}")


MODELS = [
    ("A3_E_empty_s0", r"runs_gcppo\A3_E_empty_s0"),
    ("A3_E_fourrooms_s0", r"runs_gcppo\A3_E_fourrooms_s0"),
    ("A3_shared_seq", r"runs_gcppo\A3_SEQ_step2_fourrooms_s0"),
    ("A3_shared_joint", r"runs_gcppo\A3_joint_s0"),
]

TEST_ENVS = [
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-FourRooms-v0",
]


def make_env(env_id: str, seed: int):
    horizon = 512 if env_id == "MiniGrid-FourRooms-v0" else OPTION_HORIZON
    cfg = dict(
        env_id=env_id,
        goal_pool=["at_goal"],
        option_horizon=horizon,
        seed=seed,
        use_nodrop=False,
        use_precondition_do=True,
        parent_goal=None,
        child_goal=None,
        strict_parent_mode=False,
        strict_parent_fail_fast=False,
        reset_max_tries=128,
    )
    print("[EVAL ENV CFG]", cfg)
    return GCOptionEnv(**cfg)


def main():
    rows = []
    for model_name, path_or_dir in MODELS:
        model_path = resolve_model_path(path_or_dir)
        if model_path is None:
            print(f"[SKIP] model not found: {path_or_dir}")
            continue
        for env_id in TEST_ENVS:
            result = eval_model_on_env(
                str(model_path),
                make_env,
                env_id,
                episodes=EPISODES,
                seed=SEED,
                deterministic=DETERMINISTIC,
            )
            row = {
                "model_name": model_name,
                "test_env": env_id,
                "success_rate": result["success_rate"],
                "avg_steps": result["avg_steps"],
            }
            rows.append(row)
            print(row)

    save_long_and_pivot(rows, "a3_eval_matrix", TEST_ENVS)


if __name__ == "__main__":
    main()
