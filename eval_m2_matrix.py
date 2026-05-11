import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rl.gc_option_env import GCOptionEnv

# 按你当前 M2 的训练命名来写
MODELS = [
    ("M2_E_babyaipickup_nff_s0", r"runs_gcppo\M2_E_babyaipickup_nff_s0\BabyAI-Pickup-v0_final.zip"),
    ("M2_E_fetch_nff_s0", r"runs_gcppo\M2_E_fetch_nff_s0\MiniGrid-Fetch-8x8-N3-v0_final.zip"),
    ("M2_shared_seq", r"runs_gcppo\M2_SEQ_step2_fetch_nff_s0\MiniGrid-Fetch-8x8-N3-v0_final.zip"),
    ("M2_shared_joint", r"runs_gcppo\M2_joint_nff_s0\M2_Pickup_joint_final.zip"),
]

TEST_ENVS = [
    "BabyAI-Pickup-v0",
    "MiniGrid-Fetch-8x8-N3-v0",
]

EPISODES = 200
SEED = 0
OPTION_HORIZON = 256
DETERMINISTIC = False


def make_env(env_id: str, seed: int):
    cfg = dict(
        env_id=env_id,
        goal_pool=["has_obj"],
        option_horizon=OPTION_HORIZON,
        seed=seed,
        use_nodrop=False,
        use_precondition_do=True,
        parent_goal="near_obj",
        child_goal="has_obj",
        strict_parent_mode=True,
        strict_parent_fail_fast=False,
        reset_max_tries=128,
    )
    print("[EVAL ENV CFG]", cfg)
    return GCOptionEnv(**cfg)


def eval_model_on_env(
    model_path: str,
    env_id: str,
    episodes: int = 200,
    seed: int = 0,
    deterministic: bool = False,
):
    env = make_env(env_id, seed)
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


def main():
    rows = []

    for model_name, model_path in MODELS:
        if not Path(model_path).exists():
            print(f"[SKIP] model not found: {model_path}")
            continue

        for env_id in TEST_ENVS:
            result = eval_model_on_env(
                model_path,
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

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    out_csv = out_dir / "m2_eval_matrix.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_name", "test_env", "success_rate", "avg_steps"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # 保存 pivot 版，方便直接看表
    pivot = {}
    for row in rows:
        model_name = row["model_name"]
        test_env = row["test_env"]
        success = row["success_rate"]
        if model_name not in pivot:
            pivot[model_name] = {}
        pivot[model_name][test_env] = success

    pivot_csv = out_dir / "m2_eval_matrix_pivot.csv"
    with open(pivot_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model_name"] + TEST_ENVS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name in pivot:
            row = {"model_name": model_name}
            for env_id in TEST_ENVS:
                row[env_id] = pivot[model_name].get(env_id, "")
            writer.writerow(row)

    print(f"\nSaved long CSV to: {out_csv}")
    print(f"Saved pivot CSV to: {pivot_csv}")


if __name__ == "__main__":
    main()