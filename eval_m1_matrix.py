import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rl.gc_option_env import GCOptionEnv

MODELS = [
    ("M1_E_doorkey_nff_s0", r"runs_gcppo\M1_E_doorkey_nff_s0\MiniGrid-DoorKey-6x6-v0_final.zip"),
    ("M1_E_unlockpickup_nff_s0", r"runs_gcppo\M1_E_unlockpickup_nff_s0\MiniGrid-UnlockPickup-v0_final.zip"),
    ("M1_E_unlocklocal_nff_s0", r"runs_gcppo\M1_E_unlocklocal_nff_s0\BabyAI-UnlockLocal-v0_final.zip"),
    ("M1_E_keyinbox_nff_s0", r"runs_gcppo\M1_E_keyinbox_nff_s0\BabyAI-KeyInBox-v0_final.zip"),
    ("M1_shared_core", r"runs_gcppo\M1_SEQ_step3_unlocklocal_nff_s0\BabyAI-UnlockLocal-v0_final.zip"),
    ("M1_shared_ext", r"runs_gcppo\M1_SEQ_step4_keyinbox_nff_s0\BabyAI-KeyInBox-v0_final.zip"),
    ("M1_joint_ext", r"runs_gcppo\M1_joint_ext_nff_s0\M1_UnlockDoor_joint_final.zip"),
]

TEST_ENVS = [
    "MiniGrid-DoorKey-6x6-v0",
    "MiniGrid-UnlockPickup-v0",
    "BabyAI-UnlockLocal-v0",
    "BabyAI-KeyInBox-v0",
]

EPISODES = 200
SEED = 0
OPTION_HORIZON = 256
DETERMINISTIC = False


def make_env(env_id: str, seed: int):
    cfg = dict(
        env_id=env_id,
        goal_pool=["door_open"],
        option_horizon=OPTION_HORIZON,
        seed=seed,
        use_nodrop=False,
        use_precondition_do=True,
        parent_goal="has_key",
        child_goal="door_open",
        strict_parent_mode=True,
        strict_parent_fail_fast=False,
        reset_max_tries=128,
    )
    print("[EVAL ENV CFG]", cfg)
    return GCOptionEnv(**cfg)


def eval_model_on_env(model_path: str, env_id: str, episodes: int = 200, seed: int = 0, deterministic: bool = False):
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

    env.close()
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
            result = eval_model_on_env(model_path, env_id, episodes=EPISODES, seed=SEED, deterministic=DETERMINISTIC)
            row = {
                "model_name": model_name,
                "test_env": env_id,
                "success_rate": result["success_rate"],
                "avg_steps": result["avg_steps"],
            }
            rows.append(row)
            print(row)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "m1_eval_matrix.csv"
    out_pivot_csv = out_dir / "m1_eval_matrix_pivot.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "test_env", "success_rate", "avg_steps"])
        writer.writeheader()
        writer.writerows(rows)

    # Simple pivot for success_rate.
    model_names = [m for m, _ in MODELS if any(r["model_name"] == m for r in rows)]
    env_names = list(TEST_ENVS)
    data = {(r["model_name"], r["test_env"]): r["success_rate"] for r in rows}

    with out_pivot_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model\\env"] + env_names)
        for m in model_names:
            writer.writerow([m] + [data.get((m, e), "") for e in env_names])

    print(f"Saved long CSV to: {out_csv}")
    print(f"Saved pivot CSV to: {out_pivot_csv}")


if __name__ == "__main__":
    main()
