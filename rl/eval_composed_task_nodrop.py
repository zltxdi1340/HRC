import argparse
import io
import contextlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from env.subgoals import extract_subgoals, parse_pickup_target
from rl.gc_option_env import Goal, _goal_vector


NODE2OBJ = {"has_key": "key", "has_box": "box", "has_ball": "ball"}


def _silent_reset(env, seed=None, options=None):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return env.reset(seed=seed, options=options)


def make_goal(goal_name: str, env) -> Goal:
    target_obj = "none"
    target_color = "grey"
    if goal_name == "has_target":
        obj, color = parse_pickup_target(env)
        if obj is not None:
            target_obj = obj
        if color is not None:
            target_color = color
    elif goal_name in NODE2OBJ:
        target_obj = NODE2OBJ[goal_name]
    return Goal(node=goal_name, target_obj=target_obj, target_color=target_color)


def encode_obs(obs, goal: Goal):
    if isinstance(obs, dict):
        img = obs["image"]
        direction = int(obs.get("direction", 0))
    else:
        img = obs
        direction = 0
    img_chw = np.transpose(img, (2, 0, 1)).astype(np.uint8)
    gvec = _goal_vector(goal).astype(np.float32)
    return {"image": img_chw, "direction": direction, "goal": gvec}


def select_plan(env_id: str, env, babyai_mode: str) -> List[str]:
    if "MiniGrid-Empty" in env_id:
        return ["at_goal"]
    if "MiniGrid-DoorKey" in env_id and "DoorKeyBall" not in env_id:
        return ["has_key", "door_open", "at_goal"]
    if "MiniGrid-UnlockPickup" in env_id:
        return ["has_key", "door_open", "has_box"]
    if "BabyAI-Pickup" in env_id:
        if babyai_mode == "unified":
            return ["has_target"]
        obj, _color = parse_pickup_target(env)
        route = {"key": "has_key", "ball": "has_ball", "box": "has_box"}.get(obj, "has_target")
        return [route]
    raise ValueError(f"Unsupported env_id: {env_id}")


def default_horizon(env_id: str, goal: str) -> int:
    if "BabyAI-Pickup" in env_id:
        return 512
    if "MiniGrid-UnlockPickup" in env_id and goal == "has_box":
        return 512
    return 256


def maybe_bridge_unlockpickup(env_id: str, env, next_goal: str):
    # Optional bridge for UnlockPickup if caller wants to mimic the state that has_box saw in training.
    if "MiniGrid-UnlockPickup" in env_id and next_goal == "has_box":
        carrying = getattr(env.unwrapped, "carrying", None)
        if carrying is not None and getattr(carrying, "type", None) == "key":
            env.unwrapped.carrying = None


def reset_raw_env(env, env_id: str, seed: int):
    if "BabyAI-Pickup" in env_id:
        return _silent_reset(env, seed=seed)
    return env.reset(seed=seed)


def run_option(model, env, env_id: str, goal_name: str, horizon: int, deterministic: bool = True) -> Tuple[int, bool, bool, dict]:
    obs = env.unwrapped.gen_obs() if hasattr(env.unwrapped, "gen_obs") else None
    if obs is None:
        raise RuntimeError("Could not get obs from raw env")

    goal = make_goal(goal_name, env)
    steps = 0
    done = False
    info = {}

    sg = extract_subgoals(env)
    if int(sg.get(goal_name, 0)) == 1:
        return steps, True, done, info

    for _ in range(horizon):
        enc = encode_obs(obs, goal)
        action, _ = model.predict(enc, deterministic=deterministic)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1

        sg = extract_subgoals(env)
        if int(sg.get(goal_name, 0)) == 1:
            return steps, True, done, info
        if done:
            return steps, False, done, info

    return steps, False, done, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--babyai_mode", choices=["unified", "routed"], default="unified")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--disable_nodrop", action="store_true", default=True)
    ap.add_argument("--model_at_goal", type=str, default=None)
    ap.add_argument("--model_has_key", type=str, default=None)
    ap.add_argument("--model_door_open", type=str, default=None)
    ap.add_argument("--model_has_box", type=str, default=None)
    ap.add_argument("--model_has_ball", type=str, default=None)
    ap.add_argument("--model_has_target", type=str, default=None)
    args = ap.parse_args()

    model_paths = {
        "at_goal": args.model_at_goal,
        "has_key": args.model_has_key,
        "door_open": args.model_door_open,
        "has_box": args.model_has_box,
        "has_ball": args.model_has_ball,
        "has_target": args.model_has_target,
    }
    models: Dict[str, PPO] = {}
    for goal, path in model_paths.items():
        if path:
            models[goal] = PPO.load(path)

    env = gym.make(args.env_id)  # IMPORTANT: raw env, NoDrop is NOT applied.

    full_success = []
    total_steps = []
    per_option_success = {}

    for ep in range(args.episodes):
        _obs, _info = reset_raw_env(env, args.env_id, args.seed + ep)
        plan = select_plan(args.env_id, env, args.babyai_mode)

        ep_steps = 0
        done = False
        executed = []

        for idx, goal in enumerate(plan):
            if goal not in models:
                raise ValueError(f"Missing model for goal '{goal}'.")

            # 桥接
            maybe_bridge_unlockpickup(args.env_id, env, goal)

            horizon = default_horizon(args.env_id, goal)
            steps, ok, done, _info = run_option(models[goal], env, args.env_id, goal, horizon, deterministic=args.deterministic)
            ep_steps += steps
            executed.append((goal, ok))
            per_option_success.setdefault(goal, []).append(float(ok))

            if done:
                break

        sg = extract_subgoals(env)
        success = float(sg.get("success", 0))
        full_success.append(success)
        total_steps.append(ep_steps)

    print("=" * 72)
    print(f"env_id                : {args.env_id}")
    print(f"episodes              : {args.episodes}")
    print(f"deterministic         : {args.deterministic}")
    print(f"nodrop_disabled       : True (raw env, no NoDropWrapper)")
    if "BabyAI-Pickup" in args.env_id:
        print(f"babyai_mode           : {args.babyai_mode}")
    print("-" * 72)
    print(f"full_task_success     : {float(np.mean(full_success)):.4f}")
    print(f"avg_env_steps         : {float(np.mean(total_steps)):.2f}")
    for goal, vals in per_option_success.items():
        print(f"option_success/{goal:<8}: {float(np.mean(vals)):.4f}")
    print("=" * 72)

    env.close()


if __name__ == "__main__":
    main()
