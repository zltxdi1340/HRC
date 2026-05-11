import argparse
import io
import contextlib
import numpy as np
from stable_baselines3 import PPO

from rl.gc_option_env import GCOptionEnv
from env.subgoals import extract_subgoals, parse_pickup_target


def _silent_reset(base_env, seed=None, options=None):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return base_env.reset(seed=seed, options=options)


def reset_for_eval(env: GCOptionEnv, *, seed=None, options=None, babyai_max_filter_tries=12):
    """
    Eval-only reset that keeps NoDrop disabled.
    It mirrors the user's current eval script, but does not wrap the env with NoDropWrapper.
    """
    base_env = env.env

    if "BabyAI-Pickup" in env.env_id:
        obs, info = _silent_reset(base_env, seed=seed, options=options)
    else:
        obs, info = base_env.reset(seed=seed, options=options)

    env._t = 0
    env._goal = env._sample_goal()
    reset_ok = True

    if "BabyAI-Pickup" in env.env_id:
        desired = env._desired_obj_for_goal(env._goal.node)
        if desired is not None:
            matched = False
            obj, _color = parse_pickup_target(base_env)
            if obj == desired:
                matched = True
            else:
                for _ in range(max(0, babyai_max_filter_tries - 1)):
                    obs, info = _silent_reset(base_env, options=options)
                    obj, _color = parse_pickup_target(base_env)
                    if obj == desired:
                        matched = True
                        break
            reset_ok = matched

    env._apply_preconditions()
    sg = extract_subgoals(base_env)
    env._prev_goal_val = int(sg.get(env._goal.node, 0))

    obs = base_env.unwrapped.gen_obs() if hasattr(base_env.unwrapped, "gen_obs") else obs

    info = dict(info)
    info["goal_node"] = env._goal.node
    info["reset_ok"] = reset_ok
    return env._encode_obs(obs), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--env_id", type=str, required=True)
    ap.add_argument("--goal", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--option_horizon", type=int, default=128)
    ap.add_argument("--babyai_max_filter_tries", type=int, default=12)
    ap.add_argument("--deterministic", action="store_true", default=True)
    args = ap.parse_args()

    env = GCOptionEnv(
        env_id=args.env_id,
        goal_pool=[args.goal],
        option_horizon=args.option_horizon,
        seed=args.seed,
        use_nodrop=False,  # IMPORTANT: NoDrop disabled for eval
        use_precondition_do=True,
    )

    model = PPO.load(args.model_path)

    success_list = []
    steps_list = []
    seen_list = []
    front_list = []
    pickup_attempted_list = []
    pickup_front_list = []

    valid_eps = 0
    reset_calls = 0
    max_reset_calls = args.episodes * 10

    while valid_eps < args.episodes and reset_calls < max_reset_calls:
        obs, info = reset_for_eval(
            env,
            seed=args.seed + reset_calls,
            babyai_max_filter_tries=args.babyai_max_filter_tries,
        )
        reset_calls += 1
        if not info.get("reset_ok", True):
            continue

        done = False
        steps = 0
        final_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            final_info = info

        success_list.append(float(final_info.get("is_success", False)))
        steps_list.append(steps)
        seen_list.append(float(final_info.get("debug_ever_target_seen", 0.0)))
        front_list.append(float(final_info.get("debug_ever_target_front", 0.0)))
        pickup_attempted_list.append(float(final_info.get("debug_pickup_attempted", 0.0)))
        pickup_front_list.append(float(final_info.get("debug_pickup_attempt_front", 0.0)))
        valid_eps += 1

    success_rate = float(np.mean(success_list)) if success_list else 0.0
    avg_steps = float(np.mean(steps_list)) if steps_list else 0.0
    seen_rate = float(np.mean(seen_list)) if seen_list else 0.0
    front_rate = float(np.mean(front_list)) if front_list else 0.0
    pickup_attempted_rate = float(np.mean(pickup_attempted_list)) if pickup_attempted_list else 0.0
    pickup_front_rate = float(np.mean(pickup_front_list)) if pickup_front_list else 0.0

    seen_and_success = [
        1.0 if (s > 0.5 and suc > 0.5) else 0.0
        for s, suc in zip(seen_list, success_list)
    ]
    success_given_seen = (
        float(np.sum(seen_and_success) / max(1, np.sum(seen_list)))
        if np.sum(seen_list) > 0
        else 0.0
    )

    print("=" * 60)
    print(f"env_id               : {args.env_id}")
    print(f"goal                 : {args.goal}")
    print(f"episodes_requested   : {args.episodes}")
    print(f"episodes_valid       : {valid_eps}")
    print(f"reset_calls          : {reset_calls}")
    print(f"model                : {args.model_path}")
    print(f"nodrop_disabled      : True")
    print("-" * 60)
    print(f"success_rate         : {success_rate:.4f}")
    print(f"avg_steps            : {avg_steps:.2f}")
    print(f"seen_rate            : {seen_rate:.4f}")
    print(f"front_rate           : {front_rate:.4f}")
    print(f"pickup_attempted     : {pickup_attempted_rate:.4f}")
    print(f"pickup_attempt_front : {pickup_front_rate:.4f}")
    print(f"success_given_seen   : {success_given_seen:.4f}")
    if valid_eps < args.episodes:
        print(f"warning              : only collected {valid_eps}/{args.episodes} valid episodes")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
