# train_sb3_gc_ppo_multitask.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from env.node_alias import normalize_nodes, to_new
from rl.callbacks import GoalSuccessCallback
from rl.small_cnn import SmallCNNCombined
from multitask_gc_option_env import MultiTaskGCOptionEnv


def _split_csv(text: str):
    return [x.strip() for x in text.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_ids", type=str, required=True, help="Comma-separated env ids")
    ap.add_argument("--module_name", type=str, default="")
    ap.add_argument("--parent_goal", type=str, default="")
    ap.add_argument("--child_goal", type=str, default="")
    ap.add_argument("--goals", type=str, default="opened_door")
    ap.add_argument("--total_timesteps", type=int, default=300000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--option_horizon", type=int, default=256)
    ap.add_argument("--strict_reset_max_tries", type=int, default=128)
    ap.add_argument("--no_strict_parent_fail_fast", action="store_true")
    ap.add_argument("--use_nodrop", action="store_true")
    ap.add_argument("--no_precondition_do", action="store_true", help="Disable automatic precondition do(). Use for G0.")
    ap.add_argument("--task_sampling", type=str, default="uniform", choices=["uniform", "round_robin"])
    ap.add_argument("--logdir", type=str, default="runs_granularity")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--load_model", type=str, default="")
    args = ap.parse_args()

    env_ids = [x.strip() for x in args.env_ids.split(",") if x.strip()]
    goals = normalize_nodes(_split_csv(args.goals))
    if not env_ids:
        raise ValueError("--env_ids must be non-empty")
    if not goals:
        raise ValueError("--goals must be non-empty")

    parent_goal = to_new(args.parent_goal.strip()) if args.parent_goal.strip() else None
    child_goal = to_new(args.child_goal.strip()) if args.child_goal.strip() else None
    strict_parent_mode = bool(parent_goal and child_goal)
    strict_parent_fail_fast = False if args.no_strict_parent_fail_fast else strict_parent_mode
    use_precondition_do = not args.no_precondition_do

    safe_envs = "__".join([e.replace(":", "_").replace("/", "_") for e in env_ids])
    run_name = args.run_name.strip() or f"multitask__{safe_envs}__{args.module_name or '-'.join(goals)}__seed{args.seed}"
    run_dir = Path(args.logdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "env_ids": env_ids,
        "module_name": args.module_name,
        "parent_goal": parent_goal,
        "child_goal": child_goal,
        "goal_pool": goals,
        "option_horizon": args.option_horizon,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "use_nodrop": bool(args.use_nodrop),
        "use_precondition_do": bool(use_precondition_do),
        "strict_parent_mode": bool(strict_parent_mode),
        "strict_parent_fail_fast": bool(strict_parent_fail_fast),
        "strict_reset_max_tries": int(args.strict_reset_max_tries),
        "task_sampling": args.task_sampling,
        "node_naming": "canonical_granularity_nodes",
    }
    (run_dir / "module_spec.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("ENV_IDS:", env_ids)
    print("GOALS:", goals)
    print("PARENT_GOAL:", parent_goal)
    print("CHILD_GOAL:", child_goal)
    print("STRICT_PARENT_MODE:", strict_parent_mode)
    print("USE_PRECONDITION_DO:", use_precondition_do)

    def make_env():
        return MultiTaskGCOptionEnv(
            env_ids=env_ids,
            goal_pool=goals,
            option_horizon=args.option_horizon,
            seed=args.seed,
            use_nodrop=args.use_nodrop,
            use_precondition_do=use_precondition_do,
            parent_goal=parent_goal,
            child_goal=child_goal,
            strict_parent_mode=strict_parent_mode,
            strict_parent_fail_fast=strict_parent_fail_fast,
            reset_max_tries=args.strict_reset_max_tries,
            task_sampling=args.task_sampling,
        )

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)

    ckpt_every = 10_000
    save_freq = max(1, ckpt_every // args.n_envs)
    ckpt = CheckpointCallback(save_freq=save_freq, save_path=str(run_dir), name_prefix="ckpt")
    goal_cb = GoalSuccessCallback(window=200)

    if args.load_model.strip():
        model = PPO.load(args.load_model, env=vec_env, verbose=1, tensorboard_log=str(run_dir))
        reset_flag = False
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            verbose=1,
            tensorboard_log=str(run_dir),
            n_steps=256,
            batch_size=1024,
            learning_rate=5e-5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs=dict(
                features_extractor_class=SmallCNNCombined,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
        )
        reset_flag = True

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[ckpt, goal_cb],
        reset_num_timesteps=reset_flag,
        tb_log_name=run_name,
    )

    final_name = args.module_name.strip() or "multitask"
    model.save(str(run_dir / f"{final_name}_final.zip"))
    vec_env.close()


if __name__ == "__main__":
    main()
