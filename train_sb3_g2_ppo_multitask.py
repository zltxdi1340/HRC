# train_sb3_g2_ppo_multitask.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

from rl.callbacks import GoalSuccessCallback
from rl.debug_info_callback import DebugInfoTensorboardCallback
from rl.g2_template_env import TEMPLATE_NAMES
from rl.small_cnn import SmallCNNCombined
from multitask_g2_template_env import MultiTaskG2TemplateEnv


def _split_csv(text: str):
    return [x.strip() for x in text.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_ids", type=str, required=True, help="Comma-separated source env ids")
    ap.add_argument("--template", type=str, required=True, choices=TEMPLATE_NAMES)
    ap.add_argument("--parent_goal", type=str, default="")
    ap.add_argument("--total_timesteps", type=int, default=100000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--option_horizon", type=int, default=256)
    ap.add_argument("--strict_reset_max_tries", type=int, default=128)
    ap.add_argument("--task_sampling", type=str, default="uniform", choices=["uniform", "round_robin"])
    ap.add_argument("--use_nodrop", action="store_true")
    ap.add_argument("--no_precondition_do", action="store_true")
    ap.add_argument("--logdir", type=str, default="runs_joint_door")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--load_model", type=str, default="")
    args = ap.parse_args()

    env_ids = _split_csv(args.env_ids)
    if not env_ids:
        raise ValueError("--env_ids must be non-empty")

    use_precondition_do = not args.no_precondition_do
    parent_goal = args.parent_goal.strip() or None

    safe_envs = "__".join([e.replace(":", "_").replace("/", "_") for e in env_ids])
    run_name = args.run_name.strip() or f"C_JOINT_G2__{safe_envs}__{args.template}__seed{args.seed}"
    run_dir = Path(args.logdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "env_ids": env_ids,
        "granularity": "G2",
        "aggregation_method": "C_joint_multitask",
        "template": args.template,
        "parent_goal": parent_goal,
        "option_horizon": args.option_horizon,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "use_precondition_do": use_precondition_do,
        "use_nodrop": bool(args.use_nodrop),
        "task_sampling": args.task_sampling,
        "action_role_constrained": True,
    }
    (run_dir / "module_spec.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("ENV_IDS:", env_ids)
    print("G2_TEMPLATE:", args.template)
    print("PARENT_GOAL:", parent_goal)
    print("USE_PRECONDITION_DO:", use_precondition_do)
    print("TASK_SAMPLING:", args.task_sampling)
    print("RUN_DIR:", run_dir)

    def make_env():
        return MultiTaskG2TemplateEnv(
            env_ids=env_ids,
            template=args.template,
            option_horizon=args.option_horizon,
            seed=args.seed,
            use_nodrop=args.use_nodrop,
            use_precondition_do=use_precondition_do,
            parent_goal=parent_goal,
            reset_max_tries=args.strict_reset_max_tries,
            task_sampling=args.task_sampling,
        )

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)

    ckpt_every = 10_000
    save_freq = max(1, ckpt_every // args.n_envs)
    ckpt = CheckpointCallback(save_freq=save_freq, save_path=str(run_dir), name_prefix="ckpt")
    goal_cb = GoalSuccessCallback(window=200)

    debug_cb = DebugInfoTensorboardCallback(
        keys=[
            "debug_g2_template_success_once",
            "debug_g2_door_ready_once",
            "debug_g2_target_front_once",
            "debug_g2_pickup_attempt_count",
            "debug_g2_toggle_attempt_count",
            "g2_precondition_ok",
        ],
        ep_window=200,
        prefix="g2_debug",
        alias_map={
            "debug_g2_template_success_once": "tpl_success",
            "debug_g2_door_ready_once": "door_ready",
            "debug_g2_target_front_once": "target_front",
            "debug_g2_pickup_attempt_count": "pickup_cnt",
            "debug_g2_toggle_attempt_count": "toggle_cnt",
            "g2_precondition_ok": "precond_ok",
        },
    )

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
        callback=[ckpt, goal_cb, debug_cb],
        reset_num_timesteps=reset_flag,
        tb_log_name=run_name,
    )

    model.save(str(run_dir / f"{args.template}_final.zip"))
    vec_env.close()


if __name__ == "__main__":
    main()
