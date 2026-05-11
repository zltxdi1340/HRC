# rl/train_sb3_gc_ppo.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

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
from rl.debug_info_callback import DebugInfoTensorboardCallback
from rl.gc_option_env import GCOptionEnv
from rl.small_cnn import SmallCNNCombined


def goal_pool_for_env(env_id: str) -> List[str]:
    if env_id == "MiniGrid-Empty-8x8-v0":
        return ["at_goal"]
    if env_id == "MiniGrid-FourRooms-v0":
        return ["at_goal"]
    if "GoToObject" in env_id:
        return ["near_target"]
    if env_id == "BabyAI-UnlockLocal-v0":
        return ["near_key", "has_key", "opened_door"]
    if env_id == "MiniGrid-DoorKey-6x6-v0":
        return ["near_key", "has_key", "opened_door", "at_goal"]
    if env_id == "MiniGrid-UnlockPickup-v0":
        return ["near_key", "has_key", "opened_door", "has_box"]
    if "Fetch" in env_id:
        return ["near_target", "has_target"]
    if env_id == "BabyAI-Pickup-v0":
        return ["near_target", "has_target"]
    if env_id == "BabyAI-KeyInBox-v0":
        return ["opened_box", "has_key", "opened_door"]
    return ["at_goal"]


def _split_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _unique_keep_order(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        x = to_new(x)
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def resolve_training_spec(args: argparse.Namespace) -> dict:
    edge_mode = bool(args.parent_goal.strip() or args.child_goal.strip() or args.module_name.strip())

    if args.goals.strip() and edge_mode:
        raise ValueError("Do not pass --goals together with edge args.")
    if edge_mode and not args.child_goal.strip():
        raise ValueError("Edge mode requires --child_goal.")

    if edge_mode:
        parent_goal: Optional[str] = to_new(args.parent_goal.strip()) if args.parent_goal.strip() else None
        child_goal: str = to_new(args.child_goal.strip())
        module_name: str = args.module_name.strip() or f"{child_goal}_module"

        if args.edge_goal_pool_mode == "child_only":
            goals = [child_goal]
        else:
            goals = [g for g in [parent_goal, child_goal] if g is not None]

        goals = _unique_keep_order(goals)
        return {
            "mode": "edge",
            "module_name": module_name,
            "parent_goal": parent_goal,
            "child_goal": child_goal,
            "goals": goals,
        }

    goals = goal_pool_for_env(args.env_id)
    if args.goals.strip():
        goals = _split_csv(args.goals)
    goals = _unique_keep_order(normalize_nodes(goals))
    return {
        "mode": "goal",
        "module_name": "",
        "parent_goal": None,
        "child_goal": None,
        "goals": goals,
    }


def build_run_name(args: argparse.Namespace, spec: dict) -> str:
    safe_env = args.env_id.replace(":", "_").replace("/", "_")
    if args.run_name.strip():
        return args.run_name.strip()
    if spec["mode"] == "edge":
        safe_module = (spec["module_name"] or "module").replace(":", "_").replace("/", "_")
        p = spec["parent_goal"] or "none"
        c = spec["child_goal"] or "none"
        return f"{safe_env}__{safe_module}__{p}-to-{c}__strict__h{args.option_horizon}__seed{args.seed}"
    safe_goals = "-".join(spec["goals"])
    return f"{safe_env}__{safe_goals}__h{args.option_horizon}__seed{args.seed}"


def write_module_metadata(run_dir: Path, args: argparse.Namespace, spec: dict, use_precondition_do: bool) -> None:
    payload = {
        "env_id": args.env_id,
        "training_mode": spec["mode"],
        "module_name": spec["module_name"],
        "parent_goal": spec["parent_goal"],
        "child_goal": spec["child_goal"],
        "goal_pool": spec["goals"],
        "edge_goal_pool_mode": args.edge_goal_pool_mode,
        "option_horizon": args.option_horizon,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "use_nodrop": bool(args.use_nodrop),
        "use_precondition_do": bool(use_precondition_do),
        "strict_parent_mode": bool(spec["mode"] == "edge"),
        "strict_parent_fail_fast": bool(not args.no_strict_parent_fail_fast),
        "strict_reset_max_tries": int(args.strict_reset_max_tries),
        "node_naming": "canonical_granularity_nodes",
    }
    (run_dir / "module_spec.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, required=True)
    ap.add_argument("--total_timesteps", type=int, default=300_000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--option_horizon", type=int, default=128)
    ap.add_argument("--logdir", type=str, default="runs_granularity")
    ap.add_argument("--run_name", type=str, default="", help="subdir under logdir")
    ap.add_argument("--goals", type=str, default="", help="comma-separated goals, e.g. has_key,opened_door")
    ap.add_argument("--module_name", type=str, default="", help="Edge module name")
    ap.add_argument("--parent_goal", type=str, default="", help="Parent condition for edge policy")
    ap.add_argument("--child_goal", type=str, default="", help="Child target for edge policy")
    ap.add_argument(
        "--edge_goal_pool_mode",
        type=str,
        default="child_only",
        choices=["parent_child", "child_only"],
        help="In edge mode, child_only is recommended for G1 edge modules.",
    )
    ap.add_argument("--strict_reset_max_tries", type=int, default=128)
    ap.add_argument("--no_strict_parent_fail_fast", action="store_true")
    ap.add_argument("--load_model", type=str, default="", help="path to a .zip checkpoint to resume from")
    ap.add_argument("--use_nodrop", action="store_true", help="Enable NoDropWrapper (default: off)")
    ap.add_argument("--no_precondition_do", action="store_true", help="Disable automatic precondition do(). Use for G0.")
    args = ap.parse_args()

    spec = resolve_training_spec(args)
    goals = spec["goals"]
    use_precondition_do = not args.no_precondition_do

    print("TRAINING_MODE:", spec["mode"])
    if spec["mode"] == "edge":
        print("MODULE_NAME:", spec["module_name"])
        print("PARENT_GOAL:", spec["parent_goal"])
        print("CHILD_GOAL:", spec["child_goal"])
        print("EDGE_GOAL_POOL_MODE:", args.edge_goal_pool_mode)
        print("STRICT_PARENT_MODE:", True)
        print("STRICT_PARENT_FAIL_FAST:", not args.no_strict_parent_fail_fast)
        print("STRICT_RESET_MAX_TRIES:", args.strict_reset_max_tries)
    print("GOALS:", goals)
    print("USE_PRECONDITION_DO:", use_precondition_do)

    run_name = build_run_name(args, spec)
    safe_env = args.env_id.replace(":", "_").replace("/", "_")
    run_dir = Path(args.logdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_module_metadata(run_dir, args, spec, use_precondition_do=use_precondition_do)

    def make_env():
        return GCOptionEnv(
            env_id=args.env_id,
            goal_pool=goals,
            option_horizon=args.option_horizon,
            seed=args.seed,
            use_nodrop=args.use_nodrop,
            use_precondition_do=use_precondition_do,
            parent_goal=spec["parent_goal"],
            child_goal=spec["child_goal"],
            strict_parent_mode=(spec["mode"] == "edge"),
            strict_parent_fail_fast=(not args.no_strict_parent_fail_fast),
            reset_max_tries=args.strict_reset_max_tries,
        )

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)

    ckpt_every = 10_000
    save_freq = max(1, ckpt_every // args.n_envs)
    ckpt = CheckpointCallback(save_freq=save_freq, save_path=str(run_dir), name_prefix="ckpt")
    print(f"[ckpt] every ~{save_freq} calls (~{save_freq * args.n_envs} timesteps)")
    goal_cb = GoalSuccessCallback(window=200)

    debug_keys = [
        "debug_ever_seen_door",
        "debug_ever_front_of_door",
        "debug_toggle_attempted",
        "debug_toggle_attempt_front_of_door",
        "debug_toggle_attempt_count",
        "debug_seen_door_this_step",
        "debug_front_of_door_this_step",
        "debug_ever_target_seen",
        "debug_ever_target_front",
        "debug_pickup_attempted",
        "debug_pickup_attempt_front",
        "debug_pickup_attempt_count",
        "strict_parent_mode",
        "strict_target_locked_door_exists",
        "strict_parent_failed",
    ]
    debug_cb = DebugInfoTensorboardCallback(keys=debug_keys, ep_window=200, prefix="debug")

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

    model.save(str(run_dir / f"{safe_env}_final.zip"))
    vec_env.close()


if __name__ == "__main__":
    main()
