# interventions/sampling.py
import contextlib
import json
import os
import random
from typing import Dict, List, Optional, Set

from env.subgoals import extract_subgoals, parse_target_from_mission
from interventions.do_ops import (
    do_have,
    do_door_open,
    do_near_obj,
    do_open_box,
    do_have_matching_key_for_locked_door,
)

# Canonical node set for the granularity-ablation branch.
# These are the only keys written into DI records.
TRACK_KEYS = [
    "near_key",
    "has_key",
    "near_target",
    "has_target",
    "opened_box",
    "opened_door",
    "has_box",
    "has_ball",
    "at_goal",
    "task_success",
]

# Only these nodes are allowed as external t0 interventions.
# Do NOT intervene on at_goal or task_success.
DOABLE = {
    "near_key",
    "near_target",
    "opened_box",
    "has_key",
    "has_target",
    "has_box",
    "has_ball",
    "opened_door",
}


@contextlib.contextmanager
def _silence_if_babyai(env_id: str):
    if "BabyAI" not in env_id:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _project_vars(sg: Dict) -> Dict[str, int]:
    """
    Keep only canonical node names in sampled DI records.
    Missing nodes are filled with 0 to avoid KeyError downstream.
    """
    return {k: int(sg.get(k, 0)) for k in TRACK_KEYS}


def _sample_action(env, rng: random.Random, env_id: str) -> int:
    """
    Lightweight exploration bias, not a learned policy.
    Common MiniGrid actions: 0 left, 1 right, 2 forward, 3 pickup,
    4 drop, 5 toggle, 6 done.
    """
    if "DoorKey" in env_id or "UnlockPickup" in env_id:
        r = rng.random()
        if r < 0.45:
            return 2  # forward
        if r < 0.70:
            return 5  # toggle
        if r < 0.85:
            return rng.choice([0, 1])
        return env.action_space.sample()

    if "Pickup" in env_id:
        r = rng.random()
        if r < 0.35:
            return 3  # pickup
        if r < 0.65:
            return 2  # forward
        if r < 0.85:
            return rng.choice([0, 1])
        return env.action_space.sample()

    if "GoToObject" in env_id:
        r = rng.random()
        if r < 0.50:
            return 2  # forward
        if r < 0.80:
            return rng.choice([0, 1])
        return env.action_space.sample()

    if "Fetch" in env_id:
        r = rng.random()
        if r < 0.30:
            return 3  # pickup
        if r < 0.65:
            return 2  # forward
        if r < 0.85:
            return rng.choice([0, 1])
        return env.action_space.sample()

    if "KeyInBox" in env_id:
        r = rng.random()
        if r < 0.30:
            return 5  # toggle
        if r < 0.60:
            return 2  # forward
        if r < 0.85:
            return rng.choice([0, 1])
        return env.action_space.sample()

    return env.action_space.sample()


def _do_subgoal(env, gi: str) -> bool:
    """
    Apply an interpretable t0 intervention for one canonical node.
    This is only used by Stage 1 causal discovery, not by policy training.
    """
    if gi == "near_key":
        return bool(do_near_obj(env, "key", color=None))

    if gi == "near_target":
        obj, color = parse_target_from_mission(env)
        if obj is None:
            return False
        return bool(do_near_obj(env, obj, color=color))

    if gi == "opened_box":
        return bool(do_open_box(env))

    if gi == "has_key":
        # For locked-door tasks, prefer the key matching the target locked door.
        try:
            if bool(do_have_matching_key_for_locked_door(env)):
                return True
        except Exception:
            pass
        return bool(do_have(env, "key"))

    if gi == "has_target":
        obj, color = parse_target_from_mission(env)
        if obj is None:
            return False
        return bool(do_have(env, obj, color=color))

    if gi == "has_box":
        return bool(do_have(env, "box"))

    if gi == "has_ball":
        return bool(do_have(env, "ball"))

    if gi == "opened_door":
        # In locked-door tasks, first give the matching key, then open the door.
        try:
            do_have_matching_key_for_locked_door(env)
        except Exception:
            pass
        return bool(do_door_open(env, open_=True))

    return False


def _rollout_window(env, rng: random.Random, env_id: str, delta: int, H: int, max_vars: Dict[str, int]) -> None:
    steps = 0
    for _ in range(delta):
        action = _sample_action(env, rng, env_id)
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
        cur = _project_vars(extract_subgoals(env))
        for k in TRACK_KEYS:
            max_vars[k] = int(max_vars[k] or cur.get(k, 0))
        if terminated or truncated or steps >= H:
            break


def intervention_sampling(
    env,
    IS: Set[str],
    T: int,
    H: int,
    delta: int,
    seed: int,
    out_jsonl_path: Optional[str] = None,
) -> List[Dict]:
    """
    Generate DI records. For each trajectory seed, collect:
      - baseline: intervened='none'
      - intervention: for each doable gi in IS, reset to the same seed and do(gi=1) at t0.

    Every DI variable dict uses the canonical node names in TRACK_KEYS.
    """
    env_id = getattr(env, "spec", None).id if getattr(env, "spec", None) else "unknown"
    rng = random.Random(seed)

    DI: List[Dict] = []
    writer = None
    if out_jsonl_path:
        out_dir = os.path.dirname(out_jsonl_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        writer = open(out_jsonl_path, "w", encoding="utf-8")

    def _emit(rec: Dict):
        DI.append(rec)
        if writer is not None:
            writer.write(json.dumps(rec, ensure_ascii=False) + "\n")

    try:
        for traj in range(T):
            traj_seed = seed + 1000003 * (traj + 1)

            # ---------------- baseline ----------------
            with _silence_if_babyai(env_id):
                env.reset(seed=traj_seed)
            mission = getattr(env.unwrapped, "mission", None)

            before0 = _project_vars(extract_subgoals(env))
            max0 = dict(before0)
            _rollout_window(env, rng, env_id, delta, H, max0)

            _emit({
                "env_id": env_id,
                "mission": mission,
                "traj": int(traj),
                "IS": sorted(list(IS)),
                "intervened": "none",
                "did_intervene": False,
                "vars_before": dict(before0),
                "vars_after_int": dict(before0),
                "vars_max_window": dict(max0),
                "delta": int(delta),
                "seed": int(traj_seed),
            })

            # ---------------- interventions at t0 ----------------
            doable = [g for g in sorted(IS) if g in DOABLE]
            for gi in doable:
                with _silence_if_babyai(env_id):
                    env.reset(seed=traj_seed)
                mission = getattr(env.unwrapped, "mission", None)

                before1 = _project_vars(extract_subgoals(env))
                ok = _do_subgoal(env, gi)
                after1 = _project_vars(extract_subgoals(env))
                max1 = dict(after1)
                _rollout_window(env, rng, env_id, delta, H, max1)

                _emit({
                    "env_id": env_id,
                    "mission": mission,
                    "traj": int(traj),
                    "IS": sorted(list(IS)),
                    "intervened": gi,
                    "did_intervene": bool(ok),
                    "vars_before": dict(before1),
                    "vars_after_int": dict(after1),
                    "vars_max_window": dict(max1),
                    "delta": int(delta),
                    "seed": int(traj_seed),
                })
    finally:
        if writer is not None:
            writer.close()

    return DI
