import gymnasium as gym
import minigrid  # noqa
from rl.gc_option_env import GCOptionEnv
from env.subgoals import extract_subgoals

def describe_door_and_box(env):
    u = env.unwrapped
    doors = []
    boxes = []
    for obj in u.grid.grid:
        if obj is None:
            continue
        t = getattr(obj, "type", None)
        if t == "door":
            doors.append({
                "color": getattr(obj, "color", None),
                "is_open": getattr(obj, "is_open", None),
                "is_locked": getattr(obj, "is_locked", None),
            })
        elif t == "box":
            contains = getattr(obj, "contains", None)
            boxes.append({
                "color": getattr(obj, "color", None),
                "is_open": getattr(obj, "is_open", None),
                "contains_type": getattr(contains, "type", None) if contains is not None else None,
                "contains_color": getattr(contains, "color", None) if contains is not None else None,
            })
    return doors, boxes

env = GCOptionEnv(
    env_id="BabyAI-KeyInBox-v0",
    goal_pool=["door_open"],
    option_horizon=256,
    seed=0,
    use_precondition_do=True,
    parent_goal="has_key",
    child_goal="door_open",
    strict_parent_mode=True,
    strict_parent_fail_fast=True,
    reset_max_tries=128,
)

for i in range(20):
    obs, info = env.reset()
    sg = extract_subgoals(env.env)
    carrying = getattr(env.env.unwrapped, "carrying", None)
    doors, boxes = describe_door_and_box(env.env)

    print(f"\n=== episode {i} ===")
    print("subgoals:", {k: sg.get(k, None) for k in ["open_box", "has_key", "door_open"]})
    print("carrying:", None if carrying is None else {
        "type": getattr(carrying, "type", None),
        "color": getattr(carrying, "color", None),
    })
    print("doors:", doors)
    print("boxes:", boxes)