import argparse
import gymnasium as gym

# Register MiniGrid / BabyAI envs.
try:
    import minigrid
    import minigrid.envs
except Exception as e:
    print("[WARN] failed to import minigrid env registry:", repr(e))

try:
    import minigrid.envs.babyai
except Exception as e:
    print("[WARN] failed to import minigrid babyai env registry:", repr(e))

from env.subgoals import extract_subgoals


EXPECTED_KEYS = {
    "near_key",
    "has_key",
    "near_target",
    "has_target",
    "opened_box",
    "opened_door",
    "has_box",
    "at_goal",
    "task_success",
}

OLD_KEYS = {
    "near_obj",
    "has_obj",
    "open_box",
    "door_open",
    "success",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, required=True)
    args = parser.parse_args()

    env = gym.make(args.env_id)
    obs, info = env.reset(seed=0)

    sg = extract_subgoals(env)
    keys = set(sg.keys())

    print("env_id:", args.env_id)
    print("mission:", getattr(env.unwrapped, "mission", None))

    print("\nsubgoals:")
    for k, v in sg.items():
        if not k.startswith("_"):
            print(f"  {k}: {v}")

    missing = EXPECTED_KEYS - keys
    old = OLD_KEYS & keys

    print("\ncheck:")
    print("  missing expected keys:", sorted(missing))
    print("  old keys still present:", sorted(old))

    env.close()


if __name__ == "__main__":
    main()