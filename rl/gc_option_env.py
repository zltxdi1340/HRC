# rl/gc_option_env.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import gymnasium as gym
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

from env.wrappers import NoDropWrapper
from interventions.do_ops import (
    do_have,
    do_door_open,
    do_open_box,
    do_near_obj,
    do_have_matching_key_for_locked_door,
    choose_target_locked_door,
)
from env.subgoals import extract_subgoals, parse_target_from_mission
from env.node_alias import normalize_nodes, to_new

NODE_NAMES = [
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
NODE2ID = {n: i for i, n in enumerate(NODE_NAMES)}

COLOR_NAMES = ["red", "green", "blue", "purple", "yellow", "grey"]
COLOR2ID = {c: i for i, c in enumerate(COLOR_NAMES)}
OBJ_NAMES = ["none", "key", "ball", "box"]
OBJ2ID = {o: i for i, o in enumerate(OBJ_NAMES)}

MISSION_RE = re.compile(r"pick up the (\w+) (\w+)", re.IGNORECASE)


@dataclass
class Goal:
    node: str
    target_obj: str = "none"
    target_color: str = "grey"


def _goal_vector(goal: Goal) -> np.ndarray:
    node = np.zeros(len(NODE_NAMES), dtype=np.float32)
    node[NODE2ID[goal.node]] = 1.0

    obj = np.zeros(len(OBJ_NAMES), dtype=np.float32)
    obj[OBJ2ID.get(goal.target_obj, 0)] = 1.0

    col = np.zeros(len(COLOR_NAMES), dtype=np.float32)
    col[COLOR2ID.get(goal.target_color, COLOR2ID["grey"])] = 1.0

    return np.concatenate([node, obj, col], axis=0)


class GCOptionEnv(gym.Wrapper):
    def __init__(
        self,
        env_id: str,
        goal_pool: List[str],
        option_horizon: int = 128,
        seed: int = 0,
        use_nodrop: bool = False,
        use_precondition_do: bool = True,
        parent_goal: Optional[str] = None,
        child_goal: Optional[str] = None,
        strict_parent_mode: bool = False,
        strict_parent_fail_fast: bool = True,
        reset_max_tries: int = 64,
    ):
        base = gym.make(env_id)
        if use_nodrop:
            base = NoDropWrapper(base)
        super().__init__(base)

        self.env_id = env_id
        self.goal_pool = normalize_nodes(goal_pool)
        self.option_horizon = option_horizon
        self._front_pickup_bonus_given = False
        self.rng = np.random.default_rng(seed)
        self.use_precondition_do = use_precondition_do
        self.parent_goal = to_new(parent_goal) if parent_goal else None
        self.child_goal = to_new(child_goal) if child_goal else None
        self.strict_parent_mode = bool(strict_parent_mode)
        self.strict_parent_fail_fast = bool(strict_parent_fail_fast)
        self.reset_max_tries = int(reset_max_tries)
        actions_enum = getattr(self.env.unwrapped, "actions", None)
        pickup_enum = getattr(actions_enum, "pickup", None)
        toggle_enum = getattr(actions_enum, "toggle", None)
        self._pickup_action_id = int(pickup_enum) if pickup_enum is not None else 3
        self._toggle_action_id = int(toggle_enum) if toggle_enum is not None else 5

        self._ever_target_seen = False
        self._ever_target_front = False
        self._pickup_attempts = 0
        self._pickup_attempts_when_front = 0

        # door-debug metrics
        self._ever_seen_door = False
        self._ever_front_of_door = False
        self._toggle_attempts = 0
        self._toggle_attempts_front_of_door = 0

        obs, _ = self.env.reset(seed=seed)
        img = obs["image"] if isinstance(obs, dict) else obs
        H, W, C = img.shape

        self._goal_dim = len(NODE_NAMES) + len(OBJ_NAMES) + len(COLOR_NAMES)

        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(C, H, W), dtype=np.uint8),
            "direction": gym.spaces.Discrete(4),
            "goal": gym.spaces.Box(low=0.0, high=1.0, shape=(self._goal_dim,), dtype=np.float32),
        })
        self.action_space = self.env.action_space

        self._t = 0
        self._goal: Optional[Goal] = None
        self._prev_goal_val = 0
        self._strict_target_door = None

    def _encode_obs(self, obs: Any) -> Dict[str, Any]:
        if isinstance(obs, dict):
            img = obs["image"]
            direction = int(obs.get("direction", 0))
        else:
            img = obs
            direction = 0

        img_chw = np.transpose(img, (2, 0, 1)).astype(np.uint8)
        gvec = _goal_vector(self._goal).astype(np.float32)
        return {"image": img_chw, "direction": direction, "goal": gvec}

    def _sample_goal(self) -> Goal:
        node = to_new(self.rng.choice(self.goal_pool).item())
        target_obj = "none"
        target_color = "grey"

        if node in ("near_target", "has_target"):
            obj, color = parse_target_from_mission(self.env)
            if obj is not None:
                target_obj = obj
            if color is not None:
                target_color = color

        return Goal(node=node, target_obj=target_obj, target_color=target_color)

    def _apply_preconditions(self):
        if not self.use_precondition_do or self._goal is None:
            return

        g = to_new(self._goal.node)

        if "BabyAI-KeyInBox" in self.env_id and g == "has_key":
            do_open_box(self.env)

        if "MiniGrid-UnlockPickup" in self.env_id and g == "has_box":
            do_have_matching_key_for_locked_door(self.env)
            do_door_open(self.env, open_=True)
            self.env.unwrapped.carrying = None

        if g == "opened_door":
            do_have_matching_key_for_locked_door(self.env)

        if g == "at_goal":
            do_door_open(self.env, open_=True)

    @staticmethod
    def _desired_obj_for_goal(node: str):
        node = to_new(node)
        return {
            "near_key": "key",
            "has_key": "key",
            "has_box": "box",
            "has_ball": "ball",
        }.get(node, None)

    def _target_obj_name(self) -> Optional[str]:
        if self._goal is None:
            return None
        if self._goal.target_obj != "none":
            return self._goal.target_obj
        return self._desired_obj_for_goal(self._goal.node)

    def _target_seen_in_obs(self, obs: Any) -> bool:
        target_obj = self._target_obj_name()
        if target_obj is None:
            return False
        if not isinstance(obs, dict) or "image" not in obs:
            return False
        img = obs["image"]
        obj_ids = img[..., 0]
        target_id = OBJECT_TO_IDX.get(target_obj, None)
        if target_id is None:
            return False
        return bool((obj_ids == target_id).any())

    def _target_positions(self) -> List[Tuple[int, int]]:
        target_obj = self._target_obj_name()
        if target_obj is None:
            return []
        target_color = getattr(self._goal, "target_color", None)
        u = self.env.unwrapped
        grid = u.grid
        out = []
        for idx, obj in enumerate(grid.grid):
            if obj is None:
                continue
            if getattr(obj, "type", None) != target_obj:
                continue
            if target_color not in (None, "grey") and getattr(obj, "color", None) != target_color:
                continue
            x = int(idx % grid.width)
            y = int(idx // grid.width)
            out.append((x, y))
        return out

    def _target_in_front(self) -> bool:
        target_positions = self._target_positions()
        if not target_positions:
            return False
        u = self.env.unwrapped
        if not hasattr(u, "dir_vec"):
            return False
        fx = int(u.agent_pos[0] + u.dir_vec[0])
        fy = int(u.agent_pos[1] + u.dir_vec[1])
        return (fx, fy) in target_positions

    def _current_target_door(self):
        if self._strict_target_door is not None:
            return self._strict_target_door
        return choose_target_locked_door(self.env)

    def _door_seen_in_obs(self, obs: Any) -> bool:
        if not isinstance(obs, dict) or "image" not in obs:
            return False
        target = self._current_target_door()
        img = obs["image"]
        obj_ids = img[..., 0]
        door_id = OBJECT_TO_IDX.get("door", None)
        if door_id is None:
            return False
        door_mask = (obj_ids == door_id)
        if not bool(door_mask.any()):
            return False
        if target is None:
            return True
        _x, _y, door = target
        door_color = getattr(door, "color", None)
        color_id = COLOR_TO_IDX.get(door_color, None)
        if color_id is None:
            return True
        color_ids = img[..., 1]
        return bool((door_mask & (color_ids == color_id)).any())

    def _door_in_front(self) -> bool:
        target = self._current_target_door()
        if target is None:
            return False
        x, y, _door = target
        u = self.env.unwrapped
        if not hasattr(u, "dir_vec"):
            return False
        fx = int(u.agent_pos[0] + u.dir_vec[0])
        fy = int(u.agent_pos[1] + u.dir_vec[1])
        return (fx, fy) == (int(x), int(y))

    def _filter_babyaipickup_mission(self, obs, info, options):
        if "BabyAI-Pickup" not in self.env_id or self._goal is None:
            return obs, info

        desired_obj = self._target_obj_name()
        desired_color = getattr(self._goal, "target_color", None)

        # 统一把默认占位色 grey 当成“无颜色约束”
        if desired_color == "grey":
            desired_color = None

        if desired_obj is None:
            return obs, info

        for _ in range(50):
            obj, color = parse_target_from_mission(self.env)
            obj_ok = (obj == desired_obj)
            color_ok = (desired_color is None) or (color == desired_color)
            if obj_ok and color_ok:
                break
            obs, info = self.env.reset(options=options)

        return obs, info

    def _force_parent_condition(self) -> bool:
        pg = to_new(self.parent_goal) if self.parent_goal else None
        cg = to_new(self.child_goal) if self.child_goal else None

        if not pg:
            return True

        if pg == "near_key":
            return bool(do_near_obj(self.env, "key", color=None))

        if pg == "has_key" and cg == "opened_door":
            return bool(do_have_matching_key_for_locked_door(self.env))

        if pg == "has_key":
            return bool(do_have(self.env, "key"))

        if pg == "opened_box":
            return bool(do_open_box(self.env))

        if pg == "opened_door":
            return bool(do_door_open(self.env, open_=True))

        if pg == "near_target":
            obj, color = parse_target_from_mission(self.env)
            if obj is None:
                obj = self._desired_obj_for_goal(cg or "")
            if obj is None:
                return False
            return bool(do_near_obj(self.env, obj, color=color))

        if pg == "has_target":
            obj, color = parse_target_from_mission(self.env)
            if obj is None:
                obj = self._desired_obj_for_goal(cg or "") or "box"
                color = None
            return bool(do_have(self.env, obj, color=color))

        if pg == "has_box":
            return bool(do_have(self.env, "box"))

        if pg == "has_ball":
            return bool(do_have(self.env, "ball"))

        if pg == "at_goal":
            return bool(do_door_open(self.env, open_=True))

        return True

    def _strict_reset(self, *, seed: Optional[int], options: Optional[dict]):
        last_obs = None
        last_info = {}
        for _ in range(max(1, self.reset_max_tries)):
            obs, info = self.env.reset(seed=seed, options=options)
            obs, info = self._filter_babyaipickup_mission(obs, info, options)
            self._strict_target_door = choose_target_locked_door(self.env)
            ok = self._force_parent_condition()
            sg = extract_subgoals(self.env)
            parent_ok = (self.parent_goal is None) or (int(sg.get(self.parent_goal, 0)) == 1)
            child_ok = True
            if self.child_goal is not None:
                child_ok = int(sg.get(self.child_goal, 0)) == 0
            if ok and parent_ok and child_ok:
                return obs, info, sg
            last_obs, last_info = obs, info
        if last_obs is None:
            last_obs, last_info = self.env.reset(seed=seed, options=options)
        sg = extract_subgoals(self.env)
        return last_obs, last_info, sg

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._t = 0
        self._goal = self._sample_goal()

        if self.strict_parent_mode and self.parent_goal and self.child_goal:
            obs, info, sg = self._strict_reset(seed=seed, options=options)
        else:
            obs, info = self.env.reset(seed=seed, options=options)
            obs, info = self._filter_babyaipickup_mission(obs, info, options)
            self._apply_preconditions()
            self._strict_target_door = choose_target_locked_door(self.env)
            sg = extract_subgoals(self.env)

        self._prev_goal_val = int(sg.get(self._goal.node, 0))
        obs = self.env.unwrapped.gen_obs() if hasattr(self.env.unwrapped, "gen_obs") else obs

        self._ever_target_seen = False
        self._ever_target_front = False
        self._front_pickup_bonus_given = False
        self._pickup_attempts = 0
        self._pickup_attempts_when_front = 0
        self._ever_seen_door = False
        self._ever_front_of_door = False
        self._toggle_attempts = 0
        self._toggle_attempts_front_of_door = 0

        if self._target_seen_in_obs(obs):
            self._ever_target_seen = True
        if self._target_in_front():
            self._ever_target_front = True
        if self._door_seen_in_obs(obs):
            self._ever_seen_door = True
        if self._door_in_front():
            self._ever_front_of_door = True

        info = dict(info)
        info["strict_parent_mode"] = float(self.strict_parent_mode)
        info["strict_parent_goal"] = self.parent_goal
        info["strict_child_goal"] = self.child_goal
        info["strict_target_locked_door_exists"] = float(self._strict_target_door is not None)
        if self._strict_target_door is not None:
            x, y, door = self._strict_target_door
            info["strict_target_locked_door_x"] = int(x)
            info["strict_target_locked_door_y"] = int(y)
            info["strict_target_locked_door_color"] = getattr(door, "color", None)
        return self._encode_obs(obs), info

    def step(self, action):
        pre_target_front = self._target_in_front()
        pre_door_front = self._door_in_front()
        obs, _, terminated, truncated, info = self.env.step(action)
        post_target_front = self._target_in_front()
        post_door_front = self._door_in_front()

        if int(action) == self._pickup_action_id:
            self._pickup_attempts += 1
            if pre_target_front or post_target_front:
                self._pickup_attempts_when_front += 1

        if int(action) == self._toggle_action_id:
            self._toggle_attempts += 1
            if pre_door_front or post_door_front:
                self._toggle_attempts_front_of_door += 1

        sg = extract_subgoals(self.env)
        goal_val = int(sg.get(self._goal.node, 0))
        is_success = (self._prev_goal_val == 0 and goal_val == 1)
        reward = 1.0 if is_success else 0.0
        self._prev_goal_val = goal_val
        self._t += 1

        if self.strict_parent_mode and self.parent_goal and self.strict_parent_fail_fast:
            if int(sg.get(self.parent_goal, 0)) == 0 and not is_success:
                truncated = True
                info = dict(info)
                info["strict_parent_failed"] = 1.0

        if is_success:
            terminated = True
        if self._t >= self.option_horizon:
            truncated = True

        if self._target_seen_in_obs(obs):
            self._ever_target_seen = True
        if pre_target_front or post_target_front:
            self._ever_target_front = True
        if self._door_seen_in_obs(obs):
            self._ever_seen_door = True
        if pre_door_front or post_door_front:
            self._ever_front_of_door = True

        info = dict(info)
        info["subgoals"] = sg
        info["goal"] = self._goal.__dict__ if self._goal else None
        info["goal_node"] = self._goal.node
        info["is_success"] = is_success

        # existing pickup diagnostics
        info["debug_ever_target_seen"] = float(self._ever_target_seen)
        info["debug_ever_target_front"] = float(self._ever_target_front)
        info["debug_pickup_attempted"] = float(self._pickup_attempts > 0)
        info["debug_pickup_attempt_front"] = float(self._pickup_attempts_when_front > 0)
        info["debug_pickup_attempt_count"] = int(self._pickup_attempts)

        # new door diagnostics
        info["debug_ever_seen_door"] = float(self._ever_seen_door)
        info["debug_ever_front_of_door"] = float(self._ever_front_of_door)
        info["debug_toggle_attempted"] = float(self._toggle_attempts > 0)
        info["debug_toggle_attempt_front_of_door"] = float(self._toggle_attempts_front_of_door > 0)
        info["debug_toggle_attempt_count"] = int(self._toggle_attempts)
        info["debug_seen_door_this_step"] = float(self._door_seen_in_obs(obs))
        info["debug_front_of_door_this_step"] = float(post_door_front)

        return self._encode_obs(obs), reward, terminated, truncated, info
