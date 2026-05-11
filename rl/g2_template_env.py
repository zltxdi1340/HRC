# rl/g2_template_env.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

from env.wrappers import NoDropWrapper
from env.subgoals import extract_subgoals, parse_target_from_mission
from env.node_alias import to_new
from interventions.do_ops import (
    do_have,
    do_door_open,
    do_near_obj,
    do_have_matching_key_for_locked_door,
    choose_target_door,
    choose_target_locked_door,
    do_open_box,
)

# External state nodes stay aligned with G0/G1.
EXTERNAL_NODE_NAMES = [
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

# G2 templates are policies/actions, not causal graph nodes.
TEMPLATE_NAMES = [
    "approach_key",
    "pickup_key",
    "approach_door",
    "toggle_door",
    "approach_target",
    "pickup_target",
    "approach_goal",
    "approach_box",
    "toggle_box",
]
TEMPLATE2ID = {n: i for i, n in enumerate(TEMPLATE_NAMES)}

COLOR_NAMES = ["red", "green", "blue", "purple", "yellow", "grey"]
COLOR2ID = {c: i for i, c in enumerate(COLOR_NAMES)}
OBJ_NAMES = ["none", "key", "ball", "box"]
OBJ2ID = {o: i for i, o in enumerate(OBJ_NAMES)}


@dataclass
class TemplateGoal:
    template: str
    target_obj: str = "none"
    target_color: str = "grey"


def _template_vector(goal: TemplateGoal) -> np.ndarray:
    t = np.zeros(len(TEMPLATE_NAMES), dtype=np.float32)
    t[TEMPLATE2ID[goal.template]] = 1.0

    obj = np.zeros(len(OBJ_NAMES), dtype=np.float32)
    obj[OBJ2ID.get(goal.target_obj, 0)] = 1.0

    col = np.zeros(len(COLOR_NAMES), dtype=np.float32)
    col[COLOR2ID.get(goal.target_color, COLOR2ID["grey"])] = 1.0

    return np.concatenate([t, obj, col], axis=0)


def _dir_to_face(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Optional[int]:
    fx, fy = from_pos
    tx, ty = to_pos
    dx, dy = tx - fx, ty - fy
    # MiniGrid directions: 0 right, 1 down, 2 left, 3 up.
    if (dx, dy) == (1, 0):
        return 0
    if (dx, dy) == (0, 1):
        return 1
    if (dx, dy) == (-1, 0):
        return 2
    if (dx, dy) == (0, -1):
        return 3
    return None


def _iter_grid_objects(env):
    u = env.unwrapped
    grid = u.grid
    for idx, obj in enumerate(grid.grid):
        if obj is None:
            continue
        x = int(idx % grid.width)
        y = int(idx // grid.width)
        yield x, y, obj


def _find_obj_positions(env, obj_type: str, color: str | None = None) -> List[Tuple[int, int]]:
    out = []
    for x, y, obj in _iter_grid_objects(env):
        if getattr(obj, "type", None) != obj_type:
            continue
        if color is not None and getattr(obj, "color", None) != color:
            continue
        out.append((x, y))
    return out


def _place_facing_cell(env, target_pos: Tuple[int, int]) -> bool:
    """Place agent in an empty neighboring cell and face target_pos."""
    u = env.unwrapped
    grid = u.grid
    tx, ty = int(target_pos[0]), int(target_pos[1])
    candidates = [
        (tx - 1, ty),
        (tx + 1, ty),
        (tx, ty - 1),
        (tx, ty + 1),
    ]
    for ax, ay in candidates:
        if not (0 <= ax < grid.width and 0 <= ay < grid.height):
            continue
        if grid.get(ax, ay) is not None:
            continue
        d = _dir_to_face((ax, ay), (tx, ty))
        if d is None:
            continue
        u.agent_pos = (ax, ay)
        u.agent_dir = d
        return True
    return False


def _place_facing_door(env) -> bool:
    target = choose_target_door(env)
    if target is None:
        return False
    x, y, _door = target
    return _place_facing_cell(env, (int(x), int(y)))


def _place_facing_obj(env, obj_type: str, color: str | None = None) -> bool:
    positions = _find_obj_positions(env, obj_type, color=color)
    if not positions:
        return False
    return _place_facing_cell(env, positions[0])


def _door_in_front(env) -> bool:
    target = choose_target_door(env)
    if target is None:
        return False
    x, y, _door = target
    u = env.unwrapped
    if not hasattr(u, "dir_vec"):
        return False
    fx = int(u.agent_pos[0] + u.dir_vec[0])
    fy = int(u.agent_pos[1] + u.dir_vec[1])
    return (fx, fy) == (int(x), int(y))


def _obj_in_front(env, obj_type: str, color: str | None = None) -> bool:
    u = env.unwrapped
    if not hasattr(u, "dir_vec"):
        return False
    fx = int(u.agent_pos[0] + u.dir_vec[0])
    fy = int(u.agent_pos[1] + u.dir_vec[1])
    if not (0 <= fx < u.grid.width and 0 <= fy < u.grid.height):
        return False
    obj = u.grid.get(fx, fy)
    if obj is None:
        return False
    if getattr(obj, "type", None) != obj_type:
        return False
    if color is not None and getattr(obj, "color", None) != color:
        return False
    return True


class G2TemplateEnv(gym.Wrapper):
    """
    G2 action-template option wrapper.

    It keeps the same external state semantics as G0/G1, but trains separate
    template policies with action-role constraints:
      - approach_*: mostly navigation actions
      - pickup_*: navigation + pickup/drop actions
      - toggle_*: navigation + toggle actions

    Internal readiness conditions such as door_ready are used only for G2
    template training/switching, not as external causal graph nodes.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_id: str,
        template: str,
        option_horizon: int = 128,
        seed: int = 0,
        use_nodrop: bool = False,
        use_precondition_do: bool = True,
        parent_goal: Optional[str] = None,
        reset_max_tries: int = 64,
    ):
        if template not in TEMPLATE_NAMES:
            raise ValueError(f"Unknown G2 template: {template}. Available: {TEMPLATE_NAMES}")

        base = gym.make(env_id)
        if use_nodrop:
            base = NoDropWrapper(base)
        super().__init__(base)

        self.env_id = env_id
        self.template = template
        self.option_horizon = int(option_horizon)
        self.rng = np.random.default_rng(seed)
        self.use_precondition_do = bool(use_precondition_do)
        self.parent_goal = to_new(parent_goal) if parent_goal else None
        self.reset_max_tries = int(reset_max_tries)

        actions_enum = getattr(self.env.unwrapped, "actions", None)
        self._left_action_id = int(getattr(actions_enum, "left", 0)) if actions_enum is not None else 0
        self._right_action_id = int(getattr(actions_enum, "right", 1)) if actions_enum is not None else 1
        self._forward_action_id = int(getattr(actions_enum, "forward", 2)) if actions_enum is not None else 2
        self._pickup_action_id = int(getattr(actions_enum, "pickup", 3)) if actions_enum is not None else 3
        self._drop_action_id = int(getattr(actions_enum, "drop", 4)) if actions_enum is not None else 4
        self._toggle_action_id = int(getattr(actions_enum, "toggle", 5)) if actions_enum is not None else 5

        obs, _ = self.env.reset(seed=seed)
        img = obs["image"] if isinstance(obs, dict) else obs
        H, W, C = img.shape

        self._goal_dim = len(TEMPLATE_NAMES) + len(OBJ_NAMES) + len(COLOR_NAMES)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(C, H, W), dtype=np.uint8),
            "direction": gym.spaces.Discrete(4),
            "goal": gym.spaces.Box(low=0.0, high=1.0, shape=(self._goal_dim,), dtype=np.float32),
        })

        self._allowed_actions = self._allowed_action_ids(self.template)
        self.action_space = gym.spaces.Discrete(len(self._allowed_actions))

        self._t = 0
        self._goal = self._make_template_goal()
        self._prev_success_val = 0

        # diagnostics
        self._template_success_once = False
        self._pickup_attempts = 0
        self._toggle_attempts = 0
        self._door_ready_once = False
        self._target_front_once = False

    def set_template(self, template: str, parent_goal: Optional[str] = None):
        if template not in TEMPLATE_NAMES:
            raise ValueError(f"Unknown G2 template: {template}")
        self.template = template
        self.parent_goal = to_new(parent_goal) if parent_goal else None
        self._allowed_actions = self._allowed_action_ids(template)
        self.action_space = gym.spaces.Discrete(len(self._allowed_actions))
        self._goal = self._make_template_goal()

    def _allowed_action_ids(self, template: str) -> List[int]:
        move = [self._left_action_id, self._right_action_id, self._forward_action_id]
        if template.startswith("approach_"):
            return move
        if template.startswith("pickup_"):
            # Include drop so pickup_target can learn to drop a key before picking a box.
            return move + [self._pickup_action_id, self._drop_action_id]
        if template.startswith("toggle_"):
            return move + [self._toggle_action_id]
        return move

    def _map_action(self, action) -> int:
        idx = int(action)
        idx = max(0, min(idx, len(self._allowed_actions) - 1))
        return int(self._allowed_actions[idx])

    def _make_template_goal(self) -> TemplateGoal:
        target_obj = "none"
        target_color = "grey"

        if self.template in ("approach_target", "pickup_target"):
            obj, color = parse_target_from_mission(self.env)
            if obj is not None:
                target_obj = obj
            if color is not None:
                target_color = color
        elif self.template in ("approach_key", "pickup_key"):
            target_obj = "key"
        elif self.template in ("approach_box", "toggle_box"):
            target_obj = "box"

        return TemplateGoal(template=self.template, target_obj=target_obj, target_color=target_color)

    def _encode_obs(self, obs: Any) -> Dict[str, Any]:
        if isinstance(obs, dict):
            img = obs["image"]
            direction = int(obs.get("direction", 0))
        else:
            img = obs
            direction = 0

        img_chw = np.transpose(img, (2, 0, 1)).astype(np.uint8)
        gvec = _template_vector(self._goal).astype(np.float32)
        return {"image": img_chw, "direction": direction, "goal": gvec}

    def encode_current_obs(self) -> Dict[str, Any]:
        obs = self.env.unwrapped.gen_obs() if hasattr(self.env.unwrapped, "gen_obs") else self.env.unwrapped.observation(self.env.unwrapped.grid)
        return self._encode_obs(obs)

    def _target_obj_color(self) -> Tuple[Optional[str], Optional[str]]:
        obj, color = parse_target_from_mission(self.env)
        if obj is not None:
            return obj, color
        return None, None

    def _template_success_value(self) -> int:
        sg = extract_subgoals(self.env)

        if self.template == "approach_key":
            return int(sg.get("near_key", 0))
        if self.template == "pickup_key":
            return int(sg.get("has_key", 0))

        if self.template == "approach_target":
            return int(sg.get("near_target", 0))
        if self.template == "pickup_target":
            return int(sg.get("has_target", 0))

        if self.template == "approach_goal":
            return int(sg.get("at_goal", 0))

        if self.template == "approach_door":
            return int(_door_in_front(self.env))
        if self.template == "toggle_door":
            return int(sg.get("opened_door", 0))

        if self.template == "approach_box":
            return int(_obj_in_front(self.env, "box") or sg.get("near_target", 0))
        if self.template == "toggle_box":
            return int(sg.get("opened_box", 0))

        return 0

    def external_success(self, external_node: str) -> int:
        external_node = to_new(external_node)
        return int(extract_subgoals(self.env).get(external_node, 0))

    def _apply_template_preconditions(self):
        if not self.use_precondition_do:
            return True

        t = self.template
        env_id = self.env_id

        # Optional explicit parent first.
        if self.parent_goal == "near_key":
            if not do_near_obj(self.env, "key", color=None):
                return False
        elif self.parent_goal == "has_key":
            if not do_have_matching_key_for_locked_door(self.env):
                return False
        elif self.parent_goal == "opened_door":
            do_have_matching_key_for_locked_door(self.env)
            if not do_door_open(self.env, open_=True):
                return False
        elif self.parent_goal == "near_target":
            obj, color = self._target_obj_color()
            if obj is None:
                return False
            if not do_near_obj(self.env, obj, color=color):
                return False

        # Template defaults.
        if t == "pickup_key":
            return bool(do_near_obj(self.env, "key", color=None))

        if t == "approach_door":
            return bool(do_have_matching_key_for_locked_door(self.env))

        if t == "toggle_door":
            ok_key = do_have_matching_key_for_locked_door(self.env)
            ok_face = _place_facing_door(self.env)
            return bool(ok_key and ok_face)

        if t == "approach_goal":
            # DoorKey-style navigation after door opened.
            do_have_matching_key_for_locked_door(self.env)
            return bool(do_door_open(self.env, open_=True))

        if t == "pickup_target":
            obj, color = self._target_obj_color()
            if obj is None:
                return False

            # In UnlockPickup, mimic the post-door state: door open and key may still be carried.
            if "UnlockPickup" in env_id:
                do_have_matching_key_for_locked_door(self.env)
                do_door_open(self.env, open_=True)

            return bool(do_near_obj(self.env, obj, color=color))

        if t == "approach_target":
            if self.parent_goal == "opened_door" or "UnlockPickup" in env_id:
                do_have_matching_key_for_locked_door(self.env)
                do_door_open(self.env, open_=True)
            return True

        if t == "toggle_box":
            return bool(_place_facing_obj(self.env, "box", color=None))

        return True

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._t = 0
        obs, info = self.env.reset(seed=seed, options=options)
        self._goal = self._make_template_goal()

        # BabyAI-Pickup target is mission-dependent; keep current mission, no filtering here.
        ok = self._apply_template_preconditions()

        obs = self.env.unwrapped.gen_obs() if hasattr(self.env.unwrapped, "gen_obs") else obs
        self._prev_success_val = int(self._template_success_value())

        self._template_success_once = bool(self._prev_success_val)
        self._pickup_attempts = 0
        self._toggle_attempts = 0
        self._door_ready_once = bool(_door_in_front(self.env))
        obj, color = self._target_obj_color()
        self._target_front_once = bool(obj is not None and _obj_in_front(self.env, obj, color=color))

        info = dict(info)
        info["g2_template"] = self.template
        info["g2_precondition_ok"] = float(ok)
        info["goal_node"] = self.template
        info["external_subgoals"] = extract_subgoals(self.env)
        return self._encode_obs(obs), info

    def step(self, action):
        mapped_action = self._map_action(action)

        pre_success = int(self._template_success_value())
        obs, _, terminated, truncated, info = self.env.step(mapped_action)
        post_success = int(self._template_success_value())

        self._t += 1
        reward = 1.0 if (pre_success == 0 and post_success == 1) else 0.0
        self._prev_success_val = post_success

        if mapped_action == self._pickup_action_id:
            self._pickup_attempts += 1
        if mapped_action == self._toggle_action_id:
            self._toggle_attempts += 1

        if _door_in_front(self.env):
            self._door_ready_once = True
        obj, color = self._target_obj_color()
        if obj is not None and _obj_in_front(self.env, obj, color=color):
            self._target_front_once = True
        if post_success:
            self._template_success_once = True

        if reward > 0:
            terminated = True
        if self._t >= self.option_horizon:
            truncated = True

        info = dict(info)
        info["g2_template"] = self.template
        info["mapped_action"] = int(mapped_action)
        info["subgoals"] = extract_subgoals(self.env)
        info["goal_node"] = self.template
        info["is_success"] = bool(reward > 0)
        info["debug_g2_template_success_once"] = float(self._template_success_once)
        info["debug_g2_door_ready_once"] = float(self._door_ready_once)
        info["debug_g2_target_front_once"] = float(self._target_front_once)
        info["debug_g2_pickup_attempt_count"] = int(self._pickup_attempts)
        info["debug_g2_toggle_attempt_count"] = int(self._toggle_attempts)

        return self._encode_obs(obs), reward, terminated, truncated, info
