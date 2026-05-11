from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional

import gymnasium as gym

from rl.gc_option_env import GCOptionEnv


class MultiTaskGCOptionEnv(gym.Env):
    """
    Thin wrapper that samples one task env_id at reset time and trains a single
    shared policy across multiple tasks without modifying the original
    rl/train_sb3_gc_ppo.py.

    It delegates almost everything to an inner GCOptionEnv instance.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_ids: Iterable[str],
        goal_pool: List[str],
        option_horizon: int = 256,
        seed: int = 0,
        use_nodrop: bool = False,
        use_precondition_do: bool = True,
        parent_goal: Optional[str] = None,
        child_goal: Optional[str] = None,
        strict_parent_mode: bool = False,
        strict_parent_fail_fast: bool = False,
        reset_max_tries: int = 64,
        task_sampling: str = "uniform",
    ):
        super().__init__()
        self.env_ids = list(env_ids)
        if not self.env_ids:
            raise ValueError("env_ids must be non-empty")

        self.goal_pool = list(goal_pool)
        self.option_horizon = int(option_horizon)
        self.base_seed = int(seed)
        self.use_nodrop = bool(use_nodrop)
        self.use_precondition_do = bool(use_precondition_do)
        self.parent_goal = parent_goal
        self.child_goal = child_goal
        self.strict_parent_mode = bool(strict_parent_mode)
        self.strict_parent_fail_fast = bool(strict_parent_fail_fast)
        self.reset_max_tries = int(reset_max_tries)
        self.task_sampling = task_sampling

        self._rng = random.Random(self.base_seed)
        self._reset_count = 0
        self._current_env_id: Optional[str] = None
        self.env: Optional[GCOptionEnv] = None

        # Build a probe env to expose stable spaces.
        probe = self._make_single_env(self.env_ids[0], self.base_seed)
        self.action_space = probe.action_space
        self.observation_space = probe.observation_space
        probe.close()

    def _make_single_env(self, env_id: str, seed: int) -> GCOptionEnv:
        return GCOptionEnv(
            env_id=env_id,
            goal_pool=self.goal_pool,
            option_horizon=self.option_horizon,
            seed=seed,
            use_nodrop=self.use_nodrop,
            use_precondition_do=self.use_precondition_do,
            parent_goal=self.parent_goal,
            child_goal=self.child_goal,
            strict_parent_mode=self.strict_parent_mode,
            strict_parent_fail_fast=self.strict_parent_fail_fast,
            reset_max_tries=self.reset_max_tries,
        )

    def _sample_env_id(self) -> str:
        if self.task_sampling == "round_robin":
            idx = self._reset_count % len(self.env_ids)
            return self.env_ids[idx]
        return self._rng.choice(self.env_ids)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # Close previous env to avoid leaks.
        if self.env is not None:
            self.env.close()
            self.env = None

        env_seed = self.base_seed + self._reset_count if seed is None else int(seed)
        self._current_env_id = self._sample_env_id()
        self.env = self._make_single_env(self._current_env_id, env_seed)
        self._reset_count += 1

        obs, info = self.env.reset(seed=env_seed, options=options)
        info = dict(info)
        info["task_env_id"] = self._current_env_id
        return obs, info

    def step(self, action):
        if self.env is None:
            raise RuntimeError("reset() must be called before step().")
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["task_env_id"] = self._current_env_id
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.env is None:
            return None
        return self.env.render()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    @property
    def unwrapped(self):
        if self.env is None:
            return self
        return self.env.unwrapped
