# multitask_g2_template_env.py
from __future__ import annotations

from typing import Optional, Sequence

import gymnasium as gym
import numpy as np

from rl.g2_template_env import G2TemplateEnv


class MultiTaskG2TemplateEnv(gym.Env):
    """
    Multi-task wrapper for G2 template policies.

    Put this file in:
        D:\Project\HRC_granularity\multitask_g2_template_env.py
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_ids: Sequence[str],
        template: str,
        option_horizon: int = 256,
        seed: int = 0,
        use_nodrop: bool = False,
        use_precondition_do: bool = True,
        parent_goal: Optional[str] = None,
        reset_max_tries: int = 128,
        task_sampling: str = "uniform",
    ):
        super().__init__()
        if not env_ids:
            raise ValueError("env_ids must be non-empty")
        if task_sampling not in ("uniform", "round_robin"):
            raise ValueError("task_sampling must be 'uniform' or 'round_robin'")

        self.env_ids = list(env_ids)
        self.template = template
        self.option_horizon = int(option_horizon)
        self.seed = int(seed)
        self.use_nodrop = bool(use_nodrop)
        self.use_precondition_do = bool(use_precondition_do)
        self.parent_goal = parent_goal
        self.reset_max_tries = int(reset_max_tries)
        self.task_sampling = task_sampling
        self.rng = np.random.default_rng(seed)
        self._rr_i = -1
        self._active_i = 0

        self.envs = [
            G2TemplateEnv(
                env_id=eid,
                template=template,
                option_horizon=option_horizon,
                seed=seed + 997 * i,
                use_nodrop=use_nodrop,
                use_precondition_do=use_precondition_do,
                parent_goal=parent_goal,
                reset_max_tries=reset_max_tries,
            )
            for i, eid in enumerate(self.env_ids)
        ]

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        for env in self.envs[1:]:
            if env.action_space != self.action_space:
                raise ValueError(f"Action space mismatch for template={template}: {env.action_space} vs {self.action_space}")
            if env.observation_space != self.observation_space:
                raise ValueError("Observation space mismatch across source tasks.")

    @property
    def active_env(self) -> G2TemplateEnv:
        return self.envs[self._active_i]

    @property
    def unwrapped(self):
        return self.active_env.unwrapped

    def _choose_task(self) -> int:
        if self.task_sampling == "round_robin":
            self._rr_i = (self._rr_i + 1) % len(self.envs)
            return self._rr_i
        return int(self.rng.integers(0, len(self.envs)))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._active_i = self._choose_task()
        child_seed = None if seed is None else int(seed + 104729 * self._active_i)
        obs, info = self.active_env.reset(seed=child_seed, options=options)
        info = dict(info)
        info["task_id"] = int(self._active_i)
        info["task_env_id"] = self.env_ids[self._active_i]
        info["g2_template"] = self.template
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.active_env.step(action)
        info = dict(info)
        info["task_id"] = int(self._active_i)
        info["task_env_id"] = self.env_ids[self._active_i]
        info["g2_template"] = self.template
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.active_env.render()

    def close(self):
        for env in self.envs:
            env.close()
