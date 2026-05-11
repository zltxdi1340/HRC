# env/doorkey_ball_env.py
"""
Custom MiniGrid env to create a non-chain (graph-shaped) causal structure via an AND success condition.

Task (intuition):
- Key + locked door separates two rooms.
- Goal is in the right room (behind the locked door).
- Ball is in the left/start room (independent of the door).
- Episode success ONLY if: (agent reaches Goal) AND (agent is carrying the Ball).

This encourages a DAG where success has TWO parents:
  at_goal -> success
  has_ball -> success
while still keeping the DoorKey chain for navigation:
  has_key -> door_open -> at_goal

Important:
- We do NOT force pickup. The agent must actually use the pickup action to carry the ball.
- Ball is placed in the start room; goal is placed in the other room, ensuring "AND" rather than a single chain.
"""

from __future__ import annotations

from typing import Optional, Tuple

from gymnasium import Env
from gymnasium.envs.registration import register, registry

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Ball, Wall
from minigrid.minigrid_env import MiniGridEnv


class DoorKeyBallEnv(MiniGridEnv):
    def __init__(
        self,
        size: int = 6,
        max_steps: Optional[int] = None,
        agent_view_size: int = 7,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        assert size >= 6, "size should be >= 6 for a two-room layout"

        mission_space = MissionSpace(
            mission_func=lambda: "pick up the ball, use the key to open the door, and reach the goal"
        )

        if max_steps is None:
            max_steps = 4 * size * size

        # MiniGrid versions differ slightly in ctor args, so we try both.
        try:
            super().__init__(
                mission_space=mission_space,
                grid_size=size,
                max_steps=max_steps,
                see_through_walls=False,
                agent_view_size=agent_view_size,
                render_mode=render_mode,
                **kwargs,
            )
        except TypeError:
            super().__init__(
                mission_space=mission_space,
                width=size,
                height=size,
                max_steps=max_steps,
                see_through_walls=False,
                agent_view_size=agent_view_size,
                render_mode=render_mode,
                **kwargs,
            )

        self.size = size
        self._door_pos: Optional[Tuple[int, int]] = None
        self._goal_pos: Optional[Tuple[int, int]] = None

    # ---------- helper state (no manual updates needed) ----------
    @property
    def has_ball(self) -> bool:
        c = self.unwrapped.carrying
        return (c is not None) and (getattr(c, "type", "") == "ball")

    def _at_goal(self) -> bool:
        cell = self.grid.get(*self.agent_pos)
        return (cell is not None) and (getattr(cell, "type", "") == "goal")

    # ---------- grid generation ----------
    def _gen_grid(self, width: int, height: int):
        # Create empty grid
        self.grid = Grid(width, height)

        # Surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Split into two rooms with a vertical wall
        split = width // 2
        for y in range(1, height - 1):
            self.grid.set(split, y, Wall())

        # Locked door in the split wall (middle)
        door_y = height // 2
        door = Door("yellow", is_locked=True)
        self.grid.set(split, door_y, door)
        self._door_pos = (split, door_y)

        # Place agent in LEFT room
        # (top-left corner region excluding walls)
        self.place_agent(top=(1, 1), size=(split - 1, height - 2))

        # Place KEY in LEFT room (must match door color)
        self.place_obj(Key("yellow"), top=(1, 1), size=(split - 1, height - 2))

        # Place BALL in LEFT room (independent branch)
        self.place_obj(Ball("blue"), top=(1, 1), size=(split - 1, height - 2))

        # Place GOAL in RIGHT room (behind the locked door)
        goal = Goal()
        self.place_obj(goal, top=(split + 1, 1), size=(width - split - 2, height - 2))
        # store goal position for debugging/optional checks
        # (Goal has no fixed attribute; locate it by scan if needed)
        self._goal_pos = None

        self.mission = "pick up the ball, use the key to open the door, and reach the goal"

    # ---------- success condition (AND) ----------
    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)

        # If base env terminates because it hit goal, keep termination ONLY if ball is carried.
        # Otherwise, override and allow the episode to continue.
        if terminated and self._at_goal() and (not self.has_ball):
            terminated = False
            reward = 0.0

        # If we're at goal AND carrying ball, ensure success termination & reward.
        # (Some versions already terminate on goal; we reinforce for clarity.)
        if self._at_goal() and self.has_ball:
            terminated = True
            reward = 1.0
            info = dict(info)
            info["and_success"] = True

        return obs, reward, terminated, truncated, info


# ---------- Gymnasium registration ----------
# Import this module once (e.g., at the top of hrc_stage1.py) before gym.make().
ENV_ID_6 = "MiniGrid-DoorKeyBall-6x6-v0"
if ENV_ID_6 not in registry:
    register(
        id=ENV_ID_6,
        entry_point="env.doorkey_ball_env:DoorKeyBallEnv",
        kwargs={"size": 6},
    )
