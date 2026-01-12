from __future__ import annotations

from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 4,
        max_steps: int = 30,
        reward_mode: str = "sparse",
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        pit_reward: float = -1.0,
        render_mode: Optional[str] = None,
    ):
        if reward_mode not in ("sparse", "dense"):
            raise ValueError("reward_mode must be 'sparse' or 'dense'")
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.pit_reward = pit_reward
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.pits = {(1, 1)}

        self._agent_pos = self.start_pos
        self._step_count = 0
        self._rng = np.random.default_rng()
        self._prev_dist = self._distance_to_goal(self._agent_pos)

    def _get_obs(self) -> np.ndarray:
        ax, ay = self._agent_pos
        gx, gy = self.goal_pos
        obs = np.array(
            [ax, ay, gx, gy], dtype=np.float32
        )
        obs /= float(self.grid_size - 1)
        return obs

    def _distance_to_goal(self, pos: Tuple[int, int]) -> float:
        ax, ay = pos
        gx, gy = self.goal_pos
        return abs(ax - gx) + abs(ay - gy)

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._agent_pos = self.start_pos
        self._step_count = 0
        self._prev_dist = self._distance_to_goal(self._agent_pos)
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        x, y = self._agent_pos
        if action == 0:
            y = min(self.grid_size - 1, y + 1)
        elif action == 1:
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            x = max(0, x - 1)

        self._agent_pos = (x, y)
        self._step_count += 1

        terminated = False
        reward = self.step_penalty

        if self._agent_pos == self.goal_pos:
            reward = self.goal_reward
            terminated = True
        elif self._agent_pos in self.pits:
            reward = self.pit_reward
            terminated = True
        else:
            if self.reward_mode == "dense":
                max_dist = 2 * (self.grid_size - 1)
                dist = self._distance_to_goal(self._agent_pos)
                reward = (self._prev_dist - dist) / float(max_dist)
                self._prev_dist = dist

        truncated = self._step_count >= self.max_steps
        obs = self._get_obs()
        info = {
            "is_success": self._agent_pos == self.goal_pos,
            "distance": self._distance_to_goal(self._agent_pos),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        cell = 32
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        grid[:] = (220, 220, 220)
        gx, gy = self.goal_pos
        grid[gy, gx] = (0, 200, 0)
        for (px, py) in self.pits:
            grid[py, px] = (200, 0, 0)
        ax, ay = self._agent_pos
        grid[ay, ax] = (30, 90, 200)
        grid = np.kron(grid, np.ones((cell, cell, 1), dtype=np.uint8))
        return grid

    def close(self):
        return None
