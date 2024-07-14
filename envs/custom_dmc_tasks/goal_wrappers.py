from collections import deque

import akro
import gym
import numpy as np
import matplotlib.pyplot as plt

from garagei.envs.akro_wrapper import AkroWrapperTrait


class GoalWrapper(AkroWrapperTrait, gym.Wrapper):
    def __init__(
        self,
        env,
        max_path_length,
        goal_range,
        num_goal_steps,
    ):
        super().__init__(env)

        self.max_path_length = max_path_length

        self.goal_epsilon = 1.5
        self.goal_range = goal_range
        self.num_goal_steps = num_goal_steps
        self.cur_goal = np.random.uniform(-self.goal_range, self.goal_range, (2,))
        self.num_steps = 0

        self.observation_space = akro.Box(
            low=-np.inf, high=np.inf, shape=(64 * 64 * 3 + 2,)
        )
        self.ob_info = dict(
            type="hybrid",
            pixel_shape=(64, 64, 3),
            state_shape=2,
        )

    def _transform(self, obs):
        pixels = self.env.render(mode="rgb_array", width=64, height=64).copy()
        pixels = pixels.flatten()
        return np.concatenate([pixels, self.cur_goal], axis=-1)

    def reset(self, **kwargs):
        self.cur_goal = np.random.uniform(-self.goal_range, self.goal_range, (2,))
        obs = self.env.reset(**kwargs)
        self.num_steps = 0
        return self._transform(obs)

    def compute_reward(self, info):
        self.num_steps += 1
        xposafter, yposafter = info["next_coordinates"]
        delta = np.linalg.norm(self.cur_goal - np.array([xposafter, yposafter]))
        if self.num_steps != 1 and delta <= self.goal_epsilon:
            reward = 10.0 / (self.max_path_length / self.num_goal_steps)
            self.num_steps = (
                (self.num_steps + self.num_goal_steps - 1)
                // self.num_goal_steps
                * self.num_goal_steps
            )
        else:
            reward = -0.0

        if self.num_steps % self.num_goal_steps == 0:
            self.cur_goal = np.array(
                [
                    np.random.uniform(
                        xposafter - self.goal_range, xposafter + self.goal_range
                    ),
                    np.random.uniform(
                        yposafter - self.goal_range, yposafter + self.goal_range
                    ),
                ]
            )

        return reward

    def step(self, action, **kwargs):
        next_obs, reward, done, info = self.env.step(action, **kwargs)
        reward = self.compute_reward(info)
        done = self.num_steps == self.max_path_length
        return self._transform(next_obs), reward, done, info
