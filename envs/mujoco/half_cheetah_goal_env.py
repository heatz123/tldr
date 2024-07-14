import akro
import numpy as np
from matplotlib.patches import Ellipse

from envs.mujoco.ant_env import AntEnv
from envs.mujoco.half_cheetah_env import HalfCheetahEnv
from envs.mujoco.mujoco_utils import convert_observation_to_space
from gym import utils


class HalfCheetahGoalEnv(HalfCheetahEnv):
    def __init__(
        self,
        max_path_length,
        goal_range,
        reward_type="sparse",
        **kwargs,
    ):
        self.max_path_length = max_path_length
        self.reward_type = reward_type

        self.goal_epsilon = 3.0
        self.goal_range = goal_range
        self.cur_goal = np.random.uniform(-self.goal_range, self.goal_range, (1,))
        self.cur_transient_goal = None
        self.num_steps = 0

        super().__init__(**kwargs)
        utils.EzPickle.__init__(
            self,
            max_path_length=max_path_length,
            goal_range=goal_range,
            reward_type=reward_type,
            **kwargs,
        )

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        low = np.full((1,), -float("inf"), dtype=np.float32)
        high = np.full((1,), float("inf"), dtype=np.float32)
        return akro.concat(
            self.observation_space,
            akro.Box(low=low, high=high, dtype=self.observation_space.dtype),
        )

    def reset(self, **kwargs):
        if "goal" in kwargs:
            raise Exception("Unknown situation")
            self.cur_transient_goal = kwargs["goal"]
        return super().reset()

    def reset_model(self):
        if self.cur_transient_goal is not None:
            raise Exception("Unknown situation")
            self.cur_goal = self.cur_transient_goal
            self.cur_transient_goal = None
        else:
            self.cur_goal = np.random.uniform(-self.goal_range, self.goal_range, (1,))
        self.num_steps = 0

        return super().reset_model()

    def _get_obs(self):
        obs = super()._get_obs()
        obs = np.concatenate([obs, self.cur_goal])

        return obs

    def _get_done(self):
        return self.num_steps == self.max_path_length

    def compute_reward(self, xposbefore, xposafter):
        self.num_steps += 1
        delta = np.linalg.norm(self.cur_goal - np.array([xposafter]))
        if self.reward_type == "sparse":
            if self.num_steps == self.max_path_length:
                reward = -delta
            else:
                reward = 0.0
        elif self.reward_type == "esparse":
            if self.num_steps != 1 and delta <= self.goal_epsilon:
                reward = 10.0
                self.num_steps = self.max_path_length
            else:
                reward = -0.0
        elif self.reward_type == "ddense":
            delta_before = np.linalg.norm(self.cur_goal - np.array([xposbefore]))
            reward = delta_before - delta
        elif self.reward_type == "dense":
            reward = -delta / self.max_path_length

        return reward
