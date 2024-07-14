import os

import akro
from gym import utils
import numpy as np
from envs.mujoco.half_cheetah_env import HalfCheetahEnv
from envs.mujoco.mujoco_utils import convert_observation_to_space
from gym.envs.mujoco import mujoco_env


class HalfCheetahHurdleEnv(HalfCheetahEnv):
    def __init__(self,
                 expose_obs_idxs=None,
                 expose_all_qpos=True,
                 task='default',
                 target_velocity=None,
                 model_path=None,
                 fixed_initial_state=False,

                 reward_type='sparse'):
        utils.EzPickle.__init__(**locals())

        if model_path is None:
            model_path = 'half_cheetah_hurdle.xml'

        # Settings from
        # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
        self._expose_obs_idxs = expose_obs_idxs
        self._expose_all_qpos = expose_all_qpos
        self._task = task
        self._target_velocity = target_velocity
        self.fixed_initial_state = fixed_initial_state

        # Hurdle-specific vars
        self.exteroceptive_observation = [12.0, 0, 0.5]
        self.hurdles_xpos = [-15., -13., -9., -5., -1., 3., 7., 11., 15., 19., 23., 27.]
        self.reward_type = reward_type
        self.cur_hurdle_reward = 0

        xml_path = "envs/mujoco/assets/"
        model_path = os.path.abspath(os.path.join(xml_path, model_path))

        mujoco_env.MujocoEnv.__init__(
            self,
            model_path,
            5)

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        low = np.full((2,), -float('inf'), dtype=np.float32)
        high = np.full((2,), float('inf'), dtype=np.float32)
        return akro.concat(self.observation_space, akro.Box(low=low, high=high, dtype=self.observation_space.dtype))

    def _get_obs(self):
        obs = super()._get_obs()
        x_pos1 = self.get_body_com('ffoot')[0]  # self.model.data.qpos.flat[:1]
        x_pos2 = self.get_body_com('bfoot')[0]  # self.model.data.qpos.flat[:1]
        matches = [x for x in self.hurdles_xpos if x >= x_pos2]
        if len(matches) == 0:
            matches.append(15.)
        next_hurdle_x_pos = [matches[0]]
        ff_dist_frm_next_hurdle = [np.linalg.norm(matches[0] - x_pos1)]
        bf_dist_frm_next_hurdle = [np.linalg.norm(matches[0] - x_pos2)]
        obs = np.concatenate([obs, next_hurdle_x_pos, bf_dist_frm_next_hurdle])

        return obs

    def reset_model(self):
        res = super().reset_model()
        self.cur_hurdle_reward = self.get_hurdle_reward()
        return res

    def isincollision(self):
        hurdle_size = [0.05, 1.0, 0.03]
        x_pos = self.get_body_com('ffoot')[0]  # self.model.data.qpos.flat[:1]
        matches = [x for x in self.hurdles_xpos if x >= x_pos]
        if len(matches) == 0:
            return False
        hurdle_pos = [matches[0], 0.0, 0.20]
        # names=['fthigh','bthigh']
        # names=['torso','bthigh','bshin','bfoot']
        names = ['ffoot']
        xyz_pos = []
        for i in range(0, len(names)):
            xyz_pos.append(self.get_body_com(names[i]))
        for i in range(0, len(names)):
            # xyz_position = self.get_body_com(names[i])
            cf = True
            for j in range(0, 1):
                if abs(hurdle_pos[j] - xyz_pos[i][j]) > 1.5 * hurdle_size[j]:
                    cf = False
                    break
            if cf:
                return True
        return False

    def get_hurdle_reward(self):
        hurdle_size = [0.05, 1.0, 0.03]
        x_pos = self.get_body_com('bfoot')[0]  # self.model.data.qpos.flat[:1]
        matches = [x for x in self.hurdles_xpos if x >= x_pos and x >= 0]
        hurdle_reward = -1.0 * len(matches)

        return hurdle_reward

    def compute_reward(self, **kwargs):
        # xyz_pos_before = self.get_body_com('bshin')
        # xyz_pos_after = self.get_body_com('bshin')
        # xyz_position = self.get_body_com('torso')
        # jump_reward = np.abs(self.data.get_body_xvelp("torso")[2])
        # run_reward = self.data.get_body_xvelp("torso")[0]
        # if self.isincollision():  # or (xyz_pos_after[0]-xyz_pos_before[0])<-0.01:#dist_from_hurdle < 1 and dist_from_hurdle > 0.3 and z_after<0.05:(xyz_pos_after[0]-xyz_pos_before[0])<-0.01: #
        #     collision_penality = -2.0
        # # print("collision")
        # else:
        #     collision_penality = 0.0
        # # print("not collisions")
        # hurdle_reward = self.get_hurdle_reward()
        # # print(hurdle_reward)
        # goal_reward = 0
        # goal_distance = np.linalg.norm(xyz_position - self.exteroceptive_observation)
        # if goal_distance < 1.0:
        #     goal_reward = 1000

        # reward = -1e-1 * goal_distance + hurdle_reward + goal_reward + run_reward + 3e-1 * jump_reward + collision_penality  # 1e-1*goal_distance+run_reward+jump_reward+collision_penality

        # Inject custom sparse reward
        prev_hurdle_reward = self.cur_hurdle_reward
        self.cur_hurdle_reward = self.get_hurdle_reward()

        reward = self.cur_hurdle_reward - prev_hurdle_reward

        return reward
