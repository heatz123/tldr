import numpy as np
from gym import utils
from . import mujoco_env
from PIL import Image

# Cheetah that starts upright, has short torso, and has hurdles
class HalfCheetahEnvShortHurdle(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.hurdles_xpos=[7., 30., 40., 60., 85., 135., 160., 180., 220., 260.]
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah_short_hurdles.xml", 5)
        utils.EzPickle.__init__(self)
        
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        xyz_position = self.get_body_com('torso')
        reward_run = (xposafter - xposbefore) #/ self.dt
        reward_ctrl = -0.1 * np.square(action).sum()
        next_obs= self.get_current_obs()
        collision_penalty=0.0
        hurdle_reward = 0.0
        done = False

        reward=(reward_run/self.dt)+reward_ctrl - np.abs(self.sim.data.qpos[2] - 0)
        info = dict(reward_run=reward_run, total_reward=reward)
        return next_obs, reward, done, info
    
    def compute_reward(self, obs):
        reward = float(self.is_successful(obs=obs))
        return reward

    def get_current_obs(self):
        proprioceptive_observation = self._get_obs()
        observation = np.concatenate([proprioceptive_observation,[-1.0]]).reshape(-1)
        return observation
    
    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self.get_current_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        
    def save_im(self, im, name):
        img = Image.fromarray(im.astype(np.uint8))
        img.save(name)
        
    def is_successful(self, obs=None):
        x_pos2 = self.get_body_com('bfoot')[0]
        return x_pos2 > 300