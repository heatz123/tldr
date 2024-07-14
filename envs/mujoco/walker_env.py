import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from envs.mujoco.mujoco_utils import MujocoTrait


class Walker2dEnv(MujocoTrait, mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 8)
        utils.EzPickle.__init__(self)

    def step(self, a, render=False):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = False  # not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()

        info = dict(
            coordinates=np.array([posbefore, 0.0]),
            next_coordinates=np.array([posafter, 0.0]),
            ori_obs=None,
            next_ori_obs=None,
        )

        if render:
            info["render"] = self.render(mode="rgb_array").transpose(2, 0, 1)

        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def plot_trajectory(self, trajectory, color, ax):
        # https://stackoverflow.com/a/20474765/2182622
        from matplotlib.collections import LineCollection

        linewidths = np.linspace(0.2, 1.2, len(trajectory))
        points = np.reshape(trajectory, (-1, 1, 2))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, linewidths=linewidths, color=color)
        ax.add_collection(lc)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = super()._get_coordinates_trajectories(trajectories)
        for i, traj in enumerate(coordinates_trajectories):
            traj[:, 1] = (i - len(coordinates_trajectories) / 2) / 1.25
        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        coord_dims = [0]
        eval_metrics = super().calc_eval_metrics(
            trajectories, is_option_trajectories, coord_dims
        )
        return eval_metrics
