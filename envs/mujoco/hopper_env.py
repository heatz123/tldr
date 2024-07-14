import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

from envs.mujoco.mujoco_utils import MujocoTrait

DEFAULT_VEL = 3

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class LifelongHopperEnv(MujocoTrait, mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Described in Lu et al. 2020.
    """

    def __init__(
        self,
        xml_file="hopper.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        target_vel=None,
        target_vel_in_obs=False,
        rgb_rendering_tracking=True,
    ):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self._target_vel = target_vel
        self._target_vel_in_obs = target_vel_in_obs
        self._target_vel_reward_weight = 1

        mujoco_env.MujocoEnv.__init__(self, xml_file, 4)

    """
    Required for compatibility with lifelong_rl lifelong environment setting
    """

    def get_env_state(self):
        return self.sim.get_state()

    def set_env_state(self, state):
        self.sim.set_state(state)

    """
    =================================================================
    """

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def set_target_vel(self, vel):
        self._target_vel = vel

    def get_target_vel(self):
        if self._target_vel is not None:
            return self._target_vel
        else:
            return DEFAULT_VEL

    @property
    def is_healthy(self):
        z, angle = self.sim.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(self.sim.data.qvel.flat.copy(), -10.0, 10.0)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        if self._target_vel is not None and self._target_vel_in_obs:
            target_vel = np.array([self.get_target_vel()])
        else:
            target_vel = []

        observation = np.concatenate((position, velocity, target_vel)).ravel()
        return observation

    def get_obs(self):
        return self._get_obs()

    def step(self, action, render=False):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        z, z_des = self.sim.data.qpos[1], 1.8
        height_cost = 5 * ((z - z_des) ** 2)
        vel_cost = abs(x_velocity - self.get_target_vel())
        ctrl_cost = 0.1 * np.sum(np.square(action))

        rewards = abs(self.get_target_vel())
        costs = height_cost + vel_cost + ctrl_cost

        reward = rewards - costs
        info = {
            "coordinates": np.array([x_position_before, 0.0]),
            "next_coordinates": np.array([x_position_after, 0.0]),
            "x_velocity": x_velocity,
            "z": z,
        }

        if render:
            info["render"] = self.render(mode="rgb_array").transpose(2, 0, 1)

        return self._get_obs(), reward, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

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
        coord_dims = [0, 1]
        eval_metrics = super().calc_eval_metrics(
            trajectories, is_option_trajectories, coord_dims
        )
        return eval_metrics
