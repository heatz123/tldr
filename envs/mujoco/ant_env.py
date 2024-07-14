from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import math
import os

from gym import utils
import numpy as np
from gym.envs.mujoco import mujoco_env

from envs.mujoco.mujoco_utils import MujocoTrait

from lxml import etree
from tempfile import NamedTemporaryFile


def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


# pylint: disable=missing-docstring
class AntEnv(MujocoTrait, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(
        self,
        task="motion",
        goal=None,
        expose_obs_idxs=None,
        expose_all_qpos=True,
        expose_body_coms=None,
        expose_body_comvels=None,
        expose_foot_sensors=False,
        use_alt_path=False,
        model_path=None,
        fixed_initial_state=False,
        done_allowing_step_unit=None,
        original_env=False,
        render_hw=100,
        gap_size=None,
        hill_height=None,
        half=False,
        uniform_reset=False,
        halfstep=False,
    ):
        utils.EzPickle.__init__(**locals())

        if model_path is None:
            model_path = "ant.xml"

        self._task = task
        self._goal = goal
        self._expose_obs_idxs = expose_obs_idxs
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._expose_foot_sensors = expose_foot_sensors
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.fixed_initial_state = fixed_initial_state

        self._done_allowing_step_unit = done_allowing_step_unit
        self._original_env = original_env
        self.render_hw = render_hw

        self.uniform_reset = uniform_reset
        self.halfstep = halfstep

        # Settings from
        # https://github.com/openai/gym/blob/master/gym/envs/__init__.py

        xml_path = "envs/mujoco/assets/"
        model_path = os.path.abspath(os.path.join(xml_path, model_path))

        parser = etree.XMLParser(remove_blank_text=True)
        with open(model_path, mode="rb") as f:
            xml_string = f.read()
        mjcf = etree.XML(xml_string, parser)
        # TODO fix gap size and option number
        if gap_size is not None:
            base_geom = mjcf.find(f".//geom[@name='base']")
            base_size = 5
            base_geom.attrib["size"] = f"{base_size} {base_size} 2"

            for direction in ["north", "south", "east", "west"]:
                gap_geom = mjcf.find(f".//geom[@name='gap_{direction}']")
                if gap_geom is None:
                    continue
                x = 0
                if direction == "east":
                    x = (base_size + gap_size) + 20
                elif direction == "west":
                    x = -(base_size + gap_size) - 20

                y = 0
                if direction == "north":
                    y = (base_size + gap_size) + 20
                elif direction == "south":
                    y = -(base_size + gap_size) - 20

                gap_geom.attrib["pos"] = f"{x} {y} 0"
                gap_geom.attrib["size"] = "20 20 2"

        if hill_height is not None:
            base_geom = mjcf.find(f".//geom[@name='base']")
            base_size = 5
            base_geom.attrib["size"] = f"{base_size} {base_size} 2"

            for direction in ["north", "south", "east", "west"]:
                hill_geom = mjcf.find(f".//geom[@name='hill_{direction}']")
                if hill_geom is None:
                    continue
                x = 0
                if direction == "east":
                    x = (base_size) + 30
                elif direction == "west":
                    x = -(base_size) - 30

                y = 0
                if direction == "north":
                    y = (base_size) + 30
                elif direction == "south":
                    y = -(base_size) - 30

                hill_geom.attrib["pos"] = f"{x} {y} 0"

                if (not half) or (half and direction in ["north"]):
                    print(half, direction)
                    hill_geom.attrib["size"] = f"30 30 {2 + hill_height}"
                else:
                    hill_geom.attrib["size"] = f"30 30 2"

        # save as tempfile
        with NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(etree.tostring(mjcf, pretty_print=True).decode())
        mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    def compute_reward(self, **kwargs):
        return None

    def _get_done(self):
        return False

    def step(self, a, render=False):
        if hasattr(self, "_step_count"):
            self._step_count += 1

        obsbefore = self._get_obs()
        xposbefore = self.sim.data.qpos.flat[0]
        yposbefore = self.sim.data.qpos.flat[1]
        self.do_simulation(a, self.frame_skip)
        obsafter = self._get_obs()
        xposafter = self.sim.data.qpos.flat[0]
        yposafter = self.sim.data.qpos.flat[1]

        reward = self.compute_reward(
            xposbefore=xposbefore,
            yposbefore=yposbefore,
            xposafter=xposafter,
            yposafter=yposafter,
        )
        if reward is None:
            forward_reward = (xposafter - xposbefore) / self.dt
            sideward_reward = (yposafter - yposbefore) / self.dt

            ctrl_cost = 0.5 * np.square(a).sum()
            survive_reward = 1.0
            if self._task == "forward":
                reward = forward_reward - ctrl_cost + survive_reward
            elif self._task == "backward":
                reward = -forward_reward - ctrl_cost + survive_reward
            elif self._task == "left":
                reward = sideward_reward - ctrl_cost + survive_reward
            elif self._task == "right":
                reward = -sideward_reward - ctrl_cost + survive_reward
            elif self._task == "goal":
                reward = -np.linalg.norm(np.array([xposafter, yposafter]) - self._goal)
            elif self._task == "motion":
                reward = (
                    np.max(np.abs(np.array([forward_reward, sideward_reward])))
                    - ctrl_cost
                    + survive_reward
                )

            def _get_gym_ant_reward():
                forward_reward = (xposafter - xposbefore) / self.dt
                ctrl_cost = 0.5 * np.square(a).sum()
                contact_cost = (
                    0.5
                    * 1e-3
                    * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
                )
                survive_reward = 1.0
                reward = forward_reward - ctrl_cost - contact_cost + survive_reward
                return reward

            reward = _get_gym_ant_reward()

        done = self._get_done()

        ob = self._get_obs()
        info = dict(
            # reward_forward=forward_reward,
            # reward_sideward=sideward_reward,
            # reward_ctrl=-ctrl_cost,
            # reward_survive=survive_reward,
            coordinates=np.array([xposbefore, yposbefore]),
            next_coordinates=np.array([xposafter, yposafter]),
            ori_obs=obsbefore,
            next_ori_obs=obsafter,
        )

        if render:
            info["render"] = self.render(mode="rgb_array", camera_id=0).transpose(
                2, 0, 1
            )

        return ob, reward, done, info

    def _get_obs(self):
        if self._original_env:
            return np.concatenate(
                [
                    self.sim.data.qpos.flat[2:],
                    self.sim.data.qvel.flat,
                    np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                ]
            )

        # No crfc observation
        if self._expose_all_qpos:
            obs = np.concatenate(
                [
                    self.sim.data.qpos.flat[:15],
                    self.sim.data.qvel.flat[:14],
                ]
            )
        else:
            obs = np.concatenate(
                [
                    self.sim.data.qpos.flat[2:15],
                    self.sim.data.qvel.flat[:14],
                ]
            )

        if self._expose_body_coms is not None:
            for name in self._expose_body_coms:
                com = self.get_body_com(name)
                if name not in self._body_com_indices:
                    indices = range(len(obs), len(obs) + len(com))
                    self._body_com_indices[name] = indices
                obs = np.concatenate([obs, com])

        if self._expose_body_comvels is not None:
            for name in self._expose_body_comvels:
                comvel = self.get_body_comvel(name)
                if name not in self._body_comvel_indices:
                    indices = range(len(obs), len(obs) + len(comvel))
                    self._body_comvel_indices[name] = indices
                obs = np.concatenate([obs, comvel])

        if self._expose_foot_sensors:
            obs = np.concatenate([obs, self.sim.data.sensordata])

        if self._expose_obs_idxs is not None:
            obs = obs[self._expose_obs_idxs]

        return obs

    def _get_done(self):
        return False

    def reset_model(self):
        self._step_count = 0
        self._done_internally = False

        if self.fixed_initial_state:
            qpos = self.init_qpos
            qvel = self.init_qvel
        else:
            qpos = self.init_qpos + np.random.uniform(
                size=self.sim.model.nq, low=-0.1, high=0.1
            )
            qvel = self.init_qvel + np.random.randn(self.sim.model.nv) * 0.1

        if not self._original_env:
            qpos[15:] = self.init_qpos[15:]
            qvel[14:] = 0.0

        if self.uniform_reset:
            # qpos[3:] = self.init_qpos[3:] + np.random.uniform(
            #     size=self.sim.model.nq - 2, low=-0.1, high=0.1
            # )
            qpos[:2] = np.random.uniform(size=2, low=-3.5, high=3.5)
            if self.halfstep:
                qpos[2] = 2.75 + (0.7 if qpos[0] >= -1 else 0)
            qvel = self.init_qvel + np.random.randn(self.sim.model.nv) * 0.1

        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        # self.viewer.cam.distance = self.model.stat.extent * 2.5
        pass

    @property
    def body_com_indices(self):
        return self._body_com_indices

    @property
    def body_comvel_indices(self):
        return self._body_comvel_indices

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        coord_dims = [0, 1]
        eval_metrics = super().calc_eval_metrics(
            trajectories, is_option_trajectories, coord_dims
        )
        return eval_metrics
