from collections import deque

import akro
import gym
import numpy as np
import matplotlib.pyplot as plt

from garagei.envs.akro_wrapper import AkroWrapperTrait


class RenderWrapper(AkroWrapperTrait, gym.Wrapper):
    """Wrap general environment to match the interface for METRA. Optionally wrap to output pixels."""

    def __init__(
        self,
        env: gym.Env,
        pixel: bool = False,
        floor_color: bool = False,
        wall_color: bool = False,
        hybrid: bool = False,
        **kwargs
    ):
        super().__init__(env)

        self.pixel = pixel
        self.hybrid = hybrid
        assert not (self.pixel and self.hybrid)

        if wall_color:
            assert self.pixel or self.hybrid
            walls = []

            for i in range(env.physics.model.ngeom):
                geom_name = env.physics.model.geom_id2name(i)
                if geom_name.startswith("block_"):
                    walls.append(i)
            # change the texture of the walls
            cmap = plt.get_cmap("tab20")
            for wall in walls:
                env.physics.model.geom_rgba[wall] = cmap(wall % 20)

        if floor_color:
            assert self.pixel or self.hybrid
            l = len(env.physics.model.tex_type)
            for i in range(l):
                if env.physics.model.tex_type[i] == 0:
                    height = env.physics.model.tex_height[i]
                    width = env.physics.model.tex_width[i]
                    s = env.physics.model.tex_adr[i]
                    for x in range(height):
                        for y in range(width):
                            cur_s = s + (x * width + y) * 3
                            env.physics.model.tex_rgb[cur_s : cur_s + 3] = [
                                int(x / height * 255),
                                int(y / width * 255),
                                128,
                            ]
            env.physics.model.mat_texrepeat[:, :] = 1

        if self.pixel:
            self.action_space = self.env.action_space
            self.observation_space = akro.Box(
                low=-np.inf, high=np.inf, shape=(64, 64, 3)
            )

            self.ob_info = dict(
                type="pixel",
                pixel_shape=(64, 64, 3),
            )

        elif self.hybrid:
            self.action_space = self.env.action_space
            self.observation_space = akro.Box(
                low=-np.inf, high=np.inf, shape=(64, 64, 3)
            )

            self.ob_info = dict(
                type="hybrid",
                pixel_shape=(64, 64, 3),
                state_shape=(env.observation_space.shape[0],),
            )

        self.camera_id = 0 if "camera_id" not in kwargs else kwargs["camera_id"]

    def _transform(self, obs):
        pixels = self.env.render(
            mode="rgb_array", width=64, height=64, camera_id=self.camera_id
        ).copy()

        pixels = pixels.flatten()
        return pixels

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.pixel:
            obs = self._transform(obs)
        elif self.hybrid:
            obs = np.concatenate([self._transform(obs), obs])
        return obs

    def step(self, action, render=False):
        next_obs, reward, done, info = self.env.step(action)
        info.update(
            coordinates=self.env.sim.data.qpos[
                :2
            ].copy(),  # (qpos[0] = xy[0], qpos[1] = xy[1])
        )
        if render:
            info["render"] = self.env.render(
                width=256, height=256, mode="rgb_array", camera_id=self.camera_id
            ).transpose(2, 0, 1)

        if self.pixel:
            next_obs = self._transform(next_obs)
        elif self.hybrid:
            next_obs = np.concatenate([self._transform(next_obs), next_obs])
        return next_obs, reward, done, info

    def plot_trajectory(self, trajectory, color, ax):
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        square_axis_limit = 0.0
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            self.plot_trajectory(trajectory, color, ax)

            square_axis_limit = max(
                square_axis_limit, np.max(np.abs(trajectory[:, :2]))
            )
        square_axis_limit = square_axis_limit * 1.2

        if plot_axis == "free":
            return

        if plot_axis is None:
            plot_axis = [
                -square_axis_limit,
                square_axis_limit,
                -square_axis_limit,
                square_axis_limit,
            ]

        if plot_axis is not None:
            plot_axis = [
                min(plot_axis[0], -square_axis_limit),
                max(plot_axis[1], square_axis_limit),
                min(plot_axis[2], -square_axis_limit),
                max(plot_axis[3], square_axis_limit),
            ]
            ax.axis(plot_axis)
            ax.set_aspect("equal")
        else:
            ax.axis("scaled")

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        for trajectory in trajectories:
            if trajectory["env_infos"]["coordinates"].dtype == object:
                coordinates_trajectories.append(
                    np.concatenate(
                        [
                            np.concatenate(
                                trajectory["env_infos"]["coordinates"], axis=0
                            ),
                        ]
                    )
                )
            elif trajectory["env_infos"]["coordinates"].ndim == 2:
                coordinates_trajectories.append(
                    np.concatenate(
                        [
                            trajectory["env_infos"]["coordinates"],
                        ]
                    )
                )
            elif trajectory["env_infos"]["coordinates"].ndim > 2:
                coordinates_trajectories.append(
                    np.concatenate(
                        [
                            trajectory["env_infos"]["coordinates"].reshape(-1, 2),
                        ]
                    )
                )
            else:
                assert False
        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        return {}


class FrameStackWrapper(AkroWrapperTrait, gym.Wrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)

        self.num_frames = num_frames
        self.frames = deque([], maxlen=self.num_frames)

        self.ori_pixel_shape = self.env.ob_info["pixel_shape"]
        self.ori_flat_pixel_shape = np.prod(self.ori_pixel_shape)
        self.new_pixel_shape = (
            self.ori_pixel_shape[0],
            self.ori_pixel_shape[1],
            self.ori_pixel_shape[2] * self.num_frames,
        )

        self.action_space = self.env.action_space

        if env.ob_info["type"] == "pixel":
            self.observation_space = akro.Box(
                low=-np.inf, high=np.inf, shape=self.new_pixel_shape
            )
            self.ob_info = dict(
                type="pixel",
                pixel_shape=self.new_pixel_shape,
            )
        elif env.ob_info["type"] == "hybrid":
            self.observation_space = akro.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    np.prod(self.new_pixel_shape) + np.prod(env.ob_info["state_shape"]),
                ),
            )
            self.ob_info = dict(
                type="hybrid",
                pixel_shape=self.new_pixel_shape,
                state_shape=env.ob_info["state_shape"],
            )
        else:
            raise NotImplementedError

    def _transform_observation(self, cur_obs):
        assert len(self.frames) == self.num_frames
        obs = np.concatenate(list(self.frames), axis=2)
        return np.concatenate(
            [obs.flatten(), cur_obs[self.ori_flat_pixel_shape :]], axis=-1
        )

    def _extract_pixels(self, obs):
        pixels = obs[: self.ori_flat_pixel_shape].reshape(self.ori_pixel_shape)
        return pixels

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        pixels = self._extract_pixels(obs)
        for _ in range(self.num_frames):
            self.frames.append(pixels)
        return self._transform_observation(obs)

    def step(self, action, **kwargs):
        next_obs, reward, done, info = self.env.step(action, **kwargs)
        pixels = self._extract_pixels(next_obs)
        self.frames.append(pixels)
        return self._transform_observation(next_obs), reward, done, info


class MazeRenderWrapper(RenderWrapper):
    """Wrap general environment to match the interface for METRA. Optionally wrap to output pixels."""

    def __init__(
        self,
        env: gym.Env,
        pixel: bool = False,
        floor_color: bool = False,
        wall_color: bool = False,
        hybrid: bool = False,
        **kwargs
    ):
        super().__init__(
            env,
            pixel=pixel,
            floor_color=floor_color,
            wall_color=wall_color,
            hybrid=hybrid,
            **kwargs
        )
        if self.env.unwrapped.__class__.__name__ not in ["MazeEnv"]:
            print(self.env.unwrapped.__class__.__name__)
            print("May be incompatible")

        # Container for wall geometries data
        wall_data = []

        unwrapped_env = self.env.unwrapped
        # Iterate over all geoms in the simulation
        for i, geom_name in enumerate(unwrapped_env.sim.model.geom_names):
            # Check if 'wall' is in the geom's name
            if "wall" in geom_name or "block" in geom_name:
                # Get the position and size of the geometry
                pos = unwrapped_env.sim.model.geom_pos[i]
                if unwrapped_env.__class__.__name__ == "MazeEnv":
                    pos = pos - np.array([1.2, 1.2, 0])
                else:
                    pass
                size = unwrapped_env.sim.model.geom_size[i]

                # Store the data together
                wall_data.append((geom_name, pos, size))

        assert len(wall_data) != 0, "No walls found in the environment"
        self.wall_data = wall_data

    def scatter_trajectory(self, trajectory, color, ax):
        for name, pos, size in self.wall_data:
            rectangle = plt.Rectangle(
                (pos[0] - size[0], pos[1] - size[1]),
                2 * size[0],
                2 * size[1],
                linewidth=0.01,
                edgecolor="none",
                facecolor="gray",
            )
            ax.add_patch(rectangle)

        ax.scatter(trajectory[:, 0], trajectory[:, 1], color=color, s=0.1)
        ax.set_aspect("equal")

    def plot_trajectory(self, trajectory, color, ax):
        for name, pos, size in self.wall_data:
            rectangle = plt.Rectangle(
                (pos[0] - size[0], pos[1] - size[1]),
                2 * size[0],
                2 * size[1],
                linewidth=0.01,
                edgecolor="none",
                facecolor="gray",
            )
            ax.add_patch(rectangle)

        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        square_axis_limit = 0.0
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            self.plot_trajectory(trajectory, color, ax)

            square_axis_limit = max(
                square_axis_limit, np.max(np.abs(trajectory[:, :2]))
            )
        square_axis_limit = square_axis_limit * 1.2

        plot_axis = [
            min([pos[0] for n, pos, size in self.wall_data]),
            max([pos[0] for n, pos, size in self.wall_data]),
            min([pos[1] for n, pos, size in self.wall_data]),
            max([pos[1] for n, pos, size in self.wall_data]),
        ]

        ax.axis(plot_axis)
        ax.set_aspect("equal")

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        for trajectory in trajectories:
            if trajectory["env_infos"]["coordinates"].dtype == object:
                coordinates_trajectories.append(
                    np.concatenate(
                        [
                            np.concatenate(
                                trajectory["env_infos"]["coordinates"], axis=0
                            ),
                        ]
                    )
                )
            elif trajectory["env_infos"]["coordinates"].ndim == 2:
                coordinates_trajectories.append(
                    np.concatenate(
                        [
                            trajectory["env_infos"]["coordinates"],
                        ]
                    )
                )
            elif trajectory["env_infos"]["coordinates"].ndim > 2:
                coordinates_trajectories.append(
                    np.concatenate(
                        [
                            trajectory["env_infos"]["coordinates"].reshape(-1, 2),
                        ]
                    )
                )
            else:
                assert False
        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        return {}
