import numpy as np


class DummyEnv():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_state = None
        self.last_ob = None
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}

    # @property
    # def observation_space(self):
    #     return akro.Box(low=-np.inf, high=np.inf, shape=(64, 64, 3))

    # def get_state(self, state):
    #     image = state['image']
    #     return image.flatten()

    def reset(self):
        state = super().reset()
        ob = self.get_state(state)
        self.last_state = state
        self.last_ob = ob
        return ob

    def step(self, action, render=False):
        next_state, reward, done, info = super().step(action)
        ob = self.get_state(next_state)

        coords = self.last_state['state'][:2].copy()
        next_coords = next_state['state'][:2].copy()
        info['coordinates'] = coords
        info['next_coordinates'] = next_coords
        info['ori_obs'] = self.last_state['state']
        info['next_ori_obs'] = next_state['state']
        if render:
            info['render'] = next_state['image'].transpose(2, 0, 1)

        self.last_state = next_state
        self.last_ob = ob

        return ob, reward, done, info

    def plot_trajectory(self, trajectory, color, ax):
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        square_axis_limit = 0.0
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            self.plot_trajectory(trajectory, color, ax)

            square_axis_limit = max(square_axis_limit, np.max(np.abs(trajectory[:, :2])))
        square_axis_limit = square_axis_limit * 1.2

        if plot_axis == 'free':
            return

        if plot_axis is None:
            plot_axis = [-square_axis_limit, square_axis_limit, -square_axis_limit, square_axis_limit]

        if plot_axis is not None:
            ax.axis(plot_axis)
            ax.set_aspect('equal')
        else:
            ax.axis('scaled')

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        for trajectory in trajectories:
            if trajectory['env_infos']['coordinates'].dtype == object:
                coordinates_trajectories.append(np.concatenate([
                    np.concatenate(trajectory['env_infos']['coordinates'], axis=0),
                    [trajectory['env_infos']['next_coordinates'][-1][-1]],
                ]))
            elif trajectory['env_infos']['coordinates'].ndim == 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'],
                    [trajectory['env_infos']['next_coordinates'][-1]]
                ]))
            elif trajectory['env_infos']['coordinates'].ndim > 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'].reshape(-1, 2),
                    trajectory['env_infos']['next_coordinates'].reshape(-1, 2)[-1:]
                ]))
            else:
                assert False
        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories, coord_dims=None):
        eval_metrics = {}

        goal_names = ['BottomBurner', 'LightSwitch', 'SlideCabinet', 'HingeCabinet', 'Microwave', 'Kettle']

        sum_successes = 0
        for i, goal_name in enumerate(goal_names):
            success = 0
            for traj in trajectories:
                success = max(success, traj['env_infos'][f'metric_success_task_relevant/goal_{i}'].max())
            eval_metrics[f'KitchenTask{goal_name}'] = success
            sum_successes += success
        eval_metrics[f'KitchenOverall'] = sum_successes

        return eval_metrics

import gym
import d4rl
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # env = gym.make('antmazewogoal-umaze-v2')
    env = gym.make('antmaze-umaze-v2')
    # env.viewer.cam.elevation = -100
    env.reset()
    res = env.render(mode='rgb_array')
    # env._get_viewer('rgb_array').render(width=500, height=500, camera_id=None)
    # res = env._get_viewer('rgb_array').read_pixels(width=500, height=500, depth=False)
    # camera_name='track' # check https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/locomotion/mujoco_goal_env.py#L148
    # camera_id = env.model.camera_name2id(camera_name)
    # print(camera_id)
    # res = env.render(mode='rgb_array', width=256, height=256, camera_id=camera_id)
    # env.camera._camera.set_state(lookat=(0, 0, 0), distance=10, elevation=-90, azimuth=0)
    # a = env.step(env.action_space.sample(), render=True)
    plt.imshow(res)
    plt.show()
    plt.savefig('maze.png')
