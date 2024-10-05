from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch

from iod.utils import get_option_colors
from iod.utils import record_video


def get_skill_options(self):  # self: algo (METRA)
    if self.discrete:
        eye_options = np.eye(self.dim_option)
        random_options = []
        colors = []
        for i in range(self.dim_option):
            num_trajs_per_option = self.num_random_trajectories // self.dim_option + (
                i < self.num_random_trajectories % self.dim_option
            )
            for _ in range(num_trajs_per_option):
                random_options.append(eye_options[i])
                colors.append(i)
        random_options = np.array(random_options)
        colors = np.array(colors)
        num_evals = len(random_options)

        cmap = "tab10" if self.dim_option <= 10 else "tab20"
        random_option_colors = []
        for i in range(num_evals):
            random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
        random_option_colors = np.array(random_option_colors)
    else:
        random_options = np.random.randn(self.num_random_trajectories, self.dim_option)
        if self.unit_length:
            random_options = random_options / np.linalg.norm(
                random_options, axis=1, keepdims=True
            )
        random_option_colors = get_option_colors(random_options * 4)

    return random_options, random_option_colors


def get_goal_options(self, runner):
    random_options_list = []
    random_option_colors_list = []
    goal_coordinates_list = []
    cmap = plt.get_cmap("coolwarm")
    for i in range(self.num_random_trajectories // runner._train_args.batch_size):
        goals_info = self.get_random_goals(runner._train_args.batch_size)
        random_options = goals_info["goals"]
        goal_coordinates = goals_info["coordinates"]
        random_option_colors = cmap(
            np.arange(len(random_options)) / len(random_options)
        )

        random_options_list.append(random_options)
        random_option_colors_list.append(random_option_colors)
        goal_coordinates_list.append(goal_coordinates)

    random_options = np.concatenate(random_options_list, axis=0)
    random_option_colors = np.concatenate(random_option_colors_list, axis=0)
    goal_coordinates = np.concatenate(goal_coordinates_list, axis=0)

    return random_options, random_option_colors, goal_coordinates


def draw_trajectories(
    self, runner, trajectories, option_colors, ax, goal_coordinates=None
):
    runner._env.render_trajectories(
        trajectories, option_colors, self.eval_plot_axis, ax
    )
    if goal_coordinates is not None:
        # render the goals
        if len(runner._env.observation_space.shape) == 1 and any(
            x in self.env_name for x in ["half_cheetah", "walker"]
        ):
            # get_ylim
            ymin, ymax = ax.get_ylim()
            for i, (goal, color) in enumerate(zip(goal_coordinates, option_colors)):
                ax.plot(
                    goal[0],
                    (i - len(goal_coordinates) / 2) / 1.25,
                    "*",
                    color=color,
                    markersize=3,
                )
        else:
            for goal, color in zip(goal_coordinates, option_colors):
                ax.plot(goal[0], goal[1], "*", color=color, markersize=10)


def get_eval_goals(self, runner):
    self.goal_range = None
    if self.env_name in [
        "half_cheetah",
        "ant",
        "dmc_quadruped",
        "dmc_humanoid",
        "dmc_humanoid_state",
    ]:
        if self.env_name == "half_cheetah":
            self.goal_range = 100
        elif self.env_name == "ant":
            self.goal_range = 50
        elif self.env_name == "dmc_quadruped":
            self.goal_range = 15
        elif self.env_name == "dmc_humanoid":
            self.goal_range = 10
        elif self.env_name == "dmc_humanoid_state":
            self.goal_range = 40

    # Goal-conditioned metrics
    env = runner._env
    goals = []  # list of (goal_obs, goal_info)
    if self.env_name == "kitchen":
        # goal_names = [
        #     "BottomBurner",
        #     "LightSwitch",
        #     "SlideCabinet",
        #     "HingeCabinet",
        #     "Microwave",
        #     "Kettle",
        # ]
        goal_names = [
            "_".join(c) for c in env.goal_configs
        ]  # [['bottom_burner'], ['light_switch'], ['slide_cabinet'], ['hinge_cabinet'], ['microwave'], ['kettle'], ['light_switch', 'slide_cabinet'], ['light_switch', 'hinge_cabinet'], ['light_switch', 'kettle'], ['slide_cabinet', 'hinge_cabinet'], ['slide_cabinet', 'kettle'], ['hinge_cabinet', 'kettle']]
        for i, goal_name in enumerate(goal_names):
            goal_obs = env.render_goal(goal_idx=i).copy().astype(np.float32)
            goal_obs = np.tile(goal_obs, self.frame_stack or 1).flatten()
            goals.append((goal_obs, {"goal_idx": i, "goal_name": goal_name}))
    elif self.env_name in ["antmaze-large-play", "antmaze-ultra-play"]:
        env.reset()

        base_observation = env.unwrapped._get_obs().astype(np.float32)
        print("env goals:", env.unwrapped.goals)
        for i, env_goal in enumerate(env.unwrapped.goals):
            obs_goal = base_observation.copy()
            obs_goal[:2] = env_goal
            goals.append((obs_goal, {"env_goal": env_goal}))
            # modify the environment to show the goal

    elif self.env_name in ["dmc_cheetah", "dmc_quadruped", "dmc_humanoid"]:
        for i in range(20):
            env.reset()
            state = env.physics.get_state().copy()
            if self.env_name == "dmc_cheetah":
                goal_loc = (np.random.rand(1) * 2 - 1) * self.goal_range
                state[:1] = goal_loc
            else:
                goal_loc = (np.random.rand(2) * 2 - 1) * self.goal_range
                state[:2] = goal_loc
            env.physics.set_state(state)
            if self.env_name == "dmc_humanoid":
                for _ in range(50):
                    env.step(np.zeros_like(env.action_space.sample()))
            else:
                env.step(np.zeros_like(env.action_space.sample()))
            goal_obs = (
                env.render(mode="rgb_array", width=64, height=64)
                .copy()
                .astype(np.float32)
            )
            goal_obs = np.tile(goal_obs, self.frame_stack or 1).flatten()
            goals.append((goal_obs, {"goal_loc": goal_loc}))

    elif self.env_name in ["dmc_humanoid_state"]:
        for i in range(20):
            ob = env.reset()
            state = env.physics.get_state().copy()
            goal_loc = (np.random.rand(2) * 2 - 1) * self.goal_range
            state[:2] = goal_loc
            env.physics.set_state(state)
            
            for _ in range(50):
                ob, *_ = env.step(np.zeros_like(env.action_space.sample()))

            goal_obs = ob.copy().astype(np.float32)
            goals.append((goal_obs, {"goal_loc": goal_loc}))

    elif self.env_name in [
        "ant",
        "ant_pixel",
        "half_cheetah",
    ]:
        for i in range(20):
            ob = env.reset()
            state = env.unwrapped._get_obs().copy()
            if self.env_name in ["half_cheetah"]:
                goal_loc = (np.random.rand(1) * 2 - 1) * self.goal_range
                state[:1] = goal_loc
                env.set_state(state[:9], state[9:])
            else:
                goal_loc = (np.random.rand(2) * 2 - 1) * self.goal_range
                state[:2] = goal_loc
                env.set_state(state[:15], state[15:])
            for _ in range(5):
                env.step(np.zeros_like(env.action_space.sample()))
            if self.env_name == "ant_pixel":
                goal_obs = (
                    env.render(mode="rgb_array", width=64, height=64)
                    .copy()
                    .astype(np.float32)
                )
                goal_obs = np.tile(goal_obs, self.frame_stack or 1).flatten()
            else:
                goal_obs = env._apply_normalize_obs(state).astype(np.float32)
            goals.append((goal_obs, {"goal_loc": goal_loc}))
    else:
        goals = []

    return goals


def run_eval(
    self, env, option, deterministic_policy=True, render_step=3, goal_obs=None
):
    """Returns: dict(render, coordinates)"""
    obs = env.reset()
    if option is None:
        assert goal_obs is not None
        option = goal_obs

    step = 0
    done = False
    render = []
    coordinate = []
    while step < self.max_path_length and not done:
        self.option_policy._force_use_mode_actions = deterministic_policy
        if not self.goal_reaching and goal_obs is not None:
            te_input = torch.from_numpy(np.stack([obs, goal_obs])).to(
                self.device
            )  # 2, d
            phi_s, phi_g = self.traj_encoder(self._restrict_te_obs(te_input)).mean
            phi_s, phi_g = (
                phi_s.detach().cpu().numpy(),
                phi_g.detach().cpu().numpy(),
            )
            if self.discrete:
                option = np.eye(self.dim_option)[(phi_g - phi_s).argmax()]
            else:
                option = (phi_g - phi_s) / np.linalg.norm(phi_g - phi_s)
        action, agent_info = self.option_policy.get_action(
            np.concatenate([obs, option])
        )
        next_obs, reward, done, info = env.step(action, render=False)
        obs = next_obs

        step += 1

        if step % render_step == 0:
            env_name = self.env_name

            # re-render of possible
            render_image = env.render(mode="rgb_array", width=640, height=480).copy()
            assert render_image.shape == (480, 640, 3), render_image.shape

            render.append(render_image)
            coordinate.append(info["coordinates"])

    return {"render": render, "coordinate": coordinate}


def _get_goal_conditioned_metrics(self, runner):
    eval_option_metrics = {}

    goals = get_eval_goals(self, runner)

    renders = []
    coordinates = []

    env = runner._env
    goal_metrics = defaultdict(list)
    for method in ["Adaptive"] if self.discrete else [""]:
        self.option_policy._force_use_mode_actions = True
        if len(goals) == 0:
            break

        for goal_idx, (goal_obs, goal_info) in enumerate(goals):
            if self.env_name in ["antmaze-large-play", "antmaze-ultra-play"]:
                env.unwrapped.set_target(goal_info["env_goal"])

            obs = env.reset()

            step = 0
            done = False
            success = 0
            option = None
            render = []
            coordinate = []
            while step < self.max_path_length and not done:
                if self.goal_reaching:
                    option = goal_obs
                else:
                    te_input = torch.from_numpy(np.stack([obs, goal_obs])).to(
                        self.device
                    )  # 2, d
                    phi_s, phi_g = self.traj_encoder(te_input).mean
                    phi_s, phi_g = (
                        phi_s.detach().cpu().numpy(),
                        phi_g.detach().cpu().numpy(),
                    )
                    if self.discrete:
                        if method == "Adaptive":
                            option = np.eye(self.dim_option)[(phi_g - phi_s).argmax()]
                        else:
                            if option is None:
                                option = np.eye(self.dim_option)[
                                    (phi_g - phi_s).argmax()
                                ]
                    else:
                        option = (phi_g - phi_s) / np.linalg.norm(phi_g - phi_s)

                action, agent_info = self.option_policy.get_action(
                    np.concatenate([obs, option])
                )
                next_obs, reward, done, info = env.step(action, render=True)
                obs = next_obs

                if self.env_name == "kitchen":
                    success = max(
                        success, env.compute_success(goal_info["goal_idx"])[0]
                    )
                elif self.env_name in [
                    "antmaze-large-play",
                    "antmaze-ultra-play",
                ]:
                    success = max(success, reward)  # assume sparse reward

                step += 1

                coordinate.append(info["coordinates"])

                render_step_interval = 1
                if step % render_step_interval == 0:
                    env_name = self.env_name

                    render_image = info["render"]
                    assert render_image.ndim == 3
                    if render_image.shape[-1] == 3:  # 64, 64, 3
                        render_image = render_image.transpose(2, 0, 1)

                    if self.env_name in ["antmaze-large-play", "antmaze-ultra-play"]:
                        # Turn render by 90 degrees
                        render_image = np.rot90(render_image, 1, axes=(-2, -1))

                    render.append(render_image)

            # render video
            renders.append(np.stack(render, axis=0))
            coordinates.append((np.stack(coordinate, axis=0), goal_info))

            # update metrics
            if self.env_name == "kitchen":
                goal_metrics[f'Kitchen{method}Goal{goal_info["goal_name"]}'].append(
                    success
                )
                goal_metrics[f"Kitchen{method}GoalOverall"].append(
                    success * len(goals)
                )  # we calculate the mean: so we multiply by the number of goals
                if goal_info["goal_idx"] < 6:
                    goal_metrics[f"Kitchen{method}GoalOverall6"].append(
                        success * len(goals[:6])
                    )
            elif self.env_name in ["antmaze-large-play", "antmaze-ultra-play"]:
                goal_metrics[f'Antmaze{method}Goal{goal_info["env_goal"]}'].append(
                    success
                )
                goal_metrics[f"Antmaze{method}GoalOverall"].append(success * len(goals))
            elif self.env_name in [
                "dmc_cheetah",
                "dmc_quadruped",
                "dmc_humanoid",
                "ant",
                "half_cheetah",
                "dmc_humanoid_state",
            ]:
                if self.env_name in ["dmc_cheetah"]:
                    cur_loc = env.physics.get_state()[:1]
                elif self.env_name in [
                    "dmc_quadruped",
                    "dmc_humanoid",
                    "dmc_humanoid_state",
                ]:
                    cur_loc = env.physics.get_state()[:2]
                elif self.env_name in ["half_cheetah"]:
                    cur_loc = env.unwrapped._get_obs()[:1]
                else:
                    cur_loc = env.unwrapped._get_obs()[:2]
                distance = np.linalg.norm(cur_loc - goal_info["goal_loc"])
                squared_distance = distance**2
                goal_metrics[f"Goal{method}Distance"].append(distance)
                goal_metrics[f"Goal{method}SquaredDistance"].append(squared_distance)

        # pack "renders" as trajectory: List[{"env_infos": {"render": np.ndarray}]
        trajectories = [
            {
                "env_infos": {"render": render},
                "agent_infos": {"cur_exploration": np.zeros(len(render))},
            }
            for render in renders
        ]

        if self.env_name in [
            "antmaze-large-play",
            "antmaze-ultra-play",
            "dmc_humanoid_state",
            "dmc_quadruped_state_escape",
        ]:
            goal_videos = []

            # set goal_images if possible
            for i, (goal_obs, goal_info) in enumerate(goals):
                goal_coords = (
                    goal_info["env_goal"]
                    if "env_goal" in goal_info
                    else goal_info["goal_loc"]
                )
                if self.env_name in ["half_cheetah"]:
                    goal_coords = np.zeros(2)
                    goal_coords[0] = goal_info["goal_loc"]

                # get goal images figure
                fig = Figure()
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111, frameon=False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

                if (
                    "antmaze-large-play" in self.env_name
                    or "antmaze-ultra-play" in self.env_name
                ):
                    env.scatter_trajectory(
                        np.array(goal_coords)[None], color="r", ax=ax
                    )
                else:
                    # set axis limits
                    if env_name in ["dmc_quadruped_state_escape"]:
                        self.goal_range = 12
                    ax.set_xlim(-self.goal_range, self.goal_range)
                    ax.set_ylim(-self.goal_range, self.goal_range)
                ax.scatter(goal_coords[0], goal_coords[1], color="b", marker="*", s=200)

                if not "maze" in self.env_name:
                    ax.set_aspect("equal", "box")

                def get_gc_plot(fig, ax, traj_coords):  # L, 2
                    ax.scatter(traj_coords[:, 0], traj_coords[:, 1], c="r", s=10)

                    # save it as numpy array
                    fig.canvas.draw()
                    gc_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    gc_plot_img = gc_plot.reshape(
                        fig.canvas.get_width_height()[::-1] + (3,)
                    )

                    if self.env_name in [
                        "antmaze-large-play",
                        "antmaze-ultra-play",
                        "dmc_quadruped_state_escape",
                        "dmc_humanoid_state",
                    ]:
                        # rotate the frame as counterclockwise
                        gc_plot_img = np.rot90(gc_plot_img, 1, axes=(0, 1))
                    return gc_plot_img

                traj_coords, goal_info = coordinates[i]
                gc_img = get_gc_plot(fig, ax, traj_coords)

                assert gc_img.ndim == 3
                goal_videos.append(gc_img)

            goal_images = np.stack(goal_videos, axis=0)
            goal_images = goal_images.transpose(0, 3, 1, 2)

        elif self.env_name in [
            "dmc_cheetah",
            "dmc_quadruped",
            "kitchen",
            "dmc_humanoid_state",
        ]:
            goal_images = None
        else:
            goal_images = None

        record_video(
            runner,
            f"Video_{method}_GoalConditioned",
            trajectories,
            n_cols=6,
            skip_frames=self.video_skip_frames,
            goal_images=goal_images,
        )

    goal_metrics = {key: np.mean(value) for key, value in goal_metrics.items()}
    eval_option_metrics.update(goal_metrics)

    return eval_option_metrics


def _get_train_coverage_metrics(self, runner):
    train_coverage_metrics = {}

    # Train coverage metric
    if len(self.coverage_queue) > 0:
        coverage_data = np.array(self.coverage_queue)
        if self.env_name == "kitchen":
            assert (
                coverage_data.ndim == 2 and coverage_data.shape[-1] == 6
            ), coverage_data.shape
            coverage = coverage_data.max(axis=0)
            goal_names = [
                "BottomBurner",
                "LightSwitch",
                "SlideCabinet",
                "HingeCabinet",
                "Microwave",
                "Kettle",
            ]

            for i, goal_name in enumerate(goal_names):
                train_coverage_metrics[f"TrainKitchenTask{goal_name}"] = coverage[i]
                train_coverage_metrics[f"CoincidentalRate{goal_name}"] = (
                    self.coincidental_goal_success[i]
                )
            train_coverage_metrics[f"TrainKitchenOverall"] = coverage.sum()

        else:
            total_coverage_data = np.array(self.coverage_log)
            assert coverage_data.ndim == 3
            assert (
                coverage_data.shape[-1] == 2
                if not self.env_name in ["half_cheetah", "walker"]
                else 1
            ), coverage_data.shape
            coverage_data = coverage_data.reshape(-1, coverage_data.shape[-1])
            total_coverage_data = total_coverage_data.reshape(
                -1, total_coverage_data.shape[-1]
            )
            uniq_coords = np.unique(np.floor(coverage_data), axis=0)
            total_uniq_coords = np.unique(np.floor(total_coverage_data), axis=0)
            train_coverage_metrics["TrainNumUniqueCoords"] = len(uniq_coords)
            train_coverage_metrics["TrainTotalNumUniqueCoords"] = len(total_uniq_coords)
            train_coverage_metrics["MaxDistFromOrigin"] = np.linalg.norm(
                coverage_data, axis=-1
            ).max(axis=0)
    else:
        if self.env_name == "kitchen":
            goal_names = [
                "BottomBurner",
                "LightSwitch",
                "SlideCabinet",
                "HingeCabinet",
                "Microwave",
                "Kettle",
            ]
            for i, goal_name in enumerate(goal_names):
                train_coverage_metrics[f"TrainKitchenTask{goal_name}"] = 0
            train_coverage_metrics[f"TrainKitchenOverall"] = 0
        else:
            train_coverage_metrics["TrainNumUniqueCoords"] = 0
            train_coverage_metrics["TrainTotalNumUniqueCoords"] = 0

    return train_coverage_metrics
