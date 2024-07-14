from collections import defaultdict, deque

import copy
import numpy as np
import torch

import global_context
import dowel_wrapper
from dowel import Histogram
from garage import TrajectoryBatch
from garage.misc import tensor_utils
from garage.np.algos.rl_algorithm import RLAlgorithm
from garagei import log_performance_ex
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garagei.torch.utils import compute_total_norm
from iod.utils import MeasureAndAccTime


class IOD(RLAlgorithm):
    def __init__(
        self,
        *,
        env_name,
        algo,
        env_spec,
        option_policy,
        traj_encoder,
        dual_lam,
        optimizer,
        alpha,
        max_path_length,
        n_epochs_per_eval,
        n_epochs_per_log,
        n_epochs_per_tb,
        n_epochs_per_save,
        n_epochs_per_pt_save,
        n_epochs_per_pkl_update,
        dim_option,
        num_random_trajectories,
        num_video_repeats,
        eval_record_video,
        video_skip_frames,
        eval_plot_axis,
        name="IOD",
        device=torch.device("cpu"),
        sample_cpu=True,
        num_train_per_epoch=1,
        discount=0.99,
        sd_batch_norm=False,
        trans_minibatch_size=None,
        trans_optimization_epochs=None,
        discrete=False,
        unit_length=False,
        save_replay_buffer=False,
    ):
        self.env_name = env_name
        self.algo = algo

        self.discount = discount
        self.max_path_length = max_path_length

        self.device = device
        self.sample_cpu = sample_cpu
        self.option_policy = option_policy.to(self.device)
        self.traj_encoder = traj_encoder.to(self.device)

        self.dual_lam = dual_lam.to(self.device)
        self.param_modules = {
            "traj_encoder": self.traj_encoder,
            "option_policy": self.option_policy,
            "dual_lam": self.dual_lam,
        }
        self.alpha = alpha
        self.name = name

        self.dim_option = dim_option

        self._num_train_per_epoch = num_train_per_epoch
        self._env_spec = env_spec

        self.n_epochs_per_eval = n_epochs_per_eval
        self.n_epochs_per_log = n_epochs_per_log
        self.n_epochs_per_tb = n_epochs_per_tb
        self.n_epochs_per_save = n_epochs_per_save
        self.n_epochs_per_pt_save = n_epochs_per_pt_save
        self.n_epochs_per_pkl_update = n_epochs_per_pkl_update
        self.num_random_trajectories = num_random_trajectories
        self.num_video_repeats = num_video_repeats
        self.eval_record_video = eval_record_video
        self.video_skip_frames = video_skip_frames
        self.eval_plot_axis = eval_plot_axis

        assert isinstance(optimizer, OptimizerGroupWrapper)
        self._optimizer = optimizer

        self._trans_minibatch_size = trans_minibatch_size
        self._trans_optimization_epochs = trans_optimization_epochs

        self.discrete = discrete
        self.unit_length = unit_length

        self.traj_encoder.eval()

        self.coverage_queue = deque([], maxlen=100000 // self.max_path_length)
        self.coverage_log = []
        self.coincidental_goal_success = [0 for _ in range(6)]

        self.save_replay_buffer = save_replay_buffer

    @property
    def policy(self):
        raise NotImplementedError()

    def all_parameters(self):
        for m in self.param_modules.values():
            for p in m.parameters():
                yield p

    def train_once(self, itr, paths, runner, extra_scalar_metrics={}):
        logging_enabled = (runner.step_itr + 1) % self.n_epochs_per_log == 0

        data = self.process_samples(paths)

        time_computing_metrics = [0.0]
        time_training = [0.0]

        with MeasureAndAccTime(time_training):
            tensors = self._train_once_inner(data)

        # add to queue if there's queue
        if self.env_name == "kitchen":
            goal_names = [
                "BottomBurner",
                "LightSwitch",
                "SlideCabinet",
                "HingeCabinet",
                "Microwave",
                "Kettle",
            ]

            for path in paths:
                successes = []
                for i, goal_name in enumerate(goal_names):
                    success = path["env_infos"][
                        f"metric_success_task_relevant/goal_{i}"
                    ].max()
                    successes.append(success)
                    self.coincidental_goal_success[i] += success
                self.coverage_queue.append(successes)
                self.coverage_log.append(successes)

        elif self.env_name in [
            "ant",
            "half_cheetah",
            "dmc_quadruped",
            "dmc_humanoid",
            "walker",
            "antmaze-original",
            "antmaze-large-play",
            "antmaze-ultra-play",
            "pointmaze-large",
            "dmc_humanoid_state",
            "dmc_quadruped_state_escape",
            "fetchpush",
        ]:
            if self.env_name in ["half_cheetah", "walker"]:
                coord_dims = [0]
            else:
                coord_dims = [0, 1]
            for i, path in enumerate(paths):
                self.coverage_queue.append(
                    path["env_infos"]["coordinates"][:, coord_dims]
                )
                self.coverage_log.append(
                    path["env_infos"]["coordinates"][:, coord_dims]
                )

        else:
            raise NotImplementedError
            print("You may wanna add here")

        performence = log_performance_ex(
            itr,
            TrajectoryBatch.from_trajectory_list(self._env_spec, paths),
            discount=self.discount,
        )
        discounted_returns = performence["discounted_returns"]
        undiscounted_returns = performence["undiscounted_returns"]

        if logging_enabled:
            prefix_tabular = global_context.get_metric_prefix()
            with dowel_wrapper.get_tabular().prefix(
                prefix_tabular + self.name + "/"
            ), dowel_wrapper.get_tabular("plot").prefix(
                prefix_tabular + self.name + "/"
            ):

                def _record_scalar(key, val):
                    dowel_wrapper.get_tabular().record(key, val)

                def _record_histogram(key, val):
                    dowel_wrapper.get_tabular("plot").record(key, Histogram(val))

                for k in tensors.keys():
                    if tensors[k].numel() == 1:
                        _record_scalar(f"{k}", tensors[k].item())
                    else:
                        _record_scalar(
                            f"{k}",
                            np.array2string(
                                tensors[k].detach().cpu().numpy(), suppress_small=True
                            ),
                        )
                with torch.no_grad():
                    total_norm = compute_total_norm(self.all_parameters())
                    _record_scalar("TotalGradNormAll", total_norm.item())
                    for key, module in self.param_modules.items():
                        total_norm = compute_total_norm(module.parameters())
                        _record_scalar(
                            f'TotalGradNorm{key.replace("_", " ").title().replace(" ", "")}',
                            total_norm.item(),
                        )
                for k, v in extra_scalar_metrics.items():
                    _record_scalar(k, v)
                _record_scalar("TimeComputingMetrics", time_computing_metrics[0])
                _record_scalar("TimeTraining", time_training[0])

                path_lengths = [len(path["actions"]) for path in paths]
                _record_scalar("PathLengthMean", np.mean(path_lengths))
                _record_scalar("PathLengthMax", np.max(path_lengths))
                _record_scalar("PathLengthMin", np.min(path_lengths))

                _record_histogram(
                    "ExternalDiscountedReturns", np.asarray(discounted_returns)
                )
                _record_histogram(
                    "ExternalUndiscountedReturns", np.asarray(undiscounted_returns)
                )

        return np.mean(undiscounted_returns)

    def train(self, runner):
        last_return = None

        with global_context.GlobalContext({"phase": "train", "policy": "sampling"}):
            for _ in runner.step_epochs(
                full_tb_epochs=0,
                log_period=self.n_epochs_per_log,
                tb_period=self.n_epochs_per_tb,
                pt_save_period=self.n_epochs_per_pt_save,
                pkl_update_period=self.n_epochs_per_pkl_update,
                new_save_period=self.n_epochs_per_save,
                save_replay_buffer=self.save_replay_buffer,
            ):
                for p in self.policy.values():
                    p.eval()
                self.traj_encoder.eval()

                if (
                    self.n_epochs_per_eval != 0
                    and runner.step_itr % self.n_epochs_per_eval == 0
                ):
                    self._evaluate_policy(runner)

                for p in self.policy.values():
                    p.train()
                self.traj_encoder.train()

                for _ in range(self._num_train_per_epoch):
                    time_sampling = [0.0]
                    with MeasureAndAccTime(time_sampling):
                        runner.step_path = self._get_train_trajectories(runner)
                    last_return = self.train_once(
                        runner.step_itr,
                        runner.step_path,
                        runner,
                        extra_scalar_metrics={
                            "TimeSampling": time_sampling[0],
                        },
                    )
                runner.step_itr += 1

        return last_return

    def _get_trajectories(
        self,
        runner,
        sampler_key,
        batch_size=None,
        extras=None,
        update_stats=False,
        worker_update=None,
        env_update=None,
    ):
        if batch_size is None:
            batch_size = len(extras)
        policy_sampler_key = (
            sampler_key[6:] if sampler_key.startswith("local_") else sampler_key
        )
        time_get_trajectories = [0.0]
        with MeasureAndAccTime(time_get_trajectories):
            trajectories, infos = runner.obtain_exact_trajectories(
                runner.step_itr,
                sampler_key=sampler_key,
                batch_size=batch_size,
                agent_update=self._get_policy_param_values(policy_sampler_key),
                env_update=env_update,
                worker_update=worker_update,
                extras=extras,
                update_stats=update_stats,
            )
        print(f"_get_trajectories({sampler_key}) {time_get_trajectories[0]}s")

        for traj in trajectories:
            for key in ["ori_obs", "next_ori_obs", "coordinates", "next_coordinates"]:
                if key not in traj["env_infos"]:
                    continue

        return trajectories

    def _get_train_trajectories(self, runner):
        default_kwargs = dict(
            runner=runner,
            update_stats=True,
            worker_update=dict(
                _render=False,
                _deterministic_policy=False,
            ),
            env_update=dict(_action_noise_std=None),
        )
        kwargs = dict(default_kwargs, **self._get_train_trajectories_kwargs(runner))

        paths = self._get_trajectories(**kwargs)

        return paths

    def process_samples(self, paths):
        data = defaultdict(list)
        for path in paths:
            data["obs"].append(path["observations"])
            data["next_obs"].append(path["next_observations"])
            data["actions"].append(path["actions"])
            data["rewards"].append(path["rewards"])
            data["dones"].append(path["dones"])
            data["returns"].append(
                tensor_utils.discount_cumsum(path["rewards"], self.discount)
            )

            if "pre_tanh_value" in path["agent_infos"]:
                data["pre_tanh_values"].append(path["agent_infos"]["pre_tanh_value"])
            if "log_prob" in path["agent_infos"]:
                data["log_probs"].append(path["agent_infos"]["log_prob"])
            if "option" in path["agent_infos"]:
                data["options"].append(path["agent_infos"]["option"])
                data["next_options"].append(
                    np.concatenate(
                        [
                            path["agent_infos"]["option"][1:],
                            path["agent_infos"]["option"][-1:],
                        ],
                        axis=0,
                    )
                )

            if "coordinates" in path["env_infos"]:  # for visualization purposes
                data["coordinates"].append(path["env_infos"]["coordinates"])

            if "exploration_type" in path:
                data["exploration_type"].append(path["exploration_type"])

            if "cur_exploration" in path["agent_infos"]:
                data["cur_exploration"].append(path["agent_infos"]["cur_exploration"])

        return data

    def _get_policy_param_values(self, key):
        param_dict = self.policy[key].get_param_values()
        for k in param_dict.keys():
            if self.sample_cpu:
                param_dict[k] = param_dict[k].detach().cpu()
            else:
                param_dict[k] = param_dict[k].detach()
        return param_dict

    def _generate_option_extras(self, options):
        return [{"option": option, "exploration_type": 0} for option in options]

    def _gradient_descent(self, loss, optimizer_keys, clip_grad=False):
        self._optimizer.zero_grad(keys=optimizer_keys)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(
                # do not include q function, only for traj_encoder somehow
                self._optimizer.target_parameters(keys=optimizer_keys),
                5.0,
            )

        self._optimizer.step(keys=optimizer_keys)

    def _log_eval_metrics(self, runner):
        runner.eval_log_diagnostics()
        runner.plot_log_diagnostics()
