import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import (
    FigureCanvasAgg as FigureCanvas,
)
import copy

import global_context
from garagei.replay_buffer.path_buffer_ex import PathBufferEx
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
from iod.utils import (
    get_torch_concat_obs,
    FigManager,
    record_video,
)
from iod import eval_utils
from iod import apt_utils


class AgentWrapper(object):
    """Wrapper for communicating the agent weights with the sampler."""

    def __init__(self, policies):
        assert isinstance(policies, dict) and "default_policy" in policies
        self.default_policy = policies["default_policy"]
        self.exploration_policy = policies.get("exploration_policy", None)

    def get_param_values(self):
        param_dict = {}
        default_param_dict = self.default_policy.get_param_values()
        for k in default_param_dict.keys():
            param_dict[f"default_{k}"] = default_param_dict[k].detach()

        if self.exploration_policy:
            exploration_param_dict = self.exploration_policy.get_param_values()
            for k in exploration_param_dict.keys():
                param_dict[f"exploration_{k}"] = exploration_param_dict[k].detach()

        return param_dict

    def set_param_values(self, state_dict):
        default_state_dict = {}
        exploration_state_dict = {}

        for k, v in state_dict.items():
            k: str
            if k.startswith("default_"):
                default_state_dict[k.replace("default_", "", 1)] = v
            elif k.startswith("exploration_"):
                exploration_state_dict[k.replace("exploration_", "", 1)] = v
            else:
                raise ValueError(f"Unknown key: {k}")

        self.default_policy.set_param_values(default_state_dict)
        if self.exploration_policy:
            self.exploration_policy.set_param_values(exploration_state_dict)

    def eval(self):
        self.default_policy.eval()
        if self.exploration_policy:
            self.exploration_policy.eval()

    def train(self):
        self.default_policy.train()
        if self.exploration_policy:
            self.exploration_policy.train()

    def reset(self):
        self.default_policy.reset()
        if self.exploration_policy:
            self.exploration_policy.reset()


class TLDR(IOD):

    def __init__(
        self,
        *,
        qf1,
        qf2,
        log_alpha,
        tau,
        scale_reward,
        target_coef,
        replay_buffer,
        min_buffer_size,
        num_alt_samples,
        split_group,
        dual_reg,
        dual_slack,
        pixel_shape=None,
        exploration_type=0,
        exploration_policy=None,
        exploration_qf1=None,
        exploration_qf2=None,
        exploration_log_alpha=None,
        goal_reaching=0,
        frame_stack=None,
        exploration_sac_discount=0.99,
        knn_k=12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.goal_reaching = goal_reaching

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = log_alpha.to(self.device)

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha,
        )

        self.tau = tau

        self.replay_buffer: PathBufferEx = replay_buffer
        self.min_buffer_size = min_buffer_size

        self.dual_reg = dual_reg
        self.dual_slack = dual_slack

        self.num_alt_samples = num_alt_samples
        self.split_group = split_group

        self._reward_scale_factor = scale_reward
        self._target_entropy = (
            -np.prod(self._env_spec.action_space.shape).item() / 2.0 * target_coef
        )

        self.pixel_shape = pixel_shape

        assert self._trans_optimization_epochs is not None

        self.exploration_type = exploration_type
        if self.goal_reaching:
            assert exploration_policy is not None
            assert exploration_qf1 is not None
            assert exploration_qf2 is not None
            assert exploration_log_alpha is not None

            self.exploration_policy = exploration_policy.to(self.device)
            self.exploration_qf1 = exploration_qf1.to(self.device)
            self.exploration_qf2 = exploration_qf2.to(self.device)
            self.target_exploration_qf1 = copy.deepcopy(self.exploration_qf1)
            self.target_exploration_qf2 = copy.deepcopy(self.exploration_qf2)
            self.exploration_log_alpha = exploration_log_alpha.to(self.device)

            rms = apt_utils.RMS(self.device)

            knn_clip = 0.0001
            knn_k = knn_k  # adjusting this controls temperatire of goals
            knn_avg = True
            knn_rms = False  # since rms should be only used for calculating apt reward
            self.pbe = apt_utils.PBE(
                rms, knn_clip, knn_k, knn_avg, knn_rms, self.device
            )

            self.param_modules.update(
                exploration_policy=self.exploration_policy,
                exploration_qf1=self.exploration_qf1,
                exploration_qf2=self.exploration_qf2,
                exploration_log_alpha=self.exploration_log_alpha,
            )

        policy_for_agent = {
            "default_policy": self.option_policy,
        }

        if self.goal_reaching:
            policy_for_agent.update(
                {
                    "exploration_policy": self.exploration_policy,
                }
            )

        if self.goal_reaching:
            policy_for_agent.update(
                {
                    "encoder": self.traj_encoder,
                }
            )

        self.policy_for_agent = AgentWrapper(
            policies=policy_for_agent
        )  # this should be at the end

        self.frame_stack = frame_stack

        self.exploration_sac_discount = exploration_sac_discount
        self.plot_first_2dims = True

    @property
    def policy(self):
        return {"option_policy": self.policy_for_agent}

    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    def _generate_option_extras_list(self, options):
        """
        >>> a = np.zeros((3,4))
        >>> list(a)
        [array([0., 0., 0., 0.]), array([0., 0., 0., 0.]), array([0., 0., 0., 0.])]
        """
        return [{"option": list(option)} for option in options]

    def _get_train_trajectories_kwargs(self, runner, options=None):
        if options is None:
            if self.discrete:
                extras = self._generate_option_extras(
                    np.eye(self.dim_option)[
                        np.random.randint(
                            0, self.dim_option, runner._train_args.batch_size
                        )
                    ]
                )

            else:
                random_options = np.random.randn(
                    runner._train_args.batch_size, self.dim_option
                )
                if self.unit_length:
                    random_options /= np.linalg.norm(
                        random_options, axis=-1, keepdims=True
                    )
                extras = self._generate_option_extras(random_options)

            if self.goal_reaching:
                goals = self.get_random_goals(runner._train_args.batch_size)["goals"]
                assert goals.ndim == 2  # for state spaces
                for i, extra in enumerate(extras):
                    extra["option"] = goals[i]

        else:  # options is not None
            extras = self._generate_option_extras(options)

        if self.goal_reaching:
            if self.replay_buffer.n_transitions_stored < runner._train_args.batch_size:
                for i, extra in enumerate(extras):
                    extra["exploration_type"] = (
                        2  # only use exploration policy for prefilling replay buffer
                    )
            else:
                for i, extra in enumerate(extras):
                    if i % 2 == 0:
                        extra["exploration_type"] = 1  # go-explore (tldr)
                    else:
                        extra["exploration_type"] = 0  # do not use exploration policy

        return dict(
            extras=extras,
            sampler_key="option_policy",
        )

    def get_random_goals(self, size, num_batch=None):
        """Get `size` number of goals from buffer."""
        if num_batch is None:
            num_batch = self._trans_minibatch_size
        if (
            self.replay_buffer.n_transitions_stored < num_batch
        ):  # case for evaluation for the first epoch & early training
            example_ob = np.full(self._env_spec.observation_space.shape, 1000).reshape(
                -1
            )
            if len(self._env_spec.observation_space.shape) >= 3:
                example_ob = example_ob.astype(np.uint8)
            goals = example_ob[None, :].repeat(size, axis=0)
            assert goals.ndim == 2 and goals.shape[0] == size
            return {
                "goals": goals,
                "coordinates": np.zeros((size, 2)),
            }

        # get samples as tensors
        samples = self.replay_buffer.sample_transitions(num_batch)
        data = {}
        for key, value in samples.items():
            data[key] = torch.from_numpy(value).float().to(self.device)
        with torch.no_grad():
            # calculate reward of each goals
            z1 = self.traj_encoder(data["next_obs"]).mean
            reward = self.pbe.get_reward(z1, z1).squeeze(-1)
        assert reward.ndim == 1
        # select top `size` goals
        indices = torch.argsort(reward, descending=True)[:size].cpu().numpy()
        assert indices.shape == (size,)
        goals = samples["next_obs"][indices]

        # TODO: consider diversity of the goals
        return {"goals": goals, "coordinates": samples["coordinates"][indices]}

    def _flatten_data(self, data):
        epoch_data = {}
        for key, value in data.items():
            epoch_data[key] = torch.tensor(
                np.concatenate(value, axis=0), dtype=torch.float32, device=self.device
            )
        return epoch_data

    def _update_replay_buffer(self, data):
        paths = []
        # Add paths to the replay buffer
        for i in range(len(data["actions"])):
            path = {}
            for key in data.keys():
                cur_list = data[key][i]
                if cur_list.ndim == 1:
                    cur_list = cur_list[..., np.newaxis]
                path[key] = cur_list
            paths.append(path)

        for path in paths:
            self.replay_buffer.add_path(path)

    def _sample_replay_buffer(self, replay_buffer):
        if self.goal_reaching:
            samples = replay_buffer.sample_transitions_with_goals(
                self._trans_minibatch_size,
            )
        else:
            samples = replay_buffer.sample_transitions(self._trans_minibatch_size)

        data = {}
        for key, value in samples.items():
            if value.shape[1] == 1 and "option" not in key:
                value = np.squeeze(value, axis=1)
            data[key] = torch.from_numpy(value).float().to(self.device)

        return data

    def _train_once_inner(self, path_data):
        """Inner training loop"""
        self._update_replay_buffer(path_data)

        epoch_data = self._flatten_data(path_data)

        if self.goal_reaching:
            tensors = self._train_components_tldr(epoch_data)
        else:
            tensors = self._train_components_metra(epoch_data)

        return tensors

    def _train_components_tldr(self, epoch_data):
        assert self.goal_reaching

        tensors = {}
        for _ in range(self._trans_optimization_epochs):
            v = self._sample_replay_buffer(self.replay_buffer)
            self._optimize_te(tensors, v)

            with torch.no_grad():
                self._update_rewards(tensors, v)
                self._update_tldr_rewards(tensors, v)

            self._optimize_exploration_policy(tensors, v)
            self._optimize_op(tensors, v)

        return tensors

    def _train_components_metra(self, epoch_data):
        if (
            self.replay_buffer is not None
            and self.replay_buffer.n_transitions_stored < self.min_buffer_size
        ):
            return {}

        for _ in range(self._trans_optimization_epochs):
            tensors = {}  # to keep track of losses

            v = self._sample_replay_buffer(self.replay_buffer)

            self._update_rewards(tensors, v)
            self._optimize_te(tensors, v)
            self._update_rewards(tensors, v)
            self._optimize_op(tensors, v)

        return tensors

    def _update_loss_te_tldr(self, tensors, internal_vals, traj_encoder, dual_lam):
        obs = internal_vals["obs"]
        next_obs = internal_vals["next_obs"]
        goals = internal_vals["goals"]

        phi_x, phi_y, phi_g = torch.split(
            traj_encoder(torch.cat([obs, next_obs, goals], dim=0)).mean, len(obs)
        )
        squared_dist = ((phi_x - phi_g) ** 2).sum(axis=-1)  # double V network is used
        dist = torch.sqrt(
            torch.maximum(squared_dist, torch.full_like(squared_dist, 1e-6))
        )

        cst_dist = torch.ones_like(squared_dist)
        cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
        cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)

        dual_lam = dual_lam.param.exp()

        te_obj = (
            -torch.nn.functional.softplus(500 - dist, beta=0.01).mean()
            + (dual_lam.detach() * cst_penalty).mean()
        )

        internal_vals.update({"cst_penalty": cst_penalty})
        tensors.update(
            {
                "DualCstPenalty": cst_penalty.mean(),
            }
        )

        loss_te = -te_obj

        tensors.update(
            {
                "TeObjMean": te_obj.mean(),
                "LossTe": loss_te,
            }
        )

    def _optimize_te(self, tensors, internal_vars):
        if self.goal_reaching:
            self._update_loss_te_tldr(
                tensors, internal_vars, self.traj_encoder, self.dual_lam
            )
        else:
            self._update_loss_te(tensors, internal_vars)

        self._gradient_descent(
            tensors["LossTe"], optimizer_keys=["traj_encoder"], clip_grad=False
        )

        if self.dual_reg:
            self._update_loss_dual_lam(tensors, internal_vars)
            self._gradient_descent(
                tensors["LossDualLam"],
                optimizer_keys=["dual_lam"],
            )

    def _optimize_op(self, tensors, internal_vars):
        self._update_loss_qf(tensors, internal_vars)

        self._gradient_descent(
            tensors["LossQf1"] + tensors["LossQf2"],
            optimizer_keys=["qf"],
        )

        self._update_loss_op(tensors, internal_vars)
        self._gradient_descent(
            tensors["LossSacp"],
            optimizer_keys=["option_policy"],
        )

        self._update_loss_alpha(tensors, internal_vars)
        self._gradient_descent(
            tensors["LossAlpha"],
            optimizer_keys=["log_alpha"],
        )

        sac_utils.update_targets(
            self, self.qf1, self.qf2, self.target_qf1, self.target_qf2
        )

    def _optimize_exploration_policy(self, tensors, internal_vars):
        self._update_loss_qf(tensors, internal_vars, exploration=True)

        self._gradient_descent(
            tensors["exploration_LossQf1"] + tensors["exploration_LossQf2"],
            optimizer_keys=["exploration_qf"],
            clip_grad=False,
        )

        self._update_loss_op(tensors, internal_vars, exploration=True)
        self._gradient_descent(
            tensors["exploration_LossSacp"],
            optimizer_keys=["exploration_policy"],
            clip_grad=False,
        )

        self._update_loss_alpha(tensors, internal_vars, exploration=True)
        self._gradient_descent(
            tensors["exploration_LossAlpha"],
            optimizer_keys=["exploration_log_alpha"],
            clip_grad=False,
        )

        sac_utils.update_targets(
            self,
            self.exploration_qf1,
            self.exploration_qf2,
            self.target_exploration_qf1,
            self.target_exploration_qf2,
        )

    def _update_tldr_rewards(self, tensors, v):
        rep, next_rep = v["cur_z"], v["next_z"]

        rewards = self.pbe(rep, next_rep, use_rms=True)
        rewards = rewards.flatten()
        assert rewards.ndim == 1 and rewards.shape[0] == rep.shape[0]

        tensors.update(
            {
                "AptRewardMean": rewards.mean(),
                "AptRewardStd": rewards.std(),
            }
        )

        v["exploration_rewards"] = rewards

    def _update_rewards(self, tensors, v, exp_v=None):
        obs = v["obs"]
        next_obs = v["next_obs"]
        options = v["options"]
        next_options = v["next_options"]
        actions = v["actions"]
        dones = v["dones"]
        if "dones_exp" in v:  # done signal for exploration policy
            dones_exp = v["dones_exp"]

        def get_rewards(traj_encoder, obs, next_obs, options):
            assert options.ndim == 2
            ### calc target rewards
            if self.goal_reaching:
                cur_z, next_z, goal_z = torch.split(
                    traj_encoder(torch.cat([obs, next_obs, options], dim=0)).mean,
                    len(obs),
                )
                rew = torch.norm(goal_z - cur_z, dim=1) - torch.norm(
                    goal_z - next_z, dim=1
                )
                return rew, cur_z, next_z

            cur_z = traj_encoder(obs).mean
            next_z = traj_encoder(next_obs).mean

            target_z = next_z - cur_z

            if self.discrete:
                dim_option = options.shape[1]
                masks = (
                    (
                        options
                        - (options.mean(dim=1, keepdim=True) if dim_option != 1 else 0)
                    )
                    * dim_option
                    / (dim_option - 1 if dim_option != 1 else 1)
                )
                rewards = (target_z * masks).sum(dim=1)
            else:
                inner = (target_z * options).sum(dim=1)
                rewards = inner

            return rewards, cur_z, next_z

        rewards, cur_z, next_z = get_rewards(self.traj_encoder, obs, next_obs, options)

        # For dual objectives
        v.update(
            {
                "cur_z": cur_z,
                "next_z": next_z,
            }
        )

        tensors.update(
            {
                "RewardMean": rewards.mean(),
                "RewardStd": rewards.std(),
                "RewardMax": rewards.max(),
                "RewardMin": rewards.min(),
            }
        )

        v["obs"] = obs
        v["next_obs"] = next_obs
        v["options"] = options
        v["next_options"] = next_options
        v["dones"] = dones
        v["rewards"] = rewards
        v["actions"] = actions
        if "dones_exp" in v:
            v["dones_exp"] = dones_exp

    def _update_loss_te(self, tensors, v):
        rewards = v["rewards"]

        obs = v["obs"]
        next_obs = v["next_obs"]

        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs
            y = next_obs
            phi_x = self.traj_encoder(x).mean
            phi_y = self.traj_encoder(y).mean

            cst_dist = torch.ones_like(x[:, 0])

            cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)

            te_obj = rewards.mean() + (dual_lam.detach() * cst_penalty).mean()

            v.update({"cst_penalty": cst_penalty})
            tensors.update(
                {
                    "DualCstPenalty": cst_penalty.mean(),
                }
            )
        else:
            te_obj = rewards.mean()

        loss_te = -te_obj

        tensors.update(
            {
                "TeObjMean": te_obj.mean(),
                "LossTe": loss_te,
            }
        )

    def _update_loss_dual_lam(self, tensors, v, dual_lam=None):
        if dual_lam is None:
            dual_lam = self.dual_lam
        log_dual_lam = dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (v["cst_penalty"].detach()).mean()

        tensors.update(
            {
                "DualLam": dual_lam,
                "LossDualLam": loss_dual_lam,
            }
        )

    def _update_loss_qf(self, tensors, v, exploration=False):
        obs = v["obs"]
        next_obs = v["next_obs"]
        options = v["options"]
        next_options = v["next_options"]
        actions = v["actions"]
        dones = v["dones"]

        if exploration:
            options = None
            next_options = None
            dones = v["dones_exp"]

            processed_cat_obs = self.exploration_policy.process_observations(obs)
            next_processed_cat_obs = self.exploration_policy.process_observations(
                next_obs
            )

            sac_utils.update_loss_qf(
                self,
                tensors,
                v,
                obs=processed_cat_obs,
                actions=actions,
                next_obs=next_processed_cat_obs,
                dones=dones,
                rewards=v["exploration_rewards"] * self._reward_scale_factor,
                policy=self.exploration_policy,
                qf1=self.exploration_qf1,
                qf2=self.exploration_qf2,
                log_alpha=self.exploration_log_alpha,
                target_qf1=self.target_exploration_qf1,
                target_qf2=self.target_exploration_qf2,
                description_prefix="exploration_",
                discount=self.exploration_sac_discount,
            )

        else:
            processed_cat_obs = self._get_concat_obs(
                self.option_policy.process_observations(obs), options
            )
            next_options = options
            next_processed_cat_obs = self._get_concat_obs(
                self.option_policy.process_observations(next_obs), next_options
            )

            sac_utils.update_loss_qf(
                self,
                tensors,
                v,
                obs=processed_cat_obs,
                actions=actions,
                next_obs=next_processed_cat_obs,
                dones=dones,
                rewards=v["rewards"] * self._reward_scale_factor,
                policy=self.option_policy,
                qf1=self.qf1,
                qf2=self.qf2,
                log_alpha=self.log_alpha,
                target_qf1=self.target_qf1,
                target_qf2=self.target_qf2,
                description_prefix="",
                discount=self.discount,
            )

        v.update(
            {
                "processed_cat_obs": processed_cat_obs,
                "next_processed_cat_obs": next_processed_cat_obs,
            }
        )

    def _update_loss_op(self, tensors, v, exploration=False):
        obs = v["obs"]
        options = v["options"]

        if exploration:
            processed_cat_obs = self.exploration_policy.process_observations(obs)
            sac_utils.update_loss_sacp(
                self,
                tensors,
                v,
                obs=processed_cat_obs,
                policy=self.exploration_policy,
                log_alpha=self.exploration_log_alpha,
                qf1=self.exploration_qf1,
                qf2=self.exploration_qf2,
                description_prefix="exploration_",
            )
        else:
            processed_cat_obs = self._get_concat_obs(
                self.option_policy.process_observations(obs), options
            )

            sac_utils.update_loss_sacp(
                self,
                tensors,
                v,
                obs=processed_cat_obs,
                policy=self.option_policy,
                log_alpha=self.log_alpha,
                qf1=self.qf1,
                qf2=self.qf2,
                description_prefix="",
            )

    def _update_loss_alpha(self, tensors, v, exploration=False):
        if exploration:
            sac_utils.update_loss_alpha(
                self,
                tensors,
                v,
                log_alpha=self.exploration_log_alpha,
                description_prefix="exploration_",
            )
        else:
            sac_utils.update_loss_alpha(
                self,
                tensors,
                v,
                log_alpha=self.log_alpha,
                description_prefix="",
            )

    def _evaluate_policy(self, runner):
        eval_option_metrics = {}

        random_trajectories = self._evaluate_policy_inner(
            runner, deterministic_policy=True, description_prefix=""
        )
        eval_option_metrics.update(
            runner._env.calc_eval_metrics(
                random_trajectories, is_option_trajectories=True
            )
        )

        eval_gc_metrics = eval_utils._get_goal_conditioned_metrics(self, runner)
        eval_option_metrics.update(eval_gc_metrics)

        train_coverage_metrics = eval_utils._get_train_coverage_metrics(self, runner)
        eval_option_metrics.update(train_coverage_metrics)

        self._evaluate_policy_inner(
            runner, deterministic_policy=False, description_prefix="train_"
        )

        with global_context.GlobalContext({"phase": "eval", "policy": "option"}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(
                    self._env_spec, random_trajectories
                ),
                discount=self.discount,
                additional_records=eval_option_metrics,
            )
        self._log_eval_metrics(runner)

    def _evaluate_policy_inner(
        self, runner, deterministic_policy=True, description_prefix=""
    ):
        """Get eval trajectories with the goals sampled from replay buffer"""

        if self.goal_reaching:
            random_options, random_option_colors, goal_coordinates = (
                eval_utils.get_goal_options(self, runner)
            )
            # TODO: give option_colors according to the coordinates for better visualizations
        else:
            random_options, random_option_colors = eval_utils.get_skill_options(self)
            goal_coordinates = None

        if deterministic_policy:
            trajectories_kwargs = {
                "sampler_key": "option_policy",
                "extras": self._generate_option_extras(random_options),
            }
        else:
            trajectories_kwargs = self._get_train_trajectories_kwargs(
                runner, options=random_options
            )

        random_trajectories = self._get_trajectories(
            runner,
            **trajectories_kwargs,
            worker_update=dict(
                _render=False,
                _deterministic_policy=deterministic_policy,
            ),
            env_update=dict(_action_noise_std=None),
        )

        with FigManager(runner, f"{description_prefix}TrajPlot_RandomZ") as fm:
            eval_utils.draw_trajectories(
                self,
                runner,
                random_trajectories,
                random_option_colors,
                fm.ax,
                goal_coordinates=goal_coordinates,
            )

        if self.goal_reaching and len(runner._env.observation_space.shape) == 3:
            with FigManager(runner, f"{description_prefix}GoalImages") as fm:
                # pixel-space
                num_goals = len(random_trajectories)
                grid_width = 6
                grid_height = num_goals // 6 + (num_goals % 6 > 0)

                # Create a separate figure for rendering the goals
                fig, axs = plt.subplots(
                    grid_height, grid_width, figsize=(15, grid_height * 2.5)
                )
                axs = axs.flatten()
                random_options = random_options.reshape(
                    -1, *runner._env.observation_space.shape
                )
                goal_images = random_options[:, :, :, :3].astype(np.uint8)
                i = 0
                for i, goal_image in enumerate(goal_images):
                    axs[i].imshow(goal_image)
                    axs[i].axis("off")

                # Turn off the remaining empty subplots
                for j in range(i + 1, len(axs)):
                    axs[j].axis("off")

                plt.tight_layout()

                # Render the figure to a canvas
                canvas = FigureCanvas(fig)
                canvas.draw()
                width, height = fig.get_size_inches() * fig.get_dpi()
                image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
                    int(height), int(width), 3
                )

                plt.close(fig)  # Close the figure to free memory

                fm.ax.imshow(image)
                fm.ax.axis("off")

        # Record videos
        if self.eval_record_video:
            video_trajectories, extras = self._get_video_trajectories(
                runner, deterministic_policy=deterministic_policy, return_extra=True
            )
            if self.goal_reaching and len(runner._env.observation_space.shape) == 3:
                assert extras[0]["option"].ndim == 1
                options = np.stack([extra["option"] for extra in extras], axis=0)
                goal_images = (
                    options.reshape(-1, *runner._env.observation_space.shape)[
                        :, :, :, :3
                    ]
                    .transpose(0, 3, 1, 2)
                    .astype(np.uint8)
                )
            else:
                extras = None
                goal_images = None

            record_video(
                runner,
                f"{description_prefix}Video_RandomZ",
                video_trajectories,
                skip_frames=self.video_skip_frames,
                goal_images=goal_images,
            )

        return random_trajectories

    def _get_video_trajectories(
        self, runner, deterministic_policy=True, return_extra=False, options=None
    ):
        assert self.eval_record_video
        if options is None:
            if self.discrete:
                video_options = np.eye(self.dim_option)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            else:
                if self.dim_option == 2:
                    radius = 1.0 if self.unit_length else 1.5
                    video_options = []
                    for angle in [3, 2, 1, 4]:
                        video_options.append(
                            [
                                radius * np.cos(angle * np.pi / 4),
                                radius * np.sin(angle * np.pi / 4),
                            ]
                        )
                    video_options.append([0, 0])
                    for angle in [0, 5, 6, 7]:
                        video_options.append(
                            [
                                radius * np.cos(angle * np.pi / 4),
                                radius * np.sin(angle * np.pi / 4),
                            ]
                        )
                    video_options = np.array(video_options)
                else:
                    video_options = np.random.randn(9, self.dim_option)
                    if self.unit_length:
                        video_options = video_options / np.linalg.norm(
                            video_options, axis=1, keepdims=True
                        )
                video_options = video_options.repeat(self.num_video_repeats, axis=0)

            if self.goal_reaching:
                video_options = self.get_random_goals(self.num_video_repeats)["goals"]
        else:
            video_options = options
        if not deterministic_policy:
            extras = self._get_train_trajectories_kwargs(runner, options=video_options)[
                "extras"
            ]
        else:
            extras = self._generate_option_extras(video_options)
        video_trajectories = self._get_trajectories(
            runner,
            sampler_key="local_option_policy",
            extras=extras,
            worker_update=dict(
                _render=True,
                _deterministic_policy=deterministic_policy,
            ),
        )

        if return_extra:
            return video_trajectories, extras
        return video_trajectories
