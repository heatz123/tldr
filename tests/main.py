#!/usr/bin/env python3
import dowel_wrapper

assert dowel_wrapper is not None
import dowel

import wandb

import argparse
import datetime
import functools
import os
import sys
import platform
import torch.multiprocessing as mp
import tempfile

if "mac" in platform.platform():
    pass
else:
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["EGL_DEVICE_ID"] = "0"

import better_exceptions
import numpy as np

better_exceptions.hook()

import torch

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.torch.distributions import TanhNormal

from garagei.replay_buffer.path_buffer_ex import PathBufferEx
from garagei.experiment.option_local_runner import OptionLocalRunner
from garagei.envs.consistent_normalized_env import consistent_normalize
from garagei.sampler.option_multiprocessing_sampler import OptionMultiprocessingSampler
from garagei.torch.modules.parameter_module import ParameterModule
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garagei.torch.utils import xavier_normal_ex

from iod.model_utils import (
    make_module_conf,
    make_option_policy,
    make_traj_encoder,
    make_qf,
)

from iod.tldr import TLDR
from iod.utils import get_normalizer_preset

EXP_DIR = "exp/"
if os.environ.get("START_METHOD") is not None:
    START_METHOD = os.environ["START_METHOD"]
else:
    START_METHOD = "spawn"


def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--run_group", type=str, default="Debug")
    parser.add_argument(
        "--normalizer_type", type=str, default="off", choices=["off", "preset"]
    )
    parser.add_argument("--encoder", type=int, default=0)

    parser.add_argument(
        "--env",
        type=str,
        default="antmaze-large-play",
        choices=[
            "ant",
            "half_cheetah",
            "antmaze-large-play",
            "antmaze-ultra-play",
            "dmc_quadruped_state_escape",
            "dmc_humanoid_state",
            "dmc_quadruped",
            "kitchen",
        ],
    )
    parser.add_argument("--frame_stack", type=int, default=None)
    parser.add_argument("--max_path_length", type=int, default=200)

    parser.add_argument("--use_gpu", type=int, default=1, choices=[0, 1])
    parser.add_argument("--sample_cpu", type=int, default=1, choices=[0, 1])

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_parallel", type=int, default=4)
    parser.add_argument("--n_thread", type=int, default=1)

    parser.add_argument("--n_epochs", type=int, default=1000000)
    parser.add_argument("--traj_batch_size", type=int, default=8)
    parser.add_argument("--trans_minibatch_size", type=int, default=256)
    parser.add_argument("--trans_optimization_epochs", type=int, default=50)

    parser.add_argument("--n_epochs_per_eval", type=int, default=125)
    parser.add_argument("--n_epochs_per_log", type=int, default=25)
    parser.add_argument("--n_epochs_per_save", type=int, default=1000)
    parser.add_argument("--n_epochs_per_pt_save", type=int, default=1000)
    parser.add_argument("--n_epochs_per_pkl_update", type=int, default=None)
    parser.add_argument("--num_random_trajectories", type=int, default=48)
    parser.add_argument("--num_video_repeats", type=int, default=2)
    parser.add_argument("--eval_record_video", type=int, default=1)
    parser.add_argument("--eval_plot_axis", type=float, default=None, nargs="*")
    parser.add_argument("--video_skip_frames", type=int, default=1)

    parser.add_argument("--dim_option", type=int, default=2)

    parser.add_argument("--common_lr", type=float, default=1e-4)
    parser.add_argument("--lr_op", type=float, default=None)
    parser.add_argument("--lr_te", type=float, default=None)

    parser.add_argument("--alpha", type=float, default=0.01)

    parser.add_argument("--sac_tau", type=float, default=5e-3)
    parser.add_argument("--sac_lr_q", type=float, default=None)
    parser.add_argument("--sac_lr_a", type=float, default=None)
    parser.add_argument("--exploration_sac_lr_q", type=float, default=None)
    parser.add_argument("--exploration_sac_lr_a", type=float, default=None)
    parser.add_argument("--exploration_lr_op", type=float, default=None)
    parser.add_argument("--sac_discount", type=float, default=0.99)
    parser.add_argument("--exploration_sac_discount", type=float, default=0.99)
    parser.add_argument("--sac_scale_reward", type=float, default=1.0)
    parser.add_argument("--sac_target_coef", type=float, default=1.0)
    parser.add_argument("--sac_min_buffer_size", type=int, default=10000)
    parser.add_argument("--sac_max_buffer_size", type=int, default=300000)

    parser.add_argument("--spectral_normalization", type=int, default=0, choices=[0, 1])
    parser.add_argument("--model_master_dim", type=int, default=1024)
    parser.add_argument("--model_master_num_layers", type=int, default=2)
    parser.add_argument(
        "--model_master_nonlinearity", type=str, default=None, choices=["relu", "tanh"]
    )
    parser.add_argument("--traj_encoder_dims", type=int, default=None)
    parser.add_argument("--traj_encoder_num_layers", type=int, default=None)
    parser.add_argument("--traj_encoder_layer_normalization", type=int, default=None)
    parser.add_argument("--qf_dims", type=int, default=None)
    parser.add_argument("--qf_num_layers", type=int, default=None)
    parser.add_argument("--policy_dims", type=int, default=None)
    parser.add_argument("--policy_num_layers", type=int, default=None)
    parser.add_argument("--exp_qf_dims", type=int, default=None)
    parser.add_argument("--exp_qf_num_layers", type=int, default=None)
    parser.add_argument("--exp_policy_dims", type=int, default=None)
    parser.add_argument("--exp_policy_num_layers", type=int, default=None)
    parser.add_argument("--q_layer_normalization", type=int, default=None)
    parser.add_argument("--exp_q_layer_normalization", type=int, default=None)

    parser.add_argument("--algo", type=str, default="tldr", choices=["tldr", "metra"])
    parser.add_argument("--goal_reaching", type=int, default=1, choices=[0, 1])
    parser.add_argument("--num_alt_samples", type=int, default=100)
    parser.add_argument("--split_group", type=int, default=65536)

    parser.add_argument("--discrete", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--inner", type=int, default=1, choices=[0, 1]
    )  # inner product reward for METRA
    parser.add_argument(
        "--unit_length", type=int, default=1, choices=[0, 1]
    )  # Only for continuous skills

    parser.add_argument("--dual_reg", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dual_lam", type=float, default=30)
    parser.add_argument("--dual_slack", type=float, default=1e-3)
    parser.add_argument("--dual_lr", type=float, default=None)

    parser.add_argument("--knn_k", type=int, default=12)

    parser.add_argument("--description", type=str, default="")
    return parser


args = get_argparser().parse_args()
g_start_time = int(datetime.datetime.now().timestamp())


def get_exp_name():
    exp_name = ""
    exp_name += f"sd{args.seed:03d}_"
    if "SLURM_JOB_ID" in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if "SLURM_PROCID" in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name_prefix = exp_name
    if "SLURM_RESTART_COUNT" in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f"{g_start_time}"

    exp_name += "_" + args.env
    if args.description:
        exp_name += "_" + args.description

    return exp_name, exp_name_prefix


def get_log_dir():
    exp_name, exp_name_prefix = get_exp_name()
    assert len(exp_name) <= os.pathconf("/", "PC_NAME_MAX")
    # Resolve symlinks to prevent runs from crashing in case of home nfs crashing.
    log_dir = os.path.realpath(os.path.join(EXP_DIR, args.run_group, exp_name))
    assert not os.path.exists(log_dir), f"The following path already exists: {log_dir}"

    return log_dir


def make_env(args, max_path_length):
    if args.env == "antmaze-large-play":
        import gym
        import d4rl
        from envs.d4rl.pixel_wrappers import MazeRenderWrapper

        env = gym.make("antmaze-custom-large-play-v2")
        env = MazeRenderWrapper(env, pixel=False)
    elif args.env == "antmaze-ultra-play":
        import gym
        import d4rl
        from envs.d4rl.pixel_wrappers import MazeRenderWrapper

        env = gym.make("antmaze-custom-ultra-play-v0")
        env = MazeRenderWrapper(env, pixel=False)
    elif args.env == "half_cheetah":
        from envs.mujoco.half_cheetah_env import HalfCheetahEnv

        env = HalfCheetahEnv(render_hw=100)
    elif args.env == "ant":
        from envs.mujoco.ant_env import AntEnv

        env = AntEnv(render_hw=100)
    elif args.env == "dmc_humanoid_state":
        from envs.custom_dmc_tasks import dmc

        env = dmc.make(
            "humanoid_run_pure_state",
            obs_type="states",
            frame_stack=1,
            action_repeat=2,
            seed=args.seed,
        )
    elif args.env == "dmc_quadruped_state_escape":
        from envs.custom_dmc_tasks import dmc

        env = dmc.make(
            "quadruped_escape",
            obs_type="states",
            frame_stack=1,
            action_repeat=2,
            seed=args.seed,
            task_kwargs={
                "random": args.seed,
            },
        )
    elif args.env.startswith("dmc"):  # dmc pixel-based environments
        from envs.custom_dmc_tasks import dmc
        from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper

        assert args.encoder
        if args.env == "dmc_quadruped":
            env = dmc.make(
                "quadruped_run_forward_color",
                obs_type="states",
                frame_stack=1,
                action_repeat=2,
                seed=args.seed,
            )
            env = RenderWrapper(env)
        else:
            raise NotImplementedError
    elif args.env == "kitchen":
        sys.path.append("lexa")
        from envs.lexa.mykitchen import MyKitchenEnv

        assert args.encoder  # Only support pixel-based environments
        env = MyKitchenEnv(log_per_goal=True)

    else:
        raise NotImplementedError

    if args.frame_stack is not None:
        from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper

        env = FrameStackWrapper(env, args.frame_stack)

    normalizer_type = args.normalizer_type
    normalizer_kwargs = {}

    if normalizer_type == "off":
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)
    elif normalizer_type == "preset":
        normalizer_name = args.env
        normalizer_mean, normalizer_std = get_normalizer_preset(
            f"{normalizer_name}_preset"
        )
        env = consistent_normalize(
            env,
            normalize_obs=True,
            mean=normalizer_mean,
            std=normalizer_std,
            **normalizer_kwargs,
        )

    return env


def get_env_fn(args):
    max_path_length = args.max_path_length
    contextualized_make_env = functools.partial(
        make_env, args=args, max_path_length=max_path_length
    )
    return contextualized_make_env


def get_runner(args, ctxt):
    if "WANDB_API_KEY" in os.environ:
        wandb_output_dir = tempfile.mkdtemp()

        try:
            wandb.init(
                project="",
                entity="",
                group=args.run_group,
                name=get_exp_name()[0],
                config=vars(args),
                dir=wandb_output_dir,
            )
        except:
            raise Exception("Please set project and entity in wandb.init()")

    dowel.logger.log("ARGS: " + str(args))
    if args.n_thread is not None:
        torch.set_num_threads(args.n_thread)

    set_seed(args.seed)
    runner = OptionLocalRunner(ctxt)
    device = torch.device("cuda" if args.use_gpu else "cpu")

    contextualized_make_env = get_env_fn(args)
    env = contextualized_make_env()

    module_conf = make_module_conf(args, env)

    option_policy = make_option_policy(args, **module_conf)

    traj_encoder = make_traj_encoder(args, **module_conf)

    dual_lam = ParameterModule(torch.Tensor([np.log(args.dual_lam)]))

    def _finalize_lr(lr):
        if lr is None:
            lr = args.common_lr
        else:
            assert bool(lr), "To specify a lr of 0, use a negative value"
        if lr < 0.0:
            dowel.logger.log(f"Setting lr to ZERO given {lr}")
            lr = 0.0
        return lr

    optimizers = {
        "option_policy": torch.optim.Adam(
            [
                {"params": option_policy.parameters(), "lr": _finalize_lr(args.lr_op)},
            ]
        ),
        "traj_encoder": torch.optim.Adam(
            [
                {"params": traj_encoder.parameters(), "lr": _finalize_lr(args.lr_te)},
            ]
        ),
        "dual_lam": torch.optim.Adam(
            [
                {"params": dual_lam.parameters(), "lr": _finalize_lr(args.dual_lr)},
            ]
        ),
    }

    qf1 = make_qf(args, **module_conf)
    qf2 = make_qf(args, **module_conf)

    log_alpha = ParameterModule(torch.Tensor([np.log(args.alpha)]))
    optimizers.update(
        {
            "qf": torch.optim.Adam(
                [
                    {
                        "params": list(qf1.parameters()) + list(qf2.parameters()),
                        "lr": _finalize_lr(args.sac_lr_q),
                    },
                ]
            ),
            "log_alpha": torch.optim.Adam(
                [
                    {
                        "params": log_alpha.parameters(),
                        "lr": _finalize_lr(args.sac_lr_a),
                    },
                ]
            ),
        }
    )

    if args.goal_reaching:
        exploration_policy = make_option_policy(
            args,
            input_dim=module_conf["module_obs_dim"],
            policy_kwargs=dict(
                name="exploration_policy",
            ),
            encode_goal=False,
            **module_conf,
        )

        optimizers.update(
            {
                "exploration_policy": torch.optim.Adam(
                    [
                        {
                            "params": exploration_policy.parameters(),
                            "lr": _finalize_lr(args.exploration_lr_op),
                        },
                    ]
                ),
            }
        )

        exploration_qf1 = make_qf(
            args,
            input_dim=module_conf["module_obs_dim"],
            encode_goal=False,
            **module_conf,
        )
        exploration_qf2 = make_qf(
            args,
            input_dim=module_conf["module_obs_dim"],
            encode_goal=False,
            **module_conf,
        )
        exploration_log_alpha = ParameterModule(torch.Tensor([np.log(args.alpha)]))

        optimizers.update(
            {
                "exploration_qf": torch.optim.Adam(
                    [
                        {
                            "params": list(exploration_qf1.parameters())
                            + list(exploration_qf2.parameters()),
                            "lr": _finalize_lr(args.exploration_sac_lr_q),
                        },
                    ]
                ),
                "exploration_log_alpha": torch.optim.Adam(
                    [
                        {
                            "params": exploration_log_alpha.parameters(),
                            "lr": _finalize_lr(args.exploration_sac_lr_a),
                        },
                    ]
                ),
            }
        )

    replay_buffer = PathBufferEx(
        capacity_in_transitions=int(args.sac_max_buffer_size),
        pixel_shape=module_conf["pixel_shape"],
        use_goal=(args.algo in ["tldr"]),
    )

    optimizer = OptimizerGroupWrapper(
        optimizers=optimizers,
        max_optimization_epochs=None,
    )

    algo_kwargs = dict(
        env_name=args.env,
        algo=args.algo,
        env_spec=env.spec,
        option_policy=option_policy,
        traj_encoder=traj_encoder,
        dual_lam=dual_lam,
        optimizer=optimizer,
        alpha=args.alpha,
        max_path_length=args.max_path_length,
        n_epochs_per_eval=args.n_epochs_per_eval,
        n_epochs_per_log=args.n_epochs_per_log,
        n_epochs_per_tb=args.n_epochs_per_log,
        n_epochs_per_save=args.n_epochs_per_save,
        n_epochs_per_pt_save=args.n_epochs_per_pt_save,
        n_epochs_per_pkl_update=(
            args.n_epochs_per_eval
            if args.n_epochs_per_pkl_update is None
            else args.n_epochs_per_pkl_update
        ),
        dim_option=args.dim_option,
        num_random_trajectories=args.num_random_trajectories,
        num_video_repeats=args.num_video_repeats,
        eval_record_video=args.eval_record_video,
        video_skip_frames=args.video_skip_frames,
        eval_plot_axis=args.eval_plot_axis,
        name="TLDR",
        device=device,
        sample_cpu=args.sample_cpu,
        num_train_per_epoch=1,
        trans_minibatch_size=args.trans_minibatch_size,
        trans_optimization_epochs=args.trans_optimization_epochs,
        discount=args.sac_discount,
        exploration_sac_discount=args.exploration_sac_discount,
        discrete=args.discrete,
        unit_length=args.unit_length,
        exploration_policy=exploration_policy if args.goal_reaching else None,
        exploration_qf1=exploration_qf1 if args.goal_reaching else None,
        exploration_qf2=exploration_qf2 if args.goal_reaching else None,
        exploration_log_alpha=(exploration_log_alpha if args.goal_reaching else None),
        goal_reaching=args.goal_reaching,
        frame_stack=args.frame_stack,
        knn_k=args.knn_k,
    )

    skill_common_args = dict(
        qf1=qf1,
        qf2=qf2,
        log_alpha=log_alpha,
        tau=args.sac_tau,
        scale_reward=args.sac_scale_reward,
        target_coef=args.sac_target_coef,
        replay_buffer=replay_buffer,
        min_buffer_size=args.sac_min_buffer_size,
        num_alt_samples=args.num_alt_samples,
        split_group=args.split_group,
        dual_reg=args.dual_reg,
        dual_slack=args.dual_slack,
        pixel_shape=module_conf["pixel_shape"],
    )

    algo = TLDR(
        **algo_kwargs,
        **skill_common_args,
    )

    if args.sample_cpu:
        algo.option_policy.cpu()
        algo.traj_encoder.cpu()
    else:
        algo.option_policy.to(device)
        algo.traj_encoder.to(device)

    runner.setup(
        algo=algo,
        env=env,
        make_env=contextualized_make_env,
        sampler_cls=OptionMultiprocessingSampler,
        sampler_args=dict(n_thread=args.n_thread),
        n_workers=args.n_parallel,
        worker_args=dict(),
    )
    algo.option_policy.to(device)
    algo.traj_encoder.to(device)

    return runner


@wrap_experiment(log_dir=get_log_dir(), name=get_exp_name()[0])
def run(ctxt=None):
    runner = get_runner(args, ctxt=ctxt)
    runner.train(n_epochs=args.n_epochs, batch_size=args.traj_batch_size)


if __name__ == "__main__":
    mp.set_start_method(START_METHOD)
    run()
