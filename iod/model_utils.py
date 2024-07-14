import numpy as np
import torch
import functools

from garagei.torch.modules.with_encoder import WithEncoder, Encoder
from garagei.torch.modules.gaussian_mlp_module_ex import (
    GaussianMLPTwoHeadedModuleEx,
    GaussianMLPIndependentStdModuleEx,
    GaussianMLPModuleEx,
)
from garage.torch.distributions import TanhNormal
from garagei.torch.utils import xavier_normal_ex
from garagei.torch.policies.policy_ex import PolicyEx
from garagei.torch.q_functions.continuous_mlp_q_function_ex import (
    ContinuousMLPQFunctionEx,
)


def get_gaussian_module_construction(
    args,
    *,
    hidden_sizes,
    const_std=False,
    hidden_nonlinearity=torch.relu,
    w_init=torch.nn.init.xavier_uniform_,
    init_std=1.0,
    min_std=1e-6,
    max_std=None,
    **kwargs,
):
    module_kwargs = dict()
    if const_std:
        module_cls = GaussianMLPModuleEx
        module_kwargs.update(
            dict(
                learn_std=False,
                init_std=init_std,
            )
        )
    else:
        module_cls = GaussianMLPIndependentStdModuleEx
        module_kwargs.update(
            dict(
                std_hidden_sizes=hidden_sizes,
                std_hidden_nonlinearity=hidden_nonlinearity,
                std_hidden_w_init=w_init,
                std_output_w_init=w_init,
                init_std=init_std,
                min_std=min_std,
                max_std=max_std,
            )
        )

    module_kwargs.update(
        dict(
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=w_init,
            std_parameterization="exp",
            bias=True,
            spectral_normalization=args.spectral_normalization,
            **kwargs,
        )
    )
    return module_cls, module_kwargs


def get_pixel_shape(env):
    # should be called only when args.encoder is True
    if hasattr(env, "ob_info"):
        if env.ob_info["type"] in ["hybrid", "pixel"]:
            pixel_shape = env.ob_info["pixel_shape"]
            pixel_dim = np.prod(pixel_shape)
            if env.ob_info["type"] in ["hybrid"]:
                state_shape = env.ob_info["state_shape"]
                assert len(state_shape) == 1
                state_dim = state_shape[0]
                print("state_dim", state_dim)
        else:
            pixel_shape = (64, 64, 3)
    else:
        pixel_shape = None
    return pixel_shape


def make_encoder(args, **kwargs):
    # should be called only when args.encoder is True
    return Encoder(
        use_atari_torso=True if args.goal_reaching else False,
        **kwargs,
    )


def with_encoder(module, args, encoder=None):
    if encoder is None:
        encoder = make_encoder(
            norm="layer" if args.encoder_layer_normalization else "none",
        )

    return WithEncoder(encoder=encoder, module=module)


def make_option_policy(
    args, input_dim=None, policy_kwargs=None, encode_goal=None, **kwargs
):
    master_dims = kwargs["master_dims"]
    nonlinearity = kwargs["nonlinearity"]
    action_dim = kwargs["action_dim"]
    policy_q_input_dim = kwargs["policy_q_input_dim"]
    pixel_shape = kwargs["pixel_shape"]

    if input_dim is None:
        input_dim = policy_q_input_dim

    module_kwargs = dict(
        hidden_sizes=(
            [args.policy_dims] * args.policy_num_layers
            if args.policy_dims is not None
            else master_dims
        ),
        layer_normalization=False,
    )
    if nonlinearity is not None:
        module_kwargs.update(hidden_nonlinearity=nonlinearity)

    module_cls = GaussianMLPTwoHeadedModuleEx
    module_kwargs.update(
        dict(
            max_std=np.exp(2.0),
            normal_distribution_cls=TanhNormal,  # using TanhNormal guarantees -1~1 action range
            output_w_init=functools.partial(xavier_normal_ex, gain=1.0),
            init_std=1.0,
        )
    )

    policy_module = module_cls(
        input_dim=input_dim, output_dim=action_dim, **module_kwargs
    )

    if args.encoder:
        if encode_goal is None:
            if args.goal_reaching:
                encode_goal = True
            else:
                encode_goal = False

        policy_encoder = make_encoder(
            args,
            pixel_shape=pixel_shape,
            encode_goal=encode_goal,
            use_separate_encoder=False,
        )
        policy_module = with_encoder(policy_module, args, encoder=policy_encoder)

    if policy_kwargs is None:
        policy_kwargs = dict(
            name="option_policy",
            option_info={
                "dim_option": args.dim_option,
            },
        )

    policy_kwargs["module"] = policy_module
    option_policy = PolicyEx(**policy_kwargs)

    return option_policy


def make_traj_encoder(args, **kwargs):
    master_dims = kwargs["master_dims"]
    module_obs_dim = kwargs["module_obs_dim"]
    nonlinearity = kwargs["nonlinearity"]
    pixel_shape = kwargs["pixel_shape"]

    traj_master_dims = (
        None
        if not args.traj_encoder_dims
        else [args.traj_encoder_dims] * args.traj_encoder_num_layers
    )

    traj_encoder_obs_dim = module_obs_dim
    traj_nonlinearity = nonlinearity or torch.relu
    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        hidden_sizes=master_dims if not traj_master_dims else traj_master_dims,
        hidden_nonlinearity=traj_nonlinearity,
        w_init=torch.nn.init.xavier_uniform_,
        input_dim=traj_encoder_obs_dim,
        output_dim=args.dim_option,
    )
    # main traj
    traj_encoder = module_cls(
        layer_normalization=True if args.traj_encoder_layer_normalization else False,
        **module_kwargs,
    )
    if args.encoder:
        te_encoder = make_encoder(args, pixel_shape=pixel_shape)
        traj_encoder = with_encoder(traj_encoder, args, encoder=te_encoder)

    return traj_encoder


def make_qf(args, input_dim=None, encode_goal=None, **kwargs):
    master_dims = kwargs["master_dims"]
    nonlinearity = kwargs["nonlinearity"]
    action_dim = kwargs["action_dim"]
    policy_q_input_dim = kwargs["policy_q_input_dim"]
    pixel_shape = kwargs["pixel_shape"]

    if input_dim is None:
        input_dim = policy_q_input_dim

    qf = ContinuousMLPQFunctionEx(
        obs_dim=input_dim,
        action_dim=action_dim,
        hidden_sizes=(
            [args.qf_dims] * args.qf_num_layers
            if args.qf_dims is not None
            else master_dims
        ),
        hidden_nonlinearity=nonlinearity or torch.relu,
        layer_normalization=True if args.q_layer_normalization else False,
    )
    if args.encoder:
        if encode_goal is None:
            if args.goal_reaching:
                encode_goal = True
            else:
                encode_goal = False

        qf_encoder = make_encoder(
            args, pixel_shape=pixel_shape, encode_goal=encode_goal
        )
        qf = with_encoder(qf, args, encoder=qf_encoder)
    return qf


def make_module_conf(args, env):
    # env is needed for obs_dim and action_dim
    master_dims = [args.model_master_dim] * args.model_master_num_layers

    if args.model_master_nonlinearity == "relu":
        nonlinearity = torch.relu
    elif args.model_master_nonlinearity == "tanh":
        nonlinearity = torch.tanh
    else:
        nonlinearity = None

    obs_dim = env.spec.observation_space.flat_dim
    action_dim = env.spec.action_space.flat_dim

    pixel_shape = get_pixel_shape(env) if args.encoder else None

    example_ob = env.reset()
    if args.encoder:
        example_encoder = make_encoder(args, pixel_shape=pixel_shape)
        module_obs_dim = example_encoder(
            torch.as_tensor(example_ob).float().unsqueeze(0)
        ).shape[-1]
    else:
        module_obs_dim = obs_dim

    policy_q_input_dim = (
        module_obs_dim + args.dim_option
        if not args.goal_reaching
        else module_obs_dim * 2
    )
    return {
        "master_dims": master_dims,
        "nonlinearity": nonlinearity,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "module_obs_dim": module_obs_dim,
        "pixel_shape": pixel_shape,
        "policy_q_input_dim": policy_q_input_dim,
    }
