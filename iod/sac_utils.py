import torch
from torch.nn import functional as F

import attrs
from typing import Any


@attrs.define(kw_only=True)
class ActorObsGoalCriticInfo:
    r"""
    Similar to CriticBatchInfo, but does not store the latents for the data batch.

    Instead, for the batch of observation and goal pairs which the actor is activated with,
    this stores the latents for them.
    """

    critic: Any
    zo: Any
    zg: Any


def _clip_actions(algo, actions):
    epsilon = 1e-6
    lower = torch.from_numpy(algo._env_spec.action_space.low).to(algo.device) + epsilon
    upper = torch.from_numpy(algo._env_spec.action_space.high).to(algo.device) - epsilon

    clip_up = (actions > upper).float()
    clip_down = (actions < lower).float()
    with torch.no_grad():
        clip = (upper - actions) * clip_up + (lower - actions) * clip_down

    return actions + clip


def update_loss_qf(
    algo,
    tensors,
    v,
    obs,
    actions,
    next_obs,
    dones,
    rewards,
    policy,
    log_alpha,
    qf1,
    qf2,
    target_qf1,
    target_qf2,
    description_prefix="",
    discount=0.99,
):
    with torch.no_grad():
        alpha = log_alpha.param.exp()

    q1_pred = qf1(obs, actions).flatten()
    q2_pred = qf2(obs, actions).flatten()

    next_action_dists, *_ = policy(next_obs)
    if hasattr(next_action_dists, "rsample_with_pre_tanh_value"):
        new_next_actions_pre_tanh, new_next_actions = (
            next_action_dists.rsample_with_pre_tanh_value()
        )
        new_next_action_log_probs = next_action_dists.log_prob(
            new_next_actions, pre_tanh_value=new_next_actions_pre_tanh
        )
    else:
        new_next_actions = next_action_dists.rsample()
        new_next_actions = _clip_actions(algo, new_next_actions)
        new_next_action_log_probs = next_action_dists.log_prob(new_next_actions)

    target_q_values = torch.min(
        target_qf1(next_obs, new_next_actions).flatten(),
        target_qf2(next_obs, new_next_actions).flatten(),
    )
    target_q_values = target_q_values - alpha * new_next_action_log_probs
    target_q_values = target_q_values * discount

    with torch.no_grad():
        q_target = rewards + target_q_values * (1.0 - dones)

    # critic loss weight: 0.5
    loss_qf1 = F.mse_loss(q1_pred, q_target) * 0.5
    loss_qf2 = F.mse_loss(q2_pred, q_target) * 0.5

    tensors.update(
        {
            f"{description_prefix}QTargetsMean": q_target.mean(),
            f"{description_prefix}QTdErrsMean": (
                (q_target - q1_pred).mean() + (q_target - q2_pred).mean()
            )
            / 2,
            f"{description_prefix}LossQf1": loss_qf1,
            f"{description_prefix}LossQf2": loss_qf2,
        }
    )


def update_loss_dqf(
    algo,
    tensors,
    v,
    obs,
    actions,
    next_obs,
    dones,
    rewards,
    qf1,
    target_qf1,
    description_prefix="",
):
    assert actions.ndim == 2, actions.shape
    actions = actions.argmax(dim=-1)
    with torch.no_grad():
        next_qa_values = target_qf1(next_obs)
        next_actions = qf1(next_obs).argmax(dim=-1)
        next_qa_values.ndim == 2
        assert next_actions.ndim == 1
        next_q_values = torch.gather(
            next_qa_values, dim=-1, index=next_actions.unsqueeze(-1)
        ).squeeze(-1)
        q_target = rewards + (1 - dones.float()) * algo.discount * next_q_values

    qa_values = qf1(obs)
    assert actions.shape == next_actions.shape, f"{actions.shape}, {next_actions.shape}"
    q_values = torch.gather(qa_values, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    assert (
        q_values.shape == q_target.shape and q_values.ndim == 1
    ), f"{q_values.shape}, {q_target.shape}"
    loss_qf1 = F.mse_loss(q_values, q_target) * 0.5

    tensors.update(
        {
            f"{description_prefix}QTargetsMean": q_target.mean(),
            f"{description_prefix}LossQf1": loss_qf1,
        }
    )


def gather_obs_goal_pairs(critic_batch_infos, data):
    r"""
    Returns (
        obs,
        goal,
        [ (latent_obs, latent_goal) for each critic ],
    )
    """

    obs = data.observations
    goal = torch.roll(data.next_observations, 1, dims=0)  # randomize :)
    add_goal_as_future_state = True
    if add_goal_as_future_state:
        # add future_observations
        goal = torch.stack([goal, data.future_observations], 0)
        obs = obs.expand_as(goal)

    actor_obs_goal_critic_infos = []

    for critic_batch_info in critic_batch_infos:
        zo = critic_batch_info.zx
        zg = torch.roll(critic_batch_info.zy, 1, dims=0)  # randomize in the same way:)

        if add_goal_as_future_state:
            # add future_observations
            zg = torch.stack(
                [
                    zg,
                    critic_batch_info.critic.encoder(data.future_observations),
                ],
                0,
            )
            zo = zo.expand_as(zg)

        actor_obs_goal_critic_infos.append(
            ActorObsGoalCriticInfo(
                critic=critic_batch_info.critic,
                zo=zo,
                zg=zg,
            )
        )

    return obs, goal, actor_obs_goal_critic_infos


def update_loss_quasimetric_sacp(
    algo,
    tensors,
    v,
    obs,
    policy,
    log_alpha,
    critic_batch_infos,
    data,
    description_prefix="",
):
    batch_size = obs.shape[0]
    with torch.no_grad():
        obs, goal, actor_obs_goal_critic_infos = gather_obs_goal_pairs(
            critic_batch_infos, data
        )  # obs: [2, 1024, ndim]
        obs = obs.reshape(2 * batch_size, -1)
        goal = goal.reshape(2 * batch_size, -1)

    processed_cat_obs = algo._get_concat_obs(
        algo.option_policy.process_observations(obs), goal
    )

    action_dists, *_ = policy(processed_cat_obs)
    actions = action_dists.rsample()
    actions = actions.reshape(2, batch_size, -1)
    if hasattr(action_dists, "rsample_with_pre_tanh_value"):
        new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
        new_action_log_probs = action_dists.log_prob(
            new_actions, pre_tanh_value=new_actions_pre_tanh
        )
    else:
        new_actions = action_dists.rsample()
        new_actions = _clip_actions(algo, new_actions)
        new_action_log_probs = action_dists.log_prob(new_actions)

    info = {}

    dists = []

    for idx, actor_obs_goal_critic_info in enumerate(actor_obs_goal_critic_infos):
        critic = actor_obs_goal_critic_info.critic
        with critic.requiring_grad(False):
            zp = critic.latent_dynamics(actor_obs_goal_critic_info.zo.detach(), actions)
            dist = critic.quasimetric_model(zp, actor_obs_goal_critic_info.zg.detach())
        info[f"dist_{idx:02d}"] = dist.mean()
        dists.append(dist)

    max_dist = info["dist_max"] = torch.stack(dists, -1).max(-1).values.mean()

    with torch.no_grad():
        alpha = log_alpha.param.exp()

    loss_sacp = (alpha * new_action_log_probs).mean() + max_dist

    tensors.update(
        {
            f"{description_prefix}SacpNewActionLogProbMean": new_action_log_probs.mean(),
            f"{description_prefix}LossSacp": loss_sacp,
        }
    )
    tensors.update(info)

    v.update(
        {
            "new_action_log_probs": new_action_log_probs,
        }
    )


def update_loss_sacp(
    algo,
    tensors,
    v,
    obs,
    policy,
    log_alpha,
    qf1,
    qf2,
    description_prefix="",
):
    with torch.no_grad():
        alpha = log_alpha.param.exp()

    action_dists, *_ = policy(obs)
    if hasattr(action_dists, "rsample_with_pre_tanh_value"):
        new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
        new_action_log_probs = action_dists.log_prob(
            new_actions, pre_tanh_value=new_actions_pre_tanh
        )
    else:
        new_actions = action_dists.rsample()
        new_actions = _clip_actions(algo, new_actions)
        new_action_log_probs = action_dists.log_prob(new_actions)

    min_q_values = torch.min(
        qf1(obs, new_actions).flatten(),
        qf2(obs, new_actions).flatten(),
    )

    loss_sacp = (alpha * new_action_log_probs - min_q_values).mean()

    tensors.update(
        {
            f"{description_prefix}SacpNewActionLogProbMean": new_action_log_probs.mean(),
            f"{description_prefix}LossSacp": loss_sacp,
        }
    )

    v.update(
        {
            "new_action_log_probs": new_action_log_probs,
        }
    )


def update_loss_alpha(
    algo,
    tensors,
    v,
    log_alpha,
    description_prefix="",
):
    loss_alpha = (
        -log_alpha.param * (v["new_action_log_probs"].detach() + algo._target_entropy)
    ).mean()

    tensors.update(
        {
            f"{description_prefix}Alpha": log_alpha.param.exp(),
            f"{description_prefix}LossAlpha": loss_alpha,
        }
    )


def update_targets(algo, qf1, qf2, target_qf1, target_qf2):
    """Update parameters in the target q-functions."""
    # target_qfs = [algo.target_qf1, algo.target_qf2]
    target_qfs = [target_qf1, target_qf2]
    # qfs = [algo.qf1, algo.qf2]
    qfs = [qf1, qf2]
    for target_qf, qf in zip(target_qfs, qfs):
        for t_param, param in zip(target_qf.parameters(), qf.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - algo.tau) + param.data * algo.tau)
