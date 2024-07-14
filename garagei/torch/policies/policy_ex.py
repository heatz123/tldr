import numpy as np
import torch

from garage.torch.distributions import TanhNormal
from garage.torch.policies.stochastic_policy import StochasticPolicy


class PolicyEx(StochasticPolicy):
    def __init__(
        self,
        name,
        *,
        module,
        clip_action=False,  # METAR: False
        omit_obs_idxs=None,
        option_info=None,
        force_use_mode_actions=False,
    ):
        super().__init__(env_spec=None, name=name)

        self._clip_action = clip_action
        self._omit_obs_idxs = omit_obs_idxs

        self._option_info = option_info
        self._force_use_mode_actions = force_use_mode_actions

        self._module = module

    def process_observations(self, observations):
        if self._omit_obs_idxs is not None:
            observations = observations.clone()
            observations[:, self._omit_obs_idxs] = 0
        return observations

    def forward(self, observations):
        observations = self.process_observations(observations)
        dist = self._module(observations)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()
        if hasattr(dist, "_normal"):
            info.update(
                dict(
                    normal_mean=dist._normal.mean,
                    normal_std=dist._normal.variance.sqrt(),
                )
            )

        return dist, info

    def forward_mode(self, observations):
        observations = self.process_observations(observations)
        samples = self._module.forward_mode(observations)
        return samples, dict()

    def forward_with_transform(self, observations, *, transform):
        observations = self.process_observations(observations)
        dist, dist_transformed = self._module.forward_with_transform(
            observations, transform=transform
        )
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            ret_mean_transformed = dist_transformed.mean.cpu()
            ret_log_std_transformed = (dist_transformed.variance.sqrt()).log().cpu()
            info = (
                dict(mean=ret_mean, log_std=ret_log_std),
                dict(mean=ret_mean_transformed, log_std=ret_log_std_transformed),
            )
        except NotImplementedError:
            info = (dict(), dict())
        return (dist, dist_transformed), info

    def forward_with_chunks(self, observations, *, merge):
        observations = [self.process_observations(o) for o in observations]
        dist = self._module.forward_with_chunks(observations, merge=merge)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()

        return dist, info

    def get_mode_actions(self, observations):
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = (
                    torch.as_tensor(observations)
                    .float()
                    .to(next(self.parameters()).device)
                )
            samples, info = self.forward_mode(observations)
            return samples.cpu().numpy(), {
                k: v.detach().cpu().numpy() for (k, v) in info.items()
            }

    def get_sample_actions(self, observations):
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = (
                    torch.as_tensor(observations)
                    .float()
                    .to(next(self.parameters()).device)
                )
            dist, info = self.forward(observations)
            if isinstance(dist, TanhNormal):
                pre_tanh_values, actions = dist.rsample_with_pre_tanh_value()
                log_probs = dist.log_prob(actions, pre_tanh_values)
                actions = actions.detach().cpu().numpy()
                infos = {k: v.detach().cpu().numpy() for (k, v) in info.items()}
                infos["pre_tanh_value"] = pre_tanh_values.detach().cpu().numpy()
                infos["log_prob"] = log_probs.detach().cpu().numpy()
            else:
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                actions = actions.detach().cpu().numpy()
                infos = {k: v.detach().cpu().numpy() for (k, v) in info.items()}
                infos["log_prob"] = log_probs.detach().cpu().numpy()
            return actions, infos

    def get_actions(self, observations):
        assert isinstance(observations, np.ndarray) or isinstance(
            observations, torch.Tensor
        )
        if self._force_use_mode_actions:
            actions, info = self.get_mode_actions(observations)
        else:
            actions, info = self.get_sample_actions(observations)
        if self._clip_action:
            epsilon = 1e-6
            actions = np.clip(
                actions,
                self.env_spec.action_space.low + epsilon,
                self.env_spec.action_space.high - epsilon,
            )
        return actions, info

    def get_action(self, observation, hidden_states=None):
        with torch.no_grad():
            if not isinstance(observation, torch.Tensor):
                observation = (
                    torch.as_tensor(observation)
                    .float()
                    .to(next(self.parameters()).device)
                )
            observation = observation.unsqueeze(0)
            action, agent_infos = self.get_actions(observation)
            return action[0], {k: v[0] for k, v in agent_infos.items()}


class RecurrentPolicyEx(StochasticPolicy):
    def __init__(
        self,
        name,
        *,
        module,
        clip_action=False,
        omit_obs_idxs=None,
        option_info=None,
        force_use_mode_actions=False,
    ):
        super().__init__(env_spec=None, name=name)

        self._clip_action = clip_action
        self._omit_obs_idxs = omit_obs_idxs

        self._option_info = option_info
        self._force_use_mode_actions = force_use_mode_actions

        self._module = module

    def process_observations(self, observations):
        if self._omit_obs_idxs is not None:
            observations = observations.clone()
            observations[:, self._omit_obs_idxs] = 0
        return observations

    def forward(self, observations, hidden_states=None):
        observations = self.process_observations(observations)
        dist, hidden_states = self._module(observations, hidden_states=hidden_states)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()
        if hasattr(dist, "_normal"):
            info.update(
                dict(
                    normal_mean=dist._normal.mean,
                    normal_std=dist._normal.variance.sqrt(),
                )
            )

        info.update(dict(hidden_states=hidden_states))

        return dist, info

    def forward_mode(self, observations, hidden_states=None):
        observations = self.process_observations(observations)
        samples, (hn, cn) = self._module.forward_mode(
            observations, hidden_states=hidden_states
        )
        return samples, dict(hidden_states=(hn, cn))

    def forward_with_transform(self, observations, *, transform):
        observations = self.process_observations(observations)
        dist, dist_transformed = self._module.forward_with_transform(
            observations, transform=transform
        )
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            ret_mean_transformed = dist_transformed.mean.cpu()
            ret_log_std_transformed = (dist_transformed.variance.sqrt()).log().cpu()
            info = (
                dict(mean=ret_mean, log_std=ret_log_std),
                dict(mean=ret_mean_transformed, log_std=ret_log_std_transformed),
            )
        except NotImplementedError:
            info = (dict(), dict())
        return (dist, dist_transformed), info

    def forward_with_chunks(self, observations, *, merge):
        observations = [self.process_observations(o) for o in observations]
        dist = self._module.forward_with_chunks(observations, merge=merge)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()

        return dist, info

    def get_mode_actions(self, observations, hidden_states=None):
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = (
                    torch.as_tensor(observations)
                    .float()
                    .to(next(self.parameters()).device)
                )
            samples, info = self.forward_mode(observations, hidden_states=hidden_states)
            info = {
                k: (
                    v.detach().cpu().numpy()
                    if k != "hidden_states"
                    else v  # deal with lstm hidden_states: no need to feed to env
                )
                for (k, v) in info.items()
            }
            return samples.cpu().numpy(), info

    def get_sample_actions(self, observations, hidden_states=None):
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = (
                    torch.as_tensor(observations)
                    .float()
                    .to(next(self.parameters()).device)
                )
            dist, info = self.forward(observations, hidden_states=hidden_states)
            if isinstance(dist, TanhNormal):
                pre_tanh_values, actions = dist.rsample_with_pre_tanh_value()
                log_probs = dist.log_prob(actions, pre_tanh_values)
                actions = actions.detach().cpu().numpy()
                infos = {
                    k: (
                        v.detach().cpu().numpy()
                        if k != "hidden_states"
                        else v  # deal with lstm hidden_states: no need to feed to env
                    )
                    for (k, v) in info.items()
                }
                infos["pre_tanh_value"] = pre_tanh_values.detach().cpu().numpy()
                infos["log_prob"] = log_probs.detach().cpu().numpy()
            else:
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                actions = actions.detach().cpu().numpy()
                infos = {
                    k: (
                        v.detach().cpu().numpy()
                        if k != "hidden_states"
                        else v  # deal with lstm hidden_states: no need to feed to env
                    )
                    for (k, v) in info.items()
                }
                infos["log_prob"] = log_probs.detach().cpu().numpy()
            return actions, infos

    def get_actions(self, observations, hidden_states=None):
        assert isinstance(observations, np.ndarray) or isinstance(
            observations, torch.Tensor
        )
        if self._force_use_mode_actions:
            actions, info = self.get_mode_actions(
                observations, hidden_states=hidden_states
            )
        else:
            actions, info = self.get_sample_actions(
                observations, hidden_states=hidden_states
            )
        if self._clip_action:
            epsilon = 1e-6
            actions = np.clip(
                actions,
                self.env_spec.action_space.low + epsilon,
                self.env_spec.action_space.high - epsilon,
            )
        return actions, info

    def get_action(self, observation, hidden_states=None):
        with torch.no_grad():  # maybe when this is called, we need to feed in both the context
            if not isinstance(observation, torch.Tensor):
                observation = (
                    torch.as_tensor(observation)
                    .float()
                    .to(next(self.parameters()).device)
                )
            observation = observation.unsqueeze(0)
            action, agent_infos = self.get_actions(
                observation, hidden_states=hidden_states
            )
            return action[0], {
                k: (v[0] if k != "hidden_states" else v) for k, v in agent_infos.items()
            }

    def reset(self, dones=None):
        self._module.reset(dones=dones)
