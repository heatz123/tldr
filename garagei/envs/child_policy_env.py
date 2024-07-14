from collections import defaultdict

import akro
import gym.spaces.utils
import numpy as np
import torch

from garage.envs import EnvSpec
from garage.torch.distributions import TanhNormal

from iod.utils import get_torch_concat_obs


class ChildPolicyEnv(gym.Wrapper):
    def __init__(
            self,
            env,
            cp_dict,
            cp_action_range,
            cp_unit_length,
            cp_multi_step,
            cp_num_truncate_obs,
            cp_omit_obs_idxs=None,
    ):
        super().__init__(env)

        self.child_policy = cp_dict['policy']
        self.child_policy.eval()

        self.cp_dim_action = cp_dict['dim_option']
        self.cp_action_range = cp_action_range
        self.cp_unit_length = cp_unit_length
        self.cp_multi_step = cp_multi_step
        self.cp_num_truncate_obs = cp_num_truncate_obs
        self.cp_omit_obs_idxs = cp_omit_obs_idxs
        self.cp_discrete = cp_dict['discrete']

        self.observation_space = self.env.observation_space
        if 'discrete' in cp_dict and cp_dict['discrete']:
            self.action_space = akro.Discrete(n=cp_dict['dim_option'])
        else:
            self.action_space = akro.Box(low=-1., high=1., shape=(self.cp_dim_action,))

        self.last_obs = None
        self.first_obs = None

    @property
    def spec(self):
        return EnvSpec(action_space=self.action_space,
                       observation_space=self.observation_space)

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)

        self.last_obs = ret
        self.first_obs = ret

        return ret

    def step(self, cp_action, **kwargs):
        cp_action_norm = np.linalg.norm(cp_action)
        cp_action = cp_action.copy()
        if not self.cp_discrete:
            if self.cp_unit_length:
                cp_action = cp_action / cp_action_norm
            else:
                cp_action = cp_action * self.cp_action_range

        sum_rewards = 0.
        acc_infos = defaultdict(list)

        done_final = False
        for i in range(self.cp_multi_step):
            cp_obs = self.last_obs
            cp_obs = torch.as_tensor(cp_obs)
            if self.cp_num_truncate_obs > 0:
                cp_obs = cp_obs[:-self.cp_num_truncate_obs]
            if self.cp_omit_obs_idxs is not None:
                cp_obs[self.cp_omit_obs_idxs] = 0

            cp_action = torch.as_tensor(cp_action)

            cp_input = get_torch_concat_obs(cp_obs, cp_action, dim=0).float()

            # XXX: Hacky
            # First try to use mode
            if hasattr(self.child_policy._module, 'forward_mode'):
                # Beta
                action = self.child_policy.get_mode_actions(cp_input.unsqueeze(dim=0))[0]
            else:
                # Tanhgaussian
                action_dist = self.child_policy(cp_input.unsqueeze(dim=0))[0]
                action = action_dist.mean.detach().numpy()
            action = action[0]
            # Assume that the range of the variable 'action' (= the output from self.child_policy) is [-1, 1]
            # This assumption is probably true as of now (since we only use (scaled) Beta or TanhGaussian policy)
            lb, ub = self.env.action_space.low, self.env.action_space.high
            action = lb + (action + 1) * (0.5 * (ub - lb))
            action = np.clip(action, lb, ub)

            next_obs, reward, done, info = self.env.step(action, **kwargs)

            self.last_obs = next_obs

            sum_rewards += reward
            for k, v in info.items():
                acc_infos[k].append(v)

            if info.get('done_internal', False):
                done_final = True

            if done:
                done_final = True
                break

        infos = {}
        for k, v in acc_infos.items():
            # if k in ['coordinates', 'next_coordinates', 'ori', 'next_ori']:
            #     infos[k] = np.concatenate(v).reshape(-1, v[0].shape[-1])
            # elif k in ['ori_obs', 'next_ori_obs']:
            #     infos[k] = v[-1]
            # else:
            #     if isinstance(v[0], np.ndarray):
            #         infos[k] = np.array(v)
            #     elif isinstance(v[0], tuple):
            #         infos[k] = np.array([list(l) for l in v])
            #     else:
            #         infos[k] = sum(v)
            infos[k] = v[-1]
        infos['cp_action_norm'] = cp_action_norm

        return next_obs, sum_rewards, done_final, infos
