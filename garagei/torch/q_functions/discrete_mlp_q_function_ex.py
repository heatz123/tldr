"""This modules creates a discrete Q-function network."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage.torch.modules import MLPModule


# pytorch v1.6 issue, see https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class DiscreteMLPQFunctionEx(MLPModule):
    """Implements a discrete MLP Q-value network.

    It predicts the Q-value for all possible actions based on the
    input state.

    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module): Activation function
            for intermediate dense layer(s). It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable or torch.nn.Module): Activation function
            for output dense layer. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes,
        hidden_nonlinearity=F.relu,
        hidden_w_init=nn.init.xavier_normal_,
        hidden_b_init=nn.init.zeros_,
        output_nonlinearity=None,
        output_w_init=nn.init.xavier_normal_,
        output_b_init=nn.init.zeros_,
        layer_normalization=False,
    ):

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        super().__init__(
            obs_dim,
            action_dim,
            hidden_sizes,
            hidden_nonlinearity,
            hidden_w_init,
            hidden_b_init,
            output_nonlinearity,
            output_w_init,
            output_b_init,
            layer_normalization,
        )

    def get_actions(self, observations):
        assert isinstance(observations, np.ndarray) or isinstance(
            observations, torch.Tensor
        )
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = (
                    torch.as_tensor(observations)
                    .float()
                    .to(next(self.parameters()).device)
                )
            actions = self.forward(observations).argmax(dim=-1).cpu().numpy()
            infos = {}
        return actions, infos

    def get_action(self, observation):
        with torch.no_grad():
            if not isinstance(observation, torch.Tensor):
                observation = (
                    torch.as_tensor(observation)
                    .float()
                    .to(next(self.parameters()).device)
                )
            observation = observation.unsqueeze(0)
            action, agent_infos = self.get_actions(observation)
            assert action.shape == (1,), action.shape
            action = np.eye(self.action_dim)[action[0]]
            assert action.shape == (self.action_dim,), action.shape
            return action, {k: v[0] for k, v in agent_infos.items()}

    def get_param_values(self):
        """Get the parameters to the policy.

        This method is included to ensure consistency with TF policies.

        Returns:
            dict: The parameters (in the form of the state dictionary).

        """
        return self.state_dict()

    def set_param_values(self, state_dict):
        """Set the parameters to the policy.

        This method is included to ensure consistency with TF policies.

        Args:
            state_dict (dict): State dictionary.

        """
        self.load_state_dict(state_dict)
