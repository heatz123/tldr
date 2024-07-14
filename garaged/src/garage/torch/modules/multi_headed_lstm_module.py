"""MultiHeadedMLPModule."""

import copy

import torch
import torch.nn as nn

from garagei.torch.modules.spectral_norm import spectral_norm


class MultiHeadedLSTMModule(nn.Module):
    """MultiHeadedLSTMModule Model.

    A PyTorch module composed only of a multi-layer perceptron (MLP) with
    multiple parallel output layers which maps real-valued inputs to
    real-valued outputs. The length of outputs is n_heads and shape of each
    output element is depend on each output dimension

    Args:
        n_heads (int): Number of different output layers
        input_dim (int): Dimension of the network input.
        output_dims (int or list or tuple): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module or list or tuple):
            Activation function for intermediate dense layer(s).
            It should return a torch.Tensor. Set it to None to maintain a
            linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearities (callable or torch.nn.Module or list or tuple):
            Activation function for output dense layer. It should return a
            torch.Tensor. Set it to None to maintain a linear activation.
            Size of the parameter should be 1 or equal to n_head
        output_w_inits (callable or list or tuple): Initializer function for
            the weight of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        output_b_inits (callable or list or tuple): Initializer function for
            the bias of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(
        self,
        n_heads,
        input_dim,
        output_dims,
        hidden_sizes,
        hidden_nonlinearity=torch.relu,
        hidden_w_init=nn.init.xavier_normal_,
        hidden_b_init=nn.init.zeros_,
        output_nonlinearities=None,
        output_w_inits=nn.init.xavier_normal_,
        output_b_inits=nn.init.zeros_,
        layer_normalization=False,
        bias=True,
        spectral_normalization=False,
        spectral_coef=1.0,
    ):  # todo heatz123: check the arguments are correct for lstms
        super().__init__()

        self._layers = nn.ModuleList()

        output_dims = self._check_parameter_for_output_layer(
            "output_dims", output_dims, n_heads
        )
        output_w_inits = self._check_parameter_for_output_layer(
            "output_w_inits", output_w_inits, n_heads
        )
        output_b_inits = self._check_parameter_for_output_layer(
            "output_b_inits", output_b_inits, n_heads
        )
        output_nonlinearities = self._check_parameter_for_output_layer(
            "output_nonlinearities", output_nonlinearities, n_heads
        )

        self._layers = nn.ModuleList()

        prev_size = input_dim

        # assert all elements of hidden_sizes are equal
        assert all(x == hidden_sizes[0] for x in hidden_sizes)

        size = hidden_sizes[0]
        hidden_layers = nn.Sequential()
        if spectral_normalization:
            linear_layer = spectral_norm(
                nn.LSTM(
                    prev_size,
                    size,
                    num_layers=len(hidden_sizes),
                    bias=bias,
                    batch_first=True,
                ),
                spectral_coef=spectral_coef,
            )
        else:
            linear_layer = nn.LSTM(
                prev_size,
                size,
                num_layers=len(hidden_sizes),
                bias=bias,
                batch_first=True,
            )
        for name, param in linear_layer.named_parameters():
            if "weight_ih" in name:
                hidden_w_init(param)
            elif "weight_hh" in name:
                hidden_w_init(param)
            elif "weight" in name:
                print("new weight found")
                print(name)
            elif "bias_ih" in name:
                hidden_b_init(param)
            elif "bias_hh" in name:
                hidden_b_init(param)
            elif "bias" in name:
                print("new bias found")
                print(name)

        # hidden_w_init(linear_layer.weight)
        # if bias:
        #     hidden_b_init(linear_layer.bias)
        # hidden_layers.add_module('linear', linear_layer)
        self.lstm = linear_layer

        assert not layer_normalization, "layer normalization not supported for lstms"
        # if layer_normalization:
        #     hidden_layers.add_module('layer_normalization', nn.LayerNorm(size))

        # if hidden_nonlinearity:
        #     hidden_layers.add_module('non_linearity', _NonLinearity(hidden_nonlinearity))

        self._layers.append(hidden_layers)
        prev_size = size

        self._output_layers = nn.ModuleList()
        for i in range(n_heads):
            output_layer = nn.Sequential()
            if spectral_normalization:
                linear_layer = spectral_norm(
                    nn.Linear(prev_size, output_dims[i], bias=bias),
                    spectral_coef=spectral_coef,
                )
            else:
                linear_layer = nn.Linear(prev_size, output_dims[i], bias=bias)
            output_w_inits[i](linear_layer.weight)
            if bias:
                output_b_inits[i](linear_layer.bias)
            output_layer.add_module("linear", linear_layer)

            if output_nonlinearities[i]:
                output_layer.add_module(
                    "non_linearity", _NonLinearity(output_nonlinearities[i])
                )

            self._output_layers.append(output_layer)

    @classmethod
    def _check_parameter_for_output_layer(cls, var_name, var, n_heads):
        """Check input parameters for output layer are valid.

        Args:
            var_name (str): variable name
            var (any): variable to be checked
            n_heads (int): number of head

        Returns:
            list: list of variables (length of n_heads)

        Raises:
            ValueError: if the variable is a list but length of the variable
                is not equal to n_heads

        """
        if isinstance(var, (list, tuple)):
            if len(var) == 1:
                return list(var) * n_heads
            if len(var) == n_heads:
                return var
            msg = (
                "{} should be either an integer or a collection of length "
                "n_heads ({}), but {} provided."
            )
            raise ValueError(msg.format(var_name, n_heads, var))
        return [copy.deepcopy(var) for _ in range(n_heads)]

    # pylint: disable=arguments-differ
    def forward(self, input_val, hidden_states=None):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        # #### debug
        # hidden_states = None
        # #### end debug

        x = input_val
        if hidden_states is not None:
            x, (hn, cn) = self.lstm(x, hidden_states)
        else:
            x, (hn, cn) = self.lstm(x)
        for layer in self._layers:
            x = layer(x)

        return [output_layer(x) for output_layer in self._output_layers], (hn, cn)

    def get_last_linear_layer(self):
        for m in reversed(self._layers):
            if isinstance(m, nn.Sequential):
                for l in reversed(m):
                    if isinstance(l, nn.Linear):
                        return l
            if isinstance(m, nn.Linear):
                return m
        return None

    def reset(self, dones=None):
        self.last_hidden_state = None  # note here that only one class is used
        # for all rollouts with different threads, this can be problematic
        # todo: fix
        # then what about just get the hidden state every time and feed it every time
        # to make it stateless? # TODO heatz123


class _NonLinearity(nn.Module):
    """Wrapper class for non linear function or module.

    Args:
        non_linear (callable or type): Non-linear function or type to be
            wrapped.

    """

    def __init__(self, non_linear):
        super().__init__()

        if isinstance(non_linear, type):
            self.module = non_linear()
        elif callable(non_linear):
            self.module = copy.deepcopy(non_linear)
        else:
            raise ValueError(
                "Non linear function {} is not supported".format(non_linear)
            )

    # pylint: disable=arguments-differ
    def forward(self, input_value):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values

        Returns:
            torch.Tensor: Output value

        """
        return self.module(input_value)

    # pylint: disable=missing-return-doc, missing-return-type-doc
    def __repr__(self):
        return repr(self.module)
