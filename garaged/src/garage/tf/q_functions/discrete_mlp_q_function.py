"""Discrete MLP QFunction."""

import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.models import MLPDuelingModel
from garage.tf.models import MLPModel
from garage.tf.q_functions.q_function import QFunction


class DiscreteMLPQFunction(QFunction):
    """Discrete MLP Q Function.

    This class implements a Q-value network. It predicts Q-value based on the
    input state and action. It uses an MLP to fit the function Q(s, a).

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        name (str): Name of the q-function, also serves as the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means the MLP of this q-function consists of
            two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        dueling (bool): Use dueling model or not.
        layer_normalization (bool): Bool for using layer normalization.

    """

    def __init__(
        self,
        env_spec,
        name=None,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=tf.nn.relu,
        hidden_w_init=tf.initializers.glorot_uniform(
            seed=deterministic.get_tf_seed_stream()
        ),
        hidden_b_init=tf.zeros_initializer(),
        output_nonlinearity=None,
        output_w_init=tf.initializers.glorot_uniform(
            seed=deterministic.get_tf_seed_stream()
        ),
        output_b_init=tf.zeros_initializer(),
        dueling=False,
        layer_normalization=False,
    ):
        super().__init__(name)

        self._env_spec = env_spec
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._dueling = dueling
        self._layer_normalization = layer_normalization

        self.obs_dim = env_spec.observation_space.shape
        action_dim = env_spec.action_space.flat_dim

        if not dueling:
            self.model = MLPModel(
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                hidden_w_init=hidden_w_init,
                hidden_b_init=hidden_b_init,
                output_nonlinearity=output_nonlinearity,
                output_w_init=output_w_init,
                output_b_init=output_b_init,
                layer_normalization=layer_normalization,
            )
        else:
            self.model = MLPDuelingModel(
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                hidden_w_init=hidden_w_init,
                hidden_b_init=hidden_b_init,
                output_nonlinearity=output_nonlinearity,
                output_w_init=output_w_init,
                output_b_init=output_b_init,
                layer_normalization=layer_normalization,
            )

        self._network = None

        self._initialize()

    def _initialize(self):
        """Initialize QFunction."""
        obs_ph = tf.compat.v1.placeholder(
            tf.float32, (None,) + self.obs_dim, name="obs"
        )

        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            self._network = self.model.build(obs_ph)

    @property
    def q_vals(self):
        """Return the Q values, the output of the network.

        Return:
            list[tf.Tensor]: Q values.

        """
        return self._network.outputs

    @property
    def input(self):
        """Get input.

        Return:
            tf.Tensor: QFunction Input.

        """
        return self._network.input

    # pylint: disable=arguments-differ
    def get_qval_sym(self, state_input, name):
        """Symbolic graph for q-network.

        Args:
            state_input (tf.Tensor): The state input tf.Tensor to the network.
            name (str): Network variable scope.

        Return:
            tf.Tensor: The tf.Tensor output of Discrete MLP QFunction.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            return self.model.build(state_input, name=name).outputs

    def clone(self, name):
        """Return a clone of the Q-function.

        It copies the configuration of the primitive and also the parameters.

        Args:
            name (str): Name of the newly created q-function.

        Returns:
            garage.tf.q_functions.DiscreteMLPQFunction: Clone of this object

        """
        new_qf = self.__class__(
            name=name,
            env_spec=self._env_spec,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization,
        )
        new_qf.model.parameters = self.model.parameters
        return new_qf

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state.

        """
        new_dict = self.__dict__.copy()
        del new_dict["_network"]
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        self.__dict__.update(state)
        self._initialize()
