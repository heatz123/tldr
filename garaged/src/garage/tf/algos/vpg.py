"""Vanilla Policy Gradient."""

from garage.tf.algos.npo import NPO
from garage.tf.optimizers import FirstOrderOptimizer


class VPG(NPO):
    """Vanilla Policy Gradient.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.tf.policies.StochasticPolicy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        max_path_length (int): Maximum length of a single rollout.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        fixed_horizon (bool): Whether to fix horizon.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies, as in PPO.
        max_kl_step (float): The maximum KL divergence between old and new
            policies, as in TRPO.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in garage.tf.optimizers.
        optimizer_args (dict): The arguments of the optimizer.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        use_neg_logli_entropy (bool): Whether to estimate the entropy as the
            negative log likelihood of the action.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
        flatten_input (bool): Whether to flatten input along the observation
            dimension. If True, for example, an observation with shape (2, 4)
            will be flattened to 8.
        name (str): The name of the algorithm.

    """

    def __init__(
        self,
        env_spec,
        policy,
        baseline,
        scope=None,
        max_path_length=500,
        discount=0.99,
        gae_lambda=1,
        center_adv=True,
        positive_adv=False,
        fixed_horizon=False,
        lr_clip_range=0.01,
        max_kl_step=0.01,
        optimizer=None,
        optimizer_args=None,
        policy_ent_coeff=0.0,
        use_softplus_entropy=False,
        use_neg_logli_entropy=False,
        stop_entropy_gradient=False,
        entropy_method="no_entropy",
        flatten_input=True,
        name="VPG",
    ):
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            optimizer = FirstOrderOptimizer
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
        super().__init__(
            env_spec=env_spec,
            policy=policy,
            baseline=baseline,
            scope=scope,
            max_path_length=max_path_length,
            discount=discount,
            gae_lambda=gae_lambda,
            center_adv=center_adv,
            positive_adv=positive_adv,
            fixed_horizon=fixed_horizon,
            pg_loss="vanilla",
            lr_clip_range=lr_clip_range,
            max_kl_step=max_kl_step,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            policy_ent_coeff=policy_ent_coeff,
            use_softplus_entropy=use_softplus_entropy,
            use_neg_logli_entropy=use_neg_logli_entropy,
            stop_entropy_gradient=stop_entropy_gradient,
            entropy_method=entropy_method,
            name=name,
        )
