#!/usr/bin/env python3
"""This is an example to train a task with PPO algorithm.

Here it creates InvertedDoublePendulum using gym. And uses a PPO with 1M
steps.

Results:
    AverageDiscountedReturn: 500
    RiseTime: itr 40

"""
import gym
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.policies import GaussianMLPPolicy


@wrap_experiment
def ppo_pendulum(ctxt=None, seed=1):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = GarageEnv(normalize(gym.make("InvertedDoublePendulum-v2")))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(32, 32),
                use_trust_region=True,
            ),
        )

        # NOTE: make sure when setting entropy_method to 'max', set
        # center_adv to False and turn off policy gradient. See
        # tf.algos.NPO for detailed documentation.
        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
            ),
            stop_entropy_gradient=True,
            entropy_method="max",
            policy_ent_coeff=0.02,
            center_adv=False,
        )

        runner.setup(algo, env)

        runner.train(n_epochs=120, batch_size=2048, plot=False)


ppo_pendulum(seed=1)
