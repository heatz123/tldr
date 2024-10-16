import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GarageEnv
from garage.tf.q_functions import ContinuousMLPQFunction
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv, DummyDictEnv
from tests.fixtures.models import SimpleMLPMergeModel


class TestContinuousMLPQFunction(TfGraphTestCase):

    @pytest.mark.parametrize(
        "obs_dim, action_dim",
        [
            ((1,), (1,)),
            ((2,), (2,)),
            ((1, 1), (1,)),
            ((2, 2), (2,)),
        ],
    )
    def test_q_vals(self, obs_dim, action_dim):
        env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(
            ("garage.tf.q_functions." "continuous_mlp_q_function.MLPMergeModel"),
            new=SimpleMLPMergeModel,
        ):
            qf = ContinuousMLPQFunction(env_spec=env.spec)
        env.reset()
        obs, _, _, _ = env.step(1)
        obs = obs.flatten()
        act = np.full(action_dim, 0.5).flatten()

        expected_output = np.full((1,), 0.5)

        outputs = qf.get_qval([obs], [act])
        assert np.array_equal(outputs[0], expected_output)

        outputs = qf.get_qval([obs, obs, obs], [act, act, act])

        for output in outputs:
            assert np.array_equal(output, expected_output)

    def test_q_vals_goal_conditioned(self):
        env = GarageEnv(DummyDictEnv())
        with mock.patch(
            ("garage.tf.q_functions." "continuous_mlp_q_function.MLPMergeModel"),
            new=SimpleMLPMergeModel,
        ):
            qf = ContinuousMLPQFunction(env_spec=env.spec)
        env.reset()
        obs, _, _, _ = env.step(1)
        obs = np.concatenate(
            (obs["observation"], obs["desired_goal"], obs["achieved_goal"]), axis=-1
        )
        act = np.full((1,), 0.5).flatten()

        expected_output = np.full((1,), 0.5)

        outputs = qf.get_qval([obs], [act])
        assert np.array_equal(outputs[0], expected_output)

        outputs = qf.get_qval([obs, obs, obs], [act, act, act])
        for output in outputs:
            assert np.array_equal(output, expected_output)

    @pytest.mark.parametrize(
        "obs_dim, action_dim",
        [
            ((1,), (1,)),
            ((2,), (2,)),
            ((1, 1), (1,)),
            ((2, 2), (2,)),
        ],
    )
    def test_output_shape(self, obs_dim, action_dim):
        env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(
            ("garage.tf.q_functions." "continuous_mlp_q_function.MLPMergeModel"),
            new=SimpleMLPMergeModel,
        ):
            qf = ContinuousMLPQFunction(env_spec=env.spec)
        env.reset()
        obs, _, _, _ = env.step(1)
        obs = obs.flatten()
        act = np.full(action_dim, 0.5).flatten()

        outputs = qf.get_qval([obs], [act])

        assert outputs.shape == (1, 1)

    @pytest.mark.parametrize(
        "obs_dim, action_dim",
        [
            ((1,), (1,)),
            ((2,), (2,)),
            ((1, 1), (1,)),
            ((2, 2), (2,)),
        ],
    )
    def test_get_qval_sym(self, obs_dim, action_dim):
        env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(
            ("garage.tf.q_functions." "continuous_mlp_q_function.MLPMergeModel"),
            new=SimpleMLPMergeModel,
        ):
            qf = ContinuousMLPQFunction(env_spec=env.spec)
        env.reset()
        obs, _, _, _ = env.step(1)
        obs = obs.flatten()
        act = np.full(action_dim, 0.5).flatten()

        output1 = qf.get_qval([obs], [act])

        input_var1 = tf.compat.v1.placeholder(tf.float32, shape=(None, obs.shape[0]))
        input_var2 = tf.compat.v1.placeholder(tf.float32, shape=(None, act.shape[0]))
        q_vals = qf.get_qval_sym(input_var1, input_var2, "another")
        output2 = self.sess.run(
            q_vals, feed_dict={input_var1: [obs], input_var2: [act]}
        )

        expected_output = np.full((1,), 0.5)

        assert np.array_equal(output1, output2)
        assert np.array_equal(output2[0], expected_output)

    @pytest.mark.parametrize(
        "obs_dim, action_dim",
        [
            ((1,), (1,)),
            ((2,), (2,)),
            ((1, 1), (1,)),
            ((2, 2), (2,)),
        ],
    )
    def test_is_pickleable(self, obs_dim, action_dim):
        env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(
            ("garage.tf.q_functions." "continuous_mlp_q_function.MLPMergeModel"),
            new=SimpleMLPMergeModel,
        ):
            qf = ContinuousMLPQFunction(env_spec=env.spec)
        env.reset()
        obs, _, _, _ = env.step(1)
        obs = obs.flatten()
        act = np.full(action_dim, 0.5).flatten()

        with tf.compat.v1.variable_scope(
            "ContinuousMLPQFunction/SimpleMLPMergeModel", reuse=True
        ):
            return_var = tf.compat.v1.get_variable("return_var")
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())

        output1 = qf.get_qval([obs], [act])

        h_data = pickle.dumps(qf)
        with tf.compat.v1.Session(graph=tf.Graph()):
            qf_pickled = pickle.loads(h_data)
            output2 = qf_pickled.get_qval([obs], [act])

        assert np.array_equal(output1, output2)
