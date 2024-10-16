# flake8: noqa
# pylint: skip-file
import numpy as np
import tensorflow as tf

from garage.tf.distributions.distribution import Distribution

TINY = 1e-8


class Bernoulli(Distribution):

    def __init__(self, dim, name="Bernoulli"):
        self._name = name
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars, name="kl_sym"):
        with tf.name_scope(name):
            old_p = old_dist_info_vars["p"]
            new_p = new_dist_info_vars["p"]
            kl = old_p * (tf.math.log(old_p + TINY) - tf.math.log(new_p + TINY)) + (
                1 - old_p
            ) * (tf.math.log(1 - old_p + TINY) - tf.math.log(1 - new_p + TINY))
            ndims = kl.get_shape().ndims
            return tf.reduce_sum(kl, axis=ndims - 1)

    def kl(self, old_dist_info, new_dist_info):
        old_p = old_dist_info["p"]
        new_p = new_dist_info["p"]
        kl = old_p * (np.log(old_p + TINY) - np.log(new_p + TINY)) + (1 - old_p) * (
            np.log(1 - old_p + TINY) - np.log(1 - new_p + TINY)
        )
        return np.sum(kl, axis=-1)

    def sample(self, dist_info):
        p = np.asarray(dist_info["p"])
        return np.cast["int"](np.random.uniform(low=0.0, high=1.0, size=p.shape) < p)

    def likelihood_ratio_sym(
        self, x_var, old_dist_info_vars, new_dist_info_vars, name="likelihood_ratio_sym"
    ):
        with tf.name_scope(name):
            old_p = old_dist_info_vars["p"]
            new_p = new_dist_info_vars["p"]
            ndims = old_p.get_shape().ndims
            return tf.reduce_prod(
                x_var * new_p / (old_p + TINY)
                + (1 - x_var) * (1 - new_p) / (1 - old_p + TINY),
                axis=ndims - 1,
            )

    def log_likelihood_sym(self, x_var, dist_info_vars, name="log_likelihood_sym"):
        with tf.name_scope(name):
            p = dist_info_vars["p"]
            ndims = p.get_shape().ndims
            return tf.reduce_sum(
                x_var * tf.math.log(p + TINY) + (1 - x_var) * tf.math.log(1 - p + TINY),
                axis=ndims - 1,
            )

    def log_likelihood(self, xs, dist_info):
        p = dist_info["p"]
        return np.sum(xs * np.log(p + TINY) + (1 - xs) * np.log(1 - p + TINY), axis=-1)

    def entropy(self, dist_info):
        p = dist_info["p"]
        return np.sum(-p * np.log(p + TINY) - (1 - p) * np.log(1 - p + TINY), axis=-1)

    @property
    def dist_info_keys(self):
        return ["p"]
