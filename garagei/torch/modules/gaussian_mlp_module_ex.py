import torch
from garage.torch.distributions import TanhNormal
from garage.torch.modules import MultiHeadedMLPModule
from garage.torch.modules.gaussian_mlp_module import (
    GaussianMLPModule,
    GaussianMLPIndependentStdModule,
    GaussianMLPTwoHeadedModule,
    GaussianMLPBaseModule,
)
from garage.torch.modules.mlp_module import MLPModule
from torch import nn
from torch.distributions import Normal, Categorical, MixtureSameFamily
from torch.distributions.independent import Independent


class ForwardWithTransformTrait(object):
    def forward_with_transform(self, *inputs, transform):
        mean, log_std_uncentered = self._get_mean_and_log_std(*inputs)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(
                    None if self._min_std_param is None else self._min_std_param.item()
                ),
                max=(
                    None if self._max_std_param is None else self._max_std_param.item()
                ),
            )

        if self._std_parameterization == "exp":
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.0).log()

        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pre_tanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        mean = transform(mean)
        std = transform(std)

        dist_transformed = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pre_tanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist_transformed, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist_transformed.batch_shape samples.
            dist_transformed = Independent(dist_transformed, 1)

        return dist, dist_transformed


class ForwardWithChunksTrait(object):
    def forward_with_chunks(self, *inputs, merge):
        mean = []
        log_std_uncentered = []
        for chunk_inputs in zip(*inputs):
            chunk_mean, chunk_log_std_uncentered = self._get_mean_and_log_std(
                *chunk_inputs
            )
            mean.append(chunk_mean)
            log_std_uncentered.append(chunk_log_std_uncentered)
        mean = merge(mean, batch_dim=0)
        log_std_uncentered = merge(log_std_uncentered, batch_dim=0)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(
                    None if self._min_std_param is None else self._min_std_param.item()
                ),
                max=(
                    None if self._max_std_param is None else self._max_std_param.item()
                ),
            )

        if self._std_parameterization == "exp":
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.0).log()
        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist


class ForwardModeTrait(object):
    def forward_mode(self, *inputs):
        mean, log_std_uncentered = self._get_mean_and_log_std(*inputs)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(
                    None if self._min_std_param is None else self._min_std_param.item()
                ),
                max=(
                    None if self._max_std_param is None else self._max_std_param.item()
                ),
            )

        if self._std_parameterization == "exp":
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.0).log()

        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pre_tanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist.mean


class GaussianMLPModuleEx(
    GaussianMLPModule,
    ForwardWithTransformTrait,
    ForwardWithChunksTrait,
    ForwardModeTrait,
):
    pass


class GaussianMLPIndependentStdModuleEx(
    GaussianMLPIndependentStdModule,
    ForwardWithTransformTrait,
    ForwardWithChunksTrait,
    ForwardModeTrait,
):
    pass


class GaussianMLPTwoHeadedModuleEx(
    GaussianMLPTwoHeadedModule,
    ForwardWithTransformTrait,
    ForwardWithChunksTrait,
    ForwardModeTrait,
):
    pass
