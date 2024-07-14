import numpy as np

import torch
from torch import nn

from garagei.torch.modules.spectral_norm import spectral_norm


class NormLayer(nn.Module):
    def __init__(self, name, dim=None):
        super().__init__()
        if name == "none":
            self._layer = None
        elif name == "layer":
            assert dim != None
            self._layer = nn.LayerNorm(dim)
        else:
            raise NotImplementedError(name)

    def forward(self, features):
        if self._layer is None:
            return features
        return self._layer(features)


class CNN(nn.Module):
    def __init__(
        self,
        num_inputs,
        act=nn.ELU,
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=(400, 400, 400, 400),
        spectral_normalization=False,
    ):
        super().__init__()

        self._num_inputs = num_inputs
        self._act = act()
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

        self._conv_model = []
        for i, kernel in enumerate(self._cnn_kernels):
            test = [31, 14, 6, 2]  # TODO: fix it
            if i == 0:
                prev_depth = num_inputs
            else:
                prev_depth = 2 ** (i - 1) * self._cnn_depth
            depth = 2**i * self._cnn_depth
            if spectral_normalization:
                self._conv_model.append(
                    spectral_norm(nn.Conv2d(prev_depth, depth, kernel, stride=2))
                )
            else:
                self._conv_model.append(nn.Conv2d(prev_depth, depth, kernel, stride=2))
            self._conv_model.append(
                NormLayer(
                    norm,
                    dim=(
                        depth,
                        test[i],
                        test[i],
                    ),
                )
            )
            self._conv_model.append(self._act)
        self._conv_model = nn.Sequential(*self._conv_model)

    def forward(self, data):
        output = self._conv_model(data)
        output = output.reshape(output.shape[0], -1)
        return output


class Encoder(nn.Module):
    def __init__(
        self,
        pixel_shape,
        spectral_normalization=False,
        encode_goal=False,
        use_atari_torso=True,
        use_separate_encoder=True,
        **kwargs,
    ):
        super().__init__()

        self.pixel_shape = pixel_shape
        self.pixel_dim = np.prod(pixel_shape)

        self.pixel_depth = self.pixel_shape[-1]
        if use_atari_torso:
            self.encoder = AtariTorso(self.pixel_shape, **kwargs)
        else:
            self.encoder = CNN(
                self.pixel_depth,
                spectral_normalization=spectral_normalization,
                **kwargs,
            )

        self.encode_goal = encode_goal
        if self.encode_goal:
            if use_separate_encoder:
                self.goal_encoder = AtariTorso(self.pixel_shape, **kwargs)
            else:
                self.goal_encoder = self.encoder

    def forward(self, input):
        assert input.ndim == 2
        batch_size = input.shape[0]
        if self.encode_goal:
            pixel = (
                input[..., : 2 * self.pixel_dim]
                .reshape(batch_size, 2, *self.pixel_shape)
                .reshape(batch_size * 2, *self.pixel_shape)  # N*2, 64, 64, 3
                .permute(0, 3, 1, 2)  # N*2, 3, 64, 64
            )

            state = input[..., self.pixel_dim * 2 :]
        else:
            pixel = (
                input[..., : self.pixel_dim]
                .reshape(-1, *self.pixel_shape)
                .permute(0, 3, 1, 2)
            )
            state = input[..., self.pixel_dim :]

        pixel = pixel / 255.0

        rep = self.encoder(pixel)  # N*2, d (currently interleaved)

        if self.encode_goal:
            rep = rep.reshape(batch_size, 2, -1).reshape(batch_size, -1)  # b, 2, d
        else:
            rep = rep.reshape(rep.shape[0], -1)

        output = torch.cat([rep, state], dim=-1)

        return output


class WithEncoder(nn.Module):
    def __init__(self, encoder, module):
        super().__init__()

        self.encoder = encoder
        self.module = module

    def get_rep(self, input):
        return self.encoder(input)

    def forward(self, *inputs):
        rep = self.get_rep(inputs[0])
        return self.module(rep, *inputs[1:])

    def forward_mode(self, *inputs):
        rep = self.get_rep(inputs[0])
        return self.module.forward_mode(rep, *inputs[1:])


class DimensionsSelector(nn.Module):
    def __init__(self, module, start_dim: int = 0):  # range object
        super().__init__()

        self.module = module
        self.start_dim = start_dim

    def forward(self, *inputs):
        return self.module(inputs[0][..., self.start_dim :], *inputs[1:])

    def forward_mode(self, *inputs):
        return self.module(inputs[0][..., self.start_dim :], *inputs[1:])


class AtariTorso(nn.Module):
    def __init__(
        self, input_shape: torch.Size, activate_last: bool = True, **kwargs
    ) -> None:
        super().__init__()
        assert tuple(input_shape) in [(64, 64, 3), (64, 64, 4), (64, 64, 9)]

        self.input_shape = input_shape
        self.torso = nn.Sequential(
            nn.Conv2d(self.input_shape[-1], 32, (8, 8), (4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1)),
            nn.ReLU() if activate_last else nn.Identity(),
            nn.Flatten(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input is 3 x 64 x 64, (already permuted)
        ret = self.torso(input - 0.5).unflatten(0, input.shape[:-3])
        return ret
