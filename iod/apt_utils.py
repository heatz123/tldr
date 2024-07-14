import torch
import torch.nn as nn


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class RMS(object):
    """running mean and std"""

    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (
            self.S * self.n
            + torch.var(x, dim=0) * bs
            + torch.square(delta) * self.n * bs / (self.n + bs)
        ) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class PBE(object):
    """particle-based entropy based on knn normalized by running mean"""

    # https://github.com/rll-research/url_benchmark/blob/main/utils.py#L279

    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms, device):
        self.rms = rms
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.knn_clip = knn_clip  # default: 0.0
        self.device = device

    def get_reward(self, source, target, use_rms=False):
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(
            source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1),
            dim=-1,
            p=2,
        )
        reward, _ = sim_matrix.topk(
            self.knn_k, dim=1, largest=False, sorted=True
        )  # (b1, k)
        assert self.knn_avg
        if use_rms:
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            reward /= self.rms(reward)[0]  # divide by mean
        reward = (
            torch.maximum(reward - self.knn_clip, torch.zeros_like(reward))
            if self.knn_clip >= 0.0
            else reward
        )  # (b1, k)
        reward = reward.reshape((b1, self.knn_k))  # (b1, k)
        reward = reward.mean(dim=1, keepdim=True)  # (b1, 1)
        reward = torch.log(reward + 1.0)

        return reward

    def __call__(self, rep, next_rep, use_rms=False):
        source = target = rep
        reward_1 = self.get_reward(source, target, use_rms=use_rms)
        reward_2 = self.get_reward(next_rep, target, use_rms=use_rms)

        reward = reward_2 - reward_1
        return reward
