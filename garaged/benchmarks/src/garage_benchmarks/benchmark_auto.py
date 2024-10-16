"""Automatic benchmarking."""

from garage_benchmarks.experiments.algos import ddpg_garage_tf
from garage_benchmarks.experiments.algos import ppo_garage_pytorch
from garage_benchmarks.experiments.algos import ppo_garage_tf
from garage_benchmarks.experiments.algos import td3_garage_tf
from garage_benchmarks.experiments.algos import trpo_garage_pytorch
from garage_benchmarks.experiments.algos import trpo_garage_tf
from garage_benchmarks.experiments.algos import vpg_garage_pytorch
from garage_benchmarks.experiments.algos import vpg_garage_tf
from garage_benchmarks.helper import benchmark, iterate_experiments
from garage_benchmarks.parameters import MuJoCo1M_ENV_SET


@benchmark(plot=False, auto=True)
def auto_ddpg_benchmarks():
    """Run experiments for DDPG benchmarking."""
    iterate_experiments(ddpg_garage_tf, MuJoCo1M_ENV_SET)


@benchmark(plot=False, auto=True)
def auto_ppo_benchmarks():
    """Run experiments for PPO benchmarking."""
    iterate_experiments(ppo_garage_pytorch, MuJoCo1M_ENV_SET)
    iterate_experiments(ppo_garage_tf, MuJoCo1M_ENV_SET)


@benchmark(plot=False, auto=True)
def auto_td3_benchmarks():
    """Run experiments for TD3 benchmarking."""
    td3_env_ids = [env_id for env_id in MuJoCo1M_ENV_SET if env_id != "Reacher-v2"]

    iterate_experiments(td3_garage_tf, td3_env_ids)


@benchmark(plot=False, auto=True)
def auto_trpo_benchmarks():
    """Run experiments for TRPO benchmarking."""
    iterate_experiments(trpo_garage_pytorch, MuJoCo1M_ENV_SET)
    iterate_experiments(trpo_garage_tf, MuJoCo1M_ENV_SET)


@benchmark(plot=False, auto=True)
def auto_vpg_benchmarks():
    """Run experiments for VPG benchmarking."""
    iterate_experiments(vpg_garage_pytorch, MuJoCo1M_ENV_SET)
    iterate_experiments(vpg_garage_tf, MuJoCo1M_ENV_SET)
