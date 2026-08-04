# Seed genome: HyperAgents' default go2walking reward.
# It rewards exactly what the fitness function measures, with no shaping —
# the honest starting point, and the thing every child has to beat.

from typing import Dict, Tuple

import torch
from torch import Tensor


def compute_reward(env) -> Tuple[Tensor, Dict, Dict]:
    """Reward = tracking of the commanded forward (x) velocity."""

    lin_vel_scale = 1.0
    lin_vel_temperature = 4.0
    lin_vel_error = torch.sum(
        torch.square(env.commands[:, :1] - env.base_lin_vel[:, :1]), dim=1
    )
    lin_vel_reward = torch.exp(-lin_vel_error * lin_vel_temperature)

    total_reward = lin_vel_scale * lin_vel_reward

    reward_components = {
        "lin_vel": lin_vel_scale * lin_vel_reward,
    }
    reward_scales = {
        "lin_vel": lin_vel_scale,
    }

    return total_reward, reward_components, reward_scales
