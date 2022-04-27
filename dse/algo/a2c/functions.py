# Author: baichen318@gmail.com


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


def make_env(env, configs, idx):
    def _init():
        return env(configs, idx)

    return _init


class A2CVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, device):
        """
            Return only every `skip`-th frame
        """
        super(A2CVecEnvWrapper, self).__init__(venv)
        self.reward_space = venv.get_attr("reward_space")[0]
        self.device = device

    def reset(self):
        """
            override `A2CVecEnvWrapper.reset`, refer it to:
                https://github.com/hill-a/stable-baselines/blob/14630fbac70aaa633f8a331c8efac253d3ed6115/stable_baselines/common/vec_env/base_vec_env.py#L35
            for `SubprocVecEnv.reset`, refer it to:
                https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/vec_env/subproc_vec_env.py#L127
            `SubprocVecEnv.reset` expands the dimension
        """
        return self.venv.reset()

    def step_wait(self):
        """
            override `A2CVecEnvWrapper.step_wait`, refer it to:
                https://github.com/hill-a/stable-baselines/blob/14630fbac70aaa633f8a331c8efac253d3ed6115/stable_baselines/common/vec_env/base_vec_env.py#L237
        """
        return self.venv.step_wait()


def make_vec_envs(configs, device, env):
    envs = [
        make_env(env, configs, i + 1) \
            for i in range(configs["num-parallel"])
    ]

    envs = A2CVecEnvWrapper(
        SubprocVecEnv(envs),
        device
    )

    return envs


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        _init = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01
        )
        self.linear = _init(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


def array_to_tensor(array, device=torch.device("cpu"), fmt=float):
    """
        array: <numpy.ndarray>
        device: <class 'torch.device'>
    """
    if fmt == int:
        return torch.Tensor(array).to(device).long()
    return torch.Tensor(array).to(device).float()


def tensor_to_array(tensor, device=torch.device("cpu")):
    """
        tensor: <torch.Tensor>
        device: <class 'torch.device'>
    """
    return tensor.data.cpu().numpy()
