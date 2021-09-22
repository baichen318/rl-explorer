# Author: baichen318@gmail.com

import torch
from stable_baselines3.common.vec_env import VecEnvWrapper, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


def make_env(env, configs, idx):
    def _init():
        return env(configs, idx)

    return _init

class A3CVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(A3CVecEnvWrapper, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).long().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


def make_vec_envs(env, configs, device):
    num_process = configs["num-process"]
    envs = [
        make_env(env, configs, i + 1) for i in range(num_process)
    ]

    envs = A3CVecEnvWrapper(
        SubprocVecEnv(envs),
        device
    )

    return envs