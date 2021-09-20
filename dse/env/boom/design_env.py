# Author: baichen318@gmail.com

import os
import torch
import random
import time
import gym
from gym import spaces
import numpy as np
from dse.env.boom.design_space import parse_design_space
from vlsi.boom.vlsi import online_vlsi, test_online_vlsi

class BasicEnv(gym.Env):
    """ BasicEnv """
    def __init__(self, configs):
        super(BasicEnv, self).__init__()
        self.configs = configs
        self.design_space = parse_design_space(
            self.configs["design-space"],
            basic_component=self.configs["basic-component"],
            random_state=self.configs["seed"]
        )
        self.action_list = self.construct_action_list()
        self.set_random_state(self.configs["seed"])

    def construct_action_list(self):
        action_list = []
        for k, v in self.design_space.bounds.items():
            for _v in v:
                action_list.append(_v)

        return torch.Tensor(action_list).long()

    def set_random_state(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


class BoomDesignEnv(BasicEnv):
    """ BoomDesignEnv """
    # def __init__(self, configs, seed=int(time.time())):
    def __init__(self, configs):
        super(BoomDesignEnv, self).__init__(configs)
        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = MultiDiscrete(self.design_space.dims)
        self.state = None
        self.n_step = 0
        self.last_update = 0

    def step(self, action):
        assert self.action_space.contains(action), "[ERROR]: action %d is unsupported" % action
        s, idx = 0, 0
        for dim in self.design_space.dims:
            s += dim
            if action < s:
                break
            else:
                idx += 1
        self.state[idx] = self.action_list[action]

        reward = torch.Tensor(self.design_space.evaluate_microarchitecture(self.state))
        msg = "[INFO]: state: %s, reward: %s, current best ipc: %s" % (self.state.numpy(), reward)
        self.configs["logger"].info(msg)
        if reward > self.best_reward:
            self.best_reward = reward
            self.last_update = self.n_step
            msg = "[INFO]: current best state: %s, current best reward: %.8f" % (
                self.state.numpy(),
                self.best_reward
            )
            self.configs["logger"].info(msg)
        done = bool(
            self.n_step > self.configs["total-step"] or \
            (self.n_step - self.last_update) >= self.configs["early-stopping"]
        )
        self.n_step += 1
        return self.state.clone(), reward.clone(), done, {}

    def reset(self):
        # Notice: `self.configs["batch"] // 5` is required
        self.state = self.design_space.sample_v2(1)
        self.n_step = 0
        return self.state.clone()

    def render(self):
        return None

    def close(self):
        return None
