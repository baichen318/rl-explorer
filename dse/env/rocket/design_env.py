# Author: baichen318@gmail.com

import os
import torch
import random
import time
import logging
import gym
from gym import spaces
import numpy as np
from dse.env.rocket.design_space import parse_design_space
from vlsi.rocket.vlsi import online_vlsi, test_online_vlsi

class BasicEnv(gym.Env):
    """ BasicEnv """
    def __init__(self, configs, idx):
        super(BasicEnv, self).__init__()
        self.configs = configs
        self.idx = idx
        self.design_space = parse_design_space(
            self.configs["design-space"],
            basic_component=self.configs["basic-component"],
            random_state=self.configs["seed"],
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

    def info(self, msg):
        print(msg)
        time_str = time.strftime("%Y-%m-%d-%H-%M")
        with open(self.configs["log-file"], 'a') as f:
            f.write(time_str + ' ' + msg + '\n')


class RocketDesignEnv(BasicEnv):
    """ RocketDesignEnv """
    def __init__(self, configs, idx):
        super(RocketDesignEnv, self).__init__(configs, idx)
        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = spaces.MultiDiscrete(self.design_space.dims)
        self.state = None
        self.n_step = 0
        self.last_update = 0
        self.best_reward = 0

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

        reward = self.design_space.evaluate_microarchitecture(
            self.configs,
            self.state.numpy(),
            self.idx
        )
        msg = "[INFO]: state: %s, reward: %s" % (self.state.numpy(), reward)
        self.info(msg)
        if reward > self.best_reward:
            self.best_reward = reward
            self.last_update = self.n_step
            msg = "[INFO]: current best state: %s, current best reward: %.8f" % (
                self.state.numpy(),
                self.best_reward
            )
            self.info(msg)
        done = bool(
            self.n_step > self.configs["num-env-step"] or \
            (self.n_step - self.last_update) >= self.configs["early-stopping"]
        )
        self.n_step += 1
        # construct `info`
        info = {}
        return self.state.clone(), reward.clone(), done, dict(reward=float(reward))

    def reset(self):
        self.state = self.design_space.sample(1)
        self.n_step = 0
        return self.state

    def render(self):
        return NotImplementedError

    def close(self):
        return NotImplementedError