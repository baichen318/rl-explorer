# Author: baichen318@gmail.com

import os
import torch
import random
import time
import gym
from gym import spaces
import numpy as np
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from dse.env.boom.design_space import parse_design_space
from vlsi.boom.vlsi import online_vlsi, test_online_vlsi

class BasicEnv(gym.Env):
    """ BasicEnv """
    def __init__(self, configs, idx):
        super(BasicEnv, self).__init__()
        self.configs = configs
        # NOTICE: `self.idx`, a key to distinguish different
        # gem5 repo.
        self.idx = idx
        # NOTICE: every agent should have different initial seeds,
        # so we make a small perturbation.
        seed = round(self.idx + np.random.rand() * self.configs["seed"])
        self.design_space = parse_design_space(
            self.configs["design-space"],
            basic_component=self.configs["basic-component"],
            random_state=seed
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


class BoomDesignEnv(BasicEnv):
    """ BoomDesignEnv """
    def __init__(self, configs, idx):
        super(BoomDesignEnv, self).__init__(configs)
        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = spaces.MultiDiscrete(self.design_space.dims)
        self.state = None
        self.n_step = 0
        self.last_update = 0
        self.best_reward = 0
        self.load_model()

    def load_model(self):
        self.ipc_model = joblib.load(
            os.path.join(
                "tools",
                self.configs["ppa-model"],
                self.configs["design"] + '-' + "ipc.pt"
            )
        )
        self.power_model = joblib.load(
            os.path.join(
                "tools",
                self.configs["ppa-model"],
                self.configs["design"] + '-' + "power.pt"
            )
        )
        self.area_model = joblib.load(
            os.path.join(
                "tools",
                self.configs["ppa-model"],
                self.configs["design"] + '-' + "area.pt"
            )
        )

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

        ipc, power, area = self.design_space.evaluate_microarchitecture(
            self.configs,
            self.state.numpy().astype(int),
            self.idx
        )
        ipc = self.ipc_model.predict(
            np.expand_dims(
                np.concatenate((self.state.numpy(), [ipc])),
                axis=0
            )
        )
        power = self.power_model.predict(
            np.expand_dims(
                np.concatenate((self.state.numpy(), [power])),
                axis=0
            )
        )
        area = self.area_model.predict(
            np.expand_dims(
                np.concatenate((self.state.numpy(), [area])),
                axis=0
            )
        )
        # NOTICE: scale it manually!
        power = 10 * power
        area = 1e-6 * area
        reward = torch.Tensor(
            np.concatenate((ipc, -power, -area))
        ).squeeze(0)
        msg = "[INFO]: state: %s, reward: %s" % (self.state.numpy(), reward)
        self.info(msg)
        done = bool(self.n_step > self.configs["num-env-step"])
        self.n_step += 1
        return self.state.clone(), reward, done, {}

    def reset(self):
        self.state = self.design_space.sample_v2(1)
        self.n_step = 0
        return self.state

    def render(self):
        return NotImplementedError

    def close(self):
        return NotImplementedError
