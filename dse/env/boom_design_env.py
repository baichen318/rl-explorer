# Author: baichen318@gmail.com

import os
import torch
import random
import time
import numpy as np
from dse.env.design_space import parse_design_space
from vlsi import online_vlsi, test_online_vlsi

class BasicEnv(object):
    """ BasicEnv """
    def __init__(self, configs):
        super(BasicEnv, self).__init__()
        self.configs = configs

class BoomDesignEnv(BasicEnv):
    """ BoomDesignEnv """
    # def __init__(self, configs, seed=int(time.time())):
    def __init__(self, configs, seed=1):
        super(BoomDesignEnv, self).__init__(configs)
        self.design_space = parse_design_space(
            self.configs["design-space"],
            random_state=seed,
            basic_component=self.configs["basic-component"]
        )
        self.state = None
        self.action_list = self.construct_action_list()
        # maximal `step` call times
        self.n_step = 0
        self.last_update = 0
        self.seed = seed
        self.set_random_state(self.seed)

    def set_random_state(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def construct_action_list(self):
        action_list = []
        for k, v in self.design_space.bounds.items():
            for _v in v:
                action_list.append(_v)

        return torch.Tensor(action_list)

    def step(self, action):
        for i in range(self.configs["batch"]):
            s, idx = 0, 0
            for dim in self.design_space.dims:
                s += dim
                if action[i] < s:
                    break
                else:
                    idx += 1
            # TODO: this may cause dead-loop!
            self.state[i][idx] = self.action_list[action[i]]
        reward, ipc = [-1 for i in range(self.configs["batch"])], [0 for i in range(self.configs["batch"])]
        # if `self.state` is invalid, assign -1 to the reward directly
        for i in range(self.configs["batch"]):
            if self.design_space.validate(self.state[i].numpy()):
                reward[i] = 0
        # `valid_idx`: <tuple>
        valid_idx = torch.where(torch.tensor(reward) == 0)
        # evaluate `self.state` w.r.t. VLSI
        valid_ipc = online_vlsi(
            self.configs,
            self.state[valid_idx].numpy()
        )
        for i in range(len(valid_idx[0])):
            ipc[valid_idx[0][i]] = valid_ipc[i]
        # TODO: area & power
        ipc = torch.tensor(ipc)
        reward = ipc
        if max(reward) > max(self.best_ipc):
            self.best_ipc = reward
            best_idx = int(torch.where(self.best_ipc == max(self.best_ipc))[0])
            self.last_update = self.n_step
            msg = "[INFO]: current state: %s, best state: %s, best ipc: %.8f" % (
                self.state.numpy(),
                self.state.numpy()[best_idx],
                max(self.best_ipc)
            )
            self.configs["logger"].info(msg)
        done = bool(
            self.n_step > self.configs["total-step"] or \
            (self.n_step - self.last_update) >= self.configs["early-stopping"] or \
            torch.all(reward, -1)
        )
        self.n_step += 1
        return self.state, reward, done

    def test_step(self, action):
        """
            debug version of `step`
        """
        for i in range(self.configs["batch"]):
            s, idx = 0, 0
            for dim in self.design_space.dims:
                s += dim
                if action[i] < s:
                    break
                else:
                    idx += 1
            # TODO: this may cause dead-loop!
            self.state[i][idx] = self.action_list[action[i]]
        reward, ipc = [-1 for i in range(self.configs["batch"])], [0 for i in range(self.configs["batch"])]
        # if `self.state` is invalid, assign -1 to the reward directly
        for i in range(self.configs["batch"]):
            if self.design_space.validate(self.state[i].numpy()):
                reward[i] = 0
        # `valid_idx`: <tuple>
        valid_idx = torch.where(torch.tensor(reward) == 0)
        # evaluate `self.state` w.r.t. VLSI
        valid_ipc = test_online_vlsi(
            self.configs,
            self.state[valid_idx].numpy()
        )
        for i in range(len(valid_idx[0])):
            ipc[valid_idx[0][i]] = valid_ipc[i]
        # TODO: area & power
        ipc = torch.tensor(ipc)
        reward = ipc
        if max(reward) > max(self.best_ipc):
            self.best_ipc = reward
            best_idx = int(torch.where(self.best_ipc == max(self.best_ipc))[0])
            self.last_update = self.n_step
            msg = "[INFO]: current state: %s, best state: %s, best ipc: %.8f" % (
                self.state.numpy(),
                self.state.numpy()[best_idx],
                max(self.best_ipc)
            )
            self.configs["logger"].info(msg)
        done = bool(
            self.n_step > self.configs["total-step"] or \
            (self.n_step - self.last_update) >= self.configs["early-stopping"] or \
            torch.all(reward, -1)
        )
        self.n_step += 1
        return self.state, reward, done

    def reset(self):
        self.state = self.design_space.sample_v2(self.configs["batch"])
        self.best_ipc = torch.tensor(online_vlsi(self.configs, self.state.numpy()))
        return self.state

    def test_reset(self):
        self.state = self.design_space.sample_v2(self.configs["batch"])
        self.best_ipc = torch.tensor(test_online_vlsi(self.configs, self.state.numpy()))
        return self.state
