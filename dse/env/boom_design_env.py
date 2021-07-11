# Author: baichen318@gmail.com

import os
import torch
import random
import time
import numpy as np
from design_space import parse_design_space
from vlsi.vlsi import online_vlsi
from util.util import create_logger

class BasicEnv(object):
    """ BasicProblem """
    def __init__(self, configs):
        super(BasicProblem, self).__init__()
        self.configs = configs
        self.logger = create_logger(
            os.path.dirname(self.configs["log"]),
            os.path.basename(self.configs["log"])
        )

class BoomDesignEnv(BasicEnv):
    """ BoomDesignProblem """
    def __init__(self, configs, seed=int(time.time())):
        super(BoomDesignProblem, self).__init__(configs)
        self.design_space = parse_design_space(
            self.configs["design-space"],
            random_state=seed,
            basic_component=self.configs["basic-component"]
        )
        self.state = None
        self._action_list = self.construct_action_list()
        # maximal `step` call times
        self.n_step = 0
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

    @property
    def action_list(self, index=None):
        if index:
            return self._action_list
        return self._action_list[index]

    def step(self, action):
        s, idx = 0, 0
        for dim in self.design_space.dims:
            s += dim
            if action < s:
                break
            else:
                idx += 1
        if self.state[idx] == self.action_list(index):
            # TODO: this may cause dead-loop!
            pass
        else:
            self.state[idx] = self.env.action_list(index)

        # if `self.state` is invalid, assign -1 to the reward directly
        if self.design_space.validate(self.state.numpy()):
            reward = -1
        else:
            # evaluate `self.state` w.r.t. VLSI
            ipc = online_vlsi(self.configs, self.state.numpy())
            # TODO: area & power
            reward = ipc
        if reward > self.best_ipc:
            self.best_ipc = reward
            self.last_update = self.n_step
            msg = "[INFO]: current state: %s, ipc: %.8f" % (
                self.state.numpy(),
                self.best_ipc
            )
            self.logger.info(msg)
        done = bool(
            self.n_step > self.configs["total-step"] or \
            (self.n_step - self.last_update) >= self.configs["early-stopping"]
        )
        self.n_step += 1
        return self.state, reward, done

    def reset(self):
        self.state = self.design_space.sample(1)
        self.best_ipc = online_vlsi(self.configs, self.state.numpy())
        return self.state
