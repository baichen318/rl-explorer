# Author: baichen318@gmail.com

import os
import torch
import random
import time
import numpy as np
from dse.env.rocket.design_space import parse_design_space
from vlsi.rocket.vlsi import online_vlsi, test_online_vlsi

class BasicEnv(object):
    """ BasicEnv """
    def __init__(self, configs):
        super(BasicEnv, self).__init__()
        self.configs = configs

class RocketDesignEnv(BasicEnv):
    """ RocketDesignEnv """
    # def __init__(self, configs, seed=int(time.time())):
    def __init__(self, configs, seed=2021):
        super(RocketDesignEnv, self).__init__(configs)
        self.design_space = parse_design_space(
            self.configs["design-space"],
            basic_component=self.configs["basic-component"],
            random_state=seed,
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

        return torch.Tensor(action_list).long()

    def step(self, action):
        reward = [-1 for i in range(self.configs["batch"])]
        dont_vlsi = [0 for i in range(self.configs["batch"])]
        for i in range(self.configs["batch"]):
            s, idx = 0, 0
            for dim in self.design_space.dims:
                s += dim
                if action[i] < s:
                    break
                else:
                    idx += 1
            # TODO: this may cause dead-loop!
            # we need to avoid redundant VLSI
            if self.state[i][idx] == self.action_list[action[i]]:
                # the state is not changed, so assign the reward zero
                dont_vlsi[i] = 1
                reward[i] = 0
            else:
                self.state[i][idx] = self.action_list[action[i]]
        # if `self.state` is invalid, assign -1 to the reward directly
        for i in range(self.configs["batch"]):
            if not self.design_space.validate(self.state[i].numpy()):
                # no transition
                self.configs["logger"].info("[INFO]: %s is invalid." % self.state[i].numpy())
                dont_vlsi[i] = 1
                reward[i] = 0
        # `valid_idx`: <torch.Tensor>
        vlsi_idx = torch.where(torch.Tensor(dont_vlsi) == 0)[0]
        # evaluate `self.state` w.r.t. VLSI
        if vlsi_idx.shape[0] != 0:
            ipc = torch.Tensor(
                online_vlsi(
                    self.configs,
                    self.state[vlsi_idx].numpy()
                )
            )
            k = 0
            for i in vlsi_idx:
                reward[i] = ipc[k]
                # update `self.best_ipc`
                if reward[i] > self.ipc[i]:
                    self.ipc[i] = ipc[k]
                k += 1
        reward = torch.Tensor(reward)
        msg = "[INFO]: state: %s, reward: %s, current best ipc: %s" % (self.state.numpy(), reward, self.ipc)
        self.configs["logger"].info(msg)
        if bool(torch.lt(
                self.best_ipc,
                torch.max(self.ipc)
            )
        ):
            # the best IPC is increased
            self.best_ipc = torch.max(self.ipc)
            best_idx = torch.where(self.ipc == self.best_ipc)[0]
            self.last_update = self.n_step
            msg = "[INFO]: current state: %s, global best state: %s, global best ipc: %.8f" % (
                self.state.numpy(),
                self.state.numpy()[int(best_idx)],
                self.best_ipc
            )
            self.configs["logger"].info(msg)
        done = bool(
            self.n_step > self.configs["total-step"] or \
            (self.n_step - self.last_update) >= self.configs["early-stopping"] or \
            # all ipc equals to -1
            (torch.where(reward == -1)[0].shape == reward.shape) or \
            # all ipc equals to 0
            (torch.where(reward == 0)[0].shape == reward.shape)
        )
        self.n_step += 1
        return self.state.clone(), reward.clone(), done

    def test_step(self, action):
        """
            debug version of `step`
        """
        reward = [-1 for i in range(self.configs["batch"])]
        dont_vlsi = [0 for i in range(self.configs["batch"])]
        for i in range(self.configs["batch"]):
            s, idx = 0, 0
            for dim in self.design_space.dims:
                s += dim
                if action[i] < s:
                    break
                else:
                    idx += 1
            # TODO: this may cause dead-loop!
            # we need to avoid redundant VLSI
            if self.state[i][idx] == self.action_list[action[i]]:
                # the state is not changed, so assign the reward zero
                dont_vlsi[i] = 1
                reward[i] = 0
            else:
                self.state[i][idx] = self.action_list[action[i]]
        # if `self.state` is invalid, assign -1 to the reward directly
        for i in range(self.configs["batch"]):
            if not self.design_space.validate(self.state[i].numpy()):
                # no transition
                self.configs["logger"].info("[INFO]: %s is invalid." % self.state[i].numpy())
                dont_vlsi[i] = 1
                reward[i] = 0
        # `valid_idx`: <torch.Tensor>
        vlsi_idx = torch.where(torch.Tensor(dont_vlsi) == 0)[0]
        # evaluate `self.state` w.r.t. VLSI
        if vlsi_idx.shape[0] != 0:
            ipc = torch.Tensor(
                test_online_vlsi(
                    self.configs,
                    self.state[vlsi_idx].numpy()
                )
            )
            k = 0
            for i in vlsi_idx:
                reward[i] = ipc[k]
                # update `self.best_ipc`
                if reward[i] > self.ipc[i]:
                    self.ipc[i] = ipc[k]
                k += 1
        reward = torch.Tensor(reward)
        msg = "[INFO]: state: %s, reward: %s, current best ipc: %s" % (self.state.numpy(), reward, self.ipc)
        self.configs["logger"].info(msg)
        if bool(torch.lt(
                self.best_ipc,
                torch.max(self.ipc)
            )
        ):
            # the best IPC is increased
            self.best_ipc = torch.max(self.ipc)
            best_idx = torch.where(self.ipc == self.best_ipc)[0]
            self.last_update = self.n_step
            msg = "[INFO]: current state: %s, global best state: %s, global best ipc: %.8f" % (
                self.state.numpy(),
                self.state.numpy()[int(best_idx)],
                self.best_ipc
            )
            self.configs["logger"].info(msg)
        done = bool(
            self.n_step > self.configs["total-step"] or \
            (self.n_step - self.last_update) >= self.configs["early-stopping"] or \
            # all ipc equals to -1
            (torch.where(reward == -1)[0].shape == reward.shape) or \
            # all ipc equals to 0
            (torch.where(reward == 0)[0].shape == reward.shape)
        )
        self.n_step += 1
        return self.state.clone(), reward.clone(), done

    def get_next_state(self, reward):
        """
            re-generate a design if it is invalid
        """
        f = False
        for i in range(reward.shape[0]):
            if reward[i] == 0:
                f = True
                self.state[i] = self.design_space.sample_v3(
                    1,
                    self.design_space.bounds["decodeWidth"][::-1][i]
                )
        if f:
            msg = "[INFO]: invalid state has been re-generated: %s" % self.state
            self.configs["logger"].info(msg)
        return self.state.clone()

    def re_init(self):
        self.configs["logger"].info("[INFO]: re-generate designs...")
        state = torch.Tensor()
        for i in range(len(self.ipc)):
            if self.ipc[i] == 0:
                state = torch.cat(
                    (
                        state,
                        self.design_space.sample_v3(1, int(self.state[i][4]))
                    )
                ).long()
        ipc = torch.Tensor(online_vlsi(self.configs, state.numpy()))

        # replace the current state
        k = 0
        for i in range(len(self.ipc)):
            if self.ipc[i] == 0:
                self.state[i] = state[k]
                self.ipc[i] = ipc[k]
                k += 1

    def test_re_init(self):
        self.configs["logger"].info("[INFO]: re-generate designs...")
        state = torch.Tensor()
        for i in range(len(self.ipc)):
            if self.ipc[i] == 0:
                state = torch.cat(
                    (
                        state,
                        self.design_space.sample_v3(1, int(self.state[i][4]))
                    )
                ).long()
        ipc = torch.Tensor(test_online_vlsi(self.configs, state.numpy()))

        # replace the current state
        k = 0
        for i in range(len(self.ipc)):
            if self.ipc[i] == 0:
                self.state[i] = state[k]
                self.ipc[i] = ipc[k]
                k += 1

    def reset(self):
        self.state = self.design_space.sample(self.configs["batch"])
        self.ipc = torch.Tensor(online_vlsi(self.configs, self.state.numpy()))
        self.best_ipc = torch.max(self.ipc)
        # If some configs. cannot simulate, we need to re-initialize it
        while not (torch.where(self.ipc == 0)[0].shape[0] == 0):
            self.re_init()
        msg = "[INFO]: reset state: %s, reset ipc: %s" % (self.state, self.ipc)
        self.configs["logger"].info(msg)
        return self.state.clone()

    def test_reset(self):
        self.state = self.design_space.sample(self.configs["batch"])
        self.ipc = torch.Tensor(test_online_vlsi(self.configs, self.state.numpy()))
        self.best_ipc = torch.max(self.ipc)
        # If some configs. cannot simulate, we need to re-initialize it
        while not (torch.where(self.ipc == 0)[0].shape[0] == 0):
            self.test_re_init()
        msg = "[INFO]: reset state: %s, reset ipc: %s" % (self.state, self.ipc)
        self.configs["logger"].info(msg)
        return self.state.clone()
