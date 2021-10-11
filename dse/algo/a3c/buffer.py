# Author: baichen318@gmail.com

import torch

class Buffer(object):
    def __init__(
        self,
        configs,
        obs_shape,
        action_space
    ):
        self.obs = torch.zeros(configs["n-step-td"] + 1, configs["num-process"], *obs_shape)
        self.rewards = torch.zeros(configs["n-step-td"], configs["num-process"], len(configs["metrics"]))
        self.value_preds = torch.zeros(configs["n-step-td"] + 1, configs["num-process"], action_space.n, len(configs["metrics"]))
        # TODO: action_size
        self.returns = torch.zeros(configs["n-step-td"] + 1, configs["num-process"], action_space.n, len(configs["metrics"]))
        self.action_log_probs = torch.zeros(configs["n-step-td"], configs["num-process"], 1)
        self.actions = torch.zeros(configs["n-step-td"], configs["num-process"], 1).long()
        self.masks = torch.ones(configs["n-step-td"] + 1, configs["num-process"], 1)
        self.num_agents = configs["num-process"]
        self.num_steps = configs["n-step-td"]
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        obs,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
    ):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(
        self,
        next_value,
        gamma
    ):
        self.returns[-1].copy_(next_value)
        for step in reversed(range(self.rewards.size(0))):
            for agent in range(self.num_agents):
                self.returns[step].copy_(
                    self.returns[step + 1][agent] * gamma * self.masks[step + 1][agent] + self.rewards[step][agent]
                )

