# Author: baichen318@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class NNBase(nn.Module):
    def __init__(self, hidden_size):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size[-1]


class MLPBase(NNBase):
    def __init__(self, num_inputs, hidden_size):
        super(MLPBase, self).__init__(hidden_size)
        _init = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )
        # Actor
        self.actor = nn.Sequential(
            _init(nn.Linear(num_inputs, hidden_size[0])), nn.Tanh(),
            _init(nn.Linear(hidden_size[0], hidden_size[1])), nn.Tanh(),
            _init(nn.Linear(hidden_size[1], hidden_size[2])), nn.Tanh()
        )
        # Critic
        self.critic = nn.Sequential(
            _init(nn.Linear(num_inputs, hidden_size[0])), nn.Tanh(),
            _init(nn.Linear(hidden_size[0], hidden_size[1])), nn.Tanh(),
            _init(nn.Linear(hidden_size[1], hidden_size[2])), nn.Tanh()
        )
        self.critic_linear = _init(nn.Linear(hidden_size[-1], 1))
        self.train()

    def forward(self, x, masks):
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor


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


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, hidden_size):
        super(Policy, self).__init__()
        self.base = MLPBase(obs_shape[0], hidden_size)
        self.dist = Categorical(self.base.output_size, action_space.n)

    def forward(self, inputs, masks):
        raise NotImplementedError

    def act(self, inputs, masks):
        value, actor_features = self.base(inputs, masks)
        dist = self.dist(actor_features)
        action = dist.sample()
        action_log_prob = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_prob

    def get_value(self, inputs, masks):
        value, _ = self.base(inputs, masks)

        return value

    def evaluate_actions(self, inputs, masks, action):
        value, actor_features = self.base(inputs, masks)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
