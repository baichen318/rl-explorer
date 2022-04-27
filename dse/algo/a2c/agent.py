# Author: baichen318@gmail.com


import os
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
from dse.algo.a2c.functions import make_vec_envs, array_to_tensor, tensor_to_array
from dse.algo.a2c.model import BOOMActorCriticNetwork
from dse.algo.a2c.preference import Preference
from dse.algo.a2c.buffer import Buffer
from utils import remove_suffix


class BOOMAgent(object):
    def __init__(self, configs, env):
        super(BOOMAgent, self).__init__()
        self.configs = configs
        self.device = torch.device(
            "cuda" if self.configs["use-cuda"] else "cpu"
        )
        self.envs = make_vec_envs(self.configs, self.device, env)
        self.model = BOOMActorCriticNetwork(
            self.envs.observation_space,
            self.envs.action_space,
            self.envs.reward_space
        )
        self._model = copy.deepcopy(self.model)
        self.preference = Preference(self.envs.reward_space)
        self.buffer = Buffer(
            self.envs.observation_space,
            self.envs.reward_space,
            self.configs["sample-size"]
        )
        self.training = self.set_mode()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.configs["learning-rate"]
        )
        self.mse = nn.MSELoss()

    def set_mode(self):
        if self.configs["mode"] != "train":
            self.model.eval()
            self._model.eval()
        return True if self.configs["mode"] == "train" else False

    def get_action(self, state, preference):
        state = array_to_tensor(state)
        preference = array_to_tensor(preference)
        policy, value = self.model(state, preference)
        if self.training:
            # TODO: this may cause unexpected behavior!
            policy = F.softmax(policy / self.configs["temperature"], dim=-1)
        else:
            policy = F.softmax(policy, dim=-1)
        # TODO: this may mis-align with the original design
        action = self.random_choice_prob_index(policy)
        return action

    def random_choice_prob_index(self, policy, axis=1):
        policy = tensor_to_array(policy)
        r = np.expand_dims(np.random.rand(policy.shape[1 - axis]), axis=axis)
        # TODO: np.cumsum: axis = ?
        return (policy.cumsum(axis=0) > r).argmax(axis=axis)

    def forward_transition(self, preference):
        buffer = self.buffer.batch_pool
        state = buffer["state"]
        next_state = buffer["next-state"]
        state = array_to_tensor(state)
        next_state = array_to_tensor(next_state)
        preference = array_to_tensor(preference)
        _, value = self._model(state, preference)
        policy, _ = self.model(state, preference)
        _, next_value = self._model(next_state, preference)
        value = tensor_to_array(value).squeeze()
        next_value = tensor_to_array(next_value).squeeze()
        return value, next_value, policy

    def calc_discounted_reward(self, value, next_value):
        buffer = self.buffer.batch_pool

        def _calc_discounted_reward(reward, done, value, next_value):
            discounted_reward = np.empty(
                [self.configs["num-step"], self.envs.reward_space]
            )
            # implementation of generalized advantage estimator (GAE)
            gae = np.zeros(self.envs.reward_space)
            for t in range(self.configs["num-step"] - 1, -1, -1):
                delta = reward[t] + self.configs["gamma"] * \
                    next_value[t] * (1 - done[t]) - value[t]
                gae = delta + self.configs["gamma"] * \
                    self.configs["lambda"] * (1 - done[t]) * gae
                discounted_reward[t] = gae + value[t]
            return discounted_reward

        total_discounted_reward = []
        total_adv = []
        for idx in range(self.configs["sample-size"]):
            batch_size = self.configs["num-parallel"] * self.configs["num-step"]
            for worker in range(self.configs["num-parallel"]):
                discounted_reward = _calc_discounted_reward(
                    buffer["reward"][
                        worker * self.configs["num-step"] + idx * batch_size : \
                            (worker + 1) * self.configs["num-step"] + idx * batch_size
                    ],
                    buffer["done"][
                        worker * self.configs["num-step"] + idx * batch_size : \
                            (worker + 1) * self.configs["num-step"] + idx * batch_size
                    ],
                    value[
                        worker * self.configs["num-step"] + idx * batch_size : \
                            (worker + 1) * self.configs["num-step"] + idx * batch_size
                    ],
                    next_value[
                        worker * self.configs["num-step"] + idx * batch_size : \
                            (worker + 1) * self.configs["num-step"] + idx * batch_size
                    ]
                )
                total_discounted_reward.append(discounted_reward)
        return np.concatenate(total_discounted_reward).reshape(-1, self.envs.reward_space)

    def calc_advantage(self, preference, discounted_reward, value, step):
        batch_size = self.configs["num-parallel"] * self.configs["num-step"]

        def apply_envelope_operator(discounted_reward, preference):
            prod = np.inner(discounted_reward, preference)
            mask = prod.transpose().reshape(
                self.configs["sample-size"],
                -1,
                batch_size
            ).argmax(axis=1)
            mask = mask.reshape(-1) * batch_size + \
                np.array(
                    list(range(batch_size)) * self.configs["sample-size"]
                )
            discounted_reward = discounted_reward[mask]
            return discounted_reward

        if step > self.configs["apply-envelope-operator-start"]:
            discounted_reward = apply_envelope_operator(
                discounted_reward,
                preference
            )
        adv = discounted_reward - value
        return adv, discounted_reward

    def optimize_actor_critic(self, preference, reward, adv):
        buffer = self.buffer.batch_pool
        with torch.no_grad():
            state = array_to_tensor(buffer["state"])
            action = array_to_tensor(buffer["action"], fmt=int)
            next_state = array_to_tensor(buffer["next-state"])
            reward = array_to_tensor(reward)
            preference = array_to_tensor(preference)
            adv = array_to_tensor(adv)

        # calculate scalarized advantage
        adv_w_preference = torch.bmm(
            adv.unsqueeze(1),
            preference.unsqueeze(2)
        ).squeeze()

        # standardization
        adv_w_preference = (adv_w_preference - adv_w_preference.mean()) / \
            (adv_w_preference.std() + 1e-30)

        policy, value = self.model(state, preference)
        optimal_action = Categorical(F.softmax(policy, dim=-1))

        value_w_preference = torch.bmm(
            value.unsqueeze(1),
            preference.unsqueeze(2)
        ).squeeze()

        reward_w_preference = torch.bmm(
            reward.unsqueeze(1),
            preference.unsqueeze(2)
        ).squeeze()

        # actor loss
        actor_loss = -optimal_action.log_prob(action) * adv_w_preference

        # entropy loss
        entropy = optimal_action.entropy()

        # critic loss
        critic_loss_1 = self.mse(value_w_preference, reward_w_preference)
        critic_loss_2 = self.mse(value.view(-1), reward.view(-1))

        # total loss
        loss = actor_loss.mean() + 0.5 * \
            (
                self.configs["beta"] * critic_loss_1 + \
                (1 - self.configs["beta"]) * critic_loss_2
            ) - \
            self.configs["alpha"] * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.configs["clip-grad-norm"]
        )
        self.optimizer.step()

    def schedule_lr(self, step):
        lr = self.configs["learning-rate"] - \
            (step / self.configs["max-episode"]) * self.configs["learning-rate"]
        for params in self.optimizer.param_groups:
            params["lr"] = lr
        self.configs["logger"].info("[INFO]: learning rate: {}.".format(lr))

    def save(self, step):
        if step % (
            self.configs["num-parallel"] * \
            self.configs["num-step"] * 100
        ) == 0:
            model_path = os.path.join(
                self.configs["model-path"],
                "{}.pt".format(
                    remove_suffix(
                        os.path.basename(self.configs["log-path"]),
                        ".log"
                    )
                )
            )
            torch.save(
                self.model.state_dict(),
                model_path
            )
            self.configs["logger"].info("[INFO]: save model: {}.".format(model_path))

    def sync_critic(self, step):
        if step % self.configs["update-critic-step"] == 0:
            self._model.load_state_dict(self.model.state_dict())
            self.configs["logger"].info("[INFO]: update the critic at {}.".format(step))
