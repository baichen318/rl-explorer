# Author: baichen318@gmail.com


import os
import time
import random
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
from dse.algo.a2c.functions import make_vec_envs, array_to_tensor, tensor_to_array
from dse.algo.a2c.model import BOOMActorCriticNetwork, RocketActorCriticNetwork
from dse.algo.a2c.preference import Preference
from dse.algo.a2c.buffer import Buffer
from utils import remove_suffix, if_exist


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
        self.preference = Preference(self.configs["ppa-preference"], self.envs.reward_space)
        self.buffer = Buffer(
            self.envs.observation_space,
            self.envs.reward_space,
            self.configs["sample-size"]
        )
        self.training = self.set_mode()
        self.lr = self.configs["learning-rate"]
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )
        self.temperature = self.configs["temperature"]
        self.mse = nn.MSELoss()
        # self.set_random_state(round(time.time()))
        self.set_random_state(0)

    def set_random_state(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def set_mode(self):
        if self.configs["mode"] != "train":
            self.model.eval()
            self._model.eval()
            self.load(
                os.path.join(
                    self.configs["output-path"],
                    "models",
                    self.configs["rl-model"]
                )
            )
        return True if self.configs["mode"] == "train" else False

    def get_action(self, state, preference, status=None, episode=None):
        state = array_to_tensor(state)
        preference = array_to_tensor(preference)
        policy, value = self.model(state, preference)
        if self.training:
            policy = F.softmax(policy / self.temperature, dim=-1)
            if status is not None and episode is not None:
                status.update_action_per_episode(policy, episode)
        else:
            policy = F.softmax(policy, dim=-1)
        action = self.random_choice_prob_index(policy)
        return action

    def random_choice_prob_index(self, policy, axis=1):
        policy = tensor_to_array(policy)
        r = np.expand_dims(np.random.rand(policy.shape[1 - axis]), axis=axis)
        return (policy.cumsum(axis=axis) > r).argmax(axis=axis)

    def anneal(self):
        self.temperature = 0.01 + 0.99 * self.temperature

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
            n_step = self.configs["num-parallel"] * self.configs["num-step"]
            for worker in range(self.configs["num-parallel"]):
                discounted_reward = _calc_discounted_reward(
                    buffer["reward"][
                        worker * self.configs["num-step"] + idx * n_step : \
                            (worker + 1) * self.configs["num-step"] + idx * n_step
                    ],
                    buffer["done"][
                        worker * self.configs["num-step"] + idx * n_step : \
                            (worker + 1) * self.configs["num-step"] + idx * n_step
                    ],
                    value[
                        worker * self.configs["num-step"] + idx * n_step : \
                            (worker + 1) * self.configs["num-step"] + idx * n_step
                    ],
                    next_value[
                        worker * self.configs["num-step"] + idx * n_step : \
                            (worker + 1) * self.configs["num-step"] + idx * n_step
                    ]
                )
                total_discounted_reward.append(discounted_reward)
        return np.concatenate(total_discounted_reward).reshape(-1, self.envs.reward_space)

    def calc_advantage(self, preference, discounted_reward, value, episode):
        ofs = self.configs["num-parallel"] * self.configs["num-step"]

        def apply_envelope_operator(discounted_reward, preference):
            prod = np.inner(discounted_reward, preference)
            mask = prod.transpose().reshape(
                self.configs["sample-size"],
                -1,
                ofs
            ).argmax(axis=1)
            mask = mask.reshape(-1) * ofs + \
                np.array(
                    list(range(ofs)) * self.configs["sample-size"]
                )
            discounted_reward = discounted_reward[mask]
            return discounted_reward

        if episode > self.configs["episode-when-apply-envelope-operator"]:
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
        self.actor_loss = -optimal_action.log_prob(action) * adv_w_preference
        self.actor_loss = self.actor_loss.mean()

        # entropy loss
        self.entropy = optimal_action.entropy().mean()

        # critic loss
        critic_loss_1 = self.mse(value_w_preference, reward_w_preference)
        critic_loss_2 = self.mse(value.view(-1), reward.view(-1))

        # total loss
        self.critic_loss = 0.5 * \
            (
                self.configs["beta"] * critic_loss_1 + \
                (1 - self.configs["beta"]) * critic_loss_2
            )

        self.loss = self.actor_loss.mean() + self.critic_loss - \
            self.configs["alpha"] * self.entropy

        self.optimizer.zero_grad()
        self.loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.configs["clip-grad-norm"]
        )
        self.optimizer.step()

    def schedule_lr(self, episode):
        self.lr = self.configs["learning-rate"] - \
            (episode / (self.configs["max-sequence"] * self.configs["num-step"])) * \
            self.configs["learning-rate"]
        for params in self.optimizer.param_groups:
            params["lr"] = self.lr
        # self.configs["logger"].info("[INFO]: learning rate: {}.".format(self.lr))

    def save(self, episode):
        if episode % (
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
            self.configs["logger"].info(
                "[INFO]: save model: {} at episode: {}.".format(
                    model_path, episode
                )
            )

    def load(self, path):
        if_exist(path, strict=True)
        if self.device.type == "cpu":
            self.model.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            self.model.load_state_dict(torch.load(path))
        self._model = copy.deepcopy(self.model)
        self.configs["logger"].info(
            "load the RL model from {}".format(path)
        )


    def sync_critic(self, episode):
        if episode % self.configs["update-critic-episode"] == 0:
            self._model.load_state_dict(self.model.state_dict())
            self.configs["logger"].info(
                "[INFO]: update the critic at episode: {}.".format(
                    episode
                )
            )


class RocketAgent(object):
    def __init__(self, configs, env):
        super(RocketAgent, self).__init__()
        self.configs = configs
        self.device = torch.device(
            "cuda" if self.configs["use-cuda"] else "cpu"
        )
        self.envs = make_vec_envs(self.configs, self.device, env)
        self.model = RocketActorCriticNetwork(
            self.envs.observation_space,
            self.envs.action_space,
            self.envs.reward_space
        )
        self._model = copy.deepcopy(self.model)
        self.preference = Preference(self.configs["ppa-preference"], self.envs.reward_space)
        self.buffer = Buffer(
            self.envs.observation_space,
            self.envs.reward_space,
            self.configs["sample-size"]
        )
        self.training = self.set_mode()
        self.lr = self.configs["learning-rate"]
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )
        self.temperature = self.configs["temperature"]
        self.mse = nn.MSELoss()
        self.set_random_state(round(time.time()))

    def set_random_state(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def set_mode(self):
        if self.configs["mode"] != "train":
            self.model.eval()
            self._model.eval()
            self.load(
                os.path.join(
                    self.configs["output-path"],
                    "models",
                    self.configs["rl-model"]
                )
            )
        return True if self.configs["mode"] == "train" else False

    def get_action(self, state, preference):
        state = array_to_tensor(state)
        preference = array_to_tensor(preference)
        policy, value = self.model(state, preference)
        if self.training:
            policy = F.softmax(policy / self.temperature, dim=-1)
        else:
            policy = F.softmax(policy, dim=-1)
        action = self.random_choice_prob_index(policy)
        return action

    def random_choice_prob_index(self, policy, axis=1):
        policy = tensor_to_array(policy)
        r = np.expand_dims(np.random.rand(policy.shape[1 - axis]), axis=axis)
        return (policy.cumsum(axis=axis) > r).argmax(axis=axis)

    def anneal(self):
        self.temperature = 0.01 + 0.99 * self.temperature

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
            n_step = self.configs["num-parallel"] * self.configs["num-step"]
            for worker in range(self.configs["num-parallel"]):
                discounted_reward = _calc_discounted_reward(
                    buffer["reward"][
                        worker * self.configs["num-step"] + idx * n_step : \
                            (worker + 1) * self.configs["num-step"] + idx * n_step
                    ],
                    buffer["done"][
                        worker * self.configs["num-step"] + idx * n_step : \
                            (worker + 1) * self.configs["num-step"] + idx * n_step
                    ],
                    value[
                        worker * self.configs["num-step"] + idx * n_step : \
                            (worker + 1) * self.configs["num-step"] + idx * n_step
                    ],
                    next_value[
                        worker * self.configs["num-step"] + idx * n_step : \
                            (worker + 1) * self.configs["num-step"] + idx * n_step
                    ]
                )
                total_discounted_reward.append(discounted_reward)
        return np.concatenate(total_discounted_reward).reshape(-1, self.envs.reward_space)

    def calc_advantage(self, preference, discounted_reward, value, episode):
        n_step = self.configs["num-parallel"] * self.configs["num-step"]

        def apply_envelope_operator(discounted_reward, preference):
            prod = np.inner(discounted_reward, preference)
            mask = prod.transpose().reshape(
                self.configs["sample-size"],
                -1,
                n_step
            ).argmax(axis=1)
            mask = mask.reshape(-1) * n_step + \
                np.array(
                    list(range(n_step)) * self.configs["sample-size"]
                )
            discounted_reward = discounted_reward[mask]
            return discounted_reward

        if episode > self.configs["apply-envelope-operator-start"]:
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
        self.actor_loss = -optimal_action.log_prob(action) * adv_w_preference
        self.actor_loss = self.actor_loss.mean()

        # entropy loss
        self.entropy = optimal_action.entropy().mean()

        # critic loss
        critic_loss_1 = self.mse(value_w_preference, reward_w_preference)
        critic_loss_2 = self.mse(value.view(-1), reward.view(-1))

        # total loss
        self.critic_loss = 0.5 * \
            (
                self.configs["beta"] * critic_loss_1 + \
                (1 - self.configs["beta"]) * critic_loss_2
            )

        self.loss = self.actor_loss.mean() + self.critic_loss - \
            self.configs["alpha"] * self.entropy

        self.optimizer.zero_grad()
        self.loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.configs["clip-grad-norm"]
        )
        self.optimizer.step()

    def schedule_lr(self, episode):
        self.lr = self.configs["learning-rate"] - \
            (episode / self.configs["max-episode"]) * self.configs["learning-rate"]
        for params in self.optimizer.param_groups:
            params["lr"] = self.lr
        self.configs["logger"].info("[INFO]: learning rate: {}.".format(self.lr))

    def save(self, episode, step):
        if step % (
            self.configs["num-parallel"] * \
            self.configs["num-step"]
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
            self.configs["logger"].info(
                "[INFO]: save model: {} at episode: {}, step: {}.".format(
                    model_path, episode, step
                )
            )

    def load(self, path):
        if_exist(path, strict=True)
        if self.device.type == "cpu":
            self.model.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            self.model.load_state_dict(torch.load(path))
        self._model = copy.deepcopy(self.model)
        self.configs["logger"].info(
            "load the RL model from {}".format(path)
        )


    def sync_critic(self, episode, step):
        if step % self.configs["update-critic-step"] == 0:
            self._model.load_state_dict(self.model.state_dict())
            self.configs["logger"].info(
                "[INFO]: update the critic at episode: {}, step: {}.".format(
                    episode,
                    step
                )
            )
