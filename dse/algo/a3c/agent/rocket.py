# Author: baichen318@gmail.com


import os
import time
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dse.algo.a3c.buffer import Buffer
from dse.algo.a3c.agent.agent import Agent
from dse.algo.a3c.preference import Preference
from utils.utils import remove_suffix, if_exist
from torch.distributions.categorical import Categorical
from dse.algo.a3c.model import BOOMActorCriticNetwork, RocketActorCriticNetwork
from dse.algo.a3c.functions import make_a3c_vec_envs, array_to_tensor, tensor_to_array


class RocketAgent(Agent):
    def __init__(self, configs, env):
        super(RocketAgent, self).__init__(configs)
        self.device = torch.device(
            "cuda" if self.configs["algo"]["use-cuda"] else "cpu"
        )
        self.envs = make_a3c_vec_envs(self.configs, env)
        self.model = RocketActorCriticNetwork(
            self.envs.observation_space,
            self.envs.action_space,
            self.envs.reward_space
        )
        self._model = copy.deepcopy(self.model)
        self.training = self.set_mode()
        self.preference = Preference(
            self.configs["algo"]["test"]["ppa-preference"],
            self.envs.reward_space
        )
        self.buffer = Buffer(
            self.envs.observation_space,
            self.envs.reward_space,
            self.sample_size
        )
        self.temperature = self.configs["algo"]["train"]["temperature"]
        self.lr = self.learning_rate
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )
        self.mse = nn.MSELoss()
        self.set_random_state(configs["algo"]["random-seed"])

    def set_random_state(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def set_mode(self):
        if self.mode == "test":
            self.model.eval()
            self._model.eval()
            self.load(
                os.path.join(
                    self.configs["algo"]["test"]["rl-model"]
                )
            )
        return True if self.mode == "train" else False

    def adjust_action_space(self, policy):
        """
            We need to enforce rules to process the selected actions.
            Refer to https://stats.stackexchange.com/questions/328835/enforcing-game-rules-in-alpha-go-zero
            Our rule is to use the first N candidates as the true
            action candidates. So we normalize with the first N
            candidates.
        """
        ret = self.envs.safe_env_method(
            "get_action_candidates",
            self.envs.safe_get_attr("current_state")
        )
        num_of_candidates = len(ret)
        policy = policy[:, :num_of_candidates]
        return F.normalize(policy)

    def get_action(self, state, preference):
        state = array_to_tensor(state)
        preference = array_to_tensor(preference)
        policy, value = self.model(state, preference)
        policy = self.adjust_action_space(policy)
        if self.training:
            policy = F.softmax(policy / self.temperature, dim=-1)
            # self.logger.info(
            #     "[INFO]: action prob: {}.".format(_policy)
            # )
        else:
            policy = F.softmax(policy, dim=-1)
        _policy = policy.data.cpu().numpy()
        action = self.random_choice_prob_index(policy)
        return action, _policy

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
                [self.num_step, self.envs.reward_space]
            )
            # implementation of generalized advantage estimator (GAE)
            gae = np.zeros(self.envs.reward_space)
            for t in range(self.num_step - 1, -1, -1):
                delta = reward[t] + self.gamma * \
                    next_value[t] * (1 - done[t]) - value[t]
                gae = delta + self.gamma * self.lam * (1 - done[t]) * gae
                discounted_reward[t] = gae + value[t]
            return discounted_reward

        total_discounted_reward = []
        total_adv = []
        for idx in range(self.sample_size):
            n_step = self.num_parallel * self.num_step
            for worker in range(self.num_parallel):
                start = worker * self.num_step + idx * n_step
                end = (worker + 1) * self.num_step + idx * n_step
                discounted_reward = _calc_discounted_reward(
                    buffer["reward"][start: end],
                    buffer["done"][start: end],
                    value[start: end],
                    next_value[start: end]
                )
                total_discounted_reward.append(discounted_reward)
        return np.concatenate(total_discounted_reward).reshape(-1, self.envs.reward_space)

    def envelope_operator(self, preference, discounted_reward, value, episode):
        ofs = self.num_parallel * self.num_step

        def apply_envelope_operator(discounted_reward, preference):
            prod = np.inner(discounted_reward, preference)
            mask = prod.transpose().reshape(
                self.sample_size, -1, ofs
            ).argmax(axis=1)
            mask = mask.reshape(-1) * ofs + \
                np.array(
                    list(range(ofs)) * self.sample_size
                )
            discounted_reward = discounted_reward[mask]
            return discounted_reward

        if episode > self.start_envelope:
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
        adv_w = torch.bmm(
            adv.unsqueeze(1),
            preference.unsqueeze(2)
        ).squeeze()

        # standardization
        adv_w = (adv_w - adv_w.mean()) / \
            (adv_w.std() + 1e-30)

        policy, value = self.model(state, preference)
        optimal_action = Categorical(F.softmax(policy, dim=-1))

        value_w = torch.bmm(
            value.unsqueeze(1),
            preference.unsqueeze(2)
        ).squeeze()

        reward_w = torch.bmm(
            reward.unsqueeze(1),
            preference.unsqueeze(2)
        ).squeeze()

        # actor loss
        self.actor_loss = -optimal_action.log_prob(action) * adv_w
        self.actor_loss = self.actor_loss.mean()

        # entropy loss
        self.entropy = optimal_action.entropy().mean()

        # critic loss
        critic_loss_1 = self.mse(value_w, reward_w)
        critic_loss_2 = self.mse(value.view(-1), reward.view(-1))

        # total loss
        self.critic_loss = 0.5 * (
                self.beta * critic_loss_1 + \
                (1 - self.beta) * critic_loss_2
            )

        self.loss = self.actor_loss.mean() + self.critic_loss - \
            self.alpha * self.entropy

        self.optimizer.zero_grad()
        self.loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.clip_grad_norm
        )
        self.optimizer.step()

        self.logger.info(
            "[INFO]: actor loss: {}, critic loss: {}, " \
                "entropy loss: {}.".format(
                    self.actor_loss.detach().numpy(),
                    self.critic_loss.detach().numpy(),
                    self.entropy.detach().numpy()
            )
        )

    def schedule_lr(self, episode):
        self.lr = self.learning_rate - \
            (episode / self.max_episode) * self.learning_rate
        for params in self.optimizer.param_groups:
            params["lr"] = self.lr
        self.logger.info("[INFO]: learning rate: {}.".format(
            self.lr)
        )

    def save(self, episode):
        if episode % 100 == 0:
            model_path = os.path.join(
                self.configs["model-path"],
                "{}.pt".format(
                    remove_suffix(
                        os.path.basename(
                            self.configs["log-path"]
                        ), ".log"
                    )
                )
            )
            torch.save(
                self.model.state_dict(),
                model_path
            )
            self.logger.info(
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
        self.logger.info(
            "[INFO]: load the RL model from {}".format(path)
        )

    def sync_critic(self, episode):
        if episode % self.update_critic_episode == 0:
            self._model.load_state_dict(self.model.state_dict())
            self.logger.info(
                "[INFO]: update the critic at episode: {}.".format(
                    episode
                )
            )
