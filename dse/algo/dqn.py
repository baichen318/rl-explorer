# Author: baichen318@gmail.com

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import copy
import collections
import tqdm
import math
import pickle
from util import mkdir

Transition = collections.namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)

class PolicyNetwork(torch.nn.Module):
    """ PolicyNetwork """
    def __init__(self, in_dim, out_dim):
        super(PolicyNetwork, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, out_dim),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.mlp(x)

    def save(self, p, logger):
        p = os.path.join(p, "dqn.pt")
        torch.save(self.mlp.state_dict(), p)
        logger.info("[INFO]: saving to %s" % p)

    def load(self, p, logger):
        p = os.path.join(p, "dqn.pt")
        self.mlp.load_state_dict(torch.load(p))
        logger.info("[INFO]: loading from %s" % p)

class ExplorationScheduler(object):
    """ ExplorationScheduler """
    def __init__(self, start, end, decay):
        super(ExplorationScheduler, self).__init__()
        self.start = start
        self.end = end
        self.decay = decay
        self._step = 0

    def step(self):
        # NOTICE: `self._step` is suggested to set zero initially.
        epsilon = self.end + (self.start - self.end) * math.exp(-1. * self._step / self.decay)
        self._step += 1
        return epsilon
        
class ReplayBuffer(object):
    """ ReplayBuffer """
    def __init__(self, capacity):
        super(ReplayBuffer, self).__init__()
        self.capacity = capacity
        self.buffer = []

    def push(self, *args):
        self.buffer.append(Transition(*args))
        if len(self.buffer) > self.capacity:
            del self.buffer[0]

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def save(self, path, logger):
        np.save(
            os.path.join(path, "buffer.npy"),
            np.array(self.buffer, dtype=object),
        )
        logger.info("[INFO]: save the buffer to %s" % path)

    def load(self, path, logger):
        npy = np.load(os.path.join(path, "buffer.npy"), allow_pickle=True)
        for _npy in npy:
            self.push(*_npy)
        logger.info("[INFO]: load the buffer from %s" % path)

    def __len__(self):
        return len(self.buffer)

class DQN(object):
    """ DQN """
    def __init__(self, env):
        super(DQN, self).__init__()
        self.env = env
        self.replay_buffer = ReplayBuffer(
            capacity=self.env.configs["capacity"]
        )
        self.policy = PolicyNetwork(
            in_dim=self.env.design_space.n_dim,
            out_dim=sum(self.env.design_space.dims)
        )
        # self.target_policy = copy.deepcopy(self.policy)
        self.scheduler = ExplorationScheduler(
            start=self.env.configs["epsilon-start"],
            end=self.env.configs["epsilon-end"],
            decay=self.env.configs["episode"]
        )
        # self.update_target_policy = self.env.configs["update-target-policy"]
        self.batch_size = self.env.configs["batch-size"]
        self.gamma = self.env.configs["gamma"]
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.env.configs["learning-rate"]
        )
        self.episode_durations = []
        self.set_random_state()

    def set_random_state(self):
        random.seed(self.env.seed)
        torch.manual_seed(self.env.seed)
        torch.cuda.manual_seed(self.env.seed)

    def greedy_select(self, state):
        prob = random.random()
        if prob > self.scheduler.step():
            # exploitation
            with torch.no_grad():
                # NOTICE: we use `argmax`, we can also consider `max(1)[1]`
                # to find the largest index
                return self.policy(state.float()).argmax(dim=1)
        else:
            # exploration
            return torch.tensor(
                [random.randrange(len(self.env.action_list)) \
                 for i in range(self.env.configs["batch"])]
            )

    def optimize(self):
        def f(x):
            return x.unsqueeze(0)

        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = Transition(*zip(*self.replay_buffer.sample(self.batch_size)))
        batch_state = torch.cat(tuple(map(f, transitions.state)))
        batch_action = torch.cat(tuple(map(f, transitions.action)))
        batch_next_state = torch.cat(tuple(map(f, transitions.next_state)))
        batch_reward = torch.cat(tuple(map(f, transitions.reward)))
        # current Q value
        current_q = self.policy(batch_state.float()).gather(1, batch_action.unsqueeze(0))
        # expected Q value
        max_next_q = self.policy(batch_next_state.float()).detach().max(1)[0]
        print(self.batch_size, max_next_q.shape, batch_reward.shape)
        expected_q = batch_reward + (self.gamma * max_next_q)
        # Huber loss
        loss = F.smooth_l1_loss(current_q, expected_q)
        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self, episode):
        iterator = 0
        state = self.env.reset()
        while True:
            action = self.greedy_select(state)
            next_state, reward, done = self.env.step(action)
            msg = "[INFO]: state: {}, action: {}, next_state: {}, reward: {}, done: {}".format(
                state, action, next_state, reward, done
            )
            self.env.configs["logger"].info(msg)
            for i in range(self.env.configs["batch"]):
                self.replay_buffer.push(state[i], action[i], next_state[i], reward[i])
            self.save_buffer()
            self.optimize()
            state = next_state
            iterator += 1
            if done:
                msg = "[INFO]: episode: %d, step: %d" % (episode, iterator)
                self.env.configs["logger"].info(msg)
                self.episode_durations.append(iterator)
                break

    def test_run(self, episode):
        """
            debug version of `run`
        """
        iterator = 0
        state = self.env.test_reset()
        while True:
            action = self.greedy_select(state)
            next_state, reward, done = self.env.test_step(action)
            msg = "[INFO]: state: {}, action: {}, next_state: {}, reward: {}, done: {}".format(
                state, action, next_state, reward, done
            )
            for i in range(self.env.configs["batch"]):
                self.replay_buffer.push(state[i], action[i], next_state[i], reward[i])
            self.save_buffer()
            self.optimize()
            state = next_state
            iterator += 1
            if done:
                msg = "[INFO]: episode: %d, step: %d" % (episode, iterator)
                self.env.configs["logger"].info(msg)
                self.episode_durations.append(iterator)
                break

    def save_buffer(self):
        self.replay_buffer.save(
            self.env.configs["model-path"],
            self.env.configs["logger"]
        )

    def load_buffer(self, path):
        self.replay_buffer.load(
            path,
            self.env.configs["logger"]
        )

    def save_episode(self):
        self.replay_buffer.save(
            self.env.configs["model-path"],
            self.env.configs["logger"]
        )

    def load_episode(self, path):
        self.replay_buffer.load(
            path,
            self.env.configs["logger"]
        )


    def save(self):
        self.policy.save(
            self.env.configs["model-path"],
            self.env.configs["logger"]
        )

    def load(self, path):
        self.policy.load(
            path,
            self.env.configs["logger"]
        )
    # def search(self):
    #     iterator = tqdm.tqdm(range(self.env.configs["search-round"]))
    #     for i in iterator:
    #         state = self.env.reset()
    #         self.env.step(
    #             self.policy(state.float()).argmax()
    #         )
    #     msg = "[INFO]: search done."
    #     self.env.logger.info(msg)

    # def test_search(self):
    #     iterator = tqdm.tqdm(range(self.env.configs["search-round"]))
    #     for i in iterator:
    #         state = self.env.test_reset()
    #         self.env.test_step(
    #             self.policy(state.float()).argmax()
    #         )
    #     msg = "[INFO]: search done."
    #     self.env.logger.info(msg)
