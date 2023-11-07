

import torch
import numpy as np


class Buffer(object):
    def __init__(self, observation_space, reward_space, sample_size):
        super(Buffer, self).__init__()
        self.total_state = []
        self.total_action = []
        self.total_next_state = []
        self.total_reward = []
        self.total_done = []
        self.observation_space = observation_space
        self.reward_space = reward_space
        self.sample_size = sample_size

    def reset(self):
        self.total_state = []
        self.total_action = []
        self.total_next_state = []
        self.total_reward = []
        self.total_done = []

    def insert(self, state, action, next_state, reward, done):
        """
            Each `state` is with a shape: step x parallel x dims_of_state
        """
        self.total_state.append(state)
        self.total_action.append(action)
        self.total_next_state.append(next_state)
        self.total_reward.append(reward)
        self.total_done.append(done)

    def generate_batch_with_n_step(self):
        def _generate_batch_with_n_step():
            """
                Each `total_state` is with a shape: (parallel x step x sample) x dims_of_state
                E.g.:
                    total_state:
                        | worker 1's state @step 1 |
                        | ------------------------ |
                        | worker 1's state @step 2 |
                        | ------------------------ |
                        | worker 1's state @step 3 |
                        |           ...            |
            """
            total_state = np.stack(self.total_state).transpose(
                [1, 0, 2]
            ).reshape(-1, self.observation_space)
            total_state = np.tile(total_state, (self.sample_size, 1))
            total_action = np.stack(self.total_action).transpose().reshape([-1])
            total_action = np.tile(total_action, self.sample_size)
            total_next_state = np.stack(self.total_next_state).transpose(
                [1, 0, 2]
            ).reshape(-1, self.observation_space)
            total_next_state = np.tile(total_next_state, (self.sample_size, 1))
            total_reward = np.stack(self.total_reward).transpose(
                [1, 0, 2]
            ).reshape([-1, self.reward_space])
            total_reward = np.tile(total_reward, (self.sample_size, 1))
            total_done = np.stack(self.total_done).transpose().reshape([-1])
            total_done = np.tile(total_done, self.sample_size)
            return {
                "state": total_state,
                "action": total_action,
                "next-state": total_next_state,
                "reward": total_reward,
                "done": total_done
            }
        self.batch_pool = _generate_batch_with_n_step()
