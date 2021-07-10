import random
import torch
import copy
import collections
import tqdm
import math

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

    def save(self, p):
        print("[INFO]: saving to %s" % p)
        torch.save(self.mlp.state_dict(), p)

    def load(self, p):
        print("[INFO]: loading from %s" % p)
        if isinstance(p, 'str'):
            self.mlp = torch.load(p)
        else:
            assert isinstance(p, 'dict')
            self.mlp.load_state_dict(p)

class ExplorationScheduler(object):
    """ ExplorationScheduler """
    def __init__(self, start, end, decay):
        super(ExplorationScheduler, self).__init__()
        self.start = start
        self.end = end
        self.decay = decay
        self._step = 0

    def step(self):
        # TODO: is this the best scheduler?
        # NOTICE: `self._step` is suggested to set zero initially.
        epsilon = self.start + min(self._step / self.decay, 1.0) * (end - start)
        self._step += 1
        return epsilon
        
class ReplayBuffer(object):
    """ ReplayBuffer """
    def __init__(self, capacity):
        super(ReplayBuffer, self).__init__()
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

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
        self.target_policy = copy.deepcopy(self.policy)
        self.scheduler = ExplorationScheduler(
            start=self.env.configs["epsilon-start"],
            end=self.env.configs["epsilon-end"],
            decay=self.env.configs["epsilon-decay"]
        )
        self.update_target_policy = self.env.configs["update-target-policy"]
        self.batch_size = self.env.configs["batch_size"]
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
                return self.policy(state).argmax()
        else:
            # exploration
            return torch.tensor(
                [random.randrange(len(self.env._action_list))]
            )

    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        transitions = Transition(*zip(*transitions))
        non_terminate_mask = torch.tensor(
            tuple(map(lambda s: s is not None, transitions.next_state)),
            dtype=torch.bool
        )
        non_terminate_states = torch.cat(
            [s for s in transitions.next_state if s is not None]
        )
        state_batch = torch.cat(transitions.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        states = torch.zeros((self.batch_size, ))
        for i in range(self.batch_size):
            states[i] = state_batch[i][action_batch[i]]
        qval = self.policy(states)
        # next_state
        pass

    def train(self):
        iterator = tqdm.tqdm(range(self.env.problem.configs["total-epoch"]))
        state = self.env.reset()
        for i in iterator:
            action = self.greedy_select(state)
            next_state, reward, done = self.env.step(action)
            self.env.metric.set_rewards(reward)

            # TODO: how to handle `done`?
            if done:
                reward = -1

            self.replay_buffer.push(state, action, next_state, reward)
            self.optimize()

            state = next_state

            if (i + 1) % self.update_target_policy == 0:
                self.target_policy.load(self.policy.state_dict())
                self.policy.save()
