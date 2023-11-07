

from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, configs):
        super(Agent, self).__init__()
        self.configs = configs

    @property
    def mode(self):
        return self.configs["algo"]["mode"]

    @property
    def logger(self):
        return self.configs["logger"]

    @property
    def max_episode(self):
        return self.configs["algo"]["max-episode"]

    @property
    def num_parallel(self):
        return self.configs["algo"]["num-parallel"]

    @property
    def sample_size(self):
        return self.configs["algo"]["train"]["sample-size"]

    @property
    def num_step(self):
        return self.configs["algo"]["train"]["num-step"]

    @property
    def gamma(self):
        return self.configs["algo"]["train"]["gamma"]

    @property
    def lam(self):
        return self.configs["algo"]["train"]["lambda"]

    @property
    def beta(self):
        return self.configs["algo"]["train"]["beta"]

    @property
    def alpha(self):
        return self.configs["algo"]["train"]["alpha"]

    @property
    def start_envelope(self):
        return self.configs["algo"]["train"]["episode-when-apply-envelope-operator"]

    @property
    def clip_grad_norm(self):
        return self.configs["algo"]["train"]["clip-grad-norm"]

    @property
    def learning_rate(self):
        return self.configs["algo"]["train"]["learning-rate"]

    @property
    def update_critic_episode(self):
        return self.configs["algo"]["train"]["update-critic-episode"]

    @abstractmethod
    def get_action(self, state, explore_w):
    	raise NotImplementedError
