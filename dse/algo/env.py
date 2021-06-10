# Author: baichen318@gmail.com
import random
import torch
# Author: baichen318@gmail.com

import random

class EnvMetric(object):
    """ EnvMetric """
    # reward: `(current_pp - best_pp) / baseline`
    metrics = [
        "step",
        "current_pp",
        "best_pp",
        "baseline",
        "last_update",
        "reward"
    ]

    def __init__(self):
        super(EnvMetric, self).__init__()
        self.step = 0
        self.current_pp = 0
        self.best_pp = 0
        self.baseline = 0
        self.last_update = 0
        self.reward = 0
        self.microarch = None

    def reset(self):
        for metric in EnvMetric.metrics:
            getattr(self, metric) = 0

    def save_env(self, **kwargs):
        for k, v in kwargs.items():
            getattr(self, k) = v
        if "microarch" in kwargs.keys():
            self.set_micorarch(kwargs["microarch"])

    def set_micorarch(self, microarch):
        self.microarch = microarch
        
class MicroArchEnv(object):
    """docstring for MicroArchEnv"""
    def __init__(self, problem, seed=2021):
        super(MicroArchEnv, self).__init__()
        random.seed(seed)
        self.problem = problem
        self.metric = EnvMetric()

    def reset(self):
        # Choose the starting point randomly
        init_idx = random.choice(range(self.problem.n_train))
        pp, cc, power = self.evaluate(self.problem.train_x[init_idx])
        self.metric.reset()
        self.metric.save_env(
            current_pp=pp,
            best_pp=pp,
            baseline=pp
            microarch=self.problem.train_x[init_idx]
        )
        return self.perturb()

    def perturb(self):
        # NOTICE: we support two perturbations:
        # 1. change one candidate value softly
        # 2. choose one sample within a defined radius
        def _perturb(microarch):
            ptype = random.randint(1, 2)
            if ptype == 1:
                return self.problem.space.random_walk_by_soft_mutation(microarch, self.problem.train_x)
            else:
                assert ptype == 2
                return self.problem.space.random_walk_by_neighborhood(microarch, self.problem.train_x)

        def visited(vec, matrix):
            for mat in matrix:
                if vec.equal(mat):
                    return True
            return False

        state = torch.zeros(
            [self.problem.configs["sample-perturbation"], self.problem.space.n_dim]
        )

        for i in range(self.problem.configs["sample-perturbation"]):
            new_microarch = _perturb(self.metric.microarch.copy())
            while visited(new_microarch, state):
                new_microarch = _perturb(self.metric.microarch.copy())
            state[i] = new_microarch

        self.state = state

        return state

    def step(self):
        pass

    def evaluate(self, x: torch.Tensor):
        cc, power = self.problem.get_train_label(x)
        # NOTICE: use the area to denote the pp, refer to func: `load_dataset`
        pp = cc * power
        return pp, cc, power
        