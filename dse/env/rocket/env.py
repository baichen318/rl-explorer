# Author: baichen318@gmail.com


import os
import torch
import random
import time
import gym
import numpy as np
from gym import spaces
from collections import OrderedDict
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from dse.env.rocket.design_space import parse_design_space
from simulation.rocket.simulation import Gem5Wrapper


class BasicEnv(gym.Env):
    """ BasicEnv """
    def __init__(self, configs, idx):
        super(BasicEnv, self).__init__()
        self.configs = configs
        # NOTICE: `self.idx`, a key to distinguish different gem5 repo.
        self.idx = idx
        self.design_space = parse_design_space(self.configs)
        self.dims_of_state = self.generate_dims_of_state(self.configs["design"])
        self.actions_map, self.candidate_actions = self.generate_actions_lut(
            self.configs["design"]
        )
        self.dims_of_action = len(self.candidate_actions)
        # PPA metrics
        self.dims_of_reward = 3

    def generate_actions_lut(self, design):
        """
            Example:
                actions_map = {
                    # states index  actions index
                    0: [1, 2, 3],
                    2: [4, 5, 6, 7, 8, 9, 10],
                    ...
                }
                candidate_actions = [
                    # branchPredictor x 3
                    1, 2, 3
                    # IFU x 7
                    4, 5, 6, 7, 8, 9, 10,
                    ...
                ]
        """
        actions_map, candidate_actions = OrderedDict(), []
        s_idx, a_idx = 0, 1
        for k, v in self.design_space.descriptions[design].items():
            for _v in v:
                candidate_actions.append(a_idx)
                if s_idx in actions_map.keys():
                    actions_map[s_idx].append(a_idx)
                else:
                    actions_map[s_idx] = [a_idx]
                a_idx += 1
            s_idx += 1
        return actions_map, np.array(candidate_actions)

    def generate_dims_of_state(self, design):
        return len(self.design_space.descriptions[design].keys())


class RocketEnv(BasicEnv):
    """ RocketEnv """
    def __init__(self, configs, idx):
        super(RocketEnv, self).__init__(configs, idx)
        self.observation_space = self.dims_of_state
        self.action_space = self.dims_of_action
        self.reward_space = self.dims_of_reward
        self.load_ppa_model()
        self.state = np.zeros(self.dims_of_state)

    def load_ppa_model(self):
        ppa_model_root = os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            os.path.pardir,
            os.path.pardir,
            self.configs["ppa-model"],
        )
        perf_root = os.path.join(
            ppa_model_root,
            "rocket-perf.pt"
        )
        power_root = os.path.join(
            ppa_model_root,
            "rocket-power.pt"
        )
        area_root = os.path.join(
            ppa_model_root,
            "rocket-area.pt"
        )
        self.perf_model = joblib.load(perf_root)
        self.power_model = joblib.load(power_root)
        self.area_model = joblib.load(area_root)

    def identify_component(self, action):
        """
            action: <numpy.ndarray>
        """
        # TODO: validate its correctness
        # the 1st action is encoded as 1
        _action = action + 1
        for k, v in self.actions_map.items():
            if _action in v:
                break
        # NOTICE: `+ 1` is aligned with the design space specification
        return k, v.index(_action)

    def evaluate_microarchitecture(self, state):
        manager = Gem5Wrapper(
            self.configs,
            self.design_space,
            state,
            self.idx
        )
        perf = manager.evaluate_perf()
        power, area = manager.evaluate_power_and_area()
        area *= 1e6
        perf = self.perf_model.predict(np.expand_dims(
                np.concatenate((state, [perf])),
                axis=0
            )
        )[0]
        power = self.power_model.predict(np.expand_dims(
                np.concatenate((state, [power])),
                axis=0
            )
        )[0]
        area = self.area_model.predict(np.expand_dims(
                np.concatenate((state, [area])),
                axis=0
            )
        )[0]
        # NOTICE: it is important to scale area
        # compared with performance and power
        area *= 1e-6
        # NOTICE: power and area should be negated
        return np.array(self.scale_ppa([perf, -power, -area]))

    def if_done(self, ppa):
        if ppa[0] > self.best_ppa[0] and \
            ppa[1] > self.best_ppa[1] and \
            ppa[2] > self.best_ppa[2]:
            self.last_update = self.steps
            self.best_ppa = ppa.copy()
        return (self.steps - self.last_update) > \
            self.configs["terminate-step"]

    def step(self, action):
        s_idx, a_offset = self.identify_component(action)
        # modify a component for the microarchitecture, given the action
        self.state[s_idx] = self.design_space.descriptions[
            self.configs["design"]
        ][self.design_space.components[s_idx]][a_offset]

        proxy_ppa = self.evaluate_microarchitecture(self.state)
        reward = proxy_ppa
        done = self.if_done(proxy_ppa)
        self.steps += 1
        info = {
            "perf-pred": proxy_ppa[0],
            "power-pred": proxy_ppa[1],
            "area-pred": proxy_ppa[2],
            "perf-baseline": self.ppa_baseline[0],
            "power-baseline": self.ppa_baseline[1],
            "area-baseline": self.ppa_baseline[2],
            "last-update": self.last_update
        }
        return self.state, reward, done, info

    def get_human_implementation(self):
        return np.array(self.design_space.idx_to_vec(
                692
            )
        )

    def get_human_baseline(self):
        ppa = {
            "Rocket": [0.822898, 0.007800, 682508.000000] # [0.801072362, 0.0026, 908152.038]
        }
        # negate
        baseline = ppa[self.configs["design"]]
        baseline[1] = -baseline[1]
        baseline[2] = -baseline[2]
        return np.array(baseline)

    def reset(self):
        self.steps = 0
        # self.best_reward_w_preference = -float("inf")
        self.last_update = 0
        self.state = self.get_human_implementation()
        self.ppa_baseline = self.get_human_baseline()
        self.best_ppa = self.ppa_baseline.copy()
        return self.state

    def render(self):
        return NotImplementedError

    def close(self):
        return NotImplementedError
