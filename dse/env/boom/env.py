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
from dse.env.boom.design_space import parse_design_space
from simulation.boom.simulation import Gem5Wrapper


class BasicEnv(gym.Env):
    """ BasicEnv """
    def __init__(self, configs, idx):
        super(BasicEnv, self).__init__()
        self.configs = configs
        # NOTICE: `self.idx`, a key to distinguish different gem5 repo.
        self.idx = idx
        # NOTICE: every agent should have different initial seeds,
        # so we make a small perturbation.
        seed = round(self.idx + np.random.rand())
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
            if k == "fetchWidth" or k == "decodeWidth":
                s_idx += 1
                continue
            else:
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


class BOOMEnv(BasicEnv):
    """ BOOMEnv """
    def __init__(self, configs, idx):
        super(BOOMEnv, self).__init__(configs, idx)
        self.observation_space = self.dims_of_state
        self.action_space = self.dims_of_action
        self.reward_space = self.dims_of_reward
        self.load_ppa_model()
        self.state = np.zeros(self.dims_of_state)
        self.reset()

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
            "boom-perf.pt"
        )
        power_root = os.path.join(
            ppa_model_root,
            "boom-power.pt"
        )
        area_root = os.path.join(
            ppa_model_root,
            "boom-area.pt"
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
        action += 1
        for k, v in self.actions_map.items():
            if action in v:
                break
        # NOTICE: `+ 1` is aligned with the design space specification
        return k, v.index(action) + 1

    def scale_ppa(self, ppa):
        """
            scale PPA values can help to converge
            ppa: <list>

            Example:
                ppa = self.scale_ppa([perf, power, area])
        """
        # performance
        ppa[0] *= 2
        # power
        ppa[1] *= 20
        # area
        ppa[2] *= 0.5e-6
        return ppa

    def evaluate_microarchitecture(self, state):
        manager = Gem5Wrapper(
            self.configs,
            self.design_space,
            state,
            self.idx
        )
        perf = manager.evaluate_perf()
        power, area = manager.evaluate_power_and_area()
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
        # NOTICE: power and area should be negated
        return np.array([perf, -power, -area])

    def calc_reward(self, ppa):
        def negate_ppa():
            return np.array(
                [
                    self.ppa_baseline[0],
                    abs(self.ppa_baseline[1]),
                    abs(self.ppa_baseline[2])
                ]
            )

        return (ppa - self.ppa_baseline) / \
            (negate_ppa() + np.array([1e-8, 1e-8, 1e-8]))

    def early_stopping(self, ppa):
        preference = np.ones(self.dims_of_reward)
        preference /= np.sum(preference)
        reward_w_preference = np.dot(ppa, preference)
        if reward_w_preference > self.best_reward_w_preference:
            self.best_reward_w_preference = reward_w_preference
            self.last_update = self.steps
        return (self.steps - self.last_update) > \
            self.configs["early-stopping-per-episode"]

    def step(self, action):
        s_idx, component = self.identify_component(action)
        # modify a component for the microarchitecture, given the action
        self.state[s_idx] = component
        reward = self.calc_reward(
            self.evaluate_microarchitecture(self.state)
        )
        self.steps += 1
        done = bool(
            self.steps > self.configs["max-step-per-episode"] or \
            self.early_stopping(reward)
        )
        info = {
            "perf": reward[0],
            "power": reward[1],
            "area": reward[2]
        }
        return self.state, reward, done, info

    def reset(self):
        def get_idx_of_human_baseline(start, end):
            idx = {
                "1-wide 4-fetch SonicBOOM": 391,
                "1-wide 8-fetch SonicBOOM": random.choice(range(start, end)),
                "2-wide 4-fetch SonicBOOM": 86733931,
                "2-wide 8-fetch SonicBOOM": random.choice(range(start, end)),
                "3-wide 4-fetch SonicBOOM": random.choice(range(start, end)),
                "3-wide 8-fetch SonicBOOM": 168415972,
                "4-wide 4-fetch SonicBOOM": random.choice(range(start, end)),
                "4-wide 8-fetch SonicBOOM": 202143214,
                "5-wide SonicBOOM": 215111986
            }
            return idx[self.configs["design"]]

        def get_human_baseline():
            ppa = {
                # ipc power area
                # Small SonicBOOM
                "1-wide 4-fetch SonicBOOM": [0.766128848, 0.0212, 1504764.403],
                "1-wide 8-fetch SonicBOOM": [0.766128848, 0.0212, 1504764.403],
                # Medium SonicBOOM
                "2-wide 4-fetch SonicBOOM": [1.100314122, 0.0267, 1933210.356],
                "2-wide 8-fetch SonicBOOM": [1.100314122, 0.0267, 1933210.356],
                # Large SonicBOOM
                "3-wide 4-fetch SonicBOOM": [1.312793895, 0.0457, 3205484.562],
                "3-wide 8-fetch SonicBOOM": [1.312793895, 0.0457, 3205484.562],
                # Mega SonicBOOM
                "4-wide 4-fetch SonicBOOM": [1.634452069, 0.0592, 4805888.807],
                "4-wide 8-fetch SonicBOOM": [1.634452069, 0.0592, 4805888.807],
                # Giga SonicBOOM
                "5-wide SonicBOOM": [1.644617524, 0.0715, 5069115.916]
            }
            # negate
            baseline = ppa[self.configs["design"]]
            baseline[1] = -baseline[1]
            baseline[2] = -baseline[2]
            return np.array(baseline)

        self.steps = 0
        self.best_reward_w_preference = -float("inf")
        self.last_update = 0
        idx = self.design_space.designs.index(self.configs["design"])
        start = self.design_space.acc_design_size[idx - 1] \
            if idx > 0 \
            else 0
        end = self.design_space.acc_design_size[idx]
        # we make a reset location based on human implementation or
        # a random selection
        if np.random.random() < 0.5:
            self.state = np.array(self.design_space.idx_to_vec(
                    random.choice(range(start, end + 1))
                )
            )
            self.ppa_baseline = self.evaluate_microarchitecture(self.state)
        else:
            self.state = np.array(self.design_space.idx_to_vec(
                    get_idx_of_human_baseline(start, end + 1)
                )
            )
            self.ppa_baseline = get_human_baseline()
        return self.state

    def render(self):
        return NotImplementedError

    def close(self):
        return NotImplementedError
