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
            self.design_space.vec_to_embedding(state),
            self.idx
        )
        perf, stats = manager.evaluate_perf()
        power, area = manager.evaluate_power_and_area()
        stats_feature = []
        for k, v in stats.items():
            stats_feature.append(v)
        stats_feature = np.array(stats_feature)
        perf = self.perf_model.predict(np.expand_dims(
                np.concatenate(
                    (
                        np.array(self.design_space.vec_to_embedding(
                                list(state)
                            )
                        ),
                        stats_feature,
                        [perf]
                    )
                ),
                axis=0
            )
        )[0]
        perf = self.perf_model.predict(np.expand_dims(
                np.concatenate(
                    (
                        np.array(self.design_space.vec_to_embedding(
                                list(state)
                            )
                        ),
                        stats_feature,
                        [power]
                    )
                ),
                axis=0
            )
        )[0]
        power = self.power_model.predict(np.expand_dims(
                np.concatenate(
                    (
                        np.array(self.design_space.vec_to_embedding(
                                list(state)
                            )
                        ),
                        stats_feature,
                        [area]
                    )
                ),
                axis=0
            )
        )[0]
        # NOTICE: power and area should be negated
        return np.array([perf, -power, -area])

    def if_done(self, ppa):
        if ppa[0] > self.ppa_baseline[0] and \
            ppa[1] > self.ppa_baseline[1] and \
            ppa[2] > self.ppa_baseline[2]:
            self.last_update = self.steps
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
        def get_idx_of_human_baseline():
            if "BOOM" in self.configs["design"]:
                # some human implementations are missing, so we only explore a design
                # that already has a human implementations.
                idx = {
                    "1-wide 4-fetch SonicBOOM": 17,
                    "2-wide 4-fetch SonicBOOM": 86733931,
                    "3-wide 8-fetch SonicBOOM": 168415972,
                    "4-wide 8-fetch SonicBOOM": 202143214,
                    "5-wide SonicBOOM": 215828887
                }
            else:
                assert self.configs["design"] == "Rocket", \
                    "[ERROR]: {} is not supported.".format(self.configs["design"])
                idx = {
                    "Rocket": 692
                }
            return idx[self.configs["design"]]

        return np.array(self.design_space.idx_to_vec(
                get_idx_of_human_baseline()
            )
        )

    def get_human_baseline(self):
        if "BOOM" in self.configs["design"]:
            ppa = {
                # ipc power area
                # Small SonicBOOM
                "1-wide 4-fetch SonicBOOM":
                    [0.7760953307151794434, 0.02541292458772659302, 1.504957375000000000e+06], # [0.766128848, 0.0212, 1504764.403],
                # Medium SonicBOOM
                "2-wide 4-fetch SonicBOOM":
                    [1.121231794357299805, 0.03620982170104980469, 1.939831625000000000e+06], # [1.100314122, 0.0267, 1933210.356],
                # Large SonicBOOM
                "3-wide 8-fetch SonicBOOM":
                    [1.327529311180114746, 0.06920745223760604858, 3.213275500000000000e+06], # [1.312793895, 0.0457, 3205484.562],
                # Mega SonicBOOM
                "4-wide 8-fetch SonicBOOM":
                    [1.642608880996704102, 0.07648544013500213623, 4.829858000000000000e+06], # [1.634452069, 0.0592, 4805888.807],
                # Giga SonicBOOM
                "5-wide SonicBOOM":
                    [1.653097748756408691, 0.07422129064798355103, 5.043137500000000000e+06] # [1.644617524, 0.0715, 5069115.916]
            }
        else:
            assert self.configs["design"] == "Rocket", \
                "[ERROR]: {} is not supported.".format(self.configs["design"])
            ppa = {
                "Rocket": [0.801072362, 0.0026, 908152.038]
            }
        # negate
        baseline = ppa[self.configs["design"]]
        baseline[1] = -baseline[1]
        baseline[2] = -baseline[2]
        return np.array(baseline)

    def reset(self):
        self.steps = 0
        self.best_reward_w_preference = -float("inf")
        self.last_update = 0
        self.state = self.get_human_implementation()
        self.ppa_baseline = self.get_human_baseline()
        return self.state

    def render(self):
        return NotImplementedError

    def close(self):
        return NotImplementedError
