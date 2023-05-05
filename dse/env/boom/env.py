# Author: baichen318@gmail.com

import os
import sys
import gym
import time
import random
import torch
import numpy as np
from collections import OrderedDict
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from simulation.boom.simulation import Gem5Wrapper
from dse.env.boom.design_space import parse_design_space
from utils.exceptions import EvaluationRuntimeError, UnSupportedException


class BasicEnv(gym.Env):
    """
        We apply the scaling graph:
        ISU -> IFU -> maxBrCount -> ROB -> PRF -> LSU -> D$
        Refer to: 2009TOCS: A mechanistic performance model for superscalar out-of-order processors
    """
    def __init__(self, configs, idx):
        super(BasicEnv, self).__init__()
        self.configs = configs
        # NOTICE: `self.idx`, a key to distinguish different gem5 repo.
        self.idx = idx
        self.component_do_not_touch = [
            "fetchWidth",
            "decodeWidth"
        ]
        # in-order w.r.t. the scaling graph
        self.component_touch = [
            "ISU",
            "IFU",
            "maxBrCount",
            "ROB",
            "PRF",
            "LSU",
            "branchPredictor", 
            "I-Cache",
            "D-Cache"
        ]
        self.design_space = parse_design_space(self.configs)
        assert len(self.component_do_not_touch + self.component_touch) == \
            len(self.design_space.descriptions[self.design].keys())
        self.dims_of_state = self.generate_dims_of_state()
        self.actions = self.generate_actions()
        self.dims_of_action = self.get_dims_of_action()
        # PPA metrics
        self.dims_of_reward = 3
        self.dims_of_tunable_state = len(self.component_touch)

    @property
    def design(self):
        return self.configs["algo"]["design"]

    @property
    def ppa_model_root(self):
        return self.configs["env"]["calib"]["ppa-model"]

    def get_action_candidates(self, state_idx):
        return self.actions[state_idx]

    def get_dims_of_action(self):
        """
            Since each state faces different action candidates,
            we need to adjust the action space give a state.
            So the action space should be the maximal dimension
            among all action candidates.
        """
        return max([len(v) \
            for k, v in self.actions.items()]
        )

    def generate_actions(self):
        """
            A mapping from an episode state to an action candidates.
            E.g.,
                actions = {
                    # ISU
                    0: [1, 21, 36, 41],
                    # IFU
                    1: [1, 2, 9, 10]
                    ...
                }
        """
        actions = OrderedDict()
        for idx in range(len(self.component_touch)):
            actions[idx] = \
                self.design_space.descriptions[self.design] \
                    [self.component_touch[idx]]
        return actions

    def generate_dims_of_state(self):
        """
            NOTICE: "branchPredictor", "fetchWidth", & "decodeWidth"
            are fixed in any state.
        """
        return len(self.design_space.descriptions[self.design].keys())


class BOOMEnv(BasicEnv):
    """
        BOOM environment
    """
    def __init__(self, configs, idx):
        super(BOOMEnv, self).__init__(configs, idx)
        self.observation_space = self.dims_of_state
        self.action_space = self.dims_of_action
        self.reward_space = self.dims_of_reward
        self.load_ppa_model()
        self.state = None
        self.state_idx = None

    @property
    def if_terminate(self):
        return self.state_idx == \
            self.dims_of_tunable_state - 1

    @property
    def current_state(self):
        return self.state_idx

    def load_ppa_model(self):
        perf_root = os.path.join(
            self.ppa_model_root,
            "boom-perf.pt"
        )
        power_root = os.path.join(
            self.ppa_model_root,
            "boom-power.pt"
        )
        area_root = os.path.join(
            self.ppa_model_root,
            "boom-area.pt"

        )
        self.perf_model = joblib.load(perf_root)
        self.power_model = joblib.load(power_root)
        self.area_model = joblib.load(area_root)

    def evaluate_microarchitecture(self, state):
        manager = Gem5Wrapper(
            self.configs,
            self.design_space,
            self.design_space.vec_to_embedding(state.tolist()),
            self.idx
        )
        perf, stats = manager.evaluate_perf()
        power, area = manager.evaluate_power_and_area()
        area *= 1e6
        stats_feature = []
        for k, v in stats.items():
            stats_feature.append(v)
        stats_feature = np.array(stats_feature)
        perf = self.perf_model.predict(np.expand_dims(
                np.concatenate((
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
        power = self.power_model.predict(np.expand_dims(
                np.concatenate((
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
        area = self.area_model.predict(np.expand_dims(
                np.concatenate((
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
        # NOTICE: it is important to scale area
        # compared with performance and power
        area *= 1e-6
        # NOTICE: power and area should be negated
        return np.array([perf, -power, -area])

    def increase_state_idx(self):
        self.state_idx += 1

    def reset_state_idx(self):
        self.state_idx = 0

    def state_idx_to_state(self):
        return [
            7, 2, 3, 5, 6, 8, 0, 9, 10
        ][self.state_idx]

    def step(self, action):
        """
            Take the action.
        """
        try:
            self.state[self.state_idx_to_state()] = \
                self.get_action_candidates(self.state_idx)[action]
        except IndexError as e:
            raise EvaluationRuntimeError(
                "index out of range: {} vs {}. " \
                "current state: {}.".format(
                        action,
                        len(self.get_action_candidates(self.state_idx)),
                        self.current_state
                    )
            )

        info = {}
        if self.if_terminate:
            reward = self.evaluate_microarchitecture(self.state)
            info = {
                "perf-pred": reward[0],
                "power-pred": reward[1],
                "area-pred": reward[2]
            }
            done = True
        else:
            reward = np.array([0, 0, 0])
            done = False

        # change to the next state
        self.increase_state_idx()

        return self.state, reward, done, info

    def get_human_implementation(self):
        def get_idx_of_human_baseline():
            if "BOOM" in self.configs["design"]:
                # some human implementations are missing, so we only explore a design
                # that already has a human implementations.
                idx = {
                    "1-wide 4-fetch SonicBOOM": 17,
                    "2-wide 4-fetch SonicBOOM": 39391288,
                    "3-wide 8-fetch SonicBOOM": 82366510,
                    "4-wide 8-fetch SonicBOOM": 100910681,
                    "5-wide SonicBOOM": 100912554
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
                # area should be x 1e-6
                # Small SonicBOOM
                "1-wide 4-fetch SonicBOOM":
                    [7.919527292251586914e-01, 4.375652596354484558e-02, 1.504552125], # [0.766128848, 0.0212, 1504764.403],
                # Medium SonicBOOM
                "2-wide 4-fetch SonicBOOM":
                    [1.162309169769287109, 5.288236215710639954e-02, 1.92402725], # [1.100314122, 0.0267, 1933210.356],
                # Large SonicBOOM
                "3-wide 8-fetch SonicBOOM":
                    [1.385208010673522949, 9.215448051691055298e-02, 3.219418], # [1.312793895, 0.0457, 3205484.562],
                # Mega SonicBOOM
                "4-wide 8-fetch SonicBOOM":
                    [1.699511766433715820, 1.193742826581001282e-01, 4.787427], # [1.634452069, 0.0592, 4805888.807],
                # Giga SonicBOOM
                "5-wide SonicBOOM":
                    [1.758375167846679688, 1.426837295293807983e-01, 5.025076] # [1.644617524, 0.0715, 5069115.916]
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

    def generate_init_state(self):
        self.state = np.zeros(self.observation_space).astype(int)
        if self.design == "small-SonicBOOM":
            self.state[1] = 1
            self.state[4] = 1
        elif self.design == "medium-SonicBOOM":
            self.state[1] = 1
            self.state[4] = 2
        elif self.design == "large-SonicBOOM":
            self.state[1] = 2
            self.state[4] = 3
        elif self.design == "mega-SonicBOOM":
            self.state[1] = 2
            self.state[4] = 4
        elif self.design == "giga-SonicBOOM":
            self.state[1] = 2
            self.state[4] = 5
        else:
            raise UnSupportedException(
                "design {} is not supported.".format(self.design)
            )

    def reset(self):
        self.reset_state_idx()
        self.generate_init_state()
        return self.state

    def render(self):
        return NotImplementedError

    def close(self):
        return NotImplementedError
