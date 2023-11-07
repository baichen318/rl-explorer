

import os
import sys
import gym
import time
import torch
import random
import numpy as np
from collections import OrderedDict
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
from simulation.rocket.simulation import Gem5Wrapper
from dse.env.rocket.design_space import parse_design_space
from utils.exceptions import EvaluationRuntimeError, UnSupportedException


class BasicEnv(gym.Env):
    """
        We apply the scaling graph:
        BTB -> FPU -> mulDiv -> useVM -> I$ -> D$
        Refer to: 2009TOCS: A mechanistic performance model for superscalar out-of-order processors
    """
    def __init__(self, configs, idx):
        super(BasicEnv, self).__init__()
        self.configs = configs
        # NOTICE: `self.idx`, a key to distinguish different gem5 repo.
        self.idx = idx
        # in-order w.r.t. the scaling graph
        self.component_touch = [
            "BTB",
            "FPU",
            "mulDiv",
            "useVM",
            "R. I-Cache",
            "R. D-Cache"
        ]
        self.design_space = parse_design_space(self.configs)
        assert len(self.component_touch) == \
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
            E.g., for "small-SonicBOOM":
                actions = {
                    # ISU
                    0: [1, 2, 3, 4 ,5],
                    # IFU
                    1: [1, 2, 3, 4, 5]
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

        return len(self.design_space.descriptions[self.design].keys())


class RocketEnv(BasicEnv):
    """
        Rocket environment
    """
    def __init__(self, configs, idx):
        super(RocketEnv, self).__init__(configs, idx)
        self.observation_space = self.dims_of_state
        self.action_space = self.dims_of_action
        self.reward_space = self.dims_of_reward
        self.load_ppa_model()
        self.state = None
        self.state_idx = None

    @property
    def if_terminate(self):
        """
            If a state has no zero element, then it is terminated.
        """
        return len(np.where(self.state == 0)[0]) == 0

    @property
    def current_state(self):
        return self.state_idx

    def load_ppa_model(self):
        perf_root = os.path.join(
            self.ppa_model_root,
            "rocket-perf.pt"
        )
        power_root = os.path.join(
            self.ppa_model_root,
            "rocket-power.pt"
        )
        area_root = os.path.join(
            self.ppa_model_root,
            "rocket-area.pt"

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
        """
            Hard-coded tunable state offsets.
            Please see `component_touch` in `BasicEnv`.
        """
        return [
            0, 2, 3, 4, 1, 5
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
        """
            DEPRECATED.
        """
        return np.array(self.design_space.idx_to_vec(
                692
            )
        )

    def get_human_baseline(self):
        """
            DEPRECATED.
        """
        ppa = {
            "Rocket": [0.822898, 0.007800, 682508.000000] # [0.801072362, 0.0026, 908152.038]
        }
        # negate
        baseline = ppa[self.configs["design"]]
        baseline[1] = -baseline[1]
        baseline[2] = -baseline[2]
        return np.array(baseline)

    def generate_init_state(self):
        self.state = np.zeros(self.observation_space).astype(int)

    def reset(self):
        self.reset_state_idx()
        self.generate_init_state()
        return self.state

    def render(self):
        return NotImplementedError

    def close(self):
        return NotImplementedError
