# Author: baichen318@gmail.com

import os
import sys
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.pardir, os.path.pardir, os.path.pardir, "util")
    )
)
import random
import torch
import numpy as np
from time import time
from collections import OrderedDict
from util import if_exist, load_txt
from vlsi.rocket.vlsi import Gem5Wrapper


class Space(object):
    def __init__(self, dims):
        self.dims = dims
        # calculated manually
        self.size = 3600
        self.n_dim = len(self.dims)

    def point2knob(self, p, dims):
        """convert point form (single integer) to knob form (vector)"""
        knob = []
        for dim in dims:
            knob.append(p % dim)
            p //= dim

        return knob

    def knob2point(self, knob, dims):
        """convert knob form (vector) to point form (single integer)"""
        p = 0
        for j, k in enumerate(knob):
            p += int(np.prod(dims[:j])) * k

        return p

class RocketDesignSpace(Space):
    def __init__(self, features, bounds, dims, **kwargs):
        """
            features: <list>
            bounds: <OrderedDict>: <str> - <numpy.array>
        """
        self.features = features
        self.bounds = bounds
        super().__init__(dims)
        self.set_random_state(kwargs["random_state"])
        self.basic_component = kwargs["basic_component"]
        self.visited = set()
        self.load_ppa_model()

    def set_random_state(self, random_state):
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)

    def point2knob(self, point):
        return [self.bounds[self.features[idx]][i] \
            for idx, i in enumerate(super().point2knob(point, self.dims))]

    def knob2point(self, design):
        return super().knob2point(
            [self.bounds[self.features[idx]].index(i) \
                for idx, i in enumerate(design)],
            self.dims
        )

    def _sample(self):
        def __filter(design, k, v):
            # Notice: Google sheet
            return random.sample(v, 1)[0]

        design = []
        for k, v in self.bounds.items():
            design.append(__filter(design, k, v))
        return design

    def sample(self, batch, f=None):
        samples = []
        cnt = 0
        while cnt < batch:
            design = self._sample()
            point = self.knob2point(design)
            while point in self.visited:
                design = self._sample()
                point = self.knob2point(design)
            self.visited.add(point)
            samples.append(design)
            cnt += 1
        return torch.Tensor(samples).long().squeeze(0)

    def validate(self, configs):
        return True

    def load_ppa_model(self):
        pass

    def evaluate_microarchitecture(self, configs, state, idx, split=False, test=False):
        # NOTICE: we use light-weight white-box model
        if test:
            return torch.Tensor([random.random()]).squeeze(0)
        manager = Gem5Wrapper(configs, state, idx)
        ipc = manager.evaluate_perf()
        power, area = manager.evaluate_power_and_area()
        print("[INFO]: state:", state, "IPC: %f, Power: %f, Area: %f" % (ipc, power, area))
        # TODO: PV as the reward
        if split:
            return ipc, power, area
        else:
            return ipc + power + area


def parse_design_space(design_space, **kwargs):
    bounds = OrderedDict()
    dims = []
    features = []
    for k, v in design_space.items():
        # add `features`
        features.append(k)
        if "candidates" in v.keys():
            temp = v["candidates"]
        else:
            assert "start" in v.keys() and "end" in v.keys() and \
                "stride" in v.keys(), "[ERROR]: assert failed. YAML includes errors."
            temp = np.arange(v["start"], v["end"] + 1, v["stride"])
        # generate bounds
        bounds[k] = list(temp)
        # generate dims
        dims.append(len(temp))

    return RocketDesignSpace(features, bounds, dims, **kwargs)

