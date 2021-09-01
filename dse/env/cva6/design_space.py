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

class Space(object):
    def __init__(self, dims):
        self.dims = dims
        self.size = np.prod(self.dims)
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

class CVA6DesignSpace(Space):
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

    def _sample(self, decodeWidth):
        def __filter(design, k, v):
            return random.sample(v, 1)[0]

        design = []
        for k, v in self.bounds.items():
            design.append(__filter(design, k, v))
        return design

    def sample(self, batch):
        samples = []
        cnt = 0
        while cnt < batch:
            design = self._sample(decodeWidth)
            point = self.knob2point(design)
            while point in self.visited:
                design = self._sample(decodeWidth)
                point = self.knob2point(design)
            self.visited.add(point)
            samples.append(design)
            cnt += 1
        return torch.Tensor(samples).long()

    def validate(self, configs):
        # validate w.r.t. `configs`
        # `fetchWidth` >= `decodeWidth`
        if not (configs[1] >= configs[4]):
            return False
        # `numRobEntries` % `decodeWidth` = 0
        if not (configs[5] % configs[4] == 0):
            return False
        # `numFetchBufferEntries` % `decodeWidth` = 0
        if not (self.basic_component["ifu-buffers"][configs[2]][0] % configs[4] == 0):
            return False
        # `numFetchBufferEntries` > `fetchWidth`
        if not (self.basic_component["ifu-buffers"][configs[2]][0] > configs[1]):
            return False
        # `fetchWidth` = 4 when `decodeWidth` <= 2
        if not (configs[1] == 4 and configs[4] <= 2):
            return False
        # `fetchWidth` = 8 when `decodeWidth` > 2
        if not (configs[1] == 8 and configs[4] > 2):
            return False
        return True

def parse_design_space(design_space, **kwargs):
    bounds = OrderedDict()
    dims = []
    features = []
    for k, v in design_space.items():
        # add `features`
        features.append(k)
        if 'candidates' in v.keys():
            temp = v['candidates']
        else:
            assert 'start' in v.keys() and 'end' in v.keys() and \
                'stride' in v.keys(), "[ERROR]: assert failed. YAML includes errors."
            temp = np.arange(v['start'], v['end'] + 1, v['stride'])
        # generate bounds
        bounds[k] = list(temp)
        # generate dims
        dims.append(len(temp))

    return CVA6DesignSpace(features, bounds, dims, **kwargs)
