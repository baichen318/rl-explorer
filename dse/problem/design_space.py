# Author: baichen318@gmail.com

import random
import torch
import numpy as np
from time import time
from collections import OrderedDict

class Space(object):
    def __init__(self, dims):
        self.dims = dims
        # calculated manually
        self.size = 1373552640
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

class DesignSpace(Space):
    def __init__(self, features, bounds, dims, **kwargs):
        """
            features: <list>
            bounds: <OrderedDict>: <str> - <numpy.array>
        """
        self.features = features
        self.bounds = bounds
        super().__init__(dims)
        random.seed(kwargs["random_state"])
        self.random_state = np.random.RandomState(kwargs["random_state"])
        self.basic_component = kwargs["basic_component"]

    def set_random_state(self, random_state):
        random.seed(random_state)
        self.random_state = np.random.RandomState(random_state)

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
            if k == "icache":
                if decodeWidth >= min(self.bounds[self.features[1]]):
                    return random.sample([2, 4, 6, 8], 1)[0]
                else:
                    return random.sample(v, 1)[0]
            elif k == "fetchWidth":
                if design[0] in [1, 3, 5, 7]:
                    return v[0]
                else:
                    assert design[0] in [2, 4, 6, 8], "[ERROR]: design[0]: %d" % design[0]
                    return v[1]
            elif k == "numFetchBufferEntires":
                return random.sample([i for i in v if i % decodeWidth == 0 and i > design[1]], 1)[0]
            elif k == "numRobEntries":
                return random.sample([i for i in v if i % decodeWidth == 0], 1)[0]
            elif k == "registers":
                return random.sample([idx + 1 for idx, i in enumerate(v) \
                    if self.basic_component["registers"][i - 1][0] >= (32 + decodeWidth) and \
                        self.basic_component["registers"][i - 1][1] >= (32 + decodeWidth)], 1)[0]
            elif k == "decodeWidth":
                return decodeWidth
            else:
                return random.sample(v, 1)[0]

        design = []
        for k, v in self.bounds.items():
            design.append(__filter(design, k, v))
        return design

    def sample(self, batch):
        visited = set()
        samples = []
        cnt = 0
        while cnt < batch:
            # randomly sample designs w.r.t. decodeWidth
            for decodeWidth in self.bounds[self.features[7]]:
                design = self._sample(decodeWidth)
                point = self.knob2point(design)
                while point in visited:
                    design = self._sample(decodeWidth)
                    point = self.knob2point(design)
                visited.add(point)
                samples.append(design)
            cnt += 1
        return torch.Tensor(samples)

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

    return DesignSpace(features, bounds, dims, **kwargs)
