# Author: baichen318@gmail.com

import random
import torch
import numpy as np
from time import time
from collections import OrderedDict

class Space(object):
    def __init__(self, dims, size):
        self.dims = dims
        self.size = size
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
    def __init__(self, features, bounds, dims, size, random_state=1):
        """
            features: <list>
            bounds: <OrderedDict>: <str> - <np.array>
        """
        self.features = features
        self.bounds = bounds
        # handle `size`
        size /= len(bounds["fetchWidth"])
        size /= len(bounds["numFpPhysRegisters"])
        size /= len(bounds["numStqEntries"])
        size /= len(bounds["fp_issueWidth"])
        super().__init__(dims, size)
        random.seed(random_state)
        self.random_state = np.random.RandomState(random_state)

    def set_random_state(self, random_state):
        random.seed(random_state)
        self.random_state = np.random.RandomState(random_state)

    def euclidean_dist_func(self):
        return torch.nn.PairwiseDistance(p=2)

    def random_walk_by_soft_mutation(self, vec, dataset):
        """
            vec: <torch.Tensor>
            dataset: <torch.Tensor>
        """
        idx = random.choice(range(self.n_dim))
        dist = []
        i = 0
        fdist = euclidean_dist_func()
        for data in dataset:
            dist.append(
                (
                    float(
                        fdist(
                            torch.cat([
                                vec[:idx],
                                vec[idx + 1:]
                            ]).unsqueeze(0),
                            torch.cat([
                                data[:idx],
                                data[idx + 1:]
                            ]).unsqueeze(0)
                        )
                    ),
                    i
                )
            )
            i += 1
        dist = sorted(dist, key=lambda t: t[0])
        bounds = self.bounds[self.features[idx]].copy()
        bounds.remove(vec[idx])
        for item in dist:
            if dataset[item[1]][idx] in bounds:
                return dataset[item[1]]

    def random_walk_by_neighborhood(self, vec, dataset):
        # 1. calculate all distance
        # 2. choose top-k within the neighborhood randomly
        """
            vec: <torch.Tensor>
            dataset: <torch.Tensor>
        """
        dist = []
        i = 0
        fdist = euclidean_dist_func()
        for data in dataset:
            dist.append(
                (
                    float(
                        fdist(vec.unsqueeze(0), data.unsqueeze(0))
                    ),
                    i
                )
            )
            i += 1

        dist = sorted(dist, key=lambda t: t[0])
        idx = random.choice(range(1, 10))
        return dataset[dist[idx][1]]



    # def round_vec(self, vec):
    #     _vec = []
    #     for item in vec:
    #         _vec.append(int(item))

    #     return _vec

    # def features2knob(self, vec):
    #     """
    #         vec: <numpy.ndarray>
    #     """
    #     ret = []
    #     for idx in range(self.n_dim):
    #         ret.append(
    #             np.argwhere(
    #                 self.bounds[self.features[idx]] == vec[idx]
    #             )[0][0]
    #         )

    #     return ret

    # def verify_features(self, vec):
    #     """
    #         vec: <numpy.ndarray>
    #     """
    #     _vec = self.round_vec(vec)
    #     # constraint #1
    #     if (_vec[0] & (_vec[0] - 1)):
    #         return False
    #     # constraint #2
    #     if not (_vec[1] <= _vec[0]):
    #         return False
    #     # constraint #4
    #     if not (_vec[5] >= 32 + _vec[1]):
    #         return False
    #     # constraint #5
    #     if not (_vec[6] >= 32 + _vec[1]):
    #         return False
    #     # constraint #7
    #     if not (_vec[3] % _vec[1] == 0):
    #         return False
    #     # constraint #8
    #     if not ((_vec[7] - 1) > _vec[1]):
    #         return False
    #     # constraint #9
    #     if not ((_vec[8] - 1) > _vec[1]):
    #         return False
    #     # constraint #10
    #     if not (_vec[2] > _vec[0]):
    #         return False
    #     # constraint #11
    #     if not (_vec[2] % _vec[1] == 0):
    #         return False
    #     # constraint #12
    #     if not (_vec[18] * 2 == _vec[0]):
    #         return False
    #     # constraint #13
    #     if not (_vec[5] == _vec[6]):
    #         return False
    #     # constraint #14
    #     if not (_vec[7] == _vec[8]):
    #         return False
    #     # constraint #15
    #     if not (_vec[10] == _vec[12]):
    #         return False
    #     return True


    # def random_sample(self, batch):
    #     """
    #         It cannot sample some configs. uniformly accorting to
    #         micro-architectural structures.
    #         Use `_helper` functions to accept a configuration
    #         as much as possible.
    #     """
    #     def _helper(feature, data, candidates):
    #         _candidates = np.array([], dtype=int)
    #         if feature == "decodeWidth":
    #             # merged constraint #1
    #             for c in candidates:
    #                 if c <= data[0]:
    #                     _candidates = np.append(_candidates, c)
    #             return _candidates
    #         elif feature == "numRobEntries":
    #             # merged constraint #2
    #             for c in candidates:
    #                 if c % data[1] == 0:
    #                     _candidates = np.append(_candidates, c)
    #             return _candidates
    #         elif feature == "numFetchBufferEntries":
    #             # merged constraint #3
    #             for c in candidates:
    #                 if c > data[0] and c % data[1] == 0:
    #                     _candidates = np.append(_candidates, c)
    #             return _candidates
    #         elif feature == "ICacheParams_fetchBytes":
    #             # merged constraint #4
    #             if data[0] == 4:
    #                 _candidates = np.append(_candidates, 2)
    #             else:
    #                 assert data[0] == 8
    #                 _candidates = np.append(_candidates, 4)
    #             return _candidates
    #         elif feature == "numFpPhysRegisters":
    #             # merged constraint #5
    #             _candidates = np.append(_candidates, data[5])
    #             return _candidates
    #         elif feature == "numStqEntries":
    #             # merged constraint #6
    #             _candidates = np.append(_candidates, data[7])
    #             return _candidates
    #         elif feature == "fp_issueWidth":
    #             # merged constraint #7
    #             _candidates = np.append(_candidates, data[10])
    #             return _candidates
    #         return candidates

    #     data = []

    #     visited = set()
    #     for i in range(batch):
    #         _data = np.empty(self.n_dim)
    #         for col, candidates in enumerate(self.bounds.values()):
    #             candidates = _helper(self.features[col], _data, candidates)
    #             _data.T[col] = self.random_state.choice(candidates, size=1)
    #         while (not self.verify_features(_data)) and \
    #             (not self.knob2point(_data) in visited):
    #             for col, candidates in enumerate(self.bounds.values()):
    #                 candidates = _helper(self.features[col], _data, candidates)
    #                 _data.T[col] = self.random_state.choice(candidates, size=1)
    #         _data = self.round_vec(_data)
    #         data.append(_data)
    #         visited.add(self.knob2point(_data))

    #     return np.array(data)

    # def random_sample_v2(self, decodeWidth, batch):
    #     """
    #         V2: random sample batches w.r.t. decodeWidth
    #     """
    #     def _helper(feature, data, candidates):
    #         _candidates = np.array([], dtype=int)
    #         if feature == "fetchWidth":
    #             # merged constraint #1
    #             for c in candidates:
    #                 if c >= data[1]:
    #                     _candidates = np.append(_candidates, c)
    #             return _candidates
    #         elif feature == "numRobEntries":
    #             # merged constraint #2
    #             for c in candidates:
    #                 if c % data[1] == 0:
    #                     _candidates = np.append(_candidates, c)
    #             return _candidates
    #         elif feature == "numFetchBufferEntries":
    #             # merged constraint #3
    #             for c in candidates:
    #                 if c > data[0] and c % data[1] == 0:
    #                     _candidates = np.append(_candidates, c)
    #             return _candidates
    #         elif feature == "ICacheParams_fetchBytes":
    #             # merged constraint #4
    #             if data[0] == 4:
    #                 _candidates = np.append(_candidates, 2)
    #             else:
    #                 assert data[0] == 8
    #                 _candidates = np.append(_candidates, 4)
    #             return _candidates
    #         elif feature == "numFpPhysRegisters":
    #             # merged constraint #5
    #             _candidates = np.append(_candidates, data[5])
    #             return _candidates
    #         elif feature == "numStqEntries":
    #             # merged constraint #6
    #             _candidates = np.append(_candidates, data[7])
    #             return _candidates
    #         elif feature == "fp_issueWidth":
    #             # merged constraint #7
    #             _candidates = np.append(_candidates, data[10])
    #             return _candidates
    #         return candidates

    #     data = []

    #     visited = set()
    #     for i in range(batch):
    #         _data = np.empty(self.n_dim)
    #         _data[1] = decodeWidth
    #         for col, candidates in enumerate(self.bounds.values()):
    #             if self.features[col] == "decodeWidth":
    #                 continue
    #             candidates = _helper(self.features[col], _data, candidates)
    #             _data.T[col] = self.random_state.choice(candidates, size=1)
    #         while (not self.verify_features(_data)) and \
    #             (not self.knob2point(_data.ravel()) in visited):
    #             for col, candidates in enumerate(self.bounds.values()):
    #                 if self.features[col] == "decodeWidth":
    #                     continue
    #                 candidates = _helper(self.features[col], _data, candidates)
    #                 _data.T[col] = self.random_state.choice(candidates, size=1)
    #         _data = self.round_vec(_data)
    #         data.append(_data)
    #         visited.add(self.knob2point(_data))

    #     return np.array(data)

    # def knob2point(self, knob):
    #     """
    #         knob: <numpy.ndarray>
    #     """
    #     return super().knob2point(
    #         self.features2knob(
    #             self.round_vec(knob)
    #         ),
    #         self.dims
    #     )

    # def point2knob(self, point):
    #     """
    #         point: <int>
    #     """
    #     knob = super().point2knob(
    #         point,
    #         self.dims
    #     )
    #     vec = []
    #     i = 0
    #     for k, v in self.bounds.items():
    #         vec.append(v[knob[i]])
    #         i += 1
    #     return vec

    # def random_walk(self, vec):
    #     """
    #         vec: <numpy.ndarray>
    #     """
    #     old = list(vec).copy()
    #     new = list(vec)
    #     _new = old.copy()

    #     while new == old:
    #         from_i = np.random.randint(len(old))
    #         to_v = np.random.choice(self.bounds[self.features[from_i]])
    #         new[from_i] = to_v
    #         if from_i == 5:
    #             new[6] = to_v
    #         if from_i == 6:
    #             new[5] = to_v
    #         if from_i == 7:
    #             new[8] = to_v
    #         if from_i == 8:
    #             new[7] = to_v
    #         if from_i == 10:
    #             new[12] = to_v
    #         if from_i == 12:
    #             new[10] = to_v
    #         while not self.verify_features(np.array(new)):
    #             new = _new.copy()
    #             from_i = np.random.randint(len(old))
    #             to_v = np.random.choice(self.bounds[self.features[from_i]])
    #             new[from_i] = to_v
    #             if from_i == 5:
    #                 new[6] = to_v
    #             if from_i == 6:
    #                 new[5] = to_v
    #             if from_i == 7:
    #                 new[8] = to_v
    #             if from_i == 8:
    #                 new[7] = to_v
    #             if from_i == 10:
    #                 new[12] = to_v
    #             if from_i == 12:
    #                 new[10] = to_v
    #     return np.array(new)

    # def random_walk_v2(self, vec):
    #     """
    #         vec: <numpy.ndarray>
    #     """
    #     def _helper(vec, from_i):
    #         candidates = np.array([], dtype=int)
    #         if from_i == 0:
    #             # merged constraint #1
    #             if vec[0] >= vec[1]:
    #                 return
    #             for c in self.bounds["fetchWidth"]:
    #                 if c >= vec[1]:
    #                     candidates = np.append(candidates, c)
    #         elif from_i == 3:
    #             # merged constraint #2
    #             if vec[3] % vec[1] == 0:
    #                 return
    #             else:
    #                 for c in self.bounds["numRobEntries"]:
    #                     if c % vec[1] == 0:
    #                         candidates = np.append(candidates, c)
    #         elif from_i == 2:
    #             # merged constraint #3
    #             if vec[from_i] > vec[0] and vec[from_i] % vec[1] == 0:
    #                 return
    #             else:
    #                 for c in self.bounds["numFetchBufferEntries"]:
    #                     if c > vec[0] and c % vec[1] == 0:
    #                         candidates = np.append(candidates, c)
    #         elif from_i == 18:
    #             # merged constraint #4 `ICacheParams_fetchBytes`
    #             if vec[0] == 4:
    #                 candidates = np.append(candidates, 2)
    #             else:
    #                 candidates = np.append(candidates, 4)
    #         elif from_i == 5:
    #             # merged constraint #5 `numIntPhyRegisters`
    #             candidates = np.append(candidates, vec[6])
    #         elif from_i == 6:
    #             # merged constraint #5 `numFpPhysRegisters`
    #             candidates = np.append(candidates, vec[5])
    #         elif from_i == 8:
    #             # merged constraint #6 `numStqEntries`
    #             candidates = np.append(candidates, vec[7])
    #         elif from_i == 7:
    #             # merged constraint #6 `numLdqEntries`
    #             candidates = np.append(candidates, vec[8])
    #         elif from_i == 12:
    #             # merged constraint #7 `fp_issueWidth`
    #             candidates = np.append(candidates, vec[10])
    #         elif from_i == 10:
    #             # merged constraint #7 `mem_issueWidth`
    #             candidates = np.append(candidates, vec[12])
    #         else:
    #             return

    #         to_v = np.random.choice(candidates)
    #         vec[from_i] = to_v


    #     old = list(vec).copy()
    #     new = list(vec)
    #     _new = old.copy()

    #     while new == old:
    #         from_i = np.random.randint(len(old))
    #         while from_i == 1:
    #             from_i = np.random.randint(len(old))
    #         to_v = np.random.choice(self.bounds[self.features[from_i]])
    #         new[from_i] = to_v
    #         _helper(new, from_i)
    #         while not self.verify_features(np.array(new)):
    #             new = _new.copy()
    #             from_i = np.random.randint(len(old))
    #             while from_i == 1:
    #                 from_i = np.random.randint(len(old))
    #             to_v = np.random.choice(self.bounds[self.features[from_i]])
    #             new[from_i] = to_v
    #             _helper(new, from_i)
    #     return np.array(new)

def parse_design_space(design_space, seed=round(time())):
    bounds = OrderedDict()
    dims = []
    size = 1
    features = []
    for k, v in design_space.items():
        # add `features`
        features.append(k)
        # calculate the size of the design space
        if 'candidates' in v.keys():
            temp = v['candidates']
        else:
            assert 'start' in v.keys() and 'end' in v.keys() and \
                'stride' in v.keys(), "[ERROR]: assert failed. YAML includes errors."
            temp = np.arange(v['start'], v['end'] + 1, v['stride'])
        size *= len(temp)
        # generate bounds
        bounds[k] = np.array(temp)
        # generate dims
        dims.append(len(temp))

    return DesignSpace(features, bounds, dims, size, seed)
