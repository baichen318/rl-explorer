# Author: baichen318@gmail.com

import random
import numpy as np

class Space(object):
    def __init__(self, dims, size):
        self.dims = dims,
        self.size = size

    def point2knob(p, dims):
    """convert point form (single integer) to knob form (vector)"""
    knob = []
    for dim in dims:
        knob.append(p % dim)
        p //= dim

    return knob

    def knob2point(knob, dims):
        """convert knob form (vector) to point form (single integer)"""
        p = 0
        for j, k in enumerate(knob):
            p += int(np.prod(dims[:j])) * k

        return p

class DesignSpace(Space):
    def __init__(self, features, bounds, dims, size):
        self.features = features
        self.bounds = bounds
        super().__init__(dims, size)

    def verify_features(vec):
        """
            vec: np.array
        """
        def round_vec(vec):
            _vec = []
            for item in vec:
                _vec.append(int(item))

            return _vec
        _vec = round_vec(vec)
        # fetchWidth = 2^x
        if (_vec[0] & (_vec[0] - 1)):
            return False
        # decodeWidth <= fetchWidth
        if not (_vec[1] <= _vec[0]):
            return False
        # numIntPhysRegisters >= (32 + decodeWidth)
        if not (_vec[5] >= 32 + _vec[1]):
            return False
        # numFpPhysRegisters >= (32 + decodeWidth)
        if not (_vec[6] >= 32 + _vec[1]):
            return False
        # numRobEntries % coreWidth == 0
        if not (_vec[3] % _vec[1] == 0):
            return False
        # (numLdqEntries - 1) > decodeWidth
        if not ((_vec[7] - 1) > _vec[1]):
            return False
        # (numStqEntries - 1) > decodeWidth
        if not ((_vec[8] - 1) > _vec[1]):
            return False
        # numFetchBufferEntries > fetchWidth
        if not (_vec[2] > _vec[0]):
            return False
        return True

    def random_sample(self, batch):
        data = []

        for i in range(batch):
            _data = np.empty((1, self.dim))
            for col, candidates in enumerate(self.bounds):
                if self.features[i] == "numFpPhysRegisters" or \
                    self.features[i] == "numStqEntries":
                    _data.T[col] = _data.T[col - 1]
                    continue
                if self.features[i] == "fp_issueWidth":
                    _data.T[col] = _data.T[col - 2]
                    continue
                _data.T[col] = random.choice(candidates, size=1)
            while not self.verify_features(_data[0]):
                for col, candidates in enumerate(self.bounds):
                    if self.features[i] == "numFpPhysRegisters" or \
                        self.features[i] == "numStqEntries":
                        _data.T[col] = _data.T[col - 1]
                        continue
                    if self.features[i] == "fp_issueWidth":
                        _data.T[col] = _data.T[col - 2]
                        continue
                    _data.T[col] = random.choice(candidates, size=1)
            data.append(_data.ravel())

        return data
