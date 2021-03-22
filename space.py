# Author: baichen318@gmail.com

import numpy as np

class Space(object):
    def __init__(self, dims, size):
        self.dims = dims
        self.size = size

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
        self.features = features
        self.bounds = bounds
        # handle `size`
        size /= len(bounds["numFpPhysRegisters"])
        size /= len(bounds["numStqEntries"])
        size /= len(bounds["fp_issueWidth"])
        super().__init__(dims, size)
        self.random_state = np.random.RandomState(random_state)

    def round_vec(self, vec):
        _vec = []
        for item in vec:
            _vec.append(int(item))

        return _vec

    def features2knob(self, vec):
        """
            vec: np.array
        """
        ret = []
        for idx in range(len(vec)):
            ret.append(
                np.argwhere(
                    self.bounds[self.features[idx]] == vec[idx]
                )[0][0]
            )

        return ret

    def verify_features(self, vec):
        """
            vec: np.array
        """
        _vec = self.round_vec(vec)
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

        visited = set()
        for i in range(batch):
            _data = np.empty((1, len(self.dims)))
            for col, candidates in enumerate(self.bounds.values()):
                if self.features[col] == "numFpPhysRegisters" or \
                    self.features[col] == "numStqEntries":
                    _data.T[col] = _data.T[col - 1]
                    continue
                if self.features[col] == "fp_issueWidth":
                    _data.T[col] = _data.T[col - 2]
                    continue
                _data.T[col] = self.random_state.choice(candidates, size=1)
            while (not self.verify_features(_data[0])) and \
                (not self.knob2point(
                    self.features2knob(
                        self.round_vec(_data.ravel())
                    ),
                    self.dims
                ) in visited):
                for col, candidates in enumerate(self.bounds.values()):
                    if self.features[col] == "numFpPhysRegisters" or \
                        self.features[col] == "numStqEntries":
                        _data.T[col] = _data.T[col - 1]
                        continue
                    if self.features[col] == "fp_issueWidth":
                        _data.T[col] = _data.T[col - 2]
                        continue
                    _data.T[col] = self.random_state.choice(candidates, size=1)
            _data = self.round_vec(_data.ravel())
            data.append(_data)
            visited.add(
                self.knob2point(
                    self.features2knob(_data),
                    self.dims
                )
            )

        return np.array(data)
