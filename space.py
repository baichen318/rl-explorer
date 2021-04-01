# Author: baichen318@gmail.com

import numpy as np
from collections import OrderedDict
from util import write_csv

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
        self.features = features
        self.bounds = bounds
        # handle `size`
        size /= len(bounds["fetchWidth"])
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
        for idx in range(self.n_dim):
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
        # numFetchBufferEntries % decodeWidth
        if not (_vec[2] % _vec[1] == 0):
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
        # numIntPhysRegisters == numFpPhysRegisters
        if not (_vec[5] == _vec[6]):
            return False
        # numLdqEntries == numStqEntries
        if not (_vec[7] == _vec[8]):
            return False
        # fetchWidth = 4, ICacheParams_fetchBytes = 2
        # fetchWidth = 8, ICacheParams_fetchBytes = 4
        if not (_vec[18] * 2 == _vec[0]):
            return False
        return True

    def _enumerate_design_space(self, file, idx, dataset, data):
        if idx >= self.n_dim:
            return
        for v in self.bounds[self.features[idx]]:
            data.append(v)
            if len(data) >= 9 and (data[7] != data[8]):
                data.pop()
                continue
            if len(data) >= 7 and (data[5] != data[6]):
                data.pop()
                continue
            if len(data) == self.n_dim and (self.verify_features(np.array(data))):
                dataset.append(np.array(data))
                self.size += 1
                data.pop()
            else:
                self._enumerate_design_space(file, idx + 1, dataset, data)
                data.pop()
                if len(dataset) >= 500:
                    write_csv(file, np.array(dataset), mode='a')
                    dataset.clear()

    def enumerate_design_space(self, file):
        dataset = []
        self.size = 0
        self._enumerate_design_space(file, 0, dataset, [])
        print("[INFO]: the size of the design space:", self.size)

    def random_sample(self, batch):
        """
            It cannot sample some configs. frequently
        """
        data = []

        visited = set()
        for i in range(batch):
            _data = np.empty((1, self.n_dim))
            for col, candidates in enumerate(self.bounds.values()):
                if self.features[col] == "numFetchBufferEntries" and \
                    _data.T[-1] == 5:
                    candidates = [35, 40]
                if self.features[col] == "numFetchBufferEntries" and \
                    _data.T[-1] == 3:
                    candidates = [8, 24]
                if self.features[col] == "numFpPhysRegisters" or \
                    self.features[col] == "numStqEntries":
                    _data.T[col] = _data.T[col - 1]
                    continue
                if self.features[col] == "fp_issueWidth":
                    _data.T[col] = _data.T[col - 2]
                    continue
                if self.features[col] == "ICacheParams_fetchBytes":
                    if _data.T[0] == 8:
                        _data.T[col] = 4
                    else:
                        _data.T[col] = 2
                    continue
                _data.T[col] = self.random_state.choice(candidates, size=1)
            while (not self.verify_features(_data[0])) and \
                (not self.knob2point(_data.ravel()) in visited):
                for col, candidates in enumerate(self.bounds.values()):
                    if self.features[col] == "numFetchBufferEntries" and \
                        _data.T[-1] == 5:
                        candidates = [35, 40]
                    if self.features[col] == "numFetchBufferEntries" and \
                        _data.T[-1] == 3:
                        candidates = [8, 24]
                    if self.features[col] == "numFpPhysRegisters" or \
                        self.features[col] == "numStqEntries":
                        _data.T[col] = _data.T[col - 1]
                        continue
                    if self.features[col] == "fp_issueWidth":
                        _data.T[col] = _data.T[col - 2]
                        continue
                    if self.features[col] == "ICacheParams_fetchBytes":
                        if _data.T[0] == 8:
                            _data.T[col] = 4
                        else:
                            _data.T[col] = 2
                        continue
                    _data.T[col] = self.random_state.choice(candidates, size=1)
            _data = self.round_vec(_data.ravel())
            data.append(_data)
            visited.add(self.knob2point(_data))

        return np.array(data)

    def random_sample_v2(self, batch):
        # line indicator to `data/design-space.ft`
        stage = [
            # `decodeWidth` == 1 & `fetchWdith` == 4
            (1, 34992195),
            # `decodeWidth` == 2 & `fetchWdith` == 4
            (34992196, 64091596),
            # `decodeWidth` == 3 & `fetchWdith` == 4
            (64567822, 65305196),
            # `decodeWidth` == 4 & `fetchWdith` == 4
            (65500000, 88646595),
            # `decodeWidth` == 1 & `fetchWdith` == 8
            (88646596, 100000027),
            # `decodeWidth` == 2 & `fetchWdith` == 8
            (124999979, 140367842),
            # `decodeWidth` == 3 & `fetchWdith` == 8
            (141134596, 142300995),
            # `decodeWidth` == 4 & `fetchWdith` == 8
            (142300996, 160001118),
            # `decodeWidth` == 5 & `fetchWdith` == 5
            (163296195, 161999974)
        ]
        def get_feature_from_file(line):
            f = open("data/design-space.ft", "r")
            cnt = 0
            data = None

            for i in range(1, line):
                _ = next(f)
            for i in f:
                data = i
            f.close()
            data = self.round_vec(data.strip().split(','))
            return data

        data = []
        visited = set()
        cnt = 0

        while cnt < batch:
            s = stage[cnt % len(stage)]
            line = round(np.random.uniform(s[0], s[1]))
            _data = get_feature_from_file(line)
            while self.knob2point(np.arange(_data)) in visited:
                line = round(np.random.uniform(s[0], s[1]))
                _data = get_feature_from_file(line)
            visited.add(self.knob2point(np.array(_data)))
            data.append(_data)
            cnt += 1

        return np.array(data)

    def knob2point(self, vec):
        """
            vec: `np.array`
        """
        return super().knob2point(
            self.features2knob(
                self.round_vec(vec)
            ),
            self.dims
        )

    def random_walk(self, vec):
        """
            vec: `np.array`
        """
        old = list(vec).copy()
        new = list(vec)

        cnt = 0
        while new != old and cnt < 2:
            from_i = np.random.randint(len(old))
            to_v = np.random.choice(self.bounds[from_i])
            new[from_i] = to_v
            while self.verify_features(np.array(new)):
                to_v = np.random.choice(self.bounds[from_i])
                new[from_i] = to_v
            if new != old:
                cnt += 1

        return np.array(new)

def parse_design_space(design_space):
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
            size *= len(temp)
            # generate bounds
            bounds[k] = np.array(temp)
            # generate dims
            dims.append(len(temp))
        else:
            assert 'start' in v.keys() and 'end' in v.keys() and \
                'stride' in v.keys(), "[ERROR]: assert failed. YAML includes errors."
            temp = np.arange(v['start'], v['end'] + 1, v['stride'])
            size *= len(temp)
            # generate bounds
            bounds[k] = temp
            # generate dims
            dims.append(len(temp))

    return DesignSpace(features, bounds, dims, size)
