# Author: baichen318@gmail.com

import torch
import random
from design_space import parse_design_space
from ..util.util import load_dataset

class BasicProblem(object):
    """ BasicProblem """
    def __init__(self, configs):
        super(BasicProblem, self).__init__()
        self.configs = configs

class BoomDesignProblem(BasicProblem):
    """ BoomDesignProblem """
    def __init__(self, configs):
        super(BoomDesignProblem, self).__init__(configs)
        self.space = parse_design_space(self.configs["design-space"], seed=2021)
        self.dataset_x, self.dataset_y = load_dataset(self.configs["dataset-path"])
        self.n_total = len(self.dataset_x)
        self.train_x, self.train_y, self.test_x, self.test_y = self.split_dataset(
            torch.tensor(self.dataset_x),
            torch.tensor(self.dataset_y)
        )
        self.n_train = len(self.train_x)
        self.n_test = len(self.test_x)

    def get_train_label(self, x: torch.Tensor):
        _, indices = torch.topk(
            (((self.train_x == x.unsqueeze(0))).all(dim=1)).int(),
            1,
            0
        )
        return self.train_y[indices].to(torch.float32).squeeze()

    def get_test_label(self, x: torch.Tensor):
        _, indices = torch.topk(
            (((self.test_x == x.unsqueeze(0))).all(dim=1)).int(),
            1,
            0
        )
        return self.test_y[indices].to(torch.float32).squeeze()


    def split_dataset(self, x, y):
        def _generate_idx():
            seed = 2021
            random.seed(seed)
            train_idx = range(self.n_total)
            test_idx = random.sample(train_idx, (0.15 * self.n_total))
            for i in test_idx:
                train_idx.remove(i)
            return train_idx, test_idx

        train_idx, test_idx = _generate_idx()
        train_x = x[train_idx]
        train_y = self.get_train_label(train_x)
        test_x = x[test_idx]
        test_y = self.get_test_label(test_x)
        return train_x, train_y, test_x, test_y
