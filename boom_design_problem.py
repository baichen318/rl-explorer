# Author: baichen318@gmail.com
import torch
import numpy as np
from typing import Optional
from botorch.test_functions.base import MultiObjectiveTestProblem
from util import load_dataset

class BOOMDesignProblem(MultiObjectiveTestProblem):
    """ BOOMDesignProblem
    """
    def __init__(self, configs, noise_std: Optional[float]=None, negate: bool=False):
        self.configs = configs
        self._ref_point = torch.tensor([0.0])
        self._bounds = torch.tensor([(0.0, 1.0)])
        self.total_x, self.total_y = load_dataset(configs["dataset-output-path"])
        self.x = self.total_x.copy()
        self.y = self.total_y.copy()
        self.total_x, self.total_y = torch.tensor(self.total_x), torch.tensor(self.total_y)
        self.x, self.y = torch.tensor(self.x), torch.tensor(self.y)
        self.n_dim = self.x.shape[-1]
        self.n_sample = self.x.shape[0]

        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, x: torch.Tensor) -> torch.Tensor:
        """
            similar to __getiterm__
        """
        _, indices = torch.topk(
            (((self.x.t() == x.unsqueeze(-1))).all(dim=1)).int(),
            1,
            1
        )
        return self.y[indices].squeeze()

    def remove_sampled_data(self, x: torch.Tensor):
        sampled = torch.zeros(
            self.x.size()[0],
            dtype=torch.bool
        )[:, np.newaxis]
        _, indices = torch.topk(
            ((self.x.t() == x.unsqueeze(-1)).all(dim=1)).int(),
            1,
            1
        )
        mask = sampled.index_fill_(0, indices.squeeze(), True).squeeze()
        self.x = self.x[mask[:] == False]
        self.y = self.y[mask[:] == False]
