# Author: baichen318@gmail.com


import os
import torch
import numpy as np
from typing import Optional
from utils.utils import assert_error
from baselines.boom_explorer.util.util import load_dataset
from botorch.test_functions.base import MultiObjectiveTestProblem
from dse.env.boom.design_space import parse_design_space as parse_boom_design_space
from dse.env.rocket.design_space import parse_design_space as parse_rocket_design_space


class BOOMDesignProblem(MultiObjectiveTestProblem):
    """
        BOOMDesignProblem
    """
    def __init__(self, configs, noise_std: Optional[float]=None, negate: bool=False):
        self.configs = configs
        # NOTICE: for dataset
        # after scaling, the clock cycles are [0.2948, 1.32305],
        # the power values are [0.5349999999999999, 1.5405000000000002]
        assert "BOOM" in self.configs["algo"]["design"] or \
            self.configs["algo"]["design"] == "Rocket", \
            assert_error("{} is not supported.".format(
                self.configs["design"])
            )
        self.boom = "BOOM" in self.configs["algo"]["design"]
        if self.boom:
            self._ref_point = torch.tensor([0.0, 0.0, 0.0])
            self._bounds = torch.tensor([(3.0, 0.2, 2)])
        else:
            self._ref_point = torch.tensor([0.0, 0.0, 0.0])
            self._bounds = torch.tensor([(1.0, 0.02, 1.0)])
        self.total_x, self.total_y = load_dataset(
            os.path.join(
                configs["env"]["calib"]["dataset"]
            ),
            boom=self.boom
        )
        self.total_x, self.total_y = torch.tensor(self.total_x), torch.tensor(self.total_y)
        self.x, self.y = self.total_x.clone(), self.total_y.clone()
        self.n_dim = self.x.shape[-1]
        self.n_sample = self.x.shape[0]
        if self.boom:
            self.design_space = parse_boom_design_space(self.configs)
        else:
            self.design_space = parse_rocket_design_space(self.configs)
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, x: torch.Tensor) -> torch.Tensor:
        """
            similar to __getiterm__
        """
        _, indices = torch.topk(
            ((self.x.t() == x.unsqueeze(-1)).all(dim=1)).int(),
            1,
            1
        )
        return self.y[indices].to(torch.float32).squeeze()

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

    def reset(self):
        self.x, self.y = self.total_x.clone(), self.total_y.clone()


def define_problem(configs):
    return BOOMDesignProblem(configs)
