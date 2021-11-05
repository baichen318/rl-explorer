# Author: baichen318@gmail.com

import os
import torch
import numpy as np
from typing import Optional
from botorch.test_functions.base import MultiObjectiveTestProblem
from helper import load_dataset, transform_dataset
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib


class BOOMDesignProblem(MultiObjectiveTestProblem):
    """ BOOMDesignProblem
    """
    def __init__(self, configs, noise_std: Optional[float]=None, negate: bool=False):
        self.configs = configs
        self._ref_point = torch.tensor([0.0, 0.0, 0.0])
        self._bounds = torch.tensor([(2.0, 0.2, 5.0)])
        self.total_x, self.total_y = load_dataset(configs["dataset-output-path"])
        self.total_y = transform_dataset(self.total_y)
        self.x = self.total_x.copy()
        self.y = self.total_y.copy()
        self.total_x, self.total_y = torch.tensor(self.total_x), torch.tensor(self.total_y)
        self.x, self.y = torch.tensor(self.x), torch.tensor(self.y)
        self.n_dim = self.x.shape[-1]
        self.n_sample = self.x.shape[0]
        self.design_space = self.load_design_space()
        self.load_model()
        self.idx = 1

        super().__init__(noise_std=noise_std, negate=negate)

    def load_design_space(self):
        if self.configs["design"] == "rocket":
            from dse.env.rocket.design_space import parse_design_space
            design_space = parse_design_space(
                self.configs["design-space"],
                basic_component=self.configs["basic-component"],
                random_state=self.configs["seed"]
            )
        else:
            assert self.configs["design"] == "boom", \
                "[ERROR]: %s is not supported." % self.configs["design"]
            from dse.env.boom.design_space import parse_design_space
            design_space = parse_design_space(
                self.configs["design-space"],
                basic_component=self.configs["basic-component"],
                random_state=self.configs["seed"]
            )

        return design_space

    def load_model(self):
        self.ipc_model = joblib.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.path.pardir,
                os.path.pardir,
                "tools",
                self.configs["ppa-model"],
                self.configs["design"] + '-' + "ipc.pt"
            )
        )
        self.power_model = joblib.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.path.pardir,
                os.path.pardir,
                "tools",
                self.configs["ppa-model"],
                self.configs["design"] + '-' + "power.pt"
            )
        )
        self.area_model = joblib.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.path.pardir,
                os.path.pardir,
                "tools",
                self.configs["ppa-model"],
                self.configs["design"] + '-' + "area.pt"
            )
        )

    def evaluate_true(self, x: torch.Tensor) -> torch.Tensor:
        """
            similar to __getiterm__
        """
        _, indices = torch.topk(
            (((self.x.t() == x.unsqueeze(-1))).all(dim=1)).int(),
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
        self.n_sample = self.x.shape[0]

    def evaluate_microarchitecture(self, x):
        ipc, power, area = self.design_space.evaluate_microarchitecture(
            self.configs,
            x.astype(int),
            self.idx
        )
        ipc = self.ipc_model.predict(
            np.expand_dims(
                np.concatenate((x.astype(int), [ipc])),
                axis=0
            )
        )
        power = self.power_model.predict(
            np.expand_dims(
                np.concatenate((x.astype(int), [power])),
                axis=0
            )
        )
        area = self.area_model.predict(
            np.expand_dims(
                np.concatenate((x.astype(int), [area])),
                axis=0
            )
        )
        # trasfer area to mm^2
        if self.configs["design"] == "rocket":
            ipc = 10 * ipc
            power = 10 * power
            area = area * 1e-6 * 10
        else:
            assert self.configs["design"] == "boom", \
                "[ERROR]: %s is not supported." % self.configs["design"]
            ipc = 2 * ipc
            power = 2 * 10 * power
            area = 0.5 * 1e-6 * area
        msg = "[INFO]: microarchitecture: {}, IPC: {}, power: {}, area: {}".format(
            x,
            ipc,
            power,
            area
        )
        metric = torch.Tensor(np.concatenate((ipc, power, area))).unsqueeze(0)
        metric = transform_dataset(metric)
        return metric
