import sys
import os
import torch
import gpytorch
import torch.nn as nn
import numpy as np
from botorch.models import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from time import time
from util import create_logger, read_csv
from exception import UnDefinedException

class MLP(nn.Sequential):
    """
        MLP as preprocessor of DNNGP
    """
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.add_module("linear-1", nn.Linear(input_dim, 1000))
        self.add_module("relu-1", nn.ReLU())
        self.add_module("linear-2", nn.Linear(1000, 500))
        self.add_module("relu-2", nn.ReLU())
        self.add_module("linear-3", nn.Linear(500, 50))
        self.add_module("relu-3", nn.ReLU())
        self.add_module("linear-4", nn.Linear(50, output_dim))

class DNNGP():
    """
        DNN-GP
    """
    def __init__(self, x, y, configs):
        self.configs = configs
        self.n_dim = 19
        self.n_target = 2
        self.mlp = MLP(self.n_dim, 6)
        self.learning_rate = self.configs["learning-rate"]
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        # other variables
        self.optimizer = None
        self.gp = None

        self.init(x, y)

    def init(self, x, y):
        """
            x: <torch.Tensor>
            y: <torch.Tensor>
        """
        x = x.to(self.device)
        y = y.to(self.device)
        self.mlp = self.mlp.to(self.device)
        x = self.forward_mlp(x)
        x = self.transform_xlayout(x)
        y = self.transform_ylayout(y)
        self.gp = MultiTaskGP(x, y, task_feature=-1, mean_type="linear")

    def set_train(self):
        self.gp.train()
        self.gp.likelihood.train()
        self.mlp.train()
        self.gp = self.gp.to(self.device)
        self.likelihood = self.gp.likelihood.to(self.device)
        self.mlp = self.mlp.to(self.device)

    def set_eval(self):
        self.mlp.eval()
        self.gp.eval()
        self.gp.likelihood.eval()

    def transform_xlayout(self, x):
        """
            [x1; x2; x3]  <-->  [y11, y12; y21, y22; y31, y33]
                =>
            [x1, 0; x2, 0; x3, 0; x1, 1; x2, 1; x3, 1]  <-->  [y11; y21; y31; y12; y22; y32]
        """
        nsample = x.shape[0]
        x = torch.cat([x for i in range(self.n_target)], dim=0)
        task_index = torch.zeros(nsample, 1).to(x.device)
        for i in range(1, self.n_target):
            task_index = torch.cat([task_index, i * torch.ones(nsample, 1).to(x.device)], dim=0)
        x = torch.cat([x, task_index], dim=1)
        return x

    def transform_ylayout(self, y):
        """
            [x1; x2; x3]  <-->  [y11, y12; y21, y22; y31, y33]
                =>
            [x1, 0; x2, 0; x3, 0; x1, 1; x2, 1; x3, 1]  <-->  [y11; y21; y31; y12; y22; y32]
        """
        y = y.chunk(self.n_target, dim=1)
        y = torch.cat([y[i] for i in range(len(y))], dim=0)
        return y

    def forward_mlp(self, x):
        x = self.mlp(x)
        x = x - x.min(0)[0]
        x = 2 * (x / x.max(0)[0]) - 1
        return x

    def forward(self, x, train=True):
        x = x.to(self.device)
        x = self.forward_mlp(x)
        x = self.transform_xlayout(x)
        if train:
            self.gp.set_train_data(x)
        y = self.gp(x)

        return y

    def predict(self, x):
        """
            x: <torch.Tensor>
        """
        def _transform_ylayout(y):
            """
                y: <torch.Tensor>
            """
            y = y.chunk(self.n_target, dim=0)
            return torch.cat([y[i].unsqueeze(1) for i in range(2)], dim=1)

        self.set_eval()
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            pred = self.forward(x, train=False)
        pred = _transform_ylayout(pred.mean)

        return pred

    def sample(self, x, y):
        """
            x: <list>
            y: <list>
        """
        # TODO: normalize + ref-poi
        acq_func = ExpectedHypervolumeImprovement(
            model=self.gp,
            ref_point=torch.tensor([1, 1]).to(self.device),
            partitioning=NondominatedPartitioning(
                ref_point=torch.tensor([1, 1]).to(self.device),
                Y=y.to(self.device)
            )
        ).to(self.device)
        _x = self.mlp(x.to(self.device).float())
        acqv = acq_func(_x.unsqueeze(1).to(self.device))
        top_k, idx = torch.topk(acqv, k=5)
        new_x = x[idx]
        new_y = y[idx]

        return new_x.reshape(-1, 19), new_y.reshape(-1, 2), torch.mean(top_k)

    def save(self, path):
        state_dict = {
            "mlp": self.mlp.state_dict(),
            "gp": self.gp.state_dict()
        }
        torch.save(state_dict, path)
        print("[INFO]: saving model to %s" % path)

    def load(self, mdl, x, y):
        state_dict = torch.load(mdl)
        self.mlp.load_state_dict(state_dict["mlp"])
        self.gp.load_state_dict(state_dict["gp"])
        self.set_eval()

class BayesianOptimization(object):
    """
        BayesianOptimization Framework
    """
    def __init__(self, configs, random_state=round(time())):
        self.configs = configs
        self.random_state = np.random.RandomState(random_state)

    def set_input(self, x, y):
        """
            x: <np.array>
            y: <np.array>
        """
        self.x = x.tolist()
        self.y = y.tolist()

    def run(self):
        x = []
        y = []

        for i in range(8):
            idx = self.random_state.randint(1, len(self.x))
            x.append(self.x[idx])
            y.append(self.y[idx])
            self.x.pop(idx)
            self.y.pop(idx)

        for step in range(1, self.configs["max-bo-steps"] + 1):
            print("[INFO]: Training size: %d" % len(x))
            model = DNNGP(self.configs)
            # train `DNNGP`
            model.fit(
                torch.tensor(x),
                torch.tensor(y)
            )
            # sample w.r.t. acquisition function
            new_x, new_y, acqv = model.sample(
                torch.tensor(self.x),
                torch.tensor(self.y)
            )
            # add new samples and remove sampled ones
            pidx = []
            for item in new_x:
                x.append(item)
                idx = self.x.index(item.tolist())
                self.x.pop(idx)
                self.y.pop(idx)
            for item in new_y:
                y.append(item)

        model.save(
            os.path.join(
                self.configs["model-output-path"],
                self.configs["model"] + ".mdl"
            )
        )

    def fit(self, x, y):
        """
            API for running `BayesianOptimization`
        """
        self.set_input(x, y)
        self.run()

    def predict(self, x):
        model = DNNGP(self.configs)
        model.load(
            os.path.join(
                self.configs["model-output-path"],
                self.configs["model"] + ".mdl"
            )
        )
        return model.predict(torch.tensor(x).float())
