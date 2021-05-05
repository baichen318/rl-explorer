import sys
import os
import torch
import torch.nn as nn
import numpy as np
from botorch.models import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from time import time
from search import reference
from util import create_logger, read_csv
from exception import UnDefinedException

class DNNGP():
    """
        DNN-GP
    """
    class MLP(nn.Sequential):
        """
            MLP as preprocessor of DNNGP
        """
        def __init__(self, input_dim, output_dim):
            super(DNNGP.MLP, self).__init__()
            self.add_module("linear-1", nn.Linear(input_dim, 1000))
            self.add_module("relu-1", nn.ReLU())
            self.add_module("linear-2", nn.Linear(1000, 500))
            self.add_module("relu-2", nn.ReLU())
            self.add_module("linear-3", nn.Linear(500, 50))
            self.add_module("relu-3", nn.ReLU())
            self.add_module("linear-4", nn.Linear(50, output_dim))

    def __init__(self, configs):
        self.configs = configs

        # NOTICE: 19 micro-architectural structures & 2 interesting metrics
        # TODO: MLP output dimensions
        self.mlp = DNNGP.MLP(19, 19)
        self.learning_rate = self.configs["learning-rate"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # other variables
        self.optimizer = None
        self.gp = None

    def set_train(self, x, y):
        x, y = self.transform_xlayout(x), self.transform_ylayout(y)
        self.gp = MultiTaskGP(x, y, task_feature=-1)
        self.gp.train()
        self.gp.likelihood.train()
        self.mlp.train()
        self.gp.to(self.device)
        self.gp.likelihood.to(self.device)
        self.mlp.to(self.device)

    def set_eval(self):
        self.mlp.eval()
        self.gp.eval()
        self.gp.likelihood.eval()

    def set_optimizer(self):
        parameters = [
            {"params": self.mlp.parameters()},
            {"params": self.gp.covar_module.parameters()},
            {"params": self.gp.mean_module.parameters()},
            {"params": self.gp.likelihood.parameters()}
        ]
        if self.configs["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(
                parameters,
                lr=self.learning_rate
            )
        elif self.configs["optimizer"] == "SGD":
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=self.learning_rate
            )
        else:
            raise UnDefinedException("%s" % self.configs["optimizer"])

    def transform_xlayout(self, x):
        """
            [x1; x2; x3]  <-->  [y11, y12; y21, y22; y31, y33]
                =>
            [x1, 0; x2, 0; x3, 0; x1, 1; x2, 1; x3, 1]  <-->  [y11; y21; y31; y12; y22; y32]
        """
        nsample = x.shape[0]
        x = torch.cat([x for i in range(2)], dim=0)
        task_index = torch.zeros(nsample, 1).to(x.device)
        task_index = torch.cat([task_index, 1 * torch.ones(nsample, 1).to(x.device)], dim=0)
        x = torch.cat([x, task_index], dim=1)

        return x

    def transform_ylayout(self, y):
        """
            [x1; x2; x3]  <-->  [y11, y12; y21, y22; y31, y33]
                =>
            [x1, 0; x2, 0; x3, 0; x1, 1; x2, 1; x3, 1]  <-->  [y11; y21; y31; y12; y22; y32]
        """
        y = y.chunk(2, dim=1)
        y = torch.cat([y[i] for i in range(len(y))], dim=0)

        return y

    def train(self, x, y):
        def handle(x, y):
            y = self.transform_ylayout(y)
            x = x.to(self.device).float()
            y = y.to(self.device).squeeze(1)

            return x, y

        x, y = handle(x, y)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        for i in range(int(self.configs["max-epoch"])):
            self.optimizer.zero_grad()
            _x = self.mlp(x)
            _x = self.transform_xlayout(_x)
            self.gp.set_train_data(_x)
            _y = self.gp(_x)
            loss = -mll(_y, y)
            # BP
            loss.backward()
            self.optimizer.step()
            print("[INFO]: %d iter. Loss: %.8f" % (i + 1, loss))

    def sample(self, x, y):
        """
            x: <list>
            y: <list>
        """
        # TODO: normalize + ref-point
        acq_func = ExpectedHypervolumeImprovement(
            model=self.gp,
            ref_point=reference,
            partition=NondominatedPartitioning(
                ref_point=reference,
                Y=y.to(self.device)
            )
        ).to(self.device)
        _x = self.mlp(x.to(self.device))
        acqv = acq_func(_x.unsqueeze(1).to(self.device))
        top_k, idx = torch.topk(acqv, k=1)
        new_x = x[idx]
        new_y = y[idx]

        return new_x.reshape(-1, 19), new_y.reshape(-1, 2), torch.mean(top_k)

    def fit(self, x, y, load=False, mdl=None):
        """
            train the model
            x: <torch.tensor>
            y: <torch.tensor>
        """
        self.set_train(x, y)
        self.set_optimizer()
        if load:
            self.load(mdl)
        # train
        self.train(x, y)
        self.set_eval()

    def predict(self):
        """
            test the model
        """
        print("predict")
        pass
        exit()

    def save(self, path):
        state_dict = {
            "mlp": self.mlp.state_dict(),
            "gp": self.gp.state_dict()
        }
        torch.save(state_dict, path)

    def load(self, mdl):
        state_dict = torch.load(mdl)
        self.mlp.load_state_dict(state_dict["mlp"])
        self.gp.load_state_dict(state_dict["gp"])
        

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

        sidx = self.random_state.randint(1, len(self.x))
        x.append(self.x[sidx])
        y.append(self.y[sidx])
        self.x.pop(sidx)
        self.y.pop(sidx)
        for step in range(1, self.configs["max-bo-steps"] + 1):
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
            # add new samples
            for item in new_x:
                x.append(item)
                self.x.pop(self.x.index(item))
            for item in new_y:
                y.append(item)
                self.y.pop(self.y.index(item))

        model.save(self.configs["model-output-path"])

    def fit(self, x, y):
        """
            API for running `BayesianOptimization`
        """
        self.set_input(x, y)
        self.run()
