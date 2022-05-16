import sys
import os
import torch
import gpytorch
import tqdm
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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class MLP(nn.Sequential):
    """
        MLP as preprocessor of DKLGP
    """
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        # self.add_module("linear-1", nn.utils.spectral_norm(nn.Linear(input_dim, 1000)))
        self.add_module("linear-1", nn.Linear(input_dim, 1000))
        self.add_module("relu-1", nn.ReLU())
        # self.add_module("linear-2", nn.utils.spectral_norm(nn.Linear(1000, 500)))
        self.add_module("linear-2", nn.Linear(1000, 500))
        self.add_module("relu-2", nn.ReLU())
        # self.add_module("linear-3", nn.utils.spectral_norm(nn.Linear(500, 50)))
        self.add_module("linear-3", nn.Linear(500, 50))
        self.add_module("relu-3", nn.ReLU())
        # self.add_module("linear-4", nn.utils.spectral_norm(nn.Linear(50, output_dim)))
        self.add_module("linear-4", nn.Linear(50, output_dim))
        # self.add_module("relu-4", nn.ReLU())
        # self.add_module("linear-5", nn.Linear(50, output_dim))

class DKLGP():
    """
        DKL-GP: A Version of MultiTaskGP
    """
    def __init__(self, configs, x , y, **kwargs):
        self.configs = configs
        self.n_dim = x.shape[-1]
        self.n_target = y.shape[-1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = MLP(self.n_dim, kwargs["mlp_output_dim"])
        self.mlp.apply(weights_init)
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
        x = x.to(self.device)
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
        y = y.to(self.device)
        y = y.chunk(self.n_target, dim=1)
        y = torch.cat([y[i] for i in range(len(y))], dim=0)
        return y

    def forward_mlp(self, x):
        x = x.to(self.device)
        x = self.mlp(x)
        # normalization
        x = x - x.min(0)[0]
        x = 2 * (x / x.max(0)[0]) - 1
        return x

    def train(self, x):
        x = x.to(self.device)
        x = self.forward_mlp(x)
        x = self.transform_xlayout(x)
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
            x = self.forward_mlp(x)
            x = self.transform_xlayout(x)
            pred = self.gp(x)
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

class DKLGPV2(gpytorch.models.ExactGP):
        def __init__(self, configs, train_x, train_y,
            likelihood=gpytorch.likelihoods.GaussianLikelihood()):

            super(DKLGPV2, self).__init__(train_x, train_y, likelihood)
            self.configs = configs
            self.mean_module = gpytorch.means.LinearMean(input_size=100)
            # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5)),
            #     num_dims=2,
            #     grid_size=100
            # )
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

            self.n_dim = 19
            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
            # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = torch.device("cpu")
            self.likelihood = likelihood.to(self.device)
            self.mlp = MLP(self.n_dim, 100).to(self.device)
            # global variables
            self = self.to(self.device)

        def forward(self, x):
            x = x.to(self.device)
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.mlp(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)

            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


        def fit(self, train_x, train_y):
            # Find optimal model hyperparameters
            self.train()
            self.likelihood.train()

            train_x = train_x.to(self.device)
            train_y = train_y.to(self.device)

            # Use the adam optimizer
            optimizer = torch.optim.Adam([
                {'params': self.mlp.parameters()},
                {'params': self.covar_module.parameters()},
                {'params': self.mean_module.parameters()},
                {'params': self.likelihood.parameters()},
            ], lr=self.configs["learning-rate"])

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

            def _fit():
                iterator = tqdm.tqdm(range(self.configs["max-epoch"]))
                for i in iterator:
                    # Zero backprop gradients
                    optimizer.zero_grad()
                    # Get output from model
                    output = self.forward(train_x)
                    # Calc loss and backprop derivatives
                    loss = -mll(output, train_y)
                    loss.backward()
                    iterator.set_postfix(loss=loss.item())
                    optimizer.step()

            _fit()

        def predict(self, test_x):
            self.eval()
            self.likelihood.eval()
            test_x = test_x.to(self.device)
            with torch.no_grad(), gpytorch.settings.use_toeplitz(False), \
                gpytorch.settings.fast_pred_var():
                preds = self.forward(test_x)

            return preds.mean

        def save(self, path):
            torch.save(self.state_dict(), path)
            print("[INFO]: saving model to %s" % path)

        def load(self, mdl):
            state_dict = torch.load(mdl)
            print("[INFO]: loading model to %s" % path)
            self.load_state_dict(state_dict)
            self.set_eval()
