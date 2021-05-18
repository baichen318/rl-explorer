# Author: baichen318@gmail.com

import os
import torch
import gpytorch
import tqdm
import numpy as np
from typing import Optional
from dnn_gp import DNNGP, DNNGPV2
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from sample import sample
from boom_design_problem import BOOMDesignProblem
from util import get_configs, parse_args, adrs_v2
from exception import UnDefinedException

def initialize_dnn_gp(x, y):
    model = DNNGP(configs, x, y, mlp_output_dim=6)

def fit_dnn_gp(x, y):
    model = initialize_dnn_gp(x, y)
    model.set_train()

    parameters = [
        {"params": model.mlp.parameters()},
        {"params": model.gp.covar_module.parameters()},
        {"params": model.gp.mean_module.parameters()},
        {"params": model.gp.likelihood.parameters()}
    ]
    optimizer = torch.optim.Adam(parameters, lr=configs["learning-rate"])

    iterator = tqdm.trange(configs["max-epoch"], desc="Training DNN-GP")
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.gp.likelihood, model.gp)
    y = model.transform_ylayout(y)
    for i in iterator:
        optimizer.zero_grad()
        _y = model.train(x)
        loss = -mll(_y, y)
        loss.backward()
        optimizer.step()
        iterator.set_postfix(loss=loss.item())
    model.set_eval()

    return model

def ehvi_suggest(model, problem, sampled_y, batch=1):
    """
        model: <DNNGP>
        problem: <MultiObjectiveTestProblem>
        sampled_y: <torch.Tensor>
    """
    partitioning = NondominatedPartitioning(ref_point=problem._ref_point.to(model.device), Y=sampled_y.to(model.device))

    acq_func = ExpectedHypervolumeImprovement(
        model=model.gp,
        ref_point=problem._ref_point.tolist(),
        partitioning=partitioning
    ).to(model.device)

    acq_val = acq_func(
        model.forward_mlp(
            problem.x.to(torch.float).to(model.device)
        ).unsqueeze(1).to(model.device)
    ).to(model.device)
    top_acq_val, indices = torch.topk(acq_val, k=batch)
    new_x = proble.x[indices]
    del acq_val
    return new_x.reshape(-1, problem.n_dim), torch.mean(top_acq_val)

def get_pareto_set(y: torch.Tensor):
    return y[is_non_dominated(y)]

def define_problem():
    return BOOMDesignProblem(configs)

def design_explorer(problem):
    hv = Hypervolume(ref_point=problem._ref_point)
    adrs = []
    # generate initial data
    x, y = sample(configs, problem)
    pareto_set = get_pareto_set(y)

    adrs.append(
        adrs_v2(
            get_pareto_set(problem.total_y),
            pareto_set
        )
    )
    # initialize
    # model = intialize_dnn_gp(x, y)

    # Bayesian optimization
    temp_x, temp_acq_val = [], []
    iterator = tqdm.tqdm(range(configs["max-bo-steps"]))
    for step in iterator:
        iterator.set_description("Iter %d" % (step + 1))
        # train
        model = fit_dnn_gp(x, y)
        # sample by acquisition function
        new_x, acq_val = ehvi_suggest(model, problem, y)
        temp_x.append(new_x)
        temp_acq_val.append(acq_val)
        print(new_x, acq_val)
        exit()
        # update the training data up to `max_acq_idx`, including itself
        # max_acq_idx = temp_acq_val.index(max(temp_acq_val))
        # add in to `x` and `y`
        # x = torch.cat((x, temp_x[max_acq_idx]), 0)
        # y = torch.cat((y, problem(temp_x[max_acq_idx])), 0)
        # # calculate HV
        # hv_val = hv.compute(get_pareto_set(y))
        # print("[INFO]: hyper-volume: %.8f" % hv_val)

def main():
    problem = define_problem()
    design_explorer(problem)

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    main()
