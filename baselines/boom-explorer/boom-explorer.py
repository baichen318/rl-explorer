# Author: baichen318@gmail.com

import sys, os
sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        os.path.pardir,
        os.path.pardir
    )
)
sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        os.path.pardir,
        os.path.pardir,
        "util"
    )
)
sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        os.path.pardir,
        os.path.pardir,
        "vlsi"
    )
)
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
from sample import crted_sample, random_sample_v2
from boom_design_problem import BOOMDesignProblem
from helper import get_configs, parse_args, adrs_v2, detransform_dataset, write_txt
from vis import plot_pareto_set
from exception import UnDefinedException


def initialize_dnn_gp(x, y):
    return DNNGP(configs, x, y, mlp_output_dim=configs["mlp-output-dim"])


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
    y = model.transform_ylayout(y).squeeze(1)
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
    new_x = problem.x[indices].to(torch.float32)
    del acq_val
    return new_x.reshape(-1, problem.n_dim), torch.mean(top_acq_val)


def get_pareto_set(y: torch.Tensor):
    return y[is_non_dominated(y)]


def define_problem():
    return BOOMDesignProblem(configs)


def boom_explorer_as_baseline(problem):
    hv = Hypervolume(ref_point=problem._ref_point)
    adrs = []
    # generate initial data
    if configs["design"] == "rocket":
        x, y = random_sample_v2(configs, problem, batch=configs["batch"])
    else:
        assert self.configs["design"] == "boom", \
            "[ERROR]: %s is not supported." % self.configs["design"]
        x, y = crted_sample(configs, problem)
    pareto_set = get_pareto_set(y)

    adrs.append(
        adrs_v2(
            get_pareto_set(problem.total_y),
            pareto_set
        )
    )
    # initialize
    model = initialize_dnn_gp(x, y)
    path = os.path.join(
        os.path.dirname(configs["rpt-output-path"]),
        os.path.splitext(os.path.basename(configs["rpt-output-path"]))[0] + "-%s-detail.rpt" % configs["design"]
    )

    # Bayesian optimization
    search_x, search_acq_val = [], []
    iterator = tqdm.tqdm(range(configs["max-bo-steps"]))
    for step in iterator:
        iterator.set_description("Iter %d" % (step + 1))
        # train
        model = fit_dnn_gp(x, y)
        # sample by acquisition function
        new_x, acq_val = ehvi_suggest(model, problem, y)
        search_x.append(new_x)
        search_acq_val.append(acq_val)
        # add in to `x` and `y`
        x = torch.cat((x, new_x), 0)
        y = torch.cat((y, problem.evaluate_microarchitecture(new_x.numpy().squeeze(0))), 0)
        with open(path, 'a') as f:
            msg = "[INFO]: microarchitecture: {}, metric: {}".format(
                new_x,
                problem.evaluate_microarchitecture(new_x.numpy().squeeze(0))
            )
            print(msg)
            f.write(msg + '\n')

    pareto_set = get_pareto_set(y)
    adrs.append(
        adrs_v2(
            get_pareto_set(problem.total_y),
            pareto_set
        )
    )
    print("[INFO]: pareto set: %s, size: %d" % (str(pareto_set), len(pareto_set)))
    plot_pareto_set(
        detransform_dataset(pareto_set),
        dataset_path=configs["dataset-output-path"],
        output=configs["fig-output-path"]
    )

    # write results
    # ADRS:
    write_txt(
        os.path.join(
            os.path.dirname(configs["rpt-output-path"]),
            "adrs-%s.txt" % configs["design"]
        ),
        np.array(adrs),
        fmt="%f"
    )
    # pareto set
    write_txt(
        os.path.join(
            os.path.dirname(configs["rpt-output-path"]),
            os.path.splitext(os.path.basename(configs["rpt-output-path"]))[0] + "-%s.rpt" % configs["design"]
        ),
        np.array(pareto_set),
        fmt="%f"
    )
    # model
    model.save(
        os.path.join(
            os.path.dirname(configs["model-output-path"]),
            os.path.splitext(os.path.basename(configs["model-output-path"]))[0] + "-%s.mdl" % configs["design"]
        )
    )

def main():
    problem = define_problem()
    boom_explorer_as_baseline(problem)

if __name__ == "__main__":
    configs = get_configs(parse_args().configs)
    configs["logger"] = None
    main()
